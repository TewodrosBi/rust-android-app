use crate::engine::ring_buffer::{
    create_deck_ring_buffer, DeckAudioConsumer, DeckAudioProducer, DeckStemProducers, StereoFrame,
};
use crate::stems::qnn_engine::{QnnHtpEngine, QnnSessionOptions};
use crate::stems::{
    AtomicDeckStemControls, SharedStemControls, NUM_STEMS, OUTPUT_TENSOR_FLOATS,
    STEM_FFT_WINDOW_FRAMES, STEM_HOP_FRAMES, STEREO_CHANNELS, WINDOW_FLOATS,
};
use core::f32::consts::PI;
use libm::sinf;
use portable_atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Precomputed periodic root-Hann window $w[n] = \sin(\pi n / N)$ for 50% overlap COLA.
pub struct RootHannWindow {
    pub coeffs: [f32; STEM_FFT_WINDOW_FRAMES],
}

impl RootHannWindow {
    pub fn new_boxed() -> Box<Self> {
        let mut coeffs = [0.0_f32; STEM_FFT_WINDOW_FRAMES];
        let inv_n = 1.0_f32 / (STEM_FFT_WINDOW_FRAMES as f32);
        for (n, c) in coeffs.iter_mut().enumerate() {
            *c = sinf(PI * (n as f32) * inv_n);
        }
        Box::new(Self { coeffs })
    }
}

/// Per-deck Overlap-Add (OLA) state machine emitting 4 discrete stereo stem hops.
pub struct DeckOlaProcessor {
    input_history: Box<[f32; WINDOW_FLOATS]>,
    windowed_input: Box<[f32; WINDOW_FLOATS]>,
    inferred_stems: Box<[f32; OUTPUT_TENSOR_FLOATS]>,
    ola_accumulators: Box<[[f32; WINDOW_FLOATS]; NUM_STEMS]>,
    pub hop_in_frames: Box<[StereoFrame; STEM_HOP_FRAMES]>,
    pub stem_hops_out: Box<[[f32; STEM_HOP_FRAMES * STEREO_CHANNELS]; NUM_STEMS]>,
}

impl DeckOlaProcessor {
    pub fn new(_sample_rate: f32) -> Self {
        Self {
            input_history: Box::new([0.0_f32; WINDOW_FLOATS]),
            windowed_input: Box::new([0.0_f32; WINDOW_FLOATS]),
            inferred_stems: Box::new([0.0_f32; OUTPUT_TENSOR_FLOATS]),
            ola_accumulators: Box::new([[0.0_f32; WINDOW_FLOATS]; NUM_STEMS]),
            hop_in_frames: Box::new([StereoFrame::default(); STEM_HOP_FRAMES]),
            stem_hops_out: Box::new([[0.0_f32; STEM_HOP_FRAMES * STEREO_CHANNELS]; NUM_STEMS]),
        }
    }

    /// Processes a single 1024-frame hop through the root-Hann OLA + QNN pipeline
    /// and writes the 4 separated stereo stem hops into `self.stem_hops_out`.
    pub fn process_hop(
        &mut self,
        window: &RootHannWindow,
        qnn: &mut QnnHtpEngine,
        controls: &AtomicDeckStemControls,
    ) {
        let hop_floats = STEM_HOP_FRAMES * STEREO_CHANNELS;

        // 1. Shift upper half of input history down by one hop
        self.input_history.copy_within(hop_floats..WINDOW_FLOATS, 0);

        // 2. Copy newly arrived 1024 stereo frames into the upper half of input_history
        let upper_history = &mut self.input_history[hop_floats..WINDOW_FLOATS];
        for (i, frame) in self.hop_in_frames.iter().enumerate() {
            let base = i * 2;
            upper_history[base] = frame.left;
            upper_history[base + 1] = frame.right;
        }

        // 3. Apply analysis root-Hann window
        for frame_idx in 0..STEM_FFT_WINDOW_FRAMES {
            let w = window.coeffs[frame_idx];
            let base = frame_idx * 2;
            self.windowed_input[base] = self.input_history[base] * w;
            self.windowed_input[base + 1] = self.input_history[base + 1] * w;
        }

        // 4. Execute 4-stem inference on Hexagon HTP v75 (or zero-alloc SVF fallback)
        let start = Instant::now();
        let _ = qnn.execute_inference(&self.windowed_input, &mut self.inferred_stems);
        let elapsed_us = start.elapsed().as_micros() as u64;
        controls
            .last_inference_us
            .store(elapsed_us, Ordering::Relaxed);
        controls.inferred_hops.fetch_add(1, Ordering::Relaxed);

        // 5. Apply synthesis root-Hann window, accumulate into per-stem OLA rings,
        //    and extract the completed 1024-frame hop for each of the 4 stems.
        for stem_idx in 0..NUM_STEMS {
            let stem_offset = stem_idx * WINDOW_FLOATS;
            let stem_slice = &self.inferred_stems[stem_offset..stem_offset + WINDOW_FLOATS];
            let accum = &mut self.ola_accumulators[stem_idx];

            for frame_idx in 0..STEM_FFT_WINDOW_FRAMES {
                let w = window.coeffs[frame_idx];
                let base = frame_idx * 2;
                accum[base] += stem_slice[base] * w;
                accum[base + 1] += stem_slice[base + 1] * w;
            }

            self.stem_hops_out[stem_idx].copy_from_slice(&accum[..hop_floats]);
            accum.copy_within(hop_floats..WINDOW_FLOATS, 0);
            accum[hop_floats..WINDOW_FLOATS].fill(0.0_f32);
        }
    }
}

fn configure_worker_thread_affinity() {
    unsafe {
        let mut cpuset: libc::cpu_set_t = core::mem::zeroed();
        for cpu in 4..8 {
            libc::CPU_SET(cpu, &mut cpuset);
        }
        let tid = libc::gettid();
        let _ = libc::sched_setaffinity(tid, core::mem::size_of::<libc::cpu_set_t>(), &cpuset);
        let _ = libc::setpriority(libc::PRIO_PROCESS, tid as u32, -16);
    }
}

pub struct StemSeparatorHandle {
    running: Arc<AtomicBool>,
    worker_thread: Option<JoinHandle<()>>,
    pub raw_deck_a_producer: DeckAudioProducer,
    pub raw_deck_b_producer: DeckAudioProducer,
    pub stem_controls: Arc<SharedStemControls>,
}

impl StemSeparatorHandle {
    pub fn spawn(
        downstream_stems_a: DeckStemProducers,
        downstream_stems_b: DeckStemProducers,
        sample_rate: f32,
        ring_capacity_frames: usize,
        qnn_options: QnnSessionOptions,
        context_binary: Option<&[u8]>,
    ) -> Self {
        let (raw_deck_a_producer, raw_deck_a_consumer) =
            create_deck_ring_buffer(ring_capacity_frames);
        let (raw_deck_b_producer, raw_deck_b_consumer) =
            create_deck_ring_buffer(ring_capacity_frames);

        let stem_controls = SharedStemControls::new_arc();
        let running = Arc::new(AtomicBool::new(true));

        let mut qnn_engine = QnnHtpEngine::new(qnn_options);
        if let Some(blob) = context_binary {
            let _ = qnn_engine.load_context_binary(blob);
        }
        stem_controls
            .htp_ready
            .store(qnn_engine.is_hardware_ready(), Ordering::Release);

        let running_clone = Arc::clone(&running);
        let controls_clone = Arc::clone(&stem_controls);

        let worker_thread = thread::Builder::new()
            .name("s24-htp-stems".to_string())
            .spawn(move || {
                configure_worker_thread_affinity();
                Self::worker_loop(
                    running_clone,
                    qnn_engine,
                    raw_deck_a_consumer,
                    raw_deck_b_consumer,
                    downstream_stems_a,
                    downstream_stems_b,
                    controls_clone,
                    sample_rate,
                );
            })
            .ok();

        Self {
            running,
            worker_thread,
            raw_deck_a_producer,
            raw_deck_b_producer,
            stem_controls,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn worker_loop(
        running: Arc<AtomicBool>,
        mut qnn_engine: QnnHtpEngine,
        mut raw_a_in: DeckAudioConsumer,
        mut raw_b_in: DeckAudioConsumer,
        mut stems_a_out: DeckStemProducers,
        mut stems_b_out: DeckStemProducers,
        controls: Arc<SharedStemControls>,
        sample_rate: f32,
    ) {
        let window = RootHannWindow::new_boxed();
        let mut ola_a = DeckOlaProcessor::new(sample_rate);
        let mut ola_b = DeckOlaProcessor::new(sample_rate);

        while running.load(Ordering::Relaxed) {
            let mut did_work = false;

            if stems_a_out.min_available_write_frames() >= STEM_HOP_FRAMES
                && raw_a_in.available_to_read() >= STEM_HOP_FRAMES
            {
                let read_a = raw_a_in.pop_frames_into(&mut *ola_a.hop_in_frames);
                if read_a > 0 {
                    ola_a.process_hop(&window, &mut qnn_engine, &controls.deck_a);
                    for s in 0..NUM_STEMS {
                        let _ = stems_a_out.stems[s].push_interleaved(&ola_a.stem_hops_out[s]);
                    }
                    did_work = true;
                }
            }

            if stems_b_out.min_available_write_frames() >= STEM_HOP_FRAMES
                && raw_b_in.available_to_read() >= STEM_HOP_FRAMES
            {
                let read_b = raw_b_in.pop_frames_into(&mut *ola_b.hop_in_frames);
                if read_b > 0 {
                    ola_b.process_hop(&window, &mut qnn_engine, &controls.deck_b);
                    for s in 0..NUM_STEMS {
                        let _ = stems_b_out.stems[s].push_interleaved(&ola_b.stem_hops_out[s]);
                    }
                    did_work = true;
                }
            }

            if !did_work {
                thread::sleep(Duration::from_micros(1500));
            }
        }
    }

    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(handle) = self.worker_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for StemSeparatorHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}
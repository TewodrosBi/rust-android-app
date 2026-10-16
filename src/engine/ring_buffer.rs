use crate::stems::NUM_STEMS;
use portable_atomic::{AtomicBool, AtomicF32, AtomicU64, AtomicU8, Ordering};
use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::Arc;

/// Interleaved stereo sample frame `(left, right)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StereoFrame {
    pub left: f32,
    pub right: f32,
}

/// Producer handle held by the decoder or stem worker thread for a single stereo stream.
pub struct DeckAudioProducer {
    producer: Producer<StereoFrame>,
}

impl DeckAudioProducer {
    /// Pushes interleaved `[L, R, L, R, ...]` `f32` PCM samples into the lock-free ring buffer.
    /// Returns the number of stereo frames written without blocking.
    pub fn push_interleaved(&mut self, interleaved_pcm: &[f32]) -> usize {
        let requested_frames = interleaved_pcm.len() / 2;
        if requested_frames == 0 {
            return 0;
        }

        let available = self.producer.slots();
        let to_write = requested_frames.min(available);
        if to_write == 0 {
            return 0;
        }

        if let Ok(mut chunk) = self.producer.write_chunk_uninit(to_write) {
            let (first, second) = chunk.as_mut_slices();
            let mut src_pairs = interleaved_pcm.chunks_exact(2);

            for slot in first.iter_mut() {
                if let Some(pair) = src_pairs.next() {
                    slot.write(StereoFrame {
                        left: pair[0],
                        right: pair[1],
                    });
                } else {
                    slot.write(StereoFrame::default());
                }
            }

            for slot in second.iter_mut() {
                if let Some(pair) = src_pairs.next() {
                    slot.write(StereoFrame {
                        left: pair[0],
                        right: pair[1],
                    });
                } else {
                    slot.write(StereoFrame::default());
                }
            }

            // SAFETY: All `to_write` slots across `first` and `second` have been initialized.
            unsafe {
                chunk.commit_all();
            }
            to_write
        } else {
            0
        }
    }

    /// Number of free stereo frame slots available for writing.
    #[inline(always)]
    pub fn available_frames(&self) -> usize {
        self.producer.slots()
    }
}

/// Consumer handle held exclusively by the real-time AAudio callback thread.
pub struct DeckAudioConsumer {
    consumer: Consumer<StereoFrame>,
}

impl DeckAudioConsumer {
    /// Returns the number of stereo frames currently queued and ready to be read.
    #[inline(always)]
    pub fn available_to_read(&self) -> usize {
        self.consumer.slots()
    }

    /// Pops up to `out.len()` stereo frames in a single batch read.
    /// Any remaining tail is zero-filled to prevent underrun artifacts.
    #[inline(always)]
    pub fn pop_frames_into(&mut self, out: &mut [StereoFrame]) -> usize {
        let requested = out.len();
        let available = self.consumer.slots();
        let to_read = requested.min(available);

        if to_read > 0 {
            if let Ok(chunk) = self.consumer.read_chunk(to_read) {
                let (first, second) = chunk.as_slices();
                let first_len = first.len();
                out[..first_len].copy_from_slice(first);
                if !second.is_empty() {
                    out[first_len..first_len + second.len()].copy_from_slice(second);
                }
                chunk.commit_all();
            }
        }

        for frame in out.iter_mut().skip(to_read) {
            *frame = StereoFrame {
                left: 0.0,
                right: 0.0,
            };
        }

        to_read
    }

    /// Discards up to `max_frames` from the ring buffer without copying to keep
    /// fallback and stem streams time-aligned when switching modes.
    #[inline(always)]
    pub fn discard_frames(&mut self, max_frames: usize) -> usize {
        let available = self.consumer.slots();
        let to_drop = max_frames.min(available);
        if to_drop > 0 {
            if let Ok(chunk) = self.consumer.read_chunk(to_drop) {
                chunk.commit_all();
                return to_drop;
            }
        }
        0
    }
}

/// 4-Stem SPSC producer bundle (`[Vocals, Drums, Bass, Other]`) for a single deck.
pub struct DeckStemProducers {
    pub stems: [DeckAudioProducer; NUM_STEMS],
}

impl DeckStemProducers {
    /// Returns the minimum free write slots across all 4 stem ring buffers.
    #[inline(always)]
    pub fn min_available_write_frames(&self) -> usize {
        self.stems[0]
            .available_frames()
            .min(self.stems[1].available_frames())
            .min(self.stems[2].available_frames())
            .min(self.stems[3].available_frames())
    }

    /// Pushes interleaved stereo `f32` frames for each of the 4 stems (`[Vocals, Drums, Bass, Other]`).
    #[inline(always)]
    pub fn push_all_stems(&mut self, stem_interleaved: &[&[f32]; NUM_STEMS]) -> usize {
        let mut min_written = usize::MAX;
        for i in 0..NUM_STEMS {
            let written = self.stems[i].push_interleaved(stem_interleaved[i]);
            if written < min_written {
                min_written = written;
            }
        }
        if min_written == usize::MAX {
            0
        } else {
            min_written
        }
    }
}

/// 4-Stem SPSC consumer bundle (`[Vocals, Drums, Bass, Other]`) owned by the AAudio callback.
pub struct DeckStemConsumers {
    pub stems: [DeckAudioConsumer; NUM_STEMS],
}

impl DeckStemConsumers {
    /// Returns `true` only if all 4 stem ring buffers have at least `required_frames` ready.
    #[inline(always)]
    pub fn can_read_chunk(&self, required_frames: usize) -> bool {
        self.stems[0].available_to_read() >= required_frames
            && self.stems[1].available_to_read() >= required_frames
            && self.stems[2].available_to_read() >= required_frames
            && self.stems[3].available_to_read() >= required_frames
    }

    /// Discards up to `max_frames` across all 4 stem queues when stem separation is bypassed.
    #[inline(always)]
    pub fn discard_all(&mut self, max_frames: usize) {
        for stem in &mut self.stems {
            let _ = stem.discard_frames(max_frames);
        }
    }
}

/// Creates a lock-free SPSC stereo ring buffer with the given frame capacity.
pub fn create_deck_ring_buffer(capacity_frames: usize) -> (DeckAudioProducer, DeckAudioConsumer) {
    let (producer, consumer) = RingBuffer::new(capacity_frames);
    (
        DeckAudioProducer { producer },
        DeckAudioConsumer { consumer },
    )
}

/// Creates 4 matched lock-free SPSC stereo ring buffers (`Vocals`, `Drums`, `Bass`, `Other`) for a deck.
pub fn create_deck_stem_ring_buffers(
    capacity_frames: usize,
) -> (DeckStemProducers, DeckStemConsumers) {
    let (p0, c0) = create_deck_ring_buffer(capacity_frames);
    let (p1, c1) = create_deck_ring_buffer(capacity_frames);
    let (p2, c2) = create_deck_ring_buffer(capacity_frames);
    let (p3, c3) = create_deck_ring_buffer(capacity_frames);

    (
        DeckStemProducers {
            stems: [p0, p1, p2, p3],
        },
        DeckStemConsumers {
            stems: [c0, c1, c2, c3],
        },
    )
}

/// Cache-line-aligned per-deck atomic state shared between UI/JNI and the AAudio callback.
#[repr(align(64))]
pub struct AtomicDeckControls {
    pub trim_gain: AtomicF32,
    pub channel_fader: AtomicF32,
    pub eq_low_knob: AtomicF32,
    pub eq_mid_knob: AtomicF32,
    pub eq_high_knob: AtomicF32,
    pub kill_low: AtomicBool,
    pub kill_mid: AtomicBool,
    pub kill_high: AtomicBool,
    pub pfl_cue_enabled: AtomicBool,
    /// Real-time 4-stem active toggle for this deck.
    pub stem_enabled: AtomicBool,
    /// Individual stem linear gains (default `1.0`, range `[0.0, 2.0]`).
    pub stem_vocals: AtomicF32,
    pub stem_drums: AtomicF32,
    pub stem_bass: AtomicF32,
    pub stem_other: AtomicF32,
}

impl AtomicDeckControls {
    pub const fn new() -> Self {
        Self {
            trim_gain: AtomicF32::new(1.0),
            channel_fader: AtomicF32::new(1.0),
            eq_low_knob: AtomicF32::new(0.5),
            eq_mid_knob: AtomicF32::new(0.5),
            eq_high_knob: AtomicF32::new(0.5),
            kill_low: AtomicBool::new(false),
            kill_mid: AtomicBool::new(false),
            kill_high: AtomicBool::new(false),
            pfl_cue_enabled: AtomicBool::new(false),
            stem_enabled: AtomicBool::new(false),
            stem_vocals: AtomicF32::new(1.0),
            stem_drums: AtomicF32::new(1.0),
            stem_bass: AtomicF32::new(1.0),
            stem_other: AtomicF32::new(1.0),
        }
    }
}

/// Cache-line-aligned global mixer state shared between UI/JNI and the audio thread.
#[repr(align(64))]
pub struct SharedMixerControls {
    pub deck_a: AtomicDeckControls,
    pub deck_b: AtomicDeckControls,
    pub crossfader_pos: AtomicF32,
    pub crossfader_curve: AtomicU8,
    pub master_gain: AtomicF32,
    pub cue_level_gain: AtomicF32,
    pub cue_master_mix: AtomicF32,
    /// When `true`, Left output emits Master Mix (mono-summed) and Right output
    /// emits Headphone Cue Bus (mono-summed) for split-cable DJ monitoring.
    pub split_cue_output: AtomicBool,
    pub callback_frames_processed: AtomicU64,
    pub xrun_count: AtomicU64,
}

impl SharedMixerControls {
    pub fn new_arc() -> Arc<Self> {
        Arc::new(Self {
            deck_a: AtomicDeckControls::new(),
            deck_b: AtomicDeckControls::new(),
            crossfader_pos: AtomicF32::new(0.0),
            crossfader_curve: AtomicU8::new(0),
            master_gain: AtomicF32::new(0.85),
            cue_level_gain: AtomicF32::new(0.85),
            cue_master_mix: AtomicF32::new(0.0),
            split_cue_output: AtomicBool::new(false),
            callback_frames_processed: AtomicU64::new(0),
            xrun_count: AtomicU64::new(0),
        })
    }

    #[inline(always)]
    pub fn record_frames(&self, frames: u64) {
        self.callback_frames_processed
            .fetch_add(frames, Ordering::Relaxed);
    }
}
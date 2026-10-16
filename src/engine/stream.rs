use crate::engine::audio_callback::DjAudioCallback;
use crate::engine::ring_buffer::{
    create_deck_ring_buffer, create_deck_stem_ring_buffers, DeckAudioProducer, DeckStemProducers,
    SharedMixerControls,
};
use oboe::{
    AudioOutputStream, AudioOutputStreamSafe, AudioStream, AudioStreamAsync, AudioStreamBuilder,
    ContentType, Output, PerformanceMode, SampleRateConversionQuality, SharingMode, Stereo, Usage,
};
use std::sync::Arc;

pub const NATIVE_SAMPLE_RATE: i32 = 48_000;
pub const DEFAULT_BURST_FRAMES: i32 = 96;
/// 4 seconds of stereo audio buffer per deck at 48kHz (192,000 frames)
pub const DECK_RING_BUFFER_FRAMES: usize = 48_000 * 4;
/// 1 second of stereo buffer per individual stem ring at 48kHz (48,000 frames)
pub const STEM_RING_BUFFER_FRAMES: usize = 48_000;

pub struct DjAudioEngine {
    stream: Option<AudioStreamAsync<Output, DjAudioCallback>>,
    pub sample_rate: i32,
    pub deck_a_producer: DeckAudioProducer,
    pub deck_b_producer: DeckAudioProducer,
    pub deck_a_stem_producers: Option<DeckStemProducers>,
    pub deck_b_stem_producers: Option<DeckStemProducers>,
    pub controls: Arc<SharedMixerControls>,
}

impl DjAudioEngine {
    pub fn new(sample_rate: i32, burst_frames: i32) -> Result<Self, oboe::Error> {
        let valid_rate = if sample_rate > 8_000 {
            sample_rate
        } else {
            NATIVE_SAMPLE_RATE
        };
        let valid_burst = if burst_frames >= 32 {
            burst_frames
        } else {
            DEFAULT_BURST_FRAMES
        };

        let (deck_a_producer, deck_a_consumer) = create_deck_ring_buffer(DECK_RING_BUFFER_FRAMES);
        let (deck_b_producer, deck_b_consumer) = create_deck_ring_buffer(DECK_RING_BUFFER_FRAMES);
        let (deck_a_stem_producers, deck_a_stem_consumers) =
            create_deck_stem_ring_buffers(STEM_RING_BUFFER_FRAMES);
        let (deck_b_stem_producers, deck_b_stem_consumers) =
            create_deck_stem_ring_buffers(STEM_RING_BUFFER_FRAMES);
        let controls = SharedMixerControls::new_arc();

        let callback = DjAudioCallback::new(
            deck_a_consumer,
            deck_b_consumer,
            deck_a_stem_consumers,
            deck_b_stem_consumers,
            Arc::clone(&controls),
            valid_rate as f32,
        );

        let mut stream = Self::open_low_latency_stream(callback, valid_rate, valid_burst)?;

        let frames_per_burst = stream.get_frames_per_burst().max(valid_burst);
        let target_buffer_size = frames_per_burst * 2;
        let _ = stream.set_buffer_size_in_frames(target_buffer_size);

        log::info!(
            "S24 Ultra AAudio Stream Opened: rate={} Hz, burst={} frames, buffer_size={} frames, sharing={:?}, perf={:?}",
            stream.get_sample_rate(),
            frames_per_burst,
            stream.get_buffer_size_in_frames(),
            stream.get_sharing_mode(),
            stream.get_performance_mode()
        );

        Ok(Self {
            stream: Some(stream),
            sample_rate: valid_rate,
            deck_a_producer,
            deck_b_producer,
            deck_a_stem_producers: Some(deck_a_stem_producers),
            deck_b_stem_producers: Some(deck_b_stem_producers),
            controls,
        })
    }

    fn open_low_latency_stream(
        callback: DjAudioCallback,
        sample_rate: i32,
        burst_frames: i32,
    ) -> Result<AudioStreamAsync<Output, DjAudioCallback>, oboe::Error> {
        AudioStreamBuilder::default()
            .set_direction::<Output>()
            .set_performance_mode(PerformanceMode::LowLatency)
            .set_sharing_mode(SharingMode::Exclusive)
            .set_format::<f32>()
            .set_channel_count::<Stereo>()
            .set_sample_rate(sample_rate)
            .set_sample_rate_conversion_quality(SampleRateConversionQuality::None)
            .set_frames_per_data_callback(burst_frames)
            .set_usage(Usage::Media)
            .set_content_type(ContentType::Music)
            .set_callback(callback)
            .open_stream()
    }

    pub fn start(&mut self) -> Result<(), oboe::Error> {
        if let Some(ref mut stream) = self.stream {
            stream.start()?;
            log::info!("AAudio Exclusive Low-Latency stream started.");
        }
        Ok(())
    }

    pub fn stop(&mut self) -> Result<(), oboe::Error> {
        if let Some(ref mut stream) = self.stream {
            stream.stop()?;
            log::info!("AAudio Exclusive Low-Latency stream stopped.");
        }
        Ok(())
    }
}

impl Drop for DjAudioEngine {
    fn drop(&mut self) {
        if let Some(mut stream) = self.stream.take() {
            let _ = stream.stop();
            let _ = stream.close();
        }
    }
}
pub mod qnn_engine;
pub mod separator;

use portable_atomic::{AtomicBool, AtomicF32, AtomicU64, Ordering};
use std::sync::Arc;

/// Number of demixed stereo stems produced by the Hexagon HTP v75 model.
pub const NUM_STEMS: usize = 4;

/// Stereo channels per frame.
pub const STEREO_CHANNELS: usize = 2;

/// Analysis/synthesis window length in stereo frames (42.67 ms at 48,000 Hz).
pub const STEM_FFT_WINDOW_FRAMES: usize = 2048;

/// 50% overlap hop size in stereo frames (21.33 ms at 48,000 Hz).
pub const STEM_HOP_FRAMES: usize = STEM_FFT_WINDOW_FRAMES / 2;

/// Number of interleaved floats per analysis window (`2048 * 2 = 4096`).
pub const WINDOW_FLOATS: usize = STEM_FFT_WINDOW_FRAMES * STEREO_CHANNELS;

/// Number of interleaved floats across all 4 output stems per window (`4 * 2048 * 2 = 16384`).
pub const OUTPUT_TENSOR_FLOATS: usize = NUM_STEMS * WINDOW_FLOATS;

#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StemKind {
    Vocals = 0,
    Drums = 1,
    Bass = 2,
    Other = 3,
}

impl StemKind {
    #[inline(always)]
    pub const fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Vocals,
            1 => Self::Drums,
            2 => Self::Bass,
            _ => Self::Other,
        }
    }
}

/// Cache-line-aligned lock-free atomic controls for a single deck's 4-stem mixer.
#[repr(align(64))]
pub struct AtomicDeckStemControls {
    pub enabled: AtomicBool,
    pub vocals_gain: AtomicF32,
    pub drums_gain: AtomicF32,
    pub bass_gain: AtomicF32,
    pub other_gain: AtomicF32,
    pub vocals_mute: AtomicBool,
    pub drums_mute: AtomicBool,
    pub bass_mute: AtomicBool,
    pub other_mute: AtomicBool,
    pub inferred_hops: AtomicU64,
    pub last_inference_us: AtomicU64,
}

impl AtomicDeckStemControls {
    pub const fn new() -> Self {
        Self {
            enabled: AtomicBool::new(true),
            vocals_gain: AtomicF32::new(1.0),
            drums_gain: AtomicF32::new(1.0),
            bass_gain: AtomicF32::new(1.0),
            other_gain: AtomicF32::new(1.0),
            vocals_mute: AtomicBool::new(false),
            drums_mute: AtomicBool::new(false),
            bass_mute: AtomicBool::new(false),
            other_mute: AtomicBool::new(false),
            inferred_hops: AtomicU64::new(0),
            last_inference_us: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    pub fn effective_target_gains(&self) -> [f32; NUM_STEMS] {
        let v = if self.vocals_mute.load(Ordering::Relaxed) {
            0.0_f32
        } else {
            self.vocals_gain.load(Ordering::Relaxed).clamp(0.0, 2.0)
        };
        let d = if self.drums_mute.load(Ordering::Relaxed) {
            0.0_f32
        } else {
            self.drums_gain.load(Ordering::Relaxed).clamp(0.0, 2.0)
        };
        let b = if self.bass_mute.load(Ordering::Relaxed) {
            0.0_f32
        } else {
            self.bass_gain.load(Ordering::Relaxed).clamp(0.0, 2.0)
        };
        let o = if self.other_mute.load(Ordering::Relaxed) {
            0.0_f32
        } else {
            self.other_gain.load(Ordering::Relaxed).clamp(0.0, 2.0)
        };
        [v, d, b, o]
    }
}

/// Global atomic stem controls for both Deck A and Deck B.
#[repr(align(64))]
pub struct SharedStemControls {
    pub deck_a: AtomicDeckStemControls,
    pub deck_b: AtomicDeckStemControls,
    pub htp_ready: AtomicBool,
}

impl SharedStemControls {
    pub fn new_arc() -> Arc<Self> {
        Arc::new(Self {
            deck_a: AtomicDeckStemControls::new(),
            deck_b: AtomicDeckStemControls::new(),
            htp_ready: AtomicBool::new(false),
        })
    }
}
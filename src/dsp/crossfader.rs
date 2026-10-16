use crate::dsp::isolator::ExpSmoother;
use core::f32::consts::FRAC_PI_2;
use libm::{cosf, sinf};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossfaderCurve {
    /// Constant-power cosine/sine law (+0 dB sum for uncorrelated sources).
    Smooth = 0,
    /// Linear transition across the fader travel.
    DefaultLinear = 1,
    /// Turntablist fast cut-in curve (full volume achieved at 4% travel from either edge).
    ScratchCut = 2,
}

impl CrossfaderCurve {
    #[inline(always)]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Smooth,
            1 => Self::DefaultLinear,
            _ => Self::ScratchCut,
        }
    }
}

pub struct Crossfader {
    position_smoother: ExpSmoother,
    curve: CrossfaderCurve,
}

impl Crossfader {
    /// Position is normalized in `[-1.0, 1.0]` where `-1.0` is full Deck A,
    /// `0.0` is center, and `+1.0` is full Deck B.
    pub fn new(sample_rate: f32) -> Self {
        Self {
            position_smoother: ExpSmoother::new(0.0, 3.5, sample_rate),
            curve: CrossfaderCurve::Smooth,
        }
    }

    #[inline(always)]
    pub fn set_position(&mut self, position: f32) {
        self.position_smoother
            .set_target(position.clamp(-1.0_f32, 1.0_f32));
    }

    #[inline(always)]
    pub fn set_curve(&mut self, curve: CrossfaderCurve) {
        self.curve = curve;
    }

    /// Advances the internal smoother by one sample and returns `(gain_a, gain_b)`.
    #[inline(always)]
    pub fn next_gains(&mut self) -> (f32, f32) {
        let pos = self.position_smoother.next_value();
        // Normalize [-1.0, 1.0] to x in [0.0, 1.0]
        let x = (pos + 1.0_f32) * 0.5_f32;

        match self.curve {
            CrossfaderCurve::Smooth => {
                let angle = x * FRAC_PI_2;
                (cosf(angle), sinf(angle))
            }
            CrossfaderCurve::DefaultLinear => {
                let gain_a = 1.0_f32 - x;
                let gain_b = x;
                (gain_a, gain_b)
            }
            CrossfaderCurve::ScratchCut => {
                const CUT_WINDOW: f32 = 0.04_f32;
                let gain_b = if x <= 0.002_f32 {
                    0.0_f32
                } else if x < CUT_WINDOW {
                    x / CUT_WINDOW
                } else {
                    1.0_f32
                };

                let inv_x = 1.0_f32 - x;
                let gain_a = if inv_x <= 0.002_f32 {
                    0.0_f32
                } else if inv_x < CUT_WINDOW {
                    inv_x / CUT_WINDOW
                } else {
                    1.0_f32
                };

                (gain_a, gain_b)
            }
        }
    }
}
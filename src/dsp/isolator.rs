use crate::dsp::sanitize_denormal;
use core::f32::consts::{FRAC_1_SQRT_2, PI};
use libm::{cosf, powf, sinf};

/// Transposed Direct Form II (TDF-II) 2nd-order Biquad Filter.
/// TDF-II provides superior floating-point numerical accuracy at low frequencies.
#[derive(Clone, Copy, Debug, Default)]
pub struct BiquadTdf2 {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl BiquadTdf2 {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Configures a 2nd-order Butterworth Low-Pass filter ($Q = 1/\sqrt{2}$).
    pub fn set_butterworth_lp(&mut self, cutoff_hz: f32, sample_rate: f32) {
        let omega = 2.0_f32 * PI * (cutoff_hz / sample_rate);
        let sin_w = sinf(omega);
        let cos_w = cosf(omega);
        let alpha = sin_w / (2.0_f32 * FRAC_1_SQRT_2);

        let a0_inv = 1.0_f32 / (1.0_f32 + alpha);
        let b1_raw = 1.0_f32 - cos_w;
        let b0_raw = b1_raw * 0.5_f32;

        self.b0 = b0_raw * a0_inv;
        self.b1 = b1_raw * a0_inv;
        self.b2 = b0_raw * a0_inv;
        self.a1 = (-2.0_f32 * cos_w) * a0_inv;
        self.a2 = (1.0_f32 - alpha) * a0_inv;
    }

    /// Configures a 2nd-order Butterworth High-Pass filter ($Q = 1/\sqrt{2}$).
    pub fn set_butterworth_hp(&mut self, cutoff_hz: f32, sample_rate: f32) {
        let omega = 2.0_f32 * PI * (cutoff_hz / sample_rate);
        let sin_w = sinf(omega);
        let cos_w = cosf(omega);
        let alpha = sin_w / (2.0_f32 * FRAC_1_SQRT_2);

        let a0_inv = 1.0_f32 / (1.0_f32 + alpha);
        let b_common = 1.0_f32 + cos_w;
        let b0_raw = b_common * 0.5_f32;

        self.b0 = b0_raw * a0_inv;
        self.b1 = -b_common * a0_inv;
        self.b2 = b0_raw * a0_inv;
        self.a1 = (-2.0_f32 * cos_w) * a0_inv;
        self.a2 = (1.0_f32 - alpha) * a0_inv;
    }

    /// Configures a 2nd-order Butterworth All-Pass filter ($Q = 1/\sqrt{2}$)
    /// for phase compensation of the low band at the upper crossover frequency.
    pub fn set_butterworth_ap(&mut self, cutoff_hz: f32, sample_rate: f32) {
        let omega = 2.0_f32 * PI * (cutoff_hz / sample_rate);
        let sin_w = sinf(omega);
        let cos_w = cosf(omega);
        let alpha = sin_w / (2.0_f32 * FRAC_1_SQRT_2);

        let a0_inv = 1.0_f32 / (1.0_f32 + alpha);
        let b0_raw = 1.0_f32 - alpha;
        let b1_raw = -2.0_f32 * cos_w;
        let b2_raw = 1.0_f32 + alpha;

        self.b0 = b0_raw * a0_inv;
        self.b1 = b1_raw * a0_inv;
        self.b2 = b2_raw * a0_inv;
        self.a1 = b1_raw * a0_inv;
        self.a2 = b0_raw * a0_inv;
    }

    #[inline(always)]
    pub fn process(&mut self, input: f32) -> f32 {
        let out = self.b0 * input + self.z1;
        self.z1 = sanitize_denormal(self.b1 * input - self.a1 * out + self.z2);
        self.z2 = sanitize_denormal(self.b2 * input - self.a2 * out);
        out
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// 4th-order Linkwitz-Riley filter formed by cascading two identical
/// 2nd-order Butterworth sections in series.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinkwitzRiley4 {
    stage1: BiquadTdf2,
    stage2: BiquadTdf2,
}

impl LinkwitzRiley4 {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            stage1: BiquadTdf2::new(),
            stage2: BiquadTdf2::new(),
        }
    }

    pub fn set_lowpass(&mut self, cutoff_hz: f32, sample_rate: f32) {
        self.stage1.set_butterworth_lp(cutoff_hz, sample_rate);
        self.stage2.set_butterworth_lp(cutoff_hz, sample_rate);
    }

    pub fn set_highpass(&mut self, cutoff_hz: f32, sample_rate: f32) {
        self.stage1.set_butterworth_hp(cutoff_hz, sample_rate);
        self.stage2.set_butterworth_hp(cutoff_hz, sample_rate);
    }

    #[inline(always)]
    pub fn process(&mut self, input: f32) -> f32 {
        let mid = self.stage1.process(input);
        self.stage2.process(mid)
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.stage1.reset();
        self.stage2.reset();
    }
}

/// 1-Pole Exponential Parameter Smoother to eliminate zipper noise during
/// rapid EQ knob sweeps or hard kill-switch toggles.
#[derive(Clone, Copy, Debug)]
pub struct ExpSmoother {
    current: f32,
    target: f32,
    coeff: f32,
}

impl ExpSmoother {
    pub fn new(initial: f32, time_constant_ms: f32, sample_rate: f32) -> Self {
        let tau_samples = (time_constant_ms * 0.001_f32) * sample_rate;
        let coeff = if tau_samples > 0.0_f32 {
            1.0_f32 - powf(2.718281828_f32, -1.0_f32 / tau_samples)
        } else {
            1.0_f32
        };
        Self {
            current: initial,
            target: initial,
            coeff,
        }
    }

    #[inline(always)]
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    #[inline(always)]
    pub fn next_value(&mut self) -> f32 {
        let diff = self.target - self.current;
        if diff.abs() < 1.0e-6_f32 {
            self.current = self.target;
        } else {
            self.current += self.coeff * diff;
        }
        self.current
    }
}

/// Single-channel LR4 3-band filter network.
#[derive(Clone, Copy, Debug, Default)]
struct ChannelCrossover {
    lp_low: LinkwitzRiley4,
    hp_low: LinkwitzRiley4,
    ap_low_comp: BiquadTdf2,
    lp_high: LinkwitzRiley4,
    hp_high: LinkwitzRiley4,
}

impl ChannelCrossover {
    const fn new() -> Self {
        Self {
            lp_low: LinkwitzRiley4::new(),
            hp_low: LinkwitzRiley4::new(),
            ap_low_comp: BiquadTdf2::new(),
            lp_high: LinkwitzRiley4::new(),
            hp_high: LinkwitzRiley4::new(),
        }
    }

    fn configure(&mut self, low_mid_hz: f32, mid_high_hz: f32, sample_rate: f32) {
        self.lp_low.set_lowpass(low_mid_hz, sample_rate);
        self.hp_low.set_highpass(low_mid_hz, sample_rate);
        self.ap_low_comp.set_butterworth_ap(mid_high_hz, sample_rate);
        self.lp_high.set_lowpass(mid_high_hz, sample_rate);
        self.hp_high.set_highpass(mid_high_hz, sample_rate);
    }

    #[inline(always)]
    fn split(&mut self, input: f32) -> (f32, f32, f32) {
        let low_raw = self.lp_low.process(input);
        let mid_high = self.hp_low.process(input);

        let low = self.ap_low_comp.process(low_raw);
        let mid = self.lp_high.process(mid_high);
        let high = self.hp_high.process(mid_high);

        (low, mid, high)
    }

    fn reset(&mut self) {
        self.lp_low.reset();
        self.hp_low.reset();
        self.ap_low_comp.reset();
        self.lp_high.reset();
        self.hp_high.reset();
    }
}

/// Stereo 3-Band LR4 DJ Isolator EQ with hard kill switches and de-zippering.
pub struct ThreeBandIsolator {
    left: ChannelCrossover,
    right: ChannelCrossover,
    gain_low: ExpSmoother,
    gain_mid: ExpSmoother,
    gain_high: ExpSmoother,
    raw_gain_low: f32,
    raw_gain_mid: f32,
    raw_gain_high: f32,
    kill_low: bool,
    kill_mid: bool,
    kill_high: bool,
}

impl ThreeBandIsolator {
    pub const DEFAULT_LOW_CROSSOVER_HZ: f32 = 250.0;
    pub const DEFAULT_HIGH_CROSSOVER_HZ: f32 = 4000.0;
    pub const SMOOTHING_TIME_MS: f32 = 5.0;

    pub fn new(sample_rate: f32) -> Self {
        let mut left = ChannelCrossover::new();
        let mut right = ChannelCrossover::new();
        left.configure(
            Self::DEFAULT_LOW_CROSSOVER_HZ,
            Self::DEFAULT_HIGH_CROSSOVER_HZ,
            sample_rate,
        );
        right.configure(
            Self::DEFAULT_LOW_CROSSOVER_HZ,
            Self::DEFAULT_HIGH_CROSSOVER_HZ,
            sample_rate,
        );

        Self {
            left,
            right,
            gain_low: ExpSmoother::new(1.0, Self::SMOOTHING_TIME_MS, sample_rate),
            gain_mid: ExpSmoother::new(1.0, Self::SMOOTHING_TIME_MS, sample_rate),
            gain_high: ExpSmoother::new(1.0, Self::SMOOTHING_TIME_MS, sample_rate),
            raw_gain_low: 1.0,
            raw_gain_mid: 1.0,
            raw_gain_high: 1.0,
            kill_low: false,
            kill_mid: false,
            kill_high: false,
        }
    }

    /// Converts normalized knob position `[0.0, 1.0]` into linear isolator gain:
    /// - `0.0` -> `0.0` (-inf dB full isolator kill)
    /// - `0.5` -> `1.0` (0.0 dB flat unity gain)
    /// - `1.0` -> `2.81838` (+9.0 dB boost)
    #[inline(always)]
    pub fn knob_to_linear_gain(knob_norm: f32) -> f32 {
        let clamped = knob_norm.clamp(0.0_f32, 1.0_f32);
        if clamped <= 0.001_f32 {
            0.0_f32
        } else if clamped <= 0.5_f32 {
            let t = clamped * 2.0_f32;
            t * t * t
        } else {
            let db = (clamped - 0.5_f32) * 18.0_f32;
            powf(10.0_f32, db / 20.0_f32)
        }
    }

    #[inline(always)]
    fn refresh_targets(&mut self) {
        self.gain_low
            .set_target(if self.kill_low { 0.0 } else { self.raw_gain_low });
        self.gain_mid
            .set_target(if self.kill_mid { 0.0 } else { self.raw_gain_mid });
        self.gain_high.set_target(if self.kill_high {
            0.0
        } else {
            self.raw_gain_high
        });
    }

    /// Sets linear or knob-mapped targets for Low, Mid, and High bands.
    #[inline(always)]
    pub fn set_band_gains_linear(&mut self, low: f32, mid: f32, high: f32) {
        self.raw_gain_low = low.clamp(0.0_f32, 4.0_f32);
        self.raw_gain_mid = mid.clamp(0.0_f32, 4.0_f32);
        self.raw_gain_high = high.clamp(0.0_f32, 4.0_f32);
        self.refresh_targets();
    }

    /// Sets hard kill switch states for each band with smooth de-zippered ramping.
    #[inline(always)]
    pub fn set_kills(&mut self, kill_low: bool, kill_mid: bool, kill_high: bool) {
        self.kill_low = kill_low;
        self.kill_mid = kill_mid;
        self.kill_high = kill_high;
        self.refresh_targets();
    }

    /// Processes a single stereo frame `(left, right)` in real time.
    #[inline(always)]
    pub fn process_stereo(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        let g_low = self.gain_low.next_value();
        let g_mid = self.gain_mid.next_value();
        let g_high = self.gain_high.next_value();

        let (ll, ml, hl) = self.left.split(in_l);
        let (lr, mr, hr) = self.right.split(in_r);

        let out_l = ll * g_low + ml * g_mid + hl * g_high;
        let out_r = lr * g_low + mr * g_mid + hr * g_high;

        (out_l, out_r)
    }

    pub fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
    }
}
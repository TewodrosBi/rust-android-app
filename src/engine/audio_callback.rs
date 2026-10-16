use crate::dsp::crossfader::{Crossfader, CrossfaderCurve};
use crate::dsp::isolator::{ExpSmoother, ThreeBandIsolator};
use crate::engine::ring_buffer::{
    DeckAudioConsumer, DeckStemConsumers, SharedMixerControls, StereoFrame,
};
use crate::stems::NUM_STEMS;
use oboe::{AudioOutputCallback, AudioOutputStreamSafe, DataCallbackResult, Error};
use portable_atomic::Ordering;
use std::sync::Arc;

/// Maximum stereo frames processed per internal loop iteration.
/// Pre-allocated inline inside `DjAudioCallback` to guarantee zero heap allocations.
const MAX_SCRATCH_FRAMES: usize = 256;

/// Soft-knee master bus limiter using a continuous C1-smooth transition at
/// `KNEE_START` (0.80 / -1.94 dBFS) followed by rational Padé(3,2) `tanh` saturation
/// asymptotically bounded at `1.0` (0.0 dBFS).
#[inline(always)]
pub fn soft_knee_limit(sample: f32) -> f32 {
    let abs_x = sample.abs();
    const KNEE_START: f32 = 0.80_f32;
    const HEADROOM: f32 = 1.0_f32 - KNEE_START;
    const INV_HEADROOM: f32 = 1.0_f32 / HEADROOM;

    if abs_x <= KNEE_START {
        sample
    } else {
        let sign = sample.signum();
        let excess = (abs_x - KNEE_START) * INV_HEADROOM;
        let tanh_approx = if excess >= 3.0_f32 {
            1.0_f32
        } else {
            let e2 = excess * excess;
            (excess * (27.0_f32 + e2)) / (27.0_f32 + 9.0_f32 * e2)
        };
        sign * (KNEE_START + HEADROOM * tanh_approx)
    }
}

pub struct DjAudioCallback {
    deck_a_consumer: DeckAudioConsumer,
    deck_b_consumer: DeckAudioConsumer,
    deck_a_stems: DeckStemConsumers,
    deck_b_stems: DeckStemConsumers,
    controls: Arc<SharedMixerControls>,
    isolator_a: ThreeBandIsolator,
    isolator_b: ThreeBandIsolator,
    stem_smoothers_a: [ExpSmoother; NUM_STEMS],
    stem_smoothers_b: [ExpSmoother; NUM_STEMS],
    fader_smoother_a: ExpSmoother,
    fader_smoother_b: ExpSmoother,
    trim_smoother_a: ExpSmoother,
    trim_smoother_b: ExpSmoother,
    master_smoother: ExpSmoother,
    cue_level_smoother: ExpSmoother,
    crossfader: Crossfader,
    scratch_a: [StereoFrame; MAX_SCRATCH_FRAMES],
    scratch_b: [StereoFrame; MAX_SCRATCH_FRAMES],
    scratch_stems_a: [[StereoFrame; MAX_SCRATCH_FRAMES]; NUM_STEMS],
    scratch_stems_b: [[StereoFrame; MAX_SCRATCH_FRAMES]; NUM_STEMS],
}

impl DjAudioCallback {
    pub fn new(
        deck_a_consumer: DeckAudioConsumer,
        deck_b_consumer: DeckAudioConsumer,
        deck_a_stems: DeckStemConsumers,
        deck_b_stems: DeckStemConsumers,
        controls: Arc<SharedMixerControls>,
        sample_rate: f32,
    ) -> Self {
        Self {
            deck_a_consumer,
            deck_b_consumer,
            deck_a_stems,
            deck_b_stems,
            controls,
            isolator_a: ThreeBandIsolator::new(sample_rate),
            isolator_b: ThreeBandIsolator::new(sample_rate),
            stem_smoothers_a: [
                ExpSmoother::new(1.0, 4.0, sample_rate),
                ExpSmoother::new(1.0, 4.0, sample_rate),
                ExpSmoother::new(1.0, 4.0, sample_rate),
                ExpSmoother::new(1.0, 4.0, sample_rate),
            ],
            stem_smoothers_b: [
                ExpSmoother::new(1.0, 4.0, sample_rate),
                ExpSmoother::new(1.0, 4.0, sample_rate),
                ExpSmoother::new(1.0, 4.0, sample_rate),
                ExpSmoother::new(1.0, 4.0, sample_rate),
            ],
            fader_smoother_a: ExpSmoother::new(1.0, 4.0, sample_rate),
            fader_smoother_b: ExpSmoother::new(1.0, 4.0, sample_rate),
            trim_smoother_a: ExpSmoother::new(1.0, 4.0, sample_rate),
            trim_smoother_b: ExpSmoother::new(1.0, 4.0, sample_rate),
            master_smoother: ExpSmoother::new(0.85, 5.0, sample_rate),
            cue_level_smoother: ExpSmoother::new(0.85, 5.0, sample_rate),
            crossfader: Crossfader::new(sample_rate),
            scratch_a: [StereoFrame::default(); MAX_SCRATCH_FRAMES],
            scratch_b: [StereoFrame::default(); MAX_SCRATCH_FRAMES],
            scratch_stems_a: [[StereoFrame::default(); MAX_SCRATCH_FRAMES]; NUM_STEMS],
            scratch_stems_b: [[StereoFrame::default(); MAX_SCRATCH_FRAMES]; NUM_STEMS],
        }
    }

    #[inline(always)]
    fn sync_control_targets(&mut self) {
        let ctrl = &self.controls;

        // Deck A Stem Gain Smoothers (Vocals, Drums, Bass, Other)
        self.stem_smoothers_a[0].set_target(
            ctrl.deck_a
                .stem_vocals
                .load(Ordering::Relaxed)
                .clamp(0.0_f32, 2.0_f32),
        );
        self.stem_smoothers_a[1].set_target(
            ctrl.deck_a
                .stem_drums
                .load(Ordering::Relaxed)
                .clamp(0.0_f32, 2.0_f32),
        );
        self.stem_smoothers_a[2].set_target(
            ctrl.deck_a
                .stem_bass
                .load(Ordering::Relaxed)
                .clamp(0.0_f32, 2.0_f32),
        );
        self.stem_smoothers_a[3].set_target(
            ctrl.deck_a
                .stem_other
                .load(Ordering::Relaxed)
                .clamp(0.0_f32, 2.0_f32),
        );

        // Deck A EQ & Channel Controls
        let low_a = ThreeBandIsolator::knob_to_linear_gain(
            ctrl.deck_a.eq_low_knob.load(Ordering::Relaxed),
        );
        let mid_a = ThreeBandIsolator::knob_to_linear_gain(
            ctrl.deck_a.eq_mid_knob.load(Ordering::Relaxed),
        );
        let high_a = ThreeBandIsolator::knob_to_linear_gain(
            ctrl.deck_a.eq_high_knob.load(Ordering::Relaxed),
        );
        self.isolator_a.set_band_gains_linear(low_a, mid_a, high_a);
        self.isolator_a.set_kills(
            ctrl.deck_a.kill_low.load(Ordering::Relaxed),
            ctrl.deck_a.kill_mid.load(Ordering::Relaxed),
            ctrl.deck_a.kill_high.load(Ordering::Relaxed),
        );
        self.trim_smoother_a
            .set_target(ctrl.deck_a.trim_gain.load(Ordering::Relaxed).clamp(0.0, 2.0));
        self.fader_smoother_a.set_target(
            ctrl.deck_a
                .channel_fader
                .load(Ordering::Relaxed)
                .clamp(0.0, 1.0),
        );

        // Deck B Stem Gain Smoothers (Vocals, Drums, Bass, Other)
        self.stem_smoothers_b[0].set_target(
            ctrl.deck_b
                .stem_vocals
                .load(Ordering::Relaxed)
                .clamp(0.0_f32, 2.0_f32),
        );
        self.stem_smoothers_b[1].set_target(
            ctrl.deck_b
                .stem_drums
                .load(Ordering::Relaxed)
                .clamp(0.0_f32, 2.0_f32),
        );
        self.stem_smoothers_b[2].set_target(
            ctrl.deck_b
                .stem_bass
                .load(Ordering::Relaxed)
                .clamp(0.0_f32, 2.0_f32),
        );
        self.stem_smoothers_b[3].set_target(
            ctrl.deck_b
                .stem_other
                .load(Ordering::Relaxed)
                .clamp(0.0_f32, 2.0_f32),
        );

        // Deck B EQ & Channel Controls
        let low_b = ThreeBandIsolator::knob_to_linear_gain(
            ctrl.deck_b.eq_low_knob.load(Ordering::Relaxed),
        );
        let mid_b = ThreeBandIsolator::knob_to_linear_gain(
            ctrl.deck_b.eq_mid_knob.load(Ordering::Relaxed),
        );
        let high_b = ThreeBandIsolator::knob_to_linear_gain(
            ctrl.deck_b.eq_high_knob.load(Ordering::Relaxed),
        );
        self.isolator_b.set_band_gains_linear(low_b, mid_b, high_b);
        self.isolator_b.set_kills(
            ctrl.deck_b.kill_low.load(Ordering::Relaxed),
            ctrl.deck_b.kill_mid.load(Ordering::Relaxed),
            ctrl.deck_b.kill_high.load(Ordering::Relaxed),
        );
        self.trim_smoother_b
            .set_target(ctrl.deck_b.trim_gain.load(Ordering::Relaxed).clamp(0.0, 2.0));
        self.fader_smoother_b.set_target(
            ctrl.deck_b
                .channel_fader
                .load(Ordering::Relaxed)
                .clamp(0.0, 1.0),
        );

        // Crossfader & Master Controls
        self.crossfader
            .set_position(ctrl.crossfader_pos.load(Ordering::Relaxed));
        self.crossfader.set_curve(CrossfaderCurve::from_u8(
            ctrl.crossfader_curve.load(Ordering::Relaxed),
        ));
        self.master_smoother
            .set_target(ctrl.master_gain.load(Ordering::Relaxed).clamp(0.0, 1.5));
        self.cue_level_smoother
            .set_target(ctrl.cue_level_gain.load(Ordering::Relaxed).clamp(0.0, 1.5));
    }

    /// Populates `self.scratch_a[..chunk_len]` either by summing the 4 weighted stems
    /// (when stem separation is enabled and buffered) or by falling back to the full-track queue.
    #[inline(always)]
    fn prepare_deck_a_chunk(&mut self, chunk_len: usize, stem_enabled_a: bool) {
        if stem_enabled_a && self.deck_a_stems.can_read_chunk(chunk_len) {
            for s in 0..NUM_STEMS {
                self.deck_a_stems.stems[s]
                    .pop_frames_into(&mut self.scratch_stems_a[s][..chunk_len]);
            }
            // Drain any parallel full-track fallback frames to keep queues time-aligned
            let _ = self.deck_a_consumer.discard_frames(chunk_len);

            for i in 0..chunk_len {
                let g_v = self.stem_smoothers_a[0].next_value();
                let g_d = self.stem_smoothers_a[1].next_value();
                let g_b = self.stem_smoothers_a[2].next_value();
                let g_o = self.stem_smoothers_a[3].next_value();

                let v = self.scratch_stems_a[0][i];
                let d = self.scratch_stems_a[1][i];
                let b = self.scratch_stems_a[2][i];
                let o = self.scratch_stems_a[3][i];

                self.scratch_a[i] = StereoFrame {
                    left: v.left * g_v + d.left * g_d + b.left * g_b + o.left * g_o,
                    right: v.right * g_v + d.right * g_d + b.right * g_b + o.right * g_o,
                };
            }
        } else {
            // Seamless fallback to standard full-track stereo buffer when disabled or buffering
            self.deck_a_consumer
                .pop_frames_into(&mut self.scratch_a[..chunk_len]);
            if !stem_enabled_a {
                self.deck_a_stems.discard_all(chunk_len);
            }
        }
    }

    /// Populates `self.scratch_b[..chunk_len]` either by summing the 4 weighted stems
    /// (when stem separation is enabled and buffered) or by falling back to the full-track queue.
    #[inline(always)]
    fn prepare_deck_b_chunk(&mut self, chunk_len: usize, stem_enabled_b: bool) {
        if stem_enabled_b && self.deck_b_stems.can_read_chunk(chunk_len) {
            for s in 0..NUM_STEMS {
                self.deck_b_stems.stems[s]
                    .pop_frames_into(&mut self.scratch_stems_b[s][..chunk_len]);
            }
            let _ = self.deck_b_consumer.discard_frames(chunk_len);

            for i in 0..chunk_len {
                let g_v = self.stem_smoothers_b[0].next_value();
                let g_d = self.stem_smoothers_b[1].next_value();
                let g_b = self.stem_smoothers_b[2].next_value();
                let g_o = self.stem_smoothers_b[3].next_value();

                let v = self.scratch_stems_b[0][i];
                let d = self.scratch_stems_b[1][i];
                let b = self.scratch_stems_b[2][i];
                let o = self.scratch_stems_b[3][i];

                self.scratch_b[i] = StereoFrame {
                    left: v.left * g_v + d.left * g_d + b.left * g_b + o.left * g_o,
                    right: v.right * g_v + d.right * g_d + b.right * g_b + o.right * g_o,
                };
            }
        } else {
            self.deck_b_consumer
                .pop_frames_into(&mut self.scratch_b[..chunk_len]);
            if !stem_enabled_b {
                self.deck_b_stems.discard_all(chunk_len);
            }
        }
    }
}

impl AudioOutputCallback for DjAudioCallback {
    type FrameType = (f32, oboe::Stereo);

    fn on_audio_ready(
        &mut self,
        _stream: &mut dyn AudioOutputStreamSafe,
        frames: &mut [(f32, f32)],
    ) -> DataCallbackResult {
        self.sync_control_targets();

        let stem_enabled_a = self.controls.deck_a.stem_enabled.load(Ordering::Acquire);
        let stem_enabled_b = self.controls.deck_b.stem_enabled.load(Ordering::Acquire);
        let pfl_a = self.controls.deck_a.pfl_cue_enabled.load(Ordering::Relaxed);
        let pfl_b = self.controls.deck_b.pfl_cue_enabled.load(Ordering::Relaxed);
        let split_cue = self.controls.split_cue_output.load(Ordering::Relaxed);
        let cue_mix_ratio = self
            .controls
            .cue_master_mix
            .load(Ordering::Relaxed)
            .clamp(0.0_f32, 1.0_f32);

        let total_frames = frames.len();
        let mut processed = 0usize;

        while processed < total_frames {
            let chunk_len = (total_frames - processed).min(MAX_SCRATCH_FRAMES);

            // 1. Fetch & sum 4-stem buffers (or fallback full-track stereo) into scratch_a / scratch_b
            self.prepare_deck_a_chunk(chunk_len, stem_enabled_a);
            self.prepare_deck_b_chunk(chunk_len, stem_enabled_b);

            let out_slice = &mut frames[processed..processed + chunk_len];

            for ((out_frame, src_a), src_b) in out_slice
                .iter_mut()
                .zip(self.scratch_a[..chunk_len].iter())
                .zip(self.scratch_b[..chunk_len].iter())
            {
                let trim_a = self.trim_smoother_a.next_value();
                let trim_b = self.trim_smoother_b.next_value();

                // 2. Pre-EQ Input Trim & 3-Band LR4 Isolator EQ
                let (eq_a_l, eq_a_r) = self
                    .isolator_a
                    .process_stereo(src_a.left * trim_a, src_a.right * trim_a);
                let (eq_b_l, eq_b_r) = self
                    .isolator_b
                    .process_stereo(src_b.left * trim_b, src_b.right * trim_b);

                // 3. PFL (Pre-Fade Listen) Cue Bus Summing (post-EQ, pre-channel-fader)
                let mut pfl_l = 0.0_f32;
                let mut pfl_r = 0.0_f32;
                if pfl_a {
                    pfl_l += eq_a_l;
                    pfl_r += eq_a_r;
                }
                if pfl_b {
                    pfl_l += eq_b_l;
                    pfl_r += eq_b_r;
                }

                // 4. Channel Faders & Crossfader Curve
                let ch_a = self.fader_smoother_a.next_value();
                let ch_b = self.fader_smoother_b.next_value();
                let (xf_a, xf_b) = self.crossfader.next_gains();

                let post_a_gain = ch_a * xf_a;
                let post_b_gain = ch_b * xf_b;

                let pgm_l = eq_a_l * post_a_gain + eq_b_l * post_b_gain;
                let pgm_r = eq_a_r * post_a_gain + eq_b_r * post_b_gain;

                // 5. Master Attenuation, Cue Mixing & Soft-Knee Limiter
                let master_gain = self.master_smoother.next_value();
                let cue_gain = self.cue_level_smoother.next_value();

                let master_l = pgm_l * master_gain;
                let master_r = pgm_r * master_gain;

                if split_cue {
                    let mono_master = (master_l + master_r) * 0.5_f32;
                    let mono_pfl = (pfl_l + pfl_r) * 0.5_f32;
                    let mono_cue_bus =
                        (mono_pfl * (1.0_f32 - cue_mix_ratio) + mono_master * cue_mix_ratio)
                            * cue_gain;

                    out_frame.0 = soft_knee_limit(mono_master);
                    out_frame.1 = soft_knee_limit(mono_cue_bus);
                } else {
                    let final_l = if pfl_a || pfl_b {
                        master_l * (1.0_f32 - cue_mix_ratio) + (pfl_l * cue_gain) * cue_mix_ratio
                    } else {
                        master_l
                    };
                    let final_r = if pfl_a || pfl_b {
                        master_r * (1.0_f32 - cue_mix_ratio) + (pfl_r * cue_gain) * cue_mix_ratio
                    } else {
                        master_r
                    };

                    out_frame.0 = soft_knee_limit(final_l);
                    out_frame.1 = soft_knee_limit(final_r);
                }
            }

            processed += chunk_len;
        }

        self.controls.record_frames(total_frames as u64);
        DataCallbackResult::Continue
    }

    fn on_error_before_close(
        &mut self,
        _stream: &mut dyn AudioOutputStreamSafe,
        error: Error,
    ) {
        self.controls.xrun_count.fetch_add(1, Ordering::Relaxed);
        log::error!("AAudio real-time stream error before close: {:?}", error);
    }

    fn on_error_after_close(
        &mut self,
        _stream: &mut dyn AudioOutputStreamSafe,
        error: Error,
    ) {
        log::error!("AAudio real-time stream error after close: {:?}", error);
    }
}
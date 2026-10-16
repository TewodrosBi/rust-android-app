pub mod dsp;
pub mod engine;
pub mod stems;

use crate::engine::ring_buffer::SharedMixerControls;
use crate::engine::stream::{DjAudioEngine, DECK_RING_BUFFER_FRAMES};
use crate::stems::qnn_engine::{HtpPerformanceMode, QnnSessionOptions, TensorPrecision};
use crate::stems::separator::StemSeparatorHandle;
use jni::objects::{JClass, JFloatArray, JString};
use jni::sys::{jboolean, jfloat, jint, jlong, JNI_TRUE};
use jni::JNIEnv;
use log::LevelFilter;
use portable_atomic::Ordering;
use std::fs;
use std::sync::{Arc, Mutex, Once};

static LOGGER_INIT: Once = Once::new();

pub struct EngineContext {
    engine: Mutex<DjAudioEngine>,
    controls: Arc<SharedMixerControls>,
    stem_worker: Mutex<Option<StemSeparatorHandle>>,
}

fn init_logging() {
    LOGGER_INIT.call_once(|| {
        android_logger::init_once(
            android_logger::Config::default()
                .with_max_level(LevelFilter::Info)
                .with_tag("S24UltraDJAudio"),
        );
    });
}

#[inline(always)]
unsafe fn with_context<F, R>(engine_ptr: jlong, default: R, f: F) -> R
where
    F: FnOnce(&EngineContext) -> R,
{
    if engine_ptr == 0 {
        return default;
    }
    let ctx = &*(engine_ptr as *const EngineContext);
    f(ctx)
}

/// 1. `nativeInit(sampleRate: Int, burstFrames: Int): Long`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeInit(
    _env: JNIEnv,
    _class: JClass,
    sample_rate: jint,
    burst_frames: jint,
) -> jlong {
    init_logging();

    match DjAudioEngine::new(sample_rate, burst_frames) {
        Ok(engine) => {
            let controls = Arc::clone(&engine.controls);
            let ctx = Box::new(EngineContext {
                engine: Mutex::new(engine),
                controls,
                stem_worker: Mutex::new(None),
            });
            Box::into_raw(ctx) as jlong
        }
        Err(err) => {
            log::error!("Failed to initialize DjAudioEngine: {:?}", err);
            0
        }
    }
}

/// 2. `nativeStart(enginePtr: Long): Boolean`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeStart(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
) -> jboolean {
    unsafe {
        with_context(engine_ptr, 0, |ctx| {
            if let Ok(mut eng) = ctx.engine.lock() {
                if eng.start().is_ok() {
                    return JNI_TRUE;
                }
            }
            0
        })
    }
}

/// 3. `nativeStop(enginePtr: Long): Boolean`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeStop(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
) -> jboolean {
    unsafe {
        with_context(engine_ptr, 0, |ctx| {
            if let Ok(mut eng) = ctx.engine.lock() {
                if eng.stop().is_ok() {
                    return JNI_TRUE;
                }
            }
            0
        })
    }
}

/// 4. `nativeRelease(enginePtr: Long)`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeRelease(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
) {
    if engine_ptr == 0 {
        return;
    }
    unsafe {
        let ctx = Box::from_raw(engine_ptr as *mut EngineContext);
        if let Ok(mut worker_guard) = ctx.stem_worker.lock() {
            if let Some(mut worker) = worker_guard.take() {
                worker.shutdown();
            }
        }
        if let Ok(mut eng) = ctx.engine.lock() {
            let _ = eng.stop();
        }
        drop(ctx);
    }
}

/// 5. `nativePushDeckAudio(enginePtr: Long, deckIndex: Int, pcmData: FloatArray, offset: Int, length: Int): Int`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativePushDeckAudio(
    env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    deck_index: jint,
    pcm_data: JFloatArray,
    offset: jint,
    length: jint,
) -> jint {
    if engine_ptr == 0 || length <= 0 || offset < 0 {
        return 0;
    }

    let copy_len = (length as usize).min(16_384);
    let mut stack_buf = [0.0_f32; 16_384];
    if env
        .get_float_array_region(&pcm_data, offset, &mut stack_buf[..copy_len])
        .is_err()
    {
        return 0;
    }

    unsafe {
        with_context(engine_ptr, 0, |ctx| {
            if let Ok(mut worker_guard) = ctx.stem_worker.lock() {
                if let Some(ref mut worker) = *worker_guard {
                    if deck_index == 0 {
                        let _ = worker
                            .raw_deck_a_producer
                            .push_interleaved(&stack_buf[..copy_len]);
                    } else {
                        let _ = worker
                            .raw_deck_b_producer
                            .push_interleaved(&stack_buf[..copy_len]);
                    }
                }
            }

            if let Ok(mut eng) = ctx.engine.lock() {
                let written = if deck_index == 0 {
                    eng.deck_a_producer.push_interleaved(&stack_buf[..copy_len])
                } else {
                    eng.deck_b_producer.push_interleaved(&stack_buf[..copy_len])
                };
                written as jint
            } else {
                0
            }
        })
    }
}

/// 6. `nativeSetDeckEq(enginePtr: Long, deckIndex: Int, low: Float, mid: Float, high: Float)`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeSetDeckEq(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    deck_index: jint,
    low: jfloat,
    mid: jfloat,
    high: jfloat,
) {
    unsafe {
        with_context(engine_ptr, (), |ctx| {
            let deck = if deck_index == 0 {
                &ctx.controls.deck_a
            } else {
                &ctx.controls.deck_b
            };
            deck.eq_low_knob.store(low, Ordering::Relaxed);
            deck.eq_mid_knob.store(mid, Ordering::Relaxed);
            deck.eq_high_knob.store(high, Ordering::Relaxed);
        });
    }
}

/// 7. `nativeSetDeckKills(enginePtr: Long, deckIndex: Int, killLow: Boolean, killMid: Boolean, killHigh: Boolean)`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeSetDeckKills(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    deck_index: jint,
    kill_low: jboolean,
    kill_mid: jboolean,
    kill_high: jboolean,
) {
    unsafe {
        with_context(engine_ptr, (), |ctx| {
            let deck = if deck_index == 0 {
                &ctx.controls.deck_a
            } else {
                &ctx.controls.deck_b
            };
            deck.kill_low.store(kill_low == JNI_TRUE, Ordering::Relaxed);
            deck.kill_mid.store(kill_mid == JNI_TRUE, Ordering::Relaxed);
            deck.kill_high
                .store(kill_high == JNI_TRUE, Ordering::Relaxed);
        });
    }
}

/// 8. `nativeSetDeckGains(enginePtr: Long, deckIndex: Int, trimGain: Float, channelFader: Float)`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeSetDeckGains(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    deck_index: jint,
    trim_gain: jfloat,
    channel_fader: jfloat,
) {
    unsafe {
        with_context(engine_ptr, (), |ctx| {
            let deck = if deck_index == 0 {
                &ctx.controls.deck_a
            } else {
                &ctx.controls.deck_b
            };
            deck.trim_gain.store(trim_gain, Ordering::Relaxed);
            deck.channel_fader.store(channel_fader, Ordering::Relaxed);
        });
    }
}

/// 9. `nativeSetCrossfader(enginePtr: Long, position: Float, curveType: Int)`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeSetCrossfader(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    position: jfloat,
    curve_type: jint,
) {
    unsafe {
        with_context(engine_ptr, (), |ctx| {
            ctx.controls
                .crossfader_pos
                .store(position, Ordering::Relaxed);
            ctx.controls
                .crossfader_curve
                .store(curve_type as u8, Ordering::Relaxed);
        });
    }
}

/// 10. `nativeSetMasterAndCue(enginePtr: Long, masterGain: Float, cueLevelGain: Float, cueMix: Float, splitCue: Boolean)`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeSetMasterAndCue(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    master_gain: jfloat,
    cue_level_gain: jfloat,
    cue_mix: jfloat,
    split_cue: jboolean,
) {
    unsafe {
        with_context(engine_ptr, (), |ctx| {
            ctx.controls
                .master_gain
                .store(master_gain, Ordering::Relaxed);
            ctx.controls
                .cue_level_gain
                .store(cue_level_gain, Ordering::Relaxed);
            ctx.controls
                .cue_master_mix
                .store(cue_mix, Ordering::Relaxed);
            ctx.controls
                .split_cue_output
                .store(split_cue == JNI_TRUE, Ordering::Relaxed);
        });
    }
}

/// 11. `nativeGetFramesProcessed(enginePtr: Long): Long`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeGetFramesProcessed(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
) -> jlong {
    unsafe {
        with_context(engine_ptr, 0, |ctx| {
            ctx.controls
                .callback_frames_processed
                .load(Ordering::Relaxed) as jlong
        })
    }
}

/// 12. `nativeInitStemEngine(enginePtr: Long, modelPath: String): Boolean`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeInitStemEngine(
    mut env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    model_path: JString,
) -> jboolean {
    let path_string: Option<String> = if !model_path.is_null() {
        env.get_string(&model_path).ok().map(Into::into)
    } else {
        None
    };

    let context_blob = path_string.and_then(|p| fs::read(p).ok());

    unsafe {
        with_context(engine_ptr, 0, |ctx| {
            let mut worker_guard = match ctx.stem_worker.lock() {
                Ok(g) => g,
                Err(_) => return 0,
            };

            if worker_guard.is_some() {
                return JNI_TRUE;
            }

            let mut eng = match ctx.engine.lock() {
                Ok(e) => e,
                Err(_) => return 0,
            };

            let (stems_a, stems_b) = match (
                eng.deck_a_stem_producers.take(),
                eng.deck_b_stem_producers.take(),
            ) {
                (Some(a), Some(b)) => (a, b),
                _ => return 0,
            };

            let options = QnnSessionOptions {
                perf_mode: HtpPerformanceMode::SustainedHighPerformance,
                precision: TensorPrecision::Fp16,
                ..QnnSessionOptions::default()
            };

            let worker = StemSeparatorHandle::spawn(
                stems_a,
                stems_b,
                eng.sample_rate as f32,
                DECK_RING_BUFFER_FRAMES,
                options,
                context_blob.as_deref(),
            );

            *worker_guard = Some(worker);
            JNI_TRUE
        })
    }
}

/// 13. `nativeToggleStemSeparation(enginePtr: Long, deckIndex: Int, enabled: Boolean)`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeToggleStemSeparation(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    deck_index: jint,
    enabled: jboolean,
) {
    let is_enabled = enabled == JNI_TRUE;
    unsafe {
        with_context(engine_ptr, (), |ctx| {
            let deck = if deck_index == 0 {
                &ctx.controls.deck_a
            } else {
                &ctx.controls.deck_b
            };
            deck.stem_enabled.store(is_enabled, Ordering::Release);

            if let Ok(worker_guard) = ctx.stem_worker.lock() {
                if let Some(ref worker) = *worker_guard {
                    let stem_deck = if deck_index == 0 {
                        &worker.stem_controls.deck_a
                    } else {
                        &worker.stem_controls.deck_b
                    };
                    stem_deck.enabled.store(is_enabled, Ordering::Release);
                }
            }
        });
    }
}

/// 14. `nativeSetStemGains(enginePtr: Long, deckIndex: Int, vocals: Float, drums: Float, bass: Float, other: Float)`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeSetStemGains(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    deck_index: jint,
    vocals: jfloat,
    drums: jfloat,
    bass: jfloat,
    other: jfloat,
) {
    let v = vocals.clamp(0.0_f32, 2.0_f32);
    let d = drums.clamp(0.0_f32, 2.0_f32);
    let b = bass.clamp(0.0_f32, 2.0_f32);
    let o = other.clamp(0.0_f32, 2.0_f32);

    unsafe {
        with_context(engine_ptr, (), |ctx| {
            let deck = if deck_index == 0 {
                &ctx.controls.deck_a
            } else {
                &ctx.controls.deck_b
            };
            deck.stem_vocals.store(v, Ordering::Relaxed);
            deck.stem_drums.store(d, Ordering::Relaxed);
            deck.stem_bass.store(b, Ordering::Relaxed);
            deck.stem_other.store(o, Ordering::Relaxed);
        });
    }
}

/// 15. `nativeGetLastStemInferenceUs(enginePtr: Long): Long`
#[no_mangle]
pub extern "system" fn Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeGetLastStemInferenceUs(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
) -> jlong {
    unsafe {
        with_context(engine_ptr, 0, |ctx| {
            if let Ok(worker_guard) = ctx.stem_worker.lock() {
                if let Some(ref worker) = *worker_guard {
                    let us_a = worker
                        .stem_controls
                        .deck_a
                        .last_inference_us
                        .load(Ordering::Relaxed);
                    let us_b = worker
                        .stem_controls
                        .deck_b
                        .last_inference_us
                        .load(Ordering::Relaxed);
                    return us_a.max(us_b) as jlong;
                }
            }
            0
        })
    }
}
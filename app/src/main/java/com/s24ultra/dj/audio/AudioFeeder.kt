package com.s24ultra.dj.audio

import android.os.Process
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin

class AudioFeeder(
    private val engine: NativeAudioEngine,
    private val sampleRate: Int = NativeAudioEngine.DEFAULT_SAMPLE_RATE,
    private val qnnContextBinaryPath: String = "/data/local/tmp/htp_v75_4stem.bin"
) {
    companion object {
        private const val TAG = "S24UltraAudioFeeder"

        // 1024 stereo frames (2048 interleaved floats) per push = 1 stem hop (21.33 ms @ 48 kHz)
        private const val CHUNK_FRAMES = 1024
        private const val CHUNK_FLOATS = CHUNK_FRAMES * 2

        // Maximum acceptable Hexagon HTP v75 inference time for a 1024-frame hop at 48 kHz
        private const val HOP_DEADLINE_US = 21_333L
    }

    private val isRunning = AtomicBoolean(false)
    private var feederThread: Thread? = null

    // Pre-allocated PCM buffers to guarantee zero GC allocations inside the real-time loop
    private val pcmBufferDeckA = FloatArray(CHUNK_FLOATS)
    private val pcmBufferDeckB = FloatArray(CHUNK_FLOATS)

    // Continuous phase accumulators in [0.0, 2*PI) to prevent precision loss over long runs
    private var phaseSubA = 0.0
    private var phaseVocalA = 0.0
    private var phasePadL_A = 0.0
    private var phasePadR_A = 0.0
    private var beatSampleCounterA = 0L

    private var phaseSubB = 0.0
    private var phaseVocalB = 0.0
    private var phaseHighB = 0.0
    private var beatSampleCounterB = 0L

    /**
     * Initializes the Hexagon NPU stem engine, starts the AAudio stream, and spawns
     * the high-priority real-time feeder & verification thread.
     */
    fun startVerificationHarness(): Boolean {
        if (!engine.isInitialized) {
            Log.e(TAG, "Cannot start AudioFeeder: NativeAudioEngine is not initialized.")
            return false
        }

        if (!isRunning.compareAndSet(false, true)) {
            Log.w(TAG, "AudioFeeder is already running.")
            return true
        }

        val stemReady = engine.initStemEngine(qnnContextBinaryPath)
        Log.i(TAG, "Hexagon HTP v75 Stem Engine Initialized: $stemReady (modelPath=$qnnContextBinaryPath)")

        // Set initial mixer state
        engine.setMasterAndCue(masterGain = 0.85f, cueLevelGain = 0.85f, cueMix = 0.0f, splitCue = false)
        engine.setDeckGains(NativeAudioEngine.DECK_A, trimGain = 1.0f, channelFader = 1.0f)
        engine.setDeckGains(NativeAudioEngine.DECK_B, trimGain = 1.0f, channelFader = 1.0f)
        engine.setDeckEq(NativeAudioEngine.DECK_A, low = 0.5f, mid = 0.5f, high = 0.5f)
        engine.setDeckEq(NativeAudioEngine.DECK_B, low = 0.5f, mid = 0.5f, high = 0.5f)
        engine.setCrossfader(position = 0.0f, curveType = NativeAudioEngine.CURVE_SMOOTH)

        if (!engine.start()) {
            Log.e(TAG, "Failed to start native AAudio stream.")
            isRunning.set(false)
            return false
        }

        feederThread = Thread({ runFeedAndVerifyLoop() }, "S24Ultra-AudioFeeder").apply {
            isDaemon = true
            start()
        }

        return true
    }

    /**
     * Cleanly halts the feeder thread and stops the AAudio stream.
     */
    fun stopVerificationHarness() {
        if (isRunning.compareAndSet(true, false)) {
            try {
                feederThread?.join(1_500L)
            } catch (ignored: InterruptedException) {
                Thread.currentThread().interrupt()
            } finally {
                feederThread = null
                engine.stop()
            }
        }
    }

    private fun runFeedAndVerifyLoop() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)

        val startWallMs = SystemClock.elapsedRealtime()
        var lastTelemetryMs = startWallMs
        var lastFramesProcessed = engine.getFramesProcessed()

        var totalFramesPushedA = 0L
        var totalFramesPushedB = 0L

        // Pre-fill 4 hops (4096 stereo frames = ~85 ms) into Deck A and Deck B before steady-state
        repeat(4) {
            synthesizeDeckABlock(pcmBufferDeckA, CHUNK_FRAMES)
            synthesizeDeckBBlock(pcmBufferDeckB, CHUNK_FRAMES)
            totalFramesPushedA += engine.pushDeckAudio(NativeAudioEngine.DECK_A, pcmBufferDeckA, 0, CHUNK_FLOATS)
            totalFramesPushedB += engine.pushDeckAudio(NativeAudioEngine.DECK_B, pcmBufferDeckB, 0, CHUNK_FLOATS)
        }

        Log.i(TAG, "Pre-fill complete (DeckA=$totalFramesPushedA frames, DeckB=$totalFramesPushedB frames)")

        while (isRunning.get()) {
            val nowMs = SystemClock.elapsedRealtime()
            val elapsedSec = (nowMs - startWallMs) / 1000.0f

            // 1. Keep Deck A and Deck B lock-free ring buffers topped up without busy-waiting
            var pushedAny = false

            synthesizeDeckABlock(pcmBufferDeckA, CHUNK_FRAMES)
            val writtenA = engine.pushDeckAudio(NativeAudioEngine.DECK_A, pcmBufferDeckA, 0, CHUNK_FLOATS)
            if (writtenA > 0) {
                totalFramesPushedA += writtenA
                pushedAny = true
            }

            synthesizeDeckBBlock(pcmBufferDeckB, CHUNK_FRAMES)
            val writtenB = engine.pushDeckAudio(NativeAudioEngine.DECK_B, pcmBufferDeckB, 0, CHUNK_FLOATS)
            if (writtenB > 0) {
                totalFramesPushedB += writtenB
                pushedAny = true
            }

            // 2. Drive automated DSP & NPU Stem verification stages
            driveAutomationScenario(elapsedSec)

            // 3. Emit 1 Hz real-time latency & throughput telemetry
            if (nowMs - lastTelemetryMs >= 1_000L) {
                val currentFrames = engine.getFramesProcessed()
                val deltaFrames = currentFrames - lastFramesProcessed
                val deltaSec = (nowMs - lastTelemetryMs) / 1000.0
                val measuredSampleRate = (deltaFrames / deltaSec).toInt()
                val stemLatencyUs = engine.getLastStemInferenceUs()
                val cpuHeadroomPct = if (stemLatencyUs > 0) {
                    100.0 - ((stemLatencyUs.toDouble() / HOP_DEADLINE_US.toDouble()) * 100.0)
                } else {
                    100.0
                }

                Log.i(
                    TAG,
                    String.format(
                        "[TELEMETRY t=%.1fs] AAudio Rate: %d Hz (total=%d frames) | HTP v75 Stem Hop: %d us (deadline=%d us, NPU headroom=%.1f%%)",
                        elapsedSec,
                        measuredSampleRate,
                        currentFrames,
                        stemLatencyUs,
                        HOP_DEADLINE_US,
                        cpuHeadroomPct
                    )
                )

                lastTelemetryMs = nowMs
                lastFramesProcessed = currentFrames
            }

            if (!pushedAny) {
                SystemClock.sleep(4L)
            } else {
                SystemClock.sleep(8L)
            }
        }
    }

    /**
     * Automates isolator sweeps, kill switches, 4-stem NPU toggling, and crossfader cuts
     * across a repeating 20-second verification cycle.
     */
    private fun driveAutomationScenario(elapsedSec: Float) {
        val cycleSec = elapsedSec % 20.0f
        val lfo = 0.5f + 0.5f * sin(elapsedSec * 2.0f * PI.toFloat())

        when {
            // Stage 1 (0 - 5s): Full-track stereo mix at flat unity EQ
            cycleSec < 5.0f -> {
                engine.toggleStemSeparation(NativeAudioEngine.DECK_A, false)
                engine.toggleStemSeparation(NativeAudioEngine.DECK_B, false)
                engine.setDeckKills(NativeAudioEngine.DECK_A, killLow = false, killMid = false, killHigh = false)
                engine.setDeckEq(NativeAudioEngine.DECK_A, low = 0.5f, mid = 0.5f, high = 0.5f)
                engine.setCrossfader(position = 0.0f, curveType = NativeAudioEngine.CURVE_SMOOTH)
            }

            // Stage 2 (5 - 10s): 3-Band LR4 Isolator sweep & hard bass kill test on Deck A
            cycleSec < 10.0f -> {
                val killBass = cycleSec > 8.0f
                engine.setDeckKills(NativeAudioEngine.DECK_A, killLow = killBass, killMid = false, killHigh = false)
                engine.setDeckEq(
                    deckIndex = NativeAudioEngine.DECK_A,
                    low = lfo,
                    mid = 1.0f - lfo,
                    high = 0.5f + 0.4f * sin(elapsedSec * 4.0f)
                )
                // Isolate Deck A on the crossfader
                engine.setCrossfader(position = -1.0f, curveType = NativeAudioEngine.CURVE_SMOOTH)
            }

            // Stage 3 (10 - 16s): Activate Hexagon HTP v75 4-Stem Separation on Deck A & solo stems sequentially
            cycleSec < 16.0f -> {
                engine.setDeckKills(NativeAudioEngine.DECK_A, killLow = false, killMid = false, killHigh = false)
                engine.setDeckEq(NativeAudioEngine.DECK_A, low = 0.5f, mid = 0.5f, high = 0.5f)
                engine.toggleStemSeparation(NativeAudioEngine.DECK_A, true)

                when {
                    // Solo Vocals
                    cycleSec < 11.5f -> engine.setStemGains(NativeAudioEngine.DECK_A, 1.0f, 0.0f, 0.0f, 0.0f)
                    // Solo Drums
                    cycleSec < 13.0f -> engine.setStemGains(NativeAudioEngine.DECK_A, 0.0f, 1.0f, 0.0f, 0.0f)
                    // Solo Bass
                    cycleSec < 14.5f -> engine.setStemGains(NativeAudioEngine.DECK_A, 0.0f, 0.0f, 1.0f, 0.0f)
                    // Solo Other (Harmonic Pad)
                    else -> engine.setStemGains(NativeAudioEngine.DECK_A, 0.0f, 0.0f, 0.0f, 1.0f)
                }
            }

            // Stage 4 (16 - 20s): Dual-Deck Stem Mashup (Deck A Vocals + Drums over Deck B Bass + Other)
            else -> {
                engine.toggleStemSeparation(NativeAudioEngine.DECK_A, true)
                engine.toggleStemSeparation(NativeAudioEngine.DECK_B, true)
                engine.setStemGains(NativeAudioEngine.DECK_A, vocals = 1.1f, drums = 1.0f, bass = 0.0f, other = 0.0f)
                engine.setStemGains(NativeAudioEngine.DECK_B, vocals = 0.0f, drums = 0.0f, bass = 1.1f, other = 1.0f)

                val scratchPos = sin(elapsedSec * 6.0f * PI.toFloat()).coerceIn(-1.0f, 1.0f)
                engine.setCrossfader(position = scratchPos, curveType = NativeAudioEngine.CURVE_SCRATCH_CUT)
            }
        }
    }

    /**
     * Synthesizes a multi-component stereo signal into [outInterleaved] for Deck A:
     * - Bass (65.41 Hz C2 sub-bass)
     * - Drums (120 BPM transient impulse click + exponential body decay)
     * - Vocals (523.25 Hz C5 center-panned formant-modulated tone)
     * - Other (Wide stereo detuned harmonic pad at 329.63 Hz / 392.00 Hz)
     */
    private fun synthesizeDeckABlock(outInterleaved: FloatArray, frames: Int) {
        val twoPi = 2.0 * PI
        val invSr = 1.0 / sampleRate.toDouble()
        val samplesPerBeat = (sampleRate / 2).toLong() // 120 BPM = 2 beats/sec = 24,000 frames/beat

        val incSub = twoPi * 65.41 * invSr
        val incVocal = twoPi * 523.25 * invSr
        val incPadL = twoPi * 329.63 * invSr
        val incPadR = twoPi * 392.00 * invSr

        for (i in 0 until frames) {
            val beatPos = (beatSampleCounterA % samplesPerBeat).toDouble()
            beatSampleCounterA++

            // Percussive transient burst at start of each beat
            val drumEnv = exp(-beatPos / (sampleRate * 0.018))
            val drumSample = (sin(beatPos * 0.45) * drumEnv * 0.30).toFloat()

            // Sub-bass fundamental
            val bassSample = (sin(phaseSubA) * 0.30).toFloat()

            // Center-panned Vocal harmonic
            val vocalSample = (sin(phaseVocalA) * (0.20 + 0.05 * cos(phaseSubA * 0.25))).toFloat()

            // Stereo accompaniment pad
            val padL = (sin(phasePadL_A) * 0.15).toFloat()
            val padR = (sin(phasePadR_A) * 0.15).toFloat()

            val baseIdx = i * 2
            outInterleaved[baseIdx] = (bassSample + drumSample + vocalSample + padL).coerceIn(-0.98f, 0.98f)
            outInterleaved[baseIdx + 1] = (bassSample + drumSample + vocalSample + padR).coerceIn(-0.98f, 0.98f)

            phaseSubA = (phaseSubA + incSub) % twoPi
            phaseVocalA = (phaseVocalA + incVocal) % twoPi
            phasePadL_A = (phasePadL_A + incPadL) % twoPi
            phasePadR_A = (phasePadR_A + incPadR) % twoPi
        }
    }

    /**
     * Synthesizes a complementary stereo signal into [outInterleaved] for Deck B:
     * - Bass (55.0 Hz A1 sub-bass)
     * - Off-beat percussive snare/hat transient
     * - Vocals (659.25 Hz E5 center tone)
     * - Stereo high harmonic shimmer (1318.5 Hz)
     */
    private fun synthesizeDeckBBlock(outInterleaved: FloatArray, frames: Int) {
        val twoPi = 2.0 * PI
        val invSr = 1.0 / sampleRate.toDouble()
        val samplesPerBeat = (sampleRate / 2).toLong()
        val halfBeat = samplesPerBeat / 2

        val incSub = twoPi * 55.0 * invSr
        val incVocal = twoPi * 659.25 * invSr
        val incHigh = twoPi * 1318.51 * invSr

        for (i in 0 until frames) {
            val shiftedBeatPos = ((beatSampleCounterB + halfBeat) % samplesPerBeat).toDouble()
            beatSampleCounterB++

            val hatEnv = exp(-shiftedBeatPos / (sampleRate * 0.010))
            val drumSample = (sin(shiftedBeatPos * 1.15) * hatEnv * 0.25).toFloat()

            val bassSample = (sin(phaseSubB) * 0.32).toFloat()
            val vocalSample = (sin(phaseVocalB) * 0.18).toFloat()
            val shimmerL = (sin(phaseHighB) * 0.14).toFloat()
            val shimmerR = (cos(phaseHighB) * 0.14).toFloat()

            val baseIdx = i * 2
            outInterleaved[baseIdx] = (bassSample + drumSample + vocalSample + shimmerL).coerceIn(-0.98f, 0.98f)
            outInterleaved[baseIdx + 1] = (bassSample + drumSample + vocalSample + shimmerR).coerceIn(-0.98f, 0.98f)

            phaseSubB = (phaseSubB + incSub) % twoPi
            phaseVocalB = (phaseVocalB + incVocal) % twoPi
            phaseHighB = (phaseHighB + incHigh) % twoPi
        }
    }
}
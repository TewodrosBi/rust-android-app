package com.s24ultra.dj.audio

import android.content.Context
import android.media.AudioManager
import android.util.Log
import java.io.Closeable
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class NativeAudioEngine : Closeable {

    companion object {
        private const val TAG = "NativeAudioEngine"

        const val DECK_A: Int = 0
        const val DECK_B: Int = 1

        const val CURVE_SMOOTH: Int = 0
        const val CURVE_LINEAR: Int = 1
        const val CURVE_SCRATCH_CUT: Int = 2

        const val DEFAULT_SAMPLE_RATE: Int = 48_000
        const val DEFAULT_BURST_FRAMES: Int = 96

        private val librariesLoaded = AtomicBoolean(false)

        private fun ensureNativeLibrariesLoaded() {
            if (librariesLoaded.compareAndSet(false, true)) {
                try {
                    System.loadLibrary("c++_shared")
                    System.loadLibrary("s24_dj_core")
                    Log.i(TAG, "Successfully loaded libc++_shared.so and libs24_dj_core.so")
                } catch (e: UnsatisfiedLinkError) {
                    librariesLoaded.set(false)
                    Log.e(TAG, "CRITICAL: Failed to load native audio libraries", e)
                    throw e
                }
            }
        }
    }

    init {
        ensureNativeLibrariesLoaded()
    }

    private val engineHandle = AtomicLong(0L)

    val enginePtr: Long
        get() = engineHandle.get()

    val isInitialized: Boolean
        get() = engineHandle.get() != 0L

    /**
     * Queries the Galaxy S24 Ultra hardware output properties via [AudioManager]
     * and initializes the native AAudio Exclusive MMAP engine.
     */
    fun initializeFromHardware(context: Context): Boolean {
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager
        val hwSampleRate = audioManager
            ?.getProperty(AudioManager.PROPERTY_OUTPUT_SAMPLE_RATE)
            ?.toIntOrNull()
            ?: DEFAULT_SAMPLE_RATE

        val hwBurstFrames = audioManager
            ?.getProperty(AudioManager.PROPERTY_OUTPUT_FRAMES_PER_BUFFER)
            ?.toIntOrNull()
            ?: DEFAULT_BURST_FRAMES

        return initialize(hwSampleRate, hwBurstFrames)
    }

    /**
     * Initializes the native engine with explicit [sampleRate] and [burstFrames].
     */
    @Synchronized
    fun initialize(
        sampleRate: Int = DEFAULT_SAMPLE_RATE,
        burstFrames: Int = DEFAULT_BURST_FRAMES
    ): Boolean {
        if (engineHandle.get() != 0L) {
            Log.w(TAG, "NativeAudioEngine is already initialized (ptr=0x${engineHandle.get().toString(16)})")
            return true
        }

        val ptr = nativeInit(sampleRate, burstFrames)
        if (ptr == 0L) {
            Log.e(TAG, "nativeInit($sampleRate, $burstFrames) returned nullptr")
            return false
        }

        engineHandle.set(ptr)
        Log.i(
            TAG,
            "Initialized S24 Ultra NativeAudioEngine (ptr=0x${ptr.toString(16)}, rate=$sampleRate Hz, burst=$burstFrames frames)"
        )
        return true
    }

    @Synchronized
    fun start(): Boolean {
        val ptr = engineHandle.get()
        if (ptr == 0L) return false
        return nativeStart(ptr)
    }

    @Synchronized
    fun stop(): Boolean {
        val ptr = engineHandle.get()
        if (ptr == 0L) return false
        return nativeStop(ptr)
    }

    @Synchronized
    override fun close() {
        val ptr = engineHandle.getAndSet(0L)
        if (ptr != 0L) {
            nativeStop(ptr)
            nativeRelease(ptr)
            Log.i(TAG, "Released NativeAudioEngine (ptr=0x${ptr.toString(16)})")
        }
    }

    // --- High-Level Real-Time Dispatch Helpers ---

    fun pushDeckAudio(deckIndex: Int, pcmData: FloatArray, offset: Int = 0, length: Int = pcmData.size): Int {
        val ptr = engineHandle.get()
        if (ptr == 0L) return 0
        return nativePushDeckAudio(ptr, deckIndex, pcmData, offset, length)
    }

    fun setDeckEq(deckIndex: Int, low: Float, mid: Float, high: Float) {
        val ptr = engineHandle.get()
        if (ptr != 0L) {
            nativeSetDeckEq(ptr, deckIndex, low, mid, high)
        }
    }

    fun setDeckKills(deckIndex: Int, killLow: Boolean, killMid: Boolean, killHigh: Boolean) {
        val ptr = engineHandle.get()
        if (ptr != 0L) {
            nativeSetDeckKills(ptr, deckIndex, killLow, killMid, killHigh)
        }
    }

    fun setDeckGains(deckIndex: Int, trimGain: Float, channelFader: Float) {
        val ptr = engineHandle.get()
        if (ptr != 0L) {
            nativeSetDeckGains(ptr, deckIndex, trimGain, channelFader)
        }
    }

    fun setCrossfader(position: Float, curveType: Int = CURVE_SMOOTH) {
        val ptr = engineHandle.get()
        if (ptr != 0L) {
            nativeSetCrossfader(ptr, position, curveType)
        }
    }

    fun setMasterAndCue(
        masterGain: Float,
        cueLevelGain: Float,
        cueMix: Float,
        splitCue: Boolean
    ) {
        val ptr = engineHandle.get()
        if (ptr != 0L) {
            nativeSetMasterAndCue(ptr, masterGain, cueLevelGain, cueMix, splitCue)
        }
    }

    fun getFramesProcessed(): Long {
        val ptr = engineHandle.get()
        if (ptr == 0L) return 0L
        return nativeGetFramesProcessed(ptr)
    }

    fun initStemEngine(modelPath: String): Boolean {
        val ptr = engineHandle.get()
        if (ptr == 0L) return false
        return nativeInitStemEngine(ptr, modelPath)
    }

    fun toggleStemSeparation(deckIndex: Int, enabled: Boolean) {
        val ptr = engineHandle.get()
        if (ptr != 0L) {
            nativeToggleStemSeparation(ptr, deckIndex, enabled)
        }
    }

    fun setStemGains(
        deckIndex: Int,
        vocals: Float,
        drums: Float,
        bass: Float,
        other: Float
    ) {
        val ptr = engineHandle.get()
        if (ptr != 0L) {
            nativeSetStemGains(ptr, deckIndex, vocals, drums, bass, other)
        }
    }

    fun getLastStemInferenceUs(): Long {
        val ptr = engineHandle.get()
        if (ptr == 0L) return 0L
        return nativeGetLastStemInferenceUs(ptr)
    }

    // =========================================================================
    // 15 Exact JNI External Bindings Linked to libs24_dj_core.so
    // =========================================================================

    external fun nativeInit(sampleRate: Int, burstFrames: Int): Long

    external fun nativeStart(enginePtr: Long): Boolean

    external fun nativeStop(enginePtr: Long): Boolean

    external fun nativeRelease(enginePtr: Long)

    external fun nativePushDeckAudio(
        enginePtr: Long,
        deckIndex: Int,
        pcmData: FloatArray,
        offset: Int,
        length: Int
    ): Int

    external fun nativeSetDeckEq(
        enginePtr: Long,
        deckIndex: Int,
        low: Float,
        mid: Float,
        high: Float
    )

    external fun nativeSetDeckKills(
        enginePtr: Long,
        deckIndex: Int,
        killLow: Boolean,
        killMid: Boolean,
        killHigh: Boolean
    )

    external fun nativeSetDeckGains(
        enginePtr: Long,
        deckIndex: Int,
        trimGain: Float,
        channelFader: Float
    )

    external fun nativeSetCrossfader(
        enginePtr: Long,
        position: Float,
        curveType: Int
    )

    external fun nativeSetMasterAndCue(
        enginePtr: Long,
        masterGain: Float,
        cueLevelGain: Float,
        cueMix: Float,
        splitCue: Boolean
    )

    external fun nativeGetFramesProcessed(enginePtr: Long): Long

    external fun nativeInitStemEngine(
        enginePtr: Long,
        modelPath: String
    ): Boolean

    external fun nativeToggleStemSeparation(
        enginePtr: Long,
        deckIndex: Int,
        enabled: Boolean
    )

    external fun nativeSetStemGains(
        enginePtr: Long,
        deckIndex: Int,
        vocals: Float,
        drums: Float,
        bass: Float,
        other: Float
    )

    external fun nativeGetLastStemInferenceUs(enginePtr: Long): Long
}
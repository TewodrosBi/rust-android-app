package com.s24ultra.dj

import android.app.Activity
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import com.s24ultra.dj.audio.NativeAudioEngine
import kotlin.concurrent.thread
import kotlin.math.sin

class MainActivity : Activity() {

    private var engine: NativeAudioEngine? = null
    @Volatile private var isPlaying = false
    private var feedThread: Thread? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 80, 48, 48)
        }

        val statusText = TextView(this).apply {
            text = "Snapdragon 8 Gen 3 DJ Core\nStatus: Initializing..."
            textSize = 18f
        }
        layout.addView(statusText)

        val toggleButton = Button(this).apply {
            text = "Start Audio Test"
            isEnabled = false
            setOnClickListener {
                val eng = engine ?: return@setOnClickListener
                if (!isPlaying) {
                    if (eng.start()) {
                        isPlaying = true
                        text = "Stop Audio"
                        statusText.text = "AAudio MMAP: ACTIVE\nStreaming 440Hz Sine Tone on Deck A"
                        startToneGenerator(eng)
                    } else {
                        statusText.text = "Failed to start AAudio stream."
                    }
                } else {
                    isPlaying = false
                    feedThread?.join(200)
                    eng.stop()
                    text = "Start Audio Test"
                    statusText.text = "AAudio MMAP: STOPPED"
                }
            }
        }
        layout.addView(toggleButton)

        val eqLabel = TextView(this).apply { text = "\nDeck A Bass Gain: 1.0x" }
        layout.addView(eqLabel)

        val eqSlider = SeekBar(this).apply {
            max = 200
            progress = 100
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                    val gain = progress / 100f
                    eqLabel.text = "\nDeck A Bass Gain: ${"%.2f".format(gain)}x"
                    engine?.setDeckEq(NativeAudioEngine.DECK_A, low = gain, mid = 1.0f, high = 1.0f)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }
        layout.addView(eqSlider)

        setContentView(layout)

        thread {
            try {
                val eng = NativeAudioEngine()
                if (eng.initializeFromHardware(this@MainActivity)) {
                    engine = eng
                    eng.setMasterAndCue(masterGain = 0.8f, cueLevelGain = 0.0f, cueMix = 0.0f, splitCue = false)
                    eng.setDeckGains(NativeAudioEngine.DECK_A, trimGain = 1.0f, channelFader = 1.0f)
                    eng.setCrossfader(0.0f, NativeAudioEngine.CURVE_SMOOTH)

                    runOnUiThread {
                        toggleButton.isEnabled = true
                        statusText.text = "Snapdragon 8 Gen 3 DJ Core\nEngine Ready (AAudio MMAP + NEON + QNN)"
                    }
                }
            } catch (e: Throwable) {
                runOnUiThread {
                    statusText.text = "Engine Init Failed: ${e.message}"
                }
            }
        }
    }

    private fun startToneGenerator(eng: NativeAudioEngine) {
        feedThread = thread(priority = Thread.MAX_PRIORITY) {
            val sampleRate = 48000
            val bufferFrames = 96
            val buffer = FloatArray(bufferFrames * 2)
            var phase = 0.0
            val frequency = 440.0
            val phaseInc = 2.0 * Math.PI * frequency / sampleRate

            while (isPlaying) {
                for (i in 0 until bufferFrames) {
                    val sample = (sin(phase) * 0.4f).toFloat()
                    buffer[i * 2] = sample
                    buffer[i * 2 + 1] = sample
                    phase += phaseInc
                    if (phase >= 2.0 * Math.PI) phase -= 2.0 * Math.PI
                }

                val written = eng.pushDeckAudio(NativeAudioEngine.DECK_A, buffer)
                if (written == 0) {
                    Thread.sleep(1)
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        isPlaying = false
        feedThread?.join(200)
        engine?.close()
    }
}

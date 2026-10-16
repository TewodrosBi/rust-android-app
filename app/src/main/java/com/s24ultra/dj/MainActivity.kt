package com.s24ultra.dj

import android.app.Activity
import android.content.pm.ActivityInfo
import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
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

    // State for Deck A
    private var deckAPlaying = false
    private var deckAStemsEnabled = false
    private var deckAVocals = 1.0f
    private var deckADrums = 1.0f
    private var deckABass = 1.0f
    private var deckAOther = 1.0f

    // State for Deck B
    private var deckBPlaying = false
    private var deckBStemsEnabled = false
    private var deckBVocals = 1.0f
    private var deckBDrums = 1.0f
    private var deckBBass = 1.0f
    private var deckBOther = 1.0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE

        val mainLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#121212"))
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        // --- Header Section ---
        val headerLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(16, 16, 16, 16)
            gravity = Gravity.CENTER_VERTICAL
            setBackgroundColor(Color.parseColor("#1E1E1E"))
        }

        val statusText = TextView(this).apply {
            text = "AAudio MMAP Exclusive | 48kHz"
            setTextColor(Color.GREEN)
            textSize = 16f
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        headerLayout.addView(statusText)

        val masterVolumeLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val masterLabel = TextView(this).apply { text = "Master"; setTextColor(Color.WHITE); setPadding(0, 0, 8, 0) }
        val masterVolumeSlider = SeekBar(this).apply {
            max = 100
            progress = 80
            layoutParams = LinearLayout.LayoutParams(200, ViewGroup.LayoutParams.WRAP_CONTENT)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                    engine?.setMasterAndCue(progress / 100f, 0.0f, 0.0f, false)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }
        masterVolumeLayout.addView(masterLabel)
        masterVolumeLayout.addView(masterVolumeSlider)
        headerLayout.addView(masterVolumeLayout)

        mainLayout.addView(headerLayout)

        // --- Crossfader Section ---
        val crossfaderLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(16, 8, 16, 8)
            setBackgroundColor(Color.parseColor("#181818"))
        }
        val crossfaderSlider = SeekBar(this).apply {
            max = 200
            progress = 100
            layoutParams = LinearLayout.LayoutParams(400, ViewGroup.LayoutParams.WRAP_CONTENT)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                    val pos = (progress - 100) / 100f // -1.0 to 1.0
                    engine?.setCrossfader(pos, NativeAudioEngine.CURVE_SMOOTH)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }
        val xfaderLabel = TextView(this).apply { text = "Crossfader"; setTextColor(Color.WHITE); setPadding(0, 0, 16, 0) }
        crossfaderLayout.addView(xfaderLabel)
        crossfaderLayout.addView(crossfaderSlider)
        mainLayout.addView(crossfaderLayout)

        // --- Decks Section ---
        val decksLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f
            )
        }

        decksLayout.addView(createDeckUI("Deck A", NativeAudioEngine.DECK_A))
        decksLayout.addView(createDeckUI("Deck B", NativeAudioEngine.DECK_B))

        mainLayout.addView(decksLayout)

        setContentView(mainLayout)

        // Init Engine
        thread {
            try {
                val eng = NativeAudioEngine()
                if (eng.initializeFromHardware(this@MainActivity)) {
                    engine = eng
                    eng.setMasterAndCue(masterGain = 0.8f, cueLevelGain = 0.0f, cueMix = 0.0f, splitCue = false)
                    eng.setDeckGains(NativeAudioEngine.DECK_A, trimGain = 1.0f, channelFader = 1.0f)
                    eng.setDeckGains(NativeAudioEngine.DECK_B, trimGain = 1.0f, channelFader = 1.0f)
                    eng.setCrossfader(0.0f, NativeAudioEngine.CURVE_SMOOTH)

                    if (eng.start()) {
                         isPlaying = true
                         startToneGenerator(eng)
                         runOnUiThread {
                             statusText.text = "AAudio MMAP Exclusive | 48kHz - Engine Ready"
                         }
                    } else {
                         runOnUiThread {
                             statusText.text = "Failed to start AAudio stream."
                         }
                    }
                }
            } catch (e: Throwable) {
                runOnUiThread {
                    statusText.text = "Engine Init Failed: ${e.message}"
                }
            }
        }
    }

    private fun createDeckUI(title: String, deckIndex: Int): LinearLayout {
        val deckLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f)
            setPadding(16, 16, 16, 16)
            // Add a small margin or border color depending on index
        }

        val titleText = TextView(this).apply {
            text = title
            setTextColor(Color.WHITE)
            textSize = 20f
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 16)
        }
        deckLayout.addView(titleText)

        // Waveform Placeholder
        val waveformView = android.view.View(this).apply {
            setBackgroundColor(Color.parseColor("#333333"))
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 150)
        }
        deckLayout.addView(waveformView)

        // Controls Area
        val controlsLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
            setPadding(0, 16, 0, 0)
        }

        // Column 1: Transport
        val transportLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f)
        }

        val playBtn = Button(this).apply {
            text = "Play"
            setOnClickListener {
                if (deckIndex == NativeAudioEngine.DECK_A) {
                    deckAPlaying = !deckAPlaying
                    text = if (deckAPlaying) "Pause" else "Play"
                } else {
                    deckBPlaying = !deckBPlaying
                    text = if (deckBPlaying) "Pause" else "Play"
                }
            }
        }
        val cueBtn = Button(this).apply { text = "Cue" }
        val syncBtn = Button(this).apply { text = "Sync" }
        transportLayout.addView(playBtn)
        transportLayout.addView(cueBtn)
        transportLayout.addView(syncBtn)
        controlsLayout.addView(transportLayout)

        // Column 2: Stems
        val stemsLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1.5f)
        }
        val stemToggle = Button(this).apply {
            text = "Stems: OFF"
            setOnClickListener {
                val enabled = if (deckIndex == NativeAudioEngine.DECK_A) {
                    deckAStemsEnabled = !deckAStemsEnabled
                    deckAStemsEnabled
                } else {
                    deckBStemsEnabled = !deckBStemsEnabled
                    deckBStemsEnabled
                }
                text = "Stems: ${if (enabled) "ON" else "OFF"}"
                engine?.toggleStemSeparation(deckIndex, enabled)
            }
        }
        stemsLayout.addView(stemToggle)

        val stemSlidersLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
        }

        stemSlidersLayout.addView(createVerticalSlider("Voc", 100) { v ->
            if (deckIndex == NativeAudioEngine.DECK_A) deckAVocals = v else deckBVocals = v
            updateStems(deckIndex)
        })
        stemSlidersLayout.addView(createVerticalSlider("Drm", 100) { v ->
            if (deckIndex == NativeAudioEngine.DECK_A) deckADrums = v else deckBDrums = v
            updateStems(deckIndex)
        })
        stemSlidersLayout.addView(createVerticalSlider("Bas", 100) { v ->
            if (deckIndex == NativeAudioEngine.DECK_A) deckABass = v else deckBBass = v
            updateStems(deckIndex)
        })
        stemSlidersLayout.addView(createVerticalSlider("Oth", 100) { v ->
            if (deckIndex == NativeAudioEngine.DECK_A) deckAOther = v else deckBOther = v
            updateStems(deckIndex)
        })

        stemsLayout.addView(stemSlidersLayout)
        controlsLayout.addView(stemsLayout)

        // Column 3: EQ
        val eqLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f)
        }
        var highEq = 1.0f
        var midEq = 1.0f
        var lowEq = 1.0f

        eqLayout.addView(createVerticalSlider("Hi", 100) { v -> highEq = v * 2f; engine?.setDeckEq(deckIndex, lowEq, midEq, highEq) })
        eqLayout.addView(createVerticalSlider("Mid", 100) { v -> midEq = v * 2f; engine?.setDeckEq(deckIndex, lowEq, midEq, highEq) })
        eqLayout.addView(createVerticalSlider("Low", 100) { v -> lowEq = v * 2f; engine?.setDeckEq(deckIndex, lowEq, midEq, highEq) })
        controlsLayout.addView(eqLayout)

        // Column 4: Volume (Trim & Fader)
        val volLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f)
        }
        var trim = 1.0f
        var fader = 1.0f
        volLayout.addView(createVerticalSlider("Trim", 100) { v -> trim = v * 2f; engine?.setDeckGains(deckIndex, trim, fader) })
        volLayout.addView(createVerticalSlider("Fader", 100) { v -> fader = v; engine?.setDeckGains(deckIndex, trim, fader) })
        controlsLayout.addView(volLayout)

        deckLayout.addView(controlsLayout)
        return deckLayout
    }

    private fun updateStems(deckIndex: Int) {
        if (deckIndex == NativeAudioEngine.DECK_A) {
            engine?.setStemGains(deckIndex, deckAVocals, deckADrums, deckABass, deckAOther)
        } else {
            engine?.setStemGains(deckIndex, deckBVocals, deckBDrums, deckBBass, deckBOther)
        }
    }

    private fun createVerticalSlider(label: String, initProgress: Int, onChange: (Float) -> Unit): LinearLayout {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f)
            gravity = Gravity.CENTER_HORIZONTAL
        }

        val tv = TextView(this).apply {
            text = label
            setTextColor(Color.LTGRAY)
            textSize = 10f
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 8)
        }
        container.addView(tv)

        // Simple hack: a vertical SeekBar rotated. For proper sizing, we set a fixed width which becomes height.
        val slider = SeekBar(this).apply {
            max = 100
            progress = initProgress
            rotation = 270f
            // We give it a fixed width (which becomes height after rotation) and let it wrap content for its "height" (thickness)
            val len = 200
            layoutParams = LinearLayout.LayoutParams(len, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                topMargin = len / 2
                bottomMargin = len / 2
            }
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                    onChange(progress / 100f)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }
        container.addView(slider)
        return container
    }

    private fun startToneGenerator(eng: NativeAudioEngine) {
        feedThread = thread(priority = Thread.MAX_PRIORITY) {
            val sampleRate = 48000
            val bufferFrames = 96
            val bufferA = FloatArray(bufferFrames * 2)
            val bufferB = FloatArray(bufferFrames * 2)

            var phaseA = 0.0
            val frequencyA = 440.0
            val phaseIncA = 2.0 * Math.PI * frequencyA / sampleRate

            var phaseB = 0.0
            val frequencyB = 330.0
            val phaseIncB = 2.0 * Math.PI * frequencyB / sampleRate

            while (isPlaying) {
                // Deck A Tone
                for (i in 0 until bufferFrames) {
                    val sample = if (deckAPlaying) (sin(phaseA) * 0.4f).toFloat() else 0f
                    bufferA[i * 2] = sample
                    bufferA[i * 2 + 1] = sample
                    phaseA += phaseIncA
                    if (phaseA >= 2.0 * Math.PI) phaseA -= 2.0 * Math.PI
                }

                // Deck B Tone
                for (i in 0 until bufferFrames) {
                    val sample = if (deckBPlaying) (sin(phaseB) * 0.4f).toFloat() else 0f
                    bufferB[i * 2] = sample
                    bufferB[i * 2 + 1] = sample
                    phaseB += phaseIncB
                    if (phaseB >= 2.0 * Math.PI) phaseB -= 2.0 * Math.PI
                }

                val writtenA = eng.pushDeckAudio(NativeAudioEngine.DECK_A, bufferA)
                val writtenB = eng.pushDeckAudio(NativeAudioEngine.DECK_B, bufferB)

                if (writtenA == 0 && writtenB == 0) {
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

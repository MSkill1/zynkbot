package ai.containai.zynkbot

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.BlurMaskFilter
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PathMeasure
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.service.voice.VoiceInteractionSession
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.animation.DecelerateInterpolator
import android.view.animation.LinearInterpolator
import android.widget.FrameLayout
import kotlin.math.PI
import kotlin.math.sin

/**
 * One assistant invocation, start to finish: listen, answer natively, speak, done.
 * See ZynkAssistantService for status/caveats.
 *
 * UI: no text, no notification — a Z drawn from its three strokes (two horizontals
 * and the diagonal), in Zynkbot green, at the bottom of the screen over the lock
 * screen. It draws itself in when the session opens, breathes while listening, and
 * while thinking a bright pulse runs along the strokes — top bar, diagonal, bottom
 * bar — on a loop. Gone when the session hides.
 *
 * Microphone: one stream, owned by WakeWordService (2026-10-20 rebuild). The session
 * asks it to feed a Dictation and tells it when the turn is over (onHide → endTurn).
 * Before that, the session opened its own Vosk recorder beside the wake-word loop's,
 * and a second reader on the busy mic wedged the session (OnePlus, 2026-09-03).
 */
class ZynkAssistantSession(context: Context) : VoiceInteractionSession(context) {
    companion object {
        private const val TAG = "ZynkAssistantSession"
        // Dictation timers (silence 1.5 s, no speech 6 s, cap 30 s) live in Dictation.
        private const val MAX_QUERY_WORDS = 60
        private const val LOGO_DP = 120
        private const val BOTTOM_MARGIN_DP = 140
        /** Upper bound on how long one interaction may hold the screen on. */
        private const val SCREEN_LOCK_MS = 90_000L

        /** The session currently shown, else null. MainActivity.onResume uses it to
         *  drop the Z overlay when the app itself comes to the front mid-reply. */
        @Volatile var current: ZynkAssistantSession? = null
    }

    private var screenLock: PowerManager.WakeLock? = null
    /** Set by a tap on the Z: this trigger was not wanted. Checked wherever a step
     *  would otherwise carry on (transcript in, answer out). */
    @Volatile private var cancelled = false

    /** The dictation in flight, fed from the wake-word service's stream when Hey Zynk
     *  is on, else from a stream of our own (the assistant gesture with it off). */
    @Volatile private var dictation: Dictation? = null
    private var ownStream: MicStream? = null
    @Volatile private var listenStartedAt = 0L
    @Volatile private var firstPartialLogged = false
    private val main = Handler(Looper.getMainLooper())
    private var zView: ZView? = null

    // ── overlay ──────────────────────────────────────────────────────────────

    override fun onCreateContentView(): View {
        val density = context.resources.displayMetrics.density
        val root = FrameLayout(context).apply { setBackgroundColor(Color.TRANSPARENT) }
        val size = (LOGO_DP * density).toInt()
        val z = ZView(context)
        // A tap on the Z cancels the session — the way out of an unwanted trigger.
        z.isClickable = true
        z.setOnClickListener { cancelByUser() }
        val lp = FrameLayout.LayoutParams(size, size, Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL)
        lp.bottomMargin = (BOTTOM_MARGIN_DP * density).toInt()
        root.addView(z, lp)
        zView = z
        return root
    }

    /**
     * The Z itself. Three strokes on one path, measured so both the draw-in and the
     * travelling pulse can be expressed as "the first N% of the path" / "the segment
     * from A to B along the path" — no per-stroke bookkeeping.
     */
    private class ZView(context: Context) : View(context) {
        enum class Mode { LISTENING, THINKING }

        private val green = 0xFF50FA7B.toInt()
        private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = green; style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
        }
        private val glow = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = green; style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
        }
        private val path = Path()
        private val measure = PathMeasure()
        private val segment = Path()
        private val pulseA = Path()
        private val pulseB = Path()
        private var length = 0f
        private var drawIn = 0f   // 0..1 — the Z drawing itself in when the session opens
        private var phase = 0f    // 0..1 — looping animation phase

        var mode: Mode = Mode.LISTENING
            set(value) {
                field = value
                loop.duration = if (value == Mode.THINKING) 700L else 1600L
            }

        private val intro = ValueAnimator.ofFloat(0f, 1f).apply {
            duration = 450L
            interpolator = DecelerateInterpolator()
            addUpdateListener { drawIn = it.animatedValue as Float; invalidate() }
        }
        private val loop = ValueAnimator.ofFloat(0f, 1f).apply {
            duration = 1600L
            repeatCount = ValueAnimator.INFINITE
            interpolator = LinearInterpolator()
            addUpdateListener { phase = it.animatedValue as Float; invalidate() }
        }

        init {
            // BlurMaskFilter (the glow) needs software rendering; the view is tiny.
            setLayerType(LAYER_TYPE_SOFTWARE, null)
        }

        fun start() { intro.start(); loop.start() }
        fun stop() { intro.cancel(); loop.cancel(); drawIn = 0f; invalidate() }

        override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
            val inset = w * 0.18f
            val left = inset; val right = w - inset
            val top = inset; val bottom = h - inset
            path.reset()
            path.moveTo(left, top)      // top bar
            path.lineTo(right, top)
            path.lineTo(left, bottom)   // diagonal
            path.lineTo(right, bottom)  // bottom bar
            measure.setPath(path, false)
            length = measure.length
            stroke.strokeWidth = w * 0.11f
            glow.strokeWidth = w * 0.24f
            glow.maskFilter = BlurMaskFilter(w * 0.12f, BlurMaskFilter.Blur.NORMAL)
        }

        override fun onDraw(canvas: Canvas) {
            if (length == 0f || drawIn <= 0f) return
            segment.reset()
            measure.getSegment(0f, length * drawIn, segment, true)

            when (mode) {
                Mode.LISTENING -> {
                    // Slow breathing: the whole Z swells and dims together.
                    val breathe = 0.55f + 0.45f * ((sin(phase * 2.0 * PI).toFloat() + 1f) / 2f)
                    glow.alpha = (110 * breathe).toInt()
                    stroke.alpha = (255 * breathe).toInt()
                    canvas.drawPath(segment, glow)
                    canvas.drawPath(segment, stroke)
                }
                Mode.THINKING -> {
                    // Dim base Z, with a bright pulse travelling top bar -> diagonal -> bottom bar.
                    stroke.alpha = 110
                    canvas.drawPath(segment, stroke)
                    val pulseLen = length * 0.22f
                    val head = phase * length
                    val tail = head - pulseLen
                    pulseA.reset(); pulseB.reset()
                    if (tail >= 0f) {
                        measure.getSegment(tail, head, pulseA, true)
                    } else {
                        // Wrap: the pulse's tail is still on the end of the path.
                        measure.getSegment(0f, head, pulseA, true)
                        measure.getSegment(length + tail, length, pulseB, true)
                    }
                    glow.alpha = 150
                    stroke.alpha = 255
                    canvas.drawPath(pulseA, glow); canvas.drawPath(pulseA, stroke)
                    canvas.drawPath(pulseB, glow); canvas.drawPath(pulseB, stroke)
                }
            }
        }
    }

    // ── lifecycle ────────────────────────────────────────────────────────────

    override fun onShow(args: Bundle?, showFlags: Int) {
        super.onShow(args, showFlags)
        Log.i(TAG, "Session shown — starting dictation")
        ZynkAssistantService.sessionActive = true
        current = this
        cancelled = false
        // The session window draws over the lock screen but nothing turns the display
        // ON — with the phone asleep the Z was never seen (OnePlus, 2026-09-04: session
        // shown 13:22:03, screen dark until the power button at 13:22:32). Light it,
        // the way Google Assistant does, and hold it for the interaction. setKeepAwake
        // alone only holds a screen that is already lit.
        wakeScreen()
        try { setKeepAwake(true) } catch (e: Exception) { Log.w(TAG, "setKeepAwake failed: ${e.message}") }
        zView?.let { it.visibility = View.VISIBLE; it.mode = ZView.Mode.LISTENING; it.start() }
        startListening()
    }

    override fun onHide() {
        ZynkAssistantService.sessionActive = false
        current = null
        releaseScreen()
        stopListening()
        zView?.stop()
        // The turn is over: the service goes back to listening by itself.
        try { WakeWordService.instance?.endTurn() } catch (e: Exception) { Log.w(TAG, "endTurn failed: ${e.message}") }
        super.onHide()
    }

    /**
     * The session window spans the whole screen and, by default, swallows every
     * touch — even over its transparent area. When Matt opened the app mid-reply the
     * Z sat over the UI and ate the tap on Stop (OnePlus, 2026-09-04). So only the Z's
     * own square takes touches (a tap cancels the session); everything around it
     * passes through to whatever is beneath, and no content inset so apps under it
     * are not resized.
     */
    override fun onComputeInsets(outInsets: Insets) {
        super.onComputeInsets(outInsets)
        outInsets.contentInsets.top = zView?.rootView?.height ?: 0
        outInsets.touchableInsets = Insets.TOUCHABLE_INSETS_REGION
        val z = zView
        if (z != null && z.width > 0 && z.visibility == View.VISIBLE) {
            val loc = IntArray(2)
            z.getLocationInWindow(loc)
            outInsets.touchableRegion.set(loc[0], loc[1], loc[0] + z.width, loc[1] + z.height)
        } else {
            outInsets.touchableRegion.setEmpty()
        }
    }

    /** A tap on the Z: stop whatever is in progress, close audibly, back to passive
     *  listening. Both the listening and the speaking phases are covered. */
    private fun cancelByUser() {
        if (cancelled) return
        cancelled = true
        Log.i(TAG, "Session cancelled by a tap on the Z")
        // The strongest false-trigger signal there is: the user dismissed it. It used to
        // count for nothing, so tapping every TV firing away never started the back-off
        // (2026-09-08). It is a miss, and the clip is labelled false for the verifier.
        WakeWordService.reportOutcome(false)
        stopListening()
        NativeVoiceAnswerer.stopSpeaking()
        Thread {
            if (MainActivity.fruitlessTone(context)) NativeVoiceAnswerer.playCloseTone(context)
            main.post { hide() }
        }.start()
    }

    /** The app came to the front while this session is still busy: its own Stop
     *  button takes over, so the Z has no job and would only sit over the UI. The
     *  session itself carries on (listening or speaking) and hides when done. */
    fun hideOverlay() {
        main.post { zView?.let { it.stop(); it.visibility = View.INVISIBLE } }
    }

    @Suppress("DEPRECATION")
    private fun wakeScreen() {
        try {
            val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
            val lock = pm.newWakeLock(
                PowerManager.SCREEN_BRIGHT_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP or PowerManager.ON_AFTER_RELEASE,
                "zynkbot:assistant_session"
            )
            lock.acquire(SCREEN_LOCK_MS)
            screenLock = lock
        } catch (e: Exception) { Log.w(TAG, "screen wake failed: ${e.message}") }
    }

    private fun releaseScreen() {
        val lock = screenLock ?: return
        screenLock = null
        try { if (lock.isHeld) lock.release() } catch (e: Exception) { Log.w(TAG, "screen release failed: ${e.message}") }
    }

    // ── dictation ────────────────────────────────────────────────────────────

    /**
     * The Voice settings selector (Vosk offline / OpenAI Whisper) is honoured here
     * since 2026-09-08: OpenAI when selected and a key is on file, else Vosk. An
     * OpenAI failure (not a cancel) falls back to Vosk once. Since the 2026-10-20
     * rebuild the audio comes from the wake-word service's one stream (or a stream of
     * our own when Hey Zynk is off), through [Dictation]; the timers live there.
     */
    private fun startListening() {
        val source = MainActivity.voiceInputSource(context)
        val key = OpenAiDictation.apiKey(context)
        if (source == "openai" && key == null) {
            Log.w(TAG, "[Dictation] OpenAI selected but no OPENAI_API_KEY on file — using Vosk")
        }
        val engine = if (source == "openai" && key != null) Dictation.Engine.WHISPER else Dictation.Engine.VOSK
        startDictation(engine, key)
    }

    private fun startDictation(engine: Dictation.Engine, key: String?) {
        Log.i(TAG, "[Dictation] engine=${engine.name.lowercase()}")
        val model = if (engine == Dictation.Engine.VOSK) Dictation.voskModel(context) else null
        if (engine == Dictation.Engine.VOSK && model == null) {
            Log.w(TAG, "No Vosk model on the phone — nothing to do")
            hide()
            return
        }
        listenStartedAt = System.currentTimeMillis(); firstPartialLogged = false
        lateinit var d: Dictation
        d = Dictation(engine, model, key, timers = true, listener = object : Dictation.Listener {
            override fun onPartial(text: String) {
                if (!firstPartialLogged) { firstPartialLogged = true; Log.i(TAG, "[Dictation] first words after ${System.currentTimeMillis() - listenStartedAt} ms") }
            }
            override fun onFinal(text: String?) {
                if (dictation !== d) return   // superseded or already stopped
                dictation = null
                ownStream?.close(); ownStream = null
                if (cancelled) return
                if (text == null) {
                    if (d.isCancelled) {
                        // Ended by the service (another app took the microphone, Hey Zynk
                        // turned off, the watchdog) — not a failure, so no Vosk fallback.
                        if (ZynkAssistantService.sessionActive) hide()
                    } else if (engine == Dictation.Engine.WHISPER) {
                        Log.w(TAG, "[Dictation] OpenAI failed — falling back to Vosk")
                        startDictation(Dictation.Engine.VOSK, null)
                    } else hide()
                    return
                }
                Log.i(TAG, "Transcript: \"$text\"")
                answerAndFinish(text)
            }
        })
        dictation = d
        val svc = WakeWordService.instance
        if (svc != null && svc.isOn && svc.dictate(d, ownsTurn = true)) return
        // Hey Zynk is off (the assistant gesture brought us here): a stream of our own.
        val s = MicStream(TAG)
        s.sink = d
        if (!s.open()) {
            Log.e(TAG, "Microphone could not be opened")
            dictation = null
            hide()
            return
        }
        ownStream = s
    }

    private fun stopListening() {
        val d = dictation
        dictation = null
        d?.cancel()
        ownStream?.close(); ownStream = null
    }

    // ── answer ───────────────────────────────────────────────────────────────

    /** Cheap shape test used only in strict mode: a question mark, an interrogative
     *  or imperative opening, or a Remember command. Whisper transcripts carry
     *  punctuation; Vosk ones do not, so the word list matters more there. */
    private fun looksAddressedToAssistant(t: String): Boolean {
        val s = " " + t.trim().lowercase().replace(Regex("[^a-z0-9' ]"), " ").replace(Regex("\\s+"), " ") + " "
        if (t.contains('?')) return true
        // Anywhere in the sentence, not only at the start: "in baldur's gate three, at the
        // very beginning, is there somewhere you can find a shovel" was dropped on
        // 2026-09-08 because it did not *begin* with a question word.
        val markers = listOf(
            " what ", " what's ", " whats ", " how ", " when ", " where ", " where's ", " who ", " who's ", " why ", " which ",
            " is there ", " are there ", " is it ", " does ", " do i ", " do you ", " did ", " can you ", " can i ", " could you ",
            " would you ", " will you ", " should i ", " please ", " tell me ", " remind ", " remember ", " set a ", " set an ",
            " start a ", " start the ", " open ", " play ", " search ", " look up ", " find ", " define ", " explain ",
            " give me ", " show me ", " read ", " translate ", " zynk ",
        )
        return markers.any { s.contains(it) }
    }

    private fun answerAndFinish(transcript: String) {
        // Local sanity gate, free: one word is noise, sixty is a TV programme.
        val words = transcript.trim().split(Regex("\\s+")).filter { it.isNotBlank() }
        if (words.isNotEmpty() && (words.size < 2 || words.size > MAX_QUERY_WORDS)) {
            Log.i(TAG, "Transcript rejected (${words.size} words) — not a question")
        }
        if (transcript.isBlank() || words.size < 2 || words.size > MAX_QUERY_WORDS) {
            WakeWordService.reportOutcome(false)
            // Nothing heard (a false trigger on the air conditioner, say). The closing
            // tone is optional since 2026-09-12: with a verifier most such firings are
            // the owner's own conversation, and two sounds per miss grated.
            Thread {
                if (MainActivity.fruitlessTone(context)) NativeVoiceAnswerer.playCloseTone(context)
                main.post { hide() }
            }.start()
            return
        }
        // While backing off, only something shaped like a question or a request goes
        // to the model at all; a TV line ("This is the best time of your life.") is
        // closed here as a miss without an API call (2026-09-08).
        if (WakeWordService.isStrict() && VoiceCommands.parse(transcript) == null && !looksAddressedToAssistant(transcript)) {
            Log.i(TAG, "Strict mode: transcript is not a question or request — not sent")
            WakeWordService.reportOutcome(false)
            // The user did say something and got nothing back (Pixel 2026-09-13 14:16:
            // "this is a second test just respond affirmative" dropped in silence after
            // three air-conditioner misfires). The quiet-close toggle is for empty
            // firings; a dropped sentence always gets the tone, and a word of why.
            Thread {
                NativeVoiceAnswerer.say(context, "Were you asking me a question? I'm only answering questions for a few minutes.")
                main.post { hide() }
            }.start()
            return
        }
        // Clock commands are handled here, never by the model (KI-027): it has no
        // clock, and either invented a confirmation or said it couldn't. The
        // confirmation is spoken only after the clock app accepted the intent.
        VoiceCommands.parse(transcript)?.let { cmd ->
            main.post { zView?.mode = ZView.Mode.THINKING }
            Thread {
                NativeVoiceAnswerer.playCloseTone(context)
                val ok = VoiceCommands.execute(context, cmd)
                WakeWordService.reportOutcome(true)
                Log.i(TAG, "Voice command ${cmd::class.simpleName}: ${if (ok) "done" else "FAILED"}")
                if (!cancelled) NativeVoiceAnswerer.say(context, if (ok) VoiceCommands.confirmation(cmd) else VoiceCommands.FAILED_LINE)
                main.post { hide() }
            }.start()
            return
        }
        main.post { zView?.mode = ZView.Mode.THINKING }
        Thread {
            // "Sent" tone: without it the user sits in silence for the whole network
            // round trip thinking nothing happened. Same chime the wake-word path uses.
            NativeVoiceAnswerer.playCloseTone(context)
            val spoke = NativeVoiceAnswerer.answer(context, transcript)
            // A spoken reply to an uncancelled question: label the clip real so the
            // "Send my wake-word clips" count reflects ordinary use, not just timers.
            if (spoke && !cancelled) WakeWordService.labelClip(true)
            // Whether or not it succeeded, the session's job is done either way —
            // WakeWordService's screen-off path is the one with a WebView fallback;
            // this entry point has no equivalent to fall back to yet.
            main.post { hide() }
        }.start()
    }
}

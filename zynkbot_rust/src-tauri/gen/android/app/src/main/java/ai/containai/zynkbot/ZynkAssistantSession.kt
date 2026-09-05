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
 * Microphone discipline (learned on the OnePlus, 2026-09-03): the wake-word ONNX
 * loop and Vosk must never hold AudioRecord at the same time — a second reader on
 * the busy mic wedged the session with no timeout ever firing. So: release the
 * wake-word mic BEFORE starting Vosk, never call Vosk's stop() (which joins its
 * thread) on the main thread, and re-arm the wake word natively on hide.
 */
class ZynkAssistantSession(context: Context) : VoiceInteractionSession(context) {
    companion object {
        private const val TAG = "ZynkAssistantSession"
        private const val SILENCE_MS = 1500L
        private const val SAFETY_TIMEOUT_MS = 12_000L   // hard cap; ongoing speech cannot extend it
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

    @Volatile private var speechService: org.vosk.android.SpeechService? = null
    private val silenceHandler = Handler(Looper.getMainLooper())
    // Separate runnables so a partial transcript restarts only the silence timer. One
    // removeCallbacksAndMessages(null) used to cancel the safety cap too, so continuous
    // TV speech kept the session listening until the programme paused (2026-09-04).
    private val silenceStop = Runnable { stopVoskAsync() }
    private val hardStop = Runnable { Log.i(TAG, "Listening cap reached"); stopVoskAsync() }
    private val main = Handler(Looper.getMainLooper())
    private var zView: ZView? = null

    // ── overlay ──────────────────────────────────────────────────────────────

    override fun onCreateContentView(): View {
        val density = context.resources.displayMetrics.density
        val root = FrameLayout(context).apply { setBackgroundColor(Color.TRANSPARENT) }
        val size = (LOGO_DP * density).toInt()
        val z = ZView(context)
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
        // Re-arm passive wake-word listening natively; the JS re-arm only runs when
        // the app resumes, which a hands-free flow never does. (Refused while the
        // native voice is still speaking — NativeVoiceAnswerer re-arms afterwards.)
        Thread { try { WakeWordService.instance?.resumeMicAfterSession() } catch (e: Exception) { Log.w(TAG, "re-arm failed: ${e.message}") } }.start()
        super.onHide()
    }

    /**
     * The session window spans the whole screen and, by default, swallows every
     * touch — even over its transparent area. When Matt opened the app mid-reply the
     * Z sat over the UI and ate the tap on Stop (OnePlus, 2026-09-04). The Z is
     * display-only, so claim no touchable region at all: everything passes through
     * to whatever is beneath, and no content inset so apps under it are not resized.
     */
    override fun onComputeInsets(outInsets: Insets) {
        super.onComputeInsets(outInsets)
        outInsets.contentInsets.top = zView?.rootView?.height ?: 0
        outInsets.touchableInsets = Insets.TOUCHABLE_INSETS_REGION
        outInsets.touchableRegion.setEmpty()
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

    private fun startListening() {
        val model = WakeWordService.sharedVoskModel
        if (model == null) {
            Log.w(TAG, "No Vosk model loaded yet (wake-word listener hasn't run this session) — nothing to do")
            hide()
            return
        }

        val accumulated = StringBuilder()
        val listener = object : org.vosk.android.RecognitionListener {
            override fun onPartialResult(h: String?) {
                val partial = try { org.json.JSONObject(h ?: "").optString("partial", "") } catch (_: Exception) { "" }
                if (partial.isNotBlank()) {
                    silenceHandler.removeCallbacks(silenceStop)
                    silenceHandler.postDelayed(silenceStop, SILENCE_MS)
                }
            }
            override fun onResult(h: String?) {
                val t = try { org.json.JSONObject(h ?: "").optString("text", "").trim() } catch (_: Exception) { "" }
                if (t.isNotBlank()) synchronized(accumulated) {
                    if (accumulated.isNotEmpty()) accumulated.append(" ")
                    accumulated.append(t)
                }
            }
            override fun onFinalResult(h: String?) {
                val last = try { org.json.JSONObject(h ?: "").optString("text", "").trim() } catch (_: Exception) { "" }
                val transcript = synchronized(accumulated) {
                    buildString {
                        append(accumulated)
                        if (accumulated.isNotEmpty() && last.isNotBlank()) append(" ")
                        append(last)
                    }.trim().also { accumulated.clear() }
                }
                silenceHandler.removeCallbacksAndMessages(null)
                speechService = null
                Log.i(TAG, "Transcript: \"$transcript\"")
                answerAndFinish(transcript)
            }
            override fun onError(e: Exception?) {
                Log.e(TAG, "Vosk error: ${e?.message}")
                silenceHandler.removeCallbacksAndMessages(null)
                hide()
            }
            override fun onTimeout() {
                Log.w(TAG, "Vosk timeout — no speech detected")
                hide()
            }
        }

        // Off the main thread: releasing the wake-word mic can wait up to 2s, and
        // Vosk callbacks are delivered on the main looper regardless.
        Thread {
            val freed = try { WakeWordService.instance?.releaseMicForSession() ?: true } catch (e: Exception) { false }
            if (!freed) Log.w(TAG, "Wake-word mic not confirmed released; starting Vosk anyway (safety timeout will end it)")
            try {
                val rec = org.vosk.Recognizer(model, 16000.0f)
                val service = org.vosk.android.SpeechService(rec, 16000.0f)
                speechService = service
                service.startListening(listener)
                silenceHandler.postDelayed(hardStop, SAFETY_TIMEOUT_MS)
            } catch (e: Exception) {
                Log.e(TAG, "Vosk start failed: ${e.message}")
                main.post { hide() }
            }
        }.start()
    }

    /** Vosk's stop() joins its recognizer thread; never do that on the main thread. */
    private fun stopVoskAsync() {
        val service = speechService ?: return
        Thread { try { service.stop() } catch (e: Exception) { Log.w(TAG, "Vosk stop failed: ${e.message}") } }.start()
    }

    private fun stopListening() {
        silenceHandler.removeCallbacksAndMessages(null)
        stopVoskAsync()
        speechService = null
    }

    // ── answer ───────────────────────────────────────────────────────────────

    private fun answerAndFinish(transcript: String) {
        // Local sanity gate, free: one word is noise, sixty is a TV programme.
        val words = transcript.trim().split(Regex("\\s+")).filter { it.isNotBlank() }
        if (words.isNotEmpty() && (words.size < 2 || words.size > MAX_QUERY_WORDS)) {
            Log.i(TAG, "Transcript rejected (${words.size} words) — not a question")
        }
        if (transcript.isBlank() || words.size < 2 || words.size > MAX_QUERY_WORDS) {
            // Nothing heard (a false trigger on the air conditioner, say): close
            // audibly so the user knows it fired and shut down, rather than vanishing.
            Thread {
                NativeVoiceAnswerer.playCloseTone(context)
                main.post { hide() }
            }.start()
            return
        }
        main.post { zView?.mode = ZView.Mode.THINKING }
        Thread {
            // "Sent" tone: without it the user sits in silence for the whole network
            // round trip thinking nothing happened. Same chime the wake-word path uses.
            NativeVoiceAnswerer.playCloseTone(context)
            NativeVoiceAnswerer.answer(context, transcript)
            // Whether or not it succeeded, the session's job is done either way —
            // WakeWordService's screen-off path is the one with a WebView fallback;
            // this entry point has no equivalent to fall back to yet.
            main.post { hide() }
        }.start()
    }
}

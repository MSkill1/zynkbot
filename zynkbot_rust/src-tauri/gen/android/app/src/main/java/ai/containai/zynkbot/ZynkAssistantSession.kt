package ai.containai.zynkbot

import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.service.voice.VoiceInteractionSession
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView

/**
 * One assistant invocation, start to finish: listen, answer natively, speak, done.
 * See ZynkAssistantService for status/caveats.
 *
 * UI: no text, no notification — just a pulsing Z at the bottom of the screen (over
 * the lock screen too) so the user can see the assistant is listening / thinking,
 * the way Siri's orb does. Pulse speed is the state: slow = listening, fast =
 * thinking. The overlay disappears when the session hides.
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
        private const val SAFETY_TIMEOUT_MS = 10_000L
        private const val LOGO_DP = 96
        private const val BOTTOM_MARGIN_DP = 140
    }

    private enum class State(val pulseMs: Long, val minAlpha: Float) {
        LISTENING(900L, 0.55f),
        THINKING(450L, 0.35f),
    }

    @Volatile private var speechService: org.vosk.android.SpeechService? = null
    private val silenceHandler = Handler(Looper.getMainLooper())
    private val main = Handler(Looper.getMainLooper())
    private var logo: ImageView? = null
    private var pulse: AnimatorSet? = null

    // ── overlay ──────────────────────────────────────────────────────────────

    override fun onCreateContentView(): View {
        val density = context.resources.displayMetrics.density
        val root = FrameLayout(context).apply { setBackgroundColor(Color.TRANSPARENT) }
        val size = (LOGO_DP * density).toInt()
        val img = ImageView(context).apply {
            setImageResource(R.mipmap.ic_launcher)
            scaleType = ImageView.ScaleType.FIT_CENTER
            alpha = 0f
        }
        val lp = FrameLayout.LayoutParams(size, size, Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL)
        lp.bottomMargin = (BOTTOM_MARGIN_DP * density).toInt()
        root.addView(img, lp)
        logo = img
        return root
    }

    /** Main thread only. Restarts the pulse at the speed for [state]. */
    private fun setState(state: State) {
        val img = logo ?: return
        pulse?.cancel()
        val scaleX = ObjectAnimator.ofFloat(img, View.SCALE_X, 1.0f, 1.18f)
        val scaleY = ObjectAnimator.ofFloat(img, View.SCALE_Y, 1.0f, 1.18f)
        val alpha = ObjectAnimator.ofFloat(img, View.ALPHA, state.minAlpha, 1.0f)
        for (a in listOf(scaleX, scaleY, alpha)) {
            a.repeatMode = ValueAnimator.REVERSE
            a.repeatCount = ValueAnimator.INFINITE
        }
        pulse = AnimatorSet().apply {
            playTogether(scaleX, scaleY, alpha)
            duration = state.pulseMs
            start()
        }
    }

    private fun stopPulse() {
        pulse?.cancel(); pulse = null
        logo?.alpha = 0f
    }

    // ── lifecycle ────────────────────────────────────────────────────────────

    override fun onShow(args: Bundle?, showFlags: Int) {
        super.onShow(args, showFlags)
        Log.i(TAG, "Session shown — starting dictation")
        setState(State.LISTENING)
        startListening()
    }

    override fun onHide() {
        stopListening()
        stopPulse()
        // Re-arm passive wake-word listening natively; the JS re-arm only runs when
        // the app resumes, which a hands-free flow never does.
        Thread { try { WakeWordService.instance?.resumeMicAfterSession() } catch (e: Exception) { Log.w(TAG, "re-arm failed: ${e.message}") } }.start()
        super.onHide()
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
                    silenceHandler.removeCallbacksAndMessages(null)
                    silenceHandler.postDelayed({ stopVoskAsync() }, SILENCE_MS)
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
                silenceHandler.postDelayed({ stopVoskAsync() }, SAFETY_TIMEOUT_MS)
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
        if (transcript.isBlank()) {
            hide()
            return
        }
        main.post { setState(State.THINKING) }
        Thread {
            // "Sent" tone: without it the user sits in silence for the whole network
            // round trip thinking nothing happened. Same chime the wake-word path uses.
            NativeVoiceAnswerer.playSentTone(context)
            NativeVoiceAnswerer.answer(context, transcript)
            // Whether or not it succeeded, the session's job is done either way —
            // WakeWordService's screen-off path is the one with a WebView fallback;
            // this entry point has no equivalent to fall back to yet.
            main.post { hide() }
        }.start()
    }
}

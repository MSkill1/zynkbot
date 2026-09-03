package ai.containai.zynkbot

import android.content.Context
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.service.voice.VoiceInteractionSession
import android.util.Log
import android.view.View

/**
 * One assistant invocation, start to finish: listen, answer natively, speak, done.
 * No content view — intentionally invisible, matching "no lock-screen text, no
 * notification, just an answer." See ZynkAssistantService for status/caveats.
 *
 * Simplification, flagged for later hardening once this is verified on a device:
 * no explicit wake lock is held here (unlike WakeWordService's screen-off path).
 * The OS is expected to keep a shown session's host process elevated; if testing
 * shows otherwise, port WakeWordService's wake-lock handling here too.
 */
class ZynkAssistantSession(context: Context) : VoiceInteractionSession(context) {
    companion object {
        private const val TAG = "ZynkAssistantSession"
        private const val SILENCE_MS = 1500L
        private const val SAFETY_TIMEOUT_MS = 10_000L
    }

    private var speechService: org.vosk.android.SpeechService? = null
    private val silenceHandler = Handler(Looper.getMainLooper())

    override fun onCreateContentView(): View? = null // headless: no visible UI

    override fun onShow(args: Bundle?, showFlags: Int) {
        super.onShow(args, showFlags)
        Log.i(TAG, "Session shown — starting dictation")
        startListening()
    }

    override fun onHide() {
        stopListening()
        super.onHide()
    }

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
                    silenceHandler.postDelayed({ speechService?.stop() }, SILENCE_MS)
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

        try {
            val rec = org.vosk.Recognizer(model, 16000.0f)
            val service = org.vosk.android.SpeechService(rec, 16000.0f)
            speechService = service
            service.startListening(listener)
            silenceHandler.postDelayed({ service.stop() }, SAFETY_TIMEOUT_MS)
        } catch (e: Exception) {
            Log.e(TAG, "Vosk start failed: ${e.message}")
            hide()
        }
    }

    private fun stopListening() {
        silenceHandler.removeCallbacksAndMessages(null)
        try { speechService?.stop() } catch (_: Exception) {}
        speechService = null
    }

    private fun answerAndFinish(transcript: String) {
        if (transcript.isBlank()) {
            hide()
            return
        }
        Thread {
            // "Sent" tone: without it the user sits in silence for the whole network
            // round trip thinking nothing happened. Same chime the wake-word path uses.
            NativeVoiceAnswerer.playSentTone(context)
            NativeVoiceAnswerer.answer(context, transcript)
            // Whether or not it succeeded, the session's job is done either way —
            // WakeWordService's screen-off path is the one with a WebView fallback;
            // this entry point has no equivalent to fall back to yet.
            Handler(Looper.getMainLooper()).post { hide() }
        }.start()
    }
}

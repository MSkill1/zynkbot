package ai.containai.zynkbot

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Shared "answer a transcript entirely natively" logic: call ZynkCore.nativeSendMessage,
 * speak the reply with Android's built-in TextToSpeech. No Activity, no WebView, no
 * notification. Used by both WakeWordService (screen-off wake-word path) and
 * ZynkAssistantSession (the OS assistant-role entry point), so the two triggers share
 * one implementation instead of drifting apart.
 *
 * One TTS engine per process, created lazily on first use and shut down by whichever
 * component owns the process lifecycle (WakeWordService, being the longer-lived
 * foreground service, does this in its onDestroy()).
 */
object NativeVoiceAnswerer {
    private const val TAG = "NativeVoiceAnswerer"

    @Volatile private var tts: TextToSpeech? = null

    /** Blocking: run on a background thread, never the main thread. Returns true if a
     *  reply was produced and spoken; false on any failure (caller should fall back
     *  to another delivery path rather than leave the request unanswered). */
    fun answer(context: Context, transcript: String): Boolean {
        if (transcript.isBlank()) return false
        var spoke = false
        try {
            ZynkCore.nativeSendMessage(
                transcript, "", "", "", "guardian",
                object : ZynkCore.Callback {
                    override fun onToken(token: String) {}
                    override fun onEvent(name: String, payloadJson: String) {}
                    override fun onDone(replyJson: String) {
                        val replyText = try {
                            org.json.JSONObject(replyJson).optString("reply_text", "")
                        } catch (e: Exception) {
                            Log.w(TAG, "Could not parse native reply: ${e.message}")
                            ""
                        }
                        Log.i(TAG, "Native reply: \"${replyText.take(80)}\"")
                        if (replyText.isNotBlank()) {
                            spoke = speak(context, replyText)
                        }
                    }
                    override fun onError(message: String) {
                        Log.w(TAG, "Native reply failed: $message")
                    }
                },
            )
        } catch (e: Throwable) {
            Log.w(TAG, "Native answer path unavailable: ${e.message}")
        }
        return spoke
    }

    /** Speaks [text], blocking until it finishes. False (never throws) on any failure. */
    fun speak(context: Context, text: String): Boolean {
        val latch = CountDownLatch(1)
        var ok = false
        try {
            var engine = tts
            if (engine == null) {
                val initLatch = CountDownLatch(1)
                var initOk = false
                engine = TextToSpeech(context.applicationContext) { status ->
                    initOk = status == TextToSpeech.SUCCESS
                    initLatch.countDown()
                }
                initLatch.await(5000, TimeUnit.MILLISECONDS)
                if (!initOk) { Log.w(TAG, "TTS init failed"); return false }
                engine.language = Locale.US
                tts = engine
            }
            engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {}
                override fun onDone(utteranceId: String?) { ok = true; latch.countDown() }
                @Deprecated("required override") override fun onError(utteranceId: String?) { latch.countDown() }
                override fun onError(utteranceId: String?, errorCode: Int) { latch.countDown() }
            })
            val id = "zynk-native-${System.currentTimeMillis()}"
            val result = engine.speak(text, TextToSpeech.QUEUE_FLUSH, null, id)
            if (result != TextToSpeech.SUCCESS) { Log.w(TAG, "speak() rejected"); return false }
            // Safety cap: don't block forever if the utterance callback never fires.
            latch.await(60_000, TimeUnit.MILLISECONDS)
        } catch (e: Exception) {
            Log.w(TAG, "TTS failed: ${e.message}")
        }
        return ok
    }

    fun shutdown() {
        try { tts?.shutdown() } catch (_: Exception) {}
        tts = null
    }
}

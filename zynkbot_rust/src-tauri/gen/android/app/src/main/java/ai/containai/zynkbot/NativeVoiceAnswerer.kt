package ai.containai.zynkbot

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Shared "answer a transcript entirely natively" logic: call ZynkCore.nativeSendMessage
 * and speak the reply with Android's built-in TextToSpeech. No Activity, no WebView, no
 * notification. Used by both WakeWordService (screen-off wake-word path) and
 * ZynkAssistantSession (the OS assistant-role entry point).
 *
 * Streams: tokens are spoken as soon as a sentence completes (QUEUE_ADD), so the user
 * hears the start of the answer while the rest is still generating — instead of a
 * silent 15-20s wait for the whole reply. Model-side markers (MEMORY_EXTRACT,
 * WEB_SEARCH_NEEDED) that the frontend strips are cut here too so they're never read
 * aloud. A failure is spoken, not swallowed.
 */
object NativeVoiceAnswerer {
    private const val TAG = "NativeVoiceAnswerer"
    private const val ERROR_LINE = "Zynkbot couldn't get an answer. Check the A I key in settings."

    @Volatile private var tts: TextToSpeech? = null

    /**
     * True from the start of answer() until the reply has finished playing. Hard gate:
     * WakeWordService refuses to start the microphone listener while this is true, no
     * matter who asks (the web side's timers, the session's re-arm). Without it the
     * listener came back mid-reply, heard Zynkbot's own voice, and sent the reply
     * back as a new question (OnePlus, 2026-09-04).
     */
    @Volatile var speaking: Boolean = false
        private set

    /** Set by the user tapping Stop (via WakeWordBridge.stopSpeaking). Cuts the current
     *  utterance, drops everything queued, and makes the streaming speaker ignore any
     *  further tokens so the reply ends now rather than at the end of the paragraph. */
    @Volatile private var abortRequested: Boolean = false

    /** MainActivity registers this to push speaking state into the WebView so the
     *  app's Stop button lights up during native speech (it only knew about web TTS). */
    @Volatile var onSpeakingChanged: ((Boolean) -> Unit)? = null

    fun stopSpeaking() {
        abortRequested = true
        try { tts?.stop() } catch (_: Exception) {}
    }

    /** Blocking: run on a background thread, never the main thread. Returns true if a
     *  reply was produced and spoken. On failure it speaks a short error line and
     *  returns false so the caller can also fall back to another delivery path. */
    fun answer(context: Context, transcript: String): Boolean {
        if (transcript.isBlank()) return false
        abortRequested = false
        speaking = true
        try { onSpeakingChanged?.invoke(true) } catch (_: Exception) {}
        try {
            return answerInner(context, transcript)
        } finally {
            speaking = false
            try { onSpeakingChanged?.invoke(false) } catch (_: Exception) {}
            // Speech is over: hand the microphone back to the wake word. Anything that
            // tried to re-arm while we were speaking was refused, so this is the one
            // re-arm that counts.
            try { WakeWordService.instance?.resumeMicAfterSession() } catch (_: Exception) {}
        }
    }

    private fun answerInner(context: Context, transcript: String): Boolean {
        val engine = engine(context) ?: return false
        val speaker = SentenceSpeaker(engine)
        var failure: String? = null
        try {
            ZynkCore.nativeSendMessage(
                transcript, "", "", "", "guardian",
                object : ZynkCore.Callback {
                    override fun onToken(token: String) { speaker.feed(token) }
                    override fun onEvent(name: String, payloadJson: String) {}
                    override fun onDone(replyJson: String) {
                        val replyText = try {
                            org.json.JSONObject(replyJson).optString("reply_text", "")
                        } catch (e: Exception) {
                            Log.w(TAG, "Could not parse native reply: ${e.message}"); ""
                        }
                        Log.i(TAG, "Native reply: \"${replyText.take(80)}\"")
                        speaker.finish(replyText)
                    }
                    override fun onError(message: String) { failure = message }
                },
            )
        } catch (e: Throwable) {
            failure = e.message ?: "native path unavailable"
        }
        if (failure != null) {
            Log.w(TAG, "Native reply failed: $failure")
            speaker.speakNow(ERROR_LINE)
            return false
        }
        return speaker.spokeAnything()
    }

    /** The closing chime WakeWordService plays — used both as "your question was sent"
     *  and as "nothing heard, closing" so a false trigger that captured only the air
     *  conditioner still gives audible feedback instead of vanishing silently.
     *  Blocking (<=2s). */
    fun playCloseTone(context: Context) {
        try {
            val mp = MediaPlayer()
            mp.setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANT)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            context.resources.openRawResourceFd(R.raw.wake_chime_close)?.let { afd ->
                mp.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length); afd.close()
            }
            mp.prepare()
            val latch = CountDownLatch(1)
            mp.setOnCompletionListener { it.release(); latch.countDown() }
            mp.start()
            latch.await(2000, TimeUnit.MILLISECONDS)
        } catch (e: Exception) {
            Log.w(TAG, "Sent tone failed: ${e.message}")
        }
    }

    fun shutdown() {
        try { tts?.shutdown() } catch (_: Exception) {}
        tts = null
    }

    // ── internals ────────────────────────────────────────────────────────────

    /** One TTS engine per process, created lazily; null if it can't initialise. */
    private fun engine(context: Context): TextToSpeech? {
        tts?.let { return it }
        val initLatch = CountDownLatch(1)
        var initOk = false
        val engine = TextToSpeech(context.applicationContext) { status ->
            initOk = status == TextToSpeech.SUCCESS; initLatch.countDown()
        }
        initLatch.await(5000, TimeUnit.MILLISECONDS)
        if (!initOk) { Log.w(TAG, "TTS init failed"); return null }
        engine.language = Locale.US
        tts = engine
        return engine
    }

    /**
     * Feeds streamed tokens, speaking each completed sentence immediately and queuing
     * the next behind it. finish() speaks any remainder (or the whole reply if nothing
     * ever streamed, e.g. a non-streaming backend) and blocks until playback ends.
     */
    private class SentenceSpeaker(private val engine: TextToSpeech) {
        private val buffer = StringBuilder()
        private val pending = java.util.concurrent.ConcurrentHashMap.newKeySet<String>()
        private val allDone = Object()
        private var utterances = 0
        private var stopped = false   // a marker was seen; ignore everything after it
        private var spokeSomething = false

        init {
            engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {}
                override fun onDone(utteranceId: String?) { settle(utteranceId) }
                @Deprecated("required override") override fun onError(utteranceId: String?) { settle(utteranceId) }
                override fun onError(utteranceId: String?, errorCode: Int) { settle(utteranceId) }
                private fun settle(id: String?) {
                    if (id != null) pending.remove(id)
                    if (pending.isEmpty()) synchronized(allDone) { allDone.notifyAll() }
                }
            })
        }

        @Synchronized fun feed(token: String) {
            if (stopped || abortRequested) return
            buffer.append(token)
            val cut = MARKERS.map { buffer.indexOf(it) }.filter { it >= 0 }.minOrNull()
            if (cut != null) {
                buffer.setLength(cut)
                stopped = true
            }
            drainSentences(force = stopped)
        }

        @Synchronized fun finish(fullReply: String) {
            if (utterances == 0 && buffer.isBlank()) {
                // Nothing streamed (non-streaming backend): speak the final text.
                buffer.append(fullReply)
            }
            drainSentences(force = true)
            awaitIdle(60_000)
        }

        fun speakNow(text: String) {
            enqueue(text, flush = true)
            awaitIdle(15_000)
        }

        fun spokeAnything() = spokeSomething

        /** Cut on sentence boundaries; with force, speak whatever is left too. */
        private fun drainSentences(force: Boolean) {
            while (true) {
                val idx = sentenceEnd(buffer)
                if (idx < 0) break
                val sentence = buffer.substring(0, idx + 1).trim()
                buffer.delete(0, idx + 1)
                if (sentence.isNotEmpty()) enqueue(sentence, flush = utterances == 0)
            }
            if (force && buffer.isNotBlank()) {
                enqueue(buffer.toString().trim(), flush = utterances == 0)
                buffer.setLength(0)
            }
        }

        /** Index of a sentence terminator followed by whitespace/end, else -1. Requires a
         *  few words first so "e.g." / "1." don't trigger a one-word utterance. */
        private fun sentenceEnd(sb: StringBuilder): Int {
            var i = 0
            while (i < sb.length) {
                val c = sb[i]
                val term = c == '.' || c == '?' || c == '!' || c == '\n'
                if (term && (i + 1 >= sb.length || sb[i + 1].isWhitespace()) && i >= 12) return i
                i++
            }
            return -1
        }

        private fun enqueue(text: String, flush: Boolean) {
            if (abortRequested) return
            val id = "zynk-${System.nanoTime()}"
            pending.add(id)
            val mode = if (flush) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
            val r = engine.speak(text, mode, null, id)
            if (r == TextToSpeech.SUCCESS) { utterances++; spokeSomething = true } else pending.remove(id)
        }

        private fun awaitIdle(timeoutMs: Long) {
            val deadline = System.currentTimeMillis() + timeoutMs
            synchronized(allDone) {
                while (pending.isNotEmpty()) {
                    val left = deadline - System.currentTimeMillis()
                    if (left <= 0) { Log.w(TAG, "TTS wait timed out with ${pending.size} pending"); break }
                    try { allDone.wait(left) } catch (_: InterruptedException) { break }
                }
            }
        }

        companion object {
            private val MARKERS = listOf("MEMORY_EXTRACT", "WEB_SEARCH_NEEDED")
        }
    }
}

package ai.containai.zynkbot

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.File

/**
 * One dictation fed from the microphone stream (2026-10-20 rebuild). Replaces the
 * copies that each opened their own recorder: Vosk's SpeechService in the wake-word
 * service, in the assistant session and in the page's mic bridge, and
 * OpenAiDictation's recorder.
 *
 * Vosk: chunks go to a Recognizer; the live text (finished utterances plus the one in
 * progress) is reported on every change. Whisper: chunks go through the energy
 * endpointer and the clip is uploaded when speech ends. Timers, hands-free only:
 * nothing said within 6 s → done; 1.5 s after the text stops changing → done; 30 s
 * cap. The app's mic button runs without timers and ends on [stop].
 *
 * Callbacks arrive on the main looper. [Listener.onFinal] gets the text ("" = nothing
 * heard) or null when the dictation was cancelled or failed.
 */
class Dictation(
    val engine: Engine,
    private val voskModel: org.vosk.Model?,
    private val apiKey: String?,
    private val timers: Boolean,
    private val listener: Listener,
) : MicStream.Sink {
    enum class Engine { VOSK, WHISPER }

    interface Listener {
        fun onPartial(text: String) {}
        fun onFinal(text: String?)
    }

    companion object {
        private const val TAG = "Dictation"
        const val SILENCE_MS = 1500L      // after the last change in the text
        const val NO_SPEECH_MS = 6000L    // chime, then nothing: close, as Siri does
        const val CAP_MS = 30_000L        // ongoing speech cannot extend it (TV)
        private const val RATE = MicStream.SAMPLE_RATE

        /** The Vosk model, loaded once per process and shared with the wake-word
         *  service's screen-off path. Null if it is not on the phone yet. */
        @Synchronized
        fun voskModel(context: Context): org.vosk.Model? {
            WakeWordService.sharedVoskModel?.let { return it }
            val dir = File(context.filesDir, "vosk-model")
            if (!dir.exists()) return null
            return try {
                org.vosk.Model(dir.absolutePath).also { WakeWordService.sharedVoskModel = it }
            } catch (e: Exception) {
                Log.w(TAG, "Vosk model load failed: ${e.message}"); null
            }
        }
    }

    private val main = Handler(Looper.getMainLooper())
    private val lock = Any()
    private var rec: org.vosk.Recognizer? = null
    private val committed = StringBuilder()
    private var lastPartial = ""
    private var samplesTotal = 0L
    private var samplesSinceChange = 0L
    private var sawSpeech = false
    private var chunks = 0
    private var partialsWithText = 0
    private val endpointer = if (engine == Engine.WHISPER) OpenAiDictation.Endpointer(timers) else null
    @Volatile private var done = false
    @Volatile var isCancelled = false
        private set
    val startedAt = System.currentTimeMillis()
    /** The stream owner's hook, run on the finishing thread before the listener hears
     *  anything, so the owner's state is right by the time the listener reacts. */
    @Volatile internal var onDone: (() -> Unit)? = null
    /** Samples still to drop before listening starts. The stream stays open through
     *  the chime, and its tail through the speaker and the room reached the endpointer
     *  as the start of speech (Pixel 2026-10-20 15:07, transcript "Beep!"). The owner
     *  sets this from how recently the chime ended. */
    @Volatile internal var skipSamples = 0L

    override fun onChunk(pcm: ShortArray, n: Int) {
        if (skipSamples > 0) { skipSamples -= n; return }
        val finished: String? = synchronized(lock) {
            if (done) return
            chunks++
            samplesTotal += n
            when (engine) {
                Engine.VOSK -> feedVosk(pcm, n)
                Engine.WHISPER -> if (endpointer!!.feed(pcm, n)) "speech ended" else null
            }
        }
        if (finished != null) finish(finished)
    }

    /** Under [lock]. Returns the reason the dictation is over, or null to carry on. */
    private fun feedVosk(pcm: ShortArray, n: Int): String? {
        val r = rec ?: try {
            val m = voskModel ?: return "no Vosk model"
            org.vosk.Recognizer(m, RATE.toFloat()).also { rec = it }
        } catch (e: Exception) { return "Vosk recognizer: ${e.message}" }
        var changed = false
        if (r.acceptWaveForm(pcm, n)) {
            val t = text(r.result, "text")
            if (t.isNotEmpty()) {
                if (committed.isNotEmpty()) committed.append(' ')
                committed.append(t)
                changed = true
                emit(committed.toString())
            }
            lastPartial = ""
        } else {
            val p = text(r.partialResult, "partial")
            if (p != lastPartial) {
                lastPartial = p
                if (p.isNotEmpty()) { changed = true; partialsWithText++; emit(join(committed.toString(), p)) }
            }
        }
        if (changed) { sawSpeech = true; samplesSinceChange = 0 } else samplesSinceChange += n
        if (!timers) return null
        val ms = samplesTotal * 1000 / RATE
        return when {
            !sawSpeech && ms >= NO_SPEECH_MS -> "no speech within ${NO_SPEECH_MS / 1000} s"
            sawSpeech && samplesSinceChange * 1000 / RATE >= SILENCE_MS -> "silence"
            ms >= CAP_MS -> "listening cap"
            else -> null
        }
    }

    /** The owner ends it: the mic button's second tap. The text so far is delivered. */
    fun stop() = finish("stopped")

    /** Abort: the listener gets onFinal(null). */
    fun cancel() {
        synchronized(lock) {
            if (done) return
            done = true; isCancelled = true
            closeVosk()
        }
        Log.i(TAG, "Dictation cancelled after ${elapsed()} ms")
        onDone?.invoke()
        main.post { listener.onFinal(null) }
    }

    private fun finish(reason: String) {
        val failed = reason.startsWith("no Vosk model") || reason.startsWith("Vosk recognizer")
        var full = ""
        var pcm = ByteArray(0)
        synchronized(lock) {
            if (done) return
            done = true
            when (engine) {
                Engine.VOSK -> {
                    val last = rec?.let { text(it.finalResult, "text") } ?: ""
                    closeVosk()
                    full = join(committed.toString(), last)
                }
                Engine.WHISPER -> pcm = endpointer!!.result()
            }
        }
        onDone?.invoke()
        if (failed) {
            Log.e(TAG, "Dictation failed: $reason")
            main.post { listener.onFinal(null) }
            return
        }
        when (engine) {
            Engine.VOSK -> {
                Log.i(TAG, "Vosk dictation ended ($reason) after ${elapsed()} ms: \"$full\" (chunks: $chunks, partials with text: $partialsWithText)")
                main.post { listener.onFinal(full) }
            }
            Engine.WHISPER -> {
                Log.i(TAG, "Whisper dictation ended ($reason) after ${elapsed()} ms: ${pcm.size / 1024} KB")
                if (pcm.isEmpty()) { main.post { listener.onFinal("") }; return }
                val key = apiKey
                if (key == null) { Log.w(TAG, "No OpenAI key"); main.post { listener.onFinal(null) }; return }
                Thread({
                    val text = OpenAiDictation.transcribe(key, OpenAiDictation.toWav(pcm))
                    main.post { if (!isCancelled) listener.onFinal(text) }
                }, "WhisperUpload").start()
            }
        }
    }

    private fun emit(live: String) { main.post { listener.onPartial(live) } }
    private fun closeVosk() { try { rec?.close() } catch (_: Exception) {}; rec = null }
    private fun elapsed() = System.currentTimeMillis() - startedAt
    private fun text(json: String?, field: String) =
        try { org.json.JSONObject(json ?: "").optString(field, "").trim() } catch (_: Exception) { "" }
    private fun join(a: String, b: String) = if (a.isEmpty()) b else if (b.isEmpty()) a else "$a $b"
}

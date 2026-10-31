package ai.containai.zynkbot

import android.content.Context
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.net.HttpURLConnection
import java.net.URL
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Whisper dictation: the energy endpointer and the upload (2026-09-08). Since the
 * 2026-10-20 rebuild this file no longer records; [Dictation] feeds it chunks from the
 * one microphone stream and calls [transcribe] with the result.
 *
 * Endpointing: wait up to SPEECH_WAIT_MS for the RMS energy to rise clearly above an
 * adaptive noise floor; once speech has started, stop after TRAILING_SILENCE_MS of
 * quiet; MAX_TOTAL_MS caps the whole recording (same as the Vosk cap).
 */
object OpenAiDictation {
    private const val TAG = "OpenAiDictation"
    private const val ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
    private const val ENV_KEY = "OPENAI_API_KEY"

    private const val SAMPLE_RATE = MicStream.SAMPLE_RATE
    private const val SPEECH_WAIT_MS = 8_000L
    private const val TRAILING_SILENCE_MS = 1_200L
    private const val MAX_TOTAL_MS = 30_000L
    private const val PAGE_CAP_MS = 180_000L               // mic button without timers: a hard stop only
    private const val PRE_ROLL_MS = 500L                   // kept from before speech onset
    // First real run (2026-09-08): a question asked from ~20 ft after a wake word
    // that itself measured -31 dBFS never crossed the 400 onset, so Whisper heard
    // nothing and the session closed. Speech across a room sits around 120–300
    // PCM16 RMS on a phone mic; 120 with 2.5× the noise floor keeps a quiet room
    // out and lets distant speech in.
    private const val MIN_SPEECH_RMS = 120.0               // PCM16 units; below this nothing counts as speech
    private const val ONSET_FLOOR_MULT = 2.5
    private const val PEAK_QUIET_RATIO = 0.12              // "quiet" is also anything under 12% of the loudest speech so far
    private const val SPEECH_ONSET_CHUNKS = 2              // consecutive loud chunks before "speaking"
    const val HTTP_TIMEOUT_MS = 20_000

    fun hasApiKey(context: Context): Boolean = EnvFile.read(context, ENV_KEY) != null
    fun apiKey(context: Context): String? = EnvFile.read(context, ENV_KEY)

    /** `DICTATION_STREAM=0` in the settings file turns the live stream ([OpenAiRealtime])
     *  off and keeps the upload-after-stop path, as on the desktop. */
    fun streamingEnabled(context: Context): Boolean =
        EnvFile.read(context, "DICTATION_STREAM")?.lowercase() !in setOf("0", "false", "off", "no")

    /**
     * Decides where the utterance starts and ends from chunk energy. [feed] returns
     * true when the recording is complete; [result] is then the PCM16 to upload, empty
     * when no speech came. With [timers] off (the mic button) everything is kept
     * until the owner stops, up to PAGE_CAP_MS.
     */
    class Endpointer(private val timers: Boolean) {
        private val out = ByteArrayOutputStream()
        private val preRoll = ArrayDeque<ByteArray>()
        private var preRollMs = 0L
        private var speaking = false
        private var loudRun = 0
        private var noiseFloor = 0.0
        private var elapsedMs = 0L
        private var silenceMs = 0L
        // Quietest level seen in the last ~2 s while speaking. The pre-speech floor was
        // frozen at the moment speech started; with an air conditioner running that
        // floor was 36 while the room never dropped below ~100, so "quiet" was never
        // reached and every turn ran to the 30 s cap (Pixel, 2026-09-13 14:11). Steady
        // noise is whatever the recent minimum is; a pause in speech falls to it.
        private val recentRms = ArrayDeque<Double>()
        private var recentMs = 0L
        private var speechPeak = 0.0
        private var lastLevelLog = 0L
        var onSpeechStarted: (() -> Unit)? = null

        fun feed(s: ShortArray, n: Int): Boolean {
            val chunkMs = n * 1000L / SAMPLE_RATE
            elapsedMs += chunkMs
            val bytes = toBytes(s, n)
            if (!timers) { out.write(bytes); return elapsedMs >= PAGE_CAP_MS }
            val rms = rms(s, n)
            if (!speaking) {
                // Track the quiet level, but never learn upwards from a burst that
                // may itself be the start of speech.
                noiseFloor = if (noiseFloor == 0.0) rms else if (rms < noiseFloor * 2) noiseFloor * 0.9 + rms * 0.1 else noiseFloor
                val onset = max(noiseFloor * ONSET_FLOOR_MULT, MIN_SPEECH_RMS)
                loudRun = if (rms > onset) loudRun + 1 else 0
                preRoll.addLast(bytes); preRollMs += chunkMs
                while (preRollMs > PRE_ROLL_MS && preRoll.size > 1) { preRoll.removeFirst(); preRollMs -= chunkMs }
                if (loudRun >= SPEECH_ONSET_CHUNKS) {
                    speaking = true
                    preRoll.forEach { out.write(it) }
                    preRoll.clear()
                    Log.i(TAG, "Speech started at ${elapsedMs}ms (floor≈${noiseFloor.toInt()})")
                    onSpeechStarted?.invoke()
                } else if (elapsedMs >= SPEECH_WAIT_MS) {
                    return true
                }
                return false
            }
            out.write(bytes)
            recentRms.addLast(rms); recentMs += chunkMs
            while (recentMs > 2_000L && recentRms.size > 4) { recentRms.removeFirst(); recentMs -= chunkMs }
            val floorNow = max(noiseFloor, recentRms.minOrNull() ?: noiseFloor)
            val quiet = max(floorNow * 1.5, MIN_SPEECH_RMS / 2)
            // Second way to be "quiet": far below the speaker's own loudest level.
            // A room that settles at 60–90 after speech (fan, muted TV, breathing
            // near the mic) never dropped under the floor-based threshold and the
            // recording ran to the 30 s cap (Pixel, 2026-09-14 and 2026-09-17).
            speechPeak = max(speechPeak, rms)
            val relQuiet = if (speechPeak > 200.0) speechPeak * PEAK_QUIET_RATIO else 0.0
            silenceMs = if (rms < quiet || rms < relQuiet) silenceMs + chunkMs else 0L
            if (elapsedMs - lastLevelLog >= 2_000L) {
                Log.i(TAG, "level ${elapsedMs}ms rms≈${rms.toInt()} floor≈${floorNow.toInt()} quiet<${quiet.toInt()} peak≈${speechPeak.toInt()} relQuiet<${relQuiet.toInt()}")
                lastLevelLog = elapsedMs
            }
            if (silenceMs >= TRAILING_SILENCE_MS) {
                Log.i(TAG, "Trailing silence at ${elapsedMs}ms (floor≈${floorNow.toInt()}, quiet<${quiet.toInt()}, peak≈${speechPeak.toInt()})")
                return true
            }
            if (elapsedMs >= MAX_TOTAL_MS) {
                Log.i(TAG, "Recording cap reached (floor≈${noiseFloor.toInt()}, recent min≈${(recentRms.minOrNull() ?: 0.0).toInt()})")
                return true
            }
            return false
        }

        /** The utterance as PCM16, with its pre-roll; empty if no speech was heard. */
        fun result(): ByteArray = if (!timers || speaking) out.toByteArray() else ByteArray(0)
    }

    private fun rms(s: ShortArray, n: Int): Double {
        var acc = 0.0
        for (i in 0 until n) { val v = s[i].toDouble(); acc += v * v }
        return sqrt(acc / n)
    }

    internal fun toBytes(s: ShortArray, n: Int): ByteArray {
        val b = ByteArray(n * 2)
        for (i in 0 until n) {
            b[i * 2] = (s[i].toInt() and 0xFF).toByte()
            b[i * 2 + 1] = (s[i].toInt() shr 8 and 0xFF).toByte()
        }
        return b
    }

    /** Canonical 44-byte RIFF header + PCM16 mono data. */
    fun toWav(pcm: ByteArray): ByteArray {
        val out = ByteArrayOutputStream(44 + pcm.size)
        fun int32(v: Int) { out.write(v and 0xFF); out.write(v shr 8 and 0xFF); out.write(v shr 16 and 0xFF); out.write(v shr 24 and 0xFF) }
        fun int16(v: Int) { out.write(v and 0xFF); out.write(v shr 8 and 0xFF) }
        out.write("RIFF".toByteArray()); int32(36 + pcm.size); out.write("WAVE".toByteArray())
        out.write("fmt ".toByteArray()); int32(16); int16(1); int16(1)
        int32(SAMPLE_RATE); int32(SAMPLE_RATE * 2); int16(2); int16(16)
        out.write("data".toByteArray()); int32(pcm.size); out.write(pcm)
        return out.toByteArray()
    }

    /** Blocking upload. The trimmed `text` field, or null on any HTTP / parse failure. The key is never logged. */
    /** What Whisper says when it hears nothing (a short clip with no speech). */
    private val SILENCE_PHRASES = setOf("thank you", "thanks", "thank you for watching", "thanks for watching", "bye", "bye bye", "goodbye", "you", "okay", "ok", "hmm", "mm", "uh", "so", "the", "yeah", "thank you very much")
    fun isSilencePhrase(text: String): Boolean {
        val w = text.lowercase().filter { it.isLetterOrDigit() || it.isWhitespace() }.trim().split(Regex("\\s+")).joinToString(" ")
        return w in SILENCE_PHRASES
    }

    fun transcribe(key: String, wav: ByteArray): String? =
        transcribeRaw(key, wav)?.let { guardSilence(it, (wav.size - 44).coerceAtLeast(0)) }

    /** The silence-phrase guard: a short clip ([pcmBytes] of PCM16) whose whole text is
     *  one of the model's "heard nothing" phrases becomes "". Shared by the upload and the
     *  live stream, so both paths treat a cough the same. */
    fun guardSilence(text: String, pcmBytes: Int): String {
        val secs = pcmBytes / (SAMPLE_RATE * 2.0)
        return if (secs < 2.5 && isSilencePhrase(text)) { Log.i(TAG, "dropped the model's silence phrase (${"%.1f".format(secs)}s)"); "" } else text
    }

    // OpenAI's newer transcription model (2026-10-28, was whisper-1): lower latency, half
    // the price. The desktop names the same one (lib.rs DICTATION_MODEL); the live stream
    // (OpenAiRealtime) asks for it too.
    const val MODEL = "gpt-4o-mini-transcribe"

    private fun transcribeRaw(key: String, wav: ByteArray): String? {
        val t0 = System.currentTimeMillis()
        val boundary = "----ZynkbotDictation" + System.currentTimeMillis()
        val conn = (URL(ENDPOINT).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = HTTP_TIMEOUT_MS
            readTimeout = HTTP_TIMEOUT_MS
            doOutput = true
            setRequestProperty("Authorization", "Bearer $key")
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
        }
        try {
            DataOutputStream(conn.outputStream).use { body ->
                fun field(name: String, value: String) {
                    body.writeBytes("--$boundary\r\nContent-Disposition: form-data; name=\"$name\"\r\n\r\n$value\r\n")
                }
                field("model", MODEL)
                field("language", "en")
                field("response_format", "json")
                body.writeBytes("--$boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n")
                body.write(wav)
                body.writeBytes("\r\n--$boundary--\r\n")
            }
            val code = conn.responseCode
            if (code != 200) {
                val err = try { conn.errorStream?.bufferedReader()?.readText()?.take(300) } catch (_: Exception) { null }
                Log.e(TAG, "HTTP $code from transcription endpoint: ${err ?: "(no body)"}")
                return null
            }
            val json = conn.inputStream.bufferedReader().readText()
            val text = org.json.JSONObject(json).optString("text", "").trim()
            val secs = (wav.size - 44).coerceAtLeast(0) / (SAMPLE_RATE * 2.0)
            Log.i(TAG, "Transcribed ${"%.1f".format(secs)}s of audio (${wav.size / 1024} KB) into ${text.length} chars in ${System.currentTimeMillis() - t0} ms ($MODEL)")
            return text
        } catch (e: Exception) {
            Log.e(TAG, "Upload failed: ${e.javaClass.simpleName}: ${e.message}")
            return null
        } finally {
            conn.disconnect()
        }
    }
}

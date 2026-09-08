package ai.containai.zynkbot

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.net.HttpURLConnection
import java.net.URL
import kotlin.math.max
import kotlin.math.sqrt

/**
 * One hands-free dictation through OpenAI Whisper, for when the Voice settings
 * selector says "OpenAI" (2026-09-08). Until now that selector only reached the
 * in-app mic button (useVoiceInput.js); the native assistant session always used
 * Vosk. This is the native counterpart: record from the mic with simple energy
 * endpointing, wrap the PCM as a WAV in memory, POST it to the transcription
 * endpoint, hand back the text.
 *
 * Result contract for onDone:
 *   - a String (possibly "") — the transcript; "" means no speech was heard
 *   - null — something failed (no key, mic, network, HTTP, parse) or the recorder
 *     was cancelled. Callers tell the two apart with [Recorder.isCancelled], so a
 *     cancel never triggers a Vosk fallback.
 *
 * Endpointing: wait up to SPEECH_WAIT_MS for the RMS energy to rise clearly above
 * an adaptive noise floor; once speech has started, stop after TRAILING_SILENCE_MS
 * of quiet; MAX_TOTAL_MS caps the whole recording (same as the session's Vosk cap).
 * Tuned on paper only — the thresholds have not yet been checked on a device.
 *
 * Mic discipline: the caller must have released the wake-word AudioRecord first
 * (see ZynkAssistantSession) — two readers on the mic wedge the session.
 */
object OpenAiDictation {
    private const val TAG = "OpenAiDictation"
    private const val ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
    private const val ENV_KEY = "OPENAI_API_KEY"

    private const val SAMPLE_RATE = 16_000
    private const val CHUNK_MS = 50L
    private const val CHUNK_SAMPLES = (SAMPLE_RATE * CHUNK_MS / 1000).toInt()
    private const val SPEECH_WAIT_MS = 8_000L
    private const val TRAILING_SILENCE_MS = 1_200L
    private const val MAX_TOTAL_MS = 12_000L
    private const val PRE_ROLL_CHUNKS = 10                 // 500 ms kept from before speech onset
    private const val MIN_SPEECH_RMS = 400.0               // PCM16 units; below this nothing counts as speech
    private const val SPEECH_ONSET_CHUNKS = 2              // consecutive loud chunks before "speaking"
    const val HTTP_TIMEOUT_MS = 20_000

    fun hasApiKey(context: Context): Boolean = EnvFile.read(context, ENV_KEY) != null

    /** Handle for an in-flight recording. cancel() aborts it; onDone(null) follows. */
    class Recorder internal constructor() {
        @Volatile private var cancelledFlag = false
        @Volatile internal var connection: HttpURLConnection? = null
        val isCancelled: Boolean get() = cancelledFlag
        fun cancel() {
            cancelledFlag = true
            // Unblock an upload in progress; the worker sees the flag and reports null.
            try { connection?.disconnect() } catch (_: Exception) {}
        }
    }

    /**
     * Records and transcribes on a background thread. Callbacks arrive on that
     * thread — hop to the main looper before touching UI.
     */
    fun record(context: Context, onSpeechStarted: () -> Unit, onDone: (String?) -> Unit): Recorder {
        val recorder = Recorder()
        Thread({
            val result = try {
                run(context, recorder, onSpeechStarted)
            } catch (e: Exception) {
                Log.e(TAG, "Dictation failed: ${e.javaClass.simpleName}: ${e.message}")
                null
            }
            onDone(if (recorder.isCancelled) null else result)
        }, "OpenAiDictation").start()
        return recorder
    }

    private fun run(context: Context, recorder: Recorder, onSpeechStarted: () -> Unit): String? {
        val key = EnvFile.read(context, ENV_KEY)
        if (key == null) { Log.w(TAG, "No $ENV_KEY in settings"); return null }
        val pcm = capture(recorder, onSpeechStarted) ?: return null
        if (recorder.isCancelled) return null
        if (pcm.isEmpty()) { Log.i(TAG, "No speech detected within ${SPEECH_WAIT_MS / 1000}s"); return "" }
        return transcribe(recorder, key, toWav(pcm))
    }

    // ── capture ──────────────────────────────────────────────────────────────

    /** Raw PCM16 of the utterance (with a short pre-roll), empty if no speech, null on mic failure. */
    private fun capture(recorder: Recorder, onSpeechStarted: () -> Unit): ByteArray? {
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val bufSize = max(minBuf, CHUNK_SAMPLES * 2 * 8)
        val audio = try {
            AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufSize)
        } catch (e: Exception) {
            Log.e(TAG, "AudioRecord init failed: ${e.message}"); return null
        }
        if (audio.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord not initialized"); audio.release(); return null
        }

        val out = ByteArrayOutputStream()
        val preRoll = ArrayDeque<ByteArray>()
        val samples = ShortArray(CHUNK_SAMPLES)
        var speaking = false
        var loudRun = 0
        var noiseFloor = 0.0
        var elapsedMs = 0L
        var silenceMs = 0L
        try {
            audio.startRecording()
            if (audio.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                Log.e(TAG, "AudioRecord did not start (mic busy?)"); return null
            }
            while (!recorder.isCancelled && elapsedMs < MAX_TOTAL_MS) {
                val n = audio.read(samples, 0, CHUNK_SAMPLES)
                if (n <= 0) { Log.e(TAG, "AudioRecord read returned $n"); return null }
                elapsedMs += n * 1000L / SAMPLE_RATE
                val rms = rms(samples, n)
                val bytes = toBytes(samples, n)

                if (!speaking) {
                    // Track the quiet level, but never learn upwards from a burst that
                    // may itself be the start of speech.
                    noiseFloor = if (noiseFloor == 0.0) rms else if (rms < noiseFloor * 2) noiseFloor * 0.9 + rms * 0.1 else noiseFloor
                    val onset = max(noiseFloor * 3, MIN_SPEECH_RMS)
                    loudRun = if (rms > onset) loudRun + 1 else 0
                    preRoll.addLast(bytes)
                    while (preRoll.size > PRE_ROLL_CHUNKS) preRoll.removeFirst()
                    if (loudRun >= SPEECH_ONSET_CHUNKS) {
                        speaking = true
                        preRoll.forEach { out.write(it) }
                        preRoll.clear()
                        Log.i(TAG, "Speech started at ${elapsedMs}ms (floor≈${noiseFloor.toInt()})")
                        onSpeechStarted()
                    } else if (elapsedMs >= SPEECH_WAIT_MS) {
                        return ByteArray(0)
                    }
                } else {
                    out.write(bytes)
                    val quiet = max(noiseFloor * 2, MIN_SPEECH_RMS / 2)
                    silenceMs = if (rms < quiet) silenceMs + CHUNK_MS else 0L
                    if (silenceMs >= TRAILING_SILENCE_MS) break
                }
            }
            if (recorder.isCancelled) return null
            if (!speaking) return ByteArray(0)
            if (elapsedMs >= MAX_TOTAL_MS) Log.i(TAG, "Recording cap reached")
            return out.toByteArray()
        } finally {
            try { audio.stop() } catch (_: Exception) {}
            audio.release()
        }
    }

    private fun rms(s: ShortArray, n: Int): Double {
        var acc = 0.0
        for (i in 0 until n) { val v = s[i].toDouble(); acc += v * v }
        return sqrt(acc / n)
    }

    private fun toBytes(s: ShortArray, n: Int): ByteArray {
        val b = ByteArray(n * 2)
        for (i in 0 until n) {
            b[i * 2] = (s[i].toInt() and 0xFF).toByte()
            b[i * 2 + 1] = (s[i].toInt() shr 8 and 0xFF).toByte()
        }
        return b
    }

    /** Canonical 44-byte RIFF header + PCM16 mono data. */
    private fun toWav(pcm: ByteArray): ByteArray {
        val out = ByteArrayOutputStream(44 + pcm.size)
        fun int32(v: Int) { out.write(v and 0xFF); out.write(v shr 8 and 0xFF); out.write(v shr 16 and 0xFF); out.write(v shr 24 and 0xFF) }
        fun int16(v: Int) { out.write(v and 0xFF); out.write(v shr 8 and 0xFF) }
        out.write("RIFF".toByteArray()); int32(36 + pcm.size); out.write("WAVE".toByteArray())
        out.write("fmt ".toByteArray()); int32(16); int16(1); int16(1)
        int32(SAMPLE_RATE); int32(SAMPLE_RATE * 2); int16(2); int16(16)
        out.write("data".toByteArray()); int32(pcm.size); out.write(pcm)
        return out.toByteArray()
    }

    // ── upload ───────────────────────────────────────────────────────────────

    /** The trimmed `text` field, or null on any HTTP / parse failure. The key is never logged. */
    private fun transcribe(recorder: Recorder, key: String, wav: ByteArray): String? {
        val boundary = "----ZynkbotDictation" + System.currentTimeMillis()
        val conn = (URL(ENDPOINT).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = HTTP_TIMEOUT_MS
            readTimeout = HTTP_TIMEOUT_MS
            doOutput = true
            setRequestProperty("Authorization", "Bearer $key")
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
        }
        recorder.connection = conn
        try {
            DataOutputStream(conn.outputStream).use { body ->
                fun field(name: String, value: String) {
                    body.writeBytes("--$boundary\r\nContent-Disposition: form-data; name=\"$name\"\r\n\r\n$value\r\n")
                }
                field("model", "whisper-1")
                field("language", "en")
                field("response_format", "json")
                body.writeBytes("--$boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n")
                body.write(wav)
                body.writeBytes("\r\n--$boundary--\r\n")
            }
            val code = conn.responseCode
            if (recorder.isCancelled) return null
            if (code != 200) {
                val err = try { conn.errorStream?.bufferedReader()?.readText()?.take(300) } catch (_: Exception) { null }
                Log.e(TAG, "HTTP $code from transcription endpoint: ${err ?: "(no body)"}")
                return null
            }
            val json = conn.inputStream.bufferedReader().readText()
            val text = org.json.JSONObject(json).optString("text", "").trim()
            Log.i(TAG, "Transcribed ${wav.size / 1024} KB in ${text.length} chars")
            return text
        } catch (e: Exception) {
            if (!recorder.isCancelled) Log.e(TAG, "Upload failed: ${e.javaClass.simpleName}: ${e.message}")
            return null
        } finally {
            recorder.connection = null
            conn.disconnect()
        }
    }
}

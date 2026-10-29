package ai.containai.zynkbot

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.io.ByteArrayOutputStream
import java.util.Base64
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Live dictation through OpenAI's Realtime transcription session (2026-10-28), the
 * Android copy of the desktop's openai_realtime.rs.
 *
 * The OpenAI engine used to hold the whole clip and upload one WAV when speech ended,
 * so the text arrived after a wait as long as the clip itself. This class opens a
 * WebSocket to `wss://api.openai.com/v1/realtime?intent=transcription` when the
 * dictation starts and sends the microphone audio as it is captured. The server
 * transcribes while the user is still speaking and sends the words back in
 * `...input_audio_transcription.delta` events, which [Dictation] hands to its
 * listener the same way Vosk's partial text goes.
 *
 * An addition, not a replacement: [Dictation] still keeps the clip in the energy
 * endpointer and decides when speech ended. When the socket cannot be opened within
 * [CONNECT_TIMEOUT_MS], the server reports an error, the socket closes before stop, or
 * the session ends with no text, [finish] returns null and the old one-shot upload
 * runs exactly as before.
 *
 * Protocol (the GA Realtime API; the "realtime=v1" beta shape was refused on the
 * desktop on 2026-10-28 with "The Realtime Beta API is no longer supported"):
 * `session.update` with a transcription session to configure, `input_audio_buffer.append`
 * with base64 PCM16 at 24 kHz mono, about 100 ms per message, and
 * `input_audio_buffer.commit` on stop. The API takes pcm16 at 24 kHz only, so the
 * 16 kHz microphone audio is resampled here. The key is never logged.
 */
class OpenAiRealtime(
    private val apiKey: String,
    captureRate: Int,
    /** The running transcript (finished segments plus the one being spoken) on every change, on an OkHttp thread. */
    private val onLive: (String) -> Unit,
) {
    companion object {
        private const val TAG = "OpenAiRealtime"
        /** The Realtime API takes pcm16 at this rate only. */
        const val TARGET_RATE = 24_000
        private const val URL = "wss://api.openai.com/v1/realtime?intent=transcription"
        const val CONNECT_TIMEOUT_MS = 5_000L
        /** How long [finish] waits for the transcript of the last words after committing. */
        const val FINAL_WAIT_MS = 3_000L
        /** Audio goes out in pieces of about this long (100 ms at 24 kHz), in bytes. */
        const val SEND_CHUNK_BYTES = TARGET_RATE / 10 * 2
        /** Audio held while the socket is still connecting; more than this means the connect is stuck. */
        private const val MAX_STAGED_BYTES = TARGET_RATE * 2 * 12

        private val client: OkHttpClient by lazy {
            OkHttpClient.Builder()
                .connectTimeout(CONNECT_TIMEOUT_MS, TimeUnit.MILLISECONDS)
                // Covers the upgrade handshake only; OkHttp clears it once the socket is up.
                .readTimeout(CONNECT_TIMEOUT_MS, TimeUnit.MILLISECONDS)
                .pingInterval(20, TimeUnit.SECONDS)
                .build()
        }

        fun sessionUpdate(): String =
            """{"type":"session.update","session":{"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":$TARGET_RATE},"transcription":{"model":"${OpenAiDictation.MODEL}","language":"en"},"turn_detection":{"type":"server_vad","silence_duration_ms":700}}}}}"""

        /** One `input_audio_buffer.append` message carrying [n] bytes of [pcm16] (little-endian, 24 kHz). */
        fun appendEvent(pcm16: ByteArray, n: Int = pcm16.size): String {
            val b64 = Base64.getEncoder().encodeToString(if (n == pcm16.size) pcm16 else pcm16.copyOf(n))
            return """{"type":"input_audio_buffer.append","audio":"$b64"}"""
        }

        fun parseEvent(json: String): Event {
            val v = try { MiniJson.parse(json) as? Map<*, *> } catch (e: Exception) {
                return Event.Error("bad_json", e.message ?: "bad json")
            } ?: return Event.Error("bad_json", "not an object")
            val type = v["type"] as? String ?: ""
            // Tolerant of the name family: the GA docs and the beta both say
            // "conversation.item.input_audio_transcription.*"; should the server send
            // "...audio_transcription.*" or "transcription.*" instead, the suffix decides.
            return when {
                type == "error" -> Event.Error(MiniJson.string(v, "error", "code"), MiniJson.string(v, "error", "message") ?: "unknown error")
                type == "input_audio_buffer.committed" -> Event.Committed
                type.endsWith("transcription.delta") -> Event.Delta(v["delta"] as? String ?: v["text"] as? String ?: "")
                type.endsWith("transcription.completed") || type.endsWith("transcription.done") ->
                    Event.Completed(v["transcript"] as? String ?: v["text"] as? String ?: "")
                type.endsWith("transcription.failed") -> Event.Failed(MiniJson.string(v, "error", "message") ?: "transcription failed")
                else -> Event.Other(type)
            }
        }
    }

    // ── Server events ──────────────────────────────────────────────────────

    sealed class Event {
        /** Part of the transcript of the segment being spoken now. */
        data class Delta(val text: String) : Event()
        /** The finished transcript of one segment. */
        data class Completed(val text: String) : Event()
        /** The server (or our stop) closed a segment; its transcript is still coming. */
        object Committed : Event()
        /** Transcription of one segment failed; nothing more will come for it. */
        data class Failed(val message: String) : Event()
        data class Error(val code: String?, val message: String) : Event()
        data class Other(val type: String) : Event()
    }

    /** Builds the running transcript from the events: finished segments plus the deltas
     *  of the one still being spoken. */
    class Transcript {
        private val committed = StringBuilder()
        private val partial = StringBuilder()
        /** Segments the server has closed whose transcript has not arrived yet. */
        var pending = 0
            private set

        /** Applies one event; true when the text to show has changed. */
        fun apply(ev: Event): Boolean = when (ev) {
            is Event.Delta -> if (ev.text.isEmpty()) false else { partial.append(ev.text); true }
            is Event.Completed -> {
                if (pending > 0) pending--
                val joined = join(committed.toString(), ev.text)
                committed.setLength(0); committed.append(joined)
                partial.setLength(0)
                true
            }
            Event.Committed -> { pending++; false }
            is Event.Failed -> {
                if (pending > 0) pending--
                val changed = partial.isNotEmpty()
                partial.setLength(0)
                changed
            }
            is Event.Error, is Event.Other -> false
        }

        fun text(): String = join(committed.toString(), partial.toString())

        private fun join(a: String, b: String): String {
            val x = a.trim(); val y = b.trim()
            return if (x.isEmpty()) y else if (y.isEmpty()) x else "$x $y"
        }
    }

    // ── Resampling ─────────────────────────────────────────────────────────

    /**
     * Linear-interpolation resampler that keeps its place between chunks, so a stream
     * of microphone chunks resamples the same as the whole clip would. The position is
     * kept in whole units of 1/[to] of an input sample, so it is exact and chunking never
     * shifts it.
     */
    class Resampler(from: Int, to: Int) {
        private val from = from.coerceAtLeast(1).toLong()
        private val to = to.coerceAtLeast(1).toLong()
        private var pos = 0L
        private var prev: Short? = null

        fun process(input: ShortArray, n: Int = input.size): ShortArray {
            if (from == to) return input.copyOf(n)
            if (n == 0) return ShortArray(0)
            // Virtual buffer: the last sample of the previous chunk, then this chunk.
            val lead = if (prev != null) 1 else 0
            val buf = ShortArray(n + lead)
            prev?.let { buf[0] = it }
            System.arraycopy(input, 0, buf, lead, n)
            val out = ShortArray((n.toLong() * to / from).toInt() + 2)
            var count = 0
            while (true) {
                val i = (pos / to).toInt()
                if (i + 1 >= buf.size) break
                val frac = (pos % to).toDouble() / to.toDouble()
                val a = buf[i].toDouble()
                val b = buf[i + 1].toDouble()
                val v = Math.round(a + (b - a) * frac).coerceIn(Short.MIN_VALUE.toLong(), Short.MAX_VALUE.toLong())
                out[count++] = v.toShort()
                pos += from
            }
            prev = buf[buf.size - 1]
            pos -= (buf.size - 1).toLong() * to
            return out.copyOf(count)
        }
    }

    // ── The session ────────────────────────────────────────────────────────

    private val lock = Any()
    private val resampler = Resampler(captureRate, TARGET_RATE)
    private val transcript = Transcript()
    private val staged = ByteArrayOutputStream(SEND_CHUNK_BYTES * 2)
    private val done = CountDownLatch(1)
    private var socket: WebSocket? = null
    private var opened = false
    private var stopping = false
    private var commitOnOpen = false
    private var committedSent = false
    private var commitAcked = false
    private var failed: String? = null
    private var lastError: String? = null
    private var startedAt = 0L
    private var stoppedAt = 0L
    /** The event types seen that this session does not handle, logged once each so the
     *  server's real names show in logcat. */
    private val unknownTypes = HashSet<String>()

    /** Starts connecting in the background; audio fed before the socket is up waits here. */
    fun open() {
        val req = Request.Builder()
            .url(URL)
            .header("Authorization", "Bearer $apiKey")
            .build()
        synchronized(lock) {
            startedAt = System.currentTimeMillis()
            socket = client.newWebSocket(req, listener)
        }
    }

    /** Microphone chunk at the capture rate, on the audio thread. Cheap: resample and enqueue. */
    fun feed(pcm: ShortArray, n: Int) {
        val out = resampler.process(pcm, n)
        if (out.isEmpty()) return
        val bytes = OpenAiDictation.toBytes(out, out.size)
        synchronized(lock) {
            if (failed != null || stopping) return
            if (!opened) {
                if (System.currentTimeMillis() - startedAt > CONNECT_TIMEOUT_MS) { failLocked("connect timed out"); return }
                if (staged.size() + bytes.size > MAX_STAGED_BYTES) { failLocked("connect stalled"); return }
                staged.write(bytes)
                return
            }
            staged.write(bytes)
            if (staged.size() >= SEND_CHUNK_BYTES) flushLocked()
        }
    }

    /**
     * Commits the last audio and waits for its transcript. Blocking; call off the main
     * thread. The whole transcript of the recording, or null when the session failed,
     * never opened, or heard nothing: the caller should upload the clip instead.
     */
    fun finish(): String? {
        val wasOpen: Boolean
        synchronized(lock) {
            if (failed != null) { Log.w(TAG, "Live stream had failed ($failed) — uploading the clip"); return null }
            stopping = true
            stoppedAt = System.currentTimeMillis()
            wasOpen = opened
            if (opened) commitLocked() else commitOnOpen = true
        }
        val wait = FINAL_WAIT_MS + if (wasOpen) 0 else CONNECT_TIMEOUT_MS
        val inTime = try { done.await(wait, TimeUnit.MILLISECONDS) } catch (_: InterruptedException) { false }
        synchronized(lock) {
            try { socket?.close(1000, null) } catch (_: Exception) {}
            socket = null
            if (!inTime) Log.w(TAG, "Live stream: gave up waiting for the last transcript after ${wait} ms (pending ${transcript.pending})")
            if (!committedSent) {
                Log.w(TAG, "Live stream never got to commit (${failed ?: "not open"}) — uploading the clip")
                return null
            }
            val text = transcript.text()
            if (text.isEmpty()) {
                Log.w(TAG, "Live stream ended with no text${lastError?.let { " ($it)" } ?: ""} — uploading the clip")
                return null
            }
            Log.i(TAG, "Live stream finished ${System.currentTimeMillis() - stoppedAt} ms after stop: ${text.length} chars (${OpenAiDictation.MODEL})")
            return text
        }
    }

    /** Drops the session without waiting (the dictation was cancelled or heard no speech). */
    fun abort() {
        synchronized(lock) {
            stopping = true
            try { socket?.cancel() } catch (_: Exception) {}
            socket = null
        }
        done.countDown()
    }

    private fun flushLocked() {
        if (staged.size() == 0) return
        val all = staged.toByteArray()
        staged.reset()
        var off = 0
        while (off < all.size) {
            val len = minOf(SEND_CHUNK_BYTES, all.size - off)
            val piece = if (off == 0 && len == all.size) all else all.copyOfRange(off, off + len)
            if (socket?.send(appendEvent(piece)) != true) { failLocked("send failed"); return }
            off += len
        }
    }

    private fun commitLocked() {
        flushLocked()
        if (failed != null) return
        if (socket?.send("""{"type":"input_audio_buffer.commit"}""") != true) { failLocked("commit send failed"); return }
        committedSent = true
    }

    private fun failLocked(reason: String) {
        if (failed == null) {
            failed = reason
            Log.w(TAG, "Live stream failed: $reason")
        }
        try { socket?.cancel() } catch (_: Exception) {}
        socket = null
        done.countDown()
    }

    private val listener = object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            synchronized(lock) {
                if (failed != null || socket !== webSocket) return
                opened = true
                Log.i(TAG, "Live stream connected in ${System.currentTimeMillis() - startedAt} ms (${OpenAiDictation.MODEL})")
                if (!webSocket.send(sessionUpdate())) { failLocked("session update send failed"); return }
                if (commitOnOpen) commitLocked() else flushLocked()
            }
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            val ev = parseEvent(text)
            var live: String? = null
            synchronized(lock) {
                if (socket !== webSocket) return
                when (ev) {
                    is Event.Error -> {
                        // After our commit, an "empty buffer" error only means the server's
                        // VAD had already closed the last segment: nothing more is coming.
                        val emptyCommit = ev.code?.contains("commit_empty") == true
                        if (committedSent) {
                            commitAcked = true
                            if (!emptyCommit) lastError = ev.message
                        } else {
                            failLocked("server error: ${ev.message}")
                            return
                        }
                    }
                    Event.Committed -> if (committedSent) commitAcked = true
                    is Event.Failed -> Log.w(TAG, "Live stream segment failed: ${ev.message}")
                    is Event.Other -> if (unknownTypes.add(ev.type)) Log.i(TAG, "Live stream event not handled: ${ev.type}")
                    else -> {}
                }
                if (transcript.apply(ev)) live = transcript.text()
                if (committedSent && commitAcked && transcript.pending == 0) done.countDown()
            }
            live?.let(onLive)
        }

        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
            synchronized(lock) {
                if (socket !== webSocket) return
                if (!committedSent) failLocked("server closed the session ($code)") else done.countDown()
            }
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            synchronized(lock) { if (socket === webSocket) done.countDown() }
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            synchronized(lock) {
                if (socket !== webSocket) return
                val what = response?.let { "HTTP ${it.code}" } ?: "${t.javaClass.simpleName}: ${t.message}"
                if (!committedSent) failLocked("socket: $what") else { Log.w(TAG, "Live stream dropped after commit: $what"); done.countDown() }
            }
        }
    }
}

/**
 * The few JSON shapes the Realtime events use, read without org.json: that class is a
 * stub on the JVM, so the event parsing could not have unit tests with it. Objects
 * come back as Map<String, Any?>, arrays as List<Any?>, numbers as Double.
 */
internal object MiniJson {
    fun parse(s: String): Any? {
        val p = Parser(s)
        val v = p.value()
        p.skipWs()
        require(p.i == s.length) { "trailing data at ${p.i}" }
        return v
    }

    /** The string at [path] inside nested objects, or null when absent or not a string. */
    fun string(v: Any?, vararg path: String): String? {
        var cur: Any? = v
        for (k in path) cur = (cur as? Map<*, *>)?.get(k) ?: return null
        return cur as? String
    }

    private class Parser(val s: String) {
        var i = 0
        fun skipWs() { while (i < s.length && s[i].isWhitespace()) i++ }
        private fun fail(msg: String): Nothing = throw IllegalArgumentException("$msg at $i")
        private fun next(): Char { if (i >= s.length) fail("unexpected end"); return s[i++] }

        fun value(): Any? {
            skipWs()
            if (i >= s.length) fail("unexpected end")
            return when (val c = s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> str()
                't' -> lit("true", true)
                'f' -> lit("false", false)
                'n' -> lit("null", null)
                else -> if (c == '-' || c.isDigit()) num() else fail("unexpected '$c'")
            }
        }

        private fun obj(): Map<String, Any?> {
            i++
            val m = LinkedHashMap<String, Any?>()
            skipWs()
            if (i < s.length && s[i] == '}') { i++; return m }
            while (true) {
                skipWs()
                if (i >= s.length || s[i] != '"') fail("expected key")
                val k = str()
                skipWs()
                if (next() != ':') fail("expected ':'")
                m[k] = value()
                skipWs()
                when (next()) { ',' -> continue; '}' -> return m; else -> fail("expected ',' or '}'") }
            }
        }

        private fun arr(): List<Any?> {
            i++
            val l = ArrayList<Any?>()
            skipWs()
            if (i < s.length && s[i] == ']') { i++; return l }
            while (true) {
                l.add(value())
                skipWs()
                when (next()) { ',' -> continue; ']' -> return l; else -> fail("expected ',' or ']'") }
            }
        }

        private fun str(): String {
            i++
            val sb = StringBuilder()
            while (true) {
                when (val c = next()) {
                    '"' -> return sb.toString()
                    '\\' -> when (val e = next()) {
                        '"', '\\', '/' -> sb.append(e)
                        'b' -> sb.append('\b')
                        'f' -> sb.append('\u000C')
                        'n' -> sb.append('\n')
                        'r' -> sb.append('\r')
                        't' -> sb.append('\t')
                        'u' -> {
                            if (i + 4 > s.length) fail("bad \\u escape")
                            sb.append(s.substring(i, i + 4).toInt(16).toChar()); i += 4
                        }
                        else -> fail("bad escape '\\$e'")
                    }
                    else -> sb.append(c)
                }
            }
        }

        private fun num(): Double {
            val start = i
            if (s[i] == '-') i++
            while (i < s.length && (s[i].isDigit() || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-')) i++
            return s.substring(start, i).toDoubleOrNull() ?: fail("bad number")
        }

        private fun lit(word: String, v: Any?): Any? {
            if (!s.startsWith(word, i)) fail("expected $word")
            i += word.length
            return v
        }
    }
}

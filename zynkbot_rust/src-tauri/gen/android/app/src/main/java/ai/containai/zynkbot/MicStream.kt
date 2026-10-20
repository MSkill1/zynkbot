package ai.containai.zynkbot

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.util.Log

/**
 * The one microphone stream (2026-10-20 rebuild). Opens a single AudioRecord and
 * delivers 80 ms chunks to whatever [sink] is set, on its own thread. The owner swaps
 * the sink as its state changes (detector, dictation, nothing), so the wake-word
 * detector and dictation never hold two recorders, and the recording watch only ever
 * sees one stream of ours. Picovoice's VoiceProcessor is the same shape.
 */
class MicStream(private val tag: String = "MicStream") {
    fun interface Sink { fun onChunk(pcm: ShortArray, n: Int) }

    companion object {
        const val SAMPLE_RATE = 16000
        const val CHUNK_SAMPLES = 1280   // 80 ms, the wake models' step
    }

    /** Receives every chunk on the audio thread. May be swapped at any time. */
    @Volatile var sink: Sink? = null
    /** Called on the audio thread after a read failure; the stream has stopped. */
    @Volatile var onLost: ((String) -> Unit)? = null
    /** Our AudioRecord's session id while open, else -1; the recording watch uses it
     *  to tell our stream from other apps'. Set before recording starts. */
    @Volatile var sessionId = -1
        private set
    @Volatile private var running = false
    private var thread: Thread? = null
    private var aec: AcousticEchoCanceler? = null

    val isOpen: Boolean get() = running && thread?.isAlive == true

    /** Open the microphone and start delivering chunks. False if it could not be opened. */
    @Synchronized
    fun open(): Boolean {
        if (isOpen) return true
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val bufSize = maxOf(minBuf, CHUNK_SAMPLES * 2 * 8)
        val rec = try {
            AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufSize)
        } catch (e: Exception) {
            Log.e(tag, "AudioRecord init failed: ${e.message}"); return false
        }
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(tag, "AudioRecord not initialized"); rec.release(); return false
        }
        sessionId = rec.audioSessionId
        try { rec.startRecording() } catch (e: Exception) {
            Log.e(tag, "startRecording failed: ${e.message}"); rec.release(); sessionId = -1; return false
        }
        if (rec.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
            Log.e(tag, "AudioRecord did not start (mic busy?)"); rec.release(); sessionId = -1; return false
        }
        // Ask Android to subtract the phone's own speaker output from this stream. Whether
        // the device really does it is only known by trying; the log says what happened.
        try {
            if (AcousticEchoCanceler.isAvailable()) {
                aec = AcousticEchoCanceler.create(rec.audioSessionId)?.also { it.setEnabled(true) }
                Log.i(tag, "Echo canceller: ${if (aec?.enabled == true) "enabled" else "not enabled"}")
            } else Log.i(tag, "Echo canceller: not available on this device")
        } catch (e: Exception) { Log.w(tag, "Echo canceller: ${e.message}") }
        running = true
        val t = Thread({
            val chunk = ShortArray(CHUNK_SAMPLES)
            var lost: String? = null
            while (running) {
                val n = rec.read(chunk, 0, CHUNK_SAMPLES)
                if (n == CHUNK_SAMPLES) {
                    try { sink?.onChunk(chunk, n) } catch (e: Exception) { Log.e(tag, "Sink error: ${e.javaClass.simpleName}: ${e.message}") }
                } else if (n < 0) { lost = "read returned $n"; break }
            }
            try { rec.stop() } catch (_: Exception) {}
            rec.release()
            try { aec?.release() } catch (_: Exception) {}
            aec = null
            sessionId = -1
            Log.i(tag, "Microphone stream closed${if (lost != null) " ($lost)" else ""}")
            if (lost != null && running) { running = false; onLost?.invoke(lost) }
        }, "MicStream")
        thread = t
        t.start()
        Log.i(tag, "Microphone stream open (session $sessionId)")
        return true
    }

    /** Stop delivering and release the microphone. Waits for the thread (≤ 500 ms)
     *  unless called from it. */
    @Synchronized
    fun close() {
        running = false
        val t = thread ?: return
        thread = null
        if (t !== Thread.currentThread()) {
            t.interrupt()
            try { t.join(500) } catch (_: InterruptedException) {}
        }
    }
}

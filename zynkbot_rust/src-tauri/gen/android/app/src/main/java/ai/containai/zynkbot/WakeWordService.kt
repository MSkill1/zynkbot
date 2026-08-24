package ai.containai.zynkbot

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import java.io.File
import java.nio.FloatBuffer
import java.util.Collections

class WakeWordService : Service() {

    companion object {
        const val CHANNEL_ID = "wake_word_channel"
        const val NOTIFICATION_ID = 1003
        const val TAG = "WakeWordService"

        // openWakeWord pipeline — shapes verified empirically against the ONNX models:
        //   mel:       [1, 1280] float32  →  [1, 1, 5, 32]  (5 mel frames × 32 bins per 80ms chunk)
        //   embedding: [1, 76, 32, 1]     →  [1, 1, 1, 96]  (76 mel frames → 96-dim vector)
        //   classifier:[1, 16, 96]        →  [1, 1]          (16 embeddings → probability)
        const val SAMPLE_RATE = 16000
        const val CHUNK_SAMPLES = 1280      // 80ms per chunk at 16kHz
        const val MEL_FRAMES_PER_CHUNK = 5  // mel frames produced per 1280-sample chunk
        const val MEL_BINS = 32
        const val MEL_WINDOW = 76           // frames the embedding model expects
        const val EMB_SIZE = 96
        const val EMB_WINDOW = 16           // embeddings the classifier expects
        const val COOLDOWN_CHUNKS = 50      // ~4 seconds before re-triggering

        // Set by WakeWordBridge before starting; called on detection.
        @Volatile var detectionCallback: (() -> Unit)? = null
    }

    private var ortEnv: OrtEnvironment? = null
    private var melSession: OrtSession? = null
    private var embSession: OrtSession? = null
    private var kwsSession: OrtSession? = null

    private val melBuffer = ArrayDeque<FloatArray>()
    private val embBuffer = ArrayDeque<FloatArray>()
    private var cooldownRemaining = 0

    @Volatile private var running = false
    private var audioThread: Thread? = null
    private var threshold = 0.5f

    inner class LocalBinder : android.os.Binder() {
        fun getService(): WakeWordService = this@WakeWordService
    }

    override fun onBind(intent: Intent?): IBinder = LocalBinder()

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        threshold = intent?.getFloatExtra("threshold", 0.5f) ?: 0.5f
        val modelDir = intent?.getStringExtra("modelDir") ?: return START_NOT_STICKY

        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Zynkbot")
            .setContentText("Listening for \"Hey Zynk\"")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            } else {
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            }
            startForeground(NOTIFICATION_ID, notification, type)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        Thread { loadAndStart(modelDir) }.start()
        return START_STICKY
    }

    override fun onDestroy() {
        stop()
        super.onDestroy()
    }

    private fun loadAndStart(modelDir: String) {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val env = ortEnv!!
            val dir = File(modelDir)

            melSession = env.createSession(File(dir, "melspectrogram.onnx").absolutePath)
            embSession = env.createSession(File(dir, "embedding_model.onnx").absolutePath)
            kwsSession = env.createSession(File(dir, "hey_zynk.onnx").absolutePath)

            Log.i(TAG, "ONNX models loaded. mel inputs: ${melSession!!.inputNames}, emb inputs: ${embSession!!.inputNames}, kws inputs: ${kwsSession!!.inputNames}")

            startAudioLoop()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ONNX models: ${e.message}")
            stopSelf()
        }
    }

    private fun startAudioLoop() {
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val bufSize = maxOf(minBuf, CHUNK_SAMPLES * 2 * 8)

        val audioRecord = try {
            AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufSize)
        } catch (e: Exception) {
            Log.e(TAG, "AudioRecord init failed: ${e.message}")
            stopSelf()
            return
        }

        if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord not initialized")
            audioRecord.release()
            stopSelf()
            return
        }

        running = true
        val chunk = ShortArray(CHUNK_SAMPLES)
        audioRecord.startRecording()
        Log.i(TAG, "Wake word audio capture started")

        audioThread = Thread {
            while (running) {
                val read = audioRecord.read(chunk, 0, CHUNK_SAMPLES)
                if (read == CHUNK_SAMPLES) {
                    processChunk(chunk)
                }
            }
            audioRecord.stop()
            audioRecord.release()
            Log.i(TAG, "Wake word audio capture stopped")
        }
        audioThread!!.start()
    }

    private fun processChunk(pcm16: ShortArray) {
        val env = ortEnv ?: return
        val mel = melSession ?: return
        val emb = embSession ?: return
        val kws = kwsSession ?: return

        if (cooldownRemaining > 0) { cooldownRemaining--; return }

        try {
            // Step 1: PCM16 → float32 [-1, 1]
            val audioFloat = FloatArray(CHUNK_SAMPLES) { pcm16[it].toFloat() / 32768f }

            // Step 2: Mel model — input [1, 1280], output [1, 1, 5, 32]
            // Extract the 5 mel frames (shape [MEL_FRAMES_PER_CHUNK, MEL_BINS]) and add to buffer
            val melOut = runModel(env, mel, audioFloat, longArrayOf(1, CHUNK_SAMPLES.toLong()))
            // melOut is flat [1*1*5*32 = 160], reshape into 5 frames of 32 bins
            for (f in 0 until MEL_FRAMES_PER_CHUNK) {
                val frame = FloatArray(MEL_BINS) { melOut[f * MEL_BINS + it] }
                melBuffer.addLast(frame)
            }
            while (melBuffer.size > MEL_WINDOW) melBuffer.removeFirst()
            if (melBuffer.size < MEL_WINDOW) return

            // Step 3: Embedding model — input [1, 76, 32, 1], output [1, 1, 1, 96]
            // Flatten mel buffer to [76 * 32] then reshape in model as [1, 76, 32, 1]
            val flatMel = FloatArray(MEL_WINDOW * MEL_BINS)
            melBuffer.forEachIndexed { i, frame -> frame.copyInto(flatMel, i * MEL_BINS) }
            val embOut = runModel(env, emb, flatMel, longArrayOf(1, MEL_WINDOW.toLong(), MEL_BINS.toLong(), 1L))
            // embOut is [96] after flattening [1,1,1,96]
            embBuffer.addLast(embOut)
            while (embBuffer.size > EMB_WINDOW) embBuffer.removeFirst()
            if (embBuffer.size < EMB_WINDOW) return

            // Step 4: Classifier — input [1, 16, 96], output [1, 1]
            val flatEmb = FloatArray(EMB_WINDOW * EMB_SIZE)
            embBuffer.forEachIndexed { i, e -> e.copyInto(flatEmb, i * EMB_SIZE) }
            val prob = runModel(env, kws, flatEmb, longArrayOf(1, EMB_WINDOW.toLong(), EMB_SIZE.toLong()))

            val score = prob.firstOrNull() ?: return
            if (score > threshold) {
                Log.i(TAG, "Wake word detected! score=$score threshold=$threshold")
                cooldownRemaining = COOLDOWN_CHUNKS
                embBuffer.clear()
                detectionCallback?.invoke()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}")
        }
    }

    private fun runModel(env: OrtEnvironment, session: OrtSession, data: FloatArray, shape: LongArray): FloatArray {
        val inputName = session.inputNames.iterator().next()
        val buf = FloatBuffer.allocate(data.size)
        buf.put(data); buf.rewind()
        val tensor = OnnxTensor.createTensor(env, buf, shape)
        val results = session.run(Collections.singletonMap(inputName, tensor))
        tensor.close()
        val outTensor = results[0] as OnnxTensor
        val outBuf = outTensor.floatBuffer
        val out = FloatArray(outBuf.remaining()); outBuf.get(out)
        results.close()
        return out
    }

    fun stop() {
        running = false
        audioThread?.interrupt()
        audioThread = null
        melSession?.close(); melSession = null
        embSession?.close(); embSession = null
        kwsSession?.close(); kwsSession = null
        ortEnv?.close(); ortEnv = null
        melBuffer.clear()
        embBuffer.clear()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Hey Zynk",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Listens for the wake word in the background"
                setShowBadge(false)
            }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }
}

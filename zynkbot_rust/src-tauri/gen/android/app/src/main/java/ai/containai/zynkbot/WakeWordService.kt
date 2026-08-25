package ai.containai.zynkbot

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import java.io.File
import java.nio.FloatBuffer
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class WakeWordService : Service() {

    companion object {
        const val CHANNEL_ID = "wake_word_channel"
        const val TRANSCRIPT_CHANNEL_ID = "wake_word_transcript_channel"
        const val NOTIFICATION_ID = 1003
        const val TRANSCRIPT_NOTIFICATION_ID = 1005
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

        // Set by WakeWordBridge before starting; called on detection when screen is on.
        @Volatile var detectionCallback: (() -> Unit)? = null

        // Vosk model shared from VoskBridge so screen-off dictation doesn't reload it.
        @Volatile var sharedVoskModel: org.vosk.Model? = null
    }

    private var ortEnv: OrtEnvironment? = null
    private var melSession: OrtSession? = null
    private var embSession: OrtSession? = null
    private var kwsSession: OrtSession? = null

    private val melBuffer = ArrayDeque<FloatArray>()
    private val embBuffer = ArrayDeque<FloatArray>()
    private var cooldownRemaining = 0
    private var consecutiveHighScores = 0

    @Volatile private var running = false
    @Volatile private var audioReleased = false
    private var audioThread: Thread? = null
    private var threshold = 0.5f
    private var wakeLock: PowerManager.WakeLock? = null      // detection wake lock (25s, screen-off flow)
    private var audioWakeLock: PowerManager.WakeLock? = null // CPU wake lock for audio loop while screen off

    // Keeps the CPU running the ONNX inference loop when the screen is off.
    private val screenReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                Intent.ACTION_SCREEN_OFF -> acquireAudioWakeLock()
                Intent.ACTION_SCREEN_ON  -> releaseAudioWakeLock()
            }
        }
    }

    private fun acquireAudioWakeLock() {
        if (audioWakeLock?.isHeld == true) return
        val pm = getSystemService(PowerManager::class.java)
        audioWakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "zynkbot:audio_loop")
        audioWakeLock?.acquire()
        Log.i(TAG, "Audio loop CPU wake lock acquired")
    }

    private fun releaseAudioWakeLock() {
        try { if (audioWakeLock?.isHeld == true) audioWakeLock?.release() } catch (_: Exception) {}
        audioWakeLock = null
        Log.i(TAG, "Audio loop CPU wake lock released")
    }

    inner class LocalBinder : android.os.Binder() {
        fun getService(): WakeWordService = this@WakeWordService
    }

    override fun onBind(intent: Intent?): IBinder = LocalBinder()

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_OFF)
            addAction(Intent.ACTION_SCREEN_ON)
        }
        registerReceiver(screenReceiver, filter)
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

        try {
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
        } catch (e: SecurityException) {
            Log.w(TAG, "App not in foreground, cannot start FGS: ${e.message}")
            stopSelf()
            return START_NOT_STICKY
        }

        // Pre-load Vosk model in background so it's ready for screen-off dictation.
        if (sharedVoskModel == null) {
            Thread {
                val voskDir = File(filesDir, "vosk-model")
                if (voskDir.exists()) {
                    try {
                        sharedVoskModel = org.vosk.Model(voskDir.absolutePath)
                        Log.i(TAG, "Vosk model pre-loaded for screen-off dictation")
                    } catch (e: Exception) {
                        Log.w(TAG, "Vosk pre-load failed: ${e.message}")
                    }
                }
            }.start()
        }

        Thread { loadAndStart(modelDir) }.start()
        return START_STICKY
    }

    override fun onDestroy() {
        try { unregisterReceiver(screenReceiver) } catch (_: Exception) {}
        releaseAudioWakeLock()
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
        audioReleased = false
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
            audioReleased = true
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
            val audioFloat = FloatArray(CHUNK_SAMPLES) { pcm16[it].toFloat() / 32768f }

            val melOut = runModel(env, mel, audioFloat, longArrayOf(1, CHUNK_SAMPLES.toLong()))
            for (f in 0 until MEL_FRAMES_PER_CHUNK) {
                val frame = FloatArray(MEL_BINS) { melOut[f * MEL_BINS + it] }
                melBuffer.addLast(frame)
            }
            while (melBuffer.size > MEL_WINDOW) melBuffer.removeFirst()
            if (melBuffer.size < MEL_WINDOW) return

            val flatMel = FloatArray(MEL_WINDOW * MEL_BINS)
            melBuffer.forEachIndexed { i, frame -> frame.copyInto(flatMel, i * MEL_BINS) }
            val embOut = runModel(env, emb, flatMel, longArrayOf(1, MEL_WINDOW.toLong(), MEL_BINS.toLong(), 1L))
            embBuffer.addLast(embOut)
            while (embBuffer.size > EMB_WINDOW) embBuffer.removeFirst()
            if (embBuffer.size < EMB_WINDOW) return

            val flatEmb = FloatArray(EMB_WINDOW * EMB_SIZE)
            embBuffer.forEachIndexed { i, e -> e.copyInto(flatEmb, i * EMB_SIZE) }
            val prob = runModel(env, kws, flatEmb, longArrayOf(1, EMB_WINDOW.toLong(), EMB_SIZE.toLong()))

            val score = prob.firstOrNull() ?: return
            if (score > threshold) {
                consecutiveHighScores++
                Log.d(TAG, "High score: $score (consecutive=$consecutiveHighScores, need=2)")
                if (consecutiveHighScores >= 2) {
                    Log.i(TAG, "Wake word detected! score=$score threshold=$threshold")
                    consecutiveHighScores = 0
                    cooldownRemaining = COOLDOWN_CHUNKS
                    embBuffer.clear()

                    val pm = getSystemService(PowerManager::class.java)
                    if (pm.isInteractive) {
                        // Screen is on: JS handles the full flow
                        detectionCallback?.invoke()
                    } else {
                        // Screen is off: Kotlin handles chime + dictation, then wakes screen
                        handleScreenOffDetection()
                    }
                }
            } else {
                consecutiveHighScores = 0
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}")
        }
    }

    // ── Screen-off wake word path ────────────────────────────────────────────

    private fun handleScreenOffDetection() {
        Log.i(TAG, "Screen-off wake word — handing off to detection wake lock")
        // Release the indefinite audio-loop wake lock; detection lock covers the next 25s.
        releaseAudioWakeLock()
        val pm = getSystemService(PowerManager::class.java)
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "zynkbot:screen_off_wake")
        wakeLock?.acquire(25_000L)

        // Signal the ONNX audio loop to stop so Vosk can open the mic
        running = false

        Thread {
            // Wait for AudioRecord to release (set by audio loop thread)
            var waited = 0
            while (!audioReleased && waited < 2000) { Thread.sleep(50); waited += 50 }

            // Play chime via MediaPlayer (no WebView needed)
            try {
                val mp = MediaPlayer.create(this@WakeWordService, R.raw.wake_chime)
                if (mp != null) {
                    val latch = CountDownLatch(1)
                    mp.setOnCompletionListener { it.release(); latch.countDown() }
                    mp.start()
                    latch.await(2000, TimeUnit.MILLISECONDS)
                }
            } catch (e: Exception) {
                Log.w(TAG, "Chime playback failed: ${e.message}")
            }

            startKotlinVoskDictation()
        }.start()
    }

    private fun startKotlinVoskDictation() {
        val model = sharedVoskModel ?: run {
            Log.w(TAG, "No Vosk model available for screen-off dictation")
            releaseWakeLock()
            return
        }

        val accumulated = StringBuilder()
        val silenceHandler = Handler(Looper.getMainLooper())
        var speechService: org.vosk.android.SpeechService? = null

        val listener = object : org.vosk.android.RecognitionListener {
            override fun onPartialResult(h: String?) {
                val partial = try { org.json.JSONObject(h ?: "").optString("partial", "") } catch (_: Exception) { "" }
                if (partial.isNotBlank()) {
                    silenceHandler.removeCallbacksAndMessages(null)
                    // 1.5s silence after last speech → stop
                    silenceHandler.postDelayed({ speechService?.stop() }, 1500)
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
                Log.i(TAG, "Screen-off transcript: \"$transcript\"")
                deliverTranscriptToApp(transcript)
            }
            override fun onError(e: Exception?) {
                Log.e(TAG, "Screen-off Vosk error: ${e?.message}")
                silenceHandler.removeCallbacksAndMessages(null)
                releaseWakeLock()
            }
            override fun onTimeout() {
                Log.w(TAG, "Screen-off Vosk timeout — no speech detected")
                releaseWakeLock()
            }
        }

        Handler(Looper.getMainLooper()).post {
            try {
                val rec = org.vosk.Recognizer(model, 16000.0f)
                speechService = org.vosk.android.SpeechService(rec, 16000.0f)
                speechService!!.startListening(listener)
                Log.i(TAG, "Screen-off Vosk dictation started")
                // Safety timeout: stop after 10s regardless
                silenceHandler.postDelayed({ speechService?.stop() }, 10_000)
            } catch (e: Exception) {
                Log.e(TAG, "Screen-off Vosk start failed: ${e.message}")
                releaseWakeLock()
            }
        }
    }

    private fun deliverTranscriptToApp(transcript: String) {
        if (transcript.isBlank()) {
            Log.i(TAG, "Empty transcript — not waking screen")
            releaseWakeLock()
            return
        }
        Log.i(TAG, "Delivering transcript to app: \"$transcript\"")

        // Android 10+ blocks startActivity() from background components even in foreground
        // services. Use a full-screen-intent notification — the standard mechanism for
        // screen-off wake scenarios (alarm apps, incoming calls).
        val activityIntent = Intent(this, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT)
            putExtra("wake_word_transcript", transcript)
        }
        val pendingIntent = android.app.PendingIntent.getActivity(
            this, TRANSCRIPT_NOTIFICATION_ID, activityIntent,
            android.app.PendingIntent.FLAG_UPDATE_CURRENT or android.app.PendingIntent.FLAG_IMMUTABLE
        )
        val notification = NotificationCompat.Builder(this, TRANSCRIPT_CHANNEL_ID)
            .setContentTitle("Zynkbot")
            .setContentText("\"${transcript.take(60)}\"")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_CALL)
            .setFullScreenIntent(pendingIntent, true)
            .setAutoCancel(true)
            .build()
        getSystemService(NotificationManager::class.java)
            .notify(TRANSCRIPT_NOTIFICATION_ID, notification)

        releaseWakeLock()
    }

    private fun releaseWakeLock() {
        try { if (wakeLock?.isHeld == true) wakeLock?.release() } catch (_: Exception) {}
        wakeLock = null
    }

    // ── Shared ONNX inference ────────────────────────────────────────────────

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
        consecutiveHighScores = 0
        audioThread?.interrupt()
        audioThread = null
        melSession?.close(); melSession = null
        embSession?.close(); embSession = null
        kwsSession?.close(); kwsSession = null
        ortEnv?.close(); ortEnv = null
        melBuffer.clear()
        embBuffer.clear()
        releaseWakeLock()
        releaseAudioWakeLock()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(NotificationChannel(
                CHANNEL_ID, "Hey Zynk", NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Listens for the wake word in the background"
                setShowBadge(false)
            })
            // HIGH importance required for full-screen-intent to fire on the lock screen
            nm.createNotificationChannel(NotificationChannel(
                TRANSCRIPT_CHANNEL_ID, "Wake Word Response", NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Wakes the screen to deliver a Hey Zynk voice query"
                lockscreenVisibility = NotificationCompat.VISIBILITY_PUBLIC
                setShowBadge(false)
            })
        }
    }
}

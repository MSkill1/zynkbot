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
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.util.Locale
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
    @Volatile private var isForegrounded = false // guards against double startForeground on Android 14+
    private var audioThread: Thread? = null
    private var threshold = 0.5f
    private var lastModelDir: String? = null     // stored so background detection can restart the loop
    private var loadedModelDir: String? = null   // dir the resident ONNX sessions were loaded from; null = none loaded
    private var wakeLock: PowerManager.WakeLock? = null      // detection wake lock (25s, screen-off flow)
    @Volatile private var nativeTts: TextToSpeech? = null     // lazily created; speaks native replies
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

        // Only call startForeground once per service lifecycle. On Android 14+,
        // calling startForeground(MICROPHONE) from background context throws SecurityException
        // and kills the service mid-flow (e.g. during screen-off Vosk dictation).
        if (!isForegrounded) {
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
                isForegrounded = true
            } catch (e: SecurityException) {
                Log.w(TAG, "App not in foreground, cannot start FGS: ${e.message}")
                stopSelf()
                return START_NOT_STICKY
            }
        }

        // Stop existing audio loop and wait for it to exit before starting a new one.
        // ONNX Runtime sessions are not safe for concurrent inference — if the old thread
        // is still mid-inference when the new one starts, both corrupt each other's results.
        if (running || audioThread?.isAlive == true) {
            Log.i(TAG, "Restarting audio loop (threshold=${threshold})")
            running = false
            audioThread?.interrupt()
            audioThread?.join(500) // wait for the read loop to exit (one cycle is ~80ms)
            audioThread = null
            // Keep the loaded ONNX sessions. Reloading all three models on every
            // restart was the main source of wake-word churn and raced with the
            // screen-off Vosk mic handoff (intermittent empty transcripts). The old
            // audio thread has now exited (join above), so no inference is in flight
            // and the resident sessions are safe to reuse on the new thread.
            melBuffer.clear()
            embBuffer.clear()
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

        lastModelDir = modelDir
        Thread { loadAndStart(modelDir) }.start()
        return START_STICKY
    }

    override fun onDestroy() {
        isForegrounded = false
        try { unregisterReceiver(screenReceiver) } catch (_: Exception) {}
        releaseAudioWakeLock()
        stop()
        try { nativeTts?.shutdown() } catch (_: Exception) {}
        nativeTts = null
        super.onDestroy()
    }

    private fun loadAndStart(modelDir: String) {
        try {
            if (melSession == null || embSession == null || kwsSession == null || modelDir != loadedModelDir) {
                // (Re)load only when the models aren't already resident, or the
                // directory changed. The common restart path skips this entirely.
                melSession?.close(); embSession?.close(); kwsSession?.close()
                ortEnv = OrtEnvironment.getEnvironment()
                val env = ortEnv!!
                val dir = File(modelDir)
                melSession = env.createSession(File(dir, "melspectrogram.onnx").absolutePath)
                embSession = env.createSession(File(dir, "embedding_model.onnx").absolutePath)
                kwsSession = env.createSession(File(dir, "hey_zynk.onnx").absolutePath)
                loadedModelDir = modelDir
                Log.i(TAG, "ONNX models loaded. mel inputs: ${melSession!!.inputNames}, emb inputs: ${embSession!!.inputNames}, kws inputs: ${kwsSession!!.inputNames}")
            } else {
                Log.i(TAG, "Reusing already-loaded ONNX models")
            }

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

                    if (MainActivity.isInForeground) {
                        // App is visible: JS WebView is live, call directly
                        detectionCallback?.invoke()
                    } else {
                        // App is minimized or screen is off: evaluateJavascript() silently
                        // drops on a paused WebView. Use Kotlin-native path for both cases.
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

            // Play chime via MediaPlayer using USAGE_ASSISTANT so it respects
            // assistant/notification volume rather than media volume.
            try {
                val mp = MediaPlayer()
                mp.setAudioAttributes(
                    android.media.AudioAttributes.Builder()
                        .setUsage(android.media.AudioAttributes.USAGE_ASSISTANT)
                        .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build()
                )
                resources.openRawResourceFd(R.raw.wake_chime)?.let { afd ->
                    mp.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
                    afd.close()
                }
                mp.prepare()
                val latch = CountDownLatch(1)
                mp.setOnCompletionListener { it.release(); latch.countDown() }
                mp.start()
                latch.await(2000, TimeUnit.MILLISECONDS)
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
                answerNatively(transcript)
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

    // Answer a screen-off transcript entirely natively: no Activity, no WebView,
    // no notification. Falls back to deliverTranscriptToApp() (the WebView/
    // notification path) if anything in the native path fails or produces no
    // speakable reply, so a locked-screen query still gets answered somehow.
    private fun answerNatively(transcript: String) {
        if (transcript.isBlank()) {
            Log.i(TAG, "Empty transcript — not waking screen")
            releaseWakeLock()
            return
        }
        Log.i(TAG, "Answering natively: \"$transcript\"")

        // The 25s detection wake lock (acquired for chime+dictation) is too short
        // for a network round trip plus speech; extend it for this attempt.
        try { wakeLock?.acquire(90_000L) } catch (_: Exception) {}

        Thread {
            var spoke = false
            try {
                // "" backend picks whatever AI provider is actually configured
                // (API key or Ollama/custom endpoint) — there is no working
                // on-device model on Android. TODO once the app persists the
                // user's in-app model choice natively (today it lives only in
                // the WebView's localStorage) prefer that instead.
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
                                spoke = speakNatively(replyText)
                            }
                        }
                        override fun onError(message: String) {
                            Log.w(TAG, "Native reply failed: $message")
                        }
                    },
                )
            } catch (e: Throwable) {
                // ZynkCore not yet linked into this build, or any other native
                // failure — degrade to the known-working path rather than go silent.
                Log.w(TAG, "Native answer path unavailable: ${e.message}")
            }

            if (spoke) {
                releaseWakeLock()
                playClosingTone()
            } else {
                deliverTranscriptToApp(transcript)
            }
        }.start()
    }

    /** Speaks [text] via Android's TextToSpeech, blocking until it finishes. Returns
     *  false (without throwing) if TTS could not be initialized or start speaking. */
    private fun speakNatively(text: String): Boolean {
        val latch = CountDownLatch(1)
        var ok = false
        try {
            var tts = nativeTts
            if (tts == null) {
                val initLatch = CountDownLatch(1)
                var initOk = false
                tts = TextToSpeech(this) { status -> initOk = status == TextToSpeech.SUCCESS; initLatch.countDown() }
                initLatch.await(5000, TimeUnit.MILLISECONDS)
                if (!initOk) { Log.w(TAG, "Native TTS init failed"); return false }
                tts.language = Locale.US
                nativeTts = tts
            }
            tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {}
                override fun onDone(utteranceId: String?) { ok = true; latch.countDown() }
                @Deprecated("required override") override fun onError(utteranceId: String?) { latch.countDown() }
                override fun onError(utteranceId: String?, errorCode: Int) { latch.countDown() }
            })
            val id = "zynk-native-${System.currentTimeMillis()}"
            val result = tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, id)
            if (result != TextToSpeech.SUCCESS) { Log.w(TAG, "Native TTS speak() rejected"); return false }
            // Safety cap: don't hold the wake lock forever if the utterance callback never fires.
            latch.await(60_000, TimeUnit.MILLISECONDS)
        } catch (e: Exception) {
            Log.w(TAG, "Native TTS failed: ${e.message}")
        }
        return ok
    }

    private fun playClosingTone() {
        Thread {
            try {
                val mp = MediaPlayer()
                mp.setAudioAttributes(
                    android.media.AudioAttributes.Builder()
                        .setUsage(android.media.AudioAttributes.USAGE_ASSISTANT)
                        .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build()
                )
                resources.openRawResourceFd(R.raw.wake_chime_close)?.let { afd ->
                    mp.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
                    afd.close()
                }
                mp.prepare()
                val latch = CountDownLatch(1)
                mp.setOnCompletionListener { it.release(); latch.countDown() }
                mp.start()
                latch.await(2000, TimeUnit.MILLISECONDS)
            } catch (e: Exception) {
                Log.w(TAG, "Closing tone failed: ${e.message}")
            }
        }.start()
    }

    private fun deliverTranscriptToApp(transcript: String) {
        if (transcript.isBlank()) {
            Log.i(TAG, "Empty transcript — not waking screen")
            releaseWakeLock()
            return
        }
        Log.i(TAG, "Delivering transcript to app: \"$transcript\"")

        val activityIntent = Intent(this, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT)
            putExtra("wake_word_transcript", transcript)
        }

        val pm = getSystemService(PowerManager::class.java)
        val screenOn = pm.isInteractive

        // On standard Android, try startActivity() directly when the screen is on and the
        // app is minimized. Requires SYSTEM_ALERT_WINDOW ("Draw over other apps").
        // Skip on GrapheneOS — its kernel patches block background activity launches even
        // with that permission; the notification tap is the correct path there.
        if (screenOn && !isGrapheneOS()) {
            try {
                startActivity(activityIntent)
                Log.i(TAG, "Direct startActivity succeeded")
            } catch (e: Exception) {
                Log.w(TAG, "Direct startActivity failed: ${e.message}")
            }
        }

        // Always post the notification: auto-opens for screen-locked (full-screen-intent),
        // and acts as a tap-to-open fallback for GrapheneOS or when startActivity is blocked.
        // Use FLAG_CANCEL_CURRENT so each delivery gets a fresh PendingIntent with the
        // correct transcript. FLAG_UPDATE_CURRENT + FLAG_IMMUTABLE conflict: IMMUTABLE
        // prevents UPDATE_CURRENT from changing extras, so subsequent deliveries would
        // carry the first transcript forever.
        val pendingIntent = android.app.PendingIntent.getActivity(
            this, TRANSCRIPT_NOTIFICATION_ID, activityIntent,
            android.app.PendingIntent.FLAG_CANCEL_CURRENT or android.app.PendingIntent.FLAG_IMMUTABLE
        )
        val notification = NotificationCompat.Builder(this, TRANSCRIPT_CHANNEL_ID)
            .setContentTitle("Zynkbot")
            .setContentText("\"${transcript.take(60)}\"")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_CALL)
            .setContentIntent(pendingIntent)          // fires when user taps the notification
            .setFullScreenIntent(pendingIntent, !screenOn) // fires automatically when screen is off
            .setAutoCancel(true)
            .build()
        getSystemService(NotificationManager::class.java)
            .notify(TRANSCRIPT_NOTIFICATION_ID, notification)

        releaseWakeLock()

        // Play closing tone (faster/higher pitch = "done") to signal end of listening window.
        // We do NOT restart the ONNX loop here. The JS side (TTS-aware useEffect +
        // visibilitychange) restarts it correctly after TTS finishes. Restarting from
        // Kotlin races with WakeWordBridge.stop()/start() calls and causes the loop to
        // fire __wakeWordDetected during AI generation before JS can guard it.
        Thread {
            try {
                val mp = MediaPlayer()
                mp.setAudioAttributes(
                    android.media.AudioAttributes.Builder()
                        .setUsage(android.media.AudioAttributes.USAGE_ASSISTANT)
                        .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build()
                )
                resources.openRawResourceFd(R.raw.wake_chime_close)?.let { afd ->
                    mp.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
                    afd.close()
                }
                mp.prepare()
                val latch = CountDownLatch(1)
                mp.setOnCompletionListener { it.release(); latch.countDown() }
                mp.start()
                latch.await(2000, TimeUnit.MILLISECONDS)
            } catch (e: Exception) {
                Log.w(TAG, "Closing tone failed: ${e.message}")
            }
        }.start()
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
        audioThread?.join(300)
        audioThread = null
        melSession?.close(); melSession = null
        embSession?.close(); embSession = null
        kwsSession?.close(); kwsSession = null
        ortEnv?.close(); ortEnv = null
        loadedModelDir = null   // full stop tears the models down; next start reloads
        melBuffer.clear()
        embBuffer.clear()
        releaseWakeLock()
        releaseAudioWakeLock()
    }

    // GrapheneOS blocks background activity launches (startActivity from a foreground
    // service) even when SYSTEM_ALERT_WINDOW is granted. Detect it via a system property
    // it exposes; fall back to checking Build.DISPLAY if reflection fails.
    private fun isGrapheneOS(): Boolean = try {
        val sp = Class.forName("android.os.SystemProperties")
        val get = sp.getMethod("get", String::class.java, String::class.java)
        (get.invoke(null, "org.grapheneos.version", "") as String).isNotEmpty()
    } catch (_: Exception) {
        Build.DISPLAY.contains("graphene", ignoreCase = true)
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

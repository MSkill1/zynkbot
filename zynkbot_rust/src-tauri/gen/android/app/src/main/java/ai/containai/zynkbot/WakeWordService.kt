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
import android.media.AudioManager
import android.media.AudioPlaybackConfiguration
import android.media.AudioRecordingConfiguration
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
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.nio.FloatBuffer
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class WakeWordService : Service() {

    companion object {
        const val CHANNEL_ID = "wake_word_channel"
        const val TRANSCRIPT_CHANNEL_ID = "wake_word_transcript_channel"
        const val NOTIFICATION_ID = 1003
        const val TRANSCRIPT_NOTIFICATION_ID = 1005
        const val TAG = "WakeWordService"
        /** Longest the listener stays paused for another recording before resuming (2026-09-20). */
        private const val PAUSE_WATCHDOG_MS = 120_000L

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
        // ── 2026-10-19: the Siri-shaped fixes (Pixel log, TV on, 12:48–12:49) ──
        const val NO_SPEECH_MS = 6000L      // chime, then nothing said → close (was: the 30 s cap; same value as ZynkAssistantSession)
        const val WATCHDOG_MS = 60_000L     // once a minute: listener off with no reason → restart it
        const val PLAYBACK_TAIL_MS = 1000L  // ignore detections while the phone plays audio and for this long after
        // Silero VAD v5 (assets/wake-word-models/silero_vad.onnx): 16 kHz, 512-sample frames
        // with 64 samples of context, state 2×1×128. A detection counts only if some frame in
        // the last ~2 s had a voice; with no voice for 2 s the wake models are skipped.
        const val VAD_FRAME = 512
        const val VAD_CONTEXT = 64
        const val VAD_STATE = 2 * 128
        const val VAD_WINDOW_FRAMES = 62    // ~2 s of 32 ms frames
        const val VAD_SPEECH = 0.5f         // Silero's own default; on the 2026-10-19 clip set every real wake had a frame ≥ 0.5 in its last 2 s
        const val VAD_SKIP_AFTER_FRAMES = 62
        const val TRIGGER_CLIP_CHUNKS = 40  // ~3.2 s: covers the models' full context (~2.5 s)
        const val MAX_LISTEN_MS = 30_000L   // hard cap on one dictation, ongoing speech cannot extend it (12 s cut off dictated paragraphs; matches ZynkAssistantSession since 2026-09-12)
        const val MAX_QUERY_WORDS = 60      // longer than any question; TV dialogue is not a query
        const val TRIGGER_CLIPS_KEPT = 20   // newest clips kept under files/zynkbot/wake_triggers (phones with an enforcing verifier)
        const val TRIGGER_CLIPS_KEPT_UNVERIFIED = 60   // no verifier yet: keep enough to train one (Matt's took 33 real clips)
        const val CLIPS_NEEDED_FOR_VERIFIER = 30       // real clips before "Send my wake-word clips" lights up
        // Detections on audio quieter than this are ignored. Set from the Pixel's log of
        // 2026-09-08: a real "Hey Zynk" from ~20 ft measured -31.5 dBFS; ten of the
        // twenty-two false firings that morning sat between -40 and -47. Stricter still
        // while backing off.
        // Relaxed again 2026-09-08 pm: from across the room the owner's own "Hey Zynk"
        // sits at -39 to -44 dBFS, the same band as the TV, and the -42/-38 gates ate
        // it. Loudness cannot separate the two at room distance; the verifier can.
        const val SILENCE_GATE_DB = -46.0
        const val STRICT_GATE_DB = -46.0
        // Battery (2026-09-10): the three models ran on every 80 ms chunk all night while
        // the screen-off wake lock kept the CPU up, and a Pixel went from 50% to dead.
        // Below COMPUTE_GATE_DB (well under the detection gate) for QUIET_SKIP_AFTER
        // chunks in a row, nothing is computed until a chunk is loud again. The first
        // loud chunk runs immediately, so speech onset is never skipped; the stale
        // embeddings it joins are embeddings of the same silence.
        const val COMPUTE_GATE_DB = -52.0
        const val QUIET_SKIP_AFTER = 25          // 2 s of quiet before skipping starts
        const val QUIET_STATS_EVERY_CHUNKS = 3750 // ~5 min: one log line of skipped/total
        // Enforcement is decided per user by WakeVerifier.enforcesOn(): the shipped
        // verifier is trained on one owner's voice and lists that owner's user id;
        // for anyone else it only logs and collects clips (2026-09-09).
        const val STRICT_HITS = 4           // consecutive high scores needed while backing off
        const val STRICT_SCORE = 0.90f      // per-chunk score needed while backing off
        const val MISS_WINDOW_MS = 5 * 60_000L
        const val MISSES_TO_BACK_OFF = 3
        /** Strict mode (back-off after misses) is switched off as of 2026-09-20. It was not a
         *  TV detector: three ordinary misses in five minutes (a dismissed session, a
         *  one-word transcript, a video playing nearby) locked the owner out for ten
         *  minutes, because four consecutive slices at 0.90 is above the detector's own
         *  noise on a real "Hey Zynk". The code stays, gated here, in case a gentler
         *  version is wanted later; the plan is the owner's trained verifier instead. */
        const val STRICT_MODE_ENABLED = false
        const val BACK_OFF_MS = 10 * 60_000L

        /** Called by the session / answerer when a trigger ends: `useful` = a real
         *  question was answered or a command ran; false = nothing heard or NO_QUERY. */
        @JvmStatic fun reportOutcome(useful: Boolean) { instance?.noteOutcome(useful) }

        /** Label the newest clip without touching the back-off: an answered question is
         *  a real "Hey Zynk" for training purposes, but only clock commands count as
         *  proof against the TV for the strict-mode logic (2026-09-13). */
        @JvmStatic fun labelClip(real: Boolean) { instance?.labelLastClip(real) }

        /** Counts behind the Voice-settings "Send my wake-word clips" button. */
        @JvmStatic fun clipStats(context: Context): org.json.JSONObject {
            val dir = File(context.filesDir, "zynkbot/wake_triggers")
            val wavs = dir.listFiles { f -> f.name.endsWith(".wav") } ?: emptyArray()
            var real = 0; var falseCount = 0
            for (w in wavs) {
                val stem = w.name.removeSuffix(".wav")
                if (File(dir, "$stem.real").exists()) real++ else if (File(dir, "$stem.false").exists()) falseCount++
            }
            return org.json.JSONObject().put("total", wavs.size).put("real", real).put("false", falseCount).put("needed", CLIPS_NEEDED_FOR_VERIFIER)
        }

        /** Every clip with its labels and score files, zipped into the cache dir for the
         *  share sheet. Nothing is sent by this code; the user picks where it goes. */
        @JvmStatic fun zipClips(context: Context): File? {
            val dir = File(context.filesDir, "zynkbot/wake_triggers")
            val files = dir.listFiles()?.filter { it.isFile } ?: return null
            if (files.isEmpty()) return null
            val out = File(context.cacheDir, "zynkbot-wake-clips-${SimpleDateFormat("yyyyMMdd-HHmm", Locale.US).format(Date())}.zip")
            java.util.zip.ZipOutputStream(FileOutputStream(out)).use { zip ->
                for (f in files.sortedBy { it.name }) {
                    zip.putNextEntry(java.util.zip.ZipEntry(f.name))
                    f.inputStream().use { it.copyTo(zip) }
                    zip.closeEntry()
                }
            }
            return out
        }
        /** True while backing off after repeated fruitless triggers. */
        @JvmStatic fun isStrict(): Boolean = (instance?.strictUntil ?: 0L) > System.currentTimeMillis()

        // Vosk model shared from VoskBridge so screen-off dictation doesn't reload it.
        @Volatile var sharedVoskModel: org.vosk.Model? = null

        // The running service: ZynkAssistantSession, VoskBridge and NativeVoiceAnswerer
        // reach its stream through dictate() / beginTurn() / endTurn().
        @Volatile var instance: WakeWordService? = null

        // ── 2026-10-20 rebuild: one owner, one stream, one state ──────────────
        const val TURN_MAX_MS = 180_000L       // a turn (chime → reply spoken) longer than this is closed by the watchdog
        const val DICTATION_MAX_MS = 60_000L   // a dictation that outlives its own 30 s cap by this much is aborted
        const val PAUSE_MAX_MS = PAUSE_WATCHDOG_MS   // paused for another recording: resume regardless after this
        const val CHIME_SETTLE_MS = 350L       // dictation after a chime drops this much: the chime's tail reached Whisper as speech (15:07, "Beep!")

        /** MainActivity sets this to push the state name into the page
         *  (window.__wakeWordState); null when the page is not around. */
        @Volatile var onStateChanged: ((String) -> Unit)? = null
    }

    private var ortEnv: OrtEnvironment? = null
    private var melSession: OrtSession? = null
    private var embSession: OrtSession? = null
    private var kwsSession: OrtSession? = null

    // ── speech gate (Silero VAD, 2026-10-19) ──────────────────────────────────
    // Optional: a missing or unloadable model logs once and the detector runs ungated.
    private var vadSession: OrtSession? = null
    private var vadState = FloatArray(VAD_STATE)
    private val vadContext = FloatArray(VAD_CONTEXT)
    private val vadCarry = FloatArray(VAD_FRAME + CHUNK_SAMPLES)
    private var vadCarryLen = 0
    private val vadRecent = ArrayDeque<Float>()     // last ~2 s of speech probabilities
    private var vadFramesSinceSpeech = 0
    private var vadSkipped = 0L
    private fun vadRecentMax(): Float { var m = 0f; for (p in vadRecent) if (p > m) m = p; return m }

    /** Feed one 80 ms chunk to the VAD in 512-sample frames (leftover carried over). */
    private fun runVad(env: OrtEnvironment, vad: OrtSession, audio: FloatArray) {
        System.arraycopy(audio, 0, vadCarry, vadCarryLen, audio.size)
        vadCarryLen += audio.size
        var off = 0
        val input = FloatArray(VAD_CONTEXT + VAD_FRAME)
        while (vadCarryLen - off >= VAD_FRAME) {
            System.arraycopy(vadContext, 0, input, 0, VAD_CONTEXT)
            System.arraycopy(vadCarry, off, input, VAD_CONTEXT, VAD_FRAME)
            val inT = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, (VAD_CONTEXT + VAD_FRAME).toLong()))
            val stT = OnnxTensor.createTensor(env, FloatBuffer.wrap(vadState), longArrayOf(2, 1, 128))
            val srT = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(SAMPLE_RATE.toLong())), longArrayOf())
            val results = vad.run(mapOf("input" to inT, "state" to stT, "sr" to srT))
            val p = (results[0] as OnnxTensor).floatBuffer.get(0)
            val st = (results[1] as OnnxTensor).floatBuffer
            val newState = FloatArray(VAD_STATE); st.get(newState); vadState = newState
            results.close(); inT.close(); stT.close(); srT.close()
            System.arraycopy(vadCarry, off + VAD_FRAME - VAD_CONTEXT, vadContext, 0, VAD_CONTEXT)
            vadRecent.addLast(p)
            while (vadRecent.size > VAD_WINDOW_FRAMES) vadRecent.removeFirst()
            if (p >= VAD_SPEECH) vadFramesSinceSpeech = 0 else vadFramesSinceSpeech++
            off += VAD_FRAME
        }
        if (off > 0) { System.arraycopy(vadCarry, off, vadCarry, 0, vadCarryLen - off); vadCarryLen -= off }
    }

    private fun resetVad() {
        vadState = FloatArray(VAD_STATE); java.util.Arrays.fill(vadContext, 0f)
        vadCarryLen = 0; vadRecent.clear(); vadFramesSinceSpeech = 0
    }

    // ── playback suppression (2026-10-19) ─────────────────────────────────────
    // The phone knows when it is playing sound (a notification tone, an app's "boing",
    // a video, our own chime or reply). A detection during playback or in the second
    // after it is ignored: the model heard the speaker, not a person. Trade-off, stated
    // plainly: "Hey Zynk" does not work while this phone is playing music. The echo
    // canceller MicStream attaches is the eventual way out of that; whether
    // this phone's actually removes the speaker signal is logged, not assumed.
    private var playbackCallback: AudioManager.AudioPlaybackCallback? = null
    @Volatile private var playbackActive = false
    @Volatile private var lastPlaybackEndMs = 0L
    private fun playbackRecently(): Boolean = playbackActive || System.currentTimeMillis() - lastPlaybackEndMs < PLAYBACK_TAIL_MS

    private fun registerPlaybackWatch() {
        if (playbackCallback != null) return
        try {
            val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
            val cb = object : AudioManager.AudioPlaybackCallback() {
                override fun onPlaybackConfigChanged(configs: MutableList<AudioPlaybackConfiguration>) {
                    val active = configs.isNotEmpty()
                    if (playbackActive && !active) lastPlaybackEndMs = System.currentTimeMillis()
                    playbackActive = active
                }
            }
            am.registerAudioPlaybackCallback(cb, Handler(Looper.getMainLooper()))
            playbackCallback = cb
            playbackActive = am.activePlaybackConfigurations.isNotEmpty()
        } catch (e: Exception) { Log.w(TAG, "Playback watch unavailable: ${e.message}") }
    }

    private fun unregisterPlaybackWatch() {
        val cb = playbackCallback ?: return
        playbackCallback = null
        try { (getSystemService(Context.AUDIO_SERVICE) as AudioManager).unregisterAudioPlaybackCallback(cb) } catch (_: Exception) {}
    }

    // ── watchdog ──────────────────────────────────────────────────────────────
    // Once a minute: the net under the state machine. A state whose stream should be
    // open but is not is reopened; a pause for a recording that no longer exists is
    // lifted; a turn or a dictation that never ended is closed. In a correct run this
    // never fires; when it does, the log line says which path failed to send its event.
    private val watchdogHandler = Handler(Looper.getMainLooper())
    private val watchdog = object : Runnable {
        override fun run() {
            try { watchdogTick() } catch (e: Exception) { Log.w(TAG, "Watchdog: ${e.message}") }
            watchdogHandler.postDelayed(this, WATCHDOG_MS)
        }
    }

    private fun watchdogTick() {
        val now = System.currentTimeMillis()
        when (mic.state) {
            MicState.State.LISTENING -> if (!stream.isOpen) {
                Log.w(TAG, "Watchdog: listening but the stream is closed — reopening")
                events.execute { apply(MicState.Event.StreamLost) }
            }
            MicState.State.PAUSED -> {
                val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
                val others = am.activeRecordingConfigurations.filter { it.clientAudioSessionId != stream.sessionId }
                if (others.isEmpty()) {
                    Log.w(TAG, "Watchdog: paused for a recording that no longer exists — resuming")
                    events.execute { apply(MicState.Event.ForeignEnded) }
                } else if (now - pausedAt > PAUSE_MAX_MS) {
                    // A page dictation stream that broke and was never released kept this
                    // paused for 13 minutes (Pixel, 2026-09-19). Nothing legitimately records
                    // for that long, and Android lets one app capture twice.
                    Log.w(TAG, "Watchdog: still paused for another recording after ${PAUSE_MAX_MS / 1000} s — resuming anyway")
                    events.execute { apply(MicState.Event.ForeignEnded) }
                }
            }
            MicState.State.TURN -> if (now - turnStartedAt > TURN_MAX_MS) {
                Log.w(TAG, "Watchdog: a turn has been open for ${(now - turnStartedAt) / 1000} s — closing it")
                events.execute { apply(MicState.Event.Expired) }
            }
            MicState.State.DICTATING -> if (now - (dictation?.startedAt ?: now) > DICTATION_MAX_MS) {
                Log.w(TAG, "Watchdog: a dictation has run for ${DICTATION_MAX_MS / 1000} s — aborting it")
                events.execute { apply(MicState.Event.Expired) }
            }
            else -> {}
        }
    }

    private val melBuffer = ArrayDeque<FloatArray>()
    private val embBuffer = ArrayDeque<FloatArray>()
    private var cooldownRemaining = 0
    private var consecutiveHighScores = 0

    // ── false-trigger mitigations (2026-09-08) ─────────────────────────────
    // 21 firings in 32 minutes with the TV on, scores 0.87–1.0, verified identical on
    // desktop replay: the model itself is the problem and only a retrained verifier
    // fixes detection. Until then, three cheap defences against the annoyance:
    //  1. silence gate — six of those firings were on near-silent audio;
    //  2. back-off — after three fruitless sessions in five minutes, demand a much
    //     stronger detection for ten minutes;
    //  3. faster close on no speech (in ZynkAssistantSession).
    private var verifier: WakeVerifier? = null
    private val recentMisses = ArrayDeque<Long>()      // wall-clock ms of empty / NO_QUERY sessions
    @Volatile private var strictUntil = 0L             // while now < strictUntil: 4 hits, score ≥ 0.9

    /** Stem of the newest trigger clip, so its outcome can be written next to it. */
    @Volatile private var lastClipStem: String? = null

    /** Records how the newest trigger ended, next to its clip: `.real` or `.false`.
     *  Labelled clips are the training set for the personal verifier (no upload;
     *  the files stay under files/zynkbot/wake_triggers). */
    private fun labelLastClip(real: Boolean) {
        val stem = lastClipStem ?: return
        try {
            val dir = File(filesDir, "zynkbot/wake_triggers")
            // A trigger reports its outcome more than once as it moves through the session
            // (fruitless first, then answered), so the last report has to win. Without this
            // 27 of 87 clips carried both labels and were useless for training (KI-076).
            File(dir, "$stem.${if (real) "false" else "real"}").delete()
            File(dir, "$stem.${if (real) "real" else "false"}").writeText(if (real) "real\n" else "false\n")
        } catch (e: Exception) { Log.w(TAG, "Could not label clip: ${e.message}") }
    }

    private fun noteOutcome(useful: Boolean) {
        labelLastClip(useful)
        val now = System.currentTimeMillis()
        synchronized(recentMisses) {
            if (useful) {
                recentMisses.clear()
                if (strictUntil > now) { strictUntil = 0L; Log.i(TAG, "Real question answered — leaving strict mode") }
                trainVerifierInBackground()   // one more real clip: retrain when enough have accumulated
                return
            }
            recentMisses.addLast(now)
            while (recentMisses.isNotEmpty() && now - recentMisses.first() > MISS_WINDOW_MS) recentMisses.removeFirst()
            if (STRICT_MODE_ENABLED && recentMisses.size >= MISSES_TO_BACK_OFF && strictUntil < now) {
                strictUntil = now + BACK_OFF_MS
                Log.i(TAG, "${recentMisses.size} fruitless triggers in 5 min — strict mode for 10 min (need $STRICT_HITS hits ≥ $STRICT_SCORE)")
            }
        }
    }

    private var quietRun = 0
    private var quietSkipped = 0L
    private var quietTotal = 0L

    /** RMS of one chunk in dBFS (cheap: 1280 multiplies). */
    private fun chunkLevelDb(pcm16: ShortArray): Double {
        var sum = 0.0
        for (v in pcm16) { val f = v / 32768.0; sum += f * f }
        val rms = Math.sqrt(sum / pcm16.size.coerceAtLeast(1))
        return 20.0 * Math.log10(Math.max(rms, 1e-9))
    }

    /** RMS of the last ~3 s in dBFS; -47 dB is far below speech at any distance. */
    private fun recentLevelDb(): Double {
        var sum = 0.0; var n = 0L
        for (chunk in recentChunks) { for (v in chunk) { val f = v / 32768.0; sum += f * f; n++ } }
        if (n == 0L) return -100.0
        val rms = Math.sqrt(sum / n)
        return 20.0 * Math.log10(Math.max(rms, 1e-9))
    }

    // ── microphone ownership (2026-10-20 rebuild) ─────────────────────────────
    // One state (MicState), one stream (MicStream), one entry point (apply). Every
    // start, stop, pause and re-arm that used to be its own path is an event into
    // apply(); the state machine says what to do, and this service does it. See
    // ~/Documents/Zynk_related/claude_notes/wake_word_audit_2026-10-20.md for the
    // fourteen paths and fifteen flags this replaced.
    private val mic = MicState { line -> Log.i(TAG, "State: $line") }
    private val stream = MicStream(TAG)
    private val detectorSink = MicStream.Sink { pcm, _ -> processChunk(pcm) }
    @Volatile private var dictation: Dictation? = null
    /** Events from Android callbacks and other threads, applied one at a time in order. */
    private val events = Executors.newSingleThreadExecutor { r -> Thread(r, "WakeWordEvents") }
    @Volatile private var turnStartedAt = 0L
    @Volatile private var chimeEndedAt = 0L
    @Volatile private var pausedAt = 0L
    @Volatile private var isForegrounded = false // guards against double startForeground on Android 14+
    private var threshold = 0.5f
    private var lastModelDir: String? = null     // the page's model directory; reloaded on every page-on
    private var loadedModelDir: String? = null   // dir the resident ONNX sessions were loaded from; null = none loaded
    // A load still in flight when the page turns Hey Zynk off sees the generation move
    // on and discards what it loaded (KI-097).
    private val modelLock = Any()
    private var loadGeneration = 0
    private var wakeLock: PowerManager.WakeLock? = null      // detection wake lock (25s, screen-off flow)
    private var audioWakeLock: PowerManager.WakeLock? = null // CPU wake lock for the stream while the screen is off

    /** The current state, for the page and for callers deciding whether to use the service's stream. */
    fun state(): MicState.State = mic.state
    val isOn: Boolean get() = mic.state != MicState.State.OFF

    // Other apps' recordings. Zynkbot must never take the microphone away from an app
    // the user is talking to (OnePlus 2026-09-04 14:13: the Claude app's voice mode
    // started, its start tone fired the wake word, and our session grabbed the mic).
    // Android reports every active recording; any that is not our one stream pauses
    // us, and we resume when it ends. Our own dictation runs on the same stream, so
    // it never looks foreign here (it used to, which routed every hand-off of our own
    // through this watch).
    private var recordingCallback: AudioManager.AudioRecordingCallback? = null

    // The last ~3 s of microphone audio and the model's score per chunk, so each firing
    // can be saved, listened to, and replayed offline against the exact same numbers.
    private val recentChunks = ArrayDeque<ShortArray>()
    private val recentScores = ArrayDeque<Float>()

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
        instance = this
        createNotificationChannel()
        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_OFF)
            addAction(Intent.ACTION_SCREEN_ON)
        }
        registerReceiver(screenReceiver, filter)
        registerRecordingWatch()
        registerPlaybackWatch()
        watchdogHandler.postDelayed(watchdog, WATCHDOG_MS)
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
            } catch (e: Exception) {
                // SecurityException on Android 14 (microphone type from the
                // background); ForegroundServiceStartNotAllowedException — an
                // IllegalStateException, not a SecurityException — on Android 12+
                // when the app is in the background. The sync service crashed the
                // app on exactly that one (2026-09-14); catch everything here.
                Log.w(TAG, "Cannot start as a foreground service now: ${e.javaClass.simpleName}: ${e.message}")
                stopSelf()
                return START_NOT_STICKY
            }
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

        // The page sends several starts per toggle (its effects, plus one on every lock
        // and unlock). The state machine takes the first and ignores the rest.
        lastModelDir = modelDir
        events.execute { apply(MicState.Event.PageOn) }
        return START_STICKY
    }

    override fun onDestroy() {
        isForegrounded = false
        try { unregisterReceiver(screenReceiver) } catch (_: Exception) {}
        unregisterRecordingWatch()
        unregisterPlaybackWatch()
        watchdogHandler.removeCallbacks(watchdog)
        // Synchronously: the process may be going away with us.
        apply(MicState.Event.PageOff)
        releaseWakeLock()
        releaseAudioWakeLock()
        NativeVoiceAnswerer.shutdown()
        events.shutdown()
        instance = null
        super.onDestroy()
    }

    // ── the single entry point ────────────────────────────────────────────────

    /** Apply one event and carry out what the state machine says. True if it was
     *  accepted. Synchronized: actions never interleave. May block up to 500 ms while
     *  closing the stream; never call it with a lock the audio thread may need. */
    @Synchronized
    private fun apply(event: MicState.Event): Boolean {
        val actions = mic.on(event) ?: return false
        onStateChanged?.let { cb -> try { cb(mic.state.name.lowercase()) } catch (_: Exception) {} }
        when (event) {
            is MicState.Event.Wake, is MicState.Event.TurnBegun, is MicState.Event.DictationEnded -> turnStartedAt = System.currentTimeMillis()
            is MicState.Event.ForeignStarted -> pausedAt = System.currentTimeMillis()
            else -> {}
        }
        for (a in actions) when (a) {
            MicState.Action.LOAD_MODELS -> {
                val gen = synchronized(modelLock) { ++loadGeneration }
                val dir = lastModelDir ?: continue
                Thread { loadModels(dir, gen) }.start()
            }
            MicState.Action.UNLOAD_MODELS -> unloadModels()
            MicState.Action.OPEN_STREAM -> openStream()
            MicState.Action.CLOSE_STREAM -> stream.close()
            MicState.Action.SINK_DETECTOR -> { resetDetector(); stream.sink = detectorSink }
            MicState.Action.SINK_DICTATION -> stream.sink = dictation
            MicState.Action.SINK_NONE -> stream.sink = null
            MicState.Action.ABORT_DICTATION -> { val d = dictation; dictation = null; d?.cancel() }
        }
        return true
    }

    private fun openStream() {
        stream.onLost = { why ->
            Log.e(TAG, "Microphone stream lost: $why")
            events.execute { apply(MicState.Event.StreamLost) }
        }
        if (!stream.open()) {
            Log.e(TAG, "Microphone could not be opened — the watchdog retries in a minute")
            return
        }
        val pm = getSystemService(PowerManager::class.java)
        if (!pm.isInteractive) acquireAudioWakeLock()
    }

    private fun resetDetector() {
        melBuffer.clear(); embBuffer.clear()
        consecutiveHighScores = 0
        cooldownRemaining = COOLDOWN_CHUNKS   // don't re-trigger on the tail of a reply
        resetVad()
    }

    // ── public API: the ways into a turn ──────────────────────────────────────

    /**
     * Feed the microphone to [d] until it finishes. [ownsTurn]: the caller runs a
     * whole turn (chime, answer, speech) and calls [endTurn] when it is over; false
     * (the app's mic button): the listener comes back by itself when the dictation
     * ends. False if the service is off or paused in a way that refuses it — the
     * caller then opens its own stream (see ZynkAssistantSession, VoskBridge).
     */
    fun dictate(d: Dictation, ownsTurn: Boolean): Boolean {
        synchronized(this) {
            dictation?.cancel()
            dictation = d
            val sinceChime = System.currentTimeMillis() - chimeEndedAt
            if (sinceChime < CHIME_SETTLE_MS) d.skipSamples = (CHIME_SETTLE_MS - sinceChime) * MicStream.SAMPLE_RATE / 1000
            d.onDone = {
                // On the finishing thread, before the listener hears the result, so the
                // owner's endTurn() finds the state it expects.
                if (dictation === d) { dictation = null; apply(MicState.Event.DictationEnded) }
            }
            if (!apply(MicState.Event.DictationStarted(ownsTurn))) { dictation = null; d.onDone = null; return false }
            return true
        }
    }

    /** A turn that starts without a dictation (a spoken confirmation, a reply to a
     *  page question). True if this call opened it; the caller then calls [endTurn]. */
    fun beginTurn(): Boolean = apply(MicState.Event.TurnBegun)

    /** The turn is over: back to listening (or paused, if another app records). */
    fun endTurn() { apply(MicState.Event.TurnEnded) }

    // ── models ────────────────────────────────────────────────────────────────

    private fun loadModels(modelDir: String, generation: Int) {
        try {
            // Sessions are built into locals first and installed under the lock at the
            // end, so a page-off that arrives meanwhile cannot pull them out from under
            // this thread (KI-097).
            val env = OrtEnvironment.getEnvironment()
            val dir = File(modelDir)
            val mel = env.createSession(File(dir, "melspectrogram.onnx").absolutePath)
            val emb = env.createSession(File(dir, "embedding_model.onnx").absolutePath)
            val kws = env.createSession(File(dir, "hey_zynk.onnx").absolutePath)
            var vad: OrtSession? = null
            val vadFile = File(dir, "silero_vad.onnx")
            if (vadFile.isFile) {
                try {
                    val v = env.createSession(vadFile.absolutePath)
                    vad = v
                    Log.i(TAG, "Speech gate loaded (Silero VAD, inputs ${v.inputNames})")
                } catch (e: Exception) { Log.w(TAG, "Speech gate unavailable, running without it: ${e.message}") }
            } else Log.w(TAG, "Speech gate unavailable: silero_vad.onnx not in $modelDir")
            val installed = synchronized(modelLock) {
                if (generation != loadGeneration) false else {
                    ortEnv = env
                    melSession = mel; embSession = emb; kwsSession = kws; vadSession = vad
                    loadedModelDir = modelDir
                    true
                }
            }
            if (!installed) {
                mel.close(); emb.close(); kws.close(); vad?.close()
                Log.i(TAG, "Model load abandoned: the service was stopped while the models were loading")
                return
            }
            Log.i(TAG, "ONNX models loaded. mel inputs: ${mel.inputNames}, emb inputs: ${emb.inputNames}, kws inputs: ${kws.inputNames}")
            events.execute { apply(MicState.Event.ModelsLoaded) }
            trainVerifierInBackground()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ONNX models: ${e.javaClass.simpleName}: ${e.message}")
            events.execute { apply(MicState.Event.ModelsFailed) }
            stopSelf()
        }
    }

    private fun unloadModels() {
        synchronized(modelLock) {
            loadGeneration++   // any load still in flight will see this and discard its result
            melSession?.close(); melSession = null
            embSession?.close(); embSession = null
            kwsSession?.close(); kwsSession = null
            vadSession?.close(); vadSession = null
            ortEnv = null   // OrtEnvironment.getEnvironment() is a process-wide singleton; not closed here
            loadedModelDir = null
        }
        melBuffer.clear()
        embBuffer.clear()
        consecutiveHighScores = 0
    }

    /** Siri-style ongoing learning: retrain the personal verifier from the labelled
     *  clips when enough new ones have arrived (WakeVerifierTrainer decides), then
     *  pick the new profile up. Runs at service start and after every answered turn. */
    private fun trainVerifierInBackground() {
        Thread {
            try {
                if (WakeVerifierTrainer.trainIfDue(this, loadedModelDir)) {
                    verifier = WakeVerifier.load(this)
                }
            } catch (e: Exception) { Log.w(TAG, "Verifier training: ${e.message}") }
        }.start()
    }

    private fun processChunk(pcm16: ShortArray) {
        val env = ortEnv ?: return
        val mel = melSession ?: return
        val emb = embSession ?: return
        val kws = kwsSession ?: return

        // Keep the trailing ~2 s regardless of cooldown, so a clip always ends with
        // the chunk that fired.
        recentChunks.addLast(pcm16.copyOf())
        while (recentChunks.size > TRIGGER_CLIP_CHUNKS) recentChunks.removeFirst()

        if (cooldownRemaining > 0) { cooldownRemaining--; return }

        // Quiet room: count, and skip the models once the quiet has lasted 2 s.
        quietTotal++
        if (chunkLevelDb(pcm16) < COMPUTE_GATE_DB) quietRun++ else quietRun = 0
        if (quietTotal % QUIET_STATS_EVERY_CHUNKS == 0L) {
            Log.i(TAG, "Quiet-skip: %d of %d chunks skipped in the last ~5 min (%.0f%%); no-voice skip: %d".format(
                quietSkipped, QUIET_STATS_EVERY_CHUNKS, 100.0 * quietSkipped / QUIET_STATS_EVERY_CHUNKS, vadSkipped))
            quietSkipped = 0; vadSkipped = 0
        }
        if (quietRun > QUIET_SKIP_AFTER) { quietSkipped++; return }

        try {
            val audioFloat = FloatArray(CHUNK_SAMPLES) { pcm16[it].toFloat() / 32768f }

            // Speech gate, first stage: sound but no human voice for 2 s (a fan, a hum,
            // music without singing) — the wake models are not run at all.
            vadSession?.let { vad ->
                try { runVad(env, vad, audioFloat) } catch (e: Exception) { Log.w(TAG, "Speech gate error, disabling it: ${e.message}"); vadSession = null }
                if (vadSession != null && vadFramesSinceSpeech > VAD_SKIP_AFTER_FRAMES) { vadSkipped++; return }
            }

            val melOut = runModel(env, mel, audioFloat, longArrayOf(1, CHUNK_SAMPLES.toLong()))
            for (f in 0 until MEL_FRAMES_PER_CHUNK) {
                // NOTE on features (2026-09-05): openWakeWord's reference pipeline applies
                // mel/10 + 2 here. Our classifier (hey_zynk.onnx) was trained WITHOUT it
                // (wake-word-training/train_hey_zynk.py), and enabling it on the phone
                // made the classifier saturate on any loud nearby speech (three false
                // fires in twenty seconds of ordinary talking, Pixel, build23). So the
                // phone matches the training features, un-normalized, until the classifier
                // is retrained on the reference features. Do not add the transform here
                // without retraining.
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
            recentScores.addLast(score)
            while (recentScores.size > TRIGGER_CLIP_CHUNKS) recentScores.removeFirst()
            val strict = System.currentTimeMillis() < strictUntil
            val needHits = if (strict) STRICT_HITS else 2
            val needScore = if (strict) STRICT_SCORE else threshold
            if (score > needScore) {
                consecutiveHighScores++
                Log.d(TAG, "High score: $score (consecutive=$consecutiveHighScores, need=$needHits${if (strict) ", strict" else ""})")
                if (consecutiveHighScores >= needHits) {
                    val level = recentLevelDb()
                    // The −46 dBFS quiet gate (2026-09-08, against firings on near-silent audio)
                    // threw away two real wakes at 0.99 on 2026-10-19 (−51.6 and −56.5 dBFS: a
                    // normal voice from across a quiet room). With the speech gate loaded, "was
                    // there a voice" is its job, and the level only has to rule out true silence.
                    val quietGate = if (strict) STRICT_GATE_DB else if (vadSession != null) -65.0 else SILENCE_GATE_DB
                    if (level < quietGate) {
                        Log.i(TAG, "Detection ignored: audio too quiet (%.1f dBFS, score=%.3f)".format(level, score))
                        consecutiveHighScores = 0
                        cooldownRemaining = COOLDOWN_CHUNKS / 2
                        embBuffer.clear()
                        return
                    }
                    val pooled = WakeVerifier.pooledFeatures(flatEmb, EMB_WINDOW, EMB_SIZE)
                    // Playback gate: the phone itself was making sound (a notification tone,
                    // an app's "boing", a video) in the last second — the model heard the
                    // speaker. Kept as a labelled false clip for the verifier's training.
                    if (playbackRecently()) {
                        Log.i(TAG, "Detection ignored: this phone was playing audio (score=%.3f)".format(score))
                        saveTriggerClip(score, pooled, label = false)
                        consecutiveHighScores = 0
                        cooldownRemaining = COOLDOWN_CHUNKS / 2
                        embBuffer.clear()
                        return
                    }
                    // Speech gate, second stage: the sound scored like "Hey Zynk" but no frame
                    // in the last ~2 s had a human voice in it.
                    val voice = if (vadSession != null) vadRecentMax() else -1f
                    if (vadSession != null && voice < VAD_SPEECH) {
                        Log.i(TAG, "Detection ignored: no voice in the last 2 s (vad=%.2f, score=%.3f)".format(voice, score))
                        saveTriggerClip(score, pooled, label = false)
                        consecutiveHighScores = 0
                        cooldownRemaining = COOLDOWN_CHUNKS / 2
                        embBuffer.clear()
                        return
                    }
                    val v = verifier ?: WakeVerifier.load(this).also { verifier = it }
                    val vScore = v?.score(flatEmb, EMB_WINDOW, EMB_SIZE) ?: -1f
                    val enforcing = v != null && v.enforcesOn(this)
                    val wouldReject = v != null && vScore >= 0f && vScore <= v.threshold
                    Log.i(TAG, "Wake word detected! score=$score threshold=$threshold level=%.1f dBFS vad=%.2f verifier=%.3f%s%s".format(
                        level, voice, vScore, if (wouldReject && !enforcing) " (verifier would reject, log-only)" else "", if (strict) " (strict mode)" else ""))
                    if (wouldReject && enforcing) {
                        Log.i(TAG, "Detection ignored: verifier says not the owner (%.3f <= %.2f)".format(vScore, v!!.threshold))
                        saveTriggerClip(score, pooled)
                        consecutiveHighScores = 0
                        cooldownRemaining = COOLDOWN_CHUNKS / 2
                        embBuffer.clear()
                        return
                    }
                    saveTriggerClip(score, pooled)
                    consecutiveHighScores = 0
                    cooldownRemaining = COOLDOWN_CHUNKS
                    embBuffer.clear()

                    // Every trigger takes the native path, whether or not the app is on
                    // screen (since 2026-09-07; the in-app flow is gone).
                    onWake()
                }
            } else {
                consecutiveHighScores = 0
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}")
        }
    }

    // ── Other apps' recordings ───────────────────────────────────────────────

    private fun registerRecordingWatch() {
        if (recordingCallback != null) return
        try {
            val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
            val cb = object : AudioManager.AudioRecordingCallback() {
                override fun onRecordingConfigChanged(configs: MutableList<AudioRecordingConfiguration>) {
                    onRecordingsChanged(configs)
                }
            }
            am.registerAudioRecordingCallback(cb, Handler(Looper.getMainLooper()))
            recordingCallback = cb
            onRecordingsChanged(am.activeRecordingConfigurations)
        } catch (e: Exception) {
            Log.w(TAG, "Recording watch unavailable: ${e.message}")
        }
    }

    private fun unregisterRecordingWatch() {
        val cb = recordingCallback ?: return
        recordingCallback = null
        try { (getSystemService(Context.AUDIO_SERVICE) as AudioManager).unregisterAudioRecordingCallback(cb) } catch (_: Exception) {}
    }

    /** Anything recording that is not our one stream is another app (or the page's
     *  own WebView recorder, which is as good as one). The event goes through the
     *  state machine like every other. */
    private fun onRecordingsChanged(configs: List<AudioRecordingConfiguration>) {
        val mine = stream.sessionId
        val others = configs.filter { it.clientAudioSessionId != mine }
        if (others.isNotEmpty() && !mic.foreignActive) {
            val what = others.joinToString { "src=${it.clientAudioSource}/session=${it.clientAudioSessionId}" }
            Log.i(TAG, "Another recording is active ($what)")
            events.execute { apply(MicState.Event.ForeignStarted) }
        } else if (others.isEmpty() && mic.foreignActive) {
            Log.i(TAG, "Other recording ended")
            events.execute { apply(MicState.Event.ForeignEnded) }
        }
    }

    // ── Trigger clips ────────────────────────────────────────────────────────

    /** Save the audio that fired the wake word (the last ~2 s, ending with the chunk
     *  that fired) as 16 kHz mono WAV under files/zynkbot/wake_triggers/, newest
     *  TRIGGER_CLIPS_KEPT kept. Pull with run-as (debug builds) to hear exactly what the
     *  model took for "Hey Zynk"; false triggers become training negatives if the model
     *  is retrained. Called on the audio thread; the file write happens off it. */
    /** `pooled`: the 192 verifier features at the fire point, saved as <stem>.feat for the
     *  on-phone trainer. `label`: when the service already knows the outcome (a gate
     *  rejected it) the label is written here, in the same thread as the clip, so the
     *  later "how did the turn end" report cannot race it. */
    private fun saveTriggerClip(score: Float, pooled: FloatArray? = null, label: Boolean? = null) {
        val snapshot = recentChunks.toList()
        val scoreSnap = recentScores.toList()
        if (snapshot.isEmpty()) return
        Thread {
            try {
                val dir = File(filesDir, "zynkbot/wake_triggers").apply { mkdirs() }
                val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
                val file = File(dir, "$stamp-${"%.3f".format(Locale.US, score)}.wav")
                val stem = file.name.removeSuffix(".wav")
                if (label == null) lastClipStem = stem   // a gated-out clip is not "the newest turn"
                if (pooled != null) WakeVerifierTrainer.saveFeatures(this, stem, pooled)
                if (label != null) File(dir, "$stem.${if (label) "real" else "false"}").writeText(if (label) "real\n" else "false\n")
                val dataBytes = snapshot.sumOf { it.size } * 2
                FileOutputStream(file).use { out ->
                    out.write(wavHeader(dataBytes, 16000))
                    val buf = java.nio.ByteBuffer.allocate(dataBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    for (c in snapshot) for (sample in c) buf.putShort(sample)
                    out.write(buf.array())
                }
                // Per-chunk scores next to the clip (one per line, oldest first).
                File(dir, file.name.removeSuffix(".wav") + ".scores.txt")
                    .writeText(scoreSnap.joinToString("\n") { "%.4f".format(Locale.US, it) })
                // A clip labelled real is training data and is never pruned by age alone —
                // only the unlabelled/false pool is trimmed to the cap. Before this, heavy
                // false-trigger churn (background noise, a video playing) could push the
                // total past the cap and delete an older *real* clip along with it, so the
                // "Send my wake-word clips" count went backwards even with frequent real use
                // (Matt, 2026-09-21: 12 real clips over 12 hours' use, dropped to 11).
                val allWavs = dir.listFiles { f -> f.name.endsWith(".wav") }?.toList() ?: emptyList()
                val keepCount = if (verifier?.enforcesOn(this) == true) TRIGGER_CLIPS_KEPT else TRIGGER_CLIPS_KEPT_UNVERIFIED
                val (real, rest) = allWavs.partition { File(dir, it.name.removeSuffix(".wav") + ".real").exists() }
                rest.sortedByDescending { it.name }
                    .drop(keepCount)
                    .forEach {
                        // The audio and the score trace go; a labelled clip's features and
                        // label (about 2 KB) stay, so the training set only ever grows and a
                        // retrain never trades last week's negatives for today's (Matt,
                        // 2026-10-20). An unlabelled clip is no training material: all of it goes.
                        val stem = it.name.removeSuffix(".wav"); it.delete(); File(dir, "$stem.scores.txt").delete()
                        if (!File(dir, "$stem.false").exists()) File(dir, "$stem.feat").delete()
                    }
                Log.i(TAG, "Trigger clip saved: ${file.name}")
            } catch (e: Exception) {
                Log.w(TAG, "Trigger clip not saved: ${e.message}")
            }
        }.start()
    }

    private fun wavHeader(dataBytes: Int, sampleRate: Int): ByteArray {
        val b = java.nio.ByteBuffer.allocate(44).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray()); b.putInt(36 + dataBytes); b.put("WAVE".toByteArray())
        b.put("fmt ".toByteArray()); b.putInt(16); b.putShort(1); b.putShort(1)
        b.putInt(sampleRate); b.putInt(sampleRate * 2); b.putShort(2); b.putShort(16)
        b.put("data".toByteArray()); b.putInt(dataBytes)
        return b.array()
    }

    // ── A wake ────────────────────────────────────────────────────────────────

    /** The detector fired (audio thread). The turn runs on its own thread: chime, then
     *  the assistant session if the system has bound it, else the in-service copy. */
    private fun onWake() {
        Log.i(TAG, "Wake word — starting a turn")
        // Release the indefinite audio-loop wake lock; detection lock covers the next 25s.
        releaseAudioWakeLock()
        val pm = getSystemService(PowerManager::class.java)
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "zynkbot:screen_off_wake")
        wakeLock?.acquire(25_000L)

        Thread {
            // Not listening any more by the time this thread runs (paused, turned off): no turn.
            if (!apply(MicState.Event.Wake)) { releaseWakeLock(); return@Thread }
            playTone(R.raw.wake_chime)   // the stream stays open; chunks are dropped meanwhile
            chimeEndedAt = System.currentTimeMillis()

            // Preferred path: hand the turn to the assistant session (the OS-blessed
            // route — no Activity, no WebView, no notification). Only possible while
            // Zynkbot holds the digital-assistant role and the system has bound the
            // service; otherwise keep the in-service copy so nothing regresses.
            val assistant = ZynkAssistantService.instance
            if (assistant != null && assistant.triggerSession()) {
                Log.i(TAG, "Wake word handed off to the assistant session")
                // The 25s detection lock is too short for dictation + network + TTS.
                try { wakeLock?.acquire(90_000L) } catch (_: Exception) {}
                return@Thread
            }
            // Seen on the GrapheneOS Pixel (2026-09-12): the assistant role is held but
            // Settings.Secure.voice_interaction_service was never set, so the system never
            // bound ZynkAssistantService and every trigger landed here without a trace.
            if (assistant == null) Log.w(TAG, "Assistant service not bound (voice_interaction_service unset?) — using in-service dictation")
            startInServiceDictation()
        }.start()
    }

    /** The in-service copy of a turn's dictation: Vosk, hands-free timers, then
     *  [answerNatively]. The assistant session has the better-kept copy (it honours
     *  the Vosk/Whisper selector); they merge in step 3 of the rebuild. */
    private fun startInServiceDictation() {
        val model = sharedVoskModel ?: run {
            Log.w(TAG, "No Vosk model available for screen-off dictation")
            releaseWakeLock()
            endTurn()
            return
        }
        val d = Dictation(Dictation.Engine.VOSK, model, null, timers = true, listener = object : Dictation.Listener {
            override fun onFinal(text: String?) {
                if (text == null) {
                    Log.w(TAG, "Screen-off dictation cancelled or failed")
                    releaseWakeLock()
                    endTurn()
                    return
                }
                Log.i(TAG, "Screen-off transcript: \"$text\"")
                Thread { answerNatively(text) }.start()
            }
        })
        if (!dictate(d, ownsTurn = true)) {
            Log.w(TAG, "Dictation refused in state ${mic.state}")
            releaseWakeLock()
            endTurn()
        }
    }

    // Answer a screen-off transcript entirely natively: no Activity, no WebView,
    // no notification. Falls back to deliverTranscriptToApp() (the WebView/
    // notification path) if anything in the native path fails or produces no
    // speakable reply, so a locked-screen query still gets answered somehow.
    // Shared with ZynkAssistantSession — see NativeVoiceAnswerer.
    private fun answerNatively(transcript: String) {
        if (transcript.isBlank()) {
            Log.i(TAG, "Empty transcript — not waking screen")
            noteOutcome(false)   // a wake that led to nothing: a false clip for the trainer (the session path already did this)
            releaseWakeLock()
            endTurn()
            return
        }
        // Local sanity gate, free: one word is noise, sixty is a TV programme.
        val words = transcript.trim().split(Regex("\\s+")).filter { it.isNotBlank() }
        if (words.size < 2 || words.size > MAX_QUERY_WORDS) {
            Log.i(TAG, "Transcript rejected (${words.size} words) — not a question")
            noteOutcome(false)
            releaseWakeLock()
            if (MainActivity.fruitlessTone(this)) playTone(R.raw.wake_chime_close)
            endTurn()
            return
        }
        // "Sent" tone before anything slow, as ZynkAssistantSession does: until
        // 2026-09-12 this path played it after the spoken reply, so the user waited
        // through the whole round trip in silence and then heard "sent" at the end.
        VoiceCommands.parse(transcript)?.let { cmd ->
            Thread {
                playTone(R.raw.wake_chime_close)
                val ok = VoiceCommands.execute(this, cmd)
                Log.i(TAG, "Voice command ${cmd::class.simpleName}: ${if (ok) "done" else "FAILED"}")
                NativeVoiceAnswerer.say(this, if (ok) VoiceCommands.confirmation(cmd) else VoiceCommands.FAILED_LINE)
                releaseWakeLock()
                endTurn()
            }.start()
            return
        }
        Log.i(TAG, "Answering natively: \"$transcript\"")

        // The 25s detection wake lock (acquired for chime+dictation) is too short
        // for a network round trip plus speech; extend it for this attempt.
        try { wakeLock?.acquire(90_000L) } catch (_: Exception) {}

        Thread {
            playTone(R.raw.wake_chime_close)
            val spoke = NativeVoiceAnswerer.answer(this, transcript)   // labels the clip itself
            if (spoke) {
                releaseWakeLock()
                endTurn()
            } else {
                deliverTranscriptToApp(transcript)
                // On GrapheneOS the app only opens when the notification is tapped, so
                // waiting for its JS would leave the wake word off: the turn ends here.
                endTurn()
            }
        }.start()
    }

    /** Play one bundled sound and wait for it (≤ 2 s). Media volume: the assistant
     *  stream has no user-facing slider and sat at 5/15 on the Pixel (2026-09-08). */
    private fun playTone(resId: Int) {
        try {
            val mp = MediaPlayer()
            mp.setAudioAttributes(
                android.media.AudioAttributes.Builder()
                    .setUsage(android.media.AudioAttributes.USAGE_MEDIA)
                    .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            resources.openRawResourceFd(resId)?.let { afd ->
                mp.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
                afd.close()
            }
            mp.prepare()
            val latch = CountDownLatch(1)
            mp.setOnCompletionListener { it.release(); latch.countDown() }
            mp.start()
            latch.await(2000, TimeUnit.MILLISECONDS)
        } catch (e: Exception) {
            Log.w(TAG, "Tone failed: ${e.message}")
        }
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

        // Always post the notification as a tap-to-open fallback (GrapheneOS, or when
        // startActivity is blocked). It used to carry a full-screen intent that opened
        // the app on a locked screen; that permission is restricted by Google Play to
        // alarm and calling apps, and the native assistant path answers without
        // opening the app, so the notification is now tap-to-open only.
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
            .setAutoCancel(true)
            .build()
        getSystemService(NotificationManager::class.java)
            .notify(TRANSCRIPT_NOTIFICATION_ID, notification)

        releaseWakeLock()

        // Closing tone (faster/higher pitch = "done"): the listening window is over.
        Thread { playTone(R.raw.wake_chime_close) }.start()
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
            // HIGH importance: the transcript notification should be seen at once
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

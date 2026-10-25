package ai.containai.zynkbot

import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.media.RingtoneManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.app.NotificationCompat

/**
 * Speaks the line ("Your ten-minute timer is done.") and rings for up to
 * ZynkTimer.RING_SECONDS, as a short foreground service started by the exact alarm.
 * Stops on the notification's Stop button, on ZynkTimer.stopRinging() (any "Hey Zynk",
 * a tap in the app, the assistant session opening), or when the cap runs out.
 * `shortService` type: no special-use declaration, no extra permission; it is capped
 * at three minutes by the system, far above the 30 s we use.
 */
class TimerRingService : Service() {
    private val main = Handler(Looper.getMainLooper())
    private var player: MediaPlayer? = null
    private var stopped = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ZynkTimer.ACTION_STOP_RING) { finish(); return START_NOT_STICKY }
        val line = intent?.getStringExtra("line") ?: "Your timer is done."
        ZynkTimer.ensureChannels(this)
        val stop = PendingIntent.getService(this, 21,
            Intent(this, TimerRingService::class.java).setAction(ZynkTimer.ACTION_STOP_RING),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val n = NotificationCompat.Builder(this, ZynkTimer.CHANNEL_RING)
            .setSmallIcon(android.R.drawable.ic_lock_idle_alarm)
            .setContentTitle(line)
            .setContentText("Say \"Hey Zynk\" or tap Stop")
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setOngoing(true)
            .addAction(0, "Stop", stop)
            .setContentIntent(PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE))
            .build()
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) startForeground(ZynkTimer.NOTIF_RING, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SHORT_SERVICE)
            else startForeground(ZynkTimer.NOTIF_RING, n)
        } catch (e: Exception) { Log.e(TAG, "startForeground failed: ${e.message}"); stopSelf(); return START_NOT_STICKY }

        ZynkTimer.ringing = true
        ZynkTimer.changed()
        Thread {
            // Say it first, then ring. say() pauses the wake detector while it speaks and
            // re-arms it after, so "Hey Zynk" works during the ring.
            try { NativeVoiceAnswerer.say(this, line) } catch (e: Exception) { Log.w(TAG, "say: ${e.message}") }
            if (!stopped) main.post { startRing() }
        }.start()
        main.postDelayed({ finish() }, (ZynkTimer.RING_SECONDS + 8) * 1000L) // 8 s allowance for the spoken line
        return START_NOT_STICKY
    }

    private fun startRing() {
        if (stopped) return
        try {
            val uri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
                ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION)
            val p = MediaPlayer()
            p.setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ALARM).setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION).build())
            p.setDataSource(this, uri)
            p.isLooping = true
            p.prepare()
            p.start()
            player = p
        } catch (e: Exception) { Log.w(TAG, "ring: ${e.message}") }
    }

    private fun finish() {
        if (stopped) return
        stopped = true
        try { player?.stop(); player?.release() } catch (_: Exception) {}
        player = null
        try { NativeVoiceAnswerer.stopSpeaking() } catch (_: Exception) {}
        ZynkTimer.ringing = false
        ZynkTimer.changed()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() { finish(); super.onDestroy() }

    companion object { private const val TAG = "TimerRingService" }
}

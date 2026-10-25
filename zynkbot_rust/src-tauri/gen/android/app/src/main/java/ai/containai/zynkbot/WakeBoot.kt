package ai.containai.zynkbot

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Handler
import android.os.Looper
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import java.io.File

/**
 * Hey Zynk after a reboot (2026-10-25, Matt). The listener used to start only when
 * the page started it, so after a restart nothing listened until the app was opened.
 * The page mirrors its switch here (heyZynkEnabled, threshold) and a second switch,
 * "Start listening when the phone starts", defaults on. Two chances to start:
 * ZynkAssistantService.onReady (the system binds it at boot while Zynkbot holds the
 * assistant role) and the BOOT_COMPLETED receiver. Android 15+ refuses a microphone
 * foreground service from a boot receiver outright, and whether the assistant path is
 * allowed is what the phones have to tell us; so eight seconds after each attempt we
 * look whether the service is alive and, if not, post a notification whose tap opens
 * the app, which starts the listener the ordinary way.
 */
object WakeBoot {
    private const val TAG = "WakeBoot"
    private const val PREFS = "zynk_boot"
    private const val CHANNEL = "zynk_boot"
    private const val NOTIF_ID = 7601

    private fun prefs(ctx: Context) = ctx.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun heyZynkEnabled(ctx: Context): Boolean = prefs(ctx).getBoolean("hey_zynk_enabled", false)
    fun setHeyZynkEnabled(ctx: Context, on: Boolean, threshold: Float) {
        prefs(ctx).edit().putBoolean("hey_zynk_enabled", on).putFloat("threshold", threshold).apply()
    }
    fun startOnBoot(ctx: Context): Boolean = prefs(ctx).getBoolean("start_on_boot", true)
    fun setStartOnBoot(ctx: Context, on: Boolean) { prefs(ctx).edit().putBoolean("start_on_boot", on).apply() }

    fun shouldStart(ctx: Context): Boolean = heyZynkEnabled(ctx) && startOnBoot(ctx) && WakeWordService.instance == null

    /** Try to start the listener from `reason`; verify in 8 s; notify if it is not up. */
    fun tryStart(ctx: Context, reason: String) {
        if (!shouldStart(ctx)) { WakeLog.i(TAG, "Boot start ($reason): nothing to do (enabled=${heyZynkEnabled(ctx)}, onBoot=${startOnBoot(ctx)}, running=${WakeWordService.instance != null})"); return }
        val app = ctx.applicationContext
        val intent = Intent(app, WakeWordService::class.java).apply {
            putExtra("threshold", prefs(app).getFloat("threshold", 0.72f))
            putExtra("modelDir", File(app.filesDir, "wake-word-models").absolutePath)
        }
        try {
            ContextCompat.startForegroundService(app, intent)
            WakeLog.i(TAG, "Boot start ($reason): requested")
        } catch (e: Exception) {
            WakeLog.w(TAG, "Boot start ($reason): refused at once: ${e.javaClass.simpleName}: ${e.message}")
        }
        Handler(Looper.getMainLooper()).postDelayed({
            val up = WakeWordService.instance != null
            WakeLog.i(TAG, "Boot start ($reason): after 8 s the listener is ${if (up) "up" else "NOT up"}")
            if (up) cancelNotice(app) else notice(app)
        }, 8_000L)
    }

    private fun notice(ctx: Context) {
        val nm = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "Hey Zynk after a restart", NotificationManager.IMPORTANCE_DEFAULT).apply { setSound(null, null) })
        }
        val open = PendingIntent.getActivity(ctx, 61, Intent(ctx, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK), PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val n = NotificationCompat.Builder(ctx, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Hey Zynk is off since the restart")
            .setContentText("Tap to start listening again.")
            .setContentIntent(open).setAutoCancel(true).setSilent(true)
            .build()
        try { nm.notify(NOTIF_ID, n) } catch (_: Exception) {}
    }

    fun cancelNotice(ctx: Context) {
        try { (ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager).cancel(NOTIF_ID) } catch (_: Exception) {}
    }
}

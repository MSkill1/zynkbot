package ai.containai.zynkbot

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat

/**
 * A silent notification that opens the app, posted by Rust (android_jni::post_notice)
 * when a memory card is raised and the app is not on screen (2026-10-26). The card
 * itself is kept in the database and shown when the app opens.
 */
object ZynkNotify {
    private const val CHANNEL = "zynk_cards"
    private const val NOTIF_ID = 76000

    @JvmStatic
    fun post(ctx: Context, title: String, text: String) {
        if (MainActivity.isInForeground) return
        try {
            val nm = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && nm.getNotificationChannel(CHANNEL) == null) {
                nm.createNotificationChannel(NotificationChannel(CHANNEL, "Questions about a memory", NotificationManager.IMPORTANCE_DEFAULT).apply { setSound(null, null) })
            }
            val open = PendingIntent.getActivity(ctx, 76, Intent(ctx, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
            val n = NotificationCompat.Builder(ctx, CHANNEL)
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle(title).setContentText(text)
                .setContentIntent(open).setAutoCancel(true).setSilent(true)
                .build()
            nm.notify(NOTIF_ID, n)
        } catch (_: Exception) {}
    }

    @JvmStatic
    fun cancel(ctx: Context) {
        try { (ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager).cancel(NOTIF_ID) } catch (_: Exception) {}
    }
}

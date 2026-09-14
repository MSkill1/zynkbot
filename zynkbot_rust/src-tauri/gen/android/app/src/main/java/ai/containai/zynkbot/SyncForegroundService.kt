package ai.containai.zynkbot

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat

class SyncForegroundService : Service() {

    companion object {
        private const val TAG = "SyncForegroundService"
        const val CHANNEL_ID = "zynksync_channel"
        const val NOTIFICATION_ID = 1001
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // A null intent means Android is re-creating the process to restart a sticky
        // service it had killed. There is nothing for this service to do on its own —
        // the sync server lives in the app's Rust core, which MainActivity starts —
        // and the app is in the background at that moment, so startForeground() is
        // refused (ForegroundServiceStartNotAllowedException on Android 12+). Left
        // uncaught, that refusal crashed the app every time the system reaped it
        // (tester's Pixel on Android 17 and Matt's Pixel, 2026-09-14). Decline the
        // restart and never ask to be restarted again.
        if (intent == null) {
            Log.i(TAG, "Sticky restart with no intent — nothing to run without the app; stopping")
            stopSelf(startId)
            return START_NOT_STICKY
        }

        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Zynkbot")
            .setContentText("Memory sync active")
            .setSmallIcon(android.R.drawable.ic_popup_sync)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
        } catch (e: Exception) {
            // ForegroundServiceStartNotAllowedException (API 31+), SecurityException,
            // IllegalStateException: the OS will not let us be a foreground service
            // right now. Losing the notification is harmless; crashing is not.
            Log.w(TAG, "startForeground refused: ${e.javaClass.simpleName}: ${e.message}")
            stopSelf(startId)
            return START_NOT_STICKY
        }

        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Zynkbot Sync",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Keeps memory sync running in the background"
                setShowBadge(false)
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }
}

package ai.containai.zynkbot

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/** Exact-alarm target for ZynkTimer: the timer or alarm going off, a Cancel tap on the
 *  countdown notification, and the reboot hook that puts pending ones back. */
class TimerReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val kind = intent.getStringExtra(ZynkTimer.EXTRA_KIND) ?: "timer"
        val id = intent.getIntExtra("id", 0)
        when (intent.action) {
            ZynkTimer.ACTION_FIRE -> ZynkTimer.onFired(context, kind, id)
            ZynkTimer.ACTION_CANCEL -> ZynkTimer.cancelById(context, kind, id)
            Intent.ACTION_BOOT_COMPLETED, "android.intent.action.LOCKED_BOOT_COMPLETED" -> ZynkTimer.rescheduleAfterBoot(context)
        }
    }
}

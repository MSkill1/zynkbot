package ai.containai.zynkbot

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Handler
import android.os.Looper

/**
 * After an update of the app (2026-10-28, KI-101): the install killed the process, so the
 * Hey Zynk listener was gone until the app was opened; Matt said "Hey Zynk" six times to a
 * phone that was not listening. Restart the listener the way boot does. The assistant
 * service (the Z) is bound by the system, not by us; when it has not come back after a
 * while, say so once, with a tap that opens the assistant setting.
 */
class PackageReplacedReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_MY_PACKAGE_REPLACED) return
        WakeLog.i(TAG, "App updated: restarting the listener")
        WakeBoot.tryStart(context, "package replaced")
        val app = context.applicationContext
        Handler(Looper.getMainLooper()).postDelayed({
            if (ZynkAssistantService.instance == null && holdsAssistantRole(app)) {
                WakeLog.w(TAG, "App updated: assistant service not bound yet; the Z needs a re-select or a reboot")
                WakeBoot.notice(app, "Zynkbot was updated",
                    "The Z on \"Hey Zynk\" comes back after a reboot, or after choosing Zynkbot again as the assistant. Tap to open that setting.",
                    Intent(android.provider.Settings.ACTION_VOICE_INPUT_SETTINGS), NOTIF_ID)
            }
        }, 20_000L)
    }

    private fun holdsAssistantRole(ctx: Context): Boolean = try {
        val rm = ctx.getSystemService(android.app.role.RoleManager::class.java)
        rm != null && rm.isRoleHeld(android.app.role.RoleManager.ROLE_ASSISTANT)
    } catch (_: Exception) { false }

    companion object {
        private const val TAG = "PackageReplaced"
        private const val NOTIF_ID = 7602
    }
}

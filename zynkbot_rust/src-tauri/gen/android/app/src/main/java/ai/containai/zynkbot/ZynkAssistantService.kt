package ai.containai.zynkbot

import android.os.Bundle
import android.service.voice.VoiceInteractionService
import android.util.Log

/**
 * Registers Zynkbot as the phone's digital assistant (Settings -> Apps -> Default
 * apps -> Digital assistant app). This is the OS-blessed entry point for hands-free
 * voice: its session can show over any app or the lock screen without needing
 * MainActivity's WebView to wake, and — per Android's own background-start
 * exemption list — a VoiceInteractionService provider is explicitly permitted to
 * start a microphone foreground service even when not in the foreground, which a
 * plain BOOT_COMPLETED receiver is not (see project memory: boot-start alone does
 * not work for a mic-type service on Android 14+; this is the documented exemption
 * that actually does).
 *
 * STATUS (2026-09): registered and structurally complete, but UNVERIFIED — not yet
 * built or run on a device. Open question (design doc gate G1): whether a
 * sideloaded, non-Play-Store app is even offered in the assistant picker on this
 * ROM at all. Not yet wired to WakeWordService's own ONNX detection loop — today
 * it is reachable only via the OS's own assistant-invocation gesture (long-press
 * power / home, depending on the device), which is itself the first thing to test.
 */
class ZynkAssistantService : VoiceInteractionService() {
    companion object {
        private const val TAG = "ZynkAssistantService"

        /**
         * The running instance while Zynkbot holds the assistant role, else null.
         * showSession() is an instance method, so WakeWordService needs this to hand
         * a detected wake word to the session. Same companion-static pattern as
         * WakeWordService.detectionCallback / sharedVoskModel.
         */
        @Volatile var instance: ZynkAssistantService? = null
    }

    override fun onReady() {
        super.onReady()
        instance = this
        Log.i(TAG, "Zynkbot assistant service ready")
    }

    override fun onShutdown() {
        instance = null
        super.onShutdown()
    }

    /**
     * Open a session (ZynkAssistantSession.onShow -> dictation -> native answer).
     * Called from WakeWordService on a detected "Hey Zynk" when the app is not in
     * front; the OS handles the rest under the assistant-role exemptions.
     * Returns false if the OS refused, so the caller can fall back.
     */
    fun triggerSession(): Boolean = try {
        showSession(Bundle(), 0)
        true
    } catch (e: Exception) {
        Log.w(TAG, "showSession refused: ${e.message}")
        false
    }
}

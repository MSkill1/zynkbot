package ai.containai.zynkbot

import android.content.Intent
import android.speech.RecognitionService
import android.speech.SpeechRecognizer

/**
 * Required by the assistant role, unused by Zynkbot.
 *
 * Android's VoiceInteractionServiceInfo parser treats a missing
 * android:recognitionService as a hard parse error ("No recognitionService
 * specified"), which makes the role controller reject the app as an
 * "unqualified voice interaction service" — verified against AOSP source on
 * 2026-09-03 after the ADB role assignment failed with exactly that message.
 *
 * Zynkbot does its own recognition with Vosk (WakeWordService / ZynkAssistantSession)
 * and never goes through Android's SpeechRecognizer framework, so this is a
 * minimal, honest stub: any client that binds it gets ERROR_CLIENT immediately
 * rather than a hang.
 */
class ZynkRecognitionService : RecognitionService() {
    override fun onStartListening(recognizerIntent: Intent?, listener: Callback?) {
        try { listener?.error(SpeechRecognizer.ERROR_CLIENT) } catch (_: Exception) {}
    }

    override fun onCancel(listener: Callback?) {}

    override fun onStopListening(listener: Callback?) {}
}

package ai.containai.zynkbot

import android.os.Bundle
import android.service.voice.VoiceInteractionSession
import android.service.voice.VoiceInteractionSessionService

/** Required separate service the OS binds to obtain a new session instance each
 *  time the assistant is invoked. Thin by design — all behavior lives in
 *  ZynkAssistantSession. */
class ZynkAssistantSessionService : VoiceInteractionSessionService() {
    override fun onNewSession(args: Bundle?): VoiceInteractionSession {
        return ZynkAssistantSession(this)
    }
}

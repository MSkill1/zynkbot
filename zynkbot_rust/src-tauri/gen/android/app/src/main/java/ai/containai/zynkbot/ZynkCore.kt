package ai.containai.zynkbot

/**
 * Native door into the Rust chat core.
 *
 * Lets Kotlin produce and stream a "Hey Zynk" reply entirely in native code —
 * no WebView resumed, no lock-screen notification. The Rust side lives in
 * src-tauri/src/android_jni.rs and calls the same chat core the Tauri command
 * uses (commands::chat::generate_reply), delivering results back through the
 * [Callback] below.
 *
 * NOTE (2026-09): called from WakeWordService.answerNatively() for the screen-off
 * path. If this fails or produces no speakable reply, that caller falls back to
 * the existing WebView/notification delivery, so a locked-screen query is never
 * left unanswered while this path is new. The in-app (screen-on, app open) voice
 * flow is untouched and still goes entirely through the WebView/Tauri path.
 */
object ZynkCore {
    init {
        // Already loaded in-process by Tauri's generated Rust.kt; loadLibrary is
        // idempotent, so this makes ZynkCore self-contained if called first.
        System.loadLibrary("app_lib")
    }

    /**
     * Produce a reply for [message] and stream it back through [cb]. Blocking:
     * call from a background thread (the assistant service worker), never the
     * main thread. Tokens arrive on [Callback.onToken]; the run finishes with
     * exactly one of [Callback.onDone] or [Callback.onError].
     *
     * Pass "" for [userId] to resolve the persisted device identity automatically,
     * and "" for [sessionId] to continue the current conversation thread (the one
     * the app last had on screen; see set_current_session) — Kotlin does not need
     * its own copy of that logic. The thread id comes back as a "voice-session"
     * event before the first token.
     */
    @JvmStatic
    external fun nativeSendMessage(
        message: String,
        userId: String,
        sessionId: String,
        backend: String,
        containmentMode: String,
        cb: Callback,
    )

    interface Callback {
        /** One streamed token of the reply text. */
        fun onToken(token: String)
        /** A side-channel event (memory-processing progress, contradiction) with a JSON payload. */
        fun onEvent(name: String, payloadJson: String)
        /** The reply completed; [replyJson] is the serialized ReplyResponse. */
        fun onDone(replyJson: String)
        /** The reply failed; [message] is a human-readable error. */
        fun onError(message: String)
    }
}

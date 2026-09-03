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
 * NOTE (2026-09): the door is defined and compile-verified on both sides, but is
 * not yet wired to any caller. The planned ZynkAssistantService will invoke
 * nativeSendMessage() so a locked-screen query is answered without the notification
 * path that ColorOS blocks. Until then the app's voice flow still runs through the
 * WebView/Tauri path unchanged.
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

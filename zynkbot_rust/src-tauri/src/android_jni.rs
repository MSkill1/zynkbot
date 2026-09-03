//! The native door into the chat core, callable from Kotlin.
//!
//! `WakeWordService` (or the future `ZynkAssistantService`) can call
//! `ZynkCore.nativeSendMessage(...)` to produce and stream a reply entirely in
//! native code — no WebView resumed, no lock-screen notification. Tokens and
//! completion are delivered back through a Kotlin callback object via `JniSink`,
//! which implements the same `ResponseSink` the Tauri path uses.
//!
//! Kotlin side (see gen/android/.../ZynkCore.kt):
//! ```
//! object ZynkCore {
//!     external fun nativeSendMessage(
//!         message: String, userId: String, sessionId: String,
//!         backend: String, containmentMode: String, cb: Callback)
//!     interface Callback {
//!         fun onToken(token: String)
//!         fun onEvent(name: String, payloadJson: String)
//!         fun onDone(replyJson: String)
//!         fun onError(message: String)
//!     }
//! }
//! ```
#![cfg(target_os = "android")]

use crate::response_sink::ResponseSink;
use jni::objects::{GlobalRef, JClass, JObject, JString, JValue};
use jni::{JNIEnv, JavaVM};
use std::sync::Arc;

/// Streams a reply back into a Kotlin `ZynkCore.Callback`. The sink may be invoked
/// from a worker thread of the tokio runtime we spin up below, so every call
/// re-attaches the current thread to the JVM (idempotent once attached).
struct JniSink {
    vm: JavaVM,
    callback: GlobalRef,
}

impl JniSink {
    /// Call a one-String-argument callback method (onToken / onDone / onError).
    fn call_str(&self, method: &str, arg: &str) {
        let Ok(mut env) = self.vm.attach_current_thread_as_daemon() else { return };
        let Ok(jarg) = env.new_string(arg) else { return };
        let _ = env.call_method(
            self.callback.as_obj(),
            method,
            "(Ljava/lang/String;)V",
            &[JValue::Object(&jarg)],
        );
    }

    fn done(&self, reply_json: &str) {
        self.call_str("onDone", reply_json);
    }

    fn error(&self, message: &str) {
        self.call_str("onError", message);
    }
}

impl ResponseSink for JniSink {
    fn token(&self, token: &str) {
        self.call_str("onToken", token);
    }

    fn event(&self, name: &str, payload: serde_json::Value) {
        let Ok(mut env) = self.vm.attach_current_thread_as_daemon() else { return };
        let payload_json = payload.to_string();
        let (Ok(jname), Ok(jpayload)) = (env.new_string(name), env.new_string(&payload_json)) else {
            return;
        };
        let _ = env.call_method(
            self.callback.as_obj(),
            "onEvent",
            "(Ljava/lang/String;Ljava/lang/String;)V",
            &[JValue::Object(&jname), JValue::Object(&jpayload)],
        );
    }
}

/// JNI entry point. Signature must match `ZynkCore.nativeSendMessage`.
/// Runs the async chat core to completion on a dedicated tokio runtime (so the
/// core's `tokio::spawn`ed background memory work has a runtime handle), then
/// reports the result through the callback.
#[no_mangle]
pub extern "system" fn Java_ai_containai_zynkbot_ZynkCore_nativeSendMessage<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    message: JString<'local>,
    user_id: JString<'local>,
    session_id: JString<'local>,
    backend: JString<'local>,
    containment_mode: JString<'local>,
    callback: JObject<'local>,
) {
    let message = jstring_to_string(&mut env, &message);
    let user_id = jstring_to_string(&mut env, &user_id);
    let session_id = jstring_to_string(&mut env, &session_id);
    let backend = jstring_to_string(&mut env, &backend);
    let containment_mode = jstring_to_string(&mut env, &containment_mode);

    // A global ref keeps the callback alive across the async run and across the
    // thread hops the sink makes. The JavaVM lets the sink re-attach threads.
    let vm = match env.get_java_vm() {
        Ok(vm) => vm,
        Err(e) => {
            eprintln!("[ZynkCore JNI] get_java_vm failed: {e}");
            return;
        }
    };
    let callback = match env.new_global_ref(callback) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[ZynkCore JNI] new_global_ref failed: {e}");
            return;
        }
    };

    let sink_impl = Arc::new(JniSink { vm, callback });
    let sink: Arc<dyn ResponseSink> = sink_impl.clone();

    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            sink_impl.error(&format!("failed to start async runtime: {e}"));
            return;
        }
    };

    let result = runtime.block_on(crate::commands::chat::generate_reply(
        sink,
        message,
        user_id,
        session_id,
        backend,
        containment_mode,
        None, // conversation_history
        None, // skip_containment
        None, // skip_memory_storage
        None, // kb_enabled
        None, // user_query
        None, // image_data
    ));

    match result {
        Ok(reply) => match serde_json::to_string(&reply) {
            Ok(json) => sink_impl.done(&json),
            Err(e) => sink_impl.error(&format!("failed to serialize reply: {e}")),
        },
        Err(e) => sink_impl.error(&e),
    }
}

/// Best-effort JString -> String; empty string on any failure.
fn jstring_to_string(env: &mut JNIEnv, s: &JString) -> String {
    env.get_string(s)
        .map(|js| js.to_string_lossy().into_owned())
        .unwrap_or_default()
}

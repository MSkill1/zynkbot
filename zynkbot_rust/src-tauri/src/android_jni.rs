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

/// The thread's recent turns, oldest first, in the shape generate_reply expects — the
/// same thing the web app sends from its own message list. Capped at 40 messages
/// (20 exchanges); generate_reply applies its per-model limit on top. Any failure
/// means "no history", never a failed answer.
async fn load_recent_turns(session_id: &str) -> Option<Vec<crate::ConversationTurn>> {
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url()).await.ok()?;
    let msgs = crate::conversation_history::get_messages(&pool, session_id).await;
    pool.close().await;
    let msgs = msgs.ok()?;
    if msgs.is_empty() {
        return None;
    }
    let start = msgs.len().saturating_sub(40);
    Some(
        msgs[start..]
            .iter()
            .map(|m| crate::ConversationTurn { role: m.role.clone(), content: m.content.clone() })
            .collect(),
    )
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
    let user_id_arg = jstring_to_string(&mut env, &user_id);
    let session_id_arg = jstring_to_string(&mut env, &session_id);
    let backend_arg = jstring_to_string(&mut env, &backend);
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

    // Empty user_id/session_id means "resolve for me" — Kotlin doesn't have (and
    // shouldn't need) its own copy of identity logic or session bookkeeping.
    let user_id = if user_id_arg.is_empty() {
        match crate::user_identity::get_identity() {
            Ok(identity) => identity.user_id,
            Err(e) => {
                sink_impl.error(&format!("failed to resolve user identity: {e}"));
                return;
            }
        }
    } else {
        user_id_arg
    };
    // Empty session_id means "the current thread": hands-free turns continue the
    // conversation the app last had on screen (recorded via set_current_session), so
    // "what's the capital of Arizona" / "when's a good time to go" hang together.
    // Before the app has ever recorded one, start a thread and make it current so the
    // next question lands in the same place.
    let session_id = if session_id_arg.is_empty() {
        match crate::user_identity::get_current_session_id() {
            Some(id) => id,
            None => {
                let id = format!("voice-{}", uuid::Uuid::new_v4());
                if let Err(e) = crate::user_identity::set_current_session_id(&id) {
                    eprintln!("[ZynkCore JNI] could not persist current session: {e}");
                }
                id
            }
        }
    } else {
        session_id_arg
    };
    // Tell Kotlin which thread this turn belongs to, so the finished exchange can be
    // shown in the app's chat when that thread is the one on screen.
    sink.event("voice-session", serde_json::json!({ "session_id": session_id }));
    // Empty backend means "pick whatever's actually configured" — see
    // resolve_voice_backend(). Fails closed (reports an error) rather than handing
    // generate_reply a backend guaranteed to fail, so the caller's fallback to the
    // WebView/notification path — which the user's own model selection covers — runs.
    let backend = if backend_arg.is_empty() {
        match resolve_voice_backend() {
            Some(b) => b,
            None => {
                sink_impl.error("no AI backend is configured (no API key and no custom/Ollama endpoint)");
                return;
            }
        }
    } else {
        backend_arg
    };

    // One runtime for the life of the process, not one per query: building a fresh
    // multi-thread runtime (and its worker threads) on every hands-free question is
    // waste, and anything cached per-thread would be cold each time.
    static RUNTIME: once_cell::sync::OnceCell<tokio::runtime::Runtime> = once_cell::sync::OnceCell::new();
    let runtime = match RUNTIME.get_or_try_init(|| {
        tokio::runtime::Builder::new_multi_thread().enable_all().build()
    }) {
        Ok(rt) => rt,
        Err(e) => {
            sink_impl.error(&format!("failed to start async runtime: {e}"));
            return;
        }
    };

    let result = runtime.block_on(async move {
        let history = load_recent_turns(&session_id).await;
        crate::commands::chat::generate_reply(
            sink,
            message,
            user_id,
            session_id,
            backend,
            containment_mode,
            history,
            None, // skip_containment
            None, // skip_memory_storage
            None, // kb_enabled
            None, // user_query
            None, // image_data
        )
        .await
    });

    match result {
        Ok(reply) => match serde_json::to_string(&reply) {
            Ok(json) => sink_impl.done(&json),
            Err(e) => sink_impl.error(&format!("failed to serialize reply: {e}")),
        },
        Err(e) => sink_impl.error(&e),
    }
}

/// Picks a backend string that will actually work for `generate_reply` on Android.
/// There is no viable on-device model here (Zynkbot's "local" GGUF backend needs a
/// downloaded model file that Android setup doesn't provide) — the two real options
/// are a configured cloud API key or a custom/Ollama endpoint, exactly what the
/// onboarding flow sets up. Checked in the same provider priority `generate_reply`'s
/// own has_key fallback uses, then Ollama. Returns None if nothing is configured, so
/// the caller can fail closed rather than hand generate_reply a backend that errors.
fn resolve_voice_backend() -> Option<String> {
    let has = |var: &str| !std::env::var(var).unwrap_or_default().is_empty();

    // First choice: whatever the user selected in the app's model dropdown, persisted
    // by set_preferred_backend as ZYNK_MODEL_BACKEND. Honored only if it is actually
    // usable here, so a stale value (e.g. 'local' with no model file on Android) can't
    // strand hands-free voice — it falls through to the first configured provider.
    if let Ok(pref) = std::env::var("ZYNK_MODEL_BACKEND") {
        let p = pref.to_lowercase();
        let usable = if p.contains("anthropic") || p.contains("claude") { has("ANTHROPIC_API_KEY") }
            else if p.contains("openai") || p.contains("gpt") { has("OPENAI_API_KEY") }
            else if p.contains("xai") || p.contains("grok") { has("XAI_API_KEY") }
            else if p.contains("mistral") { has("MISTRAL_API_KEY") }
            else if p == "custom" { has("CUSTOM_API_URL") && has("CUSTOM_MODEL") }
            else if p.contains("local") || p.ends_with(".gguf") {
                crate::llm::local_models::resolve_default_model_path().is_ok()
            } else { false };
        if usable {
            return Some(pref);
        }
        eprintln!("[ZynkCore JNI] preferred backend '{pref}' has no usable credentials here; falling back");
    }

    if has("ANTHROPIC_API_KEY") {
        Some("anthropic".to_string())
    } else if has("OPENAI_API_KEY") {
        Some("openai".to_string())
    } else if has("XAI_API_KEY") {
        Some("xai".to_string())
    } else if has("MISTRAL_API_KEY") {
        Some("mistral".to_string())
    } else if has("CUSTOM_API_URL") && has("CUSTOM_MODEL") {
        Some("custom".to_string())
    } else {
        None
    }
}

/// Best-effort JString -> String; empty string on any failure.
fn jstring_to_string(env: &mut JNIEnv, s: &JString) -> String {
    env.get_string(s)
        .map(|js| js.to_string_lossy().into_owned())
        .unwrap_or_default()
}

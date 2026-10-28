//! Where a chat response goes as it is produced.
//!
//! Today the only consumer is the Tauri command layer, which forwards the reply
//! to the WebView by emitting Tauri events. The native Android assistant path
//! (planned) will implement this same trait with a JNI callback into Kotlin, so
//! the chat core can run and speak a reply with no WebView resumed and no
//! lock-screen notification. Introducing the trait now is the first, behaviour-
//! preserving step of that work: `TauriSink` emits the exact same event names
//! the frontend already listens for.

use serde_json::Value;

/// A destination for a chat response as it streams in.
///
/// Implementations must be cheap to share across the async streaming closures
/// (hence `Send + Sync`, held behind an `Arc`).
pub trait ResponseSink: Send + Sync {
    /// One streamed token of the assistant's reply text.
    fn token(&self, token: &str);

    /// A named side-channel event with a JSON payload — e.g. memory-processing
    /// progress or a detected contradiction. Default no-op: sinks that don't
    /// surface these (such as a bare voice reply) simply ignore them.
    fn event(&self, _name: &str, _payload: Value) {}

    /// A card was raised and the user may not be looking: on Android, post a
    /// notification that opens the app (2026-10-26). Default no-op elsewhere.
    fn card_notice(&self, _title: &str, _text: &str) {
        #[cfg(target_os = "android")]
        crate::android_jni::post_notice(_title, _text);
    }
}

/// Forwards a response to the Tauri WebView, preserving the event names the
/// frontend depends on (`stream-token`, and whatever `event` names callers pass).
pub struct TauriSink {
    app: tauri::AppHandle,
}

impl TauriSink {
    pub fn new(app: tauri::AppHandle) -> Self {
        Self { app }
    }
}

impl ResponseSink for TauriSink {
    fn token(&self, token: &str) {
        use tauri::Emitter;
        let _ = self.app.emit("stream-token", token);
    }

    fn event(&self, name: &str, payload: Value) {
        use tauri::Emitter;
        let _ = self.app.emit(name, payload);
    }
}

/// Wraps another sink and logs how long the first word took from the moment the request
/// began (2026-10-28), so "it felt slow" has a number to read in the log.
pub struct TimedSink {
    inner: std::sync::Arc<dyn ResponseSink>,
    start: std::time::Instant,
    first_seen: std::sync::atomic::AtomicBool,
}

impl TimedSink {
    pub fn new(inner: std::sync::Arc<dyn ResponseSink>, start: std::time::Instant) -> Self {
        Self { inner, start, first_seen: std::sync::atomic::AtomicBool::new(false) }
    }
}

impl ResponseSink for TimedSink {
    fn token(&self, token: &str) {
        if !token.is_empty() && !self.first_seen.swap(true, std::sync::atomic::Ordering::SeqCst) {
            println!("[⏱️ PERF] First word on screen: {:.2}s after the request began", self.start.elapsed().as_secs_f32());
        }
        self.inner.token(token);
    }
    fn event(&self, name: &str, payload: Value) { self.inner.event(name, payload); }
    fn card_notice(&self, title: &str, text: &str) { self.inner.card_notice(title, text); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// A sink that records everything it receives, for tests that exercise the
    /// chat core without a Tauri runtime (the shape the native path will reuse).
    #[derive(Default)]
    pub struct RecordingSink {
        pub tokens: Mutex<Vec<String>>,
        pub events: Mutex<Vec<(String, Value)>>,
    }

    impl ResponseSink for RecordingSink {
        fn token(&self, token: &str) {
            self.tokens.lock().unwrap().push(token.to_string());
        }
        fn event(&self, name: &str, payload: Value) {
            self.events.lock().unwrap().push((name.to_string(), payload));
        }
    }

    #[test]
    fn recording_sink_collects_tokens_and_events() {
        // Keep a typed handle for assertions; pass a trait-object clone to the
        // code under test, exactly as the chat core will receive it.
        let recorded = Arc::new(RecordingSink::default());
        let sink: Arc<dyn ResponseSink> = recorded.clone();

        sink.token("Hello");
        sink.token(", world");
        sink.event("memory-processing-complete", serde_json::json!({ "status": "stored" }));

        assert_eq!(*recorded.tokens.lock().unwrap(), vec!["Hello", ", world"]);
        assert_eq!(recorded.events.lock().unwrap().len(), 1);
        assert_eq!(recorded.events.lock().unwrap()[0].0, "memory-processing-complete");
    }
}

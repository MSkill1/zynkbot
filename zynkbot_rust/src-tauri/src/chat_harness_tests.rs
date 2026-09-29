//! Chat harness (2026-09-29): the real chat path, end to end, against a pretend model.
//!
//! `generate_reply` is run exactly as the app runs it — safety check, memory recall,
//! prompt assembly, model call, reply parsing, memory extraction, history — with one
//! substitution: the model is a tiny OpenAI-compatible HTTP server started inside the
//! test on a loopback port. It answers the streaming chat request with whatever the
//! test scripted, and the app's follow-up "should I remember this?" request with a
//! scripted decision. Nothing inside the app is mocked; the app reaches the pretend
//! model through its own custom-endpoint code (`CUSTOM_API_URL`).
//!
//! Each test gets its own data directory (via `XDG_DATA_HOME`, which is how the app
//! finds its database on Linux), so nothing here touches a real installation. The
//! system models beside the source (`src-tauri/models/system`) are used for the safety
//! classifier, embeddings and entity extraction, as in a dev build.
//!
//! Environment variables are process-wide, so these tests hold `ENV_LOCK` and run one
//! at a time. Run: `LD_LIBRARY_PATH=$PWD/lib/vosk cargo test --lib chat_harness -- --nocapture`
//!
//! Behaviours:
//!   c01 a backend with no credentials falls back to one that can answer
//!       (the fix for a tester's phone, 2026-09-22 — typed chat failed for weeks because
//!       the selected backend could not run and nothing fell back)
//!   c02 the reply is streamed token by token, not delivered whole (KI-025 was stale)
//!   c03 a stored memory reaches the prompt the model sees
//!   c04 "Remember: …" stores the fact verbatim, whatever the model says
//!   c05 a reply carrying a memory marker becomes a stored memory, and the marker
//!       line is not shown to the user

use crate::response_sink::ResponseSink;
use axum::response::IntoResponse;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Row, SqlitePool};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn hold_env() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn rt_test<F: std::future::Future<Output = ()>>(f: F) {
    tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap().block_on(f)
}

// ---------------------------------------------------------------------------
// The pretend model: an OpenAI-compatible server on a loopback port.
// ---------------------------------------------------------------------------

struct FakeModel {
    base_url: String,
    replies: Arc<Mutex<VecDeque<String>>>,
    decision: Arc<Mutex<String>>,
    requests: Arc<Mutex<Vec<serde_json::Value>>>,
}

impl FakeModel {
    async fn start() -> FakeModel {
        let replies: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::new()));
        let decision = Arc::new(Mutex::new(r#"{"should_remember": true, "title": "Fact"}"#.to_string()));
        let requests: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));

        let (r, d, q) = (replies.clone(), decision.clone(), requests.clone());
        let handler = move |axum::Json(body): axum::Json<serde_json::Value>| {
            let (r, d, q) = (r.clone(), d.clone(), q.clone());
            async move {
                q.lock().unwrap().push(body.clone());
                let streaming = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
                if streaming {
                    // The chat request. Stream the scripted reply one word at a time,
                    // in the wire format the app's OpenAI client parses.
                    let reply = r.lock().unwrap().pop_front().unwrap_or_else(|| "Okay.".to_string());
                    let mut sse = String::new();
                    for piece in reply.split_inclusive(' ') {
                        let chunk = serde_json::json!({
                            "choices": [{ "delta": { "content": piece }, "finish_reason": null }]
                        });
                        sse.push_str(&format!("data: {}\n\n", chunk));
                    }
                    sse.push_str("data: [DONE]\n\n");
                    ([(axum::http::header::CONTENT_TYPE, "text/event-stream")], sse).into_response()
                } else {
                    // The follow-up decision request ("should I remember this?"), which
                    // the app makes without streaming and parses as JSON in the content.
                    let content = d.lock().unwrap().clone();
                    axum::Json(serde_json::json!({
                        "id": "fake", "model": "fake-model",
                        "choices": [{ "message": { "role": "assistant", "content": content }, "finish_reason": "stop" }],
                        "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
                    })).into_response()
                }
            }
        };

        let app = axum::Router::new().route("/chat/completions", axum::routing::post(handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind fake model");
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, app).await.expect("fake model server"); });

        FakeModel { base_url: format!("http://127.0.0.1:{}", port), replies, decision, requests }
    }

    fn will_say(&self, reply: &str) {
        self.replies.lock().unwrap().push_back(reply.to_string());
    }

    /// Every chat (streaming) request the app has made, in order.
    fn chat_requests(&self) -> Vec<serde_json::Value> {
        self.requests.lock().unwrap().iter()
            .filter(|r| r.get("stream").and_then(|s| s.as_bool()).unwrap_or(false))
            .cloned().collect()
    }

    /// The full text the model was sent for a request (the app sends one user message).
    fn prompt_of(req: &serde_json::Value) -> String {
        req["messages"].as_array().map(|m| m.iter()
            .filter_map(|x| x["content"].as_str()).collect::<Vec<_>>().join("\n")).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// One isolated installation: its own data directory and database.
// ---------------------------------------------------------------------------

struct Bench {
    dir: PathBuf,
    pool: SqlitePool,
    user_id: String,
    session_id: String,
}

impl Bench {
    async fn new(model: &FakeModel) -> Bench {
        let dir = std::env::temp_dir().join(format!("zynkbot-chat-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("zynkbot")).unwrap();

        // The app finds its data directory through XDG_DATA_HOME on Linux.
        std::env::set_var("XDG_DATA_HOME", &dir);
        std::env::set_var("CUSTOM_API_URL", &model.base_url);
        std::env::set_var("CUSTOM_MODEL", "fake-model");
        for k in ["CUSTOM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY",
                  "MISTRAL_API_KEY", "LOCAL_MODEL_PATH", "ZYNK_MODEL_BACKEND"] {
            std::env::remove_var(k);
        }
        assert_eq!(crate::db::get_app_data_dir(), dir.join("zynkbot"), "the app is not using the test data dir");

        let url = format!("sqlite://{}?mode=rwc", dir.join("zynkbot").join("zynkbot.db").display());
        let pool = SqlitePoolOptions::new().max_connections(8)
            .after_connect(|conn, _| Box::pin(async move {
                sqlx::query("PRAGMA foreign_keys=ON").execute(&mut *conn).await?;
                sqlx::query("PRAGMA busy_timeout=15000").execute(&mut *conn).await?;
                Ok(())
            }))
            .connect(&url).await.expect("open bench db");
        sqlx::migrate!("./migrations").run(&pool).await.expect("migrate bench db");

        Bench { dir, pool, user_id: uuid::Uuid::new_v4().to_string(), session_id: uuid::Uuid::new_v4().to_string() }
    }

    /// One typed message through the real chat path. Returns the reply as JSON so the
    /// test can read `reply_text` and friends without reaching into private fields.
    async fn ask(&self, sink: Arc<dyn ResponseSink>, backend: &str, message: &str, store_memories: bool) -> serde_json::Value {
        let resp = crate::commands::chat::generate_reply(
            sink, message.to_string(), self.user_id.clone(), self.session_id.clone(),
            backend.to_string(), "sovereign".to_string(),
            None, None, Some(!store_memories), Some(false), None, None, false,
        ).await.unwrap_or_else(|e| panic!("generate_reply failed: {}", e));
        serde_json::to_value(&resp).unwrap()
    }

    async fn memory_contents(&self) -> Vec<String> {
        sqlx::query("SELECT content FROM memories WHERE user_id = ? ORDER BY id")
            .bind(&self.user_id).fetch_all(&self.pool).await.unwrap()
            .into_iter().map(|r| r.get::<String, _>("content")).collect()
    }

    /// Memory storage runs in the background after the reply; wait for it.
    async fn wait_for_memory_containing(&self, needle: &str) -> Vec<String> {
        for _ in 0..80 {
            let all = self.memory_contents().await;
            if all.iter().any(|m| m.contains(needle)) { return all; }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
        self.memory_contents().await
    }
}

impl Drop for Bench {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.dir); }
}

/// Records what the chat path sends to the screen.
#[derive(Default)]
struct RecordingSink {
    tokens: Mutex<Vec<String>>,
}

impl ResponseSink for RecordingSink {
    fn token(&self, token: &str) { self.tokens.lock().unwrap().push(token.to_string()); }
}

fn reply_text(v: &serde_json::Value) -> String {
    v["reply_text"].as_str().unwrap_or("").to_string()
}

// ---------------------------------------------------------------------------
// c01. The selected backend has no credentials. The app must answer anyway, through
//      the first backend that can, instead of erroring. (On a phone the failing
//      selection is "local"; on this desktop local models exist beside the source, so
//      the test uses a keyless cloud provider, which takes the same fallback path.)
// ---------------------------------------------------------------------------
#[test]
fn c01_a_backend_with_no_credentials_falls_back_to_one_that_can_answer() {
    let _env = hold_env();
    rt_test(async {
        let model = FakeModel::start().await;
        let bench = Bench::new(&model).await;
        model.will_say("Montpelier is the capital of Vermont.");

        let sink = Arc::new(RecordingSink::default());
        let reply = bench.ask(sink.clone(), "anthropic", "What is the capital of Vermont?", false).await;

        assert_eq!(reply_text(&reply), "Montpelier is the capital of Vermont.",
            "the reply did not come from the fallback model: {}", reply);
        assert_eq!(model.chat_requests().len(), 1, "expected exactly one chat request to reach the model");
    });
}

// ---------------------------------------------------------------------------
// c02. The reply reaches the screen as a stream of tokens, not one block.
// ---------------------------------------------------------------------------
#[test]
fn c02_the_reply_is_streamed_token_by_token() {
    let _env = hold_env();
    rt_test(async {
        let model = FakeModel::start().await;
        let bench = Bench::new(&model).await;
        model.will_say("Four plus four is eight.");

        let sink = Arc::new(RecordingSink::default());
        let reply = bench.ask(sink.clone(), "custom", "What is four plus four?", false).await;

        let tokens = sink.tokens.lock().unwrap().clone();
        assert!(tokens.len() > 1, "reply arrived as {} token(s), expected several: {:?}", tokens.len(), tokens);
        assert_eq!(tokens.concat(), "Four plus four is eight.", "streamed tokens do not add up to the reply");
        assert_eq!(reply_text(&reply), "Four plus four is eight.");
    });
}

// ---------------------------------------------------------------------------
// c03. A memory the user stored earlier is recalled and placed in front of the model.
//      The check is on what the model was actually sent, not on what it replied.
// ---------------------------------------------------------------------------
#[test]
fn c03_a_stored_memory_reaches_the_prompt() {
    let _env = hold_env();
    rt_test(async {
        let model = FakeModel::start().await;
        let bench = Bench::new(&model).await;

        let fact = "My car is a blue Subaru Outback";
        let embedding = crate::llm::local_embeddings::generate_local_embedding(fact).expect("embed the memory");
        crate::memory::insert_memory(
            &bench.pool, Some("Car"), fact, Some("conversation"), None, Some(embedding), None, None,
            Some(&bench.user_id), "personal", true, false, Some(serde_json::json!([])), None, None, Some(fact),
        ).await.expect("insert memory");

        model.will_say("You drive a blue Subaru Outback.");
        let sink = Arc::new(RecordingSink::default());
        bench.ask(sink, "custom", "What car do I drive?", false).await;

        let requests = model.chat_requests();
        assert_eq!(requests.len(), 1);
        let prompt = FakeModel::prompt_of(&requests[0]);
        assert!(prompt.contains("Subaru"),
            "the stored memory never reached the model; prompt was:\n{}", prompt);
    });
}

impl FakeModel {
    /// What the model answers when the app asks whether to remember something.
    fn will_decide(&self, decision_json: &str) {
        *self.decision.lock().unwrap() = decision_json.to_string();
    }
}

// ---------------------------------------------------------------------------
// c04. "Remember: …" is the user's own instruction. It is stored word for word even
//      if the model's decision call says not to.
// ---------------------------------------------------------------------------
#[test]
fn c04_remember_colon_stores_the_fact_verbatim() {
    let _env = hold_env();
    rt_test(async {
        let model = FakeModel::start().await;
        let bench = Bench::new(&model).await;
        model.will_say("Got it, I'll remember that.");
        model.will_decide(r#"{"should_remember": false, "title": null}"#);

        let sink = Arc::new(RecordingSink::default());
        bench.ask(sink, "custom", "Remember: my car is blue", true).await;

        let stored = bench.wait_for_memory_containing("car is blue").await;
        assert!(stored.iter().any(|m| m.contains("car is blue")),
            "the explicit Remember was not stored; memories: {:?}", stored);
    });
}

// ---------------------------------------------------------------------------
// c05. The model marks a fact in its reply. The fact becomes a memory, and the marker
//      line is stripped from what the user sees.
// ---------------------------------------------------------------------------
#[test]
fn c05_a_reply_carrying_a_memory_marker_becomes_a_memory() {
    let _env = hold_env();
    rt_test(async {
        let model = FakeModel::start().await;
        let bench = Bench::new(&model).await;
        model.will_say("Burlington is a great town, glad it suits you.\nMEMORY_EXTRACT: The user lives in Burlington");
        model.will_decide(r#"{"should_remember": true, "title": "Lives in Burlington"}"#);

        let sink = Arc::new(RecordingSink::default());
        let reply = bench.ask(sink, "custom",
            "I moved to Burlington last spring and I really like it here so far", true).await;

        assert!(!reply_text(&reply).contains("MEMORY_EXTRACT"),
            "the marker line was shown to the user: {}", reply_text(&reply));
        let stored = bench.wait_for_memory_containing("Burlington").await;
        assert!(stored.iter().any(|m| m.contains("Burlington")),
            "the marked fact was not stored; memories: {:?}", stored);
    });
}

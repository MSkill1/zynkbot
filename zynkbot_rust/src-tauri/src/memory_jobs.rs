//! The memory job queue (migration 0023, 2026-10-24).
//!
//! After a reply, the memory work for a message (call 2, the facts; call 3, their links)
//! is a row here, and one worker runs the rows in order when the model is idle. Until
//! today each message spawned its own task; on a phone a message's work takes minutes,
//! the tasks raced each other and the user's next question, and a task gave up after
//! three interruptions, losing the facts. A row is never lost: an interruption leaves it
//! pending, a real failure is retried a few times, and rows survive an app restart.

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Everything the background work needs, as it is stored in the row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: i64,
    pub session_id: String,
    pub user_id: String,
    pub message: String,
    pub backend: String,
    pub source_type: String,
    pub containment_mode: String,
    pub explicit_remember: bool,
    pub remember_text: Option<String>,
    pub user_name: Option<String>,
    pub recent_turns: Option<String>,
    pub query_entities: Vec<String>,
    pub attempts: i64,
}

/// Real failures are retried this many times; an interruption does not count.
const MAX_ATTEMPTS: i64 = 4;

static WAKE: once_cell::sync::Lazy<tokio::sync::Notify> = once_cell::sync::Lazy::new(tokio::sync::Notify::new);
static RUNNING: AtomicBool = AtomicBool::new(false);

/// Queue the memory work for one message. Returns the row id.
pub async fn enqueue(pool: &SqlitePool, job: &Job) -> Result<i64, sqlx::Error> {
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO memory_jobs (session_id, user_id, message, backend, source_type, containment_mode,
            explicit_remember, remember_text, user_name, recent_turns, query_entities)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id")
        .bind(&job.session_id).bind(&job.user_id).bind(&job.message).bind(&job.backend)
        .bind(&job.source_type).bind(&job.containment_mode).bind(job.explicit_remember as i64)
        .bind(&job.remember_text).bind(&job.user_name).bind(&job.recent_turns)
        .bind(serde_json::to_string(&job.query_entities).unwrap_or_else(|_| "[]".into()))
        .fetch_one(pool).await?;
    WAKE.notify_one();
    Ok(id)
}

/// The oldest pending row, marked running. A row left "running" by a crash is picked up
/// again after five minutes.
pub async fn take_next(pool: &SqlitePool) -> Result<Option<Job>, sqlx::Error> {
    let row = sqlx::query_as::<_, (i64, String, String, String, String, String, String, i64, Option<String>, Option<String>, Option<String>, String, i64)>(
        "SELECT id, session_id, user_id, message, backend, source_type, containment_mode, explicit_remember,
                remember_text, user_name, recent_turns, query_entities, attempts
         FROM memory_jobs
         WHERE status = 'pending' OR (status = 'running' AND updated_at < datetime('now', '-5 minutes'))
         ORDER BY id LIMIT 1")
        .fetch_optional(pool).await?;
    let Some(r) = row else { return Ok(None) };
    sqlx::query("UPDATE memory_jobs SET status = 'running', updated_at = datetime('now') WHERE id = ?")
        .bind(r.0).execute(pool).await?;
    Ok(Some(Job {
        id: r.0, session_id: r.1, user_id: r.2, message: r.3, backend: r.4, source_type: r.5,
        containment_mode: r.6, explicit_remember: r.7 != 0, remember_text: r.8, user_name: r.9,
        recent_turns: r.10, query_entities: serde_json::from_str(&r.11).unwrap_or_default(), attempts: r.12,
    }))
}

pub async fn mark(pool: &SqlitePool, id: i64, status: &str, error: Option<&str>, count_attempt: bool) {
    let _ = sqlx::query("UPDATE memory_jobs SET status = ?, error = ?, attempts = attempts + ?, updated_at = datetime('now') WHERE id = ?")
        .bind(status).bind(error).bind(if count_attempt { 1 } else { 0 }).bind(id).execute(pool).await;
}

/// How many rows are waiting or running (the Memory Manager shows this, 2026-10-24).
pub async fn pending_count(pool: &SqlitePool) -> i64 {
    sqlx::query_scalar("SELECT COUNT(*) FROM memory_jobs WHERE status IN ('pending', 'running')")
        .fetch_one(pool).await.unwrap_or(0)
}

/// Start the worker if none is running in this process. The sink carries the Memory
/// Manager's progress events and the contradiction card; the first caller's sink is kept.
pub fn ensure_worker(sink: Arc<dyn crate::response_sink::ResponseSink>) {
    if RUNNING.swap(true, Ordering::SeqCst) { return; }
    // Tauri's runtime: the app's setup closure is not inside a tokio context, and in the
    // chat harness this keeps one worker alive across tests.
    tauri::async_runtime::spawn(async move {
        struct Done;
        impl Drop for Done { fn drop(&mut self) { RUNNING.store(false, Ordering::SeqCst); } }
        let _done = Done;
        run_worker(sink).await;
    });
}

/// The worker loop: one job at a time, oldest first, waking on `enqueue`.
async fn run_worker(sink: Arc<dyn crate::response_sink::ResponseSink>) {
    println!("[Memory jobs] worker started");
    // The app runs its migrations on its own first connection; the worker starts beside it
    // and asked for the table before it existed (Pixel, 17:19). Let the app go first.
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    loop {
        let pool = match sqlx::SqlitePool::connect(&crate::db::get_db_url()).await {
            Ok(p) => p,
            Err(e) => { println!("[Memory jobs] no database: {}", e); tokio::time::sleep(std::time::Duration::from_secs(5)).await; continue; }
        };
        let job = take_next(&pool).await;
        match job {
            Ok(Some(job)) => {
                let id = job.id;
                match process(&pool, job, sink.clone()).await {
                    Ok(()) => mark(&pool, id, "done", None, true).await,
                    Err(e) if e.contains("interrupted") => {
                        // The user is talking to the model; the row waits its turn.
                        println!("[Memory jobs] job {} waits for the model (user message in progress)", id);
                        mark(&pool, id, "pending", None, false).await;
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    }
                    Err(e) => {
                        let attempts: i64 = sqlx::query_scalar("SELECT attempts FROM memory_jobs WHERE id = ?").bind(id).fetch_one(&pool).await.unwrap_or(0);
                        let status = if attempts + 1 >= MAX_ATTEMPTS { "failed" } else { "pending" };
                        println!("[Memory jobs] job {} {}: {}", id, if status == "failed" { "failed for good" } else { "failed, will retry" }, e);
                        mark(&pool, id, status, Some(&e), true).await;
                        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                    }
                }
                pool.close().await;
            }
            Ok(None) => {
                pool.close().await;
                // Idle: wait for a new row, or look again in a while in case one was left running.
                let _ = tokio::time::timeout(std::time::Duration::from_secs(120), WAKE.notified()).await;
            }
            Err(e) => {
                println!("[Memory jobs] could not read the queue: {}", e);
                pool.close().await;
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        }
    }
}

/// One job: "Remember:" stores the words as said; otherwise call 2 lists the facts and
/// each one goes through the storage path (call 3 inside it).
async fn process(pool: &SqlitePool, job: Job, sink: Arc<dyn crate::response_sink::ResponseSink>) -> Result<(), String> {
    let _ = pool;
    let bg = crate::commands::chat::MemoryBg {
        bg_message: job.message.clone(),
        bg_user_id: job.user_id.clone(),
        bg_session_id: job.session_id.clone(),
        bg_forced_backend: job.backend.clone(),
        bg_source_type: if job.source_type == "hands_free" { "hands_free" } else { "conversation" },
        bg_containment_mode: job.containment_mode.clone(),
        bg_sink: sink,
        bg_is_explicit_remember: job.explicit_remember,
        bg_user_name: job.user_name.clone(),
        bg_recent_turns: job.recent_turns.clone(),
        query_entities: job.query_entities.clone(),
        bg_sibling_ids: Vec::new(),
    };
    if let Some(text) = job.remember_text.clone().filter(|t| !t.trim().is_empty()) {
        crate::commands::chat::run_memory_background(bg, crate::memory_prompts::ExtractedFact { fact: text, title: None, extras: Default::default() }).await;
        return Ok(());
    }
    let facts = crate::commands::chat::extract_facts(&bg).await?;
    if facts.is_empty() {
        println!("[Memory jobs] job {}: call 2 found nothing to remember", job.id);
        if let Ok(p) = sqlx::SqlitePool::connect(&crate::db::get_db_url()).await {
            crate::conversation_history::set_extraction_outcome(&p, &job.session_id, "model_declined").await;
            p.close().await;
        }
        return Ok(());
    }
    let mut stored: Vec<i32> = Vec::new();
    for fact in facts {
        let mut this = bg.clone();
        this.bg_sibling_ids = stored.clone();
        if let Some(id) = crate::commands::chat::run_memory_background(this, fact).await { stored.push(id); }
    }
    // Facts from one message are linked "related" to each other, no model call (Matt,
    // 2026-10-24): the flight, the wedding and the cat came from one breath, and three
    // unlinked nodes were a worse record of it than one memory had been.
    if stored.len() > 1 {
        if let Ok(p) = sqlx::SqlitePool::connect(&crate::db::get_db_url()).await {
            let mut made = 0;
            for i in 0..stored.len() {
                for j in (i + 1)..stored.len() {
                    if crate::memory::create_memory_link(&p, stored[j], stored[i], "related", 1.0, Some("said in the same message"), "same_message").await.is_ok() { made += 1; }
                }
            }
            println!("[Memory jobs] job {}: {} fact(s) linked to each other ({} link(s))", job.id, stored.len(), made);
            p.close().await;
        }
    }
    Ok(())
}

/// A sink for when there is no window to tell (tests, a headless start).
pub struct NullSink;
impl crate::response_sink::ResponseSink for NullSink {
    fn token(&self, _token: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn pool() -> SqlitePool {
        let p = SqlitePoolOptions::new().max_connections(1).connect("sqlite::memory:").await.unwrap();
        sqlx::migrate!("./migrations").run(&p).await.unwrap();
        p
    }
    fn job(msg: &str) -> Job {
        Job { id: 0, session_id: "s".into(), user_id: "u".into(), message: msg.into(), backend: "custom".into(),
            source_type: "conversation".into(), containment_mode: "sovereign".into(), explicit_remember: false,
            remember_text: None, user_name: Some("Matt".into()), recent_turns: None, query_entities: vec!["Louie".into()], attempts: 0 }
    }

    #[tokio::test]
    async fn rows_come_back_oldest_first_and_only_once() {
        let p = pool().await;
        enqueue(&p, &job("first")).await.unwrap();
        enqueue(&p, &job("second")).await.unwrap();
        let a = take_next(&p).await.unwrap().unwrap();
        assert_eq!(a.message, "first");
        assert_eq!(a.query_entities, vec!["Louie".to_string()]);
        let b = take_next(&p).await.unwrap().unwrap();
        assert_eq!(b.message, "second");
        assert!(take_next(&p).await.unwrap().is_none(), "a running row is not handed out twice");
        assert_eq!(pending_count(&p).await, 2);
        mark(&p, a.id, "done", None, true).await;
        mark(&p, b.id, "pending", None, false).await;
        assert_eq!(pending_count(&p).await, 1);
        let again = take_next(&p).await.unwrap().unwrap();
        assert_eq!(again.id, b.id, "a row put back to pending is picked up again");
    }
}

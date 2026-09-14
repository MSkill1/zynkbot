// conversation_history.rs — Persistent conversation log
//
// Every completed exchange (user message + assistant response) is written here
// after the response is dispatched. Runs in a background task — never blocks
// the message handler.
//
// HIPAA mode: logging is skipped entirely. Raw conversation text is more
// sensitive than extracted facts; no persistent record is appropriate.

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

// ============================================================================
// TYPES
// ============================================================================

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct ConversationSession {
    pub id: i32,
    pub session_id: String,
    pub user_id: String,
    pub title: Option<String>,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub last_active: chrono::DateTime<chrono::Utc>,
    pub message_count: i32,
    pub model_backend: Option<String>,
    pub containment_mode: Option<String>,
    /// Held at the top of the history list regardless of date.
    pub pinned: bool,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct ConversationMessage {
    pub id: i64,
    pub session_id: String,
    pub user_id: String,
    pub role: String,
    pub content: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub model_backend: Option<String>,
    pub containment_mode: Option<String>,
}

// ============================================================================
// TABLE SETUP — idempotent, safe to call on every startup
// ============================================================================

pub async fn ensure_tables(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    // Schema comes from the migrations. Repeat the timestamp normalisation from
    // migration 0002 on every start as a safety net: any writer that still omits the
    // timestamp columns gets SQLite's datetime('now') default (space format), which
    // would misorder the history list again.
    for sql in [
        "UPDATE conversation_sessions SET started_at  = replace(started_at,  ' ', 'T') || '+00:00' WHERE started_at  NOT LIKE '%T%'",
        "UPDATE conversation_sessions SET last_active = replace(last_active, ' ', 'T') || '+00:00' WHERE last_active NOT LIKE '%T%'",
        "UPDATE conversation_messages SET created_at  = replace(created_at,  ' ', 'T') || '+00:00' WHERE created_at  NOT LIKE '%T%'",
    ] {
        let fixed = sqlx::query(sql).execute(pool).await?.rows_affected();
        if fixed > 0 {
            println!("[ConvHistory] normalised {} timestamp rows", fixed);
        }
    }
    // Repair message counts that drifted: until 0.9.6-beta2 the first exchange of a
    // thread was never counted (the INSERT left the column at its default of 0 and
    // only the ON CONFLICT path added 2), so every thread read 2 short and a
    // one-exchange thread said "0 messages". Only rows that are wrong are touched.
    let recounted = sqlx::query(
        "UPDATE conversation_sessions SET message_count = (
             SELECT COUNT(*) FROM conversation_messages m WHERE m.session_id = conversation_sessions.session_id
         ) WHERE message_count <> (
             SELECT COUNT(*) FROM conversation_messages m WHERE m.session_id = conversation_sessions.session_id
         )",
    )
    .execute(pool)
    .await?
    .rows_affected();
    if recounted > 0 {
        println!("[ConvHistory] repaired message_count on {} threads", recounted);
    }
    println!("[ConvHistory] ✅ Tables ready");
    Ok(())
}

/// Give a thread a name of the user's choosing. An empty name clears the custom
/// title, which puts the automatic one (first message) back the next time a
/// message is logged.
pub async fn set_session_title(
    pool: &SqlitePool,
    session_id: &str,
    user_id: &str,
    title: &str,
) -> Result<bool, sqlx::Error> {
    let title: String = title.trim().chars().take(120).collect();
    let result = sqlx::query(
        "UPDATE conversation_sessions SET title = ? WHERE session_id = ? AND user_id = ?",
    )
    .bind(&title)
    .bind(session_id)
    .bind(user_id)
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// Make a thread visible in History the moment its first message is sent, rather
/// than after the reply has streamed and been logged. Until 0.9.6-beta2 a thread had
/// no row until `log_exchange` ran, so opening History while the first reply was
/// still coming showed nothing for it, and "Current thread" could not label it
/// (2026-09-14). Idempotent: an existing row is left exactly as it is;
/// `log_exchange` then updates it as before.
pub async fn open_session(
    pool: &SqlitePool,
    session_id: &str,
    user_id: &str,
    first_message: &str,
    model_backend: &str,
    containment_mode: &str,
) -> Result<bool, sqlx::Error> {
    let title_snippet: String = first_message.chars().take(60).collect();
    let result = sqlx::query(
        "INSERT OR IGNORE INTO conversation_sessions
             (session_id, user_id, title, started_at, last_active, message_count, model_backend, containment_mode)
         VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%S+00:00','now'), strftime('%Y-%m-%dT%H:%M:%S+00:00','now'), 0, ?, ?)",
    )
    .bind(session_id)
    .bind(user_id)
    .bind(&title_snippet)
    .bind(model_backend)
    .bind(containment_mode)
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// Pin or unpin a conversation; pinned ones are listed first.
pub async fn set_session_pinned(
    pool: &SqlitePool,
    session_id: &str,
    user_id: &str,
    pinned: bool,
) -> Result<bool, sqlx::Error> {
    let result = sqlx::query(
        "UPDATE conversation_sessions SET pinned = ? WHERE session_id = ? AND user_id = ?",
    )
    .bind(if pinned { 1 } else { 0 })
    .bind(session_id)
    .bind(user_id)
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

// ============================================================================
// WRITE — log a completed exchange
// ============================================================================

pub async fn log_exchange(
    pool: &SqlitePool,
    session_id: &str,
    user_id: &str,
    user_message: &str,
    assistant_message: &str,
    model_backend: &str,
    containment_mode: &str,
    name_thread: bool,
    input_mode: &str,
) -> Result<(), sqlx::Error> {
    // Auto-title: first 60 chars of the first message that is allowed to name the
    // thread (callers pass name_thread = false for turns that must not — fragments
    // and NO_QUERY turns never reach here; since 2026-09-08 real hands-free
    // exchanges may name a thread, otherwise a thread whose first questions were
    // spoken stayed untitled and unfindable). A thread named by nobody keeps an
    // empty title; the history panel shows a placeholder. The user can rename it.
    let title_snippet: String = if name_thread {
        user_message.chars().take(60).collect()
    } else {
        String::new()
    };

    // The row is normally created up front by open_session; a hands-free turn
    // creates it here. message_count is recomputed from the rows below after the
    // messages are stored, so it is exact even when a repeat is ignored. Before
    // 0.9.6-beta2 a fresh row was left at 0 and only the conflict path added 2, so
    // every thread's count read 2 short.
    sqlx::query(
        "INSERT INTO conversation_sessions
             (session_id, user_id, title, started_at, last_active, message_count, model_backend, containment_mode)
         VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%S+00:00','now'), strftime('%Y-%m-%dT%H:%M:%S+00:00','now'), 0, ?, ?)
         ON CONFLICT (session_id) DO UPDATE SET
             last_active   = strftime('%Y-%m-%dT%H:%M:%S+00:00','now'),
             model_backend = EXCLUDED.model_backend,
             title         = CASE WHEN coalesce(conversation_sessions.title, '') = ''
                                  THEN EXCLUDED.title ELSE conversation_sessions.title END",
    )
    .bind(session_id)
    .bind(user_id)
    .bind(&title_snippet)
    .bind(model_backend)
    .bind(containment_mode)
    .execute(pool)
    .await?;

    // INSERT OR IGNORE: the unique index from migration 0011 makes an exact repeat
    // of (session, role, second, content) a no-op instead of a second row.
    sqlx::query(
        "INSERT OR IGNORE INTO conversation_messages
             (session_id, user_id, role, content, model_backend, containment_mode, created_at, input_mode)
         VALUES (?, ?, 'user', ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%S+00:00','now'), ?)",
    )
    .bind(session_id)
    .bind(user_id)
    .bind(user_message)
    .bind(model_backend)
    .bind(containment_mode)
    .bind(input_mode)
    .execute(pool)
    .await?;

    sqlx::query(
        "INSERT OR IGNORE INTO conversation_messages
             (session_id, user_id, role, content, model_backend, containment_mode, created_at, input_mode)
         VALUES (?, ?, 'assistant', ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%S+00:00','now'), 'system')",
    )
    .bind(session_id)
    .bind(user_id)
    .bind(assistant_message)
    .bind(model_backend)
    .bind(containment_mode)
    .execute(pool)
    .await?;

    sqlx::query(
        "UPDATE conversation_sessions SET message_count = (
             SELECT COUNT(*) FROM conversation_messages m WHERE m.session_id = conversation_sessions.session_id
         ) WHERE session_id = ?",
    )
    .bind(session_id)
    .execute(pool)
    .await?;

    Ok(())
}

/// What happened to memory extraction for the latest exchange in a thread.
/// One of: gate_skipped, model_declined, duplicate, contradiction, stored, error.
/// Answers "did nothing memorable happen, or did extraction never run?" per session.
pub async fn set_extraction_outcome(pool: &SqlitePool, session_id: &str, outcome: &str) {
    let _ = sqlx::query(
        "UPDATE conversation_sessions SET last_extraction = ?, last_extraction_at = strftime('%Y-%m-%dT%H:%M:%S+00:00','now') WHERE session_id = ?",
    )
    .bind(outcome)
    .bind(session_id)
    .execute(pool)
    .await;
}

// ============================================================================
// READ — queries used by Tauri commands in lib.rs
// ============================================================================

pub async fn list_sessions(
    pool: &SqlitePool,
    user_id: &str,
    limit: i64,
    offset: i64,
) -> Result<Vec<ConversationSession>, sqlx::Error> {
    sqlx::query_as::<_, ConversationSession>(
        "SELECT id, session_id, user_id, title, started_at, last_active,
                message_count, model_backend, containment_mode, pinned
         FROM conversation_sessions
         WHERE user_id = ?
         ORDER BY pinned DESC, last_active DESC
         LIMIT ? OFFSET ?",
    )
    .bind(user_id)
    .bind(limit)
    .bind(offset)
    .fetch_all(pool)
    .await
}

pub async fn get_messages(
    pool: &SqlitePool,
    session_id: &str,
) -> Result<Vec<ConversationMessage>, sqlx::Error> {
    sqlx::query_as::<_, ConversationMessage>(
        "SELECT id, session_id, user_id, role, content, created_at,
                model_backend, containment_mode
         FROM conversation_messages
         WHERE session_id = ?
         ORDER BY created_at ASC",
    )
    .bind(session_id)
    .fetch_all(pool)
    .await
}

pub async fn search(
    pool: &SqlitePool,
    user_id: &str,
    query: &str,
    date_from: Option<&str>,
    date_to: Option<&str>,
) -> Result<Vec<ConversationSession>, sqlx::Error> {
    let date_from_val = date_from.unwrap_or("1970-01-01");
    // Append end-of-day so the full selected date is included, not just midnight.
    let date_to_val = date_to
        .map(|d| format!("{}T23:59:59Z", d))
        .unwrap_or_else(|| "2999-12-31T23:59:59Z".to_string());

    if query.trim().is_empty() {
        // Date-only filter — no text condition
        return sqlx::query_as::<_, ConversationSession>(
            "SELECT id, session_id, user_id, title, started_at, last_active,
                    message_count, model_backend, containment_mode, pinned
             FROM conversation_sessions
             WHERE user_id = ?
               AND last_active >= ?
               AND last_active <= ?
             ORDER BY pinned DESC, last_active DESC
             LIMIT 50",
        )
        .bind(user_id)
        .bind(date_from_val)
        .bind(&date_to_val)
        .fetch_all(pool)
        .await;
    }

    // LIKE for case-insensitive partial matching (SQLite LIKE is case-insensitive for ASCII).
    let pattern = format!("%{}%", query.trim());
    sqlx::query_as::<_, ConversationSession>(
        "SELECT DISTINCT s.id, s.session_id, s.user_id, s.title,
                s.started_at, s.last_active, s.message_count,
                s.model_backend, s.containment_mode, s.pinned
         FROM conversation_sessions s
         JOIN conversation_messages m ON m.session_id = s.session_id
         WHERE s.user_id = ?
           AND (m.content LIKE ? OR s.title LIKE ?)
           AND s.last_active >= ?
           AND s.last_active <= ?
         ORDER BY s.pinned DESC, s.last_active DESC
         LIMIT 50",
    )
    .bind(user_id)
    .bind(&pattern)
    .bind(&pattern)
    .bind(date_from_val)
    .bind(&date_to_val)
    .fetch_all(pool)
    .await
}

pub async fn delete_session(
    pool: &SqlitePool,
    session_id: &str,
    user_id: &str,
) -> Result<bool, sqlx::Error> {
    // user_id check prevents deleting another user's session
    let result = sqlx::query(
        "DELETE FROM conversation_sessions
         WHERE session_id = ? AND user_id = ?",
    )
    .bind(session_id)
    .bind(user_id)
    .execute(pool)
    .await?;

    Ok(result.rows_affected() > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn test_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .connect("sqlite::memory:")
            .await
            .expect("in-memory DB failed");
        sqlx::migrate!("./migrations").run(&pool).await.expect("migration failed");
        pool
    }

    async fn count(pool: &SqlitePool, sid: &str) -> i32 {
        sqlx::query_scalar("SELECT message_count FROM conversation_sessions WHERE session_id = ?")
            .bind(sid).fetch_one(pool).await.unwrap()
    }

    #[tokio::test]
    async fn thread_is_listed_from_the_first_message_and_counts_every_exchange() {
        let pool = test_pool().await;
        // The first message opens the thread before any reply exists.
        assert!(open_session(&pool, "s1", "u1", "my car is a 2019 Outback, what oil does it take", "anthropic", "guardian").await.unwrap());
        let listed = list_sessions(&pool, "u1", 10, 0).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].title.as_deref(), Some("my car is a 2019 Outback, what oil does it take"));
        assert_eq!(listed[0].message_count, 0);
        // Opening it again is a no-op.
        assert!(!open_session(&pool, "s1", "u1", "something else", "anthropic", "guardian").await.unwrap());

        log_exchange(&pool, "s1", "u1", "my car is a 2019 Outback, what oil does it take", "0W-20.", "anthropic", "guardian", true, "typed").await.unwrap();
        assert_eq!(count(&pool, "s1").await, 2, "first exchange must be counted");
        log_exchange(&pool, "s1", "u1", "and the filter?", "Subaru 15208AA170.", "anthropic", "guardian", true, "typed").await.unwrap();
        assert_eq!(count(&pool, "s1").await, 4);
        assert_eq!(get_messages(&pool, "s1").await.unwrap().len(), 4);
    }

    #[tokio::test]
    async fn a_thread_logged_without_open_session_still_counts_its_first_exchange() {
        let pool = test_pool().await;
        log_exchange(&pool, "s2", "u1", "hello", "hi", "anthropic", "guardian", true, "hands_free").await.unwrap();
        assert_eq!(count(&pool, "s2").await, 2);
    }

    #[tokio::test]
    async fn ensure_tables_repairs_drifted_counts_only() {
        let pool = test_pool().await;
        log_exchange(&pool, "s3", "u1", "q", "a", "anthropic", "guardian", true, "typed").await.unwrap();
        sqlx::query("UPDATE conversation_sessions SET message_count = 0 WHERE session_id = 's3'").execute(&pool).await.unwrap();
        ensure_tables(&pool).await.unwrap();
        assert_eq!(count(&pool, "s3").await, 2);
    }

    #[tokio::test]
    async fn rename_is_per_user_and_trimmed() {
        let pool = test_pool().await;
        open_session(&pool, "s4", "u1", "first words", "anthropic", "guardian").await.unwrap();
        assert!(set_session_title(&pool, "s4", "u1", "  Outback maintenance  ").await.unwrap());
        assert_eq!(list_sessions(&pool, "u1", 10, 0).await.unwrap()[0].title.as_deref(), Some("Outback maintenance"));
        // Another user cannot rename it.
        assert!(!set_session_title(&pool, "s4", "u2", "x").await.unwrap());
        // A later exchange never overwrites a name the user chose.
        log_exchange(&pool, "s4", "u1", "later question", "answer", "anthropic", "guardian", true, "typed").await.unwrap();
        assert_eq!(list_sessions(&pool, "u1", 10, 0).await.unwrap()[0].title.as_deref(), Some("Outback maintenance"));
        // Blank clears the custom name; the next exchange names it automatically again.
        assert!(set_session_title(&pool, "s4", "u1", "   ").await.unwrap());
        log_exchange(&pool, "s4", "u1", "fresh title source", "answer", "anthropic", "guardian", true, "typed").await.unwrap();
        assert_eq!(list_sessions(&pool, "u1", 10, 0).await.unwrap()[0].title.as_deref(), Some("fresh title source"));
    }
}

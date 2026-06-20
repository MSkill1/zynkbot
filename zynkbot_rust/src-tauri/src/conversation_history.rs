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

pub async fn ensure_tables(_pool: &SqlitePool) -> Result<(), sqlx::Error> {
    // Schema is created by the SQLite migration in db.rs — nothing to do here
    println!("[ConvHistory] ✅ Tables ready");
    Ok(())
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
) -> Result<(), sqlx::Error> {
    // Auto-title: first 60 chars of the first user message in this session.
    // ON CONFLICT leaves title unchanged if the session already exists.
    let title_snippet: String = user_message.chars().take(60).collect();

    sqlx::query(
        "INSERT INTO conversation_sessions
             (session_id, user_id, title, model_backend, containment_mode)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT (session_id) DO UPDATE SET
             last_active   = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
             message_count = conversation_sessions.message_count + 2,
             model_backend = EXCLUDED.model_backend",
    )
    .bind(session_id)
    .bind(user_id)
    .bind(&title_snippet)
    .bind(model_backend)
    .bind(containment_mode)
    .execute(pool)
    .await?;

    sqlx::query(
        "INSERT INTO conversation_messages
             (session_id, user_id, role, content, model_backend, containment_mode)
         VALUES (?, ?, 'user', ?, ?, ?)",
    )
    .bind(session_id)
    .bind(user_id)
    .bind(user_message)
    .bind(model_backend)
    .bind(containment_mode)
    .execute(pool)
    .await?;

    sqlx::query(
        "INSERT INTO conversation_messages
             (session_id, user_id, role, content, model_backend, containment_mode)
         VALUES (?, ?, 'assistant', ?, ?, ?)",
    )
    .bind(session_id)
    .bind(user_id)
    .bind(assistant_message)
    .bind(model_backend)
    .bind(containment_mode)
    .execute(pool)
    .await?;

    Ok(())
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
                message_count, model_backend, containment_mode
         FROM conversation_sessions
         WHERE user_id = ?
         ORDER BY last_active DESC
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
                    message_count, model_backend, containment_mode
             FROM conversation_sessions
             WHERE user_id = ?
               AND last_active >= ?
               AND last_active <= ?
             ORDER BY last_active DESC
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
                s.model_backend, s.containment_mode
         FROM conversation_sessions s
         JOIN conversation_messages m ON m.session_id = s.session_id
         WHERE s.user_id = ?
           AND (m.content LIKE ? OR s.title LIKE ?)
           AND s.last_active >= ?
           AND s.last_active <= ?
         ORDER BY s.last_active DESC
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

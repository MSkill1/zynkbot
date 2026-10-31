use crate::conversation_history;
use crate::conversation_engine::ConversationEngine;

#[tauri::command]
pub async fn list_conversation_sessions(
    user_id: String,
    limit: Option<i64>,
    offset: Option<i64>,
) -> Result<Vec<conversation_history::ConversationSession>, String> {
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e: sqlx::Error| e.to_string())?;
    conversation_history::list_sessions(&pool, &user_id, limit.unwrap_or(50), offset.unwrap_or(0))
        .await
        .map_err(|e: sqlx::Error| e.to_string())
}

/// The thread any of the user's devices wrote to most recently (2026-10-28, single shared
/// context): the one every device lands in when it is brought back to the front. Threads
/// with no messages (a started-and-abandoned "New") are not counted.
#[tauri::command]
pub async fn get_latest_thread(user_id: String) -> Result<Option<serde_json::Value>, String> {
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url).await.map_err(|e: sqlx::Error| e.to_string())?;
    let row: Option<(String, String, i64)> = sqlx::query_as(
        "SELECT session_id, last_active, message_count FROM conversation_sessions
         WHERE user_id = ? AND message_count > 0 ORDER BY last_active DESC LIMIT 1")
        .bind(&user_id).fetch_optional(&pool).await.map_err(|e| e.to_string())?;
    pool.close().await;
    Ok(row.map(|(session_id, last_active, message_count)| serde_json::json!({
        "session_id": session_id, "last_active": last_active, "message_count": message_count })))
}

#[tauri::command]
pub async fn get_conversation_messages(
    session_id: String,
) -> Result<Vec<conversation_history::ConversationMessage>, String> {
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e: sqlx::Error| e.to_string())?;
    conversation_history::get_messages(&pool, &session_id)
        .await
        .map_err(|e: sqlx::Error| e.to_string())
}

#[tauri::command]
pub async fn search_conversations(
    user_id: String,
    query: String,
    date_from: Option<String>,
    date_to: Option<String>,
) -> Result<Vec<conversation_history::ConversationSession>, String> {
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e: sqlx::Error| e.to_string())?;
    conversation_history::search(
        &pool,
        &user_id,
        &query,
        date_from.as_deref(),
        date_to.as_deref(),
    )
    .await
    .map_err(|e: sqlx::Error| e.to_string())
}

/// Pin a conversation to the top of History (or unpin it).
#[tauri::command]
pub async fn set_session_pinned(
    session_id: String,
    user_id: String,
    pinned: bool,
) -> Result<bool, String> {
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e: sqlx::Error| e.to_string())?;
    conversation_history::set_session_pinned(&pool, &session_id, &user_id, pinned)
        .await
        .map_err(|e: sqlx::Error| e.to_string())
}

/// Rename a conversation in History. An empty title clears the custom name.
#[tauri::command]
pub async fn set_session_title(
    session_id: String,
    user_id: String,
    title: String,
) -> Result<bool, String> {
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e: sqlx::Error| e.to_string())?;
    conversation_history::set_session_title(&pool, &session_id, &user_id, &title)
        .await
        .map_err(|e: sqlx::Error| e.to_string())
}

#[tauri::command]
pub async fn delete_conversation_session(
    session_id: String,
    user_id: String,
) -> Result<bool, String> {
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e: sqlx::Error| e.to_string())?;
    conversation_history::delete_session(&pool, &session_id, &user_id)
        .await
        .map_err(|e: sqlx::Error| e.to_string())
}

#[tauri::command]
pub async fn store_message_feedback(
    message_id: String,
    session_id: String,
    user_id: String,
    rating: i16,
    model_backend: Option<String>,
    containment_mode: Option<String>,
) -> Result<(), String> {
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e: sqlx::Error| e.to_string())?;
    sqlx::query(
        "INSERT INTO message_feedback (message_id, session_id, user_id, rating, model_backend, containment_mode)
         VALUES (?, ?, ?, ?, ?, ?)
         ON CONFLICT (message_id, user_id) DO UPDATE SET rating = EXCLUDED.rating"
    )
    .bind(&message_id)
    .bind(&session_id)
    .bind(&user_id)
    .bind(rating)
    .bind(&model_backend)
    .bind(&containment_mode)
    .execute(&pool)
    .await
    .map_err(|e: sqlx::Error| e.to_string())?;
    Ok(())
}

/// Build a conversation prompt from input, history, and recalled memories
#[tauri::command]
pub async fn build_conversation_prompt(
    user_input: String,
    conversation_history: Option<Vec<crate::conversation_engine::ConversationTurn>>,
    recalled_memories: Option<Vec<crate::conversation_engine::Memory>>,
    backend: String,
) -> Result<String, String> {
    println!("[Rust ConversationEngine] Building conversation prompt...");

    let engine = ConversationEngine::new();
    let is_api = ConversationEngine::is_api_model(&backend);

    let prompt = engine.build_prompt(
        &user_input,
        conversation_history.as_deref(),
        recalled_memories.as_deref(),
        is_api,
        None,
    );

    Ok(prompt)
}

#[tauri::command]
pub async fn clear_conversation_history(user_id: String) -> Result<serde_json::Value, String> {
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    let result = sqlx::query("DELETE FROM conversation_sessions WHERE user_id = ?")
        .bind(&user_id)
        .execute(&pool)
        .await
        .map_err(|e| format!("Failed to clear conversation history: {}", e))?;

    let deleted = result.rows_affected();
    println!("[ConvHistory] Cleared {} sessions for user {}", deleted, &user_id[..8.min(user_id.len())]);

    Ok(serde_json::json!({ "deleted_count": deleted }))
}

// ============================================================================
// A SEAL THE USER MAKES (2026-10-27): one chosen thread as a signed file. See seal.rs.
// ============================================================================

/// Seal one thread: writes `<title>-<date>.zynkseal.json` to the seals folder and returns
/// where it went. Nothing is written to the database.
#[tauri::command]
pub async fn seal_thread(session_id: String, user_id: String) -> Result<serde_json::Value, String> {
    let pool = crate::db::create_pool().await.map_err(|e| format!("Database error: {}", e))?;
    let session: Option<(Option<String>, chrono::DateTime<chrono::Utc>)> = sqlx::query_as(
        "SELECT title, started_at FROM conversation_sessions WHERE session_id = ? AND user_id = ?")
        .bind(&session_id).bind(&user_id)
        .fetch_optional(&pool).await.map_err(|e| format!("Database error: {}", e))?;
    let Some((title, started_at)) = session else {
        pool.close().await;
        return Err("No such thread".to_string());
    };
    let messages = conversation_history::get_messages(&pool, &session_id).await
        .map_err(|e| format!("Database error: {}", e))?;
    pool.close().await;
    if messages.is_empty() {
        return Err("The thread has no messages to seal".to_string());
    }
    let thread = crate::seal::SealedThread {
        session_id: session_id.clone(),
        title: title.unwrap_or_default(),
        started_at: started_at.to_rfc3339(),
        messages: messages.into_iter().map(|m| crate::seal::SealedMessage {
            role: m.role, at: m.created_at.to_rfc3339(), text: m.content,
        }).collect(),
    };
    let key = crate::seal::load_or_create_key(&crate::seal::key_path())?;
    let sealed_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let file = crate::seal::seal(thread, &key, &sealed_at);
    let dir = crate::seal::seals_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Cannot create {}: {}", dir.display(), e))?;
    let path = dir.join(crate::seal::file_name(&file.thread.title, &sealed_at));
    let json = serde_json::to_string_pretty(&file).map_err(|e| format!("Cannot write the seal: {}", e))?;
    std::fs::write(&path, json).map_err(|e| format!("Cannot write {}: {}", path.display(), e))?;
    println!("[Seal] thread {} sealed: {} messages, {}", session_id, file.thread.messages.len(), path.display());
    Ok(serde_json::json!({
        "path": path.to_string_lossy(),
        "sealed_at": sealed_at,
        "messages": file.thread.messages.len(),
        "signer": file.signer,
    }))
}

/// Check a seal file: does the text still match, and did the signer shown make it?
#[tauri::command]
pub async fn verify_seal_file(path: String) -> Result<crate::seal::Verdict, String> {
    let text = std::fs::read_to_string(&path).map_err(|e| format!("Cannot read {}: {}", path, e))?;
    let file: crate::seal::SealFile = serde_json::from_str(&text).map_err(|_| "Not a Zynkbot seal file".to_string())?;
    Ok(crate::seal::verify(&file))
}

/// This device's seal signer (public key, hex), so the user can recognise their own seals.
#[tauri::command]
pub async fn seal_signer_key() -> Result<String, String> {
    let key = crate::seal::load_or_create_key(&crate::seal::key_path())?;
    Ok(hex::encode(key.verifying_key().to_bytes()))
}

/// A searched answer that came back written by the model's own provider (2026-10-27)
/// goes into the thread the way a synthesised one did: "Web search: <query>" and the
/// answer. Without this the answer was on screen and nowhere else, and gone after a
/// restart. Not in HIPAA mode, which keeps no thread.
#[tauri::command]
pub async fn log_web_search_answer(session_id: String, user_id: String, query: String, answer: String, backend: String, containment_mode: String, sources: Option<serde_json::Value>) -> Result<(), String> {
    if containment_mode.eq_ignore_ascii_case("hipaa") { return Ok(()); }
    let pool = crate::db::create_pool().await.map_err(|e| format!("Database error: {}", e))?;
    let user_line = format!("Web search: \"{}\"", query);
    // The sources go on the row too (0028), so the list under the answer survives a restart.
    let result = conversation_history::log_exchange_with_sources(&pool, &session_id, &user_id, &user_line, &answer, &backend, &containment_mode, true, "typed", sources.as_ref()).await;
    pool.close().await;
    result.map_err(|e| format!("Could not log the searched answer: {}", e))
}

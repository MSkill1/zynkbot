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

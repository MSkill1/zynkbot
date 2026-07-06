use crate::zchat;
use crate::zynklink;
use crate::user_identity;

async fn get_pool() -> Result<sqlx::SqlitePool, String> {
    sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Database connection failed: {}", e))
}

#[tauri::command]
pub async fn zchat_send_message(
    to_device_id: String,
    message_text: String,
) -> Result<zchat::SendMessageResponse, String> {
    let pool = get_pool().await?;

    let device_id_str = user_identity::get_device_id()?;
    let user_id_str = user_identity::get_user_id()?;

    let device_id = uuid::Uuid::parse_str(&device_id_str)
        .map_err(|e| format!("Invalid device ID: {}", e))?;
    let user_id = uuid::Uuid::parse_str(&user_id_str)
        .map_err(|e| format!("Invalid user ID: {}", e))?;
    let to_device_uuid = uuid::Uuid::parse_str(&to_device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    let response = zchat::send_message(&pool, device_id, to_device_uuid, message_text, user_id).await?;

    let user_id_clone = user_id_str.clone();
    let device_id_clone = device_id_str.clone();
    let to_device_clone = to_device_id.clone();
    tokio::spawn(async move {
        let db_url = crate::db::get_db_url();
        let pool = match sqlx::SqlitePool::connect(&db_url).await {
            Ok(p) => p,
            Err(e) => {
                println!("[ZChat] Failed to connect to database for delivery: {}", e);
                return;
            }
        };
        match zynklink::deliver_zchat_to_peer(&pool, &user_id_clone, &device_id_clone, &to_device_clone).await {
            Ok(count) if count > 0 => println!("[ZChat] Delivered {} message(s) to ZynkLink peer", count),
            Ok(_) => println!("[ZChat] No messages to deliver"),
            Err(e) => println!("[ZChat] Failed to deliver (will retry on next send): {}", e),
        }
    });

    Ok(response)
}

/// Get chat messages with a specific device
#[tauri::command]
pub async fn zchat_get_messages(
    device_id: String,
    since: Option<String>,
) -> Result<zchat::GetMessagesResponse, String> {
    let pool = get_pool().await?;

    let current_device_id_str = user_identity::get_device_id()?;
    let current_device_id = uuid::Uuid::parse_str(&current_device_id_str)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    let device_uuid = uuid::Uuid::parse_str(&device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    zchat::get_messages(&pool, current_device_id, device_uuid, since).await
}

/// Receive chat messages from remote device (called during sync)
#[tauri::command]
pub async fn zchat_deliver_messages(
    messages: Vec<zchat::DeliverMessageData>,
) -> Result<zchat::DeliverMessagesResponse, String> {
    let pool = get_pool().await?;
    zchat::deliver_messages(&pool, messages).await
}

/// Mark messages as delivered
#[tauri::command]
pub async fn zchat_mark_delivered(message_ids: Vec<String>) -> Result<usize, String> {
    let pool = get_pool().await?;

    let uuids: Result<Vec<uuid::Uuid>, _> = message_ids
        .iter()
        .map(|id| uuid::Uuid::parse_str(id))
        .collect();

    let uuids = uuids.map_err(|e| format!("Invalid message ID: {}", e))?;
    zchat::mark_delivered(&pool, uuids).await
}

/// Mark messages as read
#[tauri::command]
pub async fn zchat_mark_read(message_ids: Vec<String>) -> Result<usize, String> {
    let pool = get_pool().await?;

    let uuids: Result<Vec<uuid::Uuid>, _> = message_ids
        .iter()
        .map(|id| uuid::Uuid::parse_str(id))
        .collect();

    let uuids = uuids.map_err(|e| format!("Invalid message ID: {}", e))?;
    zchat::mark_read(&pool, uuids).await
}

/// Get unread message count from a specific device
#[tauri::command]
pub async fn zchat_get_unread_count(from_device_id: String) -> Result<i64, String> {
    let pool = get_pool().await?;

    let current_device_id_str = user_identity::get_device_id()?;
    let current_device_id = uuid::Uuid::parse_str(&current_device_id_str)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    let from_device_uuid = uuid::Uuid::parse_str(&from_device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    zchat::get_unread_count(&pool, current_device_id, from_device_uuid).await
}

/// Mark all unread messages from a device as read
#[tauri::command]
pub async fn zchat_mark_all_read_from_device(from_device_id: String) -> Result<usize, String> {
    let pool = get_pool().await?;

    let current_device_id_str = user_identity::get_device_id()?;
    let current_device_id = uuid::Uuid::parse_str(&current_device_id_str)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    let from_device_uuid = uuid::Uuid::parse_str(&from_device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    let message_ids = sqlx::query_scalar::<_, uuid::Uuid>(
        r#"
        SELECT id
        FROM zchat_messages
        WHERE to_device_id = ? AND from_device_id = ? AND read_at IS NULL
        "#,
    )
    .bind(current_device_id)
    .bind(from_device_uuid)
    .fetch_all(&pool)
    .await
    .map_err(|e| format!("Failed to fetch unread messages: {}", e))?;

    if message_ids.is_empty() {
        return Ok(0);
    }

    zchat::mark_read(&pool, message_ids).await
}

#[tauri::command]
pub async fn zchat_clear_history(with_device_id: String) -> Result<(), String> {
    let pool = get_pool().await?;
    let device_uuid = uuid::Uuid::parse_str(&with_device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;
    sqlx::query(
        "DELETE FROM zchat_messages WHERE from_device_id = ? OR to_device_id = ?"
    )
    .bind(device_uuid)
    .bind(device_uuid)
    .execute(&pool)
    .await
    .map_err(|e| format!("Failed to clear chat history: {}", e))?;
    Ok(())
}

/// Get undelivered messages to a specific device (for sync)
#[tauri::command]
pub async fn zchat_get_undelivered_messages(
    to_device_id: String,
) -> Result<Vec<zchat::DeliverMessageData>, String> {
    let pool = get_pool().await?;

    let from_device_id_str = user_identity::get_device_id()?;
    let from_device_id = uuid::Uuid::parse_str(&from_device_id_str)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    let to_device_uuid = uuid::Uuid::parse_str(&to_device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    zchat::get_undelivered_messages(&pool, from_device_id, to_device_uuid).await
}

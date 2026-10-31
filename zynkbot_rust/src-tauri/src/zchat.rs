// ZChat - Device-to-Device Messaging
// Enables direct chat between synced devices without cloud storage

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
#[allow(dead_code)]
pub struct ZChatMessage {
    pub id: Uuid,
    pub from_device_id: Uuid,
    pub to_device_id: Uuid,
    pub message_text: String,
    pub sent_at: DateTime<Utc>,
    pub delivered_at: Option<DateTime<Utc>>,
    pub read_at: Option<DateTime<Utc>>,
    pub user_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ZChatMessageResponse {
    pub id: String,
    pub from_device_id: String,
    pub to_device_id: String,
    pub message_text: String,
    pub sent_at: String,
    pub delivered_at: Option<String>,
    pub read_at: Option<String>,
    pub is_mine: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct SendMessageRequest {
    pub to_device_id: String,
    pub message_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct SendMessageResponse {
    pub success: bool,
    pub message_id: String,
    pub sent_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct GetMessagesResponse {
    pub messages: Vec<ZChatMessageResponse>,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct DeliverMessagesRequest {
    pub messages: Vec<DeliverMessageData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct DeliverMessageData {
    pub id: String,
    pub from_device_id: String,
    pub to_device_id: String,
    pub message_text: String,
    pub sent_at: String,
    pub user_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct DeliverMessagesResponse {
    pub success: bool,
    pub received_count: usize,
}

/// Send a chat message to a paired device
#[allow(dead_code)]
pub async fn send_message(
    pool: &SqlitePool,
    from_device_id: Uuid,
    to_device_id: Uuid,
    message_text: String,
    user_id: Uuid,
) -> Result<SendMessageResponse, String> {
    if message_text.trim().is_empty() {
        return Err("message_text cannot be empty".to_string());
    }

    // Generate ID here — SQLite has no gen_random_uuid() default
    let id = Uuid::new_v4();

    // Insert message into database
    let result = sqlx::query_as::<_, (Uuid, DateTime<Utc>)>(
        r#"
        INSERT INTO zchat_messages
        (id, from_device_id, to_device_id, message_text, user_id)
        VALUES (?, ?, ?, ?, ?)
        RETURNING id, sent_at
        "#,
    )
    .bind(id)
    .bind(from_device_id)
    .bind(to_device_id)
    .bind(message_text)
    .bind(user_id)
    .fetch_one(pool)
    .await
    .map_err(|e| format!("Failed to send message: {}", e))?;

    println!(
        "[ZChat] Message sent from {}... to {}...",
        &from_device_id.to_string()[..12],
        &to_device_id.to_string()[..12]
    );

    Ok(SendMessageResponse {
        success: true,
        message_id: result.0.to_string(),
        sent_at: result.1.to_rfc3339(),
    })
}

/// Get chat messages with a specific device
#[allow(dead_code)]
pub async fn get_messages(
    pool: &SqlitePool,
    current_device_id: Uuid,
    device_id: Uuid,
    since: Option<String>,
) -> Result<GetMessagesResponse, String> {
    let messages = if let Some(since_str) = since {
        // Parse timestamp
        let since_ts = DateTime::parse_from_rfc3339(&since_str)
            .map_err(|e| format!("Invalid timestamp format: {}", e))?
            .with_timezone(&Utc);

        // Get messages since timestamp
        sqlx::query_as::<_, ZChatMessage>(
            r#"
            SELECT id, from_device_id, to_device_id, message_text,
                   sent_at, delivered_at, read_at, user_id
            FROM zchat_messages
            WHERE ((from_device_id = ? AND to_device_id = ?)
                OR (from_device_id = ? AND to_device_id = ?))
              AND sent_at > ?
              AND id IS NOT NULL
            ORDER BY sent_at ASC
            LIMIT 100
            "#,
        )
        .bind(current_device_id)
        .bind(device_id)
        .bind(device_id)
        .bind(current_device_id)
        .bind(since_ts)
        .fetch_all(pool)
        .await
        .map_err(|e| format!("Failed to fetch messages: {}", e))?
    } else {
        // Get last 100 messages
        let mut msgs = sqlx::query_as::<_, ZChatMessage>(
            r#"
            SELECT id, from_device_id, to_device_id, message_text,
                   sent_at, delivered_at, read_at, user_id
            FROM zchat_messages
            WHERE ((from_device_id = ? AND to_device_id = ?)
               OR (from_device_id = ? AND to_device_id = ?))
              AND id IS NOT NULL
            ORDER BY sent_at DESC
            LIMIT 100
            "#,
        )
        .bind(current_device_id)
        .bind(device_id)
        .bind(device_id)
        .bind(current_device_id)
        .fetch_all(pool)
        .await
        .map_err(|e| format!("Failed to fetch messages: {}", e))?;

        // Reverse for chronological order (oldest first)
        msgs.reverse();
        msgs
    };

    // Convert to response format with is_mine flag
    let result_messages: Vec<ZChatMessageResponse> = messages
        .into_iter()
        .map(|msg| ZChatMessageResponse {
            id: msg.id.to_string(),
            from_device_id: msg.from_device_id.to_string(),
            to_device_id: msg.to_device_id.to_string(),
            message_text: msg.message_text,
            sent_at: msg.sent_at.to_rfc3339(),
            delivered_at: msg.delivered_at.map(|dt| dt.to_rfc3339()),
            read_at: msg.read_at.map(|dt| dt.to_rfc3339()),
            is_mine: msg.from_device_id == current_device_id,
        })
        .collect();

    let count = result_messages.len();

    Ok(GetMessagesResponse {
        messages: result_messages,
        count,
    })
}

/// Receive chat messages from remote device (called during sync)
#[allow(dead_code)]
pub async fn deliver_messages(
    pool: &SqlitePool,
    messages: Vec<DeliverMessageData>,
) -> Result<DeliverMessagesResponse, String> {
    if messages.is_empty() {
        return Ok(DeliverMessagesResponse {
            success: true,
            received_count: 0,
        });
    }

    let mut received_count = 0;

    for msg in messages {
        // Parse UUIDs
        let id = Uuid::parse_str(&msg.id).map_err(|e| format!("Invalid message id: {}", e))?;
        let from_device_id = Uuid::parse_str(&msg.from_device_id)
            .map_err(|e| format!("Invalid from_device_id: {}", e))?;
        let to_device_id = Uuid::parse_str(&msg.to_device_id)
            .map_err(|e| format!("Invalid to_device_id: {}", e))?;
        let user_id =
            Uuid::parse_str(&msg.user_id).map_err(|e| format!("Invalid user_id: {}", e))?;

        // Parse timestamp
        let sent_at = DateTime::parse_from_rfc3339(&msg.sent_at)
            .map_err(|e| format!("Invalid timestamp: {}", e))?
            .with_timezone(&Utc);

        // Insert message (skip if duplicate)
        let result = sqlx::query(
            r#"
            INSERT INTO zchat_messages
            (id, from_device_id, to_device_id, message_text, sent_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO NOTHING
            "#,
        )
        .bind(id)
        .bind(from_device_id)
        .bind(to_device_id)
        .bind(&msg.message_text)
        .bind(sent_at)
        .bind(user_id)
        .execute(pool)
        .await;

        match result {
            Ok(query_result) => {
                if query_result.rows_affected() > 0 {
                    received_count += 1;
                }
            }
            Err(e) => {
                println!("[ZChat] Failed to insert message {}: {}", msg.id, e);
                continue;
            }
        }
    }

    println!("[ZChat] Received {} new message(s)", received_count);

    Ok(DeliverMessagesResponse {
        success: true,
        received_count,
    })
}

/// Mark messages as delivered
#[allow(dead_code)]
pub async fn mark_delivered(
    pool: &SqlitePool,
    message_ids: Vec<Uuid>,
) -> Result<usize, String> {
    let delivered_at = Utc::now();

    let result = if message_ids.is_empty() {
        return Ok(0);
    } else {
        let in_clause = message_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        let sql = format!(
            "UPDATE zchat_messages SET delivered_at = ? WHERE id IN ({}) AND delivered_at IS NULL",
            in_clause
        );
        let mut q = sqlx::query(&sql).bind(delivered_at);
        // IDs are stored as Uuid (binary BLOB) — see the INSERT in send_message
        // at line 95-113. Binding as id.to_string() (hex) would never match.
        for id in &message_ids { q = q.bind(*id); }
        q.execute(pool).await.map_err(|e| format!("Failed to mark messages as delivered: {}", e))?
    };

    Ok(result.rows_affected() as usize)
}

/// Mark messages as read
#[allow(dead_code)]
pub async fn mark_read(
    pool: &SqlitePool,
    message_ids: Vec<Uuid>,
) -> Result<usize, String> {
    let read_at = Utc::now();

    let result = if message_ids.is_empty() {
        return Ok(0);
    } else {
        let in_clause = message_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        let sql = format!(
            "UPDATE zchat_messages SET read_at = ? WHERE id IN ({}) AND read_at IS NULL",
            in_clause
        );
        let mut q = sqlx::query(&sql).bind(read_at);
        // IDs stored as Uuid (binary BLOB) — id.to_string() would never match.
        for id in &message_ids { q = q.bind(*id); }
        q.execute(pool).await.map_err(|e| format!("Failed to mark messages as read: {}", e))?
    };

    Ok(result.rows_affected() as usize)
}

/// Get count of unread messages from a specific device
#[allow(dead_code)]
pub async fn get_unread_count(
    pool: &SqlitePool,
    current_device_id: Uuid,
    from_device_id: Uuid,
) -> Result<i64, String> {
    let count = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT COUNT(*)
        FROM zchat_messages
        WHERE to_device_id = ? AND from_device_id = ? AND read_at IS NULL
        "#,
    )
    .bind(current_device_id)
    .bind(from_device_id)
    .fetch_one(pool)
    .await
    .map_err(|e| format!("Failed to get unread count: {}", e))?;

    Ok(count)
}

/// Get undelivered messages to a specific device (for sync)
#[allow(dead_code)]
pub async fn get_undelivered_messages(
    pool: &SqlitePool,
    from_device_id: Uuid,
    to_device_id: Uuid,
) -> Result<Vec<DeliverMessageData>, String> {
    let messages = sqlx::query_as::<_, ZChatMessage>(
        r#"
        SELECT id, from_device_id, to_device_id, message_text,
               sent_at, delivered_at, read_at, user_id
        FROM zchat_messages
        WHERE from_device_id = ? AND to_device_id = ?
          AND delivered_at IS NULL
          AND id IS NOT NULL
        ORDER BY sent_at ASC
        LIMIT 100
        "#,
    )
    .bind(from_device_id)
    .bind(to_device_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to fetch undelivered messages: {}", e))?;

    let result: Vec<DeliverMessageData> = messages
        .into_iter()
        .map(|msg| DeliverMessageData {
            id: msg.id.to_string(),
            from_device_id: msg.from_device_id.to_string(),
            to_device_id: msg.to_device_id.to_string(),
            message_text: msg.message_text,
            sent_at: msg.sent_at.to_rfc3339(),
            user_id: msg.user_id.to_string(),
        })
        .collect();

    Ok(result)
}

// ============================================================================
// HTTP route and delivery — served by the transport (crate::transport), 2026-09-17.
// ZChat owns its messages; the transport provides the server, the pinned client
// and the device registry. Delivery goes to any device in the registry, which is
// what lets a user message their own devices as well as a linked user's.
// ============================================================================

use axum::{extract::State, routing::post, Json, Router};
use std::sync::Arc;
use std::time::Duration;
use crate::transport::Transport;

/// ZChat's route bundle: message delivery, verified peers only.
pub fn routes(transport: Arc<Transport>) -> Router {
    Router::new()
        .route("/api/zchat/deliver", post(handle_zchat_deliver))
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&transport),
            crate::transport::require_verified_device,
        ))
        .with_state(transport)
}

/// Axum handler for receiving chat messages from a peer device
async fn handle_zchat_deliver(
    State(transport): State<Arc<Transport>>,
    Json(messages): Json<Vec<DeliverMessageData>>,
) -> Result<Json<serde_json::Value>, String> {
    println!("[ZChat] Received {} messages from peer", messages.len());

    // Call zchat deliver_messages function
    let result = deliver_messages(&transport.db_pool, messages).await?;

    Ok(Json(serde_json::json!({
        "success": true,
        "delivered": result.received_count
    })))
}

/// Deliver undelivered ZChat messages to a specific peer device
pub async fn deliver_to_peer(transport: &Transport, to_device_id: &str) -> Result<usize, String> {
    // Query database for peer device IP address (similar to how ZynkLink does it)
    // Note: device_id column is TEXT, so bind as string
    let peer_info = sqlx::query_as::<_, (Option<String>,)>(
        "SELECT device_ip FROM zynk_devices WHERE device_id = ?"
    )
    .bind(to_device_id)
    .fetch_optional(&transport.db_pool)
    .await
    .map_err(|e| format!("Failed to query peer device: {}", e))?
    .ok_or_else(|| format!("Peer device {} not found in database", to_device_id))?;

    // Get current device ID and parse for zchat functions
    let from_device_id = uuid::Uuid::parse_str(&transport.device_id())
        .map_err(|e| format!("Invalid device ID: {}", e))?;
    let to_device_uuid = uuid::Uuid::parse_str(to_device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    let device_ip = peer_info.0
        .ok_or_else(|| format!("No IP address found for device {}", to_device_id))?;

    // Get undelivered messages
    let messages = get_undelivered_messages(&transport.db_pool, from_device_id, to_device_uuid).await?;

    if messages.is_empty() {
        return Ok(0);
    }

    println!("[ZChat] Delivering {} messages to {}...", messages.len(), &to_device_id[..8]);

    // Extract message IDs for marking as delivered
    let message_ids: Vec<uuid::Uuid> = messages.iter()
        .filter_map(|m| uuid::Uuid::parse_str(&m.id).ok())
        .collect();

    // Send messages to peer
    let endpoint = format!("https://{}:{}/api/zchat/deliver", device_ip, transport.peer_port_by_ip(&device_ip).await);
    let client = transport.http_client.read().await.clone();
    let response = client
        .post(&endpoint)
        .json(&messages)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("Failed to deliver messages: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(format!("Message delivery failed with status {}: {}", status, error_text));
    }

    // Mark messages as delivered
    if !message_ids.is_empty() {
        mark_delivered(&transport.db_pool, message_ids).await?;
    }

    println!("[ZChat] ✓ Delivered {} messages to {}...", messages.len(), &to_device_id[..8]);

    Ok(messages.len())
}

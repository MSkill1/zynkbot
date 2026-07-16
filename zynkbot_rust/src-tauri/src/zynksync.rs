/// ZynkSync - Device-to-Device Memory Synchronization
///
/// Pure Rust implementation providing automatic memory synchronization across devices
///
/// Features:
/// - Manual device management (add devices by IP:port)
/// - Automatic sync loop (configurable interval)
/// - Conflict resolution (last-write-wins by timestamp)
/// - Background async processing with tokio
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, interval};
use sqlx::{SqlitePool, Row};
use chrono::{DateTime, Utc};
use reqwest::Client as HttpClient;
use tokio_rustls::TlsAcceptor;
use tokio::net::TcpListener;
use hyper_util::rt::{TokioIo, TokioExecutor};
use hyper_util::server::conn::auto::Builder as HyperConnBuilder;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use axum::{
    routing::post,
    Router,
    Json,
    extract::{State, ConnectInfo},
    http::StatusCode,
};
use std::net::SocketAddr;
use sha2::{Sha256, Digest};
use crate::zchat;
use crate::user_identity;
use tauri::Emitter;

/// Represents a peer device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerDevice {
    pub device_id: String,
    pub device_name: String,
    pub host: String,
    pub port: u16,
    pub url: String,
    pub last_seen: DateTime<Utc>,
    pub paired: bool,  // Whether this device is authorized to sync
    pub pairing_code: Option<String>,  // 6-digit code for pairing
    pub user_id: Option<String>,  // Host's user_id (for identity sync during pairing)
}

/// Represents a memory record for synchronization
/// Contains ALL fields from the memories table for complete sync
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncMemory {
    pub id: i32,
    pub user_id: String,
    pub session_id: Option<String>,  // Nullable in database
    pub content: String,
    pub title: Option<String>,
    pub source_type: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
    pub parent_scroll_id: Option<i32>,
    pub chunk_index: Option<i32>,
    pub namespace: String,
    pub is_syncable: bool,
    pub is_shareable: Option<bool>,
    pub embedding: Option<Vec<f32>>,  // CRITICAL: Vector embedding for semantic search
    pub link_count: Option<i32>,
    pub is_ephemeral: Option<bool>,
    pub expires_at: Option<DateTime<Utc>>,
    pub sentiment_score: Option<f32>,
    pub sentiment_label: Option<String>,
    pub event_type: Option<String>,
    pub event_date: Option<DateTime<Utc>>,
    pub entities_detected: Option<serde_json::Value>,  // NER entities for hybrid search
    #[serde(default)]
    pub original_text: Option<String>,
    #[serde(default)]
    pub relationships: Vec<MemoryRelationship>,  // Relationships from memory_links
}

/// Represents a relationship between memories (from memory_links table)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRelationship {
    pub source_memory_id: i32,  // Original ID on source device
    pub target_memory_id: i32,  // Original ID on source device
    pub relation_type: String,  // 'supports', 'contradicts', etc.
    pub confidence: f32,
    pub notes: Option<String>,
    pub created_by: String,
}

/// Result of a synchronization operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResult {
    pub peer_device_id: String,
    pub peer_device_name: String,
    pub memories_sent: usize,
    pub memories_received: usize,
    pub conversations_sent: usize,
    pub conflicts_resolved: usize,
    pub success: bool,
    pub error: Option<String>,
}

/// Memory inventory for a user (used for bidirectional "active device wins" sync)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInventory {
    pub user_id: String,
    pub memory_ids: Vec<i32>,  // Complete list of memory IDs this device has
    pub content_hashes: Vec<String>,  // SHA256 hashes of memory content for portable comparison
    pub latest_activity: Option<DateTime<Utc>>,  // Most recent memory timestamp (to determine active device)
    pub memory_count: usize,
}

/// Request to get inventory from remote device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryRequest {
    pub user_id: String,
}

/// Conversation session payload for cross-device sync
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConversationSession {
    pub session_id: String,
    pub user_id: String,
    pub title: Option<String>,
    pub started_at: DateTime<Utc>,
    pub last_active: DateTime<Utc>,
    pub message_count: i32,
    pub model_backend: Option<String>,
    pub containment_mode: Option<String>,
}

/// Conversation message payload for cross-device sync
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConversationMessage {
    pub session_id: String,
    pub user_id: String,
    pub role: String,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub model_backend: Option<String>,
    pub containment_mode: Option<String>,
    pub entry_hash: Option<String>,
    pub prev_hash: Option<String>,
}

/// Combined payload sent over the wire for conversation sync
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSyncPayload {
    pub sessions: Vec<SyncConversationSession>,
    pub messages: Vec<SyncConversationMessage>,
}

/// Core ZynkSync service managing device synchronization
pub struct ZynkSyncService {
    /// Unique identifier for this device
    device_id: String,

    /// Human-readable device name
    device_name: String,

    /// SQLite connection pool
    db_pool: SqlitePool,

    /// HTTP client for peer communication (rebuilt after each new pairing to add pinned certs)
    http_client: Arc<RwLock<HttpClient>>,

    /// Peer devices (thread-safe)
    peers: Arc<RwLock<HashMap<String, PeerDevice>>>,

    /// Last sync timestamps per peer (to track incremental syncs)
    last_sync: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,

    /// Whether auto-sync is enabled
    auto_sync_enabled: Arc<RwLock<bool>>,

    /// Sync interval in seconds
    sync_interval_secs: u64,

    /// Port this device is listening on
    server_port: Arc<RwLock<Option<u16>>>,

    /// Shutdown signal for HTTP server
    shutdown_tx: Arc<RwLock<Option<tokio::sync::oneshot::Sender<()>>>>,

    /// Current pairing code for this device
    pairing_code: Arc<RwLock<Option<String>>>,

    /// Failed pairing attempts per client IP — invalidate code after 5 misses
    failed_pairing_attempts: Arc<RwLock<HashMap<String, u32>>>,

    /// This device's TLS certificate PEM (for HTTPS server)
    cert_pem: String,

    /// This device's TLS private key PEM (for HTTPS server)
    key_pem: String,

    /// This device's TLS certificate DER (sent to peers during pairing)
    cert_der: Vec<u8>,
}

impl ZynkSyncService {
    /// Create a new ZynkSync service instance
    pub fn new(
        device_id: String,
        device_name: String,
        db_pool: SqlitePool,
        sync_interval_secs: Option<u64>,
        cert_pem: String,
        key_pem: String,
        cert_der: Vec<u8>,
    ) -> Self {
        Self {
            device_id,
            device_name,
            db_pool,
            http_client: Arc::new(RwLock::new(HttpClient::new())),
            peers: Arc::new(RwLock::new(HashMap::new())),
            last_sync: Arc::new(RwLock::new(HashMap::new())),
            auto_sync_enabled: Arc::new(RwLock::new(false)),
            sync_interval_secs: sync_interval_secs.unwrap_or(300),
            server_port: Arc::new(RwLock::new(None)),
            shutdown_tx: Arc::new(RwLock::new(None)),
            pairing_code: Arc::new(RwLock::new(None)),
            failed_pairing_attempts: Arc::new(RwLock::new(HashMap::new())),
            cert_pem,
            key_pem,
            cert_der,
        }
    }

    /// Rebuild the shared HTTP client to trust all currently stored peer certificates.
    /// Called at startup (after load_devices) and after each new pairing.
    pub async fn rebuild_http_client(&self) -> Result<(), String> {
        let rows = sqlx::query(
            "SELECT tls_cert_der FROM zynk_devices WHERE tls_cert_der IS NOT NULL AND sync_paired = 1"
        )
        .fetch_all(&self.db_pool)
        .await
        .map_err(|e| format!("Failed to load peer certs: {}", e))?;

        let cert_count = rows.len();
        // Tag every outgoing request with our device ID so the remote can
        // reject it if we've been removed from their peer list.
        let mut default_headers = reqwest::header::HeaderMap::new();
        if let Ok(val) = reqwest::header::HeaderValue::from_str(&self.device_id) {
            default_headers.insert("x-device-id", val);
        }

        let mut pinned_ders: Vec<Vec<u8>> = Vec::new();
        for row in rows {
            let cert_der: Option<Vec<u8>> = row.try_get("tls_cert_der").ok().flatten();
            if let Some(der) = cert_der {
                println!("[TLS] Pinning peer cert ({} bytes)", der.len());
                pinned_ders.push(der);
            } else {
                println!("[TLS] Warning: peer row has NULL tls_cert_der");
            }
        }
        println!("[TLS] rebuild_http_client: {} peer cert(s) in DB, {} pinned", cert_count, pinned_ders.len());

        let tls_config = crate::tls::build_pinned_client_config(pinned_ders);
        let client = reqwest::ClientBuilder::new()
            .use_preconfigured_tls(tls_config)
            .timeout(std::time::Duration::from_secs(30))
            .default_headers(default_headers)
            .build()
            .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

        *self.http_client.write().await = client;
        println!("[TLS] HTTP client rebuilt with {} pinned peer certificates", cert_count);
        Ok(())
    }

    /// Return a clone of the shared DB pool for use by Tauri commands that need DB access
    pub fn get_db_pool(&self) -> SqlitePool {
        self.db_pool.clone()
    }

    pub async fn get_http_client(&self) -> reqwest::Client {
        self.http_client.read().await.clone()
    }

    /// Generate a new 6-digit pairing code
    pub async fn generate_pairing_code(&self) -> Result<String, String> {
        // Use rand::random() instead of thread_rng() for Send compatibility
        let code = format!("{:06}", rand::random::<u32>() % 1000000);

        // Store in memory
        {
            let mut pairing_code = self.pairing_code.write().await;
            *pairing_code = Some(code.clone());
        }

        // Store in database with 10-minute expiration
        let expires_at = Utc::now() + chrono::Duration::minutes(10);
        sqlx::query(
            "INSERT INTO zynk_devices (device_id, device_name, pairing_code, pairing_code_expires_at, is_paired, port, created_at, last_seen_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT (device_id) DO UPDATE
             SET pairing_code = ?, pairing_code_expires_at = ?"
        )
        .bind(&self.device_id)
        .bind(&self.device_name)
        .bind(&code)
        .bind(expires_at)
        .bind(true)  // This device is always paired with itself (it's the host)
        .bind(57963i32)
        .bind(Utc::now())
        .bind(Utc::now())
        .bind(&code)        // ON CONFLICT SET pairing_code = ?
        .bind(expires_at)   // ON CONFLICT SET pairing_code_expires_at = ?
        .execute(&self.db_pool)
        .await
        .map_err(|e| format!("Failed to store pairing code: {}", e))?;

        println!("[ZynkSync] Generated pairing code: {} (expires in 10 minutes)", code);
        Ok(code)
    }

    /// Get the current pairing code (always generates a fresh one)
    pub async fn get_pairing_code(&self) -> Result<String, String> {
        // Always generate a fresh code with new 10-minute expiration
        // This ensures codes are never expired when shown to user
        self.generate_pairing_code().await
    }

    /// Load devices from database
    pub async fn load_devices(&self) -> Result<(), String> {
        let rows = sqlx::query(
            "SELECT device_id, device_name, device_ip, port, last_seen_at
             FROM zynk_devices
             WHERE sync_paired = 1
             ORDER BY last_seen_at DESC"
        )
        .fetch_all(&self.db_pool)
        .await
        .map_err(|e| format!("Failed to load devices: {}", e))?;

        let mut peers_map = self.peers.write().await;

        for row in rows {
            let device_id: String = row.get("device_id");

            // Skip self from peer list (don't sync with ourselves)
            if device_id == self.device_id {
                continue;
            }

            let device_name: String = row.get("device_name");
            let host: Option<String> = row.get("device_ip");
            let port: i32 = row.get("port");
            let last_seen: DateTime<Utc> = row.get("last_seen_at");

            // Skip devices without IP address
            let host = match host {
                Some(h) => h,
                None => continue,
            };

            let peer = PeerDevice {
                device_id: device_id.clone(),
                device_name,
                host: host.clone(),
                port: port as u16,
                url: format!("https://{}:{}", host, port),
                last_seen,
                paired: true,
                pairing_code: None,
                user_id: None,  // Not relevant when loading from database
            };

            peers_map.insert(device_id, peer);
        }

        println!("[ZynkSync] Loaded {} devices from database", peers_map.len());
        Ok(())
    }

    /// Add a device manually using IP address and pairing code
    /// Returns the peer device info including the host's user_id for identity sync
    pub async fn add_device(&self, host_ip: &str, pairing_code: &str) -> Result<PeerDevice, String> {
        // Validate pairing code format (6 digits)
        if pairing_code.len() != 6 || !pairing_code.chars().all(|c| c.is_numeric()) {
            return Err("Invalid pairing code. Must be 6 digits.".to_string());
        }

        let host = host_ip.to_string();
        let port: u16 = 57963;  // Fixed port for ZynkSync

        // Try to contact the device and verify pairing code
        let url = format!("https://{}:{}", host, port);
        let verify_endpoint = format!("{}/api/zynksync/verify-pairing", url);

        // Get client's user_id for validation
        let client_user_id = match user_identity::get_user_id() {
            Ok(uid) => {
                println!("[ZynkSync] Client user_id: {}", uid);
                Some(uid)
            }
            Err(e) => {
                println!("[ZynkSync] ⚠ Warning: Could not get client user_id: {}", e);
                None
            }
        };

        // Get client's memory count for smart security check
        let client_memory_count = if let Some(ref uid) = client_user_id {
            sqlx::query_scalar::<_, i64>(
                "SELECT COUNT(*) as count FROM memories WHERE user_id = ?"
            )
            .bind(uid)
            .fetch_one(&self.db_pool)
            .await
            .unwrap_or(0)
        } else {
            0
        };

        println!("[ZynkSync] Client has {} memories", client_memory_count);

        // Send pairing code for verification WITH our device info (timeout after 5 seconds)
        // NOTE: We don't send client_ip anymore - the server extracts it from the TCP connection
        println!("[ZynkSync] Sending pairing verification request to: {}", verify_endpoint);
        let mut request_body = serde_json::json!({
            "pairing_code": pairing_code,
            "client_device_id": self.device_id,
            "client_device_name": self.device_name,
            "client_memory_count": client_memory_count,
            "client_cert_der": BASE64.encode(&self.cert_der),
        });

        // Include client_user_id if available
        if let Some(ref uid) = client_user_id {
            request_body["client_user_id"] = serde_json::json!(uid);
        }

        // Use a TOFU client for the initial pairing request — the peer cert is not yet pinned.
        // The pairing code provides the authentication; the cert is pinned after this exchange.
        let tofu_client = reqwest::ClientBuilder::new()
            .danger_accept_invalid_certs(true)
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| format!("Failed to build pairing client: {}", e))?;

        let response = tofu_client
            .post(&verify_endpoint)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("Could not connect to device: {}", e))?;

        // Log response details
        let status = response.status();
        println!("[ZynkSync] Received response with status: {}", status);

        if !status.is_success() {
            // Try to read error response body for debugging
            let error_body = response.text().await.unwrap_or_else(|_| "Could not read error body".to_string());
            #[cfg(debug_assertions)]
            println!("[ZynkSync] Error response body: {}", error_body);
            return Err(format!("Pairing failed (status {}): {}", status, error_body));
        }

        // Get the raw response text first for debugging
        let response_text = response.text().await
            .map_err(|e| format!("Could not read response body: {}", e))?;

        #[cfg(debug_assertions)]
        println!("[ZynkSync] Received response body ({} bytes): {}", response_text.len(), response_text);

        // Try to parse it as JSON
        let device_info: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| format!("Invalid JSON response from device: {} | Response was: {}", e, response_text))?;

        #[cfg(debug_assertions)]
        println!("[ZynkSync] Successfully parsed response JSON: {}", serde_json::to_string_pretty(&device_info).unwrap_or_default());

        // Check for security warning in response
        if let Some(warning) = device_info.get("warning") {
            let warning_msg = warning.get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Identity change detected");
            let severity = warning.get("severity")
                .and_then(|s| s.as_str())
                .unwrap_or("medium");

            println!("[ZynkSync] ⚠️ PAIRING WARNING ({}): {}", severity, warning_msg);
            #[cfg(debug_assertions)]
            println!("[ZynkSync] Full warning: {}", serde_json::to_string_pretty(&warning).unwrap_or_default());

            // Emit Tauri event for frontend to display
            match crate::APP_HANDLE.lock() {
                Ok(app_handle_guard) => {
                    if let Some(app_handle) = app_handle_guard.as_ref() {
                        match app_handle.emit("zynksync://warning", warning.clone()) {
                            Ok(_) => println!("[ZynkSync] Warning event emitted to frontend"),
                            Err(e) => println!("[ZynkSync] Failed to emit warning event: {}", e),
                        }
                    }
                }
                Err(e) => println!("[ZynkSync] Failed to lock APP_HANDLE: {}", e),
            }

            // Log to help user understand what's happening
            println!("[ZynkSync] → This device will adopt the host's identity");
            println!("[ZynkSync] → Existing memories will be synced to the new identity");
        }

        // Extract device info from verification response
        let device_id = device_info.get("device_id")
            .and_then(|v| v.as_str())
            .ok_or("Device did not provide device_id")?
            .to_string();

        let device_name = device_info.get("device_name")
            .and_then(|v| v.as_str())
            .ok_or("Device did not provide device_name")?
            .to_string();

        // Extract host's user_id for identity sync (optional field)
        let host_user_id = device_info.get("user_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        if let Some(ref uid) = host_user_id {
            println!("[ZynkSync] Host user_id received: {}", uid);
            println!("[ZynkSync] ⚠ Identity sync will be handled by frontend with user confirmation");
        } else {
            println!("[ZynkSync] ⚠ Warning: Host did not provide user_id (identity won't be synced)");
        }

        // Check if device is already added
        {
            let peers_map = self.peers.read().await;
            if peers_map.contains_key(&device_id) {
                return Err("Device already added".to_string());
            }
        }

        // Extract the host's TLS cert DER from the pairing response (base64 encoded)
        let peer_cert_der: Option<Vec<u8>> = device_info
            .get("cert_der")
            .and_then(|v| v.as_str())
            .and_then(|b64| BASE64.decode(b64).ok());

        if let Some(ref der) = peer_cert_der {
            println!("[TLS] Received peer cert ({} bytes) — storing for pinning", der.len());
        } else {
            println!("[TLS] Warning: peer did not provide TLS certificate — connection will use TOFU");
        }

        // Store in database
        // IMPORTANT: Store the HOST's user_id (not the client's) for proper identity tracking
        sqlx::query(
            "INSERT INTO zynk_devices (device_id, device_name, device_ip, port, device_platform, is_paired, sync_paired, owner_user_id, tls_cert_der, last_seen_at, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT (device_id) DO UPDATE
             SET device_name = ?, device_ip = ?, port = ?, is_paired = ?, sync_paired = ?, owner_user_id = ?, tls_cert_der = ?, last_seen_at = ?"
        )
        .bind(&device_id)
        .bind(&device_name)
        .bind(&host)
        .bind(port as i32)
        .bind("")  // device_platform - not critical for manual addition
        .bind(true)  // is_paired
        .bind(true)  // sync_paired - this is a ZynkSync pairing
        .bind(host_user_id.as_deref())
        .bind(peer_cert_der.as_deref())
        .bind(Utc::now())
        .bind(Utc::now())
        // ON CONFLICT UPDATE values
        .bind(&device_name)
        .bind(&host)
        .bind(port as i32)
        .bind(true)
        .bind(true)  // sync_paired
        .bind(host_user_id.as_deref())
        .bind(peer_cert_der.as_deref())
        .bind(Utc::now())
        .execute(&self.db_pool)
        .await
        .map_err(|e| format!("Failed to save device: {}", e))?;

        // Rebuild the shared HTTP client to trust this peer's certificate going forward
        if let Err(e) = self.rebuild_http_client().await {
            println!("[TLS] Warning: could not rebuild HTTP client after pairing: {}", e);
        }

        // Add to peers map
        let peer = PeerDevice {
            device_id: device_id.clone(),
            device_name: device_name.clone(),
            host: host.clone(),
            port,
            url: url.clone(),
            last_seen: Utc::now(),
            paired: true,
            pairing_code: None,
            user_id: host_user_id,  // Include host's user_id for identity sync
        };

        {
            let mut peers_map = self.peers.write().await;
            peers_map.insert(device_id.clone(), peer.clone());
        }

        println!("[ZynkSync] Added device: {} ({})", device_name, device_id);
        Ok(peer)
    }

    /// Clear ZynkSync pairing data for a device. No peer notification — called directly
    /// by `handle_notify_unsynced` to avoid a round-trip loop, and by `remove_device()`.
    /// Preserves ZynkLink data (zynk_linked_directories, zynk_file_manifest, etc.).
    /// If `is_paired = 0` (no ZynkLink either) the device row is deleted entirely.
    /// If `is_paired = 1` (ZynkLink still active) the row is kept with sync_paired = 0.
    async fn clear_sync_data_db_only(&self, device_id: &str) -> Result<(), String> {
        println!("[ZynkSync] Clearing sync data for device: {}", &device_id[..device_id.len().min(8)]);

        let mut tx = self.db_pool.begin().await
            .map_err(|e| format!("Failed to start sync removal transaction: {}", e))?;

        // ZynkSync-specific tables only — do NOT touch ZynkLink tables.

        // zynk_sync_state — no FK to zynk_devices, CASCADE would never reach it
        sqlx::query("DELETE FROM zynk_sync_state WHERE source_device_id = ? OR target_device_id = ?")
            .bind(device_id).bind(device_id).execute(&mut *tx).await
            .map_err(|e| format!("Failed to delete sync state: {}", e))?;

        // zynk_device_pairings
        sqlx::query("DELETE FROM zynk_device_pairings WHERE device_a_id = ? OR device_b_id = ?")
            .bind(device_id).bind(device_id).execute(&mut *tx).await
            .map_err(|e| format!("Failed to delete device pairings: {}", e))?;

        // zynk_device_certificates — no FK to zynk_devices, CASCADE would never reach it
        sqlx::query("DELETE FROM zynk_device_certificates WHERE device_id = ?")
            .bind(device_id).execute(&mut *tx).await
            .map_err(|e| format!("Failed to delete device certificates: {}", e))?;

        // user_sync_codes — no FK to zynk_devices, CASCADE would never reach it
        sqlx::query("DELETE FROM user_sync_codes WHERE device_id = ?")
            .bind(device_id).execute(&mut *tx).await
            .map_err(|e| format!("Failed to delete user sync codes: {}", e))?;

        // Check if ZynkLink is still active for this device by querying zynklink_pairings
        // directly — do not read is_paired, which is a ZynkSync-managed column and would
        // couple the two independent trust systems.
        let has_link: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM zynklink_pairings
             WHERE (device1_id = ? OR device2_id = ?) AND is_active = 1"
        )
        .bind(device_id)
        .bind(device_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(|e| format!("Failed to check ZynkLink pairing: {}", e))?;

        if has_link > 0 {
            // ZynkLink still active — keep the device row, just clear sync fields
            sqlx::query("UPDATE zynk_devices SET sync_paired = 0, tls_cert_der = NULL WHERE device_id = ?")
                .bind(device_id).execute(&mut *tx).await
                .map_err(|e| format!("Failed to clear sync_paired: {}", e))?;
        } else {
            // No ZynkLink either — delete the device row entirely
            sqlx::query("DELETE FROM zynk_devices WHERE device_id = ?")
                .bind(device_id).execute(&mut *tx).await
                .map_err(|e| format!("Failed to remove device from database: {}", e))?;
        }

        tx.commit().await
            .map_err(|e| format!("Failed to commit sync data removal: {}", e))?;

        // Remove from in-memory peers map
        {
            let mut peers_map = self.peers.write().await;
            peers_map.remove(device_id);
        }

        println!("[ZynkSync] ✓ Cleared sync data for device {}", &device_id[..device_id.len().min(8)]);
        Ok(())
    }

    /// Clear ZynkLink pairing data for a device. Called by revoke_zynklink_pairing.
    /// Preserves ZynkSync pairing if sync_paired = 1.
    pub async fn clear_link_data(&self, device_id: &str) -> Result<(), String> {
        println!("[ZynkLink] Clearing link data for device: {}", &device_id[..device_id.len().min(8)]);

        let device_uuid = uuid::Uuid::parse_str(device_id)
            .map_err(|e| format!("Invalid device ID: {}", e))?;

        let mut tx = self.db_pool.begin().await
            .map_err(|e| format!("Failed to start link removal transaction: {}", e))?;

        // ZynkLink-specific tables

        // zynk_file_manifest — child of zynk_linked_directories
        sqlx::query("DELETE FROM zynk_file_manifest WHERE shared_directory_id IN (SELECT id FROM zynk_linked_directories WHERE device_id = ?)")
            .bind(device_id).execute(&mut *tx).await
            .map_err(|e| format!("Failed to delete file manifests: {}", e))?;

        // zynk_link_manifest — child of zynk_linked_directories
        sqlx::query("DELETE FROM zynk_link_manifest WHERE linked_directory_id IN (SELECT id FROM zynk_linked_directories WHERE device_id = ?)")
            .bind(device_id).execute(&mut *tx).await
            .map_err(|e| format!("Failed to delete link manifests: {}", e))?;

        // zynk_linked_directories
        sqlx::query("DELETE FROM zynk_linked_directories WHERE device_id = ?")
            .bind(device_id).execute(&mut *tx).await
            .map_err(|e| format!("Failed to delete linked directories: {}", e))?;

        // zynklink_codes
        sqlx::query("DELETE FROM zynklink_codes WHERE creator_device_id = ? OR accepted_by_device_id = ?")
            .bind(device_id).bind(device_id).execute(&mut *tx).await
            .map_err(|e| format!("Failed to delete ZynkLink codes: {}", e))?;

        // zynklink_pairings
        sqlx::query("DELETE FROM zynklink_pairings WHERE device1_id = ? OR device2_id = ?")
            .bind(device_id).bind(device_id).execute(&mut *tx).await
            .map_err(|e| format!("Failed to delete ZynkLink pairings: {}", e))?;

        // zchat_messages — UUID blob columns
        sqlx::query("DELETE FROM zchat_messages WHERE from_device_id = ? OR to_device_id = ?")
            .bind(device_uuid).bind(device_uuid).execute(&mut *tx).await
            .map_err(|e| format!("Failed to delete chat history: {}", e))?;

        // Check if sync pairing is still active
        let sync_paired: Option<i64> = sqlx::query_scalar(
            "SELECT sync_paired FROM zynk_devices WHERE device_id = ?"
        )
        .bind(device_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| format!("Failed to check sync_paired: {}", e))?
        .flatten();

        if sync_paired == Some(1) {
            // Sync is still active — just clear the link flag
            sqlx::query("UPDATE zynk_devices SET is_paired = 0 WHERE device_id = ?")
                .bind(device_id).execute(&mut *tx).await
                .map_err(|e| format!("Failed to clear is_paired: {}", e))?;
        } else {
            // No sync either — delete the device row entirely
            sqlx::query("DELETE FROM zynk_devices WHERE device_id = ?")
                .bind(device_id).execute(&mut *tx).await
                .map_err(|e| format!("Failed to delete device row: {}", e))?;
            // Remove from in-memory peers map (device is gone entirely)
            let mut peers_map = self.peers.write().await;
            peers_map.remove(device_id);
            // Must drop write lock before committing
            drop(peers_map);
        }

        tx.commit().await
            .map_err(|e| format!("Failed to commit link removal: {}", e))?;

        println!("[ZynkLink] ✓ Cleared link data for device {}", &device_id[..device_id.len().min(8)]);
        Ok(())
    }

    /// Remove a device and notify the peer to do the same (fire-and-forget).
    pub async fn remove_device(&self, device_id: &str) -> Result<(), String> {
        // Grab peer IP before the transaction deletes the zynk_devices row
        let peer_ip = sqlx::query_as::<_, (Option<String>,)>(
            "SELECT device_ip FROM zynk_devices WHERE device_id = ?"
        )
        .bind(device_id)
        .fetch_optional(&self.db_pool)
        .await
        .ok()
        .flatten()
        .and_then(|r| r.0);

        self.clear_sync_data_db_only(device_id).await?;

        // Best-effort: tell the peer to remove us from its list too.
        // Fire-and-forget — if they're offline the auth check blocks re-insertion anyway.
        if let Some(ip) = peer_ip {
            let local_device_id = self.device_id.clone();
            let http_client = self.http_client.read().await.clone();
            tokio::spawn(async move {
                let url = format!("https://{}:57963/api/zynksync/notify-unsynced", ip);
                let payload = serde_json::json!({ "removed_device_id": local_device_id });
                match http_client.post(&url).json(&payload).send().await {
                    Ok(_) => println!("[ZynkSync] ✓ Notified peer of unsync"),
                    Err(e) => println!("[ZynkSync] Note: could not notify peer of unsync (offline?): {}", e),
                }
            });
        }

        Ok(())
    }

    /// Check if this is the first sync between local device and peer
    /// Returns true if neither device has synced with the other before
    async fn is_first_sync(&self, peer_device_id: &str) -> Result<bool, String> {
        let local_device_id = &self.device_id;

        // Order device IDs consistently (smaller first) as per table constraint
        let (device_a, device_b) = if local_device_id.as_str() < peer_device_id {
            (local_device_id.as_str(), peer_device_id)
        } else {
            (peer_device_id, local_device_id.as_str())
        };

        let result = sqlx::query_as::<_, (Option<chrono::DateTime<chrono::Utc>>, Option<chrono::DateTime<chrono::Utc>>)>(
            "SELECT last_sync_a_to_b, last_sync_b_to_a
             FROM zynk_device_pairings
             WHERE device_a_id = ? AND device_b_id = ?"
        )
        .bind(device_a)
        .bind(device_b)
        .fetch_optional(&self.db_pool)
        .await
        .map_err(|e| format!("Failed to check sync history: {}", e))?;

        match result {
            None => {
                // No pairing record exists yet - definitely first sync
                Ok(true)
            }
            Some(record) => {
                // Check if both sync directions are NULL (never synced)
                Ok(record.0.is_none() && record.1.is_none())
            }
        }
    }

    /// Update sync timestamp after successful sync
    /// Records which direction the sync happened (local_is_active determines direction)
    async fn update_sync_timestamp(&self, peer_device_id: &str, local_is_active: bool) -> Result<(), String> {
        let local_device_id = &self.device_id;
        let now = chrono::Utc::now();

        // Order device IDs consistently (smaller first) as per table constraint
        let (device_a, device_b) = if local_device_id.as_str() < peer_device_id {
            (local_device_id.as_str(), peer_device_id)
        } else {
            (peer_device_id, local_device_id.as_str())
        };

        // Determine which timestamp to update based on sync direction
        // If local_is_active: local pushed to remote (a_to_b if local is a, b_to_a if local is b)
        let update_a_to_b = if local_device_id.as_str() < peer_device_id {
            // Local is device_a
            local_is_active
        } else {
            // Local is device_b
            !local_is_active
        };

        if update_a_to_b {
            sqlx::query(
                "INSERT INTO zynk_device_pairings (device_a_id, device_b_id, last_sync_a_to_b, paired_at)
                 VALUES (?, ?, ?, ?)
                 ON CONFLICT (device_a_id, device_b_id)
                 DO UPDATE SET last_sync_a_to_b = ?"
            )
            .bind(device_a)
            .bind(device_b)
            .bind(now)
            .bind(now)
            .bind(now)  // ON CONFLICT SET last_sync_a_to_b = ?
            .execute(&self.db_pool)
            .await
            .map_err(|e| format!("Failed to update sync timestamp: {}", e))?;
        } else {
            sqlx::query(
                "INSERT INTO zynk_device_pairings (device_a_id, device_b_id, last_sync_b_to_a, paired_at)
                 VALUES (?, ?, ?, ?)
                 ON CONFLICT (device_a_id, device_b_id)
                 DO UPDATE SET last_sync_b_to_a = ?"
            )
            .bind(device_a)
            .bind(device_b)
            .bind(now)
            .bind(now)
            .bind(now)  // ON CONFLICT SET last_sync_b_to_a = ?
            .execute(&self.db_pool)
            .await
            .map_err(|e| format!("Failed to update sync timestamp: {}", e))?;
        }

        Ok(())
    }

    /// Clear all paired devices (used during shutdown or reset)
    #[allow(dead_code)]
    pub async fn clear_all_devices(&self) -> Result<i64, String> {
        println!("[ZynkSync] Clearing all devices...");

        // Clear from database
        let result = sqlx::query("DELETE FROM zynk_devices")
            .execute(&self.db_pool)
            .await
            .map_err(|e| format!("Failed to clear devices: {}", e))?;

        let count = result.rows_affected() as i64;

        // Clear from peers map
        {
            let mut peers_map = self.peers.write().await;
            peers_map.clear();
        }

        println!("[ZynkSync] ✓ Cleared {} device(s) from database and memory", count);
        Ok(count)
    }

    /// Get local memory inventory for a user (for "active device wins" sync)
    async fn get_local_inventory(&self, user_id: &str) -> Result<MemoryInventory, String> {
        let rows = sqlx::query_as::<_, (i32, String, chrono::DateTime<chrono::Utc>, Option<chrono::DateTime<chrono::Utc>>)>(
            "SELECT id, content, created_at, updated_at
             FROM memories
             WHERE user_id = ? AND is_syncable = 1
             ORDER BY CASE WHEN COALESCE(updated_at, created_at) > created_at THEN COALESCE(updated_at, created_at) ELSE created_at END DESC"
        )
        .bind(user_id)
        .fetch_all(&self.db_pool)
        .await
        .map_err(|e| format!("Failed to get local inventory: {}", e))?;

        let memory_ids: Vec<i32> = rows.iter().map(|r| r.0).collect();

        // Compute content hashes for portable comparison across devices
        let content_hashes: Vec<String> = rows.iter().map(|r| {
            let mut hasher = Sha256::new();
            hasher.update(r.1.as_bytes());
            format!("{:x}", hasher.finalize())
        }).collect();

        // Use most recent timestamp (either created_at or updated_at)
        let latest_activity = rows.first().map(|r| {
            r.3.unwrap_or(r.2).max(r.2)
        });
        let memory_count = rows.len();


        Ok(MemoryInventory {
            user_id: user_id.to_string(),
            memory_ids,
            content_hashes,
            latest_activity,
            memory_count,
        })
    }

    /// Get memories modified since the last sync with a specific peer
    async fn get_modified_memories(
        &self,
        peer_id: &str,
        user_id: Option<&str>,
    ) -> Result<Vec<SyncMemory>, String> {
        let last_sync_time = {
            let last_sync_map = self.last_sync.read().await;
            last_sync_map.get(peer_id).copied()
        };

        let query = if let Some(last_sync) = last_sync_time {
            // Incremental sync: only memories modified since last sync
            sqlx::query(
                "SELECT id, user_id, session_id, content, title, source_type, created_at, updated_at,
                        parent_scroll_id, chunk_index, namespace, is_syncable, is_shareable,
                        embedding, link_count, is_ephemeral, expires_at, sentiment_score, sentiment_label,
                        event_type, event_date, entities_detected, original_text
                 FROM memories
                 WHERE is_syncable = 1
                   AND created_at > ?
                   AND (? IS NULL OR user_id = ?)
                 ORDER BY created_at ASC
                 LIMIT 1000"
            )
            .bind(last_sync)
            .bind(user_id)
            .bind(user_id)
        } else {
            // Full sync: all syncable memories
            sqlx::query(
                "SELECT id, user_id, session_id, content, title, source_type, created_at, updated_at,
                        parent_scroll_id, chunk_index, namespace, is_syncable, is_shareable,
                        embedding, link_count, is_ephemeral, expires_at, sentiment_score, sentiment_label,
                        event_type, event_date, entities_detected, original_text
                 FROM memories
                 WHERE is_syncable = 1
                   AND (? IS NULL OR user_id = ?)
                 ORDER BY created_at ASC
                 LIMIT 1000"
            )
            .bind(user_id)
            .bind(user_id)
        };

        let rows = query
            .fetch_all(&self.db_pool)
            .await
            .map_err(|e| format!("Database query failed: {}", e))?;

        let mut memories: Vec<SyncMemory> = rows
            .iter()
            .map(|row| {
                // Convert pgvector::Vector to Vec<f32>
                let embedding: Option<Vec<f32>> = row.try_get::<Option<Vec<u8>>, _>("embedding")
                    .ok()
                    .flatten()
                    .map(|blob| blob.chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect::<Vec<f32>>());

                SyncMemory {
                    id: row.get("id"),
                    user_id: row.get("user_id"),
                    session_id: row.get("session_id"),
                    content: row.get("content"),
                    title: row.get("title"),
                    source_type: row.get("source_type"),
                    created_at: row.get("created_at"),
                    updated_at: row.get("updated_at"),
                    parent_scroll_id: row.get("parent_scroll_id"),
                    chunk_index: row.get("chunk_index"),
                    namespace: row.get("namespace"),
                    is_syncable: row.get("is_syncable"),
                    is_shareable: row.get("is_shareable"),
                    embedding,
                    link_count: row.get("link_count"),
                    is_ephemeral: row.get("is_ephemeral"),
                    expires_at: row.get("expires_at"),
                    sentiment_score: row.get("sentiment_score"),
                    sentiment_label: row.get("sentiment_label"),
                    event_type: row.get("event_type"),
                    event_date: row.get("event_date"),
                    entities_detected: row.get("entities_detected"),
                    original_text: row.get("original_text"),
                    relationships: Vec::new(),  // Will be populated below
                }
            })
            .collect();

        // Fetch relationships for these memories
        if !memories.is_empty() {
            let memory_ids: Vec<i32> = memories.iter().map(|m| m.id).collect();

            let relationship_rows = {
                let in_clause = memory_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
                let sql = format!(
                    "SELECT source_memory_id, target_memory_id, relation_type, confidence, notes, created_by
                     FROM memory_links
                     WHERE source_memory_id IN ({}) OR target_memory_id IN ({})",
                    in_clause, in_clause
                );
                let mut q = sqlx::query(&sql);
                for id in &memory_ids { q = q.bind(id); }
                for id in &memory_ids { q = q.bind(id); }
                q.fetch_all(&self.db_pool)
                    .await
                    .map_err(|e| format!("Failed to fetch relationships: {}", e))?
            };

            // Collect all target memory IDs that aren't already in the batch
            let mut target_ids_to_fetch: std::collections::HashSet<i32> = std::collections::HashSet::new();
            for rel_row in &relationship_rows {
                let target_id: i32 = rel_row.get("target_memory_id");
                // If target memory not in current batch, we need to fetch it
                if !memories.iter().any(|m| m.id == target_id) {
                    target_ids_to_fetch.insert(target_id);
                }
            }

            // Fetch target memories that aren't in the batch yet
            if !target_ids_to_fetch.is_empty() {
                let target_ids_vec: Vec<i32> = target_ids_to_fetch.into_iter().collect();
                println!("[ZynkSync] Fetching {} target memories for relationships", target_ids_vec.len());

                let target_rows = {
                    let in_clause = target_ids_vec.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
                    let sql = format!(
                        "SELECT id, user_id, session_id, content, title, source_type, created_at, updated_at,
                                parent_scroll_id, chunk_index, namespace, is_syncable, is_shareable,
                                embedding, link_count, is_ephemeral, expires_at, sentiment_score, sentiment_label,
                                event_type, event_date, entities_detected, original_text
                         FROM memories WHERE id IN ({})",
                        in_clause
                    );
                    let mut q = sqlx::query(&sql);
                    for id in &target_ids_vec { q = q.bind(id); }
                    q.fetch_all(&self.db_pool)
                        .await
                        .map_err(|e| format!("Failed to fetch target memories: {}", e))?
                };

                // Add target memories to the batch
                for row in target_rows {
                    let embedding: Option<Vec<f32>> = row.try_get::<Option<Vec<u8>>, _>("embedding")
                        .ok()
                        .flatten()
                        .map(|blob| blob.chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect::<Vec<f32>>());

                    memories.push(SyncMemory {
                        id: row.get("id"),
                        user_id: row.get("user_id"),
                        session_id: row.get("session_id"),
                        content: row.get("content"),
                        title: row.get("title"),
                        source_type: row.get("source_type"),
                        created_at: row.get("created_at"),
                        updated_at: row.get("updated_at"),
                        parent_scroll_id: row.get("parent_scroll_id"),
                        chunk_index: row.get("chunk_index"),
                        namespace: row.get("namespace"),
                        is_syncable: row.get("is_syncable"),
                        is_shareable: row.get("is_shareable"),
                        embedding,
                        link_count: row.get("link_count"),
                        is_ephemeral: row.get("is_ephemeral"),
                        expires_at: row.get("expires_at"),
                        sentiment_score: row.get("sentiment_score"),
                        sentiment_label: row.get("sentiment_label"),
                        event_type: row.get("event_type"),
                        event_date: row.get("event_date"),
                        entities_detected: row.get("entities_detected"),
                        original_text: row.get("original_text"),
                        relationships: Vec::new(),  // Will be populated below
                    });
                }
            }

            // Group relationships by source memory
            for rel_row in relationship_rows {
                let source_id: i32 = rel_row.get("source_memory_id");
                let relationship = MemoryRelationship {
                    source_memory_id: source_id,
                    target_memory_id: rel_row.get("target_memory_id"),
                    relation_type: rel_row.get("relation_type"),
                    confidence: rel_row.get("confidence"),
                    notes: rel_row.get("notes"),
                    created_by: rel_row.get("created_by"),
                };

                // Add relationship to the source memory
                if let Some(memory) = memories.iter_mut().find(|m| m.id == source_id) {
                    memory.relationships.push(relationship);
                }
            }
        }

        Ok(memories)
    }

    /// Sync memories to a specific peer device
    pub async fn sync_to_peer(&self, peer_id: &str, user_id: Option<&str>) -> Result<SyncResult, String> {
        let peer = {
            let peers_map = self.peers.read().await;
            peers_map.get(peer_id).cloned()
                .ok_or_else(|| format!("Peer {} not found", peer_id))?
        };

        // Check if paired
        if !peer.paired {
            return Err(format!("Device {} is not paired. Enter pairing code first.", peer.device_name));
        }

        println!("[ZynkSync] Syncing to {} ({})", peer.device_name, peer.device_id);

        // Get modified memories
        let memories = self.get_modified_memories(peer_id, user_id).await?;

        if memories.is_empty() {
            println!("[ZynkSync] No new memories to sync");
            return Ok(SyncResult {
                peer_device_id: peer.device_id,
                peer_device_name: peer.device_name,
                memories_sent: 0,
                memories_received: 0,
                conversations_sent: 0,
                conflicts_resolved: 0,
                success: true,
                error: None,
            });
        }

        // Send memories to peer
        let endpoint = format!("{}/api/zynksync/receive", peer.url);
        let client = self.http_client.read().await.clone();
        let response = client
            .post(&endpoint)
            .json(&memories)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("Sync failed with status {}: {}", status, error_text));
        }

        // Update last sync timestamp
        {
            let mut last_sync_map = self.last_sync.write().await;
            last_sync_map.insert(peer.device_id.clone(), Utc::now());
        }

        println!("[ZynkSync] ✓ Sent {} memories to {}", memories.len(), peer.device_name);

        Ok(SyncResult {
            peer_device_id: peer.device_id,
            peer_device_name: peer.device_name,
            memories_sent: memories.len(),
            memories_received: 0,
            conversations_sent: 0,
            conflicts_resolved: 0,
            success: true,
            error: None,
        })
    }

    /// Deliver undelivered ZChat messages to a specific peer device
    pub async fn deliver_zchat_messages_to_peer(
        &self,
        to_device_id: &str,
    ) -> Result<usize, String> {
        // Query database for peer device IP address (similar to how ZynkLink does it)
        // Note: device_id column is TEXT, so bind as string
        let peer_info = sqlx::query_as::<_, (Option<String>,)>(
            "SELECT device_ip FROM zynk_devices WHERE device_id = ?"
        )
        .bind(to_device_id)
        .fetch_optional(&self.db_pool)
        .await
        .map_err(|e| format!("Failed to query peer device: {}", e))?
        .ok_or_else(|| format!("Peer device {} not found in database", to_device_id))?;

        // Get current device ID and parse for zchat functions
        let from_device_id = uuid::Uuid::parse_str(&self.device_id)
            .map_err(|e| format!("Invalid device ID: {}", e))?;
        let to_device_uuid = uuid::Uuid::parse_str(to_device_id)
            .map_err(|e| format!("Invalid device ID: {}", e))?;

        let device_ip = peer_info.0
            .ok_or_else(|| format!("No IP address found for device {}", to_device_id))?;

        // Get undelivered messages
        let messages = zchat::get_undelivered_messages(&self.db_pool, from_device_id, to_device_uuid).await?;

        if messages.is_empty() {
            return Ok(0);
        }

        println!("[ZChat] Delivering {} messages to {}...", messages.len(), &to_device_id[..8]);

        // Extract message IDs for marking as delivered
        let message_ids: Vec<uuid::Uuid> = messages.iter()
            .filter_map(|m| uuid::Uuid::parse_str(&m.id).ok())
            .collect();

        // Send messages to peer
        let endpoint = format!("https://{}:57963/api/zchat/deliver", device_ip);
        let client = self.http_client.read().await.clone();
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
            zchat::mark_delivered(&self.db_pool, message_ids).await?;
        }

        println!("[ZChat] ✓ Delivered {} messages to {}...", messages.len(), &to_device_id[..8]);

        Ok(messages.len())
    }

    /// Receive and store memories from a peer device
    /// IMPORTANT: Preserves original user_id so identities stay consistent across devices
    /// Also syncs relationships, mapping old memory IDs to new ones
    pub async fn receive_from_peer(&self, memories: Vec<SyncMemory>) -> Result<usize, String> {
        use std::collections::HashMap;

        let mut stored_count = 0;
        let mut id_mapping: HashMap<i32, i32> = HashMap::new();  // old_id -> new_id

        println!("[ZynkSync] Receiving {} memories with relationships (preserving original user_ids)", memories.len());

        // PHASE 1: Store/update all memories and build ID mapping
        for memory in &memories {
            // Check if memory already exists (by content AND user_id)
            let existing = sqlx::query_as::<_, (i32, chrono::DateTime<chrono::Utc>)>(
                "SELECT id, created_at FROM memories WHERE user_id = ? AND content = ?"
            )
            .bind(&memory.user_id)
            .bind(&memory.content)
            .fetch_optional(&self.db_pool)
            .await
            .map_err(|e| format!("Failed to check existing memory: {}", e))?;

            if let Some(existing_memory) = existing {
                // Memory exists - map old ID to existing ID
                id_mapping.insert(memory.id, existing_memory.0);

                // Update if incoming is newer
                if memory.created_at > existing_memory.1 {
                    println!("[ZynkSync] Updating existing memory (user: {}, newer timestamp)", memory.user_id);
                    sqlx::query(
                        "UPDATE memories
                         SET title = ?, namespace = ?, created_at = ?, session_id = ?
                         WHERE id = ?"
                    )
                    .bind(&memory.title)
                    .bind(&memory.namespace)
                    .bind(memory.created_at)
                    .bind(&memory.session_id)
                    .bind(existing_memory.0)
                    .execute(&self.db_pool)
                    .await
                    .map_err(|e| format!("Failed to update memory: {}", e))?;

                    stored_count += 1;
                }
            } else {
                // Memory doesn't exist - insert with ORIGINAL ID, user_id and ALL fields
                println!("[ZynkSync] Inserting new memory with ID: {} (user_id: {})", memory.id, memory.user_id);

                let embedding_vec: Option<Vec<u8>> = memory.embedding.as_ref().map(|v| v.iter().flat_map(|f| f.to_le_bytes()).collect());

                // Check if this ID is already used by a different memory
                let id_conflict = sqlx::query_scalar::<_, i64>(
                    "SELECT COUNT(*) FROM memories WHERE id = ?"
                )
                .bind(memory.id)
                .fetch_one(&self.db_pool)
                .await
                .map_err(|e| format!("Failed to check ID conflict: {}", e))?;

                if id_conflict > 0 {
                    // ID conflict - fetch existing memory and compare timestamps
                    let existing = sqlx::query_as::<_, (i32, String, chrono::DateTime<chrono::Utc>)>(
                        "SELECT id, content, created_at FROM memories WHERE id = ?"
                    )
                    .bind(memory.id)
                    .fetch_one(&self.db_pool)
                    .await
                    .map_err(|e| format!("Failed to fetch conflicting memory: {}", e))?;

                    if memory.created_at > existing.2 {
                        // Incoming memory is newer - replace existing
                        println!("[ZynkSync] ID conflict - incoming memory is NEWER, replacing existing ID {}", memory.id);

                        sqlx::query(
                            "UPDATE memories
                             SET user_id = ?, session_id = ?, content = ?, title = ?, source_type = ?,
                                 created_at = ?, updated_at = ?, parent_scroll_id = ?, chunk_index = ?,
                                 namespace = ?, is_syncable = ?, is_shareable = ?,
                                 embedding = ?, link_count = ?, is_ephemeral = ?, expires_at = ?,
                                 sentiment_score = ?, sentiment_label = ?, event_type = ?, event_date = ?,
                                 entities_detected = ?, original_text = ?
                             WHERE id = ?"
                        )
                        .bind(&memory.user_id)
                        .bind(&memory.session_id)
                        .bind(&memory.content)
                        .bind(&memory.title)
                        .bind(memory.source_type.as_deref())
                        .bind(memory.created_at)
                        .bind(memory.updated_at)
                        .bind(memory.parent_scroll_id)
                        .bind(memory.chunk_index)
                        .bind(&memory.namespace)
                        .bind(memory.is_syncable)
                        .bind(memory.is_shareable)
                        .bind(embedding_vec.as_deref())
                        .bind(memory.link_count)
                        .bind(memory.is_ephemeral)
                        .bind(memory.expires_at)
                        .bind(memory.sentiment_score)
                        .bind(memory.sentiment_label.as_deref())
                        .bind(memory.event_type.as_deref())
                        .bind(memory.event_date)
                        .bind(memory.entities_detected.as_ref())
                        .bind(memory.original_text.as_deref())
                        .bind(memory.id)
                        .execute(&self.db_pool)
                        .await
                        .map_err(|e| format!("Failed to update conflicting memory: {}", e))?;

                        id_mapping.insert(memory.id, memory.id);  // Same ID
                        stored_count += 1;
                    } else {
                        // Existing memory is newer or same - keep it, just map the ID
                        println!("[ZynkSync] ID conflict - existing memory is NEWER or SAME, keeping existing ID {}", memory.id);
                        id_mapping.insert(memory.id, existing.0);
                    }
                    continue;
                }

                // No conflict - insert with original ID
                sqlx::query(
                    "INSERT INTO memories (id, user_id, session_id, content, title, source_type, created_at, updated_at,
                                          parent_scroll_id, chunk_index, namespace, is_syncable, is_shareable,
                                          embedding, link_count, is_ephemeral, expires_at, sentiment_score, sentiment_label,
                                          event_type, event_date, entities_detected, original_text)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                .bind(memory.id)
                .bind(&memory.user_id)
                .bind(&memory.session_id)
                .bind(&memory.content)
                .bind(&memory.title)
                .bind(memory.source_type.as_deref())
                .bind(memory.created_at)
                .bind(memory.updated_at)
                .bind(memory.parent_scroll_id)
                .bind(memory.chunk_index)
                .bind(&memory.namespace)
                .bind(memory.is_syncable)
                .bind(memory.is_shareable)
                .bind(embedding_vec.as_deref())
                .bind(memory.link_count)
                .bind(memory.is_ephemeral)
                .bind(memory.expires_at)
                .bind(memory.sentiment_score)
                .bind(memory.sentiment_label.as_deref())
                .bind(memory.event_type.as_deref())
                .bind(memory.event_date)
                .bind(memory.entities_detected.as_ref())
                .bind(memory.original_text.as_deref())
                .execute(&self.db_pool)
                .await
                .map_err(|e| format!("Failed to insert memory: {}", e))?;

                // Map to SAME ID (we're syncing IDs now, not generating new ones)
                id_mapping.insert(memory.id, memory.id);
                stored_count += 1;
            }
        }

        // Always reset sequence to current max after any sync that touches explicit IDs.
        // This self-heals even when received=0 (all memories already existed as conflicts).
        // Without this, a device that only receives synced memories never advances its
        // sequence counter, causing duplicate key errors on the next local memory insert.
        if let Err(e) = sqlx::query(
            "SELECT 1" // SQLite uses AUTOINCREMENT — no sequence to reset
        )
        .execute(&self.db_pool)
        .await
        {
            eprintln!("[ZynkSync] Warning: Failed to reset memories sequence: {}", e);
        } else {
            println!("[ZynkSync] ✅ Sequence reset to current max ID to prevent future conflicts");
        }

        // PHASE 2: Sync relationships (IDs are now synced, so they should match)
        let mut relationships_created = 0;
        for memory in &memories {
            if memory.relationships.is_empty() {
                continue;
            }

            for rel in &memory.relationships {
                // With synced IDs, we use the original IDs directly
                // Check if both memories exist in local database
                let source_exists = sqlx::query_scalar::<_, i64>(
                    "SELECT COUNT(*) FROM memories WHERE id = ?"
                )
                .bind(rel.source_memory_id)
                .fetch_one(&self.db_pool)
                .await
                .map(|count| count > 0)
                .unwrap_or(false);

                let target_exists = sqlx::query_scalar::<_, i64>(
                    "SELECT COUNT(*) FROM memories WHERE id = ?"
                )
                .bind(rel.target_memory_id)
                .fetch_one(&self.db_pool)
                .await
                .map(|count| count > 0)
                .unwrap_or(false);

                if !source_exists {
                    eprintln!("[ZynkSync] Warning: Source memory ID {} not found locally - skipping relationship", rel.source_memory_id);
                    continue;
                }

                if !target_exists {
                    eprintln!("[ZynkSync] Warning: Target memory ID {} not found locally - will sync when target arrives", rel.target_memory_id);
                    continue;
                }

                let source_id = rel.source_memory_id;
                let target_id = rel.target_memory_id;

                // Insert relationship with synced IDs (ON CONFLICT DO NOTHING to handle duplicates)
                let result = sqlx::query(
                    "INSERT INTO memory_links (source_memory_id, target_memory_id, relation_type, confidence, notes, created_by)
                     VALUES (?, ?, ?, ?, ?, ?)
                     ON CONFLICT (source_memory_id, target_memory_id, relation_type) DO NOTHING"
                )
                .bind(source_id)
                .bind(target_id)
                .bind(&rel.relation_type)
                .bind(rel.confidence as f64)
                .bind(&rel.notes)
                .bind(&rel.created_by)
                .execute(&self.db_pool)
                .await;

                match result {
                    Ok(_) => relationships_created += 1,
                    Err(e) => eprintln!("[ZynkSync] Failed to create relationship: {}", e),
                }
            }
        }

        println!("[ZynkSync] ✓ Received and stored {} memories with {} relationships",
            stored_count, relationships_created);
        Ok(stored_count)
    }

    // -------------------------------------------------------------------------
    // Conversation history sync
    // -------------------------------------------------------------------------

    /// Fetch conversation sessions and messages modified since the last sync with this peer.
    /// On first sync (no last_sync entry) returns everything.
    async fn get_modified_conversations(
        &self,
        peer_id: &str,
        user_id: &str,
    ) -> Result<ConversationSyncPayload, String> {
        let last_sync_time = {
            let map = self.last_sync.read().await;
            map.get(peer_id).copied()
        };

        let session_rows = match last_sync_time {
            Some(since) => sqlx::query(
                "SELECT session_id, user_id, title, started_at, last_active, message_count,
                        model_backend, containment_mode
                 FROM conversation_sessions
                 WHERE user_id = ? AND last_active > ?
                 ORDER BY last_active ASC LIMIT 500"
            )
            .bind(user_id)
            .bind(since)
            .fetch_all(&self.db_pool)
            .await
            .map_err(|e| format!("Failed to fetch sessions: {}", e))?,

            None => sqlx::query(
                "SELECT session_id, user_id, title, started_at, last_active, message_count,
                        model_backend, containment_mode
                 FROM conversation_sessions
                 WHERE user_id = ?
                 ORDER BY last_active ASC LIMIT 500"
            )
            .bind(user_id)
            .fetch_all(&self.db_pool)
            .await
            .map_err(|e| format!("Failed to fetch sessions: {}", e))?,
        };

        let sessions = session_rows.iter().map(|row| SyncConversationSession {
            session_id: row.get("session_id"),
            user_id: row.get("user_id"),
            title: row.get("title"),
            started_at: row.get("started_at"),
            last_active: row.get("last_active"),
            message_count: row.get("message_count"),
            model_backend: row.get("model_backend"),
            containment_mode: row.get("containment_mode"),
        }).collect();

        let message_rows = match last_sync_time {
            Some(since) => sqlx::query(
                "SELECT session_id, user_id, role, content, created_at,
                        model_backend, containment_mode, entry_hash, prev_hash
                 FROM conversation_messages
                 WHERE user_id = ? AND created_at > ?
                 ORDER BY created_at ASC LIMIT 5000"
            )
            .bind(user_id)
            .bind(since)
            .fetch_all(&self.db_pool)
            .await
            .map_err(|e| format!("Failed to fetch messages: {}", e))?,

            None => sqlx::query(
                "SELECT session_id, user_id, role, content, created_at,
                        model_backend, containment_mode, entry_hash, prev_hash
                 FROM conversation_messages
                 WHERE user_id = ?
                 ORDER BY created_at ASC LIMIT 5000"
            )
            .bind(user_id)
            .fetch_all(&self.db_pool)
            .await
            .map_err(|e| format!("Failed to fetch messages: {}", e))?,
        };

        let messages = message_rows.iter().map(|row| SyncConversationMessage {
            session_id: row.get("session_id"),
            user_id: row.get("user_id"),
            role: row.get("role"),
            content: row.get("content"),
            created_at: row.get("created_at"),
            model_backend: row.get("model_backend"),
            containment_mode: row.get("containment_mode"),
            entry_hash: row.get("entry_hash"),
            prev_hash: row.get("prev_hash"),
        }).collect();

        Ok(ConversationSyncPayload { sessions, messages })
    }

    /// Receive and upsert conversation sessions and messages from a peer.
    /// Sessions use ON CONFLICT (session_id) to merge; messages are deduplicated
    /// by (session_id, created_at, role) since conversation_messages has no UNIQUE constraint.
    pub async fn receive_conversations_from_peer(
        &self,
        payload: ConversationSyncPayload,
    ) -> Result<(usize, usize), String> {
        let mut sessions_stored = 0usize;
        let mut messages_stored = 0usize;

        for session in &payload.sessions {
            sqlx::query(
                "INSERT INTO conversation_sessions
                     (session_id, user_id, title, started_at, last_active, message_count,
                      model_backend, containment_mode)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT (session_id) DO UPDATE SET
                     title            = COALESCE(EXCLUDED.title, conversation_sessions.title),
                     last_active      = CASE WHEN EXCLUDED.last_active > conversation_sessions.last_active THEN EXCLUDED.last_active ELSE conversation_sessions.last_active END,
                     message_count    = CASE WHEN EXCLUDED.message_count > conversation_sessions.message_count THEN EXCLUDED.message_count ELSE conversation_sessions.message_count END,
                     model_backend    = COALESCE(EXCLUDED.model_backend, conversation_sessions.model_backend),
                     containment_mode = COALESCE(EXCLUDED.containment_mode, conversation_sessions.containment_mode)"
            )
            .bind(&session.session_id)
            .bind(&session.user_id)
            .bind(&session.title)
            .bind(session.started_at)
            .bind(session.last_active)
            .bind(session.message_count)
            .bind(&session.model_backend)
            .bind(&session.containment_mode)
            .execute(&self.db_pool)
            .await
            .map_err(|e| format!("Failed to upsert session {}: {}", session.session_id, e))?;

            sessions_stored += 1;
        }

        for msg in &payload.messages {
            let result = sqlx::query(
                "INSERT INTO conversation_messages
                     (session_id, user_id, role, content, created_at,
                      model_backend, containment_mode, entry_hash, prev_hash)
                 SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?
                 WHERE NOT EXISTS (
                     SELECT 1 FROM conversation_messages
                     WHERE session_id = ? AND created_at = ? AND role = ?
                 )"
            )
            .bind(&msg.session_id)
            .bind(&msg.user_id)
            .bind(&msg.role)
            .bind(&msg.content)
            .bind(msg.created_at)
            .bind(&msg.model_backend)
            .bind(&msg.containment_mode)
            .bind(&msg.entry_hash)
            .bind(&msg.prev_hash)
            .bind(&msg.session_id)   // WHERE NOT EXISTS: session_id = ?
            .bind(msg.created_at)    // WHERE NOT EXISTS: created_at = ?
            .bind(&msg.role)         // WHERE NOT EXISTS: role = ?
            .execute(&self.db_pool)
            .await
            .map_err(|e| format!("Failed to insert message: {}", e))?;

            if result.rows_affected() > 0 {
                messages_stored += 1;
            }
        }

        // Recount message_count for all affected sessions so it stays accurate
        for session in &payload.sessions {
            let _ = sqlx::query(
                "UPDATE conversation_sessions
                 SET message_count = (
                     SELECT COUNT(*) FROM conversation_messages WHERE session_id = ?
                 )
                 WHERE session_id = ?"
            )
            .bind(&session.session_id)
            .execute(&self.db_pool)
            .await;
        }

        if messages_stored > 0 {
            println!("[ZynkSync] ✓ Conversation sync: {} new message(s) across {} session(s)",
                messages_stored, sessions_stored);
        }
        Ok((sessions_stored, messages_stored))
    }

    /// Push local conversations newer than last sync to a peer device.
    async fn push_conversations_to_peer(
        &self,
        peer: &PeerDevice,
        user_id: &str,
    ) -> Result<(usize, usize), String> {
        let payload = self.get_modified_conversations(&peer.device_id, user_id).await?;

        if payload.sessions.is_empty() && payload.messages.is_empty() {
            return Ok((0, 0));
        }

        // Verbose push detail omitted — summary printed by caller

        let endpoint = format!("{}/api/zynksync/conversations/receive", peer.url);
        let client = self.http_client.read().await.clone();
        let response = client
            .post(&endpoint)
            .json(&payload)
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| format!("Failed to push conversations: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Conversation push rejected by peer: {}", response.status()));
        }

        let result: serde_json::Value = response.json().await
            .map_err(|e| format!("Failed to parse conversation sync response: {}", e))?;

        let sessions_stored = result["sessions_stored"].as_u64().unwrap_or(0) as usize;
        let messages_stored = result["messages_stored"].as_u64().unwrap_or(0) as usize;

        if messages_stored > 0 {
            println!("[ZynkSync] ✓ Peer stored {} sessions, {} new messages", sessions_stored, messages_stored);
        }
        Ok((sessions_stored, messages_stored))
    }

    /// Bidirectional sync with "active device wins" reconciliation
    /// Compares memory inventories and syncs from the device with most recent activity
    pub async fn sync_bidirectional(&self, peer_id: &str, user_id: &str) -> Result<SyncResult, String> {
        let peer = {
            let peers_map = self.peers.read().await;
            peers_map.get(peer_id).cloned()
                .ok_or_else(|| format!("Peer {} not found", peer_id))?
        };

        if !peer.paired {
            return Err(format!("Device {} is not paired", peer.device_name));
        }

        // Verbose per-sync logging suppressed

        // Step 1: Get local inventory
        let local_inventory = self.get_local_inventory(user_id).await?;

        // Step 2: Request remote inventory
        let endpoint = format!("{}/api/zynksync/inventory", peer.url);
        let request = InventoryRequest {
            user_id: user_id.to_string(),
        };

        let client = self.http_client.read().await.clone();
        let response = client
            .post(&endpoint)
            .json(&request)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| format!("Failed to get remote inventory: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Remote inventory request failed: {}", response.status()));
        }

        let remote_inventory: MemoryInventory = response.json().await
            .map_err(|e| format!("Failed to parse remote inventory: {}", e))?;


        // Step 3: Check if this is the first sync between these devices
        let is_first_sync = self.is_first_sync(&peer.device_id).await?;

        if is_first_sync {
            println!("[ZynkSync] ⚠️  FIRST SYNC - Using additive merge (no deletions)");
        }

        // Step 4: Determine which device is "active" (source of truth)
        // CRITICAL: On first sync, prioritize memory count over timestamp to ensure complete data transfer
        let local_is_active = if is_first_sync {
            // First sync: Device with MORE memories is always the source of truth
            // This prevents incomplete transfers when a freshly-synced device has newer timestamp
            if local_inventory.memory_count == 0 && remote_inventory.memory_count == 0 {
                println!("[ZynkSync] Both devices have no memories, nothing to sync");
                return Ok(SyncResult {
                    peer_device_id: peer.device_id,
                    peer_device_name: peer.device_name,
                    memories_sent: 0,
                    memories_received: 0,
                    conversations_sent: 0,
                    conflicts_resolved: 0,
                    success: true,
                    error: None,
                });
            }
            // Device with more memories is active (pull from them)
            local_inventory.memory_count >= remote_inventory.memory_count
        } else {
            // Subsequent syncs: Use timestamp to determine which device has recent activity
            match (&local_inventory.latest_activity, &remote_inventory.latest_activity) {
                (Some(local_time), Some(remote_time)) => {
                    if local_time > remote_time {
                        true
                    } else if local_time < remote_time {
                        false
                    } else {
                        // Timestamps equal - use memory count as tie-breaker
                        local_inventory.memory_count >= remote_inventory.memory_count
                    }
                },
                (Some(_), None) => true,  // Local has memories, remote doesn't
                (None, Some(_)) => false, // Remote has memories, local doesn't
                (None, None) => {
                    println!("[ZynkSync] Both devices have no memories, nothing to sync");
                    return Ok(SyncResult {
                        peer_device_id: peer.device_id,
                        peer_device_name: peer.device_name,
                        memories_sent: 0,
                        memories_received: 0,
                        conversations_sent: 0,
                        conflicts_resolved: 0,
                        success: true,
                        error: None,
                    });
                }
            }
        };

        let mut memories_sent = 0;
        let mut memories_received = 0;

        if local_is_active {
            // LOCAL IS ACTIVE: Push our state to remote

            // FIXED: Compare by content hash instead of database ID (IDs are machine-specific!)
            let local_hashes: std::collections::HashSet<String> = local_inventory.content_hashes.iter().cloned().collect();
            let remote_hashes: std::collections::HashSet<String> = remote_inventory.content_hashes.iter().cloned().collect();

            // Create hash->ID mapping for local memories
            let hash_to_id: std::collections::HashMap<String, i32> = local_inventory.content_hashes.iter()
                .zip(local_inventory.memory_ids.iter())
                .map(|(h, id)| (h.clone(), *id))
                .collect();

            // Find memories we have that remote doesn't (by content hash)
            let hashes_to_send: Vec<String> = local_hashes.difference(&remote_hashes).cloned().collect();
            let to_send: Vec<i32> = hashes_to_send.iter().filter_map(|h| hash_to_id.get(h).copied()).collect();

            if !to_send.is_empty() {
                let memories_to_send = self.get_memories_by_ids(&to_send).await?;

                let endpoint = format!("{}/api/zynksync/receive", peer.url);
                let client = self.http_client.read().await.clone();
                let response = client
                    .post(&endpoint)
                    .json(&memories_to_send)
                    .timeout(Duration::from_secs(30))
                    .send()
                    .await
                    .map_err(|e| format!("Failed to send memories: {}", e))?;

                if !response.status().is_success() {
                    return Err(format!("Failed to send memories: {}", response.status()));
                }

                memories_sent = to_send.len();
            }

            // Handle deletions (only on subsequent syncs, not first sync)
            if !is_first_sync {
                // Find memories remote has that we don't (by content hash - deletions for remote)
                let hashes_to_delete: Vec<String> = remote_hashes.difference(&local_hashes).cloned().collect();

                if !hashes_to_delete.is_empty() {
                    println!("[ZynkSync] Remote has {} memories we don't - propagating deletion", hashes_to_delete.len());
                    // Create hash->ID mapping for remote memories
                    let remote_hash_to_id: std::collections::HashMap<String, i32> = remote_inventory.content_hashes.iter()
                        .zip(remote_inventory.memory_ids.iter())
                        .map(|(h, id)| (h.clone(), *id))
                        .collect();

                    // Map hashes to remote IDs
                    let ids_to_delete: Vec<i32> = hashes_to_delete.iter()
                        .filter_map(|h| remote_hash_to_id.get(h).copied())
                        .collect();

                    if !ids_to_delete.is_empty() {
                        println!("[ZynkSync] Requesting remote to delete {} memories", ids_to_delete.len());
                        let endpoint = format!("{}/api/zynksync/delete", peer.url);
                        let client = self.http_client.read().await.clone();
                        let response = client
                            .post(&endpoint)
                            .json(&ids_to_delete)
                            .timeout(Duration::from_secs(30))
                            .send()
                            .await
                            .map_err(|e| format!("Failed to request deletions: {}", e))?;

                        if !response.status().is_success() {
                            eprintln!("[ZynkSync] Warning: Delete request failed: {}", response.status());
                        }
                    }
                }
            } else {
                let unique_remote_memories = remote_hashes.difference(&local_hashes).count();
                if unique_remote_memories > 0 {
                    println!("[ZynkSync] Note: Remote has {} unique memories (keeping them - first sync)", unique_remote_memories);
                }
            }

        } else {
            // REMOTE IS ACTIVE: Pull their state to local

            // FIXED: Compare by content hash instead of database ID
            let local_hashes: std::collections::HashSet<String> = local_inventory.content_hashes.iter().cloned().collect();
            let remote_hashes: std::collections::HashSet<String> = remote_inventory.content_hashes.iter().cloned().collect();

            // Create hash->ID mapping for remote memories
            let remote_hash_to_id: std::collections::HashMap<String, i32> = remote_inventory.content_hashes.iter()
                .zip(remote_inventory.memory_ids.iter())
                .map(|(h, id)| (h.clone(), *id))
                .collect();

            // Find memories remote has that we don't (by content hash)
            let hashes_to_receive: Vec<String> = remote_hashes.difference(&local_hashes).cloned().collect();
            let to_receive: Vec<i32> = hashes_to_receive.iter().filter_map(|h| remote_hash_to_id.get(h).copied()).collect();

            if !to_receive.is_empty() {
                println!("[ZynkSync] Requesting {} missing memories from remote", to_receive.len());
                let endpoint = format!("{}/api/zynksync/fetch", peer.url);
                let client = self.http_client.read().await.clone();
                let response = client
                    .post(&endpoint)
                    .json(&to_receive)
                    .timeout(Duration::from_secs(30))
                    .send()
                    .await
                    .map_err(|e| format!("Failed to fetch memories: {}", e))?;

                if !response.status().is_success() {
                    return Err(format!("Failed to fetch memories: {}", response.status()));
                }

                let memories: Vec<SyncMemory> = response.json().await
                    .map_err(|e| format!("Failed to parse memories: {}", e))?;

                memories_received = self.receive_from_peer(memories).await?;
            }

            // Handle deletions (only on subsequent syncs, not first sync)
            if !is_first_sync {
                // Find memories we have that remote doesn't (by content hash - deletions for us)
                let hashes_to_delete: Vec<String> = local_hashes.difference(&remote_hashes).cloned().collect();

                if !hashes_to_delete.is_empty() {
                    println!("[ZynkSync] We have {} memories remote doesn't - deleting locally", hashes_to_delete.len());
                    // Create hash->ID mapping for local memories
                    let local_hash_to_id: std::collections::HashMap<String, i32> = local_inventory.content_hashes.iter()
                        .zip(local_inventory.memory_ids.iter())
                        .map(|(h, id)| (h.clone(), *id))
                        .collect();

                    // Map hashes to local IDs
                    let ids_to_delete: Vec<i32> = hashes_to_delete.iter()
                        .filter_map(|h| local_hash_to_id.get(h).copied())
                        .collect();

                    if !ids_to_delete.is_empty() {
                        println!("[ZynkSync] Deleting {} obsolete memories locally", ids_to_delete.len());
                        self.delete_memories_by_ids(&ids_to_delete).await?;
                    }
                }
            } else {
                let unique_local_memories = local_hashes.difference(&remote_hashes).count();
                if unique_local_memories > 0 {
                    println!("[ZynkSync] Note: We have {} unique memories (keeping them - first sync)", unique_local_memories);
                }
            }
        }

        // Sync conversation history — both devices always push their new conversations.
        // Union merge (not active/passive): each device pushes what it has, peer deduplicates.
        let conversations_sent = match self.push_conversations_to_peer(&peer, user_id).await {
            Ok((_sessions, messages)) => messages,
            Err(e) => {
                eprintln!("[ZynkSync] Conversation sync failed (non-fatal): {}", e);
                0
            }
        };

        // Update sync timestamp so future syncs are not considered "first sync"
        self.update_sync_timestamp(&peer.device_id, local_is_active).await?;

        if memories_sent > 0 || memories_received > 0 {
            println!("[ZynkSync] ✓ Sync complete - sent: {}, received: {}", memories_sent, memories_received);
        }

        Ok(SyncResult {
            peer_device_id: peer.device_id,
            peer_device_name: peer.device_name,
            memories_sent,
            memories_received,
            conversations_sent,
            conflicts_resolved: 0,
            success: true,
            error: None,
        })
    }

    /// Get specific memories by their IDs (including their relationships)
    async fn get_memories_by_ids(&self, ids: &[i32]) -> Result<Vec<SyncMemory>, String> {
        let rows = if ids.is_empty() {
            vec![]
        } else {
            let in_clause = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
            let sql = format!(
                "SELECT id, user_id, session_id, content, title, source_type, created_at, updated_at,
                        parent_scroll_id, chunk_index, namespace, is_syncable, is_shareable,
                        embedding, link_count, is_ephemeral, expires_at, sentiment_score, sentiment_label,
                        event_type, event_date, entities_detected, original_text
                 FROM memories WHERE id IN ({})",
                in_clause
            );
            let mut q = sqlx::query(&sql);
            for id in ids { q = q.bind(id); }
            q.fetch_all(&self.db_pool)
                .await
                .map_err(|e| format!("Failed to fetch memories: {}", e))?
        };

        let mut memories: Vec<SyncMemory> = rows
            .iter()
            .map(|row| {
                // Convert pgvector::Vector to Vec<f32>
                let embedding: Option<Vec<f32>> = row.try_get::<Option<Vec<u8>>, _>("embedding")
                    .ok()
                    .flatten()
                    .map(|blob| blob.chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect::<Vec<f32>>());

                SyncMemory {
                    id: row.get("id"),
                    user_id: row.get("user_id"),
                    session_id: row.get("session_id"),
                    content: row.get("content"),
                    title: row.get("title"),
                    source_type: row.get("source_type"),
                    created_at: row.get("created_at"),
                    updated_at: row.get("updated_at"),
                    parent_scroll_id: row.get("parent_scroll_id"),
                    chunk_index: row.get("chunk_index"),
                    namespace: row.get("namespace"),
                    is_syncable: row.get("is_syncable"),
                    is_shareable: row.get("is_shareable"),
                    embedding,
                    link_count: row.get("link_count"),
                    is_ephemeral: row.get("is_ephemeral"),
                    expires_at: row.get("expires_at"),
                    sentiment_score: row.get("sentiment_score"),
                    sentiment_label: row.get("sentiment_label"),
                    event_type: row.get("event_type"),
                    event_date: row.get("event_date"),
                    entities_detected: row.get("entities_detected"),
                    original_text: row.get("original_text"),
                    relationships: Vec::new(),  // Will be populated below
                }
            })
            .collect();

        // Fetch relationships for these memories
        if !ids.is_empty() {
            let relationship_rows = {
                let in_clause = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
                let sql = format!(
                    "SELECT source_memory_id, target_memory_id, relation_type, confidence, notes, created_by
                     FROM memory_links
                     WHERE source_memory_id IN ({}) OR target_memory_id IN ({})",
                    in_clause, in_clause
                );
                let mut q = sqlx::query(&sql);
                for id in ids { q = q.bind(id); }
                for id in ids { q = q.bind(id); }
                q.fetch_all(&self.db_pool)
                    .await
                    .map_err(|e| format!("Failed to fetch relationships: {}", e))?
            };

            // Group relationships by source memory
            for rel_row in relationship_rows {
                let source_id: i32 = rel_row.get("source_memory_id");
                let relationship = MemoryRelationship {
                    source_memory_id: source_id,
                    target_memory_id: rel_row.get("target_memory_id"),
                    relation_type: rel_row.get("relation_type"),
                    confidence: rel_row.get("confidence"),
                    notes: rel_row.get("notes"),
                    created_by: rel_row.get("created_by"),
                };

                // Add relationship to the source memory
                if let Some(memory) = memories.iter_mut().find(|m| m.id == source_id) {
                    memory.relationships.push(relationship);
                }
            }

            println!("[ZynkSync] Fetched {} memories with their relationships", memories.len());
        }

        Ok(memories)
    }

    /// Delete memories by their IDs (used when remote active device no longer has them)
    async fn delete_memories_by_ids(&self, ids: &[i32]) -> Result<usize, String> {
        let result = if ids.is_empty() {
            return Ok(0);
        } else {
            let in_clause = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
            let sql = format!("DELETE FROM memories WHERE id IN ({})", in_clause);
            let mut q = sqlx::query(&sql);
            for id in ids { q = q.bind(id); }
            q.execute(&self.db_pool)
                .await
                .map_err(|e| format!("Failed to delete memories: {}", e))?
        };

        let deleted_count = result.rows_affected() as usize;
        println!("[ZynkSync] Deleted {} memories", deleted_count);
        Ok(deleted_count)
    }

    /// Propagate a memory deletion to all paired devices
    /// Called when user manually deletes a memory to sync the deletion across devices
    /// FIXED: Uses content hash for portable deletion (IDs differ across machines)
    #[allow(dead_code)]
    pub async fn propagate_deletion(&self, memory_id: i32) -> Result<usize, String> {
        println!("[ZynkSync] Propagating deletion of memory #{} to paired devices", memory_id);

        // CRITICAL: Get the content hash BEFORE deletion (IDs are machine-specific!)
        let content_hash = match sqlx::query_scalar::<_, String>(
            "SELECT content FROM memories WHERE id = ?"
        )
        .bind(memory_id)
        .fetch_optional(&self.db_pool)
        .await
        {
            Ok(Some(content)) => {
                use sha2::{Digest, Sha256};
                format!("{:x}", Sha256::digest(content.as_bytes()))
            }
            Ok(None) => {
                println!("[ZynkSync] Memory #{} not found (already deleted?)", memory_id);
                return Ok(0);
            }
            Err(e) => {
                eprintln!("[ZynkSync] Failed to get content: {}", e);
                return Err(format!("Failed to get content: {}", e));
            }
        };

        println!("[ZynkSync] Memory content hash: {}", content_hash);

        // Delegate to hash-based propagation
        self.propagate_deletion_by_hash(content_hash).await
    }

    /// Propagate a memory deletion using content hash (memory already deleted)
    /// This is called when the memory has already been deleted locally and we have the hash
    /// IMPORTANT: Use this when you've already fetched the hash before deletion
    pub async fn propagate_deletion_by_hash(&self, content_hash: String) -> Result<usize, String> {
        println!("[ZynkSync] Propagating deletion by hash {} to paired devices", content_hash);

        // Get all paired peers
        let peers = {
            let peers_map = self.peers.read().await;
            peers_map.values()
                .filter(|p| p.paired)
                .cloned()
                .collect::<Vec<_>>()
        };

        if peers.is_empty() {
            println!("[ZynkSync] No paired devices to sync deletion to");
            return Ok(0);
        }

        let total_peers = peers.len();
        let mut success_count = 0;

        // Send deletion request to each peer (using content hash for portable lookup)
        for peer in peers {
            let endpoint = format!("{}/api/zynksync/delete-by-hash", peer.url);
            let payload = serde_json::json!({
                "content_hash": content_hash
            });

            let client = self.http_client.read().await.clone();
            match client
                .post(&endpoint)
                .json(&payload)
                .timeout(Duration::from_secs(10))
                .send()
                .await
            {
                Ok(response) => {
                    if response.status().is_success() {
                        success_count += 1;
                        println!("[ZynkSync] ✓ Deletion synced to {}", peer.device_name);
                    } else {
                        eprintln!("[ZynkSync] ✗ Failed to sync deletion to {}: status {}",
                            peer.device_name, response.status());
                    }
                }
                Err(e) => {
                    eprintln!("[ZynkSync] ✗ Failed to reach {}: {}", peer.device_name, e);
                }
            }
        }

        println!("[ZynkSync] ✓ Deletion propagated to {}/{} devices", success_count, total_peers);
        Ok(success_count)
    }

    /// Start background task for periodic message delivery retry
    pub async fn start_message_delivery_loop(self: Arc<Self>) {
        println!("[ZChat] Starting message delivery loop (interval: 30s)");

        let mut interval_timer = interval(Duration::from_secs(30));

        loop {
            interval_timer.tick().await;

            // Check if auto-sync is enabled (use same flag for message delivery)
            {
                let enabled = self.auto_sync_enabled.read().await;
                if !*enabled {
                    continue;
                }
            }

            // Get all paired devices
            let peers = {
                let peers_map = self.peers.read().await;
                peers_map.values().cloned().collect::<Vec<_>>()
            };

            // Try to deliver undelivered messages to each peer
            for peer in peers {
                if peer.paired {
                    match self.deliver_zchat_messages_to_peer(&peer.device_id).await {
                        Ok(0) => {}, // No messages to deliver
                        Ok(count) => println!("[ZChat] ✓ Delivered {} message(s) to {}", count, peer.device_name),
                        Err(e) if e.contains("No IP address") => {}, // Silent - common when device offline
                        Err(e) => println!("[ZChat] Delivery retry failed for {}: {}", peer.device_name, e),
                    }
                }
            }
        }
    }

    /// Send a heartbeat ping to all paired peers
    pub async fn start_heartbeat_loop(self: Arc<Self>) {
        let mut interval_timer = interval(Duration::from_secs(15));
        loop {
            interval_timer.tick().await;

            let enabled = self.auto_sync_enabled.read().await;
            if !*enabled { continue; }
            drop(enabled);

            let device_id = self.device_id.clone();
            let peers = {
                let peers_map = self.peers.read().await;
                peers_map.values().cloned().collect::<Vec<_>>()
            };

            for peer in peers {
                if !peer.paired { continue; }
                let url = format!("https://{}:{}/api/presence/heartbeat", peer.host, peer.port);
                let body = serde_json::json!({ "device_id": device_id });
                let client = self.http_client.read().await.clone();
                let _ = client
                    .post(&url)
                    .json(&body)
                    .timeout(Duration::from_secs(5))
                    .send()
                    .await;
            }
        }
    }

    /// Send goodbye signal to all paired peers (called on clean shutdown)
    pub async fn send_goodbye_to_peers(&self) {
        let device_id = self.device_id.clone();
        let peers = {
            let peers_map = self.peers.read().await;
            peers_map.values().cloned().collect::<Vec<_>>()
        };

        for peer in peers {
            if !peer.paired { continue; }
            let url = format!("https://{}:{}/api/presence/goodbye", peer.host, peer.port);
            let body = serde_json::json!({ "device_id": device_id });
            let client = self.http_client.read().await.clone();
            let _ = client
                .post(&url)
                .json(&body)
                .timeout(Duration::from_secs(3))
                .send()
                .await;
        }
        println!("[Presence] Goodbye sent to all peers");
    }

    /// Start automatic synchronization loop
    pub async fn start_auto_sync(self: Arc<Self>) {
        println!("[ZynkSync] Starting auto-sync loop (interval: {}s)", self.sync_interval_secs);

        {
            let mut enabled = self.auto_sync_enabled.write().await;
            *enabled = true;
        }

        let mut interval_timer = interval(Duration::from_secs(self.sync_interval_secs));

        loop {
            interval_timer.tick().await;

            // Check if still enabled
            {
                let enabled = self.auto_sync_enabled.read().await;
                if !*enabled {
                    println!("[ZynkSync] Auto-sync stopped");
                    break;
                }
            }

            // Get all peers
            let peers = {
                let peers_map = self.peers.read().await;
                peers_map.values().cloned().collect::<Vec<_>>()
            };

            if peers.is_empty() {
                continue;
            }

            // Auto-sync trigger — detail logged per-peer below

            // Get current user_id for syncing
            let user_id = match user_identity::get_user_id() {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("[ZynkSync] ✗ Failed to get user_id: {}", e);
                    continue;
                }
            };

            // Bidirectional sync with each peer (active device wins)
            for peer in peers {
                match self.sync_bidirectional(&peer.device_id, &user_id).await {
                    Ok(result) => {
                        if result.memories_sent > 0 || result.memories_received > 0 {
                            println!("[ZynkSync] ✓ Auto-synced with {} - sent: {}, received: {}",
                                peer.device_name, result.memories_sent, result.memories_received);
                        }
                    }
                    Err(e) => {
                        eprintln!("[ZynkSync] ✗ Auto-sync failed with {}: {}", peer.device_name, e);
                    }
                }
            }
        }
    }

    /// Stop automatic synchronization (HTTP server keeps running for ZynkLink)
    pub async fn stop_auto_sync(&self) {
        // Stop auto-sync loop only - HTTP server stays running for ZynkLink
        let mut enabled = self.auto_sync_enabled.write().await;
        *enabled = false;
        println!("[ZynkSync] Auto-sync disabled (HTTP server still running for ZynkLink)");
    }

    /// Broadcast a pause signal to all sync-paired peers so they also pause
    pub async fn broadcast_pause_to_peers(&self) -> usize {
        println!("[ZynkSync] Broadcasting pause to all paired devices");
        let peers = {
            let peers_map = self.peers.read().await;
            peers_map.values().filter(|p| p.paired).cloned().collect::<Vec<_>>()
        };
        let mut count = 0;
        for peer in peers {
            let endpoint = format!("{}/api/zynksync/pause", peer.url);
            let client = self.http_client.read().await.clone();
            match client
                .post(&endpoint)
                .header("X-Device-ID", &self.device_id)
                .timeout(Duration::from_secs(5))
                .send()
                .await
            {
                Ok(r) if r.status().is_success() => {
                    count += 1;
                    println!("[ZynkSync] ✓ Pause broadcast to {}", peer.device_name);
                }
                Ok(r) => eprintln!("[ZynkSync] ✗ Pause rejected by {}: {}", peer.device_name, r.status()),
                Err(e) => eprintln!("[ZynkSync] ✗ Could not reach {} for pause: {}", peer.device_name, e),
            }
        }
        count
    }

    /// Broadcast a resume signal to all sync-paired peers so they also resume
    pub async fn broadcast_resume_to_peers(&self) -> usize {
        println!("[ZynkSync] Broadcasting resume to all paired devices");
        let peers = {
            let peers_map = self.peers.read().await;
            peers_map.values().filter(|p| p.paired).cloned().collect::<Vec<_>>()
        };
        let mut count = 0;
        for peer in peers {
            let endpoint = format!("{}/api/zynksync/resume", peer.url);
            let client = self.http_client.read().await.clone();
            match client
                .post(&endpoint)
                .header("X-Device-ID", &self.device_id)
                .timeout(Duration::from_secs(5))
                .send()
                .await
            {
                Ok(r) if r.status().is_success() => {
                    count += 1;
                    println!("[ZynkSync] ✓ Resume broadcast to {}", peer.device_name);
                }
                Ok(r) => eprintln!("[ZynkSync] ✗ Resume rejected by {}: {}", peer.device_name, r.status()),
                Err(e) => eprintln!("[ZynkSync] ✗ Could not reach {} for resume: {}", peer.device_name, e),
            }
        }
        count
    }

    /// Check if auto-sync is currently enabled
    pub async fn is_auto_sync_enabled(&self) -> bool {
        let enabled = self.auto_sync_enabled.read().await;
        *enabled
    }

    /// Get list of peer devices
    pub async fn get_peers(&self) -> Vec<PeerDevice> {
        let peers_map = self.peers.read().await;
        peers_map.values().cloned().collect()
    }

    /// Public wrapper for get_local_inventory (for Tauri commands)
    pub async fn get_local_inventory_public(&self, user_id: &str) -> Result<MemoryInventory, String> {
        self.get_local_inventory(user_id).await
    }

    /// Get remote device's inventory via HTTP
    pub async fn get_remote_inventory_public(&self, peer_url: &str, user_id: &str) -> Result<MemoryInventory, String> {
        let endpoint = format!("{}/api/zynksync/inventory", peer_url);
        let request = InventoryRequest {
            user_id: user_id.to_string(),
        };

        let client = self.http_client.read().await.clone();
        let response = client
            .post(&endpoint)
            .json(&request)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| format!("Failed to get remote inventory: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Remote inventory request failed: {}", response.status()));
        }

        response.json().await
            .map_err(|e| format!("Failed to parse remote inventory: {}", e))
    }

    /// Request pairing with a peer device (generates code on this device)
    pub async fn request_pairing(&self, peer_id: &str) -> Result<String, String> {
        let pairing_code = self.generate_pairing_code().await?;

        // Update peer with pairing code
        {
            let mut peers_map = self.peers.write().await;
            if let Some(peer) = peers_map.get_mut(peer_id) {
                peer.pairing_code = Some(pairing_code.clone());
                println!("[ZynkSync] Generated pairing code {} for {}", pairing_code, peer.device_name);
            } else {
                return Err(format!("Peer {} not found", peer_id));
            }
        }

        Ok(pairing_code)
    }

    /// Verify pairing code and authorize peer
    pub async fn verify_pairing_code(&self, peer_id: &str, code: &str) -> Result<(), String> {
        let mut peers_map = self.peers.write().await;

        if let Some(peer) = peers_map.get_mut(peer_id) {
            match &peer.pairing_code {
                Some(expected_code) if expected_code == code => {
                    peer.paired = true;
                    peer.pairing_code = None;  // Clear code after successful pairing

                    // Also clear pairing code from database
                    let _ = sqlx::query(
                        "UPDATE zynk_devices SET pairing_code = NULL, pairing_code_expires_at = NULL
                         WHERE device_id = ?"
                    )
                    .bind(&self.device_id)
                    .execute(&self.db_pool)
                    .await;

                    println!("[ZynkSync] ✓ Paired with {} ({})", peer.device_name, peer.device_id);
                    Ok(())
                }
                Some(_) => Err("Incorrect pairing code".to_string()),
                None => Err("No pairing code set. Request pairing first.".to_string()),
            }
        } else {
            Err(format!("Peer {} not found", peer_id))
        }
    }

    /// Unpair from a device
    pub async fn unpair_device(&self, peer_id: &str) -> Result<(), String> {
        // Fully remove the device record — keeping a stale "unpaired" entry in the DB
        // causes ghost devices to reappear after app restart via load_devices().
        self.remove_device(peer_id).await
    }

    /// Start HTTP server to receive sync requests from peers
    /// Returns the actual port the server is listening on
    pub async fn start_http_server(self: Arc<Self>) -> Result<u16, String> {
        // Clean up any old process using port 57963 (handles hot reload issues)
        // NOTE: Port cleanup is now handled in lib.rs start_zynksync() before creating the service
        // Self::cleanup_port_57963();

        // Create Axum router with device-to-device endpoints
        let app = Router::new()
            .route("/api/zynksync/receive", post(handle_receive_sync))
            .route("/api/zynksync/info", axum::routing::get(handle_device_info))
            .route("/api/zynksync/verify-pairing", post(handle_verify_pairing))
            .route("/api/zchat/deliver", post(handle_zchat_deliver))
            .route("/api/identity/verify-sync-code", post(handle_verify_sync_code))
            .route("/api/identity/consume-sync-code", post(handle_consume_sync_code))
            // New endpoints for bidirectional "active device wins" sync
            .route("/api/zynksync/inventory", post(handle_get_inventory))
            .route("/api/zynksync/delete", post(handle_delete_memories))
            .route("/api/zynksync/delete-by-hash", post(handle_delete_by_hash))
            .route("/api/zynksync/fetch", post(handle_fetch_memories))
            // ZynkLink endpoints for file sharing
            .route("/api/zynklink/verify-code", post(handle_zynklink_verify_code))
            .route("/api/zynklink/accept-code", post(handle_zynklink_accept_code))
            .route("/api/zynklink/directories", post(handle_zynklink_directories))
            .route("/api/zynklink/files", post(handle_zynklink_files))
            .route("/api/zynklink/download", post(handle_zynklink_download))
            .route("/api/zynklink/deliver-chat", post(handle_zynklink_deliver_chat))
            .route("/api/zynklink/notify-unpaired", post(handle_zynklink_notify_unpaired))
            .route("/api/zynksync/notify-unsynced", post(handle_notify_unsynced))
            .route("/api/zynksync/conversations/receive", post(handle_receive_conversations))
            .route("/api/presence/heartbeat", post(handle_heartbeat))
            .route("/api/presence/goodbye", post(handle_goodbye))
            .route("/api/zynksync/pause", post(handle_pause))
            .route("/api/zynksync/resume", post(handle_resume))
            .with_state(Arc::clone(&self));

        // Build TLS config from this device's certificate
        let server_config = crate::tls::build_server_config(&self.cert_pem, &self.key_pem)
            .map_err(|e| format!("Failed to build TLS config: {}", e))?;
        let tls_acceptor = TlsAcceptor::from(Arc::new(server_config));

        // Bind TCP listener on fixed port 57963
        let port: u16 = 57963;
        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        let tcp_listener = TcpListener::bind(addr).await
            .map_err(|e| format!("Failed to bind port {}: {}", port, e))?;

        // Store the port
        {
            let mut server_port = self.server_port.write().await;
            *server_port = Some(port);
        }

        println!("[ZynkSync] HTTPS server listening on port {}", port);

        // Create shutdown channel
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        {
            let mut tx = self.shutdown_tx.write().await;
            *tx = Some(shutdown_tx);
        }

        // Accept loop: accept TCP → TLS handshake → serve per-connection via hyper
        // ConnectInfo<SocketAddr> is injected manually into each request extension so
        // handlers can read the client IP without needing axum::serve's built-in mechanism.
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => {
                        println!("[ZynkSync] HTTPS server shutting down");
                        break;
                    }
                    result = tcp_listener.accept() => {
                        let (tcp_stream, peer_addr) = match result {
                            Ok(pair) => pair,
                            Err(e) => { eprintln!("[ZynkSync] TCP accept error: {}", e); break; }
                        };
                        let acceptor = tls_acceptor.clone();
                        let router = app.clone();
                        tokio::spawn(async move {
                            let tls_stream = match acceptor.accept(tcp_stream).await {
                                Ok(s) => s,
                                Err(e) => {
                                    eprintln!("[ZynkSync] TLS handshake failed from {}: {}", peer_addr, e);
                                    return;
                                }
                            };
                            let io = TokioIo::new(tls_stream);
                            let svc = hyper::service::service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                                let mut router = router.clone();
                                async move {
                                    let (parts, body) = req.into_parts();
                                    let body = axum::body::Body::new(body);
                                    let mut req = hyper::Request::from_parts(parts, body);
                                    req.extensions_mut().insert(ConnectInfo(peer_addr));
                                    use tower::Service;
                                    router.call(req).await
                                }
                            });
                            if let Err(e) = HyperConnBuilder::new(TokioExecutor::new())
                                .serve_connection(io, svc)
                                .await
                            {
                                println!("[ZynkSync] Connection from {} closed: {}", peer_addr, e);
                            }
                        });
                    }
                }
            }
            println!("[ZynkSync] HTTPS server stopped");
        });

        Ok(port)
    }
}

/// Axum handler for receiving sync memories from a peer
async fn handle_receive_sync(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
    Json(memories): Json<Vec<SyncMemory>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?;
    check_sync_authorized(&service.db_pool, device_id).await?;

    println!("[ZynkSync] Received {} memories from peer", memories.len());
    let stored_count = service.receive_from_peer(memories).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;

    Ok(Json(serde_json::json!({ "success": true, "stored": stored_count })))
}

/// Axum handler for getting device info (used when adding devices manually)
async fn handle_device_info(
    State(service): State<Arc<ZynkSyncService>>,
) -> Result<Json<serde_json::Value>, String> {
    Ok(Json(serde_json::json!({
        "device_id": service.device_id,
        "device_name": service.device_name,
        "version": "1.0.0"
    })))
}

/// Axum handler for verifying device pairing codes
/// This implements bidirectional pairing: when a client connects to a host with a pairing code,
/// the host automatically adds the client to its peer list
async fn handle_verify_pairing(
    State(service): State<Arc<ZynkSyncService>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, String> {
    let pairing_code = request.get("pairing_code")
        .and_then(|c| c.as_str())
        .ok_or("Missing pairing_code parameter")?;

    // Extract client device info (for bidirectional pairing)
    let client_device_id = request.get("client_device_id")
        .and_then(|v| v.as_str())
        .ok_or("Missing client_device_id parameter")?;

    let client_device_name = request.get("client_device_name")
        .and_then(|v| v.as_str())
        .ok_or("Missing client_device_name parameter")?;

    // Extract client's user_id for validation (optional for backwards compatibility)
    let client_user_id = request.get("client_user_id")
        .and_then(|v| v.as_str());

    // Extract client's TLS cert DER (base64 encoded) for certificate pinning
    let client_cert_der: Option<Vec<u8>> = request
        .get("client_cert_der")
        .and_then(|v| v.as_str())
        .and_then(|b64| BASE64.decode(b64).ok());
    println!("[TLS] Client cert received: {} bytes", client_cert_der.as_ref().map_or(0, |d| d.len()));

    // Get the REAL client IP from the TCP connection (not from request body!)
    let client_ip = addr.ip().to_string();

    println!("[ZynkSync] Verifying pairing code [REDACTED] for client: {} ({}) from IP: {}",
        client_device_name, client_device_id, client_ip);

    if let Some(client_uid) = client_user_id {
        println!("[ZynkSync] Client provided user_id: {}", client_uid);
    } else {
        println!("[ZynkSync] ⚠ Warning: Client did not provide user_id (security validation disabled)");
    }

    // Rate limit: invalidate the pairing code after 5 failed attempts from the same IP.
    // Collapses attacker odds from "a million tries in ten minutes" to "5 tries, ever."
    {
        let attempts = service.failed_pairing_attempts.read().await;
        if attempts.get(&client_ip).copied().unwrap_or(0) >= 5 {
            drop(attempts);
            sqlx::query(
                "UPDATE zynk_devices SET pairing_code = NULL, pairing_code_expires_at = NULL WHERE device_id = ?"
            )
            .bind(&service.device_id)
            .execute(&service.db_pool)
            .await.ok();
            println!("[ZynkSync] ⛔ Pairing code invalidated after 5 failed attempts from {}", client_ip);
            return Err("Too many failed attempts — pairing code invalidated. Generate a new code.".to_string());
        }
    }

    // Query database to verify the pairing code
    // Note: We check against THIS device's pairing code (the host)
    let result = sqlx::query(
        "SELECT device_id, device_name, pairing_code_expires_at
         FROM zynk_devices
         WHERE pairing_code = ?
           AND pairing_code_expires_at > datetime('now')
           AND device_id = ?"  // Make sure it's OUR pairing code
    )
    .bind(pairing_code)
    .bind(&service.device_id)
    .fetch_optional(&service.db_pool)
    .await
    .map_err(|e| format!("Database query failed: {}", e))?;

    match result {
        Some(_record) => {
            // Success — clear the failed attempt counter for this IP
            service.failed_pairing_attempts.write().await.remove(&client_ip);
            println!("[ZynkSync] ✓ Pairing code verified! Auto-adding client: {} ({})",
                client_device_name, client_device_id);

            // BIDIRECTIONAL PAIRING: Automatically add the client device to our peer list
            let port: u16 = 57963;
            let peer = PeerDevice {
                device_id: client_device_id.to_string(),
                device_name: client_device_name.to_string(),
                host: client_ip.to_string(),
                port,
                url: format!("https://{}:{}", client_ip, port),
                last_seen: Utc::now(),
                paired: true,
                pairing_code: None,
                user_id: None,  // Not relevant - this is the HOST adding the CLIENT
            };

            // Store client device with its TLS cert for future pinned connections
            sqlx::query(
                "INSERT INTO zynk_devices (device_id, device_name, device_ip, port, device_platform, is_paired, sync_paired, tls_cert_der, last_seen_at, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT (device_id) DO UPDATE
                 SET device_name = excluded.device_name,
                     device_ip = excluded.device_ip,
                     port = excluded.port,
                     is_paired = excluded.is_paired,
                     sync_paired = excluded.sync_paired,
                     tls_cert_der = excluded.tls_cert_der,
                     last_seen_at = excluded.last_seen_at"
            )
            .bind(client_device_id)
            .bind(client_device_name)
            .bind(&client_ip)
            .bind(port as i32)
            .bind("")  // device_platform
            .bind(true)  // is_paired
            .bind(true)  // sync_paired
            .bind(client_cert_der.as_deref())
            .bind(Utc::now())
            .bind(Utc::now())
            .execute(&service.db_pool)
            .await
            .map_err(|e| format!("Failed to save client device: {}", e))?;

            // Rebuild HTTP client to trust this client's cert for future requests
            if let Err(e) = service.rebuild_http_client().await {
                println!("[TLS] Warning: could not rebuild HTTP client after bidirectional pairing: {}", e);
            }

            // Add to peers map
            {
                let mut peers_map = service.peers.write().await;
                peers_map.insert(client_device_id.to_string(), peer);
            }

            println!("[ZynkSync] ✓ Bidirectional pairing complete - client added to peer list");

            // Get host's user_id for identity sync and security validation
            let host_user_id = match user_identity::get_user_id() {
                Ok(uid) => {
                    println!("[ZynkSync] Host user_id: {}", uid);
                    Some(uid)
                }
                Err(e) => {
                    println!("[ZynkSync] Warning: Could not get host user_id: {} (pairing will proceed without identity validation)", e);
                    None
                }
            };

            // SMART SECURITY CHECK: Evaluate pairing safety
            // Option 4: Allow if user_ids match OR client has 0 memories (new device)
            // Warn if user_ids mismatch AND client has existing memories
            let client_memory_count = request.get("client_memory_count")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32;

            let mut warning_info: Option<serde_json::Value> = None;

            if let (Some(ref host_uid), Some(client_uid)) = (host_user_id.as_ref(), client_user_id) {
                if host_uid.as_str() != client_uid {
                    // Different user_ids detected
                    if client_memory_count > 0 {
                        // WARN: Client has existing memories and will adopt new identity
                        println!("[ZynkSync] ⚠️ WARNING: user_id mismatch with existing data - host: {}, client: {}, client memories: {}",
                            host_uid, client_uid, client_memory_count);

                        warning_info = Some(serde_json::json!({
                            "type": "identity_change",
                            "message": format!(
                                "This device will adopt the identity of '{}' and sync {} existing memories. \
                                 Your current device identity will be replaced.",
                                host_uid, client_memory_count
                            ),
                            "client_user_id": client_uid,
                            "host_user_id": host_uid,
                            "client_memory_count": client_memory_count,
                            "severity": "high"
                        }));
                    } else {
                        // OK: New device (no memories), safe to adopt identity
                        println!("[ZynkSync] ✓ New device setup: user_id will change from {} to {} (0 memories)",
                            client_uid, host_uid);
                    }
                } else {
                    // OK: Same user, multiple devices
                    println!("[ZynkSync] ✓ Security: user_id validated - both devices belong to user {}", host_uid);
                }
            } else if client_user_id.is_none() {
                println!("[ZynkSync] ⚠ Warning: Client did not provide user_id - security validation skipped");
            } else if host_user_id.is_none() {
                println!("[ZynkSync] ⚠ Warning: Host user_id not available - security validation skipped");
            }

            // Return this host's device info including TLS cert and user_id for identity sync
            let mut response = serde_json::json!({
                "device_id": service.device_id,
                "device_name": service.device_name,
                "cert_der": BASE64.encode(&service.cert_der),
            });

            if let Some(uid) = host_user_id {
                response["user_id"] = serde_json::json!(uid);
            }

            // Include warning info if present (frontend will display confirmation dialog)
            if let Some(warning) = warning_info {
                response["warning"] = warning;
            }

            // Log the exact response we're about to send for debugging
            #[cfg(debug_assertions)]
            println!("[ZynkSync] Sending response to client: {}", serde_json::to_string_pretty(&response).unwrap_or_else(|_| "Failed to serialize".to_string()));

            // Validate that the response is proper JSON before sending
            match serde_json::to_string(&response) {
                Ok(json_str) => {
                    println!("[ZynkSync] Response validated as proper JSON ({} bytes)", json_str.len());
                }
                Err(e) => {
                    println!("[ZynkSync] ERROR: Response is not valid JSON: {}", e);
                    return Err(format!("Failed to serialize response: {}", e));
                }
            }

            Ok(Json(response))
        }
        None => {
            // Wrong code — increment failed attempt counter for this IP
            let mut attempts = service.failed_pairing_attempts.write().await;
            let count = attempts.entry(client_ip.clone()).or_insert(0);
            *count += 1;
            println!("[ZynkSync] ✗ Invalid pairing code from {} ({}/5 attempts)", client_ip, count);
            Err("Invalid or expired pairing code".to_string())
        }
    }
}

/// Axum handler for heartbeat pings — updates last_seen_at for the sender
async fn handle_heartbeat(
    State(service): State<Arc<ZynkSyncService>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, String> {
    let device_id = body.get("device_id")
        .and_then(|v| v.as_str())
        .ok_or("Missing device_id")?;

    sqlx::query(
        "UPDATE zynk_devices SET last_seen_at = datetime('now') WHERE device_id = ?"
    )
    .bind(device_id)
    .execute(&service.db_pool)
    .await
    .map_err(|e| format!("Failed to update last_seen_at: {}", e))?;

    Ok(Json(serde_json::json!({ "ok": true })))
}

/// Axum handler for goodbye signal — marks sender as offline immediately
async fn handle_goodbye(
    State(service): State<Arc<ZynkSyncService>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, String> {
    let device_id = body.get("device_id")
        .and_then(|v| v.as_str())
        .ok_or("Missing device_id")?;

    sqlx::query(
        "UPDATE zynk_devices SET last_seen_at = '1970-01-01T00:00:00.000Z' WHERE device_id = ?"
    )
    .bind(device_id)
    .execute(&service.db_pool)
    .await
    .map_err(|e| format!("Failed to set offline: {}", e))?;

    Ok(Json(serde_json::json!({ "ok": true })))
}

/// Axum handler: pause auto-sync on this device (called by a peer broadcasting pause)
async fn handle_pause(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?;
    check_sync_authorized(&service.db_pool, device_id).await?;

    service.stop_auto_sync().await;
    if let Err(e) = crate::save_sync_state(false).await {
        eprintln!("[ZynkSync] Failed to persist pause state: {}", e);
    }
    if let Ok(app_guard) = crate::APP_HANDLE.lock() {
        if let Some(app) = app_guard.as_ref() {
            let _ = app.emit("zynksync-status-changed", serde_json::json!({"status": "paused"}));
        }
    }
    println!("[ZynkSync] ✅ Paused by peer {}", device_id);
    Ok(Json(serde_json::json!({"success": true, "action": "paused"})))
}

/// Axum handler: resume auto-sync on this device (called by a peer broadcasting resume)
async fn handle_resume(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?;
    check_sync_authorized(&service.db_pool, device_id).await?;

    {
        let mut enabled = service.auto_sync_enabled.write().await;
        *enabled = true;
    }
    if let Err(e) = crate::save_sync_state(true).await {
        eprintln!("[ZynkSync] Failed to persist resume state: {}", e);
    }
    if let Ok(app_guard) = crate::APP_HANDLE.lock() {
        if let Some(app) = app_guard.as_ref() {
            let _ = app.emit("zynksync-status-changed", serde_json::json!({"status": "running"}));
        }
    }
    println!("[ZynkSync] ✅ Resumed by peer {}", device_id);
    Ok(Json(serde_json::json!({"success": true, "action": "resumed"})))
}

/// Axum handler for unsync push notification.
/// Called by the peer that initiated the unsync; removes them from our device list
/// and emits a UI event so the frontend refreshes without the user having to act.
async fn handle_notify_unsynced(
    State(service): State<Arc<ZynkSyncService>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, String> {
    let removed_device_id = payload.get("removed_device_id")
        .and_then(|v| v.as_str())
        .ok_or("Missing removed_device_id")?;

    println!("[ZynkSync] Peer {} initiated unsync — removing from local list", &removed_device_id[..removed_device_id.len().min(8)]);

    // Best-effort — device may already be gone or never fully paired.
    // Call db_only to avoid firing a return notification (round-trip loop).
    match service.clear_sync_data_db_only(removed_device_id).await {
        Ok(_) => println!("[ZynkSync] ✓ Removed peer device on their request"),
        Err(e) => println!("[ZynkSync] Note: clear_sync_data_db_only on notify-unsynced failed (non-fatal): {}", e),
    }

    if let Ok(guard) = crate::APP_HANDLE.lock() {
        if let Some(app) = guard.as_ref() {
            let _ = app.emit("zynksync-device-removed", serde_json::json!({
                "device_id": removed_device_id
            }));
            // Unlink is now a unified teardown, so the ZynkLink panel must refresh too.
            let _ = app.emit("zynklink-pairing-updated", serde_json::json!({
                "unlinked": true
            }));
        }
    }

    Ok(Json(serde_json::json!({ "success": true })))
}

/// Axum handler for receiving chat messages from a peer device
async fn handle_zchat_deliver(
    State(service): State<Arc<ZynkSyncService>>,
    Json(messages): Json<Vec<zchat::DeliverMessageData>>,
) -> Result<Json<serde_json::Value>, String> {
    println!("[ZChat] Received {} messages from peer", messages.len());

    // Call zchat deliver_messages function
    let result = zchat::deliver_messages(&service.db_pool, messages).await?;

    Ok(Json(serde_json::json!({
        "success": true,
        "delivered": result.received_count
    })))
}

/// Axum handler for verifying sync codes (device-to-device authentication)
async fn handle_verify_sync_code(
    State(service): State<Arc<ZynkSyncService>>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, String> {
    let code = request.get("code")
        .and_then(|c| c.as_str())
        .ok_or("Missing code parameter")?;

    println!("[SyncCode] Verifying code: {}", code);

    // Query database to verify the sync code
    let result = sqlx::query_as::<_, (String, String)>(
        "SELECT user_id, device_id FROM user_sync_codes WHERE code = ? AND expires_at > datetime('now')"
    )
    .bind(code)
    .fetch_optional(&service.db_pool)
    .await
    .map_err(|e| format!("Database query failed: {}", e))?;

    match result {
        Some(record) => {
            println!("[SyncCode] Code verified for user: {}", record.0);
            Ok(Json(serde_json::json!({
                "user_id": record.0,
                "device_id": record.1
            })))
        }
        None => {
            Err("Invalid or expired sync code".to_string())
        }
    }
}

/// Axum handler for consuming sync codes (Device B notifies Device A of successful pairing)
async fn handle_consume_sync_code(
    State(service): State<Arc<ZynkSyncService>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, String> {
    let code = request.get("code")
        .and_then(|c| c.as_str())
        .ok_or("Missing code parameter")?;
    let remote_device_id = request.get("remote_device_id")
        .and_then(|v| v.as_str())
        .ok_or("Missing remote_device_id parameter")?;

    // Mark the code as used so it cannot be reused
    sqlx::query(
        "UPDATE user_sync_codes SET used = 1, used_at = datetime('now')
         WHERE code = ? AND used = 0"
    )
    .bind(code)
    .execute(&service.db_pool)
    .await
    .map_err(|e| format!("Failed to consume sync code: {}", e))?;

    // Record the remote device as paired (IP comes from TCP connection, not request body)
    let client_ip = addr.ip().to_string();
    sqlx::query(
        "INSERT INTO zynk_devices (device_id, device_name, device_ip, port, device_platform, is_paired, sync_paired, last_seen_at, created_at)
         VALUES (?, 'Remote Device', ?, 57963, '', 1, 1, datetime('now'), datetime('now'))
         ON CONFLICT (device_id) DO UPDATE
         SET device_ip = ?, is_paired = 1, sync_paired = 1, last_seen_at = datetime('now')"
    )
    .bind(remote_device_id)
    .bind(&client_ip)
    .bind(&client_ip)
    .execute(&service.db_pool)
    .await
    .map_err(|e| format!("Failed to record paired device: {}", e))?;

    let short_id = &remote_device_id[..remote_device_id.len().min(8)];
    println!("[SyncCode] Code consumed, paired with device: {}...", short_id);

    Ok(Json(serde_json::json!({ "success": true })))
}

/// Axum handler for getting memory inventory (for "active device wins" sync)
async fn handle_get_inventory(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<InventoryRequest>,
) -> Result<Json<MemoryInventory>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?;
    check_sync_authorized(&service.db_pool, device_id).await?;

    let inventory = service.get_local_inventory(&request.user_id).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;
    Ok(Json(inventory))
}

/// Axum handler for deleting memories by ID (used when remote is active and doesn't have them)
async fn handle_delete_memories(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
    Json(ids): Json<Vec<i32>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?;
    check_sync_authorized(&service.db_pool, device_id).await?;

    println!("[ZynkSync] Delete request for {} memories", ids.len());
    let deleted_count = service.delete_memories_by_ids(&ids).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;

    Ok(Json(serde_json::json!({ "success": true, "deleted": deleted_count })))
}

/// Axum handler for fetching specific memories by ID (used when remote needs to pull our memories)
async fn handle_fetch_memories(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
    Json(ids): Json<Vec<i32>>,
) -> Result<Json<Vec<SyncMemory>>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?;
    check_sync_authorized(&service.db_pool, device_id).await?;

    println!("[ZynkSync] Fetch request for {} memories", ids.len());
    let memories = service.get_memories_by_ids(&ids).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;
    Ok(Json(memories))
}

/// Axum handler for deleting a memory by content hash (portable across machines)
/// Used for real-time deletion propagation when user manually deletes a memory
async fn handle_delete_by_hash(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?;
    check_sync_authorized(&service.db_pool, device_id).await?;

    let content_hash = request.get("content_hash")
        .and_then(|v| v.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing content_hash parameter"}))))?;

    println!("[ZynkSync] Delete-by-hash request for hash: {}", content_hash);

    let memory_id: Option<i32> = {
        use sha2::{Digest, Sha256};
        let rows: Vec<(i32, String)> = sqlx::query_as::<_, (i32, String)>(
            "SELECT id, content FROM memories"
        )
        .fetch_all(&service.db_pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("Failed to lookup memories: {}", e)}))))?;
        rows.into_iter()
            .find(|(_, c)| format!("{:x}", Sha256::digest(c.as_bytes())) == content_hash)
            .map(|(id, _)| id)
    };

    match memory_id {
        Some(id) => {
            println!("[ZynkSync] Found memory ID {} for hash {}", id, content_hash);
            let result = sqlx::query("DELETE FROM memories WHERE id = ?")
                .bind(id)
                .execute(&service.db_pool)
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("Failed to delete memory: {}", e)}))))?;
            let deleted_count = result.rows_affected();
            println!("[ZynkSync] Deleted {} memory(s)", deleted_count);
            Ok(Json(serde_json::json!({ "success": true, "deleted": deleted_count })))
        }
        None => {
            println!("[ZynkSync] No memory found with hash {}", content_hash);
            Ok(Json(serde_json::json!({ "success": true, "deleted": 0 })))
        }
    }
}

// =============================================================================
// ZynkLink HTTP Handlers - File Sharing Between Devices
// =============================================================================

/// Verify a ZynkLink code (like verify_sync_code but for file sharing)
async fn handle_zynklink_verify_code(
    State(service): State<Arc<ZynkSyncService>>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let code = request.get("code")
        .and_then(|c| c.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "Missing code parameter".to_string()))?;

    println!("[ZynkLink] Verifying code: {}", code);

    // Query database to verify the ZynkLink code
    let result = sqlx::query_as::<_, (String, String)>(
        "SELECT creator_user_id, creator_device_id FROM zynklink_codes
         WHERE code = ? AND expires_at > datetime('now') AND is_active = 1"
    )
    .bind(code)
    .fetch_optional(&service.db_pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Database query failed: {}", e)))?;

    match result {
        Some(record) => {
            println!("[ZynkLink] Code verified for user: {}", record.0);
            Ok(Json(serde_json::json!({
                "user_id": record.0,
                "device_id": record.1
            })))
        }
        None => {
            println!("[ZynkLink] Code not found or expired: {}", code);
            Err((StatusCode::NOT_FOUND, "Invalid or expired ZynkLink code".to_string()))
        }
    }
}

/// Accept a ZynkLink code and create pairing
async fn handle_zynklink_accept_code(
    State(service): State<Arc<ZynkSyncService>>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let code = request.get("code")
        .and_then(|c| c.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "Missing code parameter".to_string()))?;

    let acceptor_user_id = request.get("acceptor_user_id")
        .and_then(|u| u.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "Missing acceptor_user_id parameter".to_string()))?;

    let acceptor_device_id = request.get("acceptor_device_id")
        .and_then(|d| d.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "Missing acceptor_device_id parameter".to_string()))?;

    let acceptor_device_ip = request.get("acceptor_device_ip")
        .and_then(|ip| ip.as_str());

    println!("[ZynkLink] Accept code request: {} from user: {}..., IP: {:?}",
             code, &acceptor_user_id[..8], acceptor_device_ip);

    // Ensure acceptor's device exists in zynk_devices (required for foreign key constraint)
    println!("[ZynkLink] Device A: Ensuring acceptor's device is registered...");
    sqlx::query(
        "INSERT INTO zynk_devices (device_id, device_name, device_ip, owner_user_id, is_paired, port, created_at, last_seen_at)
         VALUES (?, ?, ?, ?, true, 57963, datetime('now'), datetime('now'))
         ON CONFLICT (device_id) DO UPDATE
         SET device_ip = ?, owner_user_id = ?, last_seen_at = datetime('now')"
    )
    .bind(acceptor_device_id)
    .bind(&format!("Remote Device {}", &acceptor_device_id[..8]))
    .bind(acceptor_device_ip)
    .bind(acceptor_user_id)
    .bind(acceptor_device_ip)
    .bind(acceptor_user_id)
    .execute(&service.db_pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to ensure acceptor device entry: {}", e)))?;

    println!("[ZynkLink] Device A: Acceptor device registered successfully");

    // Get Device A's IP address to send back to Device B
    let creator_device_ip = {
        use std::net::UdpSocket;
        match UdpSocket::bind("0.0.0.0:0") {
            Ok(socket) => {
                match socket.connect("8.8.8.8:80") {
                    Ok(_) => {
                        socket.local_addr()
                            .map(|addr| addr.ip().to_string())
                            .ok()
                    }
                    Err(_) => None
                }
            }
            Err(_) => None
        }
    };

    println!("[ZynkLink] Device A: Our IP: {:?}", creator_device_ip);

    // Use the zynklink module function
    println!("[ZynkLink] Device A: Calling accept_zynklink_code...");
    let mut result = match crate::zynklink::accept_zynklink_code(
        &service.db_pool,
        code,
        acceptor_user_id,
        acceptor_device_id
    ).await {
        Ok(r) => {
            println!("[ZynkLink] Device A: ✅ Pairing created successfully");
            r
        }
        Err(e) => {
            println!("[ZynkLink] Device A: ❌ Failed to create pairing: {}", e);
            return Err((StatusCode::BAD_REQUEST, e));
        }
    };

    // Store Device A's (creator's) own IP in the database
    if let Some(ref creator_ip) = creator_device_ip {
        // Get the creator device ID from the code record
        let creator_device_id = sqlx::query_scalar::<_, String>(
            "SELECT creator_device_id FROM zynklink_codes WHERE code = ?"
        )
        .bind(code)
        .fetch_one(&service.db_pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to get creator device ID: {}", e)))?;

        println!("[ZynkLink] Device A: Storing our own IP {} in database", creator_ip);
        sqlx::query(
            "UPDATE zynk_devices SET device_ip = ?, last_seen_at = datetime('now') WHERE device_id = ?"
        )
        .bind(creator_ip)
        .bind(&creator_device_id)
        .execute(&service.db_pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to update creator device IP: {}", e)))?;

        println!("[ZynkLink] Device A: ✅ Our IP stored successfully");
    }

    // Add Device A's IP to the response so Device B can store it
    if let Some(result_obj) = result.as_object_mut() {
        if let Some(creator_ip) = creator_device_ip {
            result_obj.insert("creator_device_ip".to_string(), serde_json::Value::String(creator_ip));
        }
    }

    // Emit event to refresh UI immediately on Device A (the code creator)
    println!("[ZynkLink] Device A: Attempting to emit zynklink-pairing-updated event");
    match crate::APP_HANDLE.lock() {
        Ok(app_handle_guard) => {
            match app_handle_guard.as_ref() {
                Some(app_handle) => {
                    println!("[ZynkLink] Device A: APP_HANDLE acquired, emitting event");
                    match app_handle.emit("zynklink-pairing-updated", serde_json::json!({
                        "acceptor_user_id": acceptor_user_id,
                        "acceptor_device_id": acceptor_device_id
                    })) {
                        Ok(_) => println!("[ZynkLink] Device A: ✅ Event emitted successfully"),
                        Err(e) => println!("[ZynkLink] Device A: ❌ Failed to emit event: {}", e),
                    }
                }
                None => {
                    println!("[ZynkLink] Device A: ⚠️ APP_HANDLE is None - cannot emit event");
                }
            }
        }
        Err(e) => {
            println!("[ZynkLink] Device A: ❌ Failed to lock APP_HANDLE: {}", e);
        }
    }

    Ok(Json(result))
}

/// Check that the requesting device_id has an active entry in zynk_device_pairings.
/// Used to reject sync requests from devices that have been removed on this side.
async fn check_sync_authorized(
    pool: &sqlx::SqlitePool,
    device_id: &str,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    // Check zynk_devices (not zynk_device_pairings) because the pairing row is
    // created by the first sync itself — using it as the gate causes a
    // chicken-and-egg rejection of every first sync from a newly added device.
    // zynk_devices.sync_paired=1 is set during the ZynkSync pairing handshake and
    // cleared by clear_sync_data_db_only(), so it's the correct liveness indicator.
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM zynk_devices WHERE device_id = ? AND sync_paired = 1"
    )
    .bind(device_id)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))))?;

    if count == 0 {
        println!("[ZynkSync] Rejected sync from unpaired device {}", &device_id[..device_id.len().min(8)]);
        return Err((StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": "Device not paired"}))));
    }
    Ok(())
}

/// List shared directories from a device
/// Check that requester_user_id has an active ZynkLink pairing with this device's user.
async fn check_zynklink_authorized(pool: &sqlx::SqlitePool, requester_user_id: &str) -> Result<(), String> {
    let local_user_id = crate::user_identity::get_user_id()
        .map_err(|e| format!("Failed to get local user ID: {}", e))?;
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM zynklink_pairings
         WHERE is_active = 1
         AND ((user1_id = ? AND user2_id = ?) OR (user1_id = ? AND user2_id = ?))"
    )
    .bind(&local_user_id).bind(requester_user_id)
    .bind(requester_user_id).bind(&local_user_id)
    .fetch_one(pool)
    .await
    .map_err(|e| format!("Failed to check ZynkLink authorization: {}", e))?;
    if count == 0 {
        Err("Not authorized: no active ZynkLink pairing with this user".to_string())
    } else {
        Ok(())
    }
}

async fn handle_zynklink_directories(
    State(service): State<Arc<ZynkSyncService>>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let requester_user_id = request.get("requester_user_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing requester_user_id"}))))?;

    check_zynklink_authorized(&service.db_pool, requester_user_id).await
        .map_err(|e| (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": e}))))?;

    let local_device_id = crate::user_identity::get_device_id()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;
    let response = crate::zynklink::list_my_shared_directories(
        &service.db_pool,
        &local_device_id
    ).await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;

    serde_json::to_value(response)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))))
}

/// List files in a shared directory
async fn handle_zynklink_files(
    State(service): State<Arc<ZynkSyncService>>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let requester_user_id = request.get("requester_user_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing requester_user_id"}))))?;

    check_zynklink_authorized(&service.db_pool, requester_user_id).await
        .map_err(|e| (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": e}))))?;

    let share_id = request.get("share_id")
        .and_then(|s| s.as_i64())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing share_id parameter"}))))? as i32;

    println!("[ZynkLink] File list request for share_id: {}", share_id);

    // Rescan the directory before listing so newly added files always appear.
    // All file types are indexed; KB-format filtering happens on the client side.
    let _ = crate::zynklink::scan_directory(
        &service.db_pool,
        &service.device_id,
        share_id,
        None,
    ).await;

    // List files in the shared directory
    let response = crate::zynklink::list_files(
        &service.db_pool,
        share_id
    ).await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;

    serde_json::to_value(response)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))))
}

/// Download a file from a shared directory
async fn handle_zynklink_download(
    State(service): State<Arc<ZynkSyncService>>,
    Json(request): Json<serde_json::Value>,
) -> Result<axum::response::Response, (StatusCode, Json<serde_json::Value>)> {
    use tokio::io::AsyncReadExt;

    let requester_user_id = request.get("requester_user_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing requester_user_id"}))))?;

    check_zynklink_authorized(&service.db_pool, requester_user_id).await
        .map_err(|e| (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": e}))))?;

    let share_id = request.get("share_id")
        .and_then(|s| s.as_i64())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing share_id parameter"}))))? as i32;

    let relative_path = request.get("relative_path")
        .and_then(|p| p.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing relative_path parameter"}))))?;

    println!("[ZynkLink] Download request for share_id: {}, path: {}", share_id, relative_path);

    let file_path = crate::zynklink::get_file_path(
        &service.db_pool,
        share_id,
        relative_path
    ).await.map_err(|e| (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": e}))))?;

    // Open the file and read metadata for the Content-Length header. Previously this
    // function read the entire file into a single Vec<u8> via tokio::fs::read(), which
    // allocated as many GBs as the file is large — a 4GB gguf transfer would allocate
    // 4GB on the sender. Now we stream 64KB chunks via futures::stream::unfold so
    // memory stays bounded regardless of file size.
    let file = tokio::fs::File::open(&file_path).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("Failed to open file: {}", e)}))))?;
    let file_size = file.metadata().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("Failed to read file metadata: {}", e)}))))?
        .len();

    let filename = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("download")
        .to_string();

    println!("[ZynkLink] Streaming {} ({} bytes)", filename, file_size);

    let stream = futures::stream::unfold(file, |mut file| async move {
        let mut buf = vec![0u8; 65536];
        match file.read(&mut buf).await {
            Ok(0) => None,
            Ok(n) => {
                buf.truncate(n);
                Some((Ok::<_, std::io::Error>(buf), file))
            }
            Err(e) => Some((Err(e), file)),
        }
    });

    axum::response::Response::builder()
        .header("Content-Type", "application/octet-stream")
        .header("Content-Length", file_size.to_string())
        .header("Content-Disposition", format!("attachment; filename=\"{}\"", filename))
        .body(axum::body::Body::from_stream(stream))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))))
}

/// Receive and store chat messages from a ZynkLink-paired device
async fn handle_zynklink_deliver_chat(
    State(service): State<Arc<ZynkSyncService>>,
    Json(messages): Json<Vec<crate::zchat::DeliverMessageData>>,
) -> Result<axum::Json<crate::zchat::DeliverMessagesResponse>, String> {
    println!("[ZynkLink] Received {} chat message(s) for delivery", messages.len());

    // Call zchat deliver_messages function
    let result = crate::zchat::deliver_messages(&service.db_pool, messages).await?;

    println!("[ZynkLink] ✓ Delivered {} message(s)", result.received_count);
    Ok(axum::Json(result))
}

/// Receive conversation sessions and messages from a peer device
async fn handle_receive_conversations(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<ConversationSyncPayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?;
    check_sync_authorized(&service.db_pool, device_id).await?;

    let (sessions_stored, messages_stored) = service.receive_conversations_from_peer(payload).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;

    Ok(Json(serde_json::json!({
        "success": true,
        "sessions_stored": sessions_stored,
        "messages_stored": messages_stored
    })))
}

/// Handle notification that a remote device has unlinked.
/// Removes the local pairing record and emits a UI refresh event.
async fn handle_zynklink_notify_unpaired(
    State(service): State<Arc<ZynkSyncService>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<axum::Json<serde_json::Value>, String> {
    let unlinked_user_id = payload.get("unlinked_user_id")
        .and_then(|v| v.as_str())
        .ok_or("Missing unlinked_user_id")?;

    let local_user_id = crate::user_identity::get_user_id()
        .map_err(|e| format!("Failed to get local user ID: {}", e))?;
    let local_device_id = crate::user_identity::get_device_id().unwrap_or_default();

    // Resolve the peer's device_id before we clear (clear_link_data removes the pairing row)
    let device_ids = sqlx::query_as::<_, (String, String)>(
        "SELECT device1_id, device2_id FROM zynklink_pairings
         WHERE ((user1_id = ? AND user2_id = ?) OR (user1_id = ? AND user2_id = ?))"
    )
    .bind(&local_user_id).bind(unlinked_user_id)
    .bind(unlinked_user_id).bind(&local_user_id)
    .fetch_optional(&service.db_pool)
    .await
    .ok()
    .flatten();

    let peer_device_id = device_ids.as_ref().map(|(d1, d2)| {
        if d1 == &local_device_id { d2.clone() } else { d1.clone() }
    });

    // Delegate full ZynkLink cleanup to clear_link_data — preserves ZynkSync if active
    if let Some(ref peer_id) = peer_device_id {
        match service.clear_link_data(peer_id).await {
            Ok(_) => println!("[ZynkLink] ✓ Remote unlink: cleared link data for peer {}", &peer_id[..peer_id.len().min(8)]),
            Err(e) => println!("[ZynkLink] Note: clear_link_data on notify-unpaired failed (non-fatal): {}", e),
        }
    }

    if let Ok(guard) = crate::APP_HANDLE.lock() {
        if let Some(app) = guard.as_ref() {
            let _ = app.emit("zynklink-pairing-updated", serde_json::json!({
                "unlinked": true,
                "remote_user_id": unlinked_user_id
            }));
        }
    }

    Ok(axum::Json(serde_json::json!({ "success": true })))
}

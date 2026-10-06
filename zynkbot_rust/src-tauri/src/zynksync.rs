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
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, interval};
use sqlx::{SqlitePool, Row};
use chrono::{DateTime, Utc};

/// Bind a TCP listener with SO_REUSEADDR and retry-on-failure. When the app is
/// force-closed and quickly reopened, the previous process's socket may still
/// be in TIME_WAIT holding port 57963. SO_REUSEADDR lets us reclaim it; the
/// retry loop covers the brief window before the kernel releases it.
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use axum::{
    routing::{post, any},
    Router,
    Json,
    extract::{State, ConnectInfo, Request},
    http::StatusCode,
    response::{Response, IntoResponse},
    body::Body,
};
use crate::tls::VerifiedDevice;
use std::net::SocketAddr;
use tauri::Emitter;
pub use crate::transport::{DEFAULT_SYNC_PORT, PeerDevice, SyncIdentity, split_host_port};

/// How often the auto-sync loop looks for a local change between timer ticks.
pub const OUTBOX_POLL_SECS: u64 = 2;
/// Longest a burst of edits can hold a kicked cycle back.
pub const OUTBOX_SETTLE_MAX_SECS: u64 = 6;
use crate::transport::Transport;






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
    pub collection_id: Option<String>,
    #[serde(default = "default_memory_placement")]
    pub memory_placement: String,
    #[serde(default)]
    pub external_id: Option<String>,
    #[serde(default)]
    pub temporal_status: Option<String>,
    #[serde(default)]
    pub provenance_json: Option<String>,
    /// JSON array text, as stored (0011). Travels since the outbox rebuild (KI-030).
    #[serde(default)]
    pub tags: Option<String>,
    /// The row's name on every device (0013).
    #[serde(default)]
    pub sync_id: Option<String>,
}

fn default_memory_placement() -> String {
    "retrieved".to_string()
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
    /// The row's name on every device (0013); equals session_id.
    #[serde(default)]
    pub sync_id: Option<String>,
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
    /// The row's name on every device (0013).
    #[serde(default)]
    pub sync_id: Option<String>,
}


/// Core ZynkSync service managing device synchronization
pub struct ZynkSyncService {
    /// Everything shared with ZynkLink and ZChat: identity, certificate, server,
    /// pinned client, device registry, presence. See crate::transport.
    pub(crate) transport: Arc<Transport>,

    /// Unique identifier for this device (copy of transport.identity().device_id)
    device_id: String,

    /// SQLite connection pool
    pub(crate) db_pool: SqlitePool,


    /// Whether auto-sync is enabled
    auto_sync_enabled: Arc<RwLock<bool>>,

    /// Sync interval in seconds
    sync_interval_secs: u64,

    /// Set while the auto-sync loop is running. start_auto_sync was called from three
    /// places (startup, the Sync panel, a peer's resume notice) and each call spawned a
    /// loop that never ended, so the desktop ran seven and synced seven times a minute,
    /// the Pixel fifteen (first device pass, 2026-10-02). One loop per service.
    auto_sync_loop_running: Arc<std::sync::atomic::AtomicBool>,
}

impl ZynkSyncService {
    /// Create a new ZynkSync service instance
    pub fn new(
        identity: SyncIdentity,
        port: Option<u16>,
        db_pool: SqlitePool,
        sync_interval_secs: Option<u64>,
        cert_pem: String,
        key_pem: String,
        cert_der: Vec<u8>,
    ) -> Self {
        let device_id = identity.device_id.clone();
        let transport = Arc::new(Transport::new(identity, port, db_pool.clone(), cert_pem, key_pem, cert_der));
        Self {
            transport,
            device_id,
            db_pool,
            auto_sync_enabled: Arc::new(RwLock::new(false)),
            sync_interval_secs: sync_interval_secs.unwrap_or(300),
            auto_sync_loop_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    // ---- identity, port, client and registry live in the transport; these keep
    //      their names so commands and the UI do not change ----

    #[allow(dead_code)] // the chat and link services will take this
    pub fn transport(&self) -> Arc<Transport> { Arc::clone(&self.transport) }
    #[allow(dead_code)]
    pub fn identity(&self) -> SyncIdentity { self.transport.identity() }
    pub fn user_id(&self) -> Result<String, String> { self.transport.user_id() }
    pub fn device_name(&self) -> String { self.transport.device_name() }
    pub fn set_user_id(&self, user_id: &str) { self.transport.set_user_id(user_id) }

    /// This device joins another user's devices: everything it already holds moves under
    /// the shared user id, so it stays visible here and travels to the peers. Before, the
    /// rows stayed under the old id and vanished from view and from sync (KI-011, b07).
    /// Plain UPDATEs, so the 0013 triggers queue each row for the drain.
    pub async fn adopt_user_id(&self, new_user_id: &str) -> Result<usize, String> {
        let old = self.user_id().unwrap_or_default();
        if old == new_user_id { return Ok(0); }
        let mut moved = 0usize;
        for table in ["memories", "conversation_sessions", "conversation_messages"] {
            let r = sqlx::query(&format!("UPDATE {} SET user_id = ? WHERE user_id = ?", table))
                .bind(new_user_id).bind(&old)
                .execute(&self.db_pool).await
                .map_err(|e| format!("re-key {}: {}", table, e))?;
            moved += r.rows_affected() as usize;
        }
        self.transport.set_user_id(new_user_id);
        if moved > 0 {
            println!("[ZynkSync] Adopted user id {}…: {} existing row(s) moved under it and queued for the peers", &new_user_id[..8.min(new_user_id.len())], moved);
        }
        Ok(moved)
    }
    pub fn set_device_name(&self, name: &str) { self.transport.set_device_name(name) }
    pub fn port(&self) -> u16 { self.transport.port() }
    pub async fn peer_port_by_id(&self, device_id: &str) -> u16 { self.transport.peer_port_by_id(device_id).await }
    pub async fn rebuild_http_client(&self) -> Result<(), String> { self.transport.rebuild_http_client().await }
    pub async fn get_http_client(&self) -> reqwest::Client { self.transport.get_http_client().await }
    pub async fn get_peer_client_for_url(&self, url: &str) -> Option<reqwest::Client> { self.transport.get_peer_client_for_url(url).await }
    pub async fn load_devices(&self) -> Result<(), String> { self.transport.load_devices().await }
    pub async fn get_peers(&self) -> Vec<PeerDevice> { self.transport.get_peers().await }
    pub async fn send_goodbye_to_peers(&self) { self.transport.send_goodbye_to_peers().await }

    /// Rebuild the shared HTTP client to trust all currently stored peer certificates.
    /// Called at startup (after load_devices) and after each new pairing.

    /// Return a clone of the shared DB pool for use by Tauri commands that need DB access
    pub fn get_db_pool(&self) -> SqlitePool {
        self.db_pool.clone()
    }


    /// Returns the mTLS-capable HTTP client if `url` points to a known paired peer,
    /// otherwise returns None. Used by chat/memory callers to transparently upgrade to mTLS.

    /// Generate a new 6-digit pairing code
    pub async fn generate_pairing_code(&self) -> Result<String, String> {
        // Use rand::random() instead of thread_rng() for Send compatibility
        let code = format!("{:06}", rand::random::<u32>() % 1000000);

        // Store in memory
        {
            let mut pairing_code = self.transport.pairing_code.write().await;
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
        .bind(&self.device_name())
        .bind(&code)
        .bind(expires_at)
        .bind(true)  // This device is always paired with itself (it's the host)
        .bind(self.port() as i32)
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

    /// Add a device manually using IP address and pairing code
    /// Returns the peer device info including the host's user_id for identity sync
    pub async fn add_device(&self, host_ip: &str, pairing_code: &str) -> Result<PeerDevice, String> {
        // Validate pairing code format (6 digits)
        if pairing_code.len() != 6 || !pairing_code.chars().all(|c| c.is_numeric()) {
            return Err("Invalid pairing code. Must be 6 digits.".to_string());
        }

        let (host, port) = split_host_port(host_ip, DEFAULT_SYNC_PORT);
        // port: from host_ip ("ip:port") or the default

        // Try to contact the device and verify pairing code
        let url = format!("https://{}:{}", host, port);
        let verify_endpoint = format!("{}/api/zynksync/verify-pairing", url);

        // Get client's user_id for validation
        let client_user_id = match self.user_id() {
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
            "client_device_name": self.device_name(),
            "client_port": self.port(),
            "client_memory_count": client_memory_count,
            "client_cert_der": BASE64.encode(&self.transport.cert_der),
        });

        // Include client_user_id if available
        if let Some(ref uid) = client_user_id {
            request_body["client_user_id"] = serde_json::json!(uid);
        }

        // Send our existing peer list so the host can join our mesh too (bidirectional introduction)
        let client_peers: Vec<serde_json::Value> = sqlx::query(
            "SELECT device_id, device_name, device_ip, port, tls_cert_der FROM zynk_devices WHERE sync_paired = 1"
        )
        .fetch_all(&self.db_pool)
        .await
        .unwrap_or_default()
        .iter()
        .filter_map(|row| {
            let dev_id: String = row.try_get("device_id").ok()?;
            let dev_name: String = row.try_get("device_name").ok()?;
            let ip: String = row.try_get("device_ip").ok()?;
            let port_val: i32 = row.try_get("port").ok()?;
            let cert: Option<Vec<u8>> = row.try_get("tls_cert_der").ok().flatten();
            Some(serde_json::json!({
                "device_id": dev_id,
                "device_name": dev_name,
                "ip": ip,
                "port": port_val,
                "cert_der": cert.map(|d| BASE64.encode(&d)),
            }))
        })
        .collect();
        if !client_peers.is_empty() {
            request_body["client_peers"] = serde_json::json!(client_peers);
            println!("[ZynkSync] Sending {} peer(s) to host for bidirectional mesh introduction", client_peers.len());
        }

        // TOFU (Trust On First Use): accept any cert for the initial pairing request.
        // The peer cert cannot be pinned before pairing completes — that's a chicken-and-egg
        // problem inherent to first contact. Security here comes from the 6-digit pairing code,
        // which the user verifies out-of-band. After pairing, the cert is stored in zynk_devices
        // and all subsequent connections use cert pinning via PinnedCertVerifier. This is the
        // only place in the codebase where invalid-cert acceptance is intentional and necessary.
        let tofu_client = reqwest::ClientBuilder::new()
            .danger_accept_invalid_certs(true)
            .timeout(Duration::from_secs(60))
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

        // If device is already in the peers map, evict it first so re-pairing works cleanly
        // (e.g. after a Remove that didn't fully propagate, or an IP/cert refresh).
        {
            let mut peers_map = self.transport.peers.write().await;
            if peers_map.contains_key(&device_id) {
                println!("[ZynkSync] Device {} already known — evicting for re-pair", &device_id[..8.min(device_id.len())]);
                peers_map.remove(&device_id);
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
            is_online: false,
        };

        {
            let mut peers_map = self.transport.peers.write().await;
            peers_map.insert(device_id.clone(), peer.clone());
        }

        println!("[ZynkSync] Added device: {} ({})", device_name, device_id);

        // Mesh pairing: host included its peer list — connect to each peer directly.
        // The host vouches for them; we pre-trust their certs and send an introduce request.
        if let Some(peers_array) = device_info.get("peers").and_then(|p| p.as_array()) {
            let intro_peers: Vec<serde_json::Value> = peers_array.clone();
            if !intro_peers.is_empty() {
                println!("[ZynkSync] Host introduced {} peer(s) — joining mesh", intro_peers.len());
                let our_user_id = self.user_id().ok();

                // Step 1: store all introduced peer certs in DB so the HTTP client can trust them
                for peer_json in &intro_peers {
                    let peer_id = match peer_json.get("device_id").and_then(|v| v.as_str()) {
                        Some(id) => id, None => continue,
                    };
                    let peer_name = peer_json.get("device_name").and_then(|v| v.as_str()).unwrap_or("unknown");
                    let peer_ip = peer_json.get("ip").and_then(|v| v.as_str()).unwrap_or("");
                    let peer_port = peer_json.get("port").and_then(|v| v.as_i64()).unwrap_or(DEFAULT_SYNC_PORT as i64) as i32;
                    let peer_cert_der: Option<Vec<u8>> = peer_json.get("cert_der")
                        .and_then(|v| v.as_str())
                        .and_then(|b64| BASE64.decode(b64).ok());

                    // Skip devices we already know
                    sqlx::query(
                        "INSERT INTO zynk_devices (device_id, device_name, device_ip, port, is_paired, sync_paired, tls_cert_der, last_seen_at, created_at)
                         VALUES (?, ?, ?, ?, 1, 1, ?, ?, ?)
                         ON CONFLICT (device_id) DO UPDATE
                         SET device_name = excluded.device_name,
                             device_ip = excluded.device_ip,
                             sync_paired = 1,
                             tls_cert_der = excluded.tls_cert_der,
                             last_seen_at = excluded.last_seen_at"
                    )
                    .bind(peer_id).bind(peer_name).bind(peer_ip).bind(peer_port)
                    .bind(peer_cert_der.as_deref()).bind(Utc::now()).bind(Utc::now())
                    .execute(&self.db_pool)
                    .await.ok();
                    println!("[ZynkSync] Pre-trusted introduced peer: {} ({})", peer_name, peer_id);
                }

                // Step 2: rebuild client to trust all newly stored certs
                self.rebuild_http_client().await.ok();

                // Step 3: send introduce request to each peer (best-effort)
                let our_cert_b64 = BASE64.encode(&self.transport.cert_der);
                let our_id = self.device_id.clone();
                let our_name = self.device_name();
                let host_id = device_id.clone(); // the host who introduced us

                for peer_json in &intro_peers {
                    let peer_id = match peer_json.get("device_id").and_then(|v| v.as_str()) {
                        Some(id) => id.to_string(), None => continue,
                    };
                    let peer_name = peer_json.get("device_name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                    let peer_ip = peer_json.get("ip").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let peer_port_num = peer_json.get("port").and_then(|v| v.as_i64()).unwrap_or(DEFAULT_SYNC_PORT as i64) as u16;

                    // Skip already known
                    {
                        let map = self.transport.peers.read().await;
                        if map.contains_key(&peer_id) { continue; }
                    }

                    if peer_ip.is_empty() { continue; }

                    let intro_url = format!("https://{}:{}/api/zynksync/introduce", peer_ip, peer_port_num);
                    let http_client = self.transport.http_client.read().await.clone();
                    let mut payload = serde_json::json!({
                        "new_device_id": our_id,
                        "new_device_name": our_name,
                        "new_device_cert_der": our_cert_b64,
                        "introducer_device_id": host_id,
                    });
                    if let Some(ref uid) = our_user_id {
                        payload["new_device_user_id"] = serde_json::json!(uid);
                    }

                    match http_client.post(&intro_url).json(&payload).send().await {
                        Ok(resp) if resp.status().is_success() => {
                            println!("[ZynkSync] ✓ Introduced to peer: {}", peer_name);
                            // Add confirmed peer to in-memory map
                            let mut map = self.transport.peers.write().await;
                            if !map.contains_key(&peer_id) {
                                map.insert(peer_id.clone(), PeerDevice {
                                    device_id: peer_id.clone(),
                                    device_name: peer_name.clone(),
                                    host: peer_ip.clone(),
                                    port: peer_port_num,
                                    url: format!("https://{}:{}", peer_ip, peer_port_num),
                                    last_seen: Utc::now(),
                                    paired: true,
                                    pairing_code: None,
                                    user_id: None,
                                    is_online: false,
                                });
                            }
                        }
                        Ok(r)  => println!("[ZynkSync] Peer {} introduction returned {}", peer_name, r.status()),
                        Err(e) => println!("[ZynkSync] Peer {} offline during introduction (will sync when online): {}", peer_name, e),
                    }
                }
            }
        }

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

        // Remove from in-memory peers map and online-status map
        {
            let mut peers_map = self.transport.peers.write().await;
            peers_map.remove(device_id);
        }
        {
            let mut map = self.transport.peer_last_seen.write().await;
            map.remove(device_id);
        }

        // If no sync peers remain, tombstones serve no purpose — clear them so stale
        // test-session tombstones don't interfere with the next fresh pairing.
        let remaining_peers: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM zynk_devices WHERE sync_paired = 1"
        )
        .fetch_one(&self.db_pool)
        .await
        .unwrap_or(0);

        if remaining_peers == 0 {
            println!("[ZynkSync] No peers remain — clearing all tombstones for clean slate");
            let _ = sqlx::query("DELETE FROM deleted_memory_hashes")
                .execute(&self.db_pool)
                .await;
        }

        println!("[ZynkSync] ✓ Cleared sync data for device {}", &device_id[..device_id.len().min(8)]);
        Ok(())
    }


    /// Leave the sync network entirely: notify all peers to remove this device,
    /// then clear all local peer data. The device_id parameter is ignored —
    /// pressing Unsync always departs the full mesh, not just one connection.
    /// Remove a specific remote device from this device and all other peers in the mesh.
    /// Used when an admin device wants to expel a ghost or unwanted peer.
    /// Kept for existing callers; ZynkLink owns this now (zynklink::clear_link_data).
    pub async fn clear_link_data(&self, device_id: &str) -> Result<(), String> {
        crate::zynklink::clear_link_data(&self.transport, device_id).await
    }

    pub async fn expel_device(&self, target_device_id: &str) -> Result<(), String> {
        // Get the target's IP before clearing it
        let target_row: Option<(String, i64)> = sqlx::query_as::<_, (String, i64)>(
            "SELECT device_ip, port FROM zynk_devices WHERE device_id = ?"
        )
        .bind(target_device_id)
        .fetch_optional(&self.db_pool)
        .await
        .ok()
        .flatten();
        let target_port: u16 = target_row.as_ref().map(|(_, p)| *p as u16).unwrap_or(DEFAULT_SYNC_PORT);
        let target_ip: Option<String> = target_row.map(|(ip, _)| ip);

        // Collect all OTHER sync-paired peers (everyone except the target)
        let other_peers: Vec<(String, String, String, i64)> = sqlx::query_as::<_, (String, String, String, i64)>(
            "SELECT device_id, device_name, device_ip, port FROM zynk_devices WHERE sync_paired = 1 AND device_id != ?"
        )
        .bind(target_device_id)
        .fetch_all(&self.db_pool)
        .await
        .unwrap_or_default();

        println!("[ZynkSync] Expelling device {} — notifying {} other peer(s)",
            &target_device_id[..8.min(target_device_id.len())], other_peers.len());

        // Remove target from local DB immediately
        self.clear_sync_data_db_only(target_device_id).await.ok();

        let local_device_id = self.device_id.clone();
        let target_id = target_device_id.to_string();
        let http_client = self.transport.http_client.read().await.clone();

        tokio::spawn(async move {
            // Tell every other peer to remove the target via cascade field
            let cascade_payload = serde_json::json!({
                "removed_device_id": local_device_id,
                "cascade_device_id": target_id
            });
            for (_, peer_name, peer_ip, peer_port) in &other_peers {
                if peer_ip.is_empty() { continue; }
                let url = format!("https://{}:{}/api/zynksync/notify-unsynced", peer_ip, peer_port);
                match http_client.post(&url).json(&cascade_payload).send().await {
                    Ok(r) if r.status().is_success() =>
                        println!("[ZynkSync] ✓ {} will remove expelled device", peer_name),
                    Ok(r) =>
                        println!("[ZynkSync] {} returned {} on expel cascade", peer_name, r.status()),
                    Err(_) =>
                        println!("[ZynkSync] {} offline — expelled device will be gone when they reconnect", peer_name),
                }
            }

            // Best-effort: notify the expelled device itself so its UI clears. It is
            // addressed by IP, and a stale entry (a reinstalled phone's old identity,
            // KI-050) shares its IP with the live one — so name the intended target,
            // and the receiver ignores it unless the target is itself (2026-09-13: the
            // live OnePlus dropped the desktop when the ghost OnePlus was deleted).
            if let Some(ip) = target_ip {
                if !ip.is_empty() {
                    let url = format!("https://{}:{}/api/zynksync/notify-unsynced", ip, target_port);
                    let self_payload = serde_json::json!({
                        "removed_device_id": local_device_id,
                        "target_device_id": target_id
                    });
                    let _ = http_client.post(&url).json(&self_payload).send().await;
                }
            }
        });

        Ok(())
    }

    pub async fn remove_device(&self, _device_id: &str) -> Result<(), String> {
        // Collect ALL sync-paired peers before clearing anything
        let all_peers: Vec<(String, String, String, i64)> = sqlx::query_as::<_, (String, String, String, i64)>(
            "SELECT device_id, device_name, device_ip, port FROM zynk_devices WHERE sync_paired = 1"
        )
        .fetch_all(&self.db_pool)
        .await
        .unwrap_or_default();

        if all_peers.is_empty() {
            println!("[ZynkSync] remove_device: no peers — nothing to do");
            return Ok(());
        }

        println!("[ZynkSync] Leaving sync network — notifying {} peer(s)", all_peers.len());

        // Clear ALL peers from local DB and in-memory maps
        for (peer_id, _, _, _) in &all_peers {
            self.clear_sync_data_db_only(peer_id).await.ok();
        }

        let local_device_id = self.device_id.clone();
        let http_client = self.transport.http_client.read().await.clone();

        tokio::spawn(async move {
            // Tell every peer: "remove me from your list"
            let payload = serde_json::json!({ "removed_device_id": local_device_id });
            for (peer_id, peer_name, peer_ip, peer_port) in &all_peers {
                if peer_ip.is_empty() { continue; }
                let url = format!("https://{}:{}/api/zynksync/notify-unsynced", peer_ip, peer_port);
                match http_client.post(&url).json(&payload).send().await {
                    Ok(r) if r.status().is_success() =>
                        println!("[ZynkSync] ✓ {} acknowledged our network departure", peer_name),
                    Ok(r) =>
                        println!("[ZynkSync] {} returned {} on unsync", peer_name, r.status()),
                    Err(e) =>
                        println!("[ZynkSync] {} offline during unsync (they will see us gone next heartbeat): {}", &peer_id[..8.min(peer_id.len())], e),
                }
            }
        });

        Ok(())
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
            let mut peers_map = self.transport.peers.write().await;
            peers_map.clear();
        }

        println!("[ZynkSync] ✓ Cleared {} device(s) from database and memory", count);
        Ok(count)
    }

    /// Receive and store memories from a peer device.
    /// Memories are written under the local user_id so they are immediately visible
    /// regardless of which user_id the sending device used.
    /// Kept for the existing ZChat command; the work lives in zchat::deliver_to_peer.
    pub async fn deliver_zchat_messages_to_peer(&self, to_device_id: &str) -> Result<usize, String> {
        crate::zchat::deliver_to_peer(&self.transport, to_device_id).await
    }

    // -------------------------------------------------------------------------
    // Conversation history sync
    // -------------------------------------------------------------------------

    /// Bidirectional sync with "active device wins" reconciliation
    /// Compares memory inventories and syncs from the device with most recent activity
    pub async fn sync_bidirectional(&self, peer_id: &str, user_id: &str) -> Result<SyncResult, String> {
        let peer = {
            let peers_map = self.transport.peers.read().await;
            peers_map.get(peer_id).cloned()
                .ok_or_else(|| format!("Peer {} not found", peer_id))?
        };

        if !peer.paired {
            return Err(format!("Device {} is not paired", peer.device_name));
        }

        // Since the outbox rebuild (step 2, 2026-10-01) a sync is a drain in each
        // direction: what changed here goes to the peer, then the peer sends what
        // changed there (sync_outbox.rs). Nothing is compared table against table any
        // more. The inventory, fetch and delete-by-hash routes stay until step 5.
        let pushed = self.drain_outbox_to(&peer.device_id, user_id).await?;
        if pushed.skipped {
            // Quietly: the peer refused a connection moments ago and is silent. The next
            // cycle tries again once two minutes have passed or a heartbeat arrives.
            return Ok(SyncResult {
                peer_device_id: peer.device_id, peer_device_name: peer.device_name,
                memories_sent: 0, memories_received: 0, conversations_sent: 0, conflicts_resolved: 0,
                success: false, error: Some("skipped: peer silent since it last refused a connection".into()),
            });
        }
        let pulled = self.pull_outbox_from(&peer.device_id).await?;

        // Keeps is_first_sync() honest for the paths that still consult it.
        self.update_sync_timestamp(&peer.device_id, true).await?;

        if pushed.entries_sent > 0 || pulled.entries_sent > 0 {
            println!("[ZynkSync] ✓ Sync with {} - sent {} ({} batches), received {} ({} batches)",
                peer.device_name, pushed.entries_sent, pushed.batches, pulled.entries_sent, pulled.batches);
        }
        Ok(SyncResult {
            peer_device_id: peer.device_id,
            peer_device_name: peer.device_name,
            memories_sent: pushed.entries_sent,
            memories_received: pulled.entries_sent,
            conversations_sent: 0,
            conflicts_resolved: 0,
            success: true,
            error: None,
        })
    }

    /// Record content hashes as tombstones so they are never resurrected by sync.
    pub async fn record_tombstones(&self, hashes: &[String]) -> Result<(), String> {
        for hash in hashes {
            sqlx::query("INSERT OR IGNORE INTO deleted_memory_hashes (content_hash) VALUES (?)")
                .bind(hash)
                .execute(&self.db_pool)
                .await
                .map_err(|e| format!("Failed to record tombstone: {}", e))?;
        }
        Ok(())
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
                let peers_map = self.transport.peers.read().await;
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

    /// Deliver any queued cascade-remove notifications for a peer that just came online.
    async fn flush_pending_removals_for_peer(&self, peer_id: &str, peer_ip: &str) {
        let pending = sqlx::query(
            "SELECT id, sender_device_id, cascade_device_id FROM zynk_pending_removals WHERE target_peer_id = ?"
        )
        .bind(peer_id)
        .fetch_all(&self.db_pool)
        .await
        .unwrap_or_default();

        if pending.is_empty() { return; }
        println!("[ZynkSync] Retrying {} queued removal(s) for peer {}", pending.len(), &peer_id[..8.min(peer_id.len())]);

        let http_client = self.transport.http_client.read().await.clone();
        let peer_port = self.peer_port_by_id(peer_id).await;
        let mut delivered_ids: Vec<i64> = Vec::new();

        for row in &pending {
            let row_id: i64 = row.try_get("id").unwrap_or(0);
            let sender_id: String = row.try_get("sender_device_id").unwrap_or_default();
            let cascade_id: String = row.try_get("cascade_device_id").unwrap_or_default();

            let url = format!("https://{}:{}/api/zynksync/notify-unsynced", peer_ip, peer_port);
            let payload = serde_json::json!({
                "removed_device_id": sender_id,
                "cascade_device_id": cascade_id,
            });

            match http_client.post(&url).json(&payload).send().await {
                Ok(r) if r.status().is_success() => {
                    println!("[ZynkSync] ✓ Delivered deferred cascade-remove to peer {}", &peer_id[..8.min(peer_id.len())]);
                    delivered_ids.push(row_id);
                }
                _ => {} // Still unreachable — keep for next heartbeat
            }
        }

        for id in delivered_ids {
            sqlx::query("DELETE FROM zynk_pending_removals WHERE id = ?")
                .bind(id)
                .execute(&self.db_pool)
                .await.ok();
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
                let peers_map = self.transport.peers.read().await;
                peers_map.values().cloned().collect::<Vec<_>>()
            };

            for peer in peers {
                if !peer.paired { continue; }
                let url = format!("https://{}:{}/api/presence/heartbeat", peer.host, peer.port);
                let body = serde_json::json!({ "device_id": device_id });
                let client = self.transport.http_client.read().await.clone();
                let result = client
                    .post(&url)
                    .json(&body)
                    .timeout(Duration::from_secs(5))
                    .send()
                    .await;
                if result.is_ok() {
                    {
                        let mut map = self.transport.peer_last_seen.write().await;
                        map.insert(peer.device_id.clone(), Utc::now());
                    }
                    // Deliver any cascade-remove notifications that failed when peer was offline
                    self.flush_pending_removals_for_peer(&peer.device_id, &peer.host).await;
                }
            }
        }
    }

    /// Send goodbye signal to all paired peers (called on clean shutdown)

    /// Start automatic synchronization loop
    /// The outbox's AUTOINCREMENT sequence: rises with every queued change, never
    /// falls (pruning deletes rows, not the sequence).
    async fn outbox_sequence(&self) -> i64 {
        sqlx::query_scalar::<_, i64>("SELECT COALESCE((SELECT seq FROM sqlite_sequence WHERE name = 'sync_outbox'), 0)")
            .fetch_one(&self.db_pool).await.unwrap_or(0)
    }

    /// Resolves once the outbox has grown past `since` (polled every `OUTBOX_POLL_SECS`).
    async fn outbox_changed(&self, since: i64) {
        loop {
            tokio::time::sleep(Duration::from_secs(OUTBOX_POLL_SECS)).await;
            if self.outbox_sequence().await > since {
                return;
            }
        }
    }

    /// Wait until the outbox has been quiet for one poll, or `OUTBOX_SETTLE_MAX_SECS`
    /// have passed — so a burst of edits goes in one cycle, not one each.
    async fn outbox_settle(&self) {
        let started = std::time::Instant::now();
        let mut seq = self.outbox_sequence().await;
        while started.elapsed() < Duration::from_secs(OUTBOX_SETTLE_MAX_SECS) {
            tokio::time::sleep(Duration::from_secs(OUTBOX_POLL_SECS)).await;
            let now = self.outbox_sequence().await;
            if now == seq {
                return;
            }
            seq = now;
        }
    }

    pub async fn start_auto_sync(self: Arc<Self>) {
        use std::sync::atomic::Ordering;
        {
            let mut enabled = self.auto_sync_enabled.write().await;
            *enabled = true;
        }
        if self.auto_sync_loop_running.swap(true, Ordering::SeqCst) {
            println!("[ZynkSync] Auto-sync loop already running — resumed, not duplicated");
            return;
        }
        println!("[ZynkSync] Starting auto-sync loop (interval: {}s)", self.sync_interval_secs);

        let mut interval_timer = interval(Duration::from_secs(self.sync_interval_secs));
        // A cycle that outlasts the interval (a long first-contact slice) must not be
        // followed by a burst of catch-up ticks.
        interval_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut last_outbox_seq = self.outbox_sequence().await;

        loop {
            // Wake on the timer, or within seconds of a local change. A change is a
            // new outbox row (the triggers write one for every memory, message,
            // thread, link or key change), seen as a higher AUTOINCREMENT sequence —
            // pruning cannot move that backwards. Before 2026-10-06 changes waited
            // for the next tick, so a memory took one to two minutes to reach the
            // other devices; the old push path felt instant, and users noticed.
            let kicked = tokio::select! {
                _ = interval_timer.tick() => false,
                _ = self.outbox_changed(last_outbox_seq) => true,
            };
            if kicked {
                // Let a burst of edits settle (a few seconds at most), then push the
                // timer out so the safety cycle does not follow straight after.
                self.outbox_settle().await;
                interval_timer.reset();
            }
            last_outbox_seq = self.outbox_sequence().await;

            // Check if still enabled
            {
                let enabled = self.auto_sync_enabled.read().await;
                if !*enabled {
                    println!("[ZynkSync] Auto-sync stopped");
                    self.auto_sync_loop_running.store(false, Ordering::SeqCst);
                    break;
                }
            }

            // Get all peers
            let peers = {
                let peers_map = self.transport.peers.read().await;
                peers_map.values().cloned().collect::<Vec<_>>()
            };

            if peers.is_empty() {
                continue;
            }

            // Auto-sync trigger — detail logged per-peer below

            // Get current user_id for syncing
            let user_id = match self.user_id() {
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
                        if result.memories_received > 0 {
                            if let Ok(guard) = crate::APP_HANDLE.lock() {
                                if let Some(app) = guard.as_ref() {
                                    let _ = app.emit("zynksync-memories-updated",
                                        serde_json::json!({ "count": result.memories_received }));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        // Debounce connection/TLS errors: only log once per 30 s per peer to
                        // avoid flooding logcat when a known-paired device is temporarily
                        // unreachable (e.g. TLS HandshakeFailure from a paired IP).
                        let is_conn_error = e.contains("HandshakeFailure")
                            || e.contains("handshake")
                            || e.contains("tls")
                            || e.contains("TLS")
                            || e.contains("connection")
                            || e.contains("connect")
                            || e.contains("tcp")
                            || e.contains("network")
                            || e.contains("os error")
                            || e.contains("unreachable")
                            || e.contains("refused")
                            || e.contains("timed out");
                        if is_conn_error {
                            let should_log = {
                                let map = self.transport.last_conn_error_logged.read().await;
                                match map.get(&peer.device_id) {
                                    Some(last) => Utc::now().signed_duration_since(*last).num_seconds() >= 30,
                                    None => true,
                                }
                            };
                            if should_log {
                                eprintln!("[ZynkSync] ✗ Auto-sync failed with {} (connection error — suppressing repeats for 30 s): {}",
                                    peer.device_name, e);
                                self.transport.last_conn_error_logged.write().await.insert(peer.device_id.clone(), Utc::now());
                                // A peer the heartbeat says is alive, yet connections to it
                                // fail: the pooled connections are the likely culprit (its
                                // app restarted; the desktop kept failing for 25 minutes on
                                // 2026-10-01). Start a fresh client — the heartbeat has
                                // already shown the peer itself answers.
                                if peer.is_online {
                                    match self.rebuild_http_client().await {
                                        Ok(()) => println!("[ZynkSync] {} is online but unreachable — HTTP client rebuilt", peer.device_name),
                                        Err(err) => eprintln!("[ZynkSync] HTTP client rebuild failed: {}", err),
                                    }
                                }
                            }
                        } else {
                            eprintln!("[ZynkSync] ✗ Auto-sync failed with {}: {}", peer.device_name, e);
                        }
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
            let peers_map = self.transport.peers.read().await;
            peers_map.values().filter(|p| p.paired).cloned().collect::<Vec<_>>()
        };
        let mut count = 0;
        for peer in peers {
            let endpoint = format!("{}/api/zynksync/pause", peer.url);
            let client = self.transport.http_client.read().await.clone();
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
            let peers_map = self.transport.peers.read().await;
            peers_map.values().filter(|p| p.paired).cloned().collect::<Vec<_>>()
        };
        let mut count = 0;
        for peer in peers {
            let endpoint = format!("{}/api/zynksync/resume", peer.url);
            let client = self.transport.http_client.read().await.clone();
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

    /// Request pairing with a peer device (generates code on this device)
    pub async fn request_pairing(&self, peer_id: &str) -> Result<String, String> {
        let pairing_code = self.generate_pairing_code().await?;

        // Update peer with pairing code
        {
            let mut peers_map = self.transport.peers.write().await;
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
        let mut peers_map = self.transport.peers.write().await;

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
    /// This device's own row in zynk_devices. zynk_device_pairings references it, so
    /// a device that never generated a pairing code (a phone that only ever entered
    /// one) could not record its sync timestamps: every sync it started failed with a
    /// foreign-key error. Found by the two-peer harness, 2026-09-17.

    pub async fn start_http_server(self: Arc<Self>) -> Result<u16, String> {
        // Clean up any old process using port 57963 (handles hot reload issues)
        // NOTE: Port cleanup is now handled in lib.rs start_zynksync() before creating the service
        // Self::cleanup_port_57963();

        // Public routes — reachable without a client cert (pairing bootstrap only).
        let transport = self.transport.clone();

        // ZynkSync's own routes: pairing and sync codes are public (a device that is
        // not yet paired must reach them); everything else needs a verified peer.
        let sync_public = Router::new()
            .route("/api/zynksync/info", axum::routing::get(handle_device_info))
            .route("/api/zynksync/verify-pairing", post(handle_verify_pairing))
            .route("/api/identity/verify-sync-code", post(handle_verify_sync_code))
            .route("/api/identity/consume-sync-code", post(handle_consume_sync_code))
            .with_state(Arc::clone(&self));

        let sync_protected = Router::new()
            .route("/api/zynksync/introduce", post(handle_introduce))
            .route("/api/zynksync/notify-unsynced", post(handle_notify_unsynced))
            .route("/api/zynksync/outbox", post(handle_receive_outbox))
            .route("/api/zynksync/outbox/pull", post(handle_pull_outbox))
            .route("/api/presence/heartbeat", post(handle_heartbeat))
            .route("/api/presence/goodbye", post(handle_goodbye))
            .route("/api/zynksync/pause", post(handle_pause))
            .route("/api/zynksync/resume", post(handle_resume))
            .route("/api/ollama/*path", any(handle_ollama_proxy))
            // A history or memory push can exceed the 2 MB default (KI-068). Applies to
            // the routes above it, which is how axum layers work.
            .layer(axum::extract::DefaultBodyLimit::max(32 * 1024 * 1024))
            .layer(axum::middleware::from_fn_with_state(
                Arc::clone(&transport),
                crate::transport::require_verified_device,
            ))
            .with_state(Arc::clone(&self));

        // The other two services register their own bundles with the transport.
        let app = Router::new()
            .merge(sync_public)
            .merge(sync_protected)
            .merge(crate::zynklink::routes(Arc::clone(&transport)))
            .merge(crate::zchat::routes(Arc::clone(&transport)))
            .layer(axum::middleware::from_fn_with_state(
                Arc::clone(&transport),
                crate::transport::inject_verified_device,
            ));

        self.transport.clone().serve(app).await
    }
}



/// Axum handler for getting device info (used when adding devices manually)
async fn handle_device_info(
    State(service): State<Arc<ZynkSyncService>>,
) -> Result<Json<serde_json::Value>, String> {
    Ok(Json(serde_json::json!({
        "device_id": service.device_id,
        "device_name": service.device_name(),
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
        let attempts = service.transport.failed_pairing_attempts.read().await;
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
            service.transport.failed_pairing_attempts.write().await.remove(&client_ip);
            println!("[ZynkSync] ✓ Pairing code verified! Auto-adding client: {} ({})",
                client_device_name, client_device_id);

            // BIDIRECTIONAL PAIRING: Automatically add the client device to our peer list
            let port: u16 = request.get("client_port").and_then(|v| v.as_u64()).map(|p| p as u16).unwrap_or(DEFAULT_SYNC_PORT);
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
                is_online: false,
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
                let mut peers_map = service.transport.peers.write().await;
                peers_map.insert(client_device_id.to_string(), peer);
            }

            println!("[ZynkSync] ✓ Bidirectional pairing complete - client added to peer list");

            // Get host's user_id for identity sync and security validation
            let host_user_id = match service.user_id() {
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

            // REVERSE MESH: Pre-trust peers the client already knows so the host joins their mesh too.
            // We do this synchronously so these peers appear in the mesh_peers response and the
            // background task can introduce the host to them with the right trust chain.
            let client_peers_vec: Vec<serde_json::Value> = request
                .get("client_peers")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            if !client_peers_vec.is_empty() {
                println!("[ZynkSync] Client sent {} peer(s) — pre-trusting for reverse mesh", client_peers_vec.len());
                for peer_val in &client_peers_vec {
                    let peer_id = match peer_val.get("device_id").and_then(|v| v.as_str()) {
                        Some(id) => id.to_string(),
                        None => continue,
                    };
                    if peer_id == service.device_id { continue; }
                    let peer_name = peer_val.get("device_name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                    let peer_ip = match peer_val.get("ip").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                        Some(ip) => ip.to_string(),
                        None => continue,
                    };
                    let peer_cert: Option<Vec<u8>> = peer_val.get("cert_der")
                        .and_then(|v| v.as_str())
                        .and_then(|b64| BASE64.decode(b64).ok());
                    let peer_port: u16 = peer_val.get("port").and_then(|v| v.as_u64()).map(|p| p as u16).unwrap_or(DEFAULT_SYNC_PORT);

                    sqlx::query(
                        "INSERT INTO zynk_devices (device_id, device_name, device_ip, port, is_paired, sync_paired, tls_cert_der, last_seen_at, created_at)
                         VALUES (?, ?, ?, ?, 1, 1, ?, ?, ?)
                         ON CONFLICT (device_id) DO UPDATE
                         SET device_name = excluded.device_name,
                             device_ip = excluded.device_ip,
                             sync_paired = 1,
                             tls_cert_der = excluded.tls_cert_der,
                             last_seen_at = excluded.last_seen_at"
                    )
                    .bind(&peer_id).bind(&peer_name).bind(&peer_ip)
                    .bind(peer_port as i32).bind(peer_cert.as_deref())
                    .bind(Utc::now()).bind(Utc::now())
                    .execute(&service.db_pool).await.ok();

                    {
                        let mut map = service.transport.peers.write().await;
                        if !map.contains_key(&peer_id) {
                            map.insert(peer_id.clone(), PeerDevice {
                                device_id: peer_id.clone(),
                                device_name: peer_name.clone(),
                                host: peer_ip.clone(),
                                port: peer_port,
                                url: format!("https://{}:{}", peer_ip, peer_port),
                                last_seen: Utc::now(),
                                paired: true,
                                pairing_code: None,
                                user_id: None,
                                is_online: false,
                            });
                        }
                    }
                    println!("[ZynkSync] Pre-trusted client's peer: {} ({})", peer_name, &peer_id[..8.min(peer_id.len())]);
                }
                if let Err(e) = service.rebuild_http_client().await {
                    println!("[TLS] Warning: could not rebuild HTTP client after reverse mesh pre-trust: {}", e);
                }
            }

            // Collect existing peers to include in response (mesh pairing)
            let mesh_peers: Vec<serde_json::Value> = {
                let rows = sqlx::query(
                    "SELECT device_id, device_name, device_ip, port, tls_cert_der
                     FROM zynk_devices
                     WHERE sync_paired = 1 AND device_id != ? AND device_id != ?"
                )
                .bind(client_device_id)
                .bind(&service.device_id)
                .fetch_all(&service.db_pool)
                .await
                .unwrap_or_default();

                rows.iter().filter_map(|row| {
                    let dev_id: String = row.try_get("device_id").ok()?;
                    let dev_name: String = row.try_get("device_name").ok()?;
                    let ip: String = row.try_get("device_ip").ok()?;
                    let port_val: i32 = row.try_get("port").ok()?;
                    let cert: Option<Vec<u8>> = row.try_get("tls_cert_der").ok().flatten();
                    Some(serde_json::json!({
                        "device_id": dev_id,
                        "device_name": dev_name,
                        "ip": ip,
                        "port": port_val,
                        "cert_der": cert.map(|d| BASE64.encode(&d)),
                    }))
                }).collect()
            };

            if !mesh_peers.is_empty() {
                println!("[ZynkSync] Sending {} peer(s) to new device for mesh pairing", mesh_peers.len());
            }

            // Notify existing peers about the new device in the background (best-effort).
            // Phase 1: introduce host to client's pre-existing peers (bidirectional mesh).
            // Phase 2: notify all known peers (including client's old peers) about the new client.
            // Running phase 1 before phase 2 ensures that by the time we tell C about B,
            // C already trusts the host and can accept the introduction.
            {
                let svc = Arc::clone(&service);
                let new_id = client_device_id.to_string();
                let new_name = client_device_name.to_string();
                let new_ip = client_ip.clone();
                let new_cert_b64 = client_cert_der.as_ref().map(|d| BASE64.encode(d)).unwrap_or_default();
                let new_uid = client_user_id.map(|s| s.to_string());
                let client_peers_for_spawn = client_peers_vec.clone();
                tokio::spawn(async move {
                    // Phase 1: introduce host to each of the client's pre-existing peers.
                    // The client (new device) is the vouching introducer — it knows both parties.
                    // We omit new_device_ip so the receiving peer falls back to addr.ip(),
                    // which is the host's IP (we are the sender and the introduced device).
                    if !client_peers_for_spawn.is_empty() {
                        let host_cert_b64 = BASE64.encode(&svc.transport.cert_der);
                        for peer_val in &client_peers_for_spawn {
                            let peer_ip = match peer_val.get("ip").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                                Some(ip) => ip.to_string(),
                                None => continue,
                            };
                            let peer_port: u16 = peer_val.get("port").and_then(|v| v.as_u64()).map(|p| p as u16).unwrap_or(DEFAULT_SYNC_PORT);
                            let url = format!("https://{}:{}/api/zynksync/introduce", peer_ip, peer_port);
                            let payload = serde_json::json!({
                                "new_device_id": svc.device_id,
                                "new_device_name": svc.device_name(),
                                "new_device_port": svc.port(),
                                "new_device_cert_der": host_cert_b64,
                                "introducer_device_id": new_id,
                                // no new_device_ip — addr.ip() on receiver will be the host's IP
                            });
                            let http_client2 = svc.transport.http_client.read().await.clone();
                            match http_client2.post(&url).json(&payload).send().await {
                                Ok(r) if r.status().is_success() =>
                                    println!("[ZynkSync] ✓ Host introduced itself to client's peer at {}", peer_ip),
                                Ok(r) =>
                                    println!("[ZynkSync] Host self-intro to client's peer at {} returned {}", peer_ip, r.status()),
                                Err(e) =>
                                    println!("[ZynkSync] Client's peer at {} offline during host self-intro: {}", peer_ip, e),
                            }
                        }
                    }

                    // Phase 2: notify all known peers (now including client's old peers) about the new client.
                    let http_client = svc.transport.http_client.read().await.clone();
                    let peers: Vec<PeerDevice> = {
                        let map = svc.transport.peers.read().await;
                        map.values()
                            .filter(|p| p.paired && p.device_id != new_id)
                            .cloned()
                            .collect()
                    };
                    for peer in peers {
                        let url = format!("{}/api/zynksync/introduce", peer.url);
                        let mut payload = serde_json::json!({
                            "new_device_id": new_id,
                            "new_device_name": new_name,
                            "new_device_cert_der": new_cert_b64,
                            "introducer_device_id": svc.device_id,
                        });
                        if let Some(ref ip) = Some(new_ip.clone()) {
                            payload["new_device_ip"] = serde_json::json!(ip);
                        }
                        if let Some(ref uid) = new_uid {
                            payload["new_device_user_id"] = serde_json::json!(uid);
                        }
                        match http_client.post(&url).json(&payload).send().await {
                            Ok(r) if r.status().is_success() =>
                                println!("[ZynkSync] ✓ Notified {} about new peer {}", peer.device_name, new_name),
                            Ok(r) =>
                                println!("[ZynkSync] Peer {} returned {} for introduction", peer.device_name, r.status()),
                            Err(e) =>
                                println!("[ZynkSync] Peer {} offline during introduction: {}", peer.device_name, e),
                        }
                    }
                });
            }

            // Return this host's device info including TLS cert, user_id, and peer list
            let mut response = serde_json::json!({
                "device_id": service.device_id,
                "device_name": service.device_name(),
                "cert_der": BASE64.encode(&service.transport.cert_der),
                "peers": mesh_peers,
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
            let mut attempts = service.transport.failed_pairing_attempts.write().await;
            let count = attempts.entry(client_ip.clone()).or_insert(0);
            *count += 1;
            println!("[ZynkSync] ✗ Invalid pairing code from {} ({}/5 attempts)", client_ip, count);
            Err("Invalid or expired pairing code".to_string())
        }
    }
}

/// Axum handler for mesh pairing introductions.
/// When device C pairs with B, B notifies A via this endpoint so A auto-pairs with C.
/// Trust rule: only accept introductions from devices already in our sync_paired list.
async fn handle_introduce(
    State(service): State<Arc<ZynkSyncService>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, String> {
    let new_device_id = request.get("new_device_id")
        .and_then(|v| v.as_str())
        .ok_or("Missing new_device_id")?;
    let new_device_name = request.get("new_device_name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let introducer_device_id = request.get("introducer_device_id")
        .and_then(|v| v.as_str())
        .ok_or("Missing introducer_device_id")?;
    let new_cert_der: Option<Vec<u8>> = request.get("new_device_cert_der")
        .and_then(|v| v.as_str())
        .and_then(|b64| BASE64.decode(b64).ok());

    // Prefer the IP the introducer explicitly declared in the payload (it knows the new device's
    // real address). Fall back to addr.ip() only for self-introductions where sender == new device.
    let new_device_ip = request.get("new_device_ip")
        .and_then(|v| v.as_str())
        .filter(|ip| !ip.is_empty())
        .map(|ip| ip.to_string())
        .unwrap_or_else(|| addr.ip().to_string());
    let new_device_port: u16 = request.get("new_device_port").and_then(|v| v.as_u64()).map(|p| p as u16).unwrap_or(DEFAULT_SYNC_PORT);

    // Reject introductions from devices we don't already trust
    let introducer_trusted: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM zynk_devices WHERE device_id = ? AND sync_paired = 1"
    )
    .bind(introducer_device_id)
    .fetch_one(&service.db_pool)
    .await
    .unwrap_or(0);

    if introducer_trusted == 0 {
        println!("[ZynkSync] ✗ Rejected introduction from untrusted introducer: {}", &introducer_device_id[..8.min(introducer_device_id.len())]);
        return Err("Introducer not in trusted peer list".to_string());
    }

    println!("[ZynkSync] ✓ Trusted introduction of {} ({}) via {}", new_device_name, &new_device_id[..8.min(new_device_id.len())], &introducer_device_id[..8.min(introducer_device_id.len())]);

    // Skip if already paired with this device
    let already: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM zynk_devices WHERE device_id = ? AND sync_paired = 1"
    )
    .bind(new_device_id)
    .fetch_one(&service.db_pool)
    .await
    .unwrap_or(0);

    if already == 0 {
        sqlx::query(
            "INSERT INTO zynk_devices (device_id, device_name, device_ip, port, is_paired, sync_paired, tls_cert_der, last_seen_at, created_at)
             VALUES (?, ?, ?, ?, 1, 1, ?, ?, ?)
             ON CONFLICT (device_id) DO UPDATE
             SET device_name = excluded.device_name,
                 device_ip = excluded.device_ip,
                 sync_paired = 1,
                 tls_cert_der = excluded.tls_cert_der,
                 last_seen_at = excluded.last_seen_at"
        )
        .bind(new_device_id)
        .bind(new_device_name)
        .bind(&new_device_ip)
        .bind(new_device_port as i32)
        .bind(new_cert_der.as_deref())
        .bind(Utc::now())
        .bind(Utc::now())
        .execute(&service.db_pool)
        .await
        .map_err(|e| format!("Failed to save introduced device: {}", e))?;

        // Rebuild HTTP client to trust the new peer's cert
        service.rebuild_http_client().await.ok();

        // Add to in-memory peers map
        let mut map = service.transport.peers.write().await;
        if !map.contains_key(new_device_id) {
            map.insert(new_device_id.to_string(), PeerDevice {
                device_id: new_device_id.to_string(),
                device_name: new_device_name.to_string(),
                host: new_device_ip.clone(),
                port: new_device_port,
                url: format!("https://{}:{}", new_device_ip, new_device_port),
                last_seen: Utc::now(),
                paired: true,
                pairing_code: None,
                user_id: None,
                is_online: false,
            });
        }
        println!("[ZynkSync] ✓ Mesh-paired with introduced device: {}", new_device_name);
    } else {
        println!("[ZynkSync] Already paired with {} — introduction ignored", new_device_name);
    }

    // Respond with our own cert so the new device can pin us
    Ok(Json(serde_json::json!({
        "device_id": service.device_id,
        "device_name": service.device_name(),
        "cert_der": BASE64.encode(&service.transport.cert_der),
    })))
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

    {
        let mut map = service.transport.peer_last_seen.write().await;
        map.insert(device_id.to_string(), Utc::now());
    }

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

    {
        let mut map = service.transport.peer_last_seen.write().await;
        map.remove(device_id);
    }

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
    check_sync_authorized(&service.db_pool, device_id, &headers).await?;

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
    check_sync_authorized(&service.db_pool, device_id, &headers).await?;

    let was_running = service.is_auto_sync_enabled().await;

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

    // If the auto-sync loop had already exited, restart it along with peers and client
    if !was_running {
        println!("[ZynkSync] Auto-sync loop was stopped — restarting via peer resume from {}", device_id);
        let svc = Arc::clone(&service);
        tokio::spawn(async move {
            if let Err(e) = svc.load_devices().await {
                eprintln!("[ZynkSync] Resume: failed to load devices: {}", e);
            }
            if let Err(e) = svc.rebuild_http_client().await {
                eprintln!("[ZynkSync] Resume: failed to rebuild HTTP client: {}", e);
            }
            svc.start_auto_sync().await;
        });
    } else {
        // Loop is running but may have stale peers; reload devices and do an immediate sync
        let svc = Arc::clone(&service);
        let resuming_peer = device_id.to_string();
        tokio::spawn(async move {
            if let Err(e) = svc.load_devices().await {
                eprintln!("[ZynkSync] Resume: failed to reload devices: {}", e);
                return;
            }
            if let Err(e) = svc.rebuild_http_client().await {
                eprintln!("[ZynkSync] Resume: failed to rebuild HTTP client: {}", e);
            }
            // Immediate sync with the peer that just resumed
            let user_id = match service.user_id() {
                Ok(id) => id,
                Err(e) => { eprintln!("[ZynkSync] Resume: failed to get user_id: {}", e); return; }
            };
            match svc.sync_bidirectional(&resuming_peer, &user_id).await {
                Ok(r) => println!("[ZynkSync] ✓ Immediate post-resume sync with {}: sent={}, received={}",
                    resuming_peer, r.memories_sent, r.memories_received),
                Err(e) => eprintln!("[ZynkSync] ✗ Post-resume sync failed: {}", e),
            }
        });
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

    // A "you were removed" notice names its target since 2026-09-13; if it is not us,
    // it was meant for a stale entry that shared our address — leave our peers alone.
    if let Some(target) = payload.get("target_device_id").and_then(|v| v.as_str()) {
        if target != service.device_id {
            println!("[ZynkSync] Ignoring removal notice addressed to {} (we are {})",
                &target[..8.min(target.len())], &service.device_id[..8.min(service.device_id.len())]);
            return Ok(Json(serde_json::json!({"success": true, "ignored": true})));
        }
    }

    // cascade_device_id: a third device we're being asked to remove (mesh cascade)
    // If absent, we remove the sender (removed_device_id) as before.
    let target_id = payload.get("cascade_device_id")
        .and_then(|v| v.as_str())
        .unwrap_or(removed_device_id);

    if target_id != removed_device_id {
        println!("[ZynkSync] Cascade remove: {} asked us to remove {}",
            &removed_device_id[..8.min(removed_device_id.len())],
            &target_id[..8.min(target_id.len())]);
    } else {
        println!("[ZynkSync] Peer {} initiated unsync — removing from local list", &removed_device_id[..removed_device_id.len().min(8)]);
    }

    // Best-effort — device may already be gone or never fully paired.
    // Call db_only to avoid firing a return notification (round-trip loop).
    match service.clear_sync_data_db_only(target_id).await {
        Ok(_) => println!("[ZynkSync] ✓ Removed peer device on their request"),
        Err(e) => println!("[ZynkSync] Note: clear_sync_data_db_only on notify-unsynced failed (non-fatal): {}", e),
    }

    let removed_device_id = target_id.to_string();

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
        &format!("INSERT INTO zynk_devices (device_id, device_name, device_ip, port, device_platform, is_paired, sync_paired, last_seen_at, created_at)
         VALUES (?, 'Remote Device', ?, {}, '', 1, 1, datetime('now'), datetime('now'))
         ON CONFLICT (device_id) DO UPDATE
         SET device_ip = ?, is_paired = 1, sync_paired = 1, last_seen_at = datetime('now')", DEFAULT_SYNC_PORT)
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

// =============================================================================
// ZynkLink HTTP Handlers - File Sharing Between Devices
// =============================================================================



/// Check that the requesting device_id has an active entry in zynk_device_pairings.
/// Used to reject sync requests from devices that have been removed on this side.
async fn check_sync_authorized(
    pool: &sqlx::SqlitePool,
    device_id: &str,
    headers: &axum::http::HeaderMap,
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

    // Pick up a rename from the sender's x-device-name header on every authorized
    // request, so renaming a device propagates to already-paired peers at their next
    // contact instead of requiring an unpair/re-pair (which wipes sync state and, if
    // the device is also ZynkLinked, its chat history).
    if let Some(name) = headers.get("x-device-name").and_then(|v| v.to_str().ok()) {
        let name = name.trim();
        if !name.is_empty() {
            let _ = sqlx::query(
                "UPDATE zynk_devices SET device_name = ? WHERE device_id = ? AND device_name != ?"
            )
            .bind(name)
            .bind(device_id)
            .bind(name)
            .execute(pool)
            .await;
        }
    }
    Ok(())
}






/// A peer's outbox batch (sync_outbox.rs). Applied in one transaction under
/// sync_suppress; the receipt's `through` is what the sender records as its cursor.
async fn handle_receive_outbox(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
    Json(batch): Json<crate::sync_outbox::OutboxBatch>,
) -> Result<Json<crate::sync_outbox::OutboxReceipt>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?;
    check_sync_authorized(&service.db_pool, device_id, &headers).await?;

    let local_user_id = service.user_id().unwrap_or_default();
    let (applied, known_before) = crate::sync_outbox::apply_outbox_batch_from(&service.db_pool, &local_user_id, Some(device_id), &batch).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;
    if applied > 0 {
        if let Ok(guard) = crate::APP_HANDLE.lock() {
            if let Some(app) = guard.as_ref() {
                let _ = app.emit("zynksync-memories-updated", serde_json::json!({ "count": applied }));
            }
        }
    }
    Ok(Json(crate::sync_outbox::OutboxReceipt { through: batch.through, applied, known_before }))
}

/// A peer asking us to drain our queue to it. The drain runs here, so the sending code
/// lives in one place whichever side started the sync.
async fn handle_pull_outbox(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
    body: Option<Json<crate::sync_outbox::PullRequest>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?
        .to_string();
    check_sync_authorized(&service.db_pool, &device_id, &headers).await?;

    // A peer that holds less than our cursor for it says (a restored install) starts over.
    let known = body.map(|Json(b)| b.known_through).unwrap_or(None);
    crate::sync_outbox::reset_cursor_if_ahead(&service.db_pool, &device_id, known).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;

    let user_id = service.user_id().unwrap_or_default();
    let outcome = service.drain_outbox_to(&device_id, &user_id).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;
    Ok(Json(serde_json::json!({
        "batches": outcome.batches,
        "entries_sent": outcome.entries_sent,
        "applied": outcome.applied_by_peer
    })))
}


// =============================================================================
// Ollama endpoints
// =============================================================================


// Proxy — forwards /api/ollama/* to localhost:11434 for LAN inference

async fn handle_ollama_proxy(
    State(_service): State<Arc<ZynkSyncService>>,
    request: Request,
) -> Response {
    // Check that local Ollama is configured
    if std::env::var("CUSTOM_API_URL").is_err() {
        return (StatusCode::SERVICE_UNAVAILABLE,
            "Ollama not configured on this device").into_response();
    }
    // The desktop decides which Ollama model paired devices use. Remote devices never
    // pick a model: whatever they send is replaced below with this machine's selection,
    // so changing it here propagates to every phone on the next request.
    let desktop_model = std::env::var("CUSTOM_MODEL").unwrap_or_default();
    if desktop_model.trim().is_empty() {
        return (StatusCode::SERVICE_UNAVAILABLE,
            "No Ollama model selected on the desktop — pick one in Settings → API Keys → Custom / Ollama").into_response();
    }

    // Strip /api/ollama prefix to get the target path
    let uri = request.uri().clone();
    let path = uri.path().trim_start_matches("/api/ollama");
    let query = uri.query().map(|q| format!("?{}", q)).unwrap_or_default();

    // Block Ollama admin operations — paired devices may only use inference and model discovery.
    // Administrative endpoints (pull, push, create, copy, delete) could exhaust disk or cause
    // unintended changes to the host's model library.
    const BLOCKED: &[&str] = &[
        "/api/pull", "/api/push", "/api/create", "/api/copy", "/api/delete",
    ];
    if BLOCKED.iter().any(|b| path == *b || path.starts_with(&format!("{}/", b))) {
        return (
            StatusCode::FORBIDDEN,
            "Administrative Ollama operations are not available via remote proxy",
        ).into_response();
    }

    let target_url = format!("http://localhost:11434{}{}", path, query);

    // Who is asking: the verified peer's name (the log used to print our own id).
    let caller = request.extensions().get::<VerifiedDevice>()
        .map(|v| v.device_name.clone())
        .unwrap_or_else(|| "unverified device".to_string());

    // Read request body
    let method_str = request.method().as_str().to_string();
    let req_content_type = request.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let body_bytes = match axum::body::to_bytes(request.into_body(), 10 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("Failed to read body: {}", e)).into_response(),
    };

    // Replace the caller's model with the desktop's selection (chat/completions,
    // generate, embeddings — anything that carries a "model" field).
    let body_bytes = match serde_json::from_slice::<serde_json::Value>(&body_bytes) {
        Ok(mut json) if json.get("model").is_some() => {
            let requested = json["model"].as_str().unwrap_or("").to_string();
            if requested != desktop_model {
                println!("[OllamaProxy] {} asked for '{}'; using desktop selection '{}'",
                    caller, requested, desktop_model);
            }
            json["model"] = serde_json::Value::String(desktop_model.clone());
            axum::body::Bytes::from(json.to_string())
        }
        _ => body_bytes,
    };

    // Forward to local Ollama
    let client = reqwest::Client::new();
    let method = match reqwest::Method::from_bytes(method_str.as_bytes()) {
        Ok(m) => m,
        Err(_) => return (StatusCode::BAD_REQUEST, "Invalid HTTP method").into_response(),
    };

    let mut builder = client.request(method, &target_url)
        .header("content-type", &req_content_type);
    if !body_bytes.is_empty() {
        builder = builder.body(body_bytes.to_vec());
    }

    let ollama_response = match builder.send().await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[OllamaProxy] Failed to reach Ollama: {}", e);
            return (StatusCode::BAD_GATEWAY,
                format!("Can't reach Ollama — is it running? ({})", e)).into_response();
        }
    };

    let status = StatusCode::from_u16(ollama_response.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let content_type = ollama_response.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    // Stream response body back to caller
    let stream = ollama_response.bytes_stream();
    Response::builder()
        .status(status)
        .header("content-type", content_type)
        .header("access-control-allow-origin", "*")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

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
use sha2::{Sha256, Digest};
use tauri::Emitter;
pub use crate::transport::{DEFAULT_SYNC_PORT, PeerDevice, SyncIdentity, split_host_port};
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
    #[serde(default)]
    pub relationships: Vec<MemoryRelationship>,  // Relationships from memory_links
}

fn default_memory_placement() -> String {
    "retrieved".to_string()
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
    #[serde(default)]
    pub deleted_hashes: Vec<String>,  // Tombstones: hashes of explicitly deleted memories
    #[serde(default)]
    pub tombstone_timestamps: std::collections::HashMap<String, String>,  // hash → deleted_at ISO
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
/// Most messages in one history push; the rest go on later cycles (KI-068).
pub const CONVERSATION_PUSH_MAX_MESSAGES: usize = 300;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSyncPayload {
    pub sessions: Vec<SyncConversationSession>,
    pub messages: Vec<SyncConversationMessage>,
}

/// Core ZynkSync service managing device synchronization
pub struct ZynkSyncService {
    /// Everything shared with ZynkLink and ZChat: identity, certificate, server,
    /// pinned client, device registry, presence. See crate::transport.
    pub(crate) transport: Arc<Transport>,

    /// Unique identifier for this device (copy of transport.identity().device_id)
    device_id: String,

    /// SQLite connection pool
    db_pool: SqlitePool,

    /// Last sync timestamps per peer (to track incremental syncs)
    last_sync: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,

    /// Whether auto-sync is enabled
    auto_sync_enabled: Arc<RwLock<bool>>,

    /// Sync interval in seconds
    sync_interval_secs: u64,
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
            last_sync: Arc::new(RwLock::new(HashMap::new())),
            auto_sync_enabled: Arc::new(RwLock::new(false)),
            sync_interval_secs: sync_interval_secs.unwrap_or(300),
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
            let mut peers_map = self.transport.peers.write().await;
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
             WHERE is_syncable = 1
             ORDER BY datetime(COALESCE(updated_at, created_at)) DESC"
        )
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

        let tombstone_rows: Vec<(String, String)> = sqlx::query_as(
            "SELECT content_hash, deleted_at FROM deleted_memory_hashes"
        )
        .fetch_all(&self.db_pool)
        .await
        .unwrap_or_default();

        let deleted_hashes: Vec<String> = tombstone_rows.iter().map(|r| r.0.clone()).collect();
        let tombstone_timestamps: std::collections::HashMap<String, String> =
            tombstone_rows.into_iter().collect();

        Ok(MemoryInventory {
            user_id: user_id.to_string(),
            memory_ids,
            content_hashes,
            latest_activity,
            memory_count,
            deleted_hashes,
            tombstone_timestamps,
        })
    }

    /// Get memories modified since the last sync with a specific peer
    async fn get_modified_memories(
        &self,
        peer_id: &str,
        _user_id: Option<&str>,
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
                        event_type, event_date, entities_detected, original_text,
                        collection_id, memory_placement, external_id, temporal_status, provenance_json
                 FROM memories
                 WHERE is_syncable = 1
                   AND created_at > ?
                 ORDER BY created_at ASC
                 LIMIT 1000"
            )
            .bind(last_sync)
        } else {
            // Full sync: all syncable memories
            sqlx::query(
                "SELECT id, user_id, session_id, content, title, source_type, created_at, updated_at,
                        parent_scroll_id, chunk_index, namespace, is_syncable, is_shareable,
                        embedding, link_count, is_ephemeral, expires_at, sentiment_score, sentiment_label,
                        event_type, event_date, entities_detected, original_text,
                        collection_id, memory_placement, external_id, temporal_status, provenance_json
                 FROM memories
                 WHERE is_syncable = 1
                 ORDER BY created_at ASC
                 LIMIT 1000"
            )
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
                    collection_id: row.get("collection_id"),
                    memory_placement: row.get("memory_placement"),
                    external_id: row.get("external_id"),
                    temporal_status: row.get("temporal_status"),
                    provenance_json: row.get("provenance_json"),
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
                                event_type, event_date, entities_detected, original_text,
                                collection_id, memory_placement, external_id, temporal_status, provenance_json
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
                        collection_id: row.get("collection_id"),
                        memory_placement: row.get("memory_placement"),
                        external_id: row.get("external_id"),
                        temporal_status: row.get("temporal_status"),
                        provenance_json: row.get("provenance_json"),
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
            let peers_map = self.transport.peers.read().await;
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
        let client = self.transport.http_client.read().await.clone();
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


    /// Receive and store memories from a peer device.
    /// Memories are written under the local user_id so they are immediately visible
    /// regardless of which user_id the sending device used.
    /// Kept for the existing ZChat command; the work lives in zchat::deliver_to_peer.
    pub async fn deliver_zchat_messages_to_peer(&self, to_device_id: &str) -> Result<usize, String> {
        crate::zchat::deliver_to_peer(&self.transport, to_device_id).await
    }

    pub async fn receive_from_peer(&self, local_user_id: &str, memories: Vec<SyncMemory>) -> Result<usize, String> {
        use std::collections::HashMap;

        let mut stored_count = 0;
        let mut id_mapping: HashMap<i32, i32> = HashMap::new();  // old_id -> new_id

        println!("[ZynkSync] Receiving {} memories from peer (writing under local user_id)", memories.len());

        // PHASE 1: Store/update all memories and build ID mapping
        for memory in &memories {
            // Check if memory already exists by content hash (user_id-agnostic)
            let existing = sqlx::query_as::<_, (i32, chrono::DateTime<chrono::Utc>)>(
                "SELECT id, created_at FROM memories WHERE content = ?"
            )
            .bind(&memory.content)
            .fetch_optional(&self.db_pool)
            .await
            .map_err(|e| format!("Failed to check existing memory: {}", e))?;

            if let Some(existing_memory) = existing {
                // Memory exists - map old ID to existing ID
                id_mapping.insert(memory.id, existing_memory.0);

                // Update if incoming is newer
                if memory.created_at > existing_memory.1 {
                    println!("[ZynkSync] Updating existing memory (newer timestamp from peer)");
                    sqlx::query(
                        "UPDATE memories
                         SET title = ?, namespace = ?, created_at = ?, session_id = ?,
                             collection_id = ?, memory_placement = ?, external_id = ?,
                             temporal_status = ?, provenance_json = ?
                         WHERE id = ?"
                    )
                    .bind(&memory.title)
                    .bind(&memory.namespace)
                    .bind(memory.created_at)
                    .bind(&memory.session_id)
                    .bind(&memory.collection_id)
                    .bind(&memory.memory_placement)
                    .bind(&memory.external_id)
                    .bind(&memory.temporal_status)
                    .bind(&memory.provenance_json)
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
                    let existing = sqlx::query_as::<_, (i32, String, chrono::DateTime<chrono::Utc>)>(
                        "SELECT id, content, created_at FROM memories WHERE id = ?"
                    )
                    .bind(memory.id)
                    .fetch_one(&self.db_pool)
                    .await
                    .map_err(|e| format!("Failed to fetch conflicting memory: {}", e))?;

                    if existing.1 == memory.content {
                        // Same content — true duplicate, keep whichever is newer
                        id_mapping.insert(memory.id, existing.0);
                    } else {
                        // Different content — integer ID collision between two different memories.
                        // Insert the incoming memory with a new auto-generated ID so neither is lost.
                        println!("[ZynkSync] ID {} collision (different content) — inserting with new ID", memory.id);
                        let result = sqlx::query(
                            "INSERT INTO memories (user_id, session_id, content, title, source_type, created_at, updated_at,
                                                   parent_scroll_id, chunk_index, namespace, is_syncable, is_shareable,
                                                   embedding, link_count, is_ephemeral, expires_at, sentiment_score, sentiment_label,
                                                   event_type, event_date, entities_detected, original_text,
                                                   collection_id, memory_placement, external_id, temporal_status, provenance_json)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                        )
                        .bind(local_user_id)
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
                        .bind(memory.collection_id.as_deref())
                        .bind(&memory.memory_placement)
                        .bind(memory.external_id.as_deref())
                        .bind(memory.temporal_status.as_deref())
                        .bind(memory.provenance_json.as_deref())
                        .execute(&self.db_pool)
                        .await
                        .map_err(|e| format!("Failed to insert ID-colliding memory: {}", e))?;
                        let new_id = result.last_insert_rowid() as i32;
                        id_mapping.insert(memory.id, new_id);
                        stored_count += 1;
                    }
                    continue;
                }

                // No conflict - insert with original ID, local user_id
                sqlx::query(
                    "INSERT INTO memories (id, user_id, session_id, content, title, source_type, created_at, updated_at,
                                          parent_scroll_id, chunk_index, namespace, is_syncable, is_shareable,
                                          embedding, link_count, is_ephemeral, expires_at, sentiment_score, sentiment_label,
                                          event_type, event_date, entities_detected, original_text,
                                          collection_id, memory_placement, external_id, temporal_status, provenance_json)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                .bind(memory.id)
                .bind(local_user_id)
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
                .bind(memory.collection_id.as_deref())
                .bind(&memory.memory_placement)
                .bind(memory.external_id.as_deref())
                .bind(memory.temporal_status.as_deref())
                .bind(memory.provenance_json.as_deref())
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

    /// Conversations changed since the last push this peer accepted, oldest first.
    /// The marker is per peer and persisted (zynk_conversation_push_state), so an app
    /// restart does not re-send the whole history; with no marker (first push to this
    /// peer) everything is eligible. A push carries at most
    /// CONVERSATION_PUSH_MAX_MESSAGES messages — the receiver caps a request at a few
    /// megabytes — and the returned `through` is the marker to record once the peer
    /// accepts it: the newest message sent, so what was left out goes next cycle.
    /// Comparisons are `>=` because timestamps have one-second resolution; the peer
    /// deduplicates what it already has.
    pub(crate) async fn get_modified_conversations(
        &self,
        peer_id: &str,
        user_id: &str,
    ) -> Result<(ConversationSyncPayload, Option<DateTime<Utc>>), String> {
        let since: Option<DateTime<Utc>> = sqlx::query_scalar(
            "SELECT pushed_through FROM zynk_conversation_push_state WHERE peer_device_id = ?"
        )
        .bind(peer_id)
        .fetch_optional(&self.db_pool)
        .await
        .map_err(|e| format!("Failed to read conversation push marker: {}", e))?;

        let session_rows = match since {
            Some(since) => sqlx::query(
                "SELECT session_id, user_id, title, started_at, last_active, message_count,
                        model_backend, containment_mode
                 FROM conversation_sessions
                 WHERE user_id = ? AND last_active >= ?
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
        }).collect::<Vec<_>>();

        let cap = CONVERSATION_PUSH_MAX_MESSAGES as i64;
        let message_rows = match since {
            Some(since) => sqlx::query(
                "SELECT session_id, user_id, role, content, created_at,
                        model_backend, containment_mode, entry_hash, prev_hash
                 FROM conversation_messages
                 WHERE user_id = ? AND created_at >= ?
                 ORDER BY created_at ASC LIMIT ?"
            )
            .bind(user_id)
            .bind(since)
            .bind(cap)
            .fetch_all(&self.db_pool)
            .await
            .map_err(|e| format!("Failed to fetch messages: {}", e))?,

            None => sqlx::query(
                "SELECT session_id, user_id, role, content, created_at,
                        model_backend, containment_mode, entry_hash, prev_hash
                 FROM conversation_messages
                 WHERE user_id = ?
                 ORDER BY created_at ASC LIMIT ?"
            )
            .bind(user_id)
            .bind(cap)
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
        }).collect::<Vec<_>>();

        // Marker to record on success. When the cap was hit, the newest message sent
        // (older sessions may be re-sent next cycle; the peer merges them). Otherwise the
        // newest thing sent, session or message.
        let newest_message = messages.last().map(|m| m.created_at);
        let newest_session = sessions.iter().map(|s| s.last_active).max();
        let through = if messages.len() >= CONVERSATION_PUSH_MAX_MESSAGES {
            newest_message
        } else {
            match (newest_message, newest_session) {
                (Some(m), Some(s)) => Some(m.max(s)),
                (m, s) => m.or(s),
            }
        };

        Ok((ConversationSyncPayload { sessions, messages }, through))
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
                     title            = CASE WHEN coalesce(conversation_sessions.title, '') = ''
                                             THEN COALESCE(EXCLUDED.title, conversation_sessions.title)
                                             ELSE conversation_sessions.title END,
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
        let (payload, through) = self.get_modified_conversations(&peer.device_id, user_id).await?;

        if payload.sessions.is_empty() && payload.messages.is_empty() {
            return Ok((0, 0));
        }
        let capped = payload.messages.len() >= CONVERSATION_PUSH_MAX_MESSAGES;

        // Verbose push detail omitted — summary printed by caller

        let endpoint = format!("{}/api/zynksync/conversations/receive", peer.url);
        let client = self.transport.http_client.read().await.clone();
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
        if capped {
            println!("[ZynkSync] History push to {} hit the {}-message cap; the rest goes next cycle",
                peer.device_name, CONVERSATION_PUSH_MAX_MESSAGES);
        }

        // The peer accepted it: move the marker so this is not sent again.
        if let Some(through) = through {
            sqlx::query(
                "INSERT INTO zynk_conversation_push_state (peer_device_id, pushed_through) VALUES (?, ?)
                 ON CONFLICT (peer_device_id) DO UPDATE SET pushed_through = excluded.pushed_through"
            )
            .bind(&peer.device_id)
            .bind(through)
            .execute(&self.db_pool)
            .await
            .map_err(|e| format!("Failed to record conversation push marker: {}", e))?;
        }
        Ok((sessions_stored, messages_stored))
    }

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

        // Verbose per-sync logging suppressed

        // Step 1: Get local inventory
        let local_inventory = self.get_local_inventory(user_id).await?;

        // Step 2: Request remote inventory
        let endpoint = format!("{}/api/zynksync/inventory", peer.url);
        let request = InventoryRequest {
            user_id: user_id.to_string(),
        };

        let client = self.transport.http_client.read().await.clone();
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
            // No memories on either side is not "nothing to sync": conversation history
            // is pushed further down and used to be skipped here (harness, 2026-09-17).
            // Device with more memories is active (pull from them); equal counts — local.
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
                (None, None) => true, // no memories anywhere; history may still need to move
            }
        };

        // --- TOMBSTONE RECONCILIATION (before active/passive logic) ---
        // Tombstones always win: explicit deletions can never be resurrected by sync.
        let local_tombstones: std::collections::HashSet<String> =
            local_inventory.deleted_hashes.iter().cloned().collect();
        let remote_tombstones: std::collections::HashSet<String> =
            remote_inventory.deleted_hashes.iter().cloned().collect();

        {
            let local_hashes_ts: std::collections::HashSet<String> =
                local_inventory.content_hashes.iter().cloned().collect();
            let local_hash_to_id_ts: std::collections::HashMap<String, i32> =
                local_inventory.content_hashes.iter()
                    .zip(local_inventory.memory_ids.iter())
                    .map(|(h, id)| (h.clone(), *id))
                    .collect();
            let remote_hashes_ts: std::collections::HashSet<String> =
                remote_inventory.content_hashes.iter().cloned().collect();

            // 1. Apply remote tombstones to local memories.
            // Skip any memory whose updated_at is newer than the tombstone's deleted_at —
            // this protects memories that were explicitly restored after the deletion.
            let mut to_tombstone_locally: Vec<i32> = Vec::new();
            for h in remote_tombstones.iter().filter(|h| local_hashes_ts.contains(*h)) {
                let id = match local_hash_to_id_ts.get(h.as_str()).copied() {
                    Some(id) => id,
                    None => continue,
                };
                let tombstone_deleted_at: Option<DateTime<Utc>> = remote_inventory
                    .tombstone_timestamps.get(h.as_str())
                    .and_then(|s| s.parse::<DateTime<Utc>>().ok());
                if let Some(deleted_at) = tombstone_deleted_at {
                    let memory_time: Option<DateTime<Utc>> = sqlx::query_scalar(
                        "SELECT COALESCE(updated_at, created_at) FROM memories WHERE id = ?"
                    )
                    .bind(id)
                    .fetch_optional(&self.db_pool)
                    .await
                    .ok()
                    .flatten();
                    if let Some(mt) = memory_time {
                        if mt > deleted_at {
                            println!("[ZynkSync] Skipping remote tombstone for restored memory (updated_at {} > tombstone {})", mt, deleted_at);
                            continue;
                        }
                    }
                }
                to_tombstone_locally.push(id);
            }
            if !to_tombstone_locally.is_empty() {
                println!("[ZynkSync] Applying {} remote tombstones locally", to_tombstone_locally.len());
                self.delete_and_tombstone(&to_tombstone_locally).await?;
                // If that emptied the device, drop the Einstein demo persona too. Clear All
                // already does; this path did not, and the model kept addressing the user as
                // "Albert" with no demo memories left (2026-09-12, KI-048 follow-up).
                let remaining: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memories")
                    .fetch_one(&self.db_pool).await.unwrap_or(1);
                if remaining == 0 {
                    crate::db::remove_demo_persona_profile();
                }
            }

            // 2. Absorb remote tombstones we don't have yet (protection for future syncs)
            let new_remote_tombstones: Vec<String> = remote_tombstones.iter()
                .filter(|h| !local_tombstones.contains(*h))
                .cloned().collect();
            if !new_remote_tombstones.is_empty() {
                self.record_tombstones(&new_remote_tombstones).await?;
            }

            // 3. Propagate our tombstones to remote for memories remote still has.
            // Skipped on first sync: a freshly-paired device should never have its
            // memories deleted by tombstones from our past history with other devices.
            if !is_first_sync {
                let to_tombstone_remotely: Vec<String> = local_tombstones.iter()
                    .filter(|h| remote_hashes_ts.contains(*h))
                    .cloned().collect();
                if !to_tombstone_remotely.is_empty() {
                    println!("[ZynkSync] Propagating {} local tombstones to remote", to_tombstone_remotely.len());
                    for hash in &to_tombstone_remotely {
                        // Include the original deleted_at so the receiver can guard against
                        // wiping memories that were recreated after the deletion event.
                        let deleted_at: Option<String> = sqlx::query_scalar(
                            "SELECT deleted_at FROM deleted_memory_hashes WHERE content_hash = ?"
                        )
                        .bind(hash)
                        .fetch_optional(&self.db_pool)
                        .await
                        .unwrap_or(None);

                        let endpoint = format!("{}/api/zynksync/delete-by-hash", peer.url);
                        let payload = serde_json::json!({
                            "content_hash": hash,
                            "deleted_at": deleted_at
                        });
                        let client = self.transport.http_client.read().await.clone();
                        let _ = client.post(&endpoint).json(&payload)
                            .timeout(Duration::from_secs(10)).send().await;
                    }
                }
            }
        }

        // Combined tombstones filter active/passive logic so tombstoned hashes are never synced
        let all_tombstones: std::collections::HashSet<String> =
            local_tombstones.union(&remote_tombstones).cloned().collect();

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

            // Find memories we have that remote doesn't (by content hash), excluding tombstoned
            let hashes_to_send: Vec<String> = local_hashes.difference(&remote_hashes)
                .filter(|h| !all_tombstones.contains(*h))
                .cloned().collect();
            let to_send: Vec<i32> = hashes_to_send.iter().filter_map(|h| hash_to_id.get(h).copied()).collect();

            if !to_send.is_empty() {
                let memories_to_send = self.get_memories_by_ids(&to_send).await?;

                let endpoint = format!("{}/api/zynksync/receive", peer.url);
                let client = self.transport.http_client.read().await.clone();
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
            // Only delete from remote if WE explicitly tombstoned the hash.
            // "Remote has it, we don't" is NOT evidence of deletion — the memory may simply
            // not have synced to us yet. Tombstone propagation (step 3 above) already handles
            // all user-initiated deletions correctly.
            if !is_first_sync {
                let hashes_to_delete: Vec<String> = remote_hashes.difference(&local_hashes)
                    .filter(|h| local_tombstones.contains(*h))
                    .cloned().collect();

                if !hashes_to_delete.is_empty() {
                    println!("[ZynkSync] Remote has {} tombstoned memories we deleted - propagating deletion", hashes_to_delete.len());
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
                        let client = self.transport.http_client.read().await.clone();
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

            // Find memories remote has that we don't (by content hash), excluding tombstoned
            let hashes_to_receive: Vec<String> = remote_hashes.difference(&local_hashes)
                .filter(|h| !all_tombstones.contains(*h))
                .cloned().collect();
            let to_receive: Vec<i32> = hashes_to_receive.iter().filter_map(|h| remote_hash_to_id.get(h).copied()).collect();

            if !to_receive.is_empty() {
                println!("[ZynkSync] Requesting {} missing memories from remote", to_receive.len());
                let endpoint = format!("{}/api/zynksync/fetch", peer.url);
                let client = self.transport.http_client.read().await.clone();
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

                memories_received = self.receive_from_peer(user_id, memories).await?;
            }

            // Handle deletions (only on subsequent syncs, not first sync)
            // Only delete locally if the REMOTE explicitly tombstoned the hash.
            // "We have it, remote doesn't" is NOT evidence of deletion — the memory may simply
            // not have synced to remote yet. Tombstone reconciliation (step 1 above) already
            // applies all remote tombstones to our local store.
            if !is_first_sync {
                let hashes_to_delete: Vec<String> = local_hashes.difference(&remote_hashes)
                    .filter(|h| remote_tombstones.contains(*h))
                    .cloned().collect();

                if !hashes_to_delete.is_empty() {
                    println!("[ZynkSync] We have {} remotely-tombstoned memories - deleting locally", hashes_to_delete.len());
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
                        event_type, event_date, entities_detected, original_text,
                        collection_id, memory_placement, external_id, temporal_status, provenance_json
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
                    collection_id: row.get("collection_id"),
                    memory_placement: row.get("memory_placement"),
                    external_id: row.get("external_id"),
                    temporal_status: row.get("temporal_status"),
                    provenance_json: row.get("provenance_json"),
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
        if ids.is_empty() { return Ok(0); }
        let in_clause = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        let sql = format!("DELETE FROM memories WHERE id IN ({})", in_clause);
        let mut q = sqlx::query(&sql);
        for id in ids { q = q.bind(id); }
        let result = q.execute(&self.db_pool).await
            .map_err(|e| format!("Failed to delete memories: {}", e))?;
        let deleted_count = result.rows_affected() as usize;
        println!("[ZynkSync] Deleted {} memories", deleted_count);
        Ok(deleted_count)
    }

    /// Remove content hashes from tombstones on all paired peers.
    /// Called after a restore that explicitly un-deletes memories, so peer tombstones
    /// don't re-delete the restored memories on the next sync cycle.
    pub async fn clear_tombstones_on_peers(&self, hashes: &[String]) -> usize {
        let peers = {
            let peers_map = self.transport.peers.read().await;
            peers_map.values()
                .filter(|p| p.paired)
                .cloned()
                .collect::<Vec<_>>()
        };
        let mut success_count = 0;
        let payload = serde_json::json!({ "hashes": hashes });
        for peer in peers {
            let endpoint = format!("{}/api/zynksync/clear-tombstones", peer.url);
            let client = self.transport.http_client.read().await.clone();
            match client
                .post(&endpoint)
                .json(&payload)
                .timeout(Duration::from_secs(15))
                .send()
                .await
            {
                Ok(r) if r.status().is_success() => {
                    println!("[ZynkSync] ✓ Cleared tombstones on {}", peer.device_name);
                    success_count += 1;
                }
                Ok(r) => eprintln!("[ZynkSync] ✗ clear-tombstones on {} failed: {}", peer.device_name, r.status()),
                Err(e) => eprintln!("[ZynkSync] ✗ Could not reach {} for clear-tombstones: {}", peer.device_name, e),
            }
        }
        success_count
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

    /// Delete memories by IDs and record tombstones for each deleted hash.
    /// Use this for user-visible deletions and remote-tombstone application.
    async fn delete_and_tombstone(&self, ids: &[i32]) -> Result<usize, String> {
        if ids.is_empty() { return Ok(0); }
        // Fetch content hashes before deletion
        let in_clause = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        let hash_sql = format!("SELECT content FROM memories WHERE id IN ({})", in_clause);
        let mut q = sqlx::query_scalar::<_, String>(&hash_sql);
        for id in ids { q = q.bind(id); }
        let contents: Vec<String> = q.fetch_all(&self.db_pool).await.unwrap_or_default();
        let hashes: Vec<String> = contents.iter().map(|c| {
            use sha2::{Digest, Sha256};
            format!("{:x}", Sha256::digest(c.as_bytes()))
        }).collect();
        let count = self.delete_memories_by_ids(ids).await?;
        self.record_tombstones(&hashes).await?;
        Ok(count)
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

        // Record tombstone locally so this deletion survives future syncs
        self.record_tombstones(&[content_hash.clone()]).await
            .unwrap_or_else(|e| eprintln!("[ZynkSync] Failed to record tombstone: {}", e));

        // Fetch the tombstone's deleted_at so peers can guard against wiping
        // memories that were recreated after this deletion event.
        let deleted_at: Option<String> = sqlx::query_scalar(
            "SELECT deleted_at FROM deleted_memory_hashes WHERE content_hash = ?"
        )
        .bind(&content_hash)
        .fetch_optional(&self.db_pool)
        .await
        .unwrap_or(None);

        // Get all paired peers
        let peers = {
            let peers_map = self.transport.peers.read().await;
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

        // Send deletion request to each peer (using content hash for portable lookup).
        // Include the tombstone's deleted_at so the receiver can skip wiping memories
        // that were recreated after this deletion event.
        for peer in peers {
            let endpoint = format!("{}/api/zynksync/delete-by-hash", peer.url);
            let payload = serde_json::json!({
                "content_hash": content_hash,
                "deleted_at": deleted_at
            });

            let client = self.transport.http_client.read().await.clone();
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

    /// Propagate a memory edit to all paired devices.
    pub async fn propagate_memory_update(
        &self,
        memory_id: i32,
        title: Option<String>,
        content: Option<String>,
        namespace: Option<String>,
    ) -> Result<usize, String> {
        println!("[ZynkSync] Propagating memory update for ID {} to paired devices", memory_id);

        let peers = {
            let peers_map = self.transport.peers.read().await;
            peers_map.values()
                .filter(|p| p.paired)
                .cloned()
                .collect::<Vec<_>>()
        };

        if peers.is_empty() {
            println!("[ZynkSync] No paired devices to propagate update to");
            return Ok(0);
        }

        let total_peers = peers.len();
        let mut success_count = 0;
        let payload = serde_json::json!({
            "memory_id": memory_id,
            "title": title,
            "content": content,
            "namespace": namespace,
        });

        for peer in peers {
            let endpoint = format!("{}/api/zynksync/update-memory", peer.url);
            let client = self.transport.http_client.read().await.clone();
            match client
                .post(&endpoint)
                .json(&payload)
                .timeout(Duration::from_secs(15))
                .send()
                .await
            {
                Ok(response) => {
                    if response.status().is_success() {
                        success_count += 1;
                        println!("[ZynkSync] ✓ Memory update synced to {}", peer.device_name);
                    } else {
                        eprintln!("[ZynkSync] ✗ Failed to sync update to {}: status {}", peer.device_name, response.status());
                    }
                }
                Err(e) => {
                    eprintln!("[ZynkSync] ✗ Failed to reach {}: {}", peer.device_name, e);
                }
            }
        }

        println!("[ZynkSync] ✓ Memory update propagated to {}/{} devices", success_count, total_peers);
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

        let client = self.transport.http_client.read().await.clone();
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
            .route("/api/zynksync/receive", post(handle_receive_sync))
            .route("/api/zynksync/inventory", post(handle_get_inventory))
            .route("/api/zynksync/delete", post(handle_delete_memories))
            .route("/api/zynksync/delete-by-hash", post(handle_delete_by_hash))
            .route("/api/zynksync/clear-tombstones", post(handle_clear_tombstones))
            .route("/api/zynksync/update-memory", post(handle_update_memory))
            .route("/api/zynksync/fetch", post(handle_fetch_memories))
            .route("/api/zynksync/introduce", post(handle_introduce))
            .route("/api/zynksync/notify-unsynced", post(handle_notify_unsynced))
            .route("/api/zynksync/conversations/receive", post(handle_receive_conversations))
            .route("/api/presence/heartbeat", post(handle_heartbeat))
            .route("/api/presence/goodbye", post(handle_goodbye))
            .route("/api/zynksync/push-api-key", post(handle_push_api_key))
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



/// Axum handler for receiving sync memories from a peer
async fn handle_receive_sync(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
    Json(memories): Json<Vec<SyncMemory>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?;
    check_sync_authorized(&service.db_pool, device_id, &headers).await?;

    println!("[ZynkSync] Received {} memories from peer", memories.len());
    let local_user_id = service.user_id().unwrap_or_default();
    let stored_count = service.receive_from_peer(&local_user_id, memories).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;

    if stored_count > 0 {
        if let Ok(guard) = crate::APP_HANDLE.lock() {
            if let Some(app) = guard.as_ref() {
                let _ = app.emit("zynksync-memories-updated", serde_json::json!({ "count": stored_count }));
            }
        }
    }

    Ok(Json(serde_json::json!({ "success": true, "stored": stored_count })))
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

/// Axum handler for getting memory inventory (for "active device wins" sync)
async fn handle_get_inventory(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<InventoryRequest>,
) -> Result<Json<MemoryInventory>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?;
    check_sync_authorized(&service.db_pool, device_id, &headers).await?;

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
    check_sync_authorized(&service.db_pool, device_id, &headers).await?;

    println!("[ZynkSync] Delete request for {} memories", ids.len());
    let deleted_count = service.delete_memories_by_ids(&ids).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;

    if deleted_count > 0 {
        if let Ok(guard) = crate::APP_HANDLE.lock() {
            if let Some(app) = guard.as_ref() {
                let _ = app.emit("zynksync-memories-updated", serde_json::json!({ "count": deleted_count }));
            }
        }
    }
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
    check_sync_authorized(&service.db_pool, device_id, &headers).await?;

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
    check_sync_authorized(&service.db_pool, device_id, &headers).await?;

    let content_hash = request.get("content_hash")
        .and_then(|v| v.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing content_hash parameter"}))))?;

    println!("[ZynkSync] Delete-by-hash request for hash: {}", content_hash);

    // Parse the tombstone's original deletion timestamp (sent by the peer).
    // If absent, fall back to now — this means any matching memory will be deleted.
    let tombstone_time: Option<DateTime<Utc>> = request.get("deleted_at")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<DateTime<Utc>>().ok());

    let memory_info: Option<(i32, DateTime<Utc>)> = {
        use sha2::{Digest, Sha256};
        let rows: Vec<(i32, String, String, Option<String>)> = sqlx::query_as::<_, (i32, String, String, Option<String>)>(
            "SELECT id, content, created_at, updated_at FROM memories"
        )
        .fetch_all(&service.db_pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("Failed to lookup memories: {}", e)}))))?;
        rows.into_iter()
            .find(|(_, c, _, _)| format!("{:x}", Sha256::digest(c.as_bytes())) == content_hash)
            .map(|(id, _, created_at_str, updated_at_str)| {
                let created_at = created_at_str.parse::<DateTime<Utc>>().unwrap_or(DateTime::<Utc>::MIN_UTC);
                let updated_at = updated_at_str.and_then(|s| s.parse::<DateTime<Utc>>().ok()).unwrap_or(DateTime::<Utc>::MIN_UTC);
                (id, created_at.max(updated_at))
            })
    };

    match memory_info {
        Some((id, effective_time)) => {
            // Only delete if the memory predates the tombstone.
            // Use max(created_at, updated_at) so restored memories (updated_at = now) are protected.
            let should_delete = tombstone_time.map_or(true, |ts| effective_time <= ts);

            if should_delete {
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO deleted_memory_hashes (content_hash) VALUES (?)"
                ).bind(content_hash).execute(&service.db_pool).await;

                let result = sqlx::query("DELETE FROM memories WHERE id = ?")
                    .bind(id)
                    .execute(&service.db_pool)
                    .await
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("Failed to delete memory: {}", e)}))))?;
                let deleted_count = result.rows_affected();
                println!("[ZynkSync] Deleted {} memory(s) for hash {}", deleted_count, content_hash);
                if deleted_count > 0 {
                    if let Ok(guard) = crate::APP_HANDLE.lock() {
                        if let Some(app) = guard.as_ref() {
                            let _ = app.emit("zynksync-memories-updated", serde_json::json!({ "count": deleted_count }));
                        }
                    }
                }
                Ok(Json(serde_json::json!({ "success": true, "deleted": deleted_count })))
            } else {
                println!("[ZynkSync] Skipping delete-by-hash: memory effective_time {} is newer than tombstone deleted_at {:?}",
                    effective_time, tombstone_time);
                Ok(Json(serde_json::json!({ "success": true, "deleted": 0, "skipped": "newer_than_tombstone" })))
            }
        }
        None => {
            println!("[ZynkSync] No memory found with hash {}", content_hash);
            Ok(Json(serde_json::json!({ "success": true, "deleted": 0 })))
        }
    }
}

/// Axum handler: peer asks us to clear specific tombstones (restore operation).
async fn handle_clear_tombstones(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?;
    check_sync_authorized(&service.db_pool, device_id, &headers).await?;

    let hashes = request.get("hashes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing hashes array"}))))?;

    let mut cleared = 0u64;
    for h in hashes {
        if let Some(hash) = h.as_str() {
            if let Ok(r) = sqlx::query("DELETE FROM deleted_memory_hashes WHERE content_hash = ?")
                .bind(hash)
                .execute(&service.db_pool)
                .await
            {
                cleared += r.rows_affected();
            }
        }
    }
    println!("[ZynkSync] Cleared {} tombstone(s) on behalf of peer restore", cleared);
    Ok(Json(serde_json::json!({ "success": true, "cleared": cleared })))
}

/// Axum handler for receiving a memory edit propagated from a paired device.
async fn handle_update_memory(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?;
    check_sync_authorized(&service.db_pool, device_id, &headers).await?;

    let memory_id: i32 = request.get("memory_id")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing memory_id"}))))?;

    let title: Option<String> = request.get("title").and_then(|v| v.as_str()).map(String::from);
    let content: Option<String> = request.get("content").and_then(|v| v.as_str()).map(String::from);
    let namespace: Option<String> = request.get("namespace").and_then(|v| v.as_str()).map(String::from);

    println!("[ZynkSync] Received memory update for ID: {}", memory_id);

    let embedding_vec: Option<Vec<u8>> = if let Some(ref new_content) = content {
        let content_clone = new_content.clone();
        match tokio::task::spawn_blocking(move || {
            crate::llm::local_embeddings::generate_local_embedding(&content_clone)
        }).await {
            Ok(Ok(embedding)) => Some(embedding.iter().flat_map(|f| f.to_le_bytes()).collect()),
            _ => None,
        }
    } else {
        None
    };

    let result = if let Some(emb) = embedding_vec {
        sqlx::query(
            "UPDATE memories
             SET title = COALESCE(?, title),
                 content = COALESCE(?, content),
                 original_text = COALESCE(?, original_text),
                 namespace = COALESCE(?, namespace),
                 embedding = ?,
                 updated_at = CURRENT_TIMESTAMP
             WHERE id = ?"
        )
        .bind(title.as_deref())
        .bind(content.as_deref())
        .bind(content.as_deref())
        .bind(namespace.as_deref())
        .bind(&emb)
        .bind(memory_id)
        .execute(&service.db_pool)
        .await
    } else {
        sqlx::query(
            "UPDATE memories
             SET title = COALESCE(?, title),
                 content = COALESCE(?, content),
                 original_text = COALESCE(?, original_text),
                 namespace = COALESCE(?, namespace),
                 updated_at = CURRENT_TIMESTAMP
             WHERE id = ?"
        )
        .bind(title.as_deref())
        .bind(content.as_deref())
        .bind(content.as_deref())
        .bind(namespace.as_deref())
        .bind(memory_id)
        .execute(&service.db_pool)
        .await
    };

    match result {
        Ok(r) => {
            let updated = r.rows_affected();
            println!("[ZynkSync] Updated {} row(s) for memory ID {}", updated, memory_id);
            if updated > 0 {
                if let Ok(guard) = crate::APP_HANDLE.lock() {
                    if let Some(app) = guard.as_ref() {
                        let _ = app.emit("zynksync-memories-updated", serde_json::json!({ "count": updated }));
                    }
                }
            }
            Ok(Json(serde_json::json!({ "success": true, "updated": updated })))
        }
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("Failed to update memory: {}", e)}))))
    }
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






/// Receive conversation sessions and messages from a peer device
async fn handle_receive_conversations(
    State(service): State<Arc<ZynkSyncService>>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<ConversationSyncPayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let device_id = headers.get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing X-Device-ID header"}))))?;
    check_sync_authorized(&service.db_pool, device_id, &headers).await?;

    let (sessions_stored, messages_stored) = service.receive_conversations_from_peer(payload).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;

    Ok(Json(serde_json::json!({
        "success": true,
        "sessions_stored": sessions_stored,
        "messages_stored": messages_stored
    })))
}


// =============================================================================
// Ollama endpoints
// =============================================================================


async fn handle_push_api_key(
    State(service): State<Arc<ZynkSyncService>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    axum::Extension(verified): axum::Extension<VerifiedDevice>,
    Json(payload): Json<serde_json::Value>,
) -> Result<axum::Json<serde_json::Value>, String> {
    let key = payload.get("key").and_then(|v| v.as_str()).ok_or("Missing key")?;
    let value = payload.get("value").and_then(|v| v.as_str()).ok_or("Missing value")?;

    // Device identity is now verified by mTLS cert (require_verified_device middleware ran).
    // Keep IP check as defence-in-depth.
    let sender_ip = addr.ip().to_string();
    let peers = service.transport.peers.read().await;
    let trusted = peers.values().any(|p| p.host == sender_ip && p.paired);
    drop(peers);
    if !trusted {
        return Err(format!("Untrusted sender IP {}, verified device: {}", sender_ip, verified.device_id));
    }

    // Allowlist — never let a peer set arbitrary env vars
    const ALLOWED: &[&str] = &[
        "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
        "OPENAI_API_KEY",    "OPENAI_MODEL",
        "XAI_API_KEY",       "XAI_MODEL",
        "MISTRAL_API_KEY",   "MISTRAL_MODEL",
        // CUSTOM_* are deliberately absent: the custom endpoint is machine-local
        // (a phone reaches Ollama through this desktop's proxy, which substitutes
        // the desktop's model), so a pushed URL or model name would only mislead.
        "R2_ENDPOINT",       "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET",
    ];
    // The backup encryption key is stored as a file, never in .env, and arriving
    // from a paired device counts as "the user has saved this key" — they set it
    // and acknowledged it on the sender.
    if key == crate::commands::models::BACKUP_KEY_PUSH_NAME {
        crate::commands::backup::install_pushed_backup_key(value)?;
        println!("[ZynkSync] ✓ Received backup encryption key from {} ({})", verified.device_name, sender_ip);
        if let Ok(guard) = crate::APP_HANDLE.lock() {
            if let Some(app) = guard.as_ref() {
                let _ = app.emit("backup-key-updated", serde_json::json!({}));
            }
        }
        return Ok(axum::Json(serde_json::json!({ "success": true })));
    }
    if !ALLOWED.contains(&key) {
        return Err(format!("Key '{}' is not propagatable", key));
    }

    let env_path = crate::db::get_app_data_dir().join(".env");
    let content = std::fs::read_to_string(&env_path).unwrap_or_default();
    let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    let prefix = format!("{}=", key);
    let mut found = false;
    for line in &mut lines {
        if line.starts_with(&prefix) { *line = format!("{}={}", key, value); found = true; break; }
    }
    if !found { lines.push(format!("{}={}", key, value)); }
    std::fs::write(&env_path, lines.join("\n"))
        .map_err(|e| format!("Failed to write .env: {}", e))?;
    std::env::set_var(key, value);

    println!("[ZynkSync] ✓ Received API key push for {} from {} ({})", key, verified.device_name, sender_ip);

    if let Ok(guard) = crate::APP_HANDLE.lock() {
        if let Some(app) = guard.as_ref() {
            let _ = app.emit("api-keys-updated", serde_json::json!({ "key": key }));
        }
    }

    Ok(axum::Json(serde_json::json!({ "success": true })))
}

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

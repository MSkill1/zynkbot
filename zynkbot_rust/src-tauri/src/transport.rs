//! Transport — what every peer-to-peer feature shares, and nothing they don't.
//!
//! One module owns the things that are genuinely common to ZynkSync (memories,
//! history, keys), ZynkLink (files) and ZChat (messages): this device's identity
//! and certificate, the HTTPS server and its TLS accept loop, the pinned mTLS
//! client, the device registry (`zynk_devices`) and presence. The three services
//! are clients of it; each owns only its own tables and routes. This is the
//! SDK's Core-layer "Transport" module (docs/SDK_VISION.md); extracted from
//! `zynksync.rs` on 2026-09-17 with the two-peer harness as the safety net.
//!
//! Design rules:
//! - identity and port are values the transport owns, never re-read from disk
//!   (several transports can coexist in one process — the test harness);
//! - the sync port is defined once (`DEFAULT_SYNC_PORT`), an instance may
//!   override it, and a peer's port is always taken from its stored row;
//! - a pairing of either kind (same user's device, or another user's) pins a
//!   certificate here, so every route can be verified the same way.

use chrono::{DateTime, Utc};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as HyperConnBuilder;
use serde::{Deserialize, Serialize};
use sqlx::{Row, SqlitePool};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpSocket};
use tokio::sync::RwLock;
use tokio_rustls::TlsAcceptor;

use crate::tls::PeerCertDer;
use crate::user_identity;

/// The well-known port every Zynkbot listens on for ZynkSync, ZynkLink and ZChat.
/// Defined once: an instance may override it (the sync test harness runs several
/// peers in one process), and a peer's port is always taken from its stored
/// device row, never assumed.
pub const DEFAULT_SYNC_PORT: u16 = 57963;

/// Who this device is. Owned by the transport rather than read from the identity
/// files at every call, so two transports can coexist in one process and an
/// identity change (pairing adoption, reset, rename) reaches the running service
/// at once.
#[derive(Clone, Debug)]
pub struct SyncIdentity {
    pub user_id: String,
    pub device_id: String,
    pub device_name: String,
}

impl SyncIdentity {
    /// The production identity: whatever the identity files say right now.
    pub fn from_files() -> Result<Self, String> {
        let id = user_identity::get_identity()?;
        Ok(Self { user_id: id.user_id, device_id: id.device_id, device_name: user_identity::get_device_name() })
    }
}

/// A device this one is paired with (either kind of pairing), as held in memory
/// for addressing. The durable record is the `zynk_devices` row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerDevice {
    pub device_id: String,
    pub device_name: String,
    pub host: String,
    pub port: u16,
    pub url: String,
    pub last_seen: DateTime<Utc>,
    pub paired: bool,                  // authorised to sync
    pub pairing_code: Option<String>,  // 6-digit code while pairing
    pub user_id: Option<String>,       // the host's user id (identity adoption during pairing)
    #[serde(default)]
    pub is_online: bool,               // heartbeat received within the last 45 s
}

/// "192.168.0.5" -> ("192.168.0.5", default); "192.168.0.5:4444" -> ("192.168.0.5", 4444).
pub fn split_host_port(s: &str, default: u16) -> (String, u16) {
    let s = s.trim();
    if let Some((h, p)) = s.rsplit_once(':') {
        if !h.contains(':') {
            if let Ok(port) = p.parse::<u16>() { return (h.to_string(), port); }
        }
    }
    (s.to_string(), default)
}

/// Host and port out of a URL such as "https://192.168.0.5:57963/api/...".
pub fn extract_host_port(url: &str) -> Option<(String, u16)> {
    let without_scheme = url.trim_start_matches("https://").trim_start_matches("http://");
    let host_port = without_scheme.split('/').next()?;
    if let Some(colon_pos) = host_port.rfind(':') {
        let host = host_port[..colon_pos].to_string();
        let port: u16 = host_port[colon_pos + 1..].parse().ok()?;
        Some((host, port))
    } else {
        Some((host_port.to_string(), if url.starts_with("https://") { 443 } else { 80 }))
    }
}

/// Bind with SO_REUSEADDR so a quick restart can reclaim a port still in TIME_WAIT
/// from the previous process; retries briefly while the old socket lets go.
async fn bind_with_reuseaddr(addr: SocketAddr, port: u16) -> Result<TcpListener, String> {
    let mut last_err = None;
    for attempt in 1..=6u32 {
        let socket = TcpSocket::new_v4().map_err(|e| format!("Failed to create socket: {}", e))?;
        if let Err(e) = socket.set_reuseaddr(true) {
            return Err(format!("Failed to set SO_REUSEADDR: {}", e));
        }
        match socket.bind(addr) {
            Ok(()) => match socket.listen(1024) {
                Ok(listener) => return Ok(listener),
                Err(e) => last_err = Some(format!("listen() failed on port {}: {}", port, e)),
            },
            Err(e) => last_err = Some(format!("bind() failed on port {} (attempt {}): {}", port, attempt, e)),
        }
        tokio::time::sleep(std::time::Duration::from_millis(500 * attempt as u64)).await;
    }
    Err(last_err.unwrap_or_else(|| "bind failed".to_string()))
}

/// The shared peer-to-peer layer for one device.
pub struct Transport {
    identity: std::sync::RwLock<SyncIdentity>,
    /// The port this instance listens on (DEFAULT_SYNC_PORT in production; 0 = any free port).
    port: u16,
    pub(crate) db_pool: SqlitePool,
    /// This device's TLS material (PEM for the server, DER handed to peers at pairing).
    pub(crate) cert_pem: String,
    pub(crate) key_pem: String,
    pub(crate) cert_der: Vec<u8>,
    /// Outbound client trusting every pinned peer certificate; rebuilt after each pairing.
    pub(crate) http_client: Arc<RwLock<reqwest::Client>>,
    /// Paired devices by device id.
    pub(crate) peers: Arc<RwLock<HashMap<String, PeerDevice>>>,
    /// Last heartbeat per peer device id (not persisted).
    pub(crate) peer_last_seen: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    /// Debounce for connection-error logging per peer.
    pub(crate) last_conn_error_logged: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    /// Failed pairing attempts per client IP — the code is invalidated after 5.
    pub(crate) failed_pairing_attempts: Arc<RwLock<HashMap<String, u32>>>,
    /// The pairing code currently on offer.
    pub(crate) pairing_code: Arc<RwLock<Option<String>>>,
    /// The port the listener actually got (differs from `port` only when it was 0).
    pub(crate) server_port: Arc<RwLock<Option<u16>>>,
    pub(crate) shutdown_tx: Arc<RwLock<Option<tokio::sync::oneshot::Sender<()>>>>,
}

impl Transport {
    pub fn new(identity: SyncIdentity, port: Option<u16>, db_pool: SqlitePool, cert_pem: String, key_pem: String, cert_der: Vec<u8>) -> Self {
        Self {
            identity: std::sync::RwLock::new(identity),
            port: port.unwrap_or(DEFAULT_SYNC_PORT),
            db_pool,
            cert_pem, key_pem, cert_der,
            http_client: Arc::new(RwLock::new(reqwest::Client::new())),
            peers: Arc::new(RwLock::new(HashMap::new())),
            peer_last_seen: Arc::new(RwLock::new(HashMap::new())),
            last_conn_error_logged: Arc::new(RwLock::new(HashMap::new())),
            failed_pairing_attempts: Arc::new(RwLock::new(HashMap::new())),
            pairing_code: Arc::new(RwLock::new(None)),
            server_port: Arc::new(RwLock::new(None)),
            shutdown_tx: Arc::new(RwLock::new(None)),
        }
    }

    // ---- identity and port -------------------------------------------------

    pub fn identity(&self) -> SyncIdentity { self.identity.read().unwrap().clone() }
    pub fn device_id(&self) -> String { self.identity.read().unwrap().device_id.clone() }
    pub fn device_name(&self) -> String { self.identity.read().unwrap().device_name.clone() }

    /// This device's user id. Err when the identity has no user id (never in practice).
    pub fn user_id(&self) -> Result<String, String> {
        let id = self.identity.read().unwrap().user_id.clone();
        if id.is_empty() { Err("No user id".to_string()) } else { Ok(id) }
    }

    /// Called when this device adopts another user id (pairing) or resets.
    pub fn set_user_id(&self, user_id: &str) { self.identity.write().unwrap().user_id = user_id.to_string(); }

    /// Called when the user renames this device.
    pub fn set_device_name(&self, name: &str) { self.identity.write().unwrap().device_name = name.to_string(); }

    /// The port this instance listens on (0 until the listener reports the port it was given).
    pub fn port(&self) -> u16 {
        if self.port != 0 { return self.port; }
        self.server_port.try_read().ok().and_then(|g| *g).unwrap_or(0)
    }

    /// The port a known peer listens on, from its stored row; the default if unknown.
    pub async fn peer_port_by_id(&self, device_id: &str) -> u16 {
        sqlx::query_scalar::<_, i64>("SELECT port FROM zynk_devices WHERE device_id = ?")
            .bind(device_id).fetch_optional(&self.db_pool).await.ok().flatten()
            .map(|p| p as u16).unwrap_or(DEFAULT_SYNC_PORT)
    }

    /// The port of the peer last seen at this address, from its stored row; the default if unknown.
    pub async fn peer_port_by_ip(&self, device_ip: &str) -> u16 {
        sqlx::query_scalar::<_, i64>("SELECT port FROM zynk_devices WHERE device_ip = ? ORDER BY last_seen_at DESC LIMIT 1")
            .bind(device_ip).fetch_optional(&self.db_pool).await.ok().flatten()
            .map(|p| p as u16).unwrap_or(DEFAULT_SYNC_PORT)
    }

    // ---- device registry -----------------------------------------------------

    /// This device's own row in zynk_devices. zynk_device_pairings references it, so
    /// a device that never generated a pairing code (a phone that only ever entered
    /// one) could not record its sync timestamps: every sync it started failed with a
    /// foreign-key error. Found by the two-peer harness, 2026-09-17.
    pub async fn ensure_own_device_row(&self) -> Result<(), String> {
        sqlx::query(
            "INSERT INTO zynk_devices (device_id, device_name, is_paired, port, created_at, last_seen_at)
             VALUES (?, ?, 1, ?, ?, ?)
             ON CONFLICT (device_id) DO UPDATE SET device_name = excluded.device_name, port = excluded.port"
        )
        .bind(self.device_id()).bind(self.device_name()).bind(self.port() as i32).bind(Utc::now()).bind(Utc::now())
        .execute(&self.db_pool).await
        .map(|_| ())
        .map_err(|e| format!("Failed to record this device: {}", e))
    }

    /// Fill the in-memory peer map from the stored device rows.
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

        let own_id = self.device_id();
        let mut peers_map = self.peers.write().await;
        for row in rows {
            let device_id: String = row.get("device_id");
            if device_id == own_id { continue; } // never sync with ourselves
            let device_name: String = row.get("device_name");
            let host: Option<String> = row.get("device_ip");
            let port: i32 = row.get("port");
            let last_seen: DateTime<Utc> = row.get("last_seen_at");
            let host = match host { Some(h) => h, None => continue }; // no address yet
            peers_map.insert(device_id.clone(), PeerDevice {
                device_id,
                device_name,
                host: host.clone(),
                port: port as u16,
                url: format!("https://{}:{}", host, port),
                last_seen,
                paired: true,
                pairing_code: None,
                user_id: None,
                is_online: false,
            });
        }
        println!("[Transport] Loaded {} devices from database", peers_map.len());
        Ok(())
    }

    /// The peer list for the UI: stored names win over the names cached at pairing
    /// (a rename arrives through the x-device-name header and is written to the row),
    /// and online = a heartbeat in the last 45 s.
    pub async fn get_peers(&self) -> Vec<PeerDevice> {
        let stored_names: HashMap<String, String> =
            sqlx::query("SELECT device_id, device_name FROM zynk_devices WHERE sync_paired = 1")
                .fetch_all(&self.db_pool)
                .await
                .map(|rows| rows.iter().filter_map(|r| {
                    let id: String = r.try_get("device_id").ok()?;
                    let name: String = r.try_get("device_name").ok()?;
                    if name.trim().is_empty() { None } else { Some((id, name)) }
                }).collect())
                .unwrap_or_default();
        let peers_map = self.peers.read().await;
        let online_map = self.peer_last_seen.read().await;
        let threshold = Utc::now() - chrono::Duration::seconds(45);
        peers_map.values().map(|peer| {
            let mut p = peer.clone();
            if let Some(name) = stored_names.get(&peer.device_id) { p.device_name = name.clone(); }
            p.is_online = online_map.get(&peer.device_id).map_or(false, |&t| t > threshold);
            p
        }).collect()
    }

    // ---- outbound client --------------------------------------------------------

    /// Rebuild the outbound client to trust every stored peer certificate. Called
    /// after startup's load_devices and after each new pairing. Every request carries
    /// x-device-id (so a peer that removed us can refuse) and x-device-name (so a
    /// rename reaches peers at their next contact, with no re-pair).
    pub async fn rebuild_http_client(&self) -> Result<(), String> {
        let rows = sqlx::query(
            "SELECT tls_cert_der FROM zynk_devices WHERE tls_cert_der IS NOT NULL AND sync_paired = 1"
        )
        .fetch_all(&self.db_pool)
        .await
        .map_err(|e| format!("Failed to load peer certs: {}", e))?;

        let cert_count = rows.len();
        let mut default_headers = reqwest::header::HeaderMap::new();
        if let Ok(val) = reqwest::header::HeaderValue::from_str(&self.device_id()) {
            default_headers.insert("x-device-id", val);
        }
        // HeaderValue rejects non-ASCII; a unicode name just does not propagate this way.
        if let Ok(val) = reqwest::header::HeaderValue::from_str(&self.device_name()) {
            default_headers.insert("x-device-name", val);
        }

        let mut pinned_ders: Vec<Vec<u8>> = Vec::new();
        for row in rows {
            let cert_der: Option<Vec<u8>> = row.try_get("tls_cert_der").ok().flatten();
            match cert_der {
                Some(der) => pinned_ders.push(der),
                None => println!("[TLS] Warning: peer row has NULL tls_cert_der"),
            }
        }
        println!("[TLS] rebuild_http_client: {} peer cert(s) in DB, {} pinned", cert_count, pinned_ders.len());

        let tls_config = match crate::tls::build_pinned_client_config_with_cert(pinned_ders.clone(), &self.cert_pem, &self.key_pem) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("[TLS] Failed to build client config with cert: {}, falling back to no-cert", e);
                crate::tls::build_pinned_client_config(pinned_ders)
            }
        };
        let client = reqwest::ClientBuilder::new()
            .use_preconfigured_tls(tls_config)
            .timeout(std::time::Duration::from_secs(30))
            .default_headers(default_headers)
            .build()
            .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

        *self.http_client.write().await = client;
        Ok(())
    }

    pub async fn get_http_client(&self) -> reqwest::Client {
        self.http_client.read().await.clone()
    }

    /// The pinned client if `url` points at a known paired peer, else None.
    pub async fn get_peer_client_for_url(&self, url: &str) -> Option<reqwest::Client> {
        let (host, port) = extract_host_port(url)?;
        let peers = self.peers.read().await;
        let is_peer = peers.values().any(|p| p.host == host && p.port == port);
        if is_peer { Some(self.get_http_client().await) } else { None }
    }

    // ---- presence ------------------------------------------------------------------

    /// Tell every paired peer we are going offline (best effort, short timeout).
    pub async fn send_goodbye_to_peers(&self) {
        let device_id = self.device_id();
        let peers: Vec<PeerDevice> = self.peers.read().await.values().cloned().collect();
        for peer in peers {
            if !peer.paired { continue; }
            let url = format!("https://{}:{}/api/presence/goodbye", peer.host, peer.port);
            let body = serde_json::json!({ "device_id": device_id });
            let client = self.http_client.read().await.clone();
            let _ = client.post(&url).json(&body).timeout(std::time::Duration::from_secs(3)).send().await;
        }
        println!("[Presence] Goodbye sent to all peers");
    }

    // ---- server ---------------------------------------------------------------------

    /// Bind the listener, record this device's own row, and serve `app` over TLS
    /// until shutdown. Each connection's peer certificate (if presented) and address
    /// are attached to every request as extensions, for the verification middleware.
    /// Returns the port actually bound.
    pub async fn serve(self: Arc<Self>, app: axum::Router) -> Result<u16, String> {
        let server_config = crate::tls::build_server_config_with_optional_client_auth(&self.cert_pem, &self.key_pem)
            .map_err(|e| format!("Failed to build TLS config: {}", e))?;
        let tls_acceptor = TlsAcceptor::from(Arc::new(server_config));

        let requested = self.port;
        let addr = SocketAddr::from(([0, 0, 0, 0], requested));
        let tcp_listener = bind_with_reuseaddr(addr, requested).await?;
        // With port 0 the OS picked one; report what we actually got.
        let port: u16 = tcp_listener.local_addr().map(|a| a.port()).unwrap_or(requested);
        *self.server_port.write().await = Some(port);
        self.ensure_own_device_row().await?;
        println!("[Transport] HTTPS server listening on port {}", port);

        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        *self.shutdown_tx.write().await = Some(shutdown_tx);

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => { println!("[Transport] HTTPS server shutting down"); break; }
                    result = tcp_listener.accept() => {
                        let (tcp_stream, peer_addr) = match result {
                            Ok(pair) => pair,
                            Err(e) => { eprintln!("[Transport] TCP accept error: {}", e); break; }
                        };
                        let acceptor = tls_acceptor.clone();
                        let router = app.clone();
                        tokio::spawn(async move {
                            let tls_stream = match acceptor.accept(tcp_stream).await {
                                Ok(s) => s,
                                Err(e) => {
                                    let e_str = e.to_string();
                                    if e_str.contains("HandshakeFailure") || e_str.contains("handshake")
                                        || e_str.contains("AlertReceived") || e_str.contains("corrupt message") {
                                        #[cfg(debug_assertions)]
                                        eprintln!("[Transport] TLS handshake failed from {} (debug): {}", peer_addr, e);
                                    } else {
                                        eprintln!("[Transport] TLS accept error from {}: {}", peer_addr, e);
                                    }
                                    return;
                                }
                            };
                            let peer_cert_der: Option<PeerCertDer> = tls_stream.get_ref().1
                                .peer_certificates().and_then(|certs| certs.first())
                                .map(|c| PeerCertDer(c.as_ref().to_vec()));
                            let io = TokioIo::new(tls_stream);
                            let svc = hyper::service::service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                                let mut router = router.clone();
                                let peer_cert_der = peer_cert_der.clone();
                                async move {
                                    let (parts, body) = req.into_parts();
                                    let body = axum::body::Body::new(body);
                                    let mut req = hyper::Request::from_parts(parts, body);
                                    req.extensions_mut().insert(axum::extract::ConnectInfo(peer_addr));
                                    if let Some(cert) = peer_cert_der { req.extensions_mut().insert(cert); }
                                    use tower::Service;
                                    router.call(req).await
                                }
                            });
                            if let Err(e) = HyperConnBuilder::new(TokioExecutor::new()).serve_connection(io, svc).await {
                                println!("[Transport] Connection from {} closed: {}", peer_addr, e);
                            }
                        });
                    }
                }
            }
            println!("[Transport] HTTPS server stopped");
        });
        Ok(port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_port_parsing_defaults_and_overrides() {
        assert_eq!(split_host_port("192.168.0.5", DEFAULT_SYNC_PORT), ("192.168.0.5".into(), 57963));
        assert_eq!(split_host_port("192.168.0.5:4444", DEFAULT_SYNC_PORT), ("192.168.0.5".into(), 4444));
        assert_eq!(split_host_port(" 10.0.0.2:1 ", DEFAULT_SYNC_PORT), ("10.0.0.2".into(), 1));
        assert_eq!(split_host_port("10.0.0.2:notaport", DEFAULT_SYNC_PORT), ("10.0.0.2:notaport".into(), 57963));
        assert_eq!(extract_host_port("https://192.168.0.5:57963/api/x"), Some(("192.168.0.5".into(), 57963)));
    }
}

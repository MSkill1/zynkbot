use std::sync::Arc;
use crate::zynksync::{PeerDevice, SyncResult};
use tauri::Emitter;

/// Start ZynkSync auto-sync loop (HTTP server already running from app launch)
#[tauri::command]
pub async fn start_zynksync(_sync_interval_secs: Option<u64>) -> Result<String, String> {
    println!("[ZynkSync] Starting auto-sync loop...");

    let service = {
        let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
        match global_service.as_ref() {
            Some(svc) => {
                if svc.is_auto_sync_enabled().await {
                    println!("[ZynkSync] ⚠️ Auto-sync already running");
                    return Err("ZynkSync auto-sync is already running. Stop it first.".to_string());
                }
                Arc::clone(svc)
            }
            None => {
                println!("[ZynkSync] ❌ HTTP server not running");
                return Err("HTTP server not started. Please restart the app.".to_string());
            }
        }
    };

    if let Err(e) = service.load_devices().await {
        println!("[ZynkSync] ⚠️ Failed to load devices: {}", e);
    }
    if let Err(e) = service.rebuild_http_client().await {
        println!("[ZynkSync] ⚠️ Failed to rebuild HTTP client with peer certs: {}", e);
    }

    println!("[ZynkSync] Generating pairing code...");
    match service.generate_pairing_code().await {
        Ok(code) => {
            println!("[ZynkSync] ✅ Pairing code: {} (expires in 10 minutes)", code);
        }
        Err(e) => {
            println!("[ZynkSync] ⚠️ Failed to generate pairing code: {}", e);
        }
    }

    let service_clone = Arc::clone(&service);
    tokio::spawn(async move {
        service_clone.start_auto_sync().await;
    });

    let service_clone = Arc::clone(&service);
    tokio::spawn(async move {
        service_clone.start_message_delivery_loop().await;
    });

    let service_clone = Arc::clone(&service);
    tokio::spawn(async move {
        service_clone.start_heartbeat_loop().await;
    });

    println!("[ZynkSync] ✅ Auto-sync started successfully");

    if let Err(e) = crate::save_sync_state(true).await {
        eprintln!("[ZynkSync] ⚠️ Failed to save sync state: {}", e);
    }

    let device_id = crate::user_identity::get_device_id()
        .map_err(|e| format!("Failed to get device ID: {}", e))?;
    Ok(device_id)
}

/// Stop ZynkSync auto-sync loop (HTTP server keeps running for ZynkLink)
#[tauri::command]
pub async fn stop_zynksync() -> Result<(), String> {
    println!("[ZynkSync] Stopping auto-sync...");

    let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
    if let Some(service) = global_service.as_ref() {
        if !service.is_auto_sync_enabled().await {
            println!("[ZynkSync] ⚠️ Auto-sync not running");
            return Err("ZynkSync auto-sync is not running".to_string());
        }

        service.stop_auto_sync().await;
        println!("[ZynkSync] ✅ Auto-sync stopped (HTTP server still running for ZynkLink)");

        if let Err(e) = crate::save_sync_state(false).await {
            eprintln!("[ZynkSync] ⚠️ Failed to save sync state: {}", e);
        }

        Ok(())
    } else {
        Err("HTTP server not started. Please restart the app.".to_string())
    }
}

/// Check if ZynkSync auto-sync is running
#[tauri::command]
pub async fn get_zynksync_status() -> Result<bool, String> {
    let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => Ok(service.is_auto_sync_enabled().await),
        None => Ok(false),
    }
}

/// Get list of discovered peer devices
#[tauri::command]
pub async fn get_zynksync_peers() -> Result<Vec<PeerDevice>, String> {
    let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => Ok(service.get_peers().await),
        None => Ok(Vec::new()),
    }
}

/// Manually trigger sync to a specific peer
#[tauri::command]
pub async fn sync_to_peer(peer_id: String, user_id: Option<String>) -> Result<SyncResult, String> {
    let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => service.sync_to_peer(&peer_id, user_id.as_deref()).await,
        None => Err("ZynkSync not started".to_string()),
    }
}

/// Receive memories from a peer device
#[tauri::command]
pub async fn receive_sync_memories(memories: Vec<crate::zynksync::SyncMemory>) -> Result<usize, String> {
    let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => service.receive_from_peer(memories).await,
        None => Err("ZynkSync not started".to_string()),
    }
}

/// Request pairing with a peer (generates 6-digit code)
#[tauri::command]
pub async fn request_device_pairing(peer_id: String) -> Result<String, String> {
    let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => service.request_pairing(&peer_id).await,
        None => Err("ZynkSync not started".to_string()),
    }
}

/// Verify pairing code and authorize peer
#[tauri::command]
pub async fn verify_pairing_code(peer_id: String, code: String) -> Result<(), String> {
    let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => service.verify_pairing_code(&peer_id, &code).await,
        None => Err("ZynkSync not started".to_string()),
    }
}

/// Unpair from a device
#[tauri::command]
pub async fn unpair_device(peer_id: String) -> Result<(), String> {
    let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => service.unpair_device(&peer_id).await,
        None => Err("ZynkSync not started".to_string()),
    }
}

/// Add a device manually by IP address and pairing code
#[tauri::command]
pub async fn add_zynksync_device(host_ip: String, pairing_code: String) -> Result<PeerDevice, String> {
    let service = {
        let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
        match global_service.as_ref() {
            Some(s) => std::sync::Arc::clone(s),
            None => return Err("ZynkSync not started".to_string()),
        }
    };
    let peer = service.add_device(&host_ip, &pairing_code).await?;
    if let Err(e) = service.rebuild_http_client().await {
        eprintln!("[ZynkSync] Warning: failed to rebuild HTTP client after pairing: {}", e);
    }
    Ok(peer)
}

/// Remove a manually added device
#[tauri::command]
pub async fn remove_zynksync_device(device_id: String) -> Result<(), String> {
    let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => service.remove_device(&device_id).await,
        None => Err("ZynkSync not started".to_string()),
    }
}

/// Get this device's pairing code for sharing
#[tauri::command]
pub async fn get_zynksync_pairing_code() -> Result<String, String> {
    let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => service.get_pairing_code().await,
        None => Err("ZynkSync not started".to_string()),
    }
}

/// Check sync status with all peers and emit event if user action needed
#[tauri::command]
pub async fn check_sync_status_with_peers(app: tauri::AppHandle, user_id: String) -> Result<serde_json::Value, String> {
    let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
    let service = match global_service.as_ref() {
        Some(svc) => svc,
        None => return Err("ZynkSync not started".to_string()),
    };

    let peers = service.get_peers().await;
    let paired_peers: Vec<_> = peers.iter().filter(|p| p.paired).collect();

    if paired_peers.is_empty() {
        return Ok(serde_json::json!({
            "needs_prompt": false,
            "reason": "no_peers"
        }));
    }

    let local_inventory = service.get_local_inventory_public(&user_id).await?;

    let mut local_is_more_recent = false;
    let mut peers_with_different_counts = Vec::new();

    for peer in paired_peers {
        match service.get_remote_inventory_public(&peer.url, &user_id).await {
            Ok(remote_inventory) => {
                let is_more_recent = match (&local_inventory.latest_activity, &remote_inventory.latest_activity) {
                    (Some(local_time), Some(remote_time)) => local_time > remote_time,
                    (Some(_), None) => true,
                    _ => false,
                };

                if is_more_recent && local_inventory.memory_count != remote_inventory.memory_count {
                    local_is_more_recent = true;
                    peers_with_different_counts.push(serde_json::json!({
                        "device_id": peer.device_id,
                        "device_name": peer.device_name,
                        "local_count": local_inventory.memory_count,
                        "remote_count": remote_inventory.memory_count,
                        "local_time": local_inventory.latest_activity,
                        "remote_time": remote_inventory.latest_activity,
                    }));
                }
            }
            Err(e) => {
                println!("[ZynkSync] Warning: Could not get inventory from {}: {}", peer.device_name, e);
            }
        }
    }

    if local_is_more_recent && !peers_with_different_counts.is_empty() {
        let _ = app.emit("sync_prompt_needed", serde_json::json!({
            "local_memory_count": local_inventory.memory_count,
            "peers": peers_with_different_counts,
        }));

        Ok(serde_json::json!({
            "needs_prompt": true,
            "local_memory_count": local_inventory.memory_count,
            "peers": peers_with_different_counts,
        }))
    } else {
        Ok(serde_json::json!({
            "needs_prompt": false,
            "reason": "no_difference"
        }))
    }
}

/// Force all paired peers to sync to this device's state (including deletions)
#[tauri::command]
pub async fn broadcast_sync_to_all_peers(user_id: String) -> Result<Vec<SyncResult>, String> {
    let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
    let service = match global_service.as_ref() {
        Some(svc) => Arc::clone(svc),
        None => return Err("ZynkSync not started".to_string()),
    };

    drop(global_service);

    let peers = service.get_peers().await;
    let paired_peers: Vec<_> = peers.into_iter().filter(|p| p.paired).collect();

    if paired_peers.is_empty() {
        return Ok(Vec::new());
    }

    println!("[ZynkSync] Broadcasting sync to {} paired devices", paired_peers.len());

    let mut results = Vec::new();
    for peer in paired_peers {
        match service.sync_bidirectional(&peer.device_id, &user_id).await {
            Ok(result) => {
                println!("[ZynkSync] ✓ Synced with {}: sent={}, received={}",
                    peer.device_name, result.memories_sent, result.memories_received);
                results.push(result);
            }
            Err(e) => {
                println!("[ZynkSync] ✗ Failed to sync with {}: {}", peer.device_name, e);
                results.push(SyncResult {
                    peer_device_id: peer.device_id,
                    peer_device_name: peer.device_name,
                    memories_sent: 0,
                    memories_received: 0,
                    conflicts_resolved: 0,
                    success: false,
                    error: Some(e),
                });
            }
        }
    }

    Ok(results)
}

/// Get the local IP address of this device
#[tauri::command]
pub async fn get_local_ip() -> Result<String, String> {
    use std::net::UdpSocket;

    match UdpSocket::bind("0.0.0.0:0") {
        Ok(socket) => {
            match socket.connect("8.8.8.8:80") {
                Ok(_) => {
                    match socket.local_addr() {
                        Ok(addr) => Ok(addr.ip().to_string()),
                        Err(e) => Err(format!("Failed to get local address: {}", e))
                    }
                }
                Err(e) => Err(format!("Failed to connect: {}", e))
            }
        }
        Err(e) => Err(format!("Failed to bind socket: {}", e))
    }
}

/// Clear all memories for a specific user
#[tauri::command]
pub async fn clear_all_memories(user_id: String, propagate: Option<bool>) -> Result<serde_json::Value, String> {
    let should_propagate = propagate.unwrap_or(true);

    println!("[Memory] Clearing all memories for user: {} (propagate: {})", user_id, should_propagate);

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    let content_hashes = if should_propagate {
        let contents: Vec<String> = sqlx::query_scalar::<_, String>(
            "SELECT content FROM memories WHERE user_id = ?"
        )
        .bind(&user_id)
        .fetch_all(&pool)
        .await
        .map_err(|e| format!("Failed to get memory contents: {}", e))?;
        use sha2::{Digest, Sha256};
        contents.iter().map(|c| format!("{:x}", Sha256::digest(c.as_bytes()))).collect()
    } else {
        Vec::new()
    };

    if should_propagate && !content_hashes.is_empty() {
        println!("[Memory] Found {} memories to delete and propagate", content_hashes.len());
    } else if !should_propagate {
        println!("[Memory] Deletion propagation disabled (identity adoption cleanup)");
    }

    let result = sqlx::query("DELETE FROM memories WHERE user_id = ?")
        .bind(&user_id)
        .execute(&pool)
        .await
        .map_err(|e| format!("Failed to clear memories: {}", e))?;

    let deleted_count = result.rows_affected();
    println!("[Memory] Deleted {} memories from local database", deleted_count);

    if should_propagate && !content_hashes.is_empty() {
        println!("[Memory] Propagating {} deletions to paired devices...", content_hashes.len());

        let service = {
            let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
            global_service.as_ref().cloned()
        };

        if let Some(service) = service {
            let mut propagated = 0;
            for hash in content_hashes {
                match service.propagate_deletion_by_hash(hash).await {
                    Ok(count) => propagated += count,
                    Err(e) => eprintln!("[Memory] Failed to propagate deletion: {}", e),
                }
            }
            println!("[Memory] ✓ Propagated deletions to {} device(s)", propagated);
        } else {
            println!("[Memory] ⚠️ ZynkSync not initialized - deletions not propagated");
        }
    }

    let profile_path = crate::db::get_user_profile_path();
    if profile_path.exists() {
        std::fs::remove_file(&profile_path).ok();
        println!("[Memory] Deleted user_profile.json");
    }

    Ok(serde_json::json!({
        "success": true,
        "deleted_count": deleted_count
    }))
}

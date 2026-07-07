use std::sync::Arc;
use crate::zynklink;
use crate::kb_rag;
use tauri::Emitter;

/// Generate a ZynkLink code for file sharing
#[tauri::command]
pub async fn generate_zynklink_code() -> Result<serde_json::Value, String> {
    println!("[ZynkLink] Generate code command invoked");

    let user_id = crate::user_identity::get_user_id()
        .map_err(|e| { println!("[ZynkLink] Failed to get user_id: {}", e); e })?;
    let device_id = crate::user_identity::get_device_id()
        .map_err(|e| { println!("[ZynkLink] Failed to get device_id: {}", e); e })?;

    println!("[ZynkLink] user_id: {}..., device_id: {}...", &user_id[..8], &device_id[..8]);

    let pool = {
        let guard = crate::ZYNKSYNC_SERVICE.lock().await;
        match guard.as_ref() {
            Some(service) => service.get_db_pool(),
            None => {
                println!("[ZynkLink] Service not started, falling back to create_pool");
                drop(guard);
                crate::db::create_pool().await.map_err(|e| format!("Failed to connect to database: {}", e))?
            }
        }
    };

    let device_name = hostname::get()
        .map_err(|e| format!("Failed to get hostname: {}", e))?
        .to_string_lossy()
        .to_string();

    sqlx::query(
        "INSERT INTO zynk_devices (device_id, device_name, owner_user_id, is_paired, port, created_at, last_seen_at)
         VALUES (?, ?, ?, true, 57963, datetime('now'), datetime('now'))
         ON CONFLICT (device_id) DO UPDATE
         SET owner_user_id = ?, last_seen_at = datetime('now')"
    )
    .bind(&device_id)
    .bind(&device_name)
    .bind(&user_id)
    .bind(&user_id)
    .execute(&pool)
    .await
    .map_err(|e| { println!("[ZynkLink] Failed to ensure device entry: {}", e); format!("Failed to ensure device entry: {}", e) })?;

    println!("[ZynkLink] Generating code...");
    let code = zynklink::generate_zynklink_code(&pool, &user_id, &device_id).await
        .map_err(|e| { println!("[ZynkLink] Code generation failed: {}", e); e })?;

    println!("[ZynkLink] Code generated successfully: {}", code);

    Ok(serde_json::json!({
        "success": true,
        "code": code
    }))
}

/// Link with remote device using ZynkLink code
#[tauri::command]
pub async fn link_with_zynklink_code(app: tauri::AppHandle, code: String, device_ip: String) -> Result<serde_json::Value, String> {
    println!("[ZynkLink] Device B: Linking with code {} to remote device {}", code, device_ip);

    let user_id = crate::user_identity::get_user_id()?;
    let device_id = crate::user_identity::get_device_id()?;

    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("Failed to build link client: {}", e))?;
    let verify_url = format!("https://{}:57963/api/zynklink/verify-code", device_ip);

    println!("[ZynkLink] Device B: Verifying code with remote device...");

    let response = client
        .post(&verify_url)
        .json(&serde_json::json!({ "code": code }))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("Failed to reach remote device: {}", e))?;

    if !response.status().is_success() {
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("Code verification failed: {}", error_text));
    }

    #[derive(serde::Deserialize)]
    struct VerifyResponse {
        user_id: String,
        device_id: String,
    }

    let verify_data: VerifyResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    println!("[ZynkLink] Device B: Code verified! Remote user_id: {}...", &verify_data.user_id[..8]);

    let accept_url = format!("https://{}:57963/api/zynklink/accept-code", device_ip);

    let local_ip = {
        use std::net::UdpSocket;
        match UdpSocket::bind("0.0.0.0:0") {
            Ok(socket) => {
                match socket.connect("8.8.8.8:80") {
                    Ok(_) => {
                        socket.local_addr()
                            .map(|addr| addr.ip().to_string())
                            .unwrap_or_else(|_| "unknown".to_string())
                    }
                    Err(_) => "unknown".to_string()
                }
            }
            Err(_) => "unknown".to_string()
        }
    };

    println!("[ZynkLink] Device B: Requesting pairing acceptance (our IP: {})...", local_ip);

    let accept_response = client
        .post(&accept_url)
        .json(&serde_json::json!({
            "code": code,
            "acceptor_user_id": user_id,
            "acceptor_device_id": device_id,
            "acceptor_device_ip": local_ip
        }))
        .send()
        .await
        .map_err(|e| format!("Failed to create pairing: {}", e))?;

    if !accept_response.status().is_success() {
        let error_text = accept_response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("Pairing acceptance failed: {}", error_text));
    }

    let accept_data: serde_json::Value = accept_response
        .json()
        .await
        .map_err(|e| format!("Failed to parse accept response: {}", e))?;

    let creator_device_ip = accept_data.get("creator_device_ip")
        .and_then(|ip| ip.as_str())
        .unwrap_or(&device_ip);

    println!("[ZynkLink] Device B: Pairing created on remote device! Creator IP: {}", creator_device_ip);

    let pool = {
        let guard = crate::ZYNKSYNC_SERVICE.lock().await;
        match guard.as_ref() {
            Some(service) => service.get_db_pool(),
            None => {
                drop(guard);
                crate::db::create_pool().await.map_err(|e| format!("Failed to connect to database: {}", e))?
            }
        }
    };

    println!("[ZynkLink] Device B: Ensuring local device is registered with our IP {}...", local_ip);
    let device_name = hostname::get()
        .map_err(|e| format!("Failed to get hostname: {}", e))?
        .to_string_lossy()
        .to_string();

    sqlx::query(
        "INSERT INTO zynk_devices (device_id, device_name, device_ip, owner_user_id, is_paired, port, created_at, last_seen_at)
         VALUES (?, ?, ?, ?, true, 57963, datetime('now'), datetime('now'))
         ON CONFLICT (device_id) DO UPDATE
         SET device_ip = ?, owner_user_id = ?, last_seen_at = datetime('now')"
    )
    .bind(&device_id)
    .bind(&device_name)
    .bind(&local_ip)
    .bind(&user_id)
    .bind(&local_ip)
    .bind(&user_id)
    .execute(&pool)
    .await
    .map_err(|e| format!("Failed to ensure local device entry: {}", e))?;

    println!("[ZynkLink] Device B: Ensuring remote device is registered with IP {}...", creator_device_ip);
    sqlx::query(
        "INSERT INTO zynk_devices (device_id, device_name, device_ip, owner_user_id, is_paired, port, created_at, last_seen_at)
         VALUES (?, ?, ?, ?, true, 57963, datetime('now'), datetime('now'))
         ON CONFLICT (device_id) DO UPDATE
         SET device_ip = ?, owner_user_id = ?, last_seen_at = datetime('now')"
    )
    .bind(&verify_data.device_id)
    .bind(&format!("Remote Device {}", &verify_data.device_id[..8]))
    .bind(creator_device_ip)
    .bind(&verify_data.user_id)
    .bind(creator_device_ip)
    .bind(&verify_data.user_id)
    .execute(&pool)
    .await
    .map_err(|e| format!("Failed to ensure remote device entry: {}", e))?;

    println!("[ZynkLink] Device B: Both devices registered successfully");

    let (user1_id, user2_id, device1_id, device2_id) = if user_id.as_str() < verify_data.user_id.as_str() {
        (user_id.clone(), verify_data.user_id.clone(), device_id.clone(), verify_data.device_id.clone())
    } else {
        (verify_data.user_id.clone(), user_id.clone(), verify_data.device_id.clone(), device_id.clone())
    };

    sqlx::query(
        "INSERT INTO zynklink_pairings (user1_id, user2_id, device1_id, device2_id, is_active)
         VALUES (?, ?, ?, ?, 1)
         ON CONFLICT (user1_id, user2_id) DO UPDATE SET is_active = 1, linked_at = datetime('now')"
    )
    .bind(&user1_id)
    .bind(&user2_id)
    .bind(&device1_id)
    .bind(&device2_id)
    .execute(&pool)
    .await
    .map_err(|e| format!("Failed to create local pairing: {}", e))?;

    println!("[ZynkLink] Device B: Local pairing record created!");

    println!("[ZynkLink] Device B: Emitting zynklink-pairing-updated event");
    let _ = app.emit("zynklink-pairing-updated", serde_json::json!({
        "remote_user_id": verify_data.user_id,
        "remote_device_id": verify_data.device_id
    }));

    Ok(serde_json::json!({
        "success": true,
        "message": format!("Successfully linked for file sharing with user {}", verify_data.user_id),
        "remote_user_id": verify_data.user_id,
        "remote_device_id": verify_data.device_id
    }))
}

/// List all ZynkLink pairings
#[tauri::command]
pub async fn list_zynklink_pairings() -> Result<serde_json::Value, String> {
    let user_id = crate::user_identity::get_user_id()?;
    let device_id = crate::user_identity::get_device_id()?;

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    zynklink::list_zynklink_pairings(&pool, &user_id, &device_id).await
}

/// Unlink from a device — FULL teardown of BOTH file-sharing (ZynkLink) AND
/// conversation sync (ZynkSync), including the pinned TLS cert, on BOTH devices.
///
/// Link and sync share a single trust relationship (the peer's pinned cert in
/// `zynk_devices.tls_cert_der`), so an "unlink" must tear everything down and force
/// a full re-pair. This previously removed only the ZynkLink file-sharing pairing,
/// leaving the sync pairing + pinned cert intact — so devices kept syncing after an
/// unlink and fell into a 401 loop whenever only one side tore down. We now delegate
/// to `remove_device()`, which deletes the peer's cert + all pairing rows locally and
/// fires `notify-unsynced` so the peer performs the same full removal.
#[tauri::command]
pub async fn revoke_zynklink_pairing(app: tauri::AppHandle, linked_user_id: String) -> Result<serde_json::Value, String> {
    let user_id = crate::user_identity::get_user_id()?;
    let local_device_id = crate::user_identity::get_device_id()?;

    // Grab the running sync service — it owns the peer-notification path and the
    // cert-pinned HTTP client. Unlink is a UI action, so the service is expected up.
    let service = {
        let guard = crate::ZYNKSYNC_SERVICE.lock().await;
        match guard.as_ref() {
            Some(s) => std::sync::Arc::clone(s),
            None => return Err("Sync service not running — cannot unlink cleanly".to_string()),
        }
    };
    let pool = service.get_db_pool();

    // Resolve the peer device_id from the pairing before we tear it down.
    let peer = sqlx::query_as::<_, (String, String)>(
        "SELECT device1_id, device2_id FROM zynklink_pairings
         WHERE ((user1_id = ? AND user2_id = ?) OR (user1_id = ? AND user2_id = ?))"
    )
    .bind(&user_id).bind(&linked_user_id)
    .bind(&linked_user_id).bind(&user_id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| format!("Failed to look up pairing: {}", e))?;

    let peer_device_id = match peer {
        Some((d1, d2)) => if d1 == local_device_id { d2 } else { d1 },
        None => return Err("No active pairing found with this user".to_string()),
    };

    // Full teardown locally + notify the peer to do the same (fire-and-forget).
    service.remove_device(&peer_device_id).await?;

    // A trusted peer is gone — rebuild the cert-pinned HTTP client so it stops
    // trusting the now-removed device.
    if let Err(e) = service.rebuild_http_client().await {
        eprintln!("[ZynkLink] Warning: failed to rebuild HTTP client after unlink: {}", e);
    }

    // Refresh both panels instantly (they also poll, but this avoids the lag).
    let _ = app.emit("zynklink-pairing-updated", serde_json::json!({ "unlinked": true }));
    let _ = app.emit("zynksync-device-removed", serde_json::json!({ "device_id": peer_device_id }));

    Ok(serde_json::json!({
        "success": true,
        "message": "Unlinked and unsynced. Both devices must re-pair to reconnect."
    }))
}

/// Pause or resume a ZynkLink pairing (session-only; clears on restart)
#[tauri::command]
pub async fn toggle_zynklink_pause(linked_device_id: String, paused: bool) -> Result<serde_json::Value, String> {
    zynklink::set_link_paused(&linked_device_id, paused).await;
    Ok(serde_json::json!({ "success": true, "paused": paused }))
}

/// Share a directory
#[tauri::command]
pub async fn share_directory(local_path: String, share_name: String, is_readable: bool, is_writable: bool) -> Result<serde_json::Value, String> {
    let user_id = crate::user_identity::get_user_id()?;
    let device_id = crate::user_identity::get_device_id()?;

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let device_name = hostname::get()
        .map_err(|e| format!("Failed to get hostname: {}", e))?
        .to_string_lossy()
        .to_string();

    sqlx::query(
        "INSERT INTO zynk_devices (device_id, device_name, owner_user_id, is_paired, port, created_at, last_seen_at)
         VALUES (?, ?, ?, true, 57963, datetime('now'), datetime('now'))
         ON CONFLICT (device_id) DO UPDATE
         SET owner_user_id = ?, last_seen_at = datetime('now')"
    )
    .bind(&device_id)
    .bind(&device_name)
    .bind(&user_id)
    .execute(&pool)
    .await
    .map_err(|e| format!("Failed to ensure device entry: {}", e))?;

    let request = zynklink::ShareDirectoryRequest {
        local_path,
        share_name,
        is_readable,
        is_writable,
    };

    let response = zynklink::share_directory(&pool, &device_id, request).await?;
    Ok(serde_json::to_value(response).map_err(|e| format!("Serialization error: {}", e))?)
}

/// Unshare a directory
#[tauri::command]
pub async fn unshare_directory(share_id: i32) -> Result<serde_json::Value, String> {
    let device_id = crate::user_identity::get_device_id()?;

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    zynklink::unshare_directory(&pool, &device_id, share_id).await
}

/// List my shared directories
#[tauri::command]
pub async fn list_my_shared_directories() -> Result<serde_json::Value, String> {
    let device_id = crate::user_identity::get_device_id()?;

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let response = zynklink::list_my_shared_directories(&pool, &device_id).await?;
    Ok(serde_json::to_value(response).map_err(|e| format!("Serialization error: {}", e))?)
}

/// List remote shared directories from paired devices
#[tauri::command]
pub async fn list_remote_directories() -> Result<serde_json::Value, String> {
    let user_id = crate::user_identity::get_user_id()?;
    let device_id = crate::user_identity::get_device_id()?;

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let paired_devices = sqlx::query_as::<_, (String, String, Option<String>)>(
        "SELECT
            CASE WHEN device1_id = ? THEN device2_id ELSE device1_id END as device_id,
            CASE WHEN user1_id = ? THEN user2_id ELSE user1_id END as user_id,
            zd.device_ip
         FROM zynklink_pairings zp
         LEFT JOIN zynk_devices zd ON (CASE WHEN device1_id = ? THEN device2_id ELSE device1_id END) = zd.device_id
         WHERE (user1_id = ? OR user2_id = ?) AND is_active = 1"
    )
    .bind(&device_id)
    .bind(&user_id)
    .bind(&device_id)
    .bind(&user_id)
    .bind(&user_id)
    .fetch_all(&pool)
    .await
    .map_err(|e| format!("Failed to get paired devices: {}", e))?;

    let mut all_directories = Vec::new();
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    for (remote_device_id, _remote_user_id, device_ip_opt) in paired_devices {
        if let Some(device_ip) = device_ip_opt {
            let url = format!("https://{}:57963/api/zynklink/directories", device_ip);

            match client
                .post(&url)
                .json(&serde_json::json!({
                    "device_id": remote_device_id,
                    "requester_user_id": user_id
                }))
                .send()
                .await
            {
                Ok(response) => {
                    if response.status().is_success() {
                        let _ = sqlx::query(
                            "UPDATE zynk_devices SET last_seen_at = datetime('now') WHERE device_id = ?"
                        )
                        .bind(&remote_device_id)
                        .execute(&pool)
                        .await;

                        match response.json::<serde_json::Value>().await {
                            Ok(data) => {
                                if let Some(dirs) = data.get("shared_directories").and_then(|d| d.as_array()) {
                                    for dir in dirs {
                                        all_directories.push(dir.clone());
                                    }
                                }
                            }
                            Err(e) => {
                                println!("[ZynkLink] Failed to parse response from {}: {}", device_ip, e);
                            }
                        }
                    } else {
                        println!("[ZynkLink] HTTP error from {}: {}", device_ip, response.status());
                    }
                }
                Err(_e) => {}
            }
        }
    }

    Ok(serde_json::json!({
        "shared_directories": all_directories
    }))
}

/// Scan a shared directory and index files
#[tauri::command]
pub async fn scan_shared_directory(share_id: i32, max_files: Option<usize>) -> Result<serde_json::Value, String> {
    let device_id = crate::user_identity::get_device_id()?;

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    zynklink::scan_directory(&pool, &device_id, share_id, max_files).await
}

/// List files in a shared directory
#[tauri::command]
pub async fn list_shared_files(share_id: i32, device_id: String) -> Result<serde_json::Value, String> {
    let local_device_id = crate::user_identity::get_device_id()?;
    let local_user_id = crate::user_identity::get_user_id()?;

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    if device_id == local_device_id {
        println!("[ZynkLink] Listing files for local share {} (device {}...)", share_id, &device_id[..8]);
        let response = zynklink::list_files(&pool, share_id).await?;
        return Ok(serde_json::to_value(response).map_err(|e| format!("Serialization error: {}", e))?);
    }

    let device_ip_opt = sqlx::query_as::<_, (Option<String>,)>(
        "SELECT device_ip FROM zynk_devices WHERE device_id = ?"
    )
    .bind(&device_id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| format!("Failed to get device IP: {}", e))?
    .and_then(|r| r.0);

    if let Some(device_ip) = device_ip_opt {
        println!("[ZynkLink] Fetching files for remote share {} from device {}... at {}", share_id, &device_id[..8], device_ip);

        let client = reqwest::Client::builder()
            .danger_accept_invalid_certs(true)
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        let url = format!("https://{}:57963/api/zynklink/files", device_ip);

        match client
            .post(&url)
            .json(&serde_json::json!({
                "share_id": share_id,
                "requester_user_id": local_user_id
            }))
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    match response.json::<serde_json::Value>().await {
                        Ok(data) => {
                            println!("[ZynkLink] Fetched {} files from remote share",
                                data.get("files").and_then(|f| f.as_array()).map(|a| a.len()).unwrap_or(0));
                            return Ok(data);
                        }
                        Err(e) => {
                            return Err(format!("Failed to parse response: {}", e));
                        }
                    }
                } else {
                    return Err(format!("HTTP error: {}", response.status()));
                }
            }
            Err(e) => {
                return Err(format!("Failed to connect to remote device: {}", e));
            }
        }
    } else {
        return Err("No IP address available for remote device".to_string());
    }
}

/// Get file path for download
#[tauri::command]
pub async fn get_shared_file_path(share_id: i32, relative_path: String) -> Result<String, String> {
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let path = zynklink::get_file_path(&pool, share_id, &relative_path).await?;
    Ok(path.to_string_lossy().to_string())
}

/// Cancel an in-flight ZynkLink download
#[tauri::command]
pub fn cancel_zynklink_download(relative_path: String) -> Result<(), String> {
    let registry = crate::DOWNLOAD_CANCELS.lock()
        .map_err(|e| format!("Cancel registry lock poisoned: {}", e))?;
    if let Some(flag) = registry.get(&relative_path) {
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
        println!("[ZynkLink] Cancel requested for: {}", relative_path);
        Ok(())
    } else {
        Err(format!("No active download found for: {}", relative_path))
    }
}

/// Download a file to the Knowledge Base folder
#[tauri::command]
pub async fn download_to_knowledge_base(
    app: tauri::AppHandle,
    share_id: i32,
    relative_path: String,
    device_id: String,
    user_id: String
) -> Result<String, String> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    println!("[KB Download] Starting download - share_id: {}, path: {}, device: {}...",
        share_id, relative_path, &device_id[..8]);

    let local_device_id = crate::user_identity::get_device_id()?;
    let local_user_id = crate::user_identity::get_user_id()?;
    println!("[KB Download] Local device: {}...", &local_device_id[..8]);

    let kb_path = kb_rag::get_kb_folder_path(&user_id)?;

    println!("[KB Download] Target directory: {}", kb_path.display());

    let filename = std::path::Path::new(&relative_path)
        .file_name()
        .ok_or("Invalid filename")?
        .to_string_lossy()
        .to_string();

    let destination_path = kb_path.join(&filename);
    let destination_path_str = destination_path.to_string_lossy().to_string();
    println!("[KB Download] Destination: {}", destination_path.display());

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    if device_id == local_device_id {
        println!("[KB Download] Local file - copying from local share");
        let source_path = zynklink::get_file_path(&pool, share_id, &relative_path).await?;
        println!("[KB Download] Source path: {}", source_path.display());

        tokio::fs::copy(&source_path, &destination_path)
            .await
            .map_err(|e| format!("Failed to copy file: {}", e))?;

        println!("[KB Download] ✓ Local file copied successfully");
    } else {
        println!("[KB Download] Remote file - downloading from device {}...", &device_id[..8]);

        let device_ip_opt = sqlx::query_as::<_, (Option<String>,)>(
            "SELECT device_ip FROM zynk_devices WHERE device_id = ?"
        )
        .bind(&device_id)
        .fetch_optional(&pool)
        .await
        .map_err(|e| format!("Failed to get device IP: {}", e))?
        .and_then(|r| r.0);

        let device_ip = device_ip_opt.ok_or("No IP address available for remote device")?;
        println!("[KB Download] Downloading from {}", device_ip);

        let client = reqwest::Client::builder()
            .danger_accept_invalid_certs(true)
            .build()
            .map_err(|e| format!("Failed to build download client: {}", e))?;
        let url = format!("https://{}:57963/api/zynklink/download", device_ip);

        let response = client
            .post(&url)
            .json(&serde_json::json!({
                "share_id": share_id,
                "relative_path": relative_path,
                "requester_user_id": local_user_id
            }))
            .send()
            .await
            .map_err(|e| format!("Failed to connect to remote device: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("HTTP error: {}", response.status()));
        }

        let total_bytes = response.content_length();

        let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let mut registry = crate::DOWNLOAD_CANCELS.lock()
                .map_err(|e| format!("Cancel registry lock poisoned: {}", e))?;
            registry.insert(relative_path.clone(), cancel_flag.clone());
        }
        struct CancelGuard(String);
        impl Drop for CancelGuard {
            fn drop(&mut self) {
                if let Ok(mut r) = crate::DOWNLOAD_CANCELS.lock() {
                    r.remove(&self.0);
                }
            }
        }
        let _cancel_guard = CancelGuard(relative_path.clone());

        let temp_path = format!("{}.part", destination_path_str);
        let mut file = tokio::fs::File::create(&temp_path).await
            .map_err(|e| format!("Failed to create destination file: {}", e))?;

        let _ = app.emit("zynklink:download:start", serde_json::json!({
            "share_id": share_id,
            "relative_path": &relative_path,
            "destination": &destination_path_str,
            "total_bytes": total_bytes,
        }));

        let mut bytes_written: u64 = 0;
        let mut last_event_at: u64 = 0;
        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            if cancel_flag.load(std::sync::atomic::Ordering::Relaxed) {
                drop(file);
                let _ = tokio::fs::remove_file(&temp_path).await;
                let _ = app.emit("zynklink:download:cancelled", serde_json::json!({
                    "share_id": share_id,
                    "relative_path": &relative_path,
                    "bytes_written": bytes_written,
                }));
                return Err("Cancelled by user".to_string());
            }

            let chunk = chunk_result
                .map_err(|e| format!("Network read error mid-transfer: {}", e))?;
            file.write_all(&chunk).await
                .map_err(|e| format!("Failed to write chunk to disk: {}", e))?;
            bytes_written += chunk.len() as u64;

            if bytes_written - last_event_at >= 262_144 {
                let _ = app.emit("zynklink:download:progress", serde_json::json!({
                    "share_id": share_id,
                    "relative_path": &relative_path,
                    "bytes_written": bytes_written,
                    "total_bytes": total_bytes,
                }));
                last_event_at = bytes_written;
            }
        }

        file.flush().await
            .map_err(|e| format!("Failed to flush destination file: {}", e))?;
        drop(file);

        tokio::fs::rename(&temp_path, &destination_path).await
            .map_err(|e| format!("Failed to finalize download (rename .part): {}", e))?;

        let _ = app.emit("zynklink:download:complete", serde_json::json!({
            "share_id": share_id,
            "relative_path": &relative_path,
            "destination": &destination_path_str,
            "total_bytes": bytes_written,
        }));

        println!("[KB Download] ✓ Remote file streamed to disk ({} bytes)", bytes_written);
    }

    println!("[KB Download] Indexing file into knowledge base...");

    let file_content = tokio::fs::read_to_string(&destination_path)
        .await
        .map_err(|e| format!("Failed to read file for indexing: {}", e))?;

    match kb_rag::index_text_as_document(&pool, &user_id, &filename, &file_content).await {
        Ok(doc_id) => {
            println!("[KB Download] ✓ File indexed successfully (doc_id: {})", doc_id);
        }
        Err(e) => {
            println!("[KB Download] ⚠️  Warning: File saved but indexing failed: {}", e);
        }
    }

    println!("[KB Download] ✓ Complete: {}", destination_path.display());
    Ok(destination_path.to_string_lossy().to_string())
}

/// Download a file to a custom location
#[tauri::command]
pub async fn download_to_custom_location(
    app: tauri::AppHandle,
    share_id: i32,
    relative_path: String,
    device_id: String,
    destination_path: String
) -> Result<String, String> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    let local_device_id = crate::user_identity::get_device_id()?;
    let local_user_id = crate::user_identity::get_user_id()?;

    if device_id == local_device_id {
        let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
            .await
            .map_err(|e| format!("Failed to connect to database: {}", e))?;

        let source_path = zynklink::get_file_path(&pool, share_id, &relative_path).await?;
        tokio::fs::copy(&source_path, &destination_path)
            .await
            .map_err(|e| format!("Failed to copy file: {}", e))?;
    } else {
        let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
            .await
            .map_err(|e| format!("Failed to connect to database: {}", e))?;

        let device_ip_opt = sqlx::query_as::<_, (Option<String>,)>(
            "SELECT device_ip FROM zynk_devices WHERE device_id = ?"
        )
        .bind(&device_id)
        .fetch_optional(&pool)
        .await
        .map_err(|e| format!("Failed to get device IP: {}", e))?
        .and_then(|r| r.0);

        let device_ip = device_ip_opt.ok_or("No IP address available for remote device")?;

        let client = reqwest::Client::builder()
            .danger_accept_invalid_certs(true)
            .build()
            .map_err(|e| format!("Failed to build download client: {}", e))?;
        let url = format!("https://{}:57963/api/zynklink/download", device_ip);

        let response = client
            .post(&url)
            .json(&serde_json::json!({
                "share_id": share_id,
                "relative_path": relative_path,
                "requester_user_id": local_user_id
            }))
            .send()
            .await
            .map_err(|e| format!("Failed to connect to remote device: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("HTTP error: {}", response.status()));
        }

        let total_bytes = response.content_length();

        let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let mut registry = crate::DOWNLOAD_CANCELS.lock()
                .map_err(|e| format!("Cancel registry lock poisoned: {}", e))?;
            registry.insert(relative_path.clone(), cancel_flag.clone());
        }
        struct CancelGuard(String);
        impl Drop for CancelGuard {
            fn drop(&mut self) {
                if let Ok(mut r) = crate::DOWNLOAD_CANCELS.lock() {
                    r.remove(&self.0);
                }
            }
        }
        let _cancel_guard = CancelGuard(relative_path.clone());

        let temp_path = format!("{}.part", destination_path);
        let mut file = tokio::fs::File::create(&temp_path).await
            .map_err(|e| format!("Failed to create destination file: {}", e))?;

        let _ = app.emit("zynklink:download:start", serde_json::json!({
            "share_id": share_id,
            "relative_path": &relative_path,
            "destination": &destination_path,
            "total_bytes": total_bytes,
        }));

        let mut bytes_written: u64 = 0;
        let mut last_event_at: u64 = 0;
        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            if cancel_flag.load(std::sync::atomic::Ordering::Relaxed) {
                drop(file);
                let _ = tokio::fs::remove_file(&temp_path).await;
                let _ = app.emit("zynklink:download:cancelled", serde_json::json!({
                    "share_id": share_id,
                    "relative_path": &relative_path,
                    "bytes_written": bytes_written,
                }));
                return Err("Cancelled by user".to_string());
            }

            let chunk = chunk_result
                .map_err(|e| format!("Network read error mid-transfer: {}", e))?;
            file.write_all(&chunk).await
                .map_err(|e| format!("Failed to write chunk to disk: {}", e))?;
            bytes_written += chunk.len() as u64;

            if bytes_written - last_event_at >= 262_144 {
                let _ = app.emit("zynklink:download:progress", serde_json::json!({
                    "share_id": share_id,
                    "relative_path": &relative_path,
                    "bytes_written": bytes_written,
                    "total_bytes": total_bytes,
                }));
                last_event_at = bytes_written;
            }
        }

        file.flush().await
            .map_err(|e| format!("Failed to flush destination file: {}", e))?;
        drop(file);

        tokio::fs::rename(&temp_path, &destination_path).await
            .map_err(|e| format!("Failed to finalize download (rename .part): {}", e))?;

        let _ = app.emit("zynklink:download:complete", serde_json::json!({
            "share_id": share_id,
            "relative_path": &relative_path,
            "destination": &destination_path,
            "total_bytes": bytes_written,
        }));
    }

    Ok(destination_path)
}

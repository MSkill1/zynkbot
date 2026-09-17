#![allow(dead_code)]

/// ZynkLink - Device-to-Device File Sharing
///
/// Pure Rust implementation of file sharing between paired Zynkbot devices
///
/// Features:
/// - Share local directories with other devices
/// - Browse and download files from paired devices
/// - Directory scanning and indexing
/// - HTTP-based file serving
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc};
use tokio::fs;
use once_cell::sync::Lazy;
use std::collections::HashSet;
use tokio::sync::Mutex;

/// In-memory set of paused linked device IDs (session-only, clears on restart).
static PAUSED_LINKS: Lazy<Mutex<HashSet<String>>> =
    Lazy::new(|| Mutex::new(HashSet::new()));

pub async fn set_link_paused(device_id: &str, paused: bool) {
    let mut set = PAUSED_LINKS.lock().await;
    if paused {
        set.insert(device_id.to_string());
    } else {
        set.remove(device_id);
    }
}

pub async fn is_link_paused(device_id: &str) -> bool {
    PAUSED_LINKS.lock().await.contains(device_id)
}

/// Represents a shared directory
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SharedDirectory {
    pub id: i32,
    pub device_id: String,
    pub local_path: String,
    pub share_name: String,
    pub is_readable: bool,
    pub is_writable: bool,
    pub created_at: DateTime<Utc>,
}

/// Represents a file in a shared directory
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SharedFile {
    pub id: i32,
    pub share_id: i32,
    pub relative_path: String,
    pub file_size: i64,
    pub last_modified: DateTime<Utc>,
    pub indexed_at: DateTime<Utc>,
}

/// Request to share a new directory
#[derive(Debug, Deserialize)]
pub struct ShareDirectoryRequest {
    pub local_path: String,
    pub share_name: String,
    pub is_readable: bool,
    pub is_writable: bool,
}

/// Response when sharing a directory
#[derive(Debug, Serialize)]
pub struct ShareDirectoryResponse {
    pub success: bool,
    pub share_id: Option<i32>,
    pub error: Option<String>,
}

/// Request to scan a directory for files
#[derive(Debug, Deserialize)]
pub struct ScanDirectoryRequest {
    pub max_files: Option<usize>,
}

/// Response with list of files
#[derive(Debug, Serialize)]
pub struct ListFilesResponse {
    pub files: Vec<SharedFile>,
}

/// Response with list of shared directories
#[derive(Debug, Serialize)]
pub struct ListDirectoriesResponse {
    pub shared_directories: Vec<SharedDirectory>,
}

/// Share a local directory
pub async fn share_directory(
    pool: &SqlitePool,
    device_id: &str,
    request: ShareDirectoryRequest,
) -> Result<ShareDirectoryResponse, String> {
    // Verify directory exists and is accessible
    let path = Path::new(&request.local_path);
    if !path.exists() {
        return Ok(ShareDirectoryResponse {
            success: false,
            share_id: None,
            error: Some("Directory does not exist".to_string()),
        });
    }

    if !path.is_dir() {
        return Ok(ShareDirectoryResponse {
            success: false,
            share_id: None,
            error: Some("Path is not a directory".to_string()),
        });
    }

    // Insert into database or update if already exists
    let result = sqlx::query_as::<_, (i32,)>(
        "INSERT INTO zynk_linked_directories (device_id, local_path, share_name, is_readable, is_writable, created_at)
         VALUES (?, ?, ?, ?, ?, datetime('now'))
         ON CONFLICT (device_id, local_path)
         DO UPDATE SET
             share_name = EXCLUDED.share_name,
             is_readable = EXCLUDED.is_readable,
             is_writable = EXCLUDED.is_writable,
             updated_at = datetime('now')
         RETURNING id"
    )
    .bind(device_id)
    .bind(&request.local_path)
    .bind(&request.share_name)
    .bind(request.is_readable)
    .bind(request.is_writable)
    .fetch_one(pool)
    .await
    .map_err(|e| format!("Failed to share directory: {}", e))?;

    Ok(ShareDirectoryResponse {
        success: true,
        share_id: Some(result.0),
        error: None,
    })
}

/// Unshare a directory
pub async fn unshare_directory(
    pool: &SqlitePool,
    device_id: &str,
    share_id: i32,
) -> Result<serde_json::Value, String> {
    // First delete all file entries
    sqlx::query("DELETE FROM zynk_file_manifest WHERE shared_directory_id = ?")
        .bind(share_id)
        .execute(pool)
        .await
        .map_err(|e| format!("Failed to delete file entries: {}", e))?;

    // Delete the share (only if owned by this device)
    let result = sqlx::query(
        "DELETE FROM zynk_linked_directories WHERE id = ? AND device_id = ?"
    )
    .bind(share_id)
    .bind(device_id)
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to unshare directory: {}", e))?;

    if result.rows_affected() == 0 {
        return Err("Share not found or not owned by this device".to_string());
    }

    Ok(serde_json::json!({
        "success": true
    }))
}

/// List directories shared by this device
pub async fn list_my_shared_directories(
    pool: &SqlitePool,
    device_id: &str,
) -> Result<ListDirectoriesResponse, String> {
    let directories = sqlx::query_as::<_, SharedDirectory>(
        "SELECT id, device_id, local_path, share_name, is_readable, is_writable, created_at
         FROM zynk_linked_directories
         WHERE device_id = ?
         ORDER BY created_at DESC"
    )
    .bind(device_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to list shared directories: {}", e))?;

    Ok(ListDirectoriesResponse {
        shared_directories: directories,
    })
}

/// List directories shared by ZynkLink paired users (remote directories)
pub async fn list_remote_directories(
    pool: &SqlitePool,
    current_user_id: &str,
) -> Result<ListDirectoriesResponse, String> {
    // Get all device IDs from ZynkLink paired users
    let paired_device_ids = sqlx::query_as::<_, (String,)>(
        "SELECT
            CASE WHEN user1_id = ? THEN device2_id ELSE device1_id END as device_id
         FROM zynklink_pairings
         WHERE (user1_id = ? OR user2_id = ?) AND is_active = 1"
    )
    .bind(current_user_id)
    .bind(current_user_id)
    .bind(current_user_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to get ZynkLink paired devices: {}", e))?;

    if paired_device_ids.is_empty() {
        return Ok(ListDirectoriesResponse {
            shared_directories: vec![],
        });
    }

    // Get shared directories from ZynkLink paired devices
    let device_ids: Vec<String> = paired_device_ids.into_iter().map(|r| r.0).collect();

    let directories = if device_ids.is_empty() {
        vec![]
    } else {
        let in_clause = device_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        let sql = format!(
            "SELECT id, device_id, local_path, share_name, is_readable, is_writable, created_at
             FROM zynk_linked_directories
             WHERE device_id IN ({}) ORDER BY created_at DESC",
            in_clause
        );
        let mut q = sqlx::query_as::<_, SharedDirectory>(&sql);
        for id in &device_ids { q = q.bind(id); }
        q.fetch_all(pool).await.map_err(|e| format!("Failed to list remote directories: {}", e))?
    };

    Ok(ListDirectoriesResponse {
        shared_directories: directories,
    })
}

/// Scan a directory and index all files.
/// Collects all file records into memory first, then atomically clears and
/// replaces the manifest — so a failed scan never leaves the manifest empty.
pub async fn scan_directory(
    pool: &SqlitePool,
    device_id: &str,
    share_id: i32,
    max_files: Option<usize>,
) -> Result<serde_json::Value, String> {
    // Get the share
    let share = sqlx::query_as::<_, SharedDirectory>(
        "SELECT id, device_id, local_path, share_name, is_readable, is_writable, created_at
         FROM zynk_linked_directories
         WHERE id = ? AND device_id = ?"
    )
    .bind(share_id)
    .bind(device_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("Failed to get share: {}", e))?;

    let share = share.ok_or_else(|| "Share not found or not owned by this device".to_string())?;

    let base_path = PathBuf::from(&share.local_path);
    let max = max_files.unwrap_or(1000);

    // Collect all files into memory first; if this fails, the existing manifest is untouched.
    let mut collected: Vec<(String, i64, DateTime<Utc>)> = Vec::new();
    collect_files_recursive(&base_path, &base_path, &mut collected, max)
        .await
        .map_err(|e| format!("Failed to scan directory: {}", e))?;

    let files_indexed = collected.len();

    // Scan succeeded — now atomically replace the manifest.
    sqlx::query("DELETE FROM zynk_file_manifest WHERE shared_directory_id = ?")
        .bind(share_id)
        .execute(pool)
        .await
        .map_err(|e| format!("Failed to clear file entries: {}", e))?;

    for (relative_path, file_size, last_modified) in collected {
        sqlx::query(
            "INSERT INTO zynk_file_manifest (shared_directory_id, relative_path, file_size, last_modified, indexed_at)
             VALUES (?, ?, ?, ?, datetime('now'))"
        )
        .bind(share_id)
        .bind(&relative_path)
        .bind(file_size)
        .bind(last_modified)
        .execute(pool)
        .await
        .map_err(|e| format!("Failed to insert file record: {}", e))?;
    }

    Ok(serde_json::json!({
        "success": true,
        "files_indexed": files_indexed
    }))
}

/// Recursively collect file metadata into `out`; returns Err if the root directory
/// cannot be read (caller decides whether to abort or ignore).
async fn collect_files_recursive(
    base_path: &Path,
    current_path: &Path,
    out: &mut Vec<(String, i64, DateTime<Utc>)>,
    max_files: usize,
) -> Result<(), String> {
    if out.len() >= max_files {
        return Ok(());
    }

    let mut entries = fs::read_dir(current_path)
        .await
        .map_err(|e| format!("Failed to read directory: {}", e))?;

    while let Some(entry) = entries.next_entry().await.map_err(|e| e.to_string())? {
        if out.len() >= max_files {
            break;
        }

        let path = entry.path();
        let metadata = match fs::metadata(&path).await {
            Ok(m) => m,
            Err(_) => continue,
        };

        if metadata.is_file() {
            let relative_path = path.strip_prefix(base_path)
                .map_err(|e| e.to_string())?
                .to_string_lossy()
                .to_string();

            let file_size = metadata.len() as i64;
            let last_modified = metadata.modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| DateTime::<Utc>::from_timestamp(d.as_secs() as i64, 0).unwrap_or_default())
                .unwrap_or_else(Utc::now);

            out.push((relative_path, file_size, last_modified));
        } else if metadata.is_dir() {
            Box::pin(collect_files_recursive(base_path, &path, out, max_files)).await?;
        }
    }

    Ok(())
}

/// List files in a shared directory
pub async fn list_files(
    pool: &SqlitePool,
    share_id: i32,
) -> Result<ListFilesResponse, String> {
    let files = sqlx::query_as::<_, SharedFile>(
        "SELECT id, shared_directory_id as share_id, relative_path, file_size, last_modified, indexed_at
         FROM zynk_file_manifest
         WHERE shared_directory_id = ?
         ORDER BY relative_path"
    )
    .bind(share_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to list files: {}", e))?;

    Ok(ListFilesResponse { files })
}

/// Get the full filesystem path for a file in a share
pub async fn get_file_path(
    pool: &SqlitePool,
    share_id: i32,
    relative_path: &str,
) -> Result<PathBuf, String> {
    // Get the share
    let share = sqlx::query_as::<_, SharedDirectory>(
        "SELECT id, device_id, local_path, share_name, is_readable, is_writable, created_at
         FROM zynk_linked_directories
         WHERE id = ?"
    )
    .bind(share_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("Failed to get share: {}", e))?;

    let share = share.ok_or_else(|| "Share not found".to_string())?;

    if !share.is_readable {
        return Err("Share is not readable".to_string());
    }

    // Construct full path
    let base_path = PathBuf::from(&share.local_path);
    let full_path = base_path.join(relative_path);

    // Security check: ensure path is within share directory
    let canonical_base = base_path.canonicalize()
        .map_err(|e| format!("Failed to canonicalize base path: {}", e))?;
    let canonical_full = full_path.canonicalize()
        .map_err(|e| format!("Failed to canonicalize file path: {}", e))?;

    if !canonical_full.starts_with(&canonical_base) {
        return Err("Path traversal detected".to_string());
    }

    // Verify file exists
    if !canonical_full.is_file() {
        return Err("File not found or is not a file".to_string());
    }

    Ok(canonical_full)
}

// =============================================================================
// ZynkLink Code Generation & Acceptance
// =============================================================================

/// Generate a ZynkLink code for file sharing (6-digit, matches ZynkSync format)
pub async fn generate_zynklink_code(
    pool: &SqlitePool,
    user_id: &str,
    device_id: &str,
) -> Result<String, String> {
    println!("[ZynkLink] Generating code for user: {}..., device: {}...", &user_id[..8], &device_id[..8]);

    // Generate unique 6-digit code (matching ZynkSync format)
    let code = loop {
        let random_num = rand::random::<u32>() % 900000 + 100000;
        let candidate = format!("{:06}", random_num);

        // Check if code already exists
        let exists = sqlx::query("SELECT code FROM zynklink_codes WHERE code = ?")
            .bind(&candidate)
            .fetch_optional(pool)
            .await
            .map_err(|e| format!("Database query failed: {}", e))?;

        if exists.is_none() {
            break candidate;
        }
    };

    println!("[ZynkLink] Generated code: {}", code);
    println!("[ZynkLink] Inserting into database...");

    // Insert into database (expires in 5 minutes, matching ZynkSync)
    sqlx::query(
        "INSERT INTO zynklink_codes (code, creator_user_id, creator_device_id, expires_at, is_active)
         VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '+5 minutes'), true)"
    )
    .bind(&code)
    .bind(user_id)
    .bind(device_id)
    .execute(pool)
    .await
    .map_err(|e| {
        println!("[ZynkLink] Database insert failed: {}", e);
        format!("Failed to generate ZynkLink code: {}", e)
    })?;

    println!("[ZynkLink] Code inserted successfully (expires in 5 min)");

    Ok(code)
}

/// Accept a ZynkLink code and create file sharing pairing
pub async fn accept_zynklink_code(
    pool: &SqlitePool,
    code: &str,
    acceptor_user_id: &str,
    acceptor_device_id: &str,
) -> Result<serde_json::Value, String> {
    // Get the code from database
    let code_record = sqlx::query_as::<_, (String, String, String, bool, Option<String>)>(
        "SELECT creator_user_id, creator_device_id, code, is_active, accepted_by_user_id
         FROM zynklink_codes
         WHERE code = ? AND (expires_at IS NULL OR expires_at > datetime('now'))"
    )
    .bind(code)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("Failed to fetch code: {}", e))?;

    let (creator_user_id, creator_device_id, _code, is_active, accepted_by) = code_record
        .ok_or_else(|| "Invalid or expired ZynkLink code".to_string())?;

    // Check if code is already used
    if !is_active {
        return Err("This ZynkLink code has already been used".to_string());
    }

    if accepted_by.is_some() {
        return Err("This ZynkLink code has already been accepted".to_string());
    }

    // Can't accept your own code from the same device
    // Allow same user_id (user linking their own devices), but not same device_id
    if creator_device_id == acceptor_device_id {
        return Err("You cannot accept your own ZynkLink code from the same device".to_string());
    }

    // Mark code as accepted
    sqlx::query(
        "UPDATE zynklink_codes
         SET accepted_by_user_id = ?, accepted_by_device_id = ?, accepted_at = datetime('now'), is_active = false
         WHERE code = ?"
    )
    .bind(acceptor_user_id)
    .bind(acceptor_device_id)
    .bind(code)
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to mark code as accepted: {}", e))?;

    // Create bidirectional pairing (user1 < user2 for consistency)
    let (user1_id, user2_id, device1_id, device2_id) = if creator_user_id.as_str() < acceptor_user_id {
        (creator_user_id.clone(), acceptor_user_id.to_string(), creator_device_id.clone(), acceptor_device_id.to_string())
    } else {
        (acceptor_user_id.to_string(), creator_user_id.clone(), acceptor_device_id.to_string(), creator_device_id.clone())
    };

    // Insert pairing (ON CONFLICT do nothing if already paired)
    sqlx::query(
        "INSERT INTO zynklink_pairings (user1_id, user2_id, device1_id, device2_id, is_active)
         VALUES (?, ?, ?, ?, true)
         ON CONFLICT (device1_id, device2_id) DO UPDATE SET is_active = true, linked_at = datetime('now')"
    )
    .bind(&user1_id)
    .bind(&user2_id)
    .bind(&device1_id)
    .bind(&device2_id)
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to create ZynkLink pairing: {}", e))?;

    Ok(serde_json::json!({
        "success": true,
        "message": format!("Successfully linked for file sharing with user {}",
                         if creator_user_id == acceptor_user_id { &user2_id } else { &creator_user_id })
    }))
}

/// List all users linked for file sharing
pub async fn list_zynklink_pairings(
    pool: &SqlitePool,
    user_id: &str,
    device_id: &str,
) -> Result<serde_json::Value, String> {
    let pairings = sqlx::query_as::<_, (String, String, Option<String>, DateTime<Utc>, Option<DateTime<Utc>>)>(
        "SELECT
            CASE WHEN user1_id = ? THEN user2_id ELSE user1_id END as linked_user_id,
            CASE WHEN device1_id = ? THEN device2_id ELSE device1_id END as linked_device_id,
            zd.device_name,
            zp.linked_at,
            zd.last_seen_at
         FROM zynklink_pairings zp
         LEFT JOIN zynk_devices zd ON (CASE WHEN device1_id = ? THEN device2_id ELSE device1_id END) = zd.device_id
         WHERE (device1_id = ? OR device2_id = ?) AND is_active = 1
         ORDER BY zp.linked_at DESC"
    )
    .bind(user_id)
    .bind(device_id)
    .bind(device_id)
    .bind(device_id)
    .bind(device_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to fetch ZynkLink pairings: {}", e))?;

    let now = Utc::now();
    let mut linked_users: Vec<serde_json::Value> = Vec::new();
    for (linked_user_id, linked_device_id, device_name, linked_at, last_seen_at) in pairings {
        let is_online = last_seen_at
            .map(|seen| (now - seen).num_seconds() < 20)
            .unwrap_or(false);
        let is_paused = is_link_paused(&linked_device_id).await;
        linked_users.push(serde_json::json!({
            "user_id": linked_user_id,
            "device_id": linked_device_id,
            "device_name": device_name,
            "linked_at": linked_at,
            "is_online": is_online,
            "last_seen_at": last_seen_at,
            "is_paused": is_paused
        }));
    }

    Ok(serde_json::json!({
        "linked_users": linked_users
    }))
}

/// Revoke a ZynkLink pairing
pub async fn revoke_zynklink_pairing(
    pool: &SqlitePool,
    user_id: &str,
    linked_user_id: &str,
) -> Result<serde_json::Value, String> {
    // Get device IDs before deleting the pairing
    let pairing = sqlx::query_as::<_, (String, String)>(
        "SELECT device1_id, device2_id FROM zynklink_pairings
         WHERE ((user1_id = ? AND user2_id = ?) OR (user1_id = ? AND user2_id = ?))
         AND is_active = 1"
    )
    .bind(user_id)
    .bind(linked_user_id)
    .bind(linked_user_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("Failed to find pairing: {}", e))?
    .ok_or_else(|| "No active pairing found with this user".to_string())?;

    let (device1_id, device2_id) = pairing;

    // Get the other device's IP address BEFORE we clear it (for notification)
    let other_device_id = if device1_id == crate::user_identity::get_device_id()? {
        device2_id.clone()
    } else {
        device1_id.clone()
    };

    let other_device_ip = sqlx::query_as::<_, (Option<String>,)>(
        "SELECT device_ip FROM zynk_devices WHERE device_id = ?"
    )
    .bind(&other_device_id)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
    .and_then(|r| r.0);

    // DELETE chat messages between these devices.
    // zchat_messages stores device IDs as UUID blobs, not text strings.
    let uuid1 = uuid::Uuid::parse_str(&device1_id)
        .map_err(|e| format!("Invalid device1 UUID: {}", e))?;
    let uuid2 = uuid::Uuid::parse_str(&device2_id)
        .map_err(|e| format!("Invalid device2 UUID: {}", e))?;
    let deleted_messages = sqlx::query(
        "DELETE FROM zchat_messages
         WHERE (from_device_id = ? AND to_device_id = ?)
            OR (from_device_id = ? AND to_device_id = ?)"
    )
    .bind(uuid1)
    .bind(uuid2)
    .bind(uuid2)
    .bind(uuid1)
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to delete chat messages: {}", e))?;

    println!("[ZynkLink] Deleted {} chat messages between devices", deleted_messages.rows_affected());

    // Clear IP addresses from both devices
    sqlx::query(
        "UPDATE zynk_devices SET device_ip = NULL
         WHERE device_id = ? OR device_id = ?"
    )
    .bind(&device1_id)
    .bind(&device2_id)
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to clear IP addresses: {}", e))?;

    // DELETE the pairing (not just deactivate)
    // NOTE: shared directories are local configuration and intentionally preserved —
    // the user's folder sharing setup should survive a device being unlinked.
    let result = sqlx::query(
        "DELETE FROM zynklink_pairings
         WHERE ((user1_id = ? AND user2_id = ?) OR (user1_id = ? AND user2_id = ?))"
    )
    .bind(user_id)
    .bind(linked_user_id)
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to delete ZynkLink pairing: {}", e))?;

    if result.rows_affected() == 0 {
        return Err("No pairing found with this user".to_string());
    }

    println!("[ZynkLink] ✓ Unlinked from user {}", &linked_user_id[..8]);

    // Remove the remote device record if it's now fully orphaned — best-effort, non-fatal
    match sqlx::query(
        "DELETE FROM zynk_devices
         WHERE device_id = ?
           AND NOT EXISTS (SELECT 1 FROM zynk_device_pairings WHERE device_a_id = device_id OR device_b_id = device_id)
           AND NOT EXISTS (SELECT 1 FROM zynklink_pairings WHERE is_active = 1 AND (device1_id = device_id OR device2_id = device_id))"
    )
    .bind(&other_device_id)
    .execute(pool)
    .await {
        Ok(r) if r.rows_affected() > 0 => println!("[ZynkLink] Cleaned up orphaned device record {}", &other_device_id[..8]),
        Err(e) => println!("[ZynkLink] Note: orphaned device cleanup failed (non-fatal): {}", e),
        _ => {}
    }

    // Sweep expired / already-used ZynkLink codes — best-effort
    match sqlx::query(
        "DELETE FROM zynklink_codes
         WHERE is_active = 0
            OR (expires_at IS NOT NULL AND expires_at < datetime('now'))"
    )
    .execute(pool)
    .await {
        Ok(r) if r.rows_affected() > 0 => println!("[ZynkLink] Swept {} expired/used ZynkLink code(s)", r.rows_affected()),
        Err(e) => println!("[ZynkLink] Note: code sweep failed (non-fatal): {}", e),
        _ => {}
    }

    // Best-effort push: tell the remote device to remove its pairing record too.
    // Fire-and-forget — if the remote is offline the auth check on their server
    // will block any further file/chat access anyway.
    if let Some(ip) = other_device_ip {
        let notify_url = format!("https://{}:{}/api/zynklink/notify-unpaired", ip, crate::zynksync::DEFAULT_SYNC_PORT);
        let payload = serde_json::json!({ "unlinked_user_id": user_id });
        tokio::spawn(async move {
            match reqwest::Client::builder()
                .danger_accept_invalid_certs(true)
                .timeout(std::time::Duration::from_secs(5))
                .build()
            {
                Ok(client) => {
                    match client.post(&notify_url).json(&payload).send().await {
                        Ok(_) => println!("[ZynkLink] ✓ Unlink notification sent to remote device"),
                        Err(e) => println!("[ZynkLink] Note: could not notify remote device of unlink (offline?): {}", e),
                    }
                }
                Err(e) => println!("[ZynkLink] Note: could not build notify client: {}", e),
            }
        });
    }

    Ok(serde_json::json!({
        "success": true,
        "message": "Unlinked successfully."
    }))
}

/// Deliver undelivered ZChat messages to a ZynkLink-paired device
pub async fn deliver_zchat_to_peer(
    pool: &SqlitePool,
    current_user_id: &str,
    current_device_id: &str,
    to_device_id: &str,
) -> Result<usize, String> {
    use crate::zchat;
    use uuid::Uuid;

    // Get the paired device's IP address — must match BOTH current and target device IDs
    // so messages are never misrouted when a device has multiple ZynkLink pairings.
    let device_info = sqlx::query_as::<_, (String, Option<String>)>(
        "SELECT
            CASE WHEN device1_id = ? THEN device2_id ELSE device1_id END as paired_device_id,
            zd.device_ip
         FROM zynklink_pairings zp
         LEFT JOIN zynk_devices zd ON (CASE WHEN device1_id = ? THEN device2_id ELSE device1_id END) = zd.device_id
         WHERE (user1_id = ? OR user2_id = ?)
           AND (device1_id = ? OR device2_id = ?)
           AND (device1_id = ? OR device2_id = ?)
           AND is_active = 1"
    )
    .bind(current_device_id)
    .bind(current_device_id)
    .bind(current_user_id)
    .bind(current_user_id)
    .bind(current_device_id)
    .bind(current_device_id)
    .bind(to_device_id)
    .bind(to_device_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("Failed to get paired device info: {}", e))?;

    let (paired_device_id, device_ip_opt) = device_info
        .ok_or_else(|| format!("No ZynkLink pairing found for device {}", &to_device_id[..8]))?;

    if is_link_paused(&paired_device_id).await {
        return Ok(0);
    }

    let device_ip = device_ip_opt
        .ok_or_else(|| format!("No IP address registered for device {}", &paired_device_id[..8]))?;

    // Get current device ID for the "from" field
    let from_device_id = crate::user_identity::get_device_id()?;
    let from_device_uuid = Uuid::parse_str(&from_device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;
    let to_device_uuid = Uuid::parse_str(to_device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    // Get undelivered messages for this device
    let messages = zchat::get_undelivered_messages(pool, from_device_uuid, to_device_uuid).await?;

    if messages.is_empty() {
        return Ok(0);
    }

    println!("[ZynkLink] Delivering {} message(s) to device {}... at {}",
        messages.len(), &to_device_id[..8], device_ip);

    // Send messages via HTTPS (peer cert not pinned here — TOFU accepted for ZynkLink chat)
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let url = format!("https://{}:{}/api/zynklink/deliver-chat", device_ip, crate::zynksync::DEFAULT_SYNC_PORT);

    let response = match client
        .post(&url)
        .json(&messages)
        .send()
        .await {
        Ok(resp) => {
            // Update last_seen_at on successful connection
            let _ = sqlx::query(
                "UPDATE zynk_devices SET last_seen_at = datetime('now') WHERE device_id = ?"
            )
            .bind(to_device_id)
            .execute(pool)
            .await;
            resp
        },
        Err(_e) => {
            // Device is offline - messages will remain queued for later delivery
            println!("[ZynkLink] Device {} is offline - messages queued for delivery when device comes back online", &to_device_id[..8]);
            return Err(format!("Device is offline - message will be delivered when device reconnects"));
        }
    };

    if !response.status().is_success() {
        return Err(format!("HTTP error: {}", response.status()));
    }

    // Mark messages as delivered
    let message_ids: Vec<Uuid> = messages
        .iter()
        .filter_map(|m| Uuid::parse_str(&m.id).ok())
        .collect();
    zchat::mark_delivered(pool, message_ids).await?;

    println!("[ZynkLink] ✓ Delivered {} message(s)", messages.len());
    Ok(messages.len())
}

// ============================================================================
// HTTP routes — served by the transport (crate::transport), 2026-09-17.
// ZynkLink owns these handlers and its own tables (zynklink_pairings, shares);
// the transport provides the server, certificates and the device registry.
// ============================================================================

use axum::{extract::State, routing::post, Json, Router, http::StatusCode};
use std::sync::Arc;
use crate::transport::{Transport, DEFAULT_SYNC_PORT};
use tauri::Emitter;

/// ZynkLink's route bundle. The two code-exchange routes are public (a device that is
/// not yet linked must reach them); the rest are authenticated per request by
/// `check_zynklink_authorized` against zynklink_pairings — moving them behind mTLS
/// waits on link pairing pinning certificates (ROADMAP: ZynkLink mTLS cert exchange).
pub fn routes(transport: Arc<Transport>) -> Router {
    Router::new()
        .route("/api/zynklink/verify-code", post(handle_zynklink_verify_code))
        .route("/api/zynklink/accept-code", post(handle_zynklink_accept_code))
        .route("/api/zynklink/directories", post(handle_zynklink_directories))
        .route("/api/zynklink/files", post(handle_zynklink_files))
        .route("/api/zynklink/download", post(handle_zynklink_download))
        .route("/api/zynklink/deliver-chat", post(handle_zynklink_deliver_chat))
        .route("/api/zynklink/notify-unpaired", post(handle_zynklink_notify_unpaired))
        .with_state(transport)
}

/// Verify a ZynkLink code (like verify_sync_code but for file sharing)
async fn handle_zynklink_verify_code(
    State(transport): State<Arc<Transport>>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let code = request.get("code")
        .and_then(|c| c.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "Missing code parameter".to_string()))?;

    println!("[ZynkLink] Verifying code: {}", code);

    // Query database to verify the ZynkLink code, also fetch device_name via join
    let result = sqlx::query_as::<_, (String, String, Option<String>)>(
        "SELECT lc.creator_user_id, lc.creator_device_id, zd.device_name
         FROM zynklink_codes lc
         LEFT JOIN zynk_devices zd ON zd.device_id = lc.creator_device_id
         WHERE lc.code = ? AND lc.expires_at > datetime('now') AND lc.is_active = 1"
    )
    .bind(code)
    .fetch_optional(&transport.db_pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Database query failed: {}", e)))?;

    match result {
        Some(record) => {
            println!("[ZynkLink] Code verified for user: {}", record.0);
            Ok(Json(serde_json::json!({
                "user_id": record.0,
                "device_id": record.1,
                "device_name": record.2
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
    State(transport): State<Arc<Transport>>,
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

    let acceptor_device_name = request.get("acceptor_device_name")
        .and_then(|n| n.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("Remote Device {}", &acceptor_device_id[..8.min(acceptor_device_id.len())]));

    println!("[ZynkLink] Accept code request: {} from user: {}..., IP: {:?}, name: {}",
             code, &acceptor_user_id[..8], acceptor_device_ip, acceptor_device_name);

    // Ensure acceptor's device exists in zynk_devices (required for foreign key constraint)
    println!("[ZynkLink] Device A: Ensuring acceptor's device is registered...");
    sqlx::query(
        &format!("INSERT INTO zynk_devices (device_id, device_name, device_ip, owner_user_id, is_paired, port, created_at, last_seen_at)
         VALUES (?, ?, ?, ?, true, {}, datetime('now'), datetime('now'))
         ON CONFLICT (device_id) DO UPDATE
         SET device_ip = ?, owner_user_id = ?, device_name = excluded.device_name, last_seen_at = datetime('now')", DEFAULT_SYNC_PORT)
    )
    .bind(acceptor_device_id)
    .bind(&acceptor_device_name)
    .bind(acceptor_device_ip)
    .bind(acceptor_user_id)
    .bind(acceptor_device_ip)
    .bind(acceptor_user_id)
    .execute(&transport.db_pool)
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
    let mut result = match accept_zynklink_code(
        &transport.db_pool,
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
        .fetch_one(&transport.db_pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to get creator device ID: {}", e)))?;

        println!("[ZynkLink] Device A: Storing our own IP {} in database", creator_ip);
        sqlx::query(
            "UPDATE zynk_devices SET device_ip = ?, last_seen_at = datetime('now') WHERE device_id = ?"
        )
        .bind(creator_ip)
        .bind(&creator_device_id)
        .execute(&transport.db_pool)
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

/// List shared directories from a device
/// Check that requester_user_id has an active ZynkLink pairing with this device's user.
async fn check_zynklink_authorized(transport: &Transport, requester_user_id: &str) -> Result<(), String> {
    let local_user_id = transport.user_id()
        .map_err(|e| format!("Failed to get local user ID: {}", e))?;
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM zynklink_pairings
         WHERE is_active = 1
         AND ((user1_id = ? AND user2_id = ?) OR (user1_id = ? AND user2_id = ?))"
    )
    .bind(&local_user_id).bind(requester_user_id)
    .bind(requester_user_id).bind(&local_user_id)
    .fetch_one(&transport.db_pool)
    .await
    .map_err(|e| format!("Failed to check ZynkLink authorization: {}", e))?;
    if count == 0 {
        Err("Not authorized: no active ZynkLink pairing with this user".to_string())
    } else {
        Ok(())
    }
}

async fn handle_zynklink_directories(
    State(transport): State<Arc<Transport>>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let requester_user_id = request.get("requester_user_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing requester_user_id"}))))?;

    check_zynklink_authorized(&transport, requester_user_id).await
        .map_err(|e| (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": e}))))?;

    let local_device_id = Ok::<String, String>(transport.device_id())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;
    let response = list_my_shared_directories(
        &transport.db_pool,
        &local_device_id
    ).await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;

    serde_json::to_value(response)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))))
}

/// List files in a shared directory
async fn handle_zynklink_files(
    State(transport): State<Arc<Transport>>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let requester_user_id = request.get("requester_user_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing requester_user_id"}))))?;

    check_zynklink_authorized(&transport, requester_user_id).await
        .map_err(|e| (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": e}))))?;

    let share_id = request.get("share_id")
        .and_then(|s| s.as_i64())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing share_id parameter"}))))? as i32;

    println!("[ZynkLink] File list request for share_id: {}", share_id);

    // List files in the shared directory
    let response = list_files(
        &transport.db_pool,
        share_id
    ).await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))))?;

    serde_json::to_value(response)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))))
}

/// Download a file from a shared directory
async fn handle_zynklink_download(
    State(transport): State<Arc<Transport>>,
    Json(request): Json<serde_json::Value>,
) -> Result<axum::response::Response, (StatusCode, Json<serde_json::Value>)> {
    use tokio::io::AsyncReadExt;

    let requester_user_id = request.get("requester_user_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing requester_user_id"}))))?;

    check_zynklink_authorized(&transport, requester_user_id).await
        .map_err(|e| (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": e}))))?;

    let share_id = request.get("share_id")
        .and_then(|s| s.as_i64())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing share_id parameter"}))))? as i32;

    let relative_path = request.get("relative_path")
        .and_then(|p| p.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing relative_path parameter"}))))?;

    println!("[ZynkLink] Download request for share_id: {}, path: {}", share_id, relative_path);

    let file_path = get_file_path(
        &transport.db_pool,
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
    State(transport): State<Arc<Transport>>,
    Json(messages): Json<Vec<crate::zchat::DeliverMessageData>>,
) -> Result<axum::Json<crate::zchat::DeliverMessagesResponse>, String> {
    println!("[ZynkLink] Received {} chat message(s) for delivery", messages.len());

    // Call zchat deliver_messages function
    let result = crate::zchat::deliver_messages(&transport.db_pool, messages).await?;

    println!("[ZynkLink] ✓ Delivered {} message(s)", result.received_count);
    Ok(axum::Json(result))
}

/// Handle notification that a remote device has unlinked.
/// Removes the local pairing record and emits a UI refresh event.
async fn handle_zynklink_notify_unpaired(
    State(transport): State<Arc<Transport>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<axum::Json<serde_json::Value>, String> {
    let unlinked_device_id = payload.get("unlinked_device_id")
        .and_then(|v| v.as_str())
        // fall back to legacy field name for older clients
        .or_else(|| payload.get("unlinked_user_id").and_then(|v| v.as_str()))
        .ok_or("Missing unlinked_device_id")?;

    // Delegate full ZynkLink cleanup to clear_link_data — preserves ZynkSync if active
    match clear_link_data(&transport, unlinked_device_id).await {
        Ok(_) => println!("[ZynkLink] ✓ Remote unlink: cleared link data for peer {}", &unlinked_device_id[..unlinked_device_id.len().min(8)]),
        Err(e) => println!("[ZynkLink] Note: clear_link_data on notify-unpaired failed (non-fatal): {}", e),
    }

    if let Ok(guard) = crate::APP_HANDLE.lock() {
        if let Some(app) = guard.as_ref() {
            let _ = app.emit("zynklink-pairing-updated", serde_json::json!({
                "unlinked": true,
                "remote_device_id": unlinked_device_id
            }));
        }
    }

    Ok(axum::Json(serde_json::json!({ "success": true })))
}

/// Clear ZynkLink pairing data for a device. Called by revoke_zynklink_pairing.
/// Preserves ZynkSync pairing if sync_paired = 1.
pub async fn clear_link_data(transport: &Transport, device_id: &str) -> Result<(), String> {
    println!("[ZynkLink] Clearing link data for device: {}", &device_id[..device_id.len().min(8)]);

    let device_uuid = uuid::Uuid::parse_str(device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    let mut tx = transport.db_pool.begin().await
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
        let mut peers_map = transport.peers.write().await;
        peers_map.remove(device_id);
        // Must drop write lock before committing
        drop(peers_map);
    }

    tx.commit().await
        .map_err(|e| format!("Failed to commit link removal: {}", e))?;

    println!("[ZynkLink] ✓ Cleared link data for device {}", &device_id[..device_id.len().min(8)]);
    Ok(())
}

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
         VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
         ON CONFLICT (device_id, local_path)
         DO UPDATE SET
             share_name = EXCLUDED.share_name,
             is_readable = EXCLUDED.is_readable,
             is_writable = EXCLUDED.is_writable,
             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
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

/// Scan a directory and index all files
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

    // Clear existing file entries
    sqlx::query("DELETE FROM zynk_file_manifest WHERE shared_directory_id = ?")
        .bind(share_id)
        .execute(pool)
        .await
        .map_err(|e| format!("Failed to clear file entries: {}", e))?;

    // Scan directory recursively
    let base_path = PathBuf::from(&share.local_path);
    let mut files_indexed = 0;
    let max = max_files.unwrap_or(1000);

    if let Err(e) = scan_directory_recursive(pool, share_id, &base_path, &base_path, &mut files_indexed, max).await {
        return Err(format!("Failed to scan directory: {}", e));
    }

    Ok(serde_json::json!({
        "success": true,
        "files_indexed": files_indexed
    }))
}

/// Recursively scan directory helper
async fn scan_directory_recursive(
    pool: &SqlitePool,
    share_id: i32,
    base_path: &Path,
    current_path: &Path,
    files_indexed: &mut usize,
    max_files: usize,
) -> Result<(), String> {
    if *files_indexed >= max_files {
        return Ok(());
    }

    let mut entries = fs::read_dir(current_path)
        .await
        .map_err(|e| format!("Failed to read directory: {}", e))?;

    while let Some(entry) = entries.next_entry().await.map_err(|e| e.to_string())? {
        if *files_indexed >= max_files {
            break;
        }

        let path = entry.path();
        let metadata = match fs::metadata(&path).await {
            Ok(m) => m,
            Err(_) => continue, // Skip files we can't read
        };

        if metadata.is_file() {
            // Get relative path
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

            // Upsert file record — rescans should refresh size/mtime, not crash on the
            // existing (shared_directory_id, relative_path) UNIQUE constraint.
            sqlx::query(
                "INSERT INTO zynk_file_manifest (shared_directory_id, relative_path, file_size, last_modified, indexed_at)
                 VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                 ON CONFLICT(shared_directory_id, relative_path) DO UPDATE SET
                     file_size = excluded.file_size,
                     last_modified = excluded.last_modified,
                     indexed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
            )
            .bind(share_id)
            .bind(&relative_path)
            .bind(file_size)
            .bind(last_modified)
            .execute(pool)
            .await
            .map_err(|e| format!("Failed to insert file record: {}", e))?;

            *files_indexed += 1;
        } else if metadata.is_dir() {
            // Recurse into subdirectory (Box::pin required for async recursion)
            Box::pin(scan_directory_recursive(pool, share_id, base_path, &path, files_indexed, max_files)).await?;
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
         WHERE code = ? AND (expires_at IS NULL OR expires_at > strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
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
         SET accepted_by_user_id = ?, accepted_by_device_id = ?, accepted_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), is_active = false
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
         ON CONFLICT (user1_id, user2_id) DO UPDATE SET is_active = true, linked_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
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
    let pairings = sqlx::query_as::<_, (String, String, DateTime<Utc>, Option<DateTime<Utc>>)>(
        "SELECT
            CASE WHEN user1_id = ? THEN user2_id ELSE user1_id END as linked_user_id,
            CASE WHEN device1_id = ? THEN device2_id ELSE device1_id END as linked_device_id,
            zp.linked_at,
            zd.last_seen_at
         FROM zynklink_pairings zp
         LEFT JOIN zynk_devices zd ON (CASE WHEN device1_id = ? THEN device2_id ELSE device1_id END) = zd.device_id
         WHERE (user1_id = ? OR user2_id = ?) AND is_active = 1
         ORDER BY zp.linked_at DESC"
    )
    .bind(user_id)
    .bind(device_id)
    .bind(device_id)
    .bind(user_id)
    .bind(user_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to fetch ZynkLink pairings: {}", e))?;

    let now = Utc::now();
    let mut linked_users: Vec<serde_json::Value> = Vec::new();
    for (linked_user_id, linked_device_id, linked_at, last_seen_at) in pairings {
        let is_online = last_seen_at
            .map(|seen| (now - seen).num_seconds() < 20)
            .unwrap_or(false);
        let is_paused = is_link_paused(&linked_device_id).await;
        linked_users.push(serde_json::json!({
            "user_id": linked_user_id,
            "device_id": linked_device_id,
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

    // Remove the remote device record if it's now fully orphaned (no sync or link pairings left)
    let deleted_device = sqlx::query(
        "DELETE FROM zynk_devices
         WHERE device_id = ?
           AND NOT EXISTS (SELECT 1 FROM zynk_device_pairings WHERE device1_id = device_id OR device2_id = device_id)
           AND NOT EXISTS (SELECT 1 FROM zynklink_pairings WHERE is_active = 1 AND (device1_id = device_id OR device2_id = device_id))"
    )
    .bind(&other_device_id)
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to clean up orphaned device: {}", e))?;
    if deleted_device.rows_affected() > 0 {
        println!("[ZynkLink] Cleaned up orphaned device record {}", &other_device_id[..8]);
    }

    // Sweep expired / already-used ZynkLink codes
    let deleted_codes = sqlx::query(
        "DELETE FROM zynklink_codes
         WHERE is_active = 0
            OR (expires_at IS NOT NULL AND expires_at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
    )
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to sweep expired codes: {}", e))?;
    if deleted_codes.rows_affected() > 0 {
        println!("[ZynkLink] Swept {} expired/used ZynkLink code(s)", deleted_codes.rows_affected());
    }

    // Best-effort push: tell the remote device to remove its pairing record too.
    // Fire-and-forget — if the remote is offline the auth check on their server
    // will block any further file/chat access anyway.
    if let Some(ip) = other_device_ip {
        let notify_url = format!("https://{}:57963/api/zynklink/notify-unpaired", ip);
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

    // Get the paired device's IP address
    let device_info = sqlx::query_as::<_, (String, Option<String>)>(
        "SELECT
            CASE WHEN device1_id = ? THEN device2_id ELSE device1_id END as paired_device_id,
            zd.device_ip
         FROM zynklink_pairings zp
         LEFT JOIN zynk_devices zd ON (CASE WHEN device1_id = ? THEN device2_id ELSE device1_id END) = zd.device_id
         WHERE (user1_id = ? OR user2_id = ?)
           AND (device1_id = ? OR device2_id = ?)
           AND is_active = 1"
    )
    .bind(current_device_id)
    .bind(current_device_id)
    .bind(current_user_id)
    .bind(current_user_id)
    .bind(current_device_id)
    .bind(current_device_id)
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

    let url = format!("https://{}:57963/api/zynklink/deliver-chat", device_ip);

    let response = match client
        .post(&url)
        .json(&messages)
        .send()
        .await {
        Ok(resp) => {
            // Update last_seen_at on successful connection
            let _ = sqlx::query(
                "UPDATE zynk_devices SET last_seen_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE device_id = ?"
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

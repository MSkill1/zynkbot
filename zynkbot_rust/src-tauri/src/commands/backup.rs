use aes_gcm::{Aes256Gcm, Key, Nonce, aead::{Aead, KeyInit}};
use hmac::{Hmac, Mac};
use sha2::digest::KeyInit as HmacKeyInit;
use rand::RngCore;
use sha2::{Sha256, Digest};
use chrono::Utc;
use std::collections::BTreeMap;

type HmacSha256 = Hmac<Sha256>;

// --- Key management ---

fn key_path() -> std::path::PathBuf {
    crate::db::get_app_data_dir().join("backup.key")
}

fn key_acknowledged_path() -> std::path::PathBuf {
    crate::db::get_app_data_dir().join("backup_key_acknowledged")
}

fn load_or_create_key() -> Result<[u8; 32], String> {
    let path = key_path();
    if path.exists() {
        let hex_str = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read backup key: {}", e))?;
        let bytes = hex::decode(hex_str.trim())
            .map_err(|e| format!("Invalid backup key format: {}", e))?;
        if bytes.len() != 32 {
            return Err("Backup key has wrong length".to_string());
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(&bytes);
        Ok(key)
    } else {
        let mut key = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut key);
        std::fs::write(&path, hex::encode(&key))
            .map_err(|e| format!("Failed to write backup key: {}", e))?;
        Ok(key)
    }
}

#[tauri::command]
pub async fn get_backup_key() -> Result<String, String> {
    load_or_create_key().map(|k| hex::encode(&k))
}

/// Returns the current backup key and whether the user has acknowledged saving it.
#[tauri::command]
pub async fn get_backup_key_status() -> Result<serde_json::Value, String> {
    let key = load_or_create_key()?;
    let acknowledged = key_acknowledged_path().exists();
    Ok(serde_json::json!({
        "key": hex::encode(&key),
        "acknowledged": acknowledged
    }))
}

/// Records that the user has saved their backup key.
#[tauri::command]
pub async fn acknowledge_backup_key() -> Result<(), String> {
    std::fs::write(key_acknowledged_path(), b"1")
        .map_err(|e| format!("Failed to write acknowledged flag: {}", e))
}

/// Derives a 32-byte AES-256 key from a passphrase using Argon2id and saves it as
/// the backup key. Uses a fixed domain-separator salt so the same passphrase always
/// produces the same key on any device — enabling passphrase-based recovery.
#[tauri::command]
pub async fn derive_key_from_passphrase(passphrase: String) -> Result<String, String> {
    use argon2::{Argon2, Algorithm, Version, Params};

    if passphrase.trim().is_empty() {
        return Err("Passphrase cannot be empty".to_string());
    }

    // Fixed salt — intentionally deterministic for cross-device recovery
    const SALT: &[u8] = b"zynkbot-backup-v1-key-derivation";

    let params = Params::new(65536, 3, 1, Some(32))
        .map_err(|e| format!("Invalid Argon2 params: {:?}", e))?;
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

    let mut key = [0u8; 32];
    argon2.hash_password_into(passphrase.as_bytes(), SALT, &mut key)
        .map_err(|e| format!("Key derivation failed: {:?}", e))?;

    std::fs::write(key_path(), hex::encode(&key))
        .map_err(|e| format!("Failed to save derived key: {}", e))?;
    std::fs::write(key_acknowledged_path(), b"1")
        .map_err(|e| format!("Failed to write acknowledged flag: {}", e))?;

    Ok(hex::encode(&key))
}

// --- Encryption ---

fn encrypt_bytes(key: &[u8; 32], plaintext: &[u8]) -> Result<Vec<u8>, String> {
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
    let mut nonce_bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher.encrypt(nonce, plaintext)
        .map_err(|e| format!("Encryption failed: {:?}", e))?;
    let mut result = nonce_bytes.to_vec();
    result.extend_from_slice(&ciphertext);
    Ok(result)
}

fn decrypt_bytes(key: &[u8; 32], data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 12 {
        return Err("Encrypted data too short".to_string());
    }
    let (nonce_bytes, ciphertext) = data.split_at(12);
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher.decrypt(nonce, ciphertext)
        .map_err(|_| "Decryption failed — wrong key or corrupted data".to_string())
}

// --- AWS Sig V4 (minimal, for R2/S3 PUT and GET) ---

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = <HmacSha256 as HmacKeyInit>::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn sha256_hex(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

struct R2Config {
    endpoint: String,
    bucket: String,
    access_key: String,
    secret_key: String,
}

fn load_r2_config() -> Result<R2Config, String> {
    Ok(R2Config {
        endpoint: std::env::var("R2_ENDPOINT")
            .map_err(|_| "R2 backup not configured — add R2_ENDPOINT in Settings → API Keys".to_string())?,
        bucket: std::env::var("R2_BUCKET").unwrap_or_else(|_| "zynkbot-backups".to_string()),
        access_key: std::env::var("R2_ACCESS_KEY_ID")
            .map_err(|_| "R2_ACCESS_KEY_ID not configured".to_string())?,
        secret_key: std::env::var("R2_SECRET_ACCESS_KEY")
            .map_err(|_| "R2_SECRET_ACCESS_KEY not configured".to_string())?,
    })
}

// Returns signed headers for an S3 PUT or GET request.
// `path` must start with `/` and include the bucket: `/{bucket}/{key}`
fn sig_v4_headers(
    method: &str,
    host: &str,
    path: &str,
    payload: &[u8],
    access_key: &str,
    secret_key: &str,
) -> BTreeMap<String, String> {
    let now = Utc::now();
    let datetime = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date = now.format("%Y%m%d").to_string();
    const REGION: &str = "auto";
    const SERVICE: &str = "s3";

    let payload_hash = sha256_hex(payload);
    let canonical_headers = format!(
        "host:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
        host, payload_hash, datetime
    );
    let signed_headers = "host;x-amz-content-sha256;x-amz-date";

    let canonical_request = format!(
        "{}\n{}\n\n{}\n{}\n{}",
        method, path, canonical_headers, signed_headers, payload_hash
    );

    let credential_scope = format!("{}/{}/{}/aws4_request", date, REGION, SERVICE);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        datetime,
        credential_scope,
        sha256_hex(canonical_request.as_bytes())
    );

    let signing_key = {
        let k1 = hmac_sha256(format!("AWS4{}", secret_key).as_bytes(), date.as_bytes());
        let k2 = hmac_sha256(&k1, REGION.as_bytes());
        let k3 = hmac_sha256(&k2, SERVICE.as_bytes());
        hmac_sha256(&k3, b"aws4_request")
    };

    let signature = hex::encode(hmac_sha256(&signing_key, string_to_sign.as_bytes()));

    let mut headers = BTreeMap::new();
    headers.insert("Authorization".to_string(), format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        access_key, credential_scope, signed_headers, signature
    ));
    headers.insert("x-amz-date".to_string(), datetime);
    headers.insert("x-amz-content-sha256".to_string(), payload_hash);
    headers
}

async fn r2_put(cfg: &R2Config, object_key: &str, data: Vec<u8>) -> Result<(), String> {
    let endpoint = cfg.endpoint.trim_end_matches('/');
    let host = endpoint
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    let path = format!("/{}/{}", cfg.bucket, object_key);
    let url = format!("{}{}", endpoint, path);

    let headers = sig_v4_headers("PUT", host, &path, &data, &cfg.access_key, &cfg.secret_key);

    let mut req = reqwest::Client::new().put(&url).body(data);
    for (k, v) in &headers {
        req = req.header(k, v);
    }

    let resp = req.send().await.map_err(|e| format!("R2 PUT request failed: {}", e))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("R2 PUT error {}: {}", status, body));
    }
    Ok(())
}

async fn r2_get(cfg: &R2Config, object_key: &str) -> Result<Vec<u8>, String> {
    let endpoint = cfg.endpoint.trim_end_matches('/');
    let host = endpoint
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    let path = format!("/{}/{}", cfg.bucket, object_key);
    let url = format!("{}{}", endpoint, path);

    let headers = sig_v4_headers("GET", host, &path, &[], &cfg.access_key, &cfg.secret_key);

    let mut req = reqwest::Client::new().get(&url);
    for (k, v) in &headers {
        req = req.header(k, v);
    }

    let resp = req.send().await.map_err(|e| format!("R2 GET request failed: {}", e))?;
    if resp.status() == 404 {
        return Err("No backup found on this account. Back up first.".to_string());
    }
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("R2 GET error {}: {}", status, body));
    }
    resp.bytes().await
        .map(|b| b.to_vec())
        .map_err(|e| format!("Failed to read R2 response: {}", e))
}

// --- Tauri commands ---

#[tauri::command]
pub async fn get_r2_config_status() -> Result<serde_json::Value, String> {
    let configured = std::env::var("R2_ENDPOINT").is_ok()
        && std::env::var("R2_ACCESS_KEY_ID").is_ok()
        && std::env::var("R2_SECRET_ACCESS_KEY").is_ok();
    Ok(serde_json::json!({ "configured": configured }))
}

#[tauri::command]
pub async fn backup_memories_to_r2(user_id: String) -> Result<serde_json::Value, String> {
    let key = load_or_create_key()?;
    let cfg = load_r2_config()?;

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url()).await
        .map_err(|e| format!("DB connect failed: {}", e))?;

    let rows: Vec<(
        Option<String>, String, Option<String>, Option<String>, Option<String>, String,
        bool, bool, bool, Option<String>, Option<String>, Option<String>, String,
    )> = sqlx::query_as(
        "SELECT title, content, source_type, session_id, user_id, namespace,
                is_syncable, is_shareable, is_ephemeral, entities_detected,
                event_type, original_text, CAST(created_at AS TEXT)
         FROM memories
         WHERE user_id = ? AND namespace != '_zynkbot'
         ORDER BY created_at"
    )
    .bind(&user_id)
    .fetch_all(&pool)
    .await
    .map_err(|e| format!("Query failed: {}", e))?;

    let count = rows.len();
    let memories_json: Vec<serde_json::Value> = rows.into_iter().map(|r| {
        serde_json::json!({
            "title": r.0,
            "content": r.1,
            "source_type": r.2,
            "session_id": r.3,
            "user_id": r.4,
            "namespace": r.5,
            "is_syncable": r.6,
            "is_shareable": r.7,
            "is_ephemeral": r.8,
            "entities_detected": r.9,
            "event_type": r.10,
            "original_text": r.11,
            "created_at": r.12,
        })
    }).collect();

    let session_rows: Vec<(String, String, Option<String>, String, String, i64, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT session_id, user_id, title, CAST(started_at AS TEXT), CAST(last_active AS TEXT),
                message_count, model_backend, containment_mode
         FROM conversation_sessions
         WHERE user_id = ?
         ORDER BY started_at"
    )
    .bind(&user_id)
    .fetch_all(&pool)
    .await
    .map_err(|e| format!("Session query failed: {}", e))?;

    let sessions_json: Vec<serde_json::Value> = session_rows.into_iter().map(|r| {
        serde_json::json!({
            "session_id": r.0,
            "user_id": r.1,
            "title": r.2,
            "started_at": r.3,
            "last_active": r.4,
            "message_count": r.5,
            "model_backend": r.6,
            "containment_mode": r.7,
        })
    }).collect();

    let message_rows: Vec<(String, String, String, String, String, Option<String>, Option<String>, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT session_id, user_id, role, content, CAST(created_at AS TEXT),
                model_backend, containment_mode, entry_hash, prev_hash
         FROM conversation_messages
         WHERE user_id = ?
         ORDER BY session_id, created_at"
    )
    .bind(&user_id)
    .fetch_all(&pool)
    .await
    .map_err(|e| format!("Message query failed: {}", e))?;

    let session_count = sessions_json.len();
    let message_count = message_rows.len();
    let messages_json: Vec<serde_json::Value> = message_rows.into_iter().map(|r| {
        serde_json::json!({
            "session_id": r.0,
            "user_id": r.1,
            "role": r.2,
            "content": r.3,
            "created_at": r.4,
            "model_backend": r.5,
            "containment_mode": r.6,
            "entry_hash": r.7,
            "prev_hash": r.8,
        })
    }).collect();

    pool.close().await;

    let bundle = serde_json::json!({
        "version": 2,
        "memories": memories_json,
        "conversations": {
            "sessions": sessions_json,
            "messages": messages_json,
        }
    });

    let payload = serde_json::to_vec(&bundle)
        .map_err(|e| format!("Serialization failed: {}", e))?;
    let encrypted = encrypt_bytes(&key, &payload)?;

    r2_put(&cfg, "backup.enc", encrypted).await?;

    println!("[Backup] Backed up {} memories, {} sessions, {} messages for user {}",
        count, session_count, message_count, user_id);
    Ok(serde_json::json!({
        "success": true,
        "count": count,
        "session_count": session_count,
        "message_count": message_count,
        "message": format!("Backed up {} memories, {} conversations ({} messages)", count, session_count, message_count)
    }))
}

#[tauri::command]
pub async fn restore_memories_from_r2(user_id: String) -> Result<serde_json::Value, String> {
    let key = load_or_create_key()?;
    let cfg = load_r2_config()?;

    let encrypted = r2_get(&cfg, "backup.enc").await?;
    let plaintext = decrypt_bytes(&key, &encrypted)?;
    let parsed: serde_json::Value = serde_json::from_slice(&plaintext)
        .map_err(|e| format!("Failed to parse backup data: {}", e))?;

    // v1 payload was a raw array of memories; v2 wraps memories + conversations.
    let (memories, sessions, messages): (Vec<serde_json::Value>, Vec<serde_json::Value>, Vec<serde_json::Value>) = match &parsed {
        serde_json::Value::Array(arr) => (arr.clone(), Vec::new(), Vec::new()),
        serde_json::Value::Object(_) => {
            let memories = parsed.get("memories").and_then(|v| v.as_array()).cloned().unwrap_or_default();
            let sessions = parsed.get("conversations").and_then(|c| c.get("sessions")).and_then(|v| v.as_array()).cloned().unwrap_or_default();
            let messages = parsed.get("conversations").and_then(|c| c.get("messages")).and_then(|v| v.as_array()).cloned().unwrap_or_default();
            (memories, sessions, messages)
        }
        _ => return Err("Unexpected backup payload shape".to_string()),
    };

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url()).await
        .map_err(|e| format!("DB connect failed: {}", e))?;

    let mut inserted = 0usize;
    let mut skipped = 0usize;
    let mut restored_hashes: Vec<String> = Vec::new();
    let mut sessions_inserted = 0usize;
    let mut sessions_skipped = 0usize;
    let mut messages_inserted = 0usize;
    let mut messages_skipped = 0usize;

    for mem in &memories {
        let content = match mem["content"].as_str() {
            Some(c) if !c.is_empty() => c.to_string(),
            _ => { skipped += 1; continue; }
        };

        let exists: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM memories WHERE content = ? AND user_id = ?"
        )
        .bind(&content)
        .bind(&user_id)
        .fetch_one(&pool)
        .await
        .unwrap_or(false);

        if exists { skipped += 1; continue; }

        let now = Utc::now().to_rfc3339();
        let result = sqlx::query(
            "INSERT INTO memories (
                title, content, source_type, session_id, user_id, namespace,
                is_syncable, is_shareable, is_ephemeral, entities_detected,
                event_type, original_text, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(mem["title"].as_str())
        .bind(&content)
        .bind(mem["source_type"].as_str())
        .bind(mem["session_id"].as_str())
        .bind(mem["user_id"].as_str().unwrap_or(&user_id))
        .bind(mem["namespace"].as_str().unwrap_or("general"))
        .bind(mem["is_syncable"].as_bool().unwrap_or(true))
        .bind(mem["is_shareable"].as_bool().unwrap_or(false))
        .bind(mem["is_ephemeral"].as_bool().unwrap_or(false))
        .bind(mem["entities_detected"].as_str())
        .bind(mem["event_type"].as_str())
        .bind(mem["original_text"].as_str())
        .bind(mem["created_at"].as_str())
        .bind(&now)
        .execute(&pool)
        .await;

        match result {
            Ok(_) => {
                let hash = format!("{:x}", Sha256::digest(content.as_bytes()));
                restored_hashes.push(hash);
                inserted += 1;
            }
            Err(e) => eprintln!("[Backup] Insert failed for memory: {}", e),
        }
    }

    // Clear local tombstones for restored memories so sync doesn't re-delete them.
    if !restored_hashes.is_empty() {
        for hash in &restored_hashes {
            let _ = sqlx::query("DELETE FROM deleted_memory_hashes WHERE content_hash = ?")
                .bind(hash)
                .execute(&pool)
                .await;
        }
        println!("[Backup] Cleared local tombstones for {} restored memories", restored_hashes.len());
    }

    // Restore conversation sessions (skip if session_id already exists).
    for sess in &sessions {
        let session_id = match sess["session_id"].as_str() {
            Some(s) if !s.is_empty() => s,
            _ => { sessions_skipped += 1; continue; }
        };

        let exists: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM conversation_sessions WHERE session_id = ?"
        )
        .bind(session_id)
        .fetch_one(&pool)
        .await
        .unwrap_or(false);

        if exists { sessions_skipped += 1; continue; }

        let result = sqlx::query(
            "INSERT INTO conversation_sessions (
                session_id, user_id, title, started_at, last_active,
                message_count, model_backend, containment_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(session_id)
        .bind(sess["user_id"].as_str().unwrap_or(&user_id))
        .bind(sess["title"].as_str())
        .bind(sess["started_at"].as_str())
        .bind(sess["last_active"].as_str())
        .bind(sess["message_count"].as_i64().unwrap_or(0))
        .bind(sess["model_backend"].as_str())
        .bind(sess["containment_mode"].as_str())
        .execute(&pool)
        .await;

        match result {
            Ok(_) => sessions_inserted += 1,
            Err(e) => eprintln!("[Backup] Session insert failed: {}", e),
        }
    }

    // Restore conversation messages (skip if same session_id + created_at + role exists).
    for msg in &messages {
        let session_id = match msg["session_id"].as_str() {
            Some(s) if !s.is_empty() => s,
            _ => { messages_skipped += 1; continue; }
        };
        let role = match msg["role"].as_str() {
            Some(r) if !r.is_empty() => r,
            _ => { messages_skipped += 1; continue; }
        };
        let created_at = match msg["created_at"].as_str() {
            Some(c) if !c.is_empty() => c,
            _ => { messages_skipped += 1; continue; }
        };
        let content = match msg["content"].as_str() {
            Some(c) => c,
            _ => { messages_skipped += 1; continue; }
        };

        let exists: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM conversation_messages
             WHERE session_id = ? AND created_at = ? AND role = ?"
        )
        .bind(session_id)
        .bind(created_at)
        .bind(role)
        .fetch_one(&pool)
        .await
        .unwrap_or(false);

        if exists { messages_skipped += 1; continue; }

        let result = sqlx::query(
            "INSERT INTO conversation_messages (
                session_id, user_id, role, content, created_at,
                model_backend, containment_mode, entry_hash, prev_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(session_id)
        .bind(msg["user_id"].as_str().unwrap_or(&user_id))
        .bind(role)
        .bind(content)
        .bind(created_at)
        .bind(msg["model_backend"].as_str())
        .bind(msg["containment_mode"].as_str())
        .bind(msg["entry_hash"].as_str())
        .bind(msg["prev_hash"].as_str())
        .execute(&pool)
        .await;

        match result {
            Ok(_) => messages_inserted += 1,
            Err(e) => eprintln!("[Backup] Message insert failed: {}", e),
        }
    }

    pool.close().await;

    // Push to peers: clear their tombstones then sync the restored memories.
    if inserted > 0 {
        let service_opt = {
            let guard = crate::ZYNKSYNC_SERVICE.lock().await;
            guard.as_ref().map(std::sync::Arc::clone)
        };
        if let Some(service) = service_opt {
            let cleared = service.clear_tombstones_on_peers(&restored_hashes).await;
            println!("[Backup] Cleared tombstones on {} peer(s)", cleared);

            let peers = service.get_peers().await;
            for peer in peers.iter().filter(|p| p.paired) {
                match service.sync_bidirectional(&peer.device_id, &user_id).await {
                    Ok(r) => println!("[Backup] Synced {} memories to {}", r.memories_sent, r.peer_device_name),
                    Err(e) => eprintln!("[Backup] Sync to {} failed: {}", peer.device_name, e),
                }
            }
        }
    }

    println!(
        "[Backup] Restored {} memories ({} skipped), {} sessions ({} skipped), {} messages ({} skipped) for user {}",
        inserted, skipped, sessions_inserted, sessions_skipped, messages_inserted, messages_skipped, user_id
    );
    Ok(serde_json::json!({
        "success": true,
        "inserted": inserted,
        "skipped": skipped,
        "sessions_inserted": sessions_inserted,
        "sessions_skipped": sessions_skipped,
        "messages_inserted": messages_inserted,
        "messages_skipped": messages_skipped,
        "message": format!(
            "Restored {} memories ({} existed), {} conversations ({} messages)",
            inserted, skipped, sessions_inserted, messages_inserted
        )
    }))
}

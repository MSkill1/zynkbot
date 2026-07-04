/// One-Time Sync Code Manager
///
/// Generates temporary 6-digit codes for secure device syncing.
/// Codes expire after 5 minutes and are one-time use only.
///
/// Uses PostgreSQL database for persistence, enabling cross-instance sync code verification.
use serde::{Deserialize, Serialize};
use sqlx::{SqlitePool, Row};
use chrono::{DateTime, Utc, Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncCodeInfo {
    pub user_id: String,
    pub device_id: String,
    pub expires_in: i64,  // seconds remaining
    pub used: bool,
}

/// Generate a unique 6-digit sync code
pub async fn generate_sync_code(
    user_id: &str,
    device_id: &str,
    pool: &SqlitePool,
) -> Result<String, String> {
    // Generate unique 6-digit code
    let code = loop {
        // Use rand::random() which is Send-safe (no ThreadRng needed)
        let random_num = rand::random::<u32>() % 900000 + 100000;
        let candidate = format!("{:06}", random_num);

        // Check if code already exists
        let exists = sqlx::query("SELECT code FROM user_sync_codes WHERE code = ?")
            .bind(&candidate)
            .fetch_optional(pool)
            .await
            .map_err(|e| format!("Database query failed: {}", e))?;

        if exists.is_none() {
            break candidate;
        }
    };

    // Calculate expiration (5 minutes from now)
    let expires_at = Utc::now() + Duration::minutes(5);

    // Store code in database
    sqlx::query(
        "INSERT INTO user_sync_codes (code, user_id, device_id, expires_at)
         VALUES (?, ?, ?, ?)"
    )
    .bind(&code)
    .bind(user_id)
    .bind(device_id)
    .bind(expires_at)
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to store sync code: {}", e))?;

    // Clean up expired codes
    cleanup_expired_codes(pool).await?;

    println!("[SyncCode] Generated code {} for user {}... (expires in 5 min)",
             code, &user_id[..8.min(user_id.len())]);

    Ok(code)
}

/// Verify sync code and return user_id if valid (consumes the code)
#[allow(dead_code)]
pub async fn verify_sync_code(
    code: &str,
    pool: &SqlitePool,
) -> Result<String, String> {
    // Get code data
    let row = sqlx::query(
        "SELECT user_id, expires_at, used
         FROM user_sync_codes
         WHERE code = ?"
    )
    .bind(code)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("Database query failed: {}", e))?;

    let row = match row {
        Some(r) => r,
        None => return Err("Invalid code".to_string()),
    };

    let user_id: String = row.get("user_id");
    let expires_at: DateTime<Utc> = row.get("expires_at");
    let used: bool = row.get("used");

    // Check if expired
    if Utc::now() > expires_at {
        // Delete expired code
        sqlx::query("DELETE FROM user_sync_codes WHERE code = ?")
            .bind(code)
            .execute(pool)
            .await
            .map_err(|e| format!("Failed to delete expired code: {}", e))?;

        return Err("Code expired".to_string());
    }

    // Check if already used
    if used {
        return Err("Code already used".to_string());
    }

    // Mark as used
    sqlx::query(
        "UPDATE user_sync_codes
         SET used = TRUE, used_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
         WHERE code = ?"
    )
    .bind(code)
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to mark code as used: {}", e))?;

    println!("[SyncCode] Verified and consumed code {} for user {}...",
             code, &user_id[..8.min(user_id.len())]);

    Ok(user_id)
}

/// Get information about a sync code without consuming it
#[allow(dead_code)]
pub async fn get_code_info(
    code: &str,
    pool: &SqlitePool,
) -> Result<SyncCodeInfo, String> {
    let row = sqlx::query(
        "SELECT user_id, device_id, expires_at, used
         FROM user_sync_codes
         WHERE code = ?"
    )
    .bind(code)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("Database query failed: {}", e))?;

    let row = match row {
        Some(r) => r,
        None => return Err("Invalid code".to_string()),
    };

    let user_id: String = row.get("user_id");
    let device_id: String = row.get("device_id");
    let expires_at: DateTime<Utc> = row.get("expires_at");
    let used: bool = row.get("used");

    // Check if expired
    if Utc::now() > expires_at {
        // Delete expired code
        sqlx::query("DELETE FROM user_sync_codes WHERE code = ?")
            .bind(code)
            .execute(pool)
            .await
            .map_err(|e| format!("Failed to delete expired code: {}", e))?;

        return Err("Code expired".to_string());
    }

    let expires_in = (expires_at - Utc::now()).num_seconds();

    Ok(SyncCodeInfo {
        user_id,
        device_id,
        expires_in,
        used,
    })
}

/// Remove expired codes from database.
/// Replaces the PostgreSQL `cleanup_expired_sync_codes()` stored function that does not
/// exist in SQLite. Plain DELETE — same semantics, just inlined.
pub async fn cleanup_expired_codes(pool: &SqlitePool) -> Result<(), String> {
    sqlx::query("DELETE FROM user_sync_codes WHERE expires_at < datetime('now')")
        .execute(pool)
        .await
        .map_err(|e| format!("Failed to cleanup expired codes: {}", e))?;

    Ok(())
}

/// Get number of active (non-expired, non-used) codes
#[allow(dead_code)]
pub async fn get_active_codes_count(pool: &SqlitePool) -> Result<i64, String> {
    cleanup_expired_codes(pool).await?;

    let row = sqlx::query(
        "SELECT COUNT(*) as count
         FROM user_sync_codes
         WHERE NOT used AND expires_at > datetime('now')"
    )
    .fetch_one(pool)
    .await
    .map_err(|e| format!("Database query failed: {}", e))?;

    let count: i64 = row.get("count");
    Ok(count)
}

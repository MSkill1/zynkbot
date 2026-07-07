use crate::user_identity;
use crate::sync_codes;

#[tauri::command]
pub async fn generate_sync_code() -> Result<serde_json::Value, String> {
    let user_id = user_identity::get_user_id()?;
    let device_id = user_identity::get_device_id()?;

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    let code = sync_codes::generate_sync_code(&user_id, &device_id, &pool).await?;

    Ok(serde_json::json!({
        "code": code,
        "expires_in": 300  // 5 minutes
    }))
}

#[tauri::command]
pub async fn verify_sync_code_info(code: String) -> Result<sync_codes::SyncCodeInfo, String> {
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    sync_codes::get_code_info(&code, &pool).await
}

#[tauri::command]
pub async fn sync_with_code(code: String, device_ip: String) -> Result<serde_json::Value, String> {
    println!("[SyncCode] Device B: Verifying code {} with remote device {}", code, device_ip);

    // Step 1: Verify code on REMOTE device (Device A) via HTTPS (TOFU — cert not yet pinned).
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("Failed to build sync client: {}", e))?;
    let verify_url = format!("https://{}:57963/api/identity/verify-sync-code", device_ip);

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

    println!("[SyncCode] Device B: Code verified! Remote user_id: {}...", &verify_data.user_id[..8]);

    // Step 2: Set Device B's user_id to match Device A
    user_identity::set_user_id(&verify_data.user_id)?;
    let identity = user_identity::get_identity()?;

    println!("[SyncCode] Device B: Linked to user {}...", &identity.user_id[..8]);

    // Step 3: Notify remote device to consume code and establish pairing
    let consume_url = format!("https://{}:57963/api/identity/consume-sync-code", device_ip);
    let _consume_response = client
        .post(&consume_url)
        .json(&serde_json::json!({
            "code": code,
            "remote_device_id": identity.device_id
        }))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await;
    // Ignore errors on consume - pairing is already established via matching user_ids

    println!("[SyncCode] Device B: Pairing complete with remote device");

    Ok(serde_json::json!({
        "success": true,
        "user_id": identity.user_id,
        "device_id": identity.device_id,
        "remote_device_id": verify_data.device_id,
        "message": "Successfully linked to remote device"
    }))
}

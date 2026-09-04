use crate::user_identity;

#[tauri::command]
pub async fn get_user_identity() -> Result<user_identity::UserIdentity, String> {
    user_identity::get_identity()
}

#[tauri::command]
pub async fn set_user_identity(user_id: String) -> Result<(), String> {
    user_identity::set_user_id(&user_id)
}

#[tauri::command]
pub async fn reset_user_identity() -> Result<user_identity::UserIdentity, String> {
    let (new_user_id, new_device_id) = user_identity::reset_all_identity()?;
    println!("[Identity] Reset complete - New user_id: {}, New device_id: {}", new_user_id, new_device_id);
    user_identity::get_identity()
}

/// Migrate all memories from old user_id to new user_id.
/// Used during identity adoption to preserve memories instead of deleting them.
#[tauri::command]
pub async fn migrate_user_memories(old_user_id: String, new_user_id: String) -> Result<i64, String> {
    println!("[Memory] Migrating memories from {} to {}", old_user_id, new_user_id);

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    let result = sqlx::query("UPDATE memories SET user_id = ? WHERE user_id = ?")
        .bind(&new_user_id)
        .bind(&old_user_id)
        .execute(&pool)
        .await
        .map_err(|e| format!("Failed to migrate memories: {}", e))?;

    let migrated_count = result.rows_affected() as i64;
    println!("[Memory] ✓ Migrated {} memories to new user_id", migrated_count);
    Ok(migrated_count)
}

/// The name this device shows on the sync network — the user's custom name if one has
/// been set (via the pairing-time prompt or the sync panel), else the hostname/
/// Android-XXXX default.
#[tauri::command]
pub async fn get_device_name() -> String {
    crate::user_identity::get_device_name()
}

/// Whether the user has ever set a custom device name. Used to decide whether to show
/// the "what should this device be called?" prompt the first time the user pairs.
#[tauri::command]
pub async fn has_custom_device_name() -> bool {
    crate::user_identity::has_custom_device_name()
}

/// Set this device's name. Takes effect immediately for any new pairing, and for
/// already-paired peers at their next sync contact (see check_sync_authorized's
/// x-device-name handling in zynksync.rs) — no unpair/re-pair needed. Also rebuilds the
/// running sync client so its own default headers pick up the new name right away,
/// rather than waiting for the next pairing or app restart.
/// Record the conversation thread the app has on screen. Hands-free ("Hey Zynk")
/// questions answered natively continue this thread — with its history — rather than
/// each starting a new one. Called whenever the app's session changes.
#[tauri::command]
pub async fn set_current_session(session_id: String) -> Result<(), String> {
    crate::user_identity::set_current_session_id(&session_id)
}

#[tauri::command]
pub async fn get_current_session() -> Option<String> {
    crate::user_identity::get_current_session_id()
}

#[tauri::command]
pub async fn set_device_name(name: String) -> Result<(), String> {
    crate::user_identity::set_device_name(&name)?;
    let global_service = crate::ZYNKSYNC_SERVICE.lock().await;
    if let Some(service) = global_service.as_ref() {
        service.rebuild_http_client().await.ok();
    }
    Ok(())
}

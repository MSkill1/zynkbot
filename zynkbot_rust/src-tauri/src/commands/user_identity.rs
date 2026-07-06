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

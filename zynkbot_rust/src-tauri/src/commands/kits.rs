//! Kits: the System Controls section, the buttons above the message box and the list window talk to these.
//! The mechanism itself is in `crate::kits`.

use serde::Serialize;

async fn pool() -> Result<sqlx::SqlitePool, String> {
    sqlx::SqlitePool::connect(&crate::db::get_db_url()).await.map_err(|e| format!("Database: {}", e))
}

#[derive(Serialize)]
pub struct KitView {
    #[serde(flatten)]
    pub kit: crate::kits::Kit,
    pub enabled: bool,
    pub answers: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize)]
pub struct KitsView {
    pub kits: Vec<KitView>,
    /// Name and icon only; nothing behind them.
    pub future: Vec<(String, String)>,
}

#[tauri::command]
pub async fn list_kits() -> Result<KitsView, String> {
    let pool = pool().await?;
    let mut kits = Vec::new();
    for k in crate::kits::all() {
        kits.push(KitView {
            kit: k.clone(),
            enabled: crate::kits::is_enabled(&pool, &k.id).await,
            answers: crate::kits::answers(&pool, &k.id).await,
        });
    }
    pool.close().await;
    Ok(KitsView { kits, future: crate::kits::FUTURE_KITS.iter().map(|(n, i)| (n.to_string(), i.to_string())).collect() })
}

/// Switch a kit on or off; the opening answers travel with "on" (and may be edited later).
#[tauri::command]
pub async fn set_kit_enabled(kit_id: String, enabled: bool, answers: Option<serde_json::Map<String, serde_json::Value>>) -> Result<(), String> {
    if crate::kits::get(&kit_id).is_none() { return Err("unknown kit".into()); }
    let pool = pool().await?;
    crate::kits::set_enabled(&pool, &kit_id, enabled).await?;
    if let Some(a) = answers { crate::kits::set_answers(&pool, &kit_id, &a).await?; }
    pool.close().await;
    println!("[Kits] {} {}", kit_id, if enabled { "on" } else { "off" });
    Ok(())
}

#[tauri::command]
pub async fn kit_items(kit_id: String, user_id: String) -> Result<Vec<crate::kits::KitItem>, String> {
    let pool = pool().await?;
    let items = crate::kits::list_items(&pool, &user_id, &kit_id).await;
    pool.close().await;
    Ok(items)
}

/// An item typed into the list window.
#[tauri::command]
pub async fn kit_add_item(kit_id: String, user_id: String, name: String, section: Option<String>, quantity: Option<String>) -> Result<i32, String> {
    let kit = crate::kits::get(&kit_id).ok_or("unknown kit")?;
    let pool = pool().await?;
    let r = crate::kits::add_item(&pool, &user_id, kit, &name, section.as_deref(), quantity.as_deref(), None).await;
    pool.close().await;
    r
}

/// Tick (bought) or untick (needed).
#[tauri::command]
pub async fn kit_set_item_state(memory_id: i32, state: String) -> Result<(), String> {
    let pool = pool().await?;
    let r = crate::kits::set_state(&pool, memory_id, &state).await;
    pool.close().await;
    r
}

#[tauri::command]
pub async fn kit_remove_item(memory_id: i32) -> Result<(), String> {
    let pool = pool().await?;
    let r = crate::kits::remove_item(&pool, memory_id).await;
    pool.close().await;
    r
}

/// "Forget everything from this kit": returns how many memories went.
#[tauri::command]
pub async fn forget_kit(kit_id: String, user_id: String) -> Result<usize, String> {
    let pool = pool().await?;
    let r = crate::kits::forget(&pool, &user_id, &kit_id).await;
    pool.close().await;
    r
}

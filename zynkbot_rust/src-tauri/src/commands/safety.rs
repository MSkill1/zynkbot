use crate::{containment, safety_classifier};

/// Check whether text is allowed under the active containment mode
#[tauri::command]
pub async fn check_containment(text: String, mode: String) -> Result<Option<String>, String> {
    println!("[ContainmentCheck] Checking text with mode: {}", mode);

    let layer = containment::ContainmentLayer::new(&mode)?;
    let result = layer.enforce(&text).await;

    if result.is_none() {
        println!("[ContainmentCheck] ✅ Content allowed");
    } else {
        println!("[ContainmentCheck] ⚠️ Content flagged");
    }

    Ok(result)
}

/// Set the containment mode and remember it. The page holds the mode and sends it with
/// every request; this saves the choice in `app_settings` so the next start opens in it.
/// Until 2026-11-10 nothing was saved and every start went back to Sovereign, Child and
/// HIPAA included.
#[tauri::command]
pub async fn set_containment_mode(mode: String) -> Result<String, String> {
    containment::ContainmentLayer::new(&mode)
        .map_err(|e| format!("Invalid containment mode: {}", e))?;

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url()).await
        .map_err(|e| format!("Database connection failed: {}", e))?;
    let saved = save_containment_mode(&pool, &mode).await;
    pool.close().await;
    saved?;

    Ok(format!("Containment mode set to: {}", mode))
}

/// The mode saved by `set_containment_mode`; Sovereign when none was saved or the saved
/// word is not a mode this build knows.
#[tauri::command]
pub async fn get_containment_mode() -> Result<String, String> {
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url()).await
        .map_err(|e| format!("Database connection failed: {}", e))?;
    let mode = saved_containment_mode(&pool).await;
    pool.close().await;
    Ok(mode)
}

pub(crate) async fn save_containment_mode(pool: &sqlx::SqlitePool, mode: &str) -> Result<(), String> {
    sqlx::query("INSERT INTO app_settings (key, value, updated_at) VALUES ('containment_mode', ?, datetime('now')) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at")
        .bind(mode.to_lowercase())
        .execute(pool).await
        .map(|_| ())
        .map_err(|e| format!("Failed to save the containment mode: {}", e))
}

pub(crate) async fn saved_containment_mode(pool: &sqlx::SqlitePool) -> String {
    let saved: Option<String> = sqlx::query_scalar("SELECT value FROM app_settings WHERE key = 'containment_mode'")
        .fetch_optional(pool).await.ok().flatten();
    match saved {
        Some(m) if containment::ContainmentLayer::new(&m).is_ok() => m,
        _ => "sovereign".to_string(),
    }
}

/// Initialize the Candle-based safety classifier on app startup
#[tauri::command]
pub async fn initialize_safety_models() -> Result<String, String> {
    println!("[Safety] Initializing Candle-based safety classifier...");

    safety_classifier::initialize()
        .map_err(|e| format!("Failed to initialize: {}", e))?;

    Ok("Safety models initialized successfully".to_string())
}

/// Check whether a question message contains memory-worthy information
#[tauri::command]
pub async fn check_question_worthiness(question: String) -> Result<serde_json::Value, String> {
    use crate::question_extractor::QuestionMemoryExtractor;

    println!("[Rust Question] Checking question worthiness...");

    let extractor = QuestionMemoryExtractor::new();
    let result = extractor.contains_memory_worthy_info(&question);

    Ok(serde_json::json!({
        "has_info": result.has_info,
        "confidence": result.confidence,
        "signals": result.signals,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_containment_mode_is_kept() {
        let pool = sqlx::sqlite::SqlitePoolOptions::new().max_connections(1).connect("sqlite::memory:").await.unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        // Nothing saved: Sovereign.
        assert_eq!(saved_containment_mode(&pool).await, "sovereign");
        save_containment_mode(&pool, "child").await.unwrap();
        assert_eq!(saved_containment_mode(&pool).await, "child");
        save_containment_mode(&pool, "HIPAA").await.unwrap();
        assert_eq!(saved_containment_mode(&pool).await, "hipaa");
        // A word this build does not know is not handed back as a mode.
        sqlx::query("UPDATE app_settings SET value = 'nonsense' WHERE key = 'containment_mode'").execute(&pool).await.unwrap();
        assert_eq!(saved_containment_mode(&pool).await, "sovereign");
    }
}

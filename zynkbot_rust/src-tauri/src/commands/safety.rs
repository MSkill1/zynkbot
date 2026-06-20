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

/// Validate that a containment mode string is recognised
#[tauri::command]
pub async fn set_containment_mode(mode: String) -> Result<String, String> {
    containment::ContainmentLayer::new(&mode)
        .map_err(|e| format!("Invalid containment mode: {}", e))?;

    Ok(format!("Containment mode set to: {}", mode))
}

/// Return the default containment mode (stateless for now)
#[tauri::command]
pub async fn get_containment_mode() -> Result<String, String> {
    Ok("guardian".to_string())
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

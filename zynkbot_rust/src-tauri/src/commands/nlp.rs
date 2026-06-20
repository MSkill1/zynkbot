use crate::nlp_enhancer::NLPEnhancer;
use crate::question_extractor::QuestionMemoryExtractor;
use crate::conversation_engine::ConversationEngine;

/// Extract named entities using Candle BERT NER (Pure Rust ML)
#[tauri::command]
pub async fn extract_entities(content: String) -> Result<serde_json::Value, String> {
    println!("[Rust NER] Extracting entities with Candle BERT...");

    let enhancer = NLPEnhancer::new();
    let entities = enhancer.extract_entities(&content);

    let entities_json: Vec<serde_json::Value> = entities.iter().map(|e| {
        serde_json::json!({
            "word": e.word,
            "label": e.label,
            "score": e.score,
            "start": e.start,
            "end": e.end,
        })
    }).collect();

    Ok(serde_json::json!(entities_json))
}

/// Extract facts from a user question
#[tauri::command]
pub async fn extract_facts_from_question(question: String) -> Result<serde_json::Value, String> {
    println!("[Rust Question] Extracting facts from question...");

    let extractor = QuestionMemoryExtractor::new();
    let facts = extractor.extract_facts(&question);

    let json = serde_json::to_value(facts)
        .map_err(|e| format!("Failed to serialize facts: {}", e))?;

    Ok(json)
}

/// Check if content is memory-worthy
#[tauri::command]
pub async fn check_memory_worthiness(content: String) -> Result<bool, String> {
    let engine = ConversationEngine::new();
    let is_worthy = engine.is_memory_worthy(&content);

    println!("[Rust ConversationEngine] Memory worthy: {}", is_worthy);

    Ok(is_worthy)
}

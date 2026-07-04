// Zynkbot Tauri Backend
// Uses Candle for ML (pure Rust, no ONNX conflicts)

// Module declarations
pub mod commands;           // Tauri command handlers (extracted from lib.rs)
pub mod safety_classifier;  // TinyBERT toxicity classifier (Candle-based)
mod containment;  // Safety enforcement using toxic-bert + OpenAI API for Child mode
mod nlp_enhancer;
mod memory;
mod question_extractor;
mod llm_fact_extractor;  // LLM-based fact extraction (uses same LLM as conversation)
mod llm;
mod conversation_engine;
mod zynksync;  // Phase 9: Device-to-device memory synchronization
mod user_identity;  // User and device identity management
mod sync_codes;  // One-time sync codes for device pairing
mod zchat;  // Device-to-device messaging
mod zynklink;  // Device-to-device file sharing
mod web_search;  // DuckDuckGo web search
mod knowledge_base;  // External reference document system
mod kb_rag;  // Knowledge Base RAG: Document chunking, indexing, semantic search
mod conversation_history;  // Persistent conversation log with full-text search
mod db;  // Database connection pool
mod tls; // TLS certificate management for ZynkSync/ZynkLink/ZChat

use serde::{Deserialize, Serialize};
use tauri::Emitter;
// use chrono::Utc;  // Unused - commented out

/// Normalize common voice-transcription misspellings of the brand name.
/// Whisper and other STT engines frequently transcribe "Zynkbot" as Zincbot,
/// Zinkbot, Sinkbot, Zyncbot, or "Zinc bot". Fix these before any processing.
fn normalize_brand_names(text: String) -> String {
    // Case-insensitive replacement preserving the canonical spelling
    let variants = [
        "Zinc bot", "zinc bot", "Zink bot", "zink bot",
        "Zincbot", "zincbot", "Zinkbot", "zinkbot",
        "Sinkbot", "sinkbot", "Zyncbot", "zyncbot",
        "ZincBot", "ZinkBot", "SinkBot", "ZyncBot",
    ];
    let mut result = text;
    for variant in &variants {
        // Preserve leading case of the original word
        let replacement = if variant.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
            "Zynkbot"
        } else {
            "zynkbot"
        };
        result = result.replace(variant, replacement);
    }
    result
}

fn blob_to_f32(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Extract full name, preferred name, and age from a free-text onboarding response.
/// Tries specific patterns first, falls back to first two-word capitalized sequence.
pub(crate) fn extract_names_from_response(response: &str) -> (Option<String>, Option<String>, Option<u32>) {
    use regex::Regex;

    // Preferred name: look for explicit "go by", "call me", etc.
    let preferred_patterns = [
        r"(?:go by|call me|prefer(?:red)?(?:\s+name)?(?:\s+is)?|nickname(?:\s+is)?|known as)\s+([A-Z][a-z]{1,})",
        r"(?i)(?:go by|call me|prefer(?:red)?(?:\s+name)?(?:\s+is)?|nickname(?:\s+is)?|known as)\s+([A-Za-z]{2,})",
        r"(?i)I(?:'m| am)\s+([A-Z][a-z]{2,})\b",
    ];

    let mut preferred_name: Option<String> = None;
    for pattern in &preferred_patterns {
        if let Ok(re) = Regex::new(pattern) {
            if let Some(cap) = re.captures(response) {
                if let Some(m) = cap.get(1) {
                    let word = m.as_str().trim().to_string();
                    let name = {
                        let mut c = word.chars();
                        match c.next() {
                            None => String::new(),
                            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                        }
                    };
                    if !name.is_empty() {
                        preferred_name = Some(name);
                        break;
                    }
                }
            }
        }
    }

    // Full name: explicit "full name is X Y" first, then any two+ consecutive Title Case words
    let full_name_patterns = [
        r"(?i)(?:full name is|full name:)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        r"(?i)(?:my name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
    ];

    let mut full_name: Option<String> = None;
    for pattern in &full_name_patterns {
        if let Ok(re) = Regex::new(pattern) {
            if let Some(cap) = re.captures(response) {
                if let Some(m) = cap.get(1) {
                    full_name = Some(m.as_str().trim().to_string());
                    break;
                }
            }
        }
    }

    if preferred_name.is_none() {
        if let Some(ref name) = full_name {
            preferred_name = name.split_whitespace().next().map(|s| s.to_string());
        }
    }

    // Age: look for a 2-digit number between 18 and 99 (preceded by "I'm", "am", or standalone)
    let age = if let Ok(re) = Regex::new(r"(?i)(?:I(?:'m| am)|age(?:\s+is)?)?\s*\b(\d{2})\b") {
        re.captures_iter(response)
            .filter_map(|cap| cap.get(1)?.as_str().parse::<u32>().ok())
            .find(|&n| n >= 18 && n <= 99)
    } else {
        None
    };

    (full_name, preferred_name, age)
}

/// Read the user profile JSON file, returning an empty object if it doesn't exist.
pub(crate) fn read_user_profile() -> serde_json::Value {
    let path = crate::db::get_user_profile_path();
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}))
}

/// Write the user profile JSON file.
pub(crate) fn write_user_profile(profile: &serde_json::Value) -> Result<(), String> {
    let path = crate::db::get_user_profile_path();
    let data = serde_json::to_string_pretty(profile)
        .map_err(|e| format!("Failed to serialize profile: {}", e))?;
    std::fs::write(&path, data)
        .map_err(|e| format!("Failed to write user profile: {}", e))
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Memory {
    id: i32,
    title: Option<String>,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<String>,
    namespace: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_syncable: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_shareable: Option<bool>,
    created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    updated_at: Option<String>,  // Made optional - Flask doesn't always return this
    #[serde(skip_serializing_if = "Option::is_none")]
    similarity: Option<f64>,
    // New NLP feature fields
    #[serde(skip_serializing_if = "Option::is_none")]
    event_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    event_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    link_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_ephemeral: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    entities_detected: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    original_text: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct ReplyResponse {
    reply_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    recalled_memories: Option<Vec<Memory>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    containment_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    blocked: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    web_search_needed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    web_search_query: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    original_query: Option<String>,
}

// ============================================================================
// TAURI COMMANDS
// (Model/API-key commands live in commands/models.rs)
// ============================================================================

// get_models and open_models_folder → commands/models.rs

/// Conversation history item (user or assistant message)
#[derive(Serialize, Deserialize)]
struct ConversationTurn {
    role: String,
    content: String,
}

/// Extract only factual declarative statements from a message
/// Filters out questions, instructions, and conversational filler
/// Example: "Don't search, just tell me what you think. I am traveling to Japan" → "I am traveling to Japan"
#[allow(dead_code)]
fn extract_factual_statements(message: &str) -> String {
    // Filter out casual acknowledgments (ok, thanks, lol, etc.)
    let trimmed_lower = message.trim().to_lowercase();
    let acknowledgments = [
        "ok", "okay", "thanks", "thank you", "got it", "sure",
        "yeah", "yep", "nope", "lol", "haha", "cool", "nice",
        "great", "alright", "fine", "k"
    ];
    if acknowledgments.contains(&trimmed_lower.as_str()) {
        return String::new();
    }

    // Explicit "Remember: ..." prefix — user is directly commanding storage.
    // Bypass all factual-pattern filtering and return the content after the colon.
    if trimmed_lower.starts_with("remember:") {
        if let Some(colon_pos) = message.find(':') {
            let content = message[colon_pos + 1..].trim().to_string();
            if !content.is_empty() {
                return content;
            }
        }
    }

    // Instruction/command patterns to filter out
    let instruction_patterns = [
        "don't", "please", "tell me", "let me know", "can you", "could you",
        "would you", "will you", "show me", "give me", "help me",
        "do a", "do not", "search", "think about", "what do you think",
        "remember to", "remind me", "don't forget", "make sure to",  // Commands
        "just do", "just tell", "just show", "just give",  // Specific "just" commands only
    ];

    // Question words that start interrogative sentences — never factual on their own
    let question_starters = [
        "what ", "how ", "why ", "when ", "where ", "who ", "which ", "whose ", "whom ",
        "is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ",
        "would ", "should ", "will ", "has ", "have ", "had ",
    ];

    // Helper function to check if a clause is factual
    let is_factual_clause = |clause: &str| -> bool {
        let lower = clause.trim().to_lowercase();

        if lower.is_empty() {
            return false;
        }

        // Reject questions even if they contain "is", "was", etc.
        let is_question = question_starters.iter().any(|q| lower.starts_with(q));
        if is_question {
            return false;
        }

        // First-person statements ("I ...", "My ...") are always factual — the user
        // is describing themselves, not commanding the AI. Check this before instruction
        // filtering so "I don't know what to do" isn't rejected by the "don't" pattern.
        let is_first_person = lower.starts_with("i ")
            || lower.starts_with("i'")
            || lower.starts_with("my ");
        if is_first_person {
            return true;
        }

        // Filter out instruction patterns for non-first-person clauses
        let has_instruction = instruction_patterns.iter().any(|pattern| lower.contains(pattern));
        if has_instruction {
            return false;
        }

        // Keep other declarative statements
        lower.starts_with("the ")
            || lower.starts_with("we ")
            || lower.starts_with("they ")
            || lower.contains(" am ")
            || lower.contains(" is ")
            || lower.contains(" was ")
            || lower.contains(" were ")
            || lower.contains(" have ")
            || lower.contains(" had ")
    };

    // Split by sentence-ending punctuation
    let sentences: Vec<&str> = message
        .split(&['.', '!', '?'][..])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let mut factual_clauses = Vec::new();

    for sentence in sentences {
        // Split only on contrasting conjunctions (but, however)
        // Don't split on "and" as it often continues the same statement
        // Example: "I hated General Patton and what he did in World War 2" should stay together
        let normalized = sentence
            .replace(", but ", "|||")
            .replace(" but ", "|||")
            .replace(", however ", "|||")
            .replace(" however ", "|||");

        let clauses: Vec<&str> = normalized.split("|||").collect();

        // Filter each clause independently
        for clause in clauses {
            if is_factual_clause(clause) {
                factual_clauses.push(clause.trim().to_string());
            }
        }
    }

    // Join remaining factual clauses
    let result = factual_clauses.join(". ");

    // Filter very short/meaningless extractions (< 10 chars)
    if result.len() < 10 {
        return String::new();
    }

    // Return factual statements, or original if nothing extracted
    if result.is_empty() {
        message.to_string()
    } else {
        result
    }
}

fn generate_title_from_content(content: &str) -> String {
    let content = content.trim();

    // Remove leading "I" or "My" for more natural titles
    let normalized = if content.to_lowercase().starts_with("i ") {
        &content[2..]
    } else if content.to_lowercase().starts_with("my ") {
        &content[3..]
    } else {
        content
    };

    // Take first sentence or 50 chars, whichever is shorter
    let first_sentence = normalized
        .split(['.', '!', '?'])
        .next()
        .unwrap_or(normalized)
        .trim();

    if first_sentence.len() <= 50 {
        // Capitalize first letter
        let mut chars = first_sentence.chars();
        match chars.next() {
            None => String::new(),
            Some(f) => f.to_uppercase().chain(chars).collect(),
        }
    } else {
        // Truncate at word boundary
        let truncated = &first_sentence[..50];
        let last_space = truncated.rfind(' ').unwrap_or(50);
        let mut result = truncated[..last_space].trim().to_string();

        // Capitalize first letter
        if let Some(first_char) = result.chars().next() {
            result = first_char.to_uppercase().chain(result.chars().skip(1)).collect();
        }

        result
    }
}

// get_api_keys, set_api_key, remove_api_key → commands/models.rs

// ============================================================================
// MEMORY DECISION HELPERS (LLM-based)
// ============================================================================

/// Local Call 2: classify relationships between an extracted fact and similar existing memories.
/// Fires in the background only when MEMORY_EXTRACT produced a fact.
/// Returns (optional_title, relationship_classifications)
async fn ask_llm_for_relationships(
    extracted_fact: &str,
    similar_memories: &[(i32, String, Option<String>, f32)],
    backend: &str,
    local_session: Option<llm::local_models::LocalModelSession>,
) -> Result<(Option<String>, Vec<RelationshipClassification>), String> {
    let similar_memories_text = if similar_memories.is_empty() {
        "(none)".to_string()
    } else {
        similar_memories.iter()
            .map(|(id, content, title, sim)| {
                let title_str = title.as_ref().map(|t| format!(" ({})", t)).unwrap_or_default();
                format!("Memory #{}{} (similarity: {:.1}%)\n{}", id, title_str, sim * 100.0, content)
            })
            .collect::<Vec<_>>()
            .join("\n\n---\n\n")
    };

    let prompt = format!(
        r#"You are a memory relationship classifier. A new personal fact was extracted from a user message. Generate a title for it and classify how it relates to similar existing memories.

NEW FACT:
{}

SIMILAR EXISTING MEMORIES:
{}

RELATIONSHIP TYPES:
- "contradicts": Same attribute, different values — or opposite claims on the same topic
- "supports": Reinforces or agrees with the existing memory
- "elaborates": Adds detail or context to the existing memory
- "caused_by": This fact describes a cause of the existing memory's effect
- "reminds_of": Loosely related or tangentially connected
- "none": No meaningful relationship

⚠️ High semantic similarity does NOT mean agreement — opposite claims = contradiction.

Return ONLY valid JSON starting with {{:
{{
  "title": "Concise descriptive title for the new fact (max 50 chars)",
  "relationships": [
    {{
      "memory_id": 123,
      "relationship_type": "contradicts|supports|elaborates|caused_by|reminds_of|none",
      "reason": "Brief explanation",
      "confidence": 0.95
    }}
  ]
}}"#,
        extracted_fact, similar_memories_text
    );

    let response = if backend.contains("anthropic") {
        call_anthropic_for_memory_decision(&prompt).await?
    } else if backend.contains("openai") {
        call_openai_for_memory_decision(&prompt).await?
    } else if backend.contains("xai") {
        call_xai_for_memory_decision(&prompt).await?
    } else if let Some(sess) = local_session {
        // Paired-call path: use the session loaded for the main conversation call.
        // No second disk read — the model is already in memory.
        println!("[Memory Relations] Using pre-loaded model session for Call 2 (no second disk load)");
        let prompt_clone = prompt.clone();
        tokio::task::spawn_blocking(move || {
            let messages = vec![llm::Message {
                role: "user".to_string(),
                content: prompt_clone,
            }];
            sess.generate(messages, Some(4096), Some(0.1), Some(RELATIONSHIP_SCHEMA))
        })
        .await
        .map_err(|e| format!("Task panicked: {}", e))?
        .map_err(|e| e.to_string())?
        .content
    } else {
        // Fresh load fallback (session unavailable or API fallback path)
        call_local_for_memory_decision(&prompt, backend, Some(RELATIONSHIP_SCHEMA)).await?
    };

    // Extract JSON from response (handle markdown code blocks)
    let json_str = if response.contains("```json") {
        response.split("```json").nth(1).unwrap_or(&response)
            .split("```").next().unwrap_or(&response).trim()
    } else if response.contains("```") {
        response.split("```").nth(1).unwrap_or(&response).trim()
    } else if let Some(start) = response.find('{') {
        &response[start..]
    } else {
        response.trim()
    };

    #[derive(serde::Deserialize)]
    struct RelationshipResult {
        title: Option<String>,
        relationships: Option<Vec<RelationshipClassification>>,
    }

    match serde_json::from_str::<RelationshipResult>(json_str) {
        Ok(result) => {
            let rels = result.relationships.unwrap_or_default()
                .into_iter()
                .filter(|r| r.relationship_type != "none" && r.memory_id > 0)
                .collect();
            Ok((result.title, rels))
        }
        Err(e) => {
            println!("[Memory Relations] Failed to parse JSON: {} — raw: {}", e, &json_str[..json_str.len().min(120)]);
            Err(format!("Failed to parse relationship JSON: {}", e))
        }
    }
}

/// Helper function to ask LLM whether to create a memory and generate a title
/// Returns (should_remember, optional_title)
#[allow(dead_code)]
async fn ask_llm_about_memory(
    message: &str,
    conversation_history: &str,
    backend: &str,
) -> Result<(bool, Option<String>), String> {
    // Build prompt
    let prompt = format!(
        r#"You are a memory decision system. Analyze the user's message and decide if it should be stored as a long-term memory.

CONVERSATION CONTEXT:
{}

USER MESSAGE:
{}

TASK:
1. Decide if this message contains information worth remembering long-term
2. If yes, generate a concise, descriptive title (max 50 characters)

Remember if the message contains:
- Personal facts (preferences, possessions, experiences)
- Important decisions or plans (INCLUDING future intentions like "I'm going to...")
- Significant statements about the user's life
- Information the user might want recalled later
- Goals, projects, or commitments the user mentions
- Emotional states or personal struggles (e.g., "I've been feeling really down", "I think my girlfriend is cheating on me", "I'm embarrassed about something")
- Relationship situations, tensions, or concerns the user is navigating

Do NOT remember:
- Simple questions without context
- Short acknowledgments (ok, thanks, etc.)
- Generic conversation without personal information
- Commands or instructions to the assistant
- Questions asking the user to recall their own memories, history, experiences, or achievements ("What theories have I come up with?", "What do I believe?", "What's my career been like?"). These are memory-recall requests — not new facts to store.

✓ ALWAYS remember statements like:
  • "I'm going to start a book series"
  • "My dog's name is Max"
  • "I'm planning to move to Boston"
  • "I work at Google"
  • "I graduated in 2015"

⚠️ CRITICAL OUTPUT REQUIREMENTS:
- Respond with ONLY valid JSON
- Do NOT include any explanatory text before or after the JSON
- Do NOT use markdown code blocks (no ```json or ```)
- Start your response immediately with the opening brace {{

OUTPUT FORMAT:
{{
  "should_remember": true/false,
  "title": "Title here (only if should_remember is true)"
}}

Return the JSON now:"#,
        conversation_history, message
    );

    // Call LLM based on backend (same backend as main conversation!)
    let response = if backend.contains("anthropic") {
        call_anthropic_for_memory_decision(&prompt).await?
    } else if backend.contains("openai") {
        call_openai_for_memory_decision(&prompt).await?
    } else if backend.contains("xai") {
        call_xai_for_memory_decision(&prompt).await?
    } else if backend.ends_with(".gguf") || backend == "local" {
        call_local_for_memory_decision(&prompt, backend, Some(MEMORY_DECISION_SCHEMA)).await?
    } else {
        println!("[Memory Decision] Unknown backend '{}' - skipping memory decision", backend);
        return Ok((false, None));
    };

    // Parse JSON response
    #[derive(serde::Deserialize)]
    struct MemoryDecision {
        should_remember: bool,
        title: Option<String>,
    }

    // Try to extract JSON from response (handles cases where LLM adds extra text)
    let json_str = if let Some(start) = response.find('{') {
        if let Some(end) = response.rfind('}') {
            &response[start..=end]
        } else {
            &response
        }
    } else {
        &response
    };

    match serde_json::from_str::<MemoryDecision>(json_str) {
        Ok(decision) => {
            println!("[Memory Decision] LLM decision: should_remember={}, title={:?}",
                     decision.should_remember, decision.title);
            Ok((decision.should_remember, decision.title))
        }
        Err(e) => {
            println!("[Memory Decision] Failed to parse LLM response: {}", e);
            println!("[Memory Decision] Response was: {}", response);
            // Fallback to no memory on parse error
            Ok((false, None))
        }
    }
}

/// Call Anthropic Claude for memory decision
async fn call_anthropic_for_memory_decision(prompt: &str) -> Result<String, String> {
    use crate::llm::anthropic;
    use crate::llm::Message;

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| "ANTHROPIC_API_KEY not set".to_string())?;

    let messages = vec![Message {
        role: "user".to_string(),
        content: prompt.to_string(),
    }];

    match anthropic::send_message(
        &api_key,
        "claude-haiku-4-5-20251001",  // Fast, cheap model for decisions
        messages,
        None,
        Some(4096),  // Match main conversation limit - handles large recalled memories
        Some(0.3),  // Low temperature for consistency
    ).await {
        Ok(response) => Ok(response.content),
        Err(e) => Err(format!("Anthropic API error: {}", e))
    }
}

/// Call OpenAI GPT for memory decision
async fn call_openai_for_memory_decision(prompt: &str) -> Result<String, String> {
    use crate::llm::openai;
    use crate::llm::Message;

    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY not set".to_string())?;

    let messages = vec![Message {
        role: "user".to_string(),
        content: prompt.to_string(),
    }];

    match openai::send_message(
        &api_key,
        "gpt-4o-mini",
        messages,
        Some(4096),  // Match main conversation limit
        Some(0.3),
    ).await {
        Ok(response) => Ok(response.content),
        Err(e) => Err(format!("OpenAI API error: {}", e))
    }
}

/// Call xAI Grok for memory decision
async fn call_xai_for_memory_decision(prompt: &str) -> Result<String, String> {
    use crate::llm::xai;
    use crate::llm::Message;

    let api_key = std::env::var("XAI_API_KEY")
        .map_err(|_| "XAI_API_KEY not set".to_string())?;

    let messages = vec![Message {
        role: "user".to_string(),
        content: prompt.to_string(),
    }];

    match xai::send_message(
        &api_key,
        "grok-3",
        messages,
        Some(4096),  // Match main conversation limit
        Some(0.3),
    ).await {
        Ok(response) => Ok(response.content),
        Err(e) => Err(format!("xAI API error: {}", e))
    }
}

/// Call local GGUF model for memory decision
async fn call_local_for_memory_decision(prompt: &str, backend: &str, json_schema: Option<&'static str>) -> Result<String, String> {
    use crate::llm::Message;

    let model_path = if backend.ends_with(".gguf") {
        backend.to_string()
    } else {
        std::env::var("LOCAL_MODEL_PATH")
            .unwrap_or_else(|_| "models/user/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf".to_string())
    };

    let messages = vec![Message {
        role: "user".to_string(),
        content: prompt.to_string(),
    }];

    let model_path_clone = model_path.clone();
    let response = tokio::task::spawn_blocking(move || {
        if let Some(schema) = json_schema {
            // Grammar-constrained: model physically cannot produce invalid JSON
            llm::local_models::generate_with_local_model_constrained(
                &model_path_clone,
                messages,
                schema,
            )
        } else {
            llm::local_models::generate_with_local_model(
                &model_path_clone,
                messages,
                Some(4096),
                Some(0.3),
            )
        }
    })
    .await
    .map_err(|e| format!("Failed to run local model task: {}", e))?
    .map_err(|e| e.to_string())?;

    Ok(response.content)
}

// JSON schemas for grammar-constrained local model calls.
// These are converted to GBNF grammars at call time — the model cannot produce
// tokens that would violate the schema, so JSON parse errors become impossible.

const RELATIONSHIP_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "relationships": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "memory_id": {"type": "integer"},
          "relationship_type": {"type": "string"},
          "reason": {"type": "string"},
          "confidence": {"type": "number"}
        },
        "required": ["memory_id", "relationship_type", "reason"]
      }
    }
  },
  "required": ["title", "relationships"]
}"#;

const MEMORY_DECISION_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "should_remember": {"type": "boolean"},
    "title": {"type": "string"}
  },
  "required": ["should_remember"]
}"#;

const MEMORY_DECISION_WITH_RELATIONSHIPS_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "should_remember": {"type": "boolean"},
    "title": {"type": "string"},
    "relationships": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "memory_id": {"type": "integer"},
          "relationship_type": {"type": "string"},
          "reason": {"type": "string"},
          "confidence": {"type": "number"}
        },
        "required": ["memory_id", "relationship_type"]
      }
    }
  },
  "required": ["should_remember"]
}"#;

// ============================================================================
// ENHANCED MEMORY DECISION WITH RELATIONSHIP CLASSIFICATION
// ============================================================================

/// Deserialize memory_id from either an integer or a string (DeepSeek returns "none" as a string
/// when there is no relationship, instead of omitting the field or using null/0).
fn deserialize_memory_id<'de, D>(deserializer: D) -> Result<i32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    #[allow(dead_code)]
    enum MemoryIdValue { Int(i32), Str(String) }
    match MemoryIdValue::deserialize(deserializer)? {
        MemoryIdValue::Int(i) => Ok(i),
        MemoryIdValue::Str(_) => Ok(-1), // "none" or any string → sentinel -1, filtered downstream
    }
}

/// Relationship classification returned by LLM
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct RelationshipClassification {
    #[serde(deserialize_with = "deserialize_memory_id")]
    memory_id: i32,
    relationship_type: String,  // "contradicts", "supports", "elaborates", "caused_by", "reminds_of", "none"
    reason: String,
    confidence: Option<f32>,  // 0.0-1.0 confidence score
}

/// Enhanced memory decision that also classifies relationships with similar memories
/// Returns (should_remember, optional_title, relationship_classifications)
async fn ask_llm_about_memory_with_relationships(
    message: &str,
    conversation_history: &str,
    similar_memories: &[(i32, String, Option<String>, f32)],  // (id, content, title, similarity)
    backend: &str,
) -> Result<(bool, Option<String>, Vec<RelationshipClassification>), String> {
    // If no similar memories, fall back to simple decision
    if similar_memories.is_empty() {
        let (should_remember, title) = ask_llm_about_memory(message, conversation_history, backend).await?;
        return Ok((should_remember, title, Vec::new()));
    }

    // Build similar memories section
    let similar_memories_text = similar_memories.iter()
        .map(|(id, content, title, sim)| {
            let title_str = title.as_ref().map(|t| format!(" ({})", t)).unwrap_or_default();
            format!("Memory #{}{} (similarity: {:.1}%)\n{}", id, title_str, sim * 100.0, content)
        })
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    // Build enhanced prompt
    let prompt = format!(
        r#"You are a memory decision and relationship classification system.

CONVERSATION CONTEXT:
{}

USER MESSAGE:
{}

SIMILAR EXISTING MEMORIES:
{}

TASK:
1. Decide if this message should be stored as a long-term memory
2. If yes, generate a concise, descriptive title (max 50 characters)
3. For EACH similar memory above, classify the relationship:

RELATIONSHIP TYPES (in priority order):
- "contradicts": Direct contradiction (e.g., "I believe X" vs "I don't believe X", "My dog is Max" vs "My dog is Wendy")
  → CRITICAL: If the user affirms X but an existing memory negates X (or vice versa), it is ALWAYS a contradiction, even if they share similar topics or high semantic similarity!
- "supports": Reinforces or agrees with the existing memory
- "elaborates": Adds detail or context to the existing memory
- "caused_by": This memory describes a cause of the existing memory's effect
- "reminds_of": Loosely related or tangentially connected
- "none": No meaningful relationship — return this if the connection is incidental

⚠️ CONTRADICTION DETECTION RULES (HIGHEST PRIORITY):
1. Focus on the MAIN CLAIM, not descriptive phrases
2. If one memory says "I believe/have/am X" and another says "I don't believe/have/am X", that is a CONTRADICTION
3. High semantic similarity does NOT mean agreement - opposite claims about the same topic are contradictions!
4. Ignore negations in descriptive clauses (e.g., "not a personal deity" describes what something is)

EXAMPLES OF CONTRADICTIONS:
✓ "I believe in Spinoza's God" vs "I do not believe in Spinoza's God"
  → Main claims oppose (belief vs disbelief)
✓ "My dog is named Max" vs "My dog's name is Wendy"
  → Same attribute, different values
✓ "I live in New York" vs "I live in Boston"
  → Same attribute (location), different values
✓ "I'm 30 years old" vs "I am 25"
  → Same attribute (age), different values
✓ "I work at Google" vs "I work at Microsoft"
  → Same attribute (employer), different values
✓ "I'm married to Sarah" vs "My spouse is Emma"
  → Same relationship type, different people
✓ "I'm planning to travel to Japan this summer" vs "I traveled to Japan a couple months ago"
  → Future plan contradicts past completion — can't be planning a trip you already took

EXAMPLES OF NON-CONTRADICTIONS:
✗ "I believe in Spinoza's God - a pantheistic view" vs "Spinoza's God is not a personal deity"
  → Both describe the same concept, "not a personal deity" is a description
✗ "My dog Max is 5 years old" vs "My dog's name is Max"
  → Different details about the same subject (age vs name)
✗ "I live in Brooklyn" vs "I live in New York City"
  → Brooklyn is part of NYC (hierarchy, not contradiction)
✗ "I went to Japan last year" vs "I'm traveling to Japan this summer"
  → Two separate trips — a past visit doesn't prevent a future one
✗ "I'm thinking about leaving my job at X" vs "I work at X"
  → A future intention does not negate a current state — both can be true simultaneously

Remember if the message contains:
- Personal facts (preferences, possessions, experiences)
- Important decisions or plans (INCLUDING future intentions like "I'm going to...")
- Significant statements about the user's life
- Information the user might want recalled later
- Goals, projects, or commitments the user mentions
- Emotional states or personal struggles (e.g., "I've been feeling really down", "I think my girlfriend is cheating on me", "I'm embarrassed about something")
- Relationship situations, tensions, or concerns the user is navigating

Do NOT remember:
- Simple questions without context
- Short acknowledgments (ok, thanks, etc.)
- Generic conversation without personal information
- Commands or instructions to the assistant
- Questions asking the user to recall their own memories, history, experiences, or achievements ("What theories have I come up with?", "What do I believe?", "What's my career been like?"). These are memory-recall requests — not new facts to store.

✓ ALWAYS remember statements like:
  • "I'm going to start a book series"
  • "My dog's name is Max"
  • "I'm planning to move to Boston"
  • "I work at Google"
  • "I graduated in 2015"

CRITICAL: If the message is a FACTUAL STATEMENT (even if it contradicts existing memories),
set should_remember=true and flag the contradictions in the relationships array.
The user will decide which memory to keep via the UI.

Examples of factual statements that should ALWAYS be remembered:
✓ "I graduated from Princeton in 1900" → should_remember=true (even if it contradicts birth year)
✓ "My dog's name is Max" → should_remember=true (even if existing memory says "Wendy")
✓ "I live in Boston" → should_remember=true (even if existing memory says "New York")
✓ "I work at Google" → should_remember=true (even if existing memory says "Microsoft")

The contradiction detection is SEPARATE from the should_remember decision!

⚠️ CRITICAL OUTPUT REQUIREMENTS:
- Respond with ONLY valid JSON
- Do NOT include any explanatory text before or after the JSON
- Do NOT use markdown code blocks (no ```json or ```)
- Do NOT include analysis, commentary, or reasoning outside the JSON
- Start your response immediately with the opening brace {{

OUTPUT FORMAT:
{{
  "should_remember": true/false,
  "title": "Title here (only if should_remember is true)",
  "relationships": [
    {{
      "memory_id": 123,
      "relationship_type": "contradicts|supports|elaborates|caused_by|reminds_of|none",
      "reason": "Brief explanation of the relationship",
      "confidence": 0.95
    }}
  ]
}}

Return the JSON now:"#,
        conversation_history, message, similar_memories_text
    );

    // Call LLM based on backend
    let response = if backend.contains("anthropic") {
        call_anthropic_for_memory_decision(&prompt).await?
    } else if backend.contains("openai") {
        call_openai_for_memory_decision(&prompt).await?
    } else if backend.contains("xai") {
        call_xai_for_memory_decision(&prompt).await?
    } else if backend.ends_with(".gguf") || backend == "local" {
        call_local_for_memory_decision(&prompt, backend, Some(MEMORY_DECISION_WITH_RELATIONSHIPS_SCHEMA)).await?
    } else {
        println!("[Memory Decision] Unknown backend '{}' - skipping", backend);
        return Ok((false, None, Vec::new()));
    };

    // Parse JSON response
    #[derive(serde::Deserialize)]
    struct EnhancedMemoryDecision {
        should_remember: bool,
        title: Option<String>,
        relationships: Option<Vec<RelationshipClassification>>,
    }

    // Extract JSON from response (handle markdown code blocks and raw JSON)
    let json_str = if response.contains("```json") {
        // Try to extract from markdown code block first
        if let Some(start) = response.find("```json") {
            let json_start = start + 7; // Skip past ```json
            if let Some(end) = response[json_start..].find("```") {
                response[json_start..json_start + end].trim()
            } else {
                // No closing ```, try to find JSON object
                &response[json_start..]
            }
        } else {
            &response
        }
    } else if response.contains("```") {
        // Handle generic code blocks ```...```
        if let Some(start) = response.find("```") {
            let json_start = start + 3;
            if let Some(end) = response[json_start..].find("```") {
                response[json_start..json_start + end].trim()
            } else {
                &response[json_start..]
            }
        } else {
            &response
        }
    } else if let Some(start) = response.find('{') {
        // Fallback: extract raw JSON by finding braces
        if let Some(end) = response.rfind('}') {
            &response[start..=end]
        } else {
            &response
        }
    } else {
        &response
    };

    // Some local models (e.g. dolphin) output double-underscored keys like memory__id
    let json_normalized = json_str.replace("memory__id", "memory_id").replace("relationship__type", "relationship_type");
    let json_str = json_normalized.as_str();

    match serde_json::from_str::<EnhancedMemoryDecision>(json_str) {
        Ok(decision) => {
            let relationships = decision.relationships.unwrap_or_default();
            println!("[Memory Decision] ✅ Successfully parsed JSON response");
            println!("[Memory Decision] LLM decision: should_remember={}, title={:?}, {} relationships",
                     decision.should_remember, decision.title, relationships.len());

            // Log relationship classifications
            if !relationships.is_empty() {
                println!("[Memory Decision] Relationships classified:");
                for rel in &relationships {
                    if rel.relationship_type != "none" {
                        println!("[Memory Decision]   Memory #{}: {} (confidence: {:.2}) - {}",
                                 rel.memory_id, rel.relationship_type,
                                 rel.confidence.unwrap_or(0.0), rel.reason);
                    } else {
                        println!("[Memory Decision]   Memory #{}: NONE - {}",
                                 rel.memory_id, rel.reason);
                    }
                }
            }

            Ok((decision.should_remember, decision.title, relationships))
        }
        Err(e) => {
            println!("[Memory Decision] ❌ Failed to parse JSON response: {}", e);
            println!("[Memory Decision] Attempted to parse: {}", json_str);
            println!("[Memory Decision] Full response was: {}", response);

            // Fallback: try simple decision without relationships
            println!("[Memory Decision] Falling back to simple memory decision (no relationship detection)");
            match ask_llm_about_memory(message, conversation_history, backend).await {
                Ok((should_remember, title)) => {
                    println!("[Memory Decision] ⚠️ Using fallback decision: should_remember={}, title={:?}",
                             should_remember, title);
                    Ok((should_remember, title, Vec::new()))
                },
                Err(fallback_err) => {
                    println!("[Memory Decision] ❌ Fallback also failed: {}", fallback_err);
                    Ok((false, None, Vec::new()))
                }
            }
        }
    }
}

/// Send message with memory - PURE RUST IMPLEMENTATION
/// No Flask dependency - handles everything in Rust
#[allow(clippy::too_many_arguments)]
#[tauri::command]
async fn send_message_with_memory(
    app: tauri::AppHandle,
    message: String,
    user_id: String,
    session_id: String,
    backend: String,
    containment_mode: String,
    conversation_history: Option<Vec<ConversationTurn>>,
    skip_containment: Option<bool>,
    skip_memory_storage: Option<bool>,  // NEW: Skip storing this message as a memory (for web search synthesis)
    _kb_enabled: Option<bool>,  // NEW: Enable Knowledge Base RAG search
    user_query: Option<String>,  // Clean question without attached file content — used for safety/memory/search
) -> Result<ReplyResponse, String> {
    use conversation_engine::ConversationEngine;

    // Normalize brand name misspellings from voice transcription before any processing
    let message = normalize_brand_names(message);

    let _request_start = std::time::Instant::now();
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  💬 NEW CHAT REQUEST                                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("▶ {}", message);
    println!();
    println!("[RUST] Backend: {}", backend);

    // Child mode: Force OpenAI backend for all responses
    let mut forced_backend = backend.clone();
    if containment_mode.to_lowercase() == "child" {
        println!("[RUST] 🧒 Child mode detected - forcing OpenAI backend");
        forced_backend = "openai".to_string();
    }

    // When a file is attached, `message` contains the full dump (file content + question).
    // `user_query` carries only the clean question — use it for safety checks, memory search,
    // KB search, and conversation history. Falls back to `message` when no file is attached.
    let query = user_query.unwrap_or_else(|| message.clone());

    // STEP 1: Safety check via containment layer (skip for internal operations like web search synthesis)
    let mut warning_prefix: Option<String> = None;

    if !skip_containment.unwrap_or(false) {
        let step_start = std::time::Instant::now();

        // Child mode: Use OpenAI Moderation API for safety check
        if containment_mode.to_lowercase() == "child" {
            println!("[RUST] 🧒 Child mode - checking with OpenAI Moderation API...");
            let layer = containment::ContainmentLayer::new(&containment_mode)?;

            match layer.check_openai_moderation(&query).await {
                Ok(Some(block_message)) => {
                    // Content blocked by OpenAI moderation
                    println!("[RUST] 🛑 Content BLOCKED by OpenAI Moderation");
                    println!("[⏱️ PERF] OpenAI moderation check: {:.3}s", step_start.elapsed().as_secs_f32());
                    return Ok(ReplyResponse {
                        reply_text: block_message,
                        recalled_memories: None,
                        model_backend: Some(forced_backend),
                        containment_mode: Some(containment_mode),
                        schema: None,
                        blocked: Some(true),
                        web_search_needed: None,
                        web_search_query: None,
                        original_query: None,
                    });
                }
                Ok(None) => {
                    // Content passed OpenAI moderation
                    println!("[RUST] ✅ Content passed OpenAI Moderation - proceeding with normal flow");
                }
                Err(e) => {
                    // API error - block for safety in Child mode
                    println!("[RUST] ❌ OpenAI Moderation API error: {}", e);
                    return Ok(ReplyResponse {
                        reply_text: format!(
                            "I'm having trouble checking if this is safe for Child Mode. Please try again later.\n\nError: {}",
                            e
                        ),
                        recalled_memories: None,
                        model_backend: Some(forced_backend),
                        containment_mode: Some(containment_mode),
                        schema: None,
                        blocked: Some(true),
                        web_search_needed: None,
                        web_search_query: None,
                        original_query: None,
                    });
                }
            }
            println!("[⏱️ PERF] OpenAI moderation check: {:.3}s", step_start.elapsed().as_secs_f32());
        } else {
            // Non-child modes: Use standard containment check
            let safety_check = commands::safety::check_containment(query.clone(), containment_mode.clone()).await;
            println!("[⏱️ PERF] Safety check: {:.3}s", step_start.elapsed().as_secs_f32());

            match safety_check {
                Ok(Some(message)) => {
                    // Check if this is a warning (Sovereign mode) or a block
                    if message.starts_with("[WARN_ALLOW]") {
                        // Sovereign mode: Extract warning, continue with LLM
                        println!("[RUST] ⚠️ Sovereign mode warning - will prepend to LLM response");
                        warning_prefix = Some(message.trim_start_matches("[WARN_ALLOW]").trim().to_string());
                    } else {
                        // Hard block (Guardian, HIPAA)
                        println!("[RUST] 🛑 Content BLOCKED by containment layer");
                        return Ok(ReplyResponse {
                            reply_text: message,
                            recalled_memories: None,
                            model_backend: Some(forced_backend),
                            containment_mode: Some(containment_mode),
                            schema: None,
                            blocked: Some(true),
                            web_search_needed: None,
                            web_search_query: None,
                            original_query: None,
                        });
                    }
                }
                Ok(None) => {
                    // Content passed safety check
                }
                Err(e) => {
                    println!("[RUST] ⚠️ Safety check failed: {}", e);
                    // Continue anyway - don't block on safety check errors
                }
            }
        }
    }

    // STEP 2: Recall memories using entity-based + semantic hybrid search
    // (Contradiction check moved to async background task after memory storage)
    // (Web search detection moved to main LLM response parsing)
    let step_start = std::time::Instant::now();

    // Extract entities from user's message using BERT NER
    let query_text = query.clone();
    let extracted_entities = tokio::task::spawn_blocking(move || {
        let enhancer = nlp_enhancer::NLPEnhancer::new();
        enhancer.extract_entities(&query_text)
    })
    .await
    .map_err(|e| format!("Failed to run entity extraction: {}", e))?;

    // Convert Entity objects to strings (just the word values, lowercased for comparison)
    let all_entities: Vec<String> = extracted_entities
        .into_iter()
        .map(|e| e.word.to_lowercase())  // IMPORTANT: Lowercase to match SQL's LOWER(elem->>'word')
        .collect();

    // Filter out stop words to improve entity matching quality
    // Stop words dilute the entity overlap score in hybrid search
    let stop_words: std::collections::HashSet<&str> = vec![
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
        "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
        "out", "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "s", "t", "can", "will", "just", "don", "should", "now",
        ".", ",", "!", "?", ";", ":", "-", "'", "\"",
        // Add common words that aren't meaningful as entities
        "never", "always", "anything", "something", "nothing", "everything",
    ].into_iter().collect();

    let query_entities: Vec<String> = all_entities
        .into_iter()
        .filter(|e| !stop_words.contains(e.as_str()) && e.len() > 1)  // Keep entities longer than 1 char
        .collect();

    println!("[RUST] ✅ Extracted {} meaningful entities from query: {:?}",
        query_entities.len(),
        query_entities.iter().take(10).collect::<Vec<_>>());

    // Connect to database
    let db_url = crate::db::get_db_url();

    let db_pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;
    println!("[⏱️ PERF] Database connection: {:.3}s", step_start.elapsed().as_secs_f32());

    // Generate embedding for hybrid search
    let step_start = std::time::Instant::now();
    let query_text_for_embedding = query.clone();
    let query_embedding = tokio::task::spawn_blocking(move || {
        llm::local_embeddings::generate_local_embedding(&query_text_for_embedding)
    })
    .await
    .map_err(|e| format!("Failed to run embedding task: {}", e))?
    .map_err(|e| format!("Failed to generate embedding: {}", e))?;

    // HYBRID SEARCH: Use hybrid_search for weighted entity + semantic scoring
    // Search limit matches build_memory_context caps: API gets more headroom, local stays lean
    let memory_search_limit = if ConversationEngine::is_api_model(&forced_backend) { 15 } else { 10 };

    // Smart namespace filtering: search ONLY system memories if query is about Zynkbot
    let is_zynkbot_query = is_query_about_zynkbot(&query);
    let (search_user_id, namespace_filter) = if is_zynkbot_query {
        ("system", Some("_zynkbot"))  // Search system user's memories in _zynkbot namespace
    } else {
        (user_id.as_str(), None)  // Search user's memories, filter system memories manually below
    };

    let recalled_memories = memory::hybrid_search(
        &db_pool,
        query_embedding,
        query_entities.clone(),  // Clone so we can use it again for entity search
        Some(search_user_id),
        None,  // Don't filter by session - search ALL memories across all sessions!
        namespace_filter,  // Search system namespace if Zynkbot query, else all namespaces
        memory_search_limit,
    )
    .await
    .map_err(|e| format!("Memory search failed: {}", e))?;

    // Filter out system memories for non-Zynkbot queries
    let recalled_memories: Vec<memory::Memory> = if is_zynkbot_query {
        recalled_memories  // Keep all (should only be system memories anyway)
    } else {
        recalled_memories.into_iter()
            .filter(|m| m.namespace != "_zynkbot")
            .collect()
    };

    // ONE-HOP GRAPH TRAVERSAL: For each recalled memory, pull in directly linked
    // memories via "elaborates", "contradicts", or "resolves" relationships.
    // Capped at 3 additional memories to prevent prompt bloat.
    let mut linked_memories: Vec<memory::Memory> = Vec::new();
    if !is_zynkbot_query {
        let already_included_ids: std::collections::HashSet<i32> = recalled_memories.iter()
            .map(|m| m.id)
            .collect();
        let mut already_included_content: std::collections::HashSet<String> = recalled_memories.iter()
            .map(|m| m.content.clone())
            .collect();

        for mem in &recalled_memories {
            if let Ok(links) = memory::get_memory_links(&db_pool, mem.id).await {
                for link in links {
                    if link.relation_type != "elaborates" && link.relation_type != "contradicts" && link.relation_type != "resolves" {
                        continue;
                    }
                    let linked_id = if link.source_memory_id == mem.id {
                        link.target_memory_id
                    } else {
                        link.source_memory_id
                    };
                    if already_included_ids.contains(&linked_id)
                        || linked_memories.iter().any(|m| m.id == linked_id)
                    {
                        continue;
                    }
                    if let Ok(Some(linked_mem)) = memory::get_memory(&db_pool, linked_id).await {
                        if already_included_content.contains(&linked_mem.content) {
                            continue;
                        }
                        println!("[RUST] 🔗 Graph traversal: memory #{} → linked #{} ({:?}) via '{}'",
                            mem.id, linked_id, linked_mem.title, link.relation_type);
                        already_included_content.insert(linked_mem.content.clone());
                        linked_memories.push(linked_mem);
                    }
                }
            }
        }

        if !linked_memories.is_empty() {
            println!("[RUST] ✅ Graph traversal added {} linked memories", linked_memories.len());
        }
    }

    let total_memories = recalled_memories.len() + linked_memories.len();
    println!("[RUST] ✅ Found {} relevant memories ({} hybrid search + {} linked)",
             total_memories, recalled_memories.len(), linked_memories.len());
    println!("[⏱️ PERF] Vector search: {:.3}s", step_start.elapsed().as_secs_f32());

    // Prepare similar memories for relationship classification (reuse the same memories we found!)
    // Format: Vec<(id, content, title, similarity)>
    let _similar_memories_for_relationships: Vec<(i32, String, Option<String>, f32)> = recalled_memories
        .iter()
        .map(|m| (m.id, m.content.clone(), m.title.clone(), m.similarity.unwrap_or(0.0) as f32))
        .collect();

    // Determine if this is an API model (for adaptive context limits)
    let is_api_model = ConversationEngine::is_api_model(&forced_backend);

    // STEP 3: Knowledge Base RAG search (opt-in via UI button)
    // Only searches when user clicks "Search Knowledge Base" button
    let kb_enabled = _kb_enabled.unwrap_or(false);
    let mut kb_context = String::new();

    if kb_enabled {
        let kb_start = std::time::Instant::now();

        // EXPLICIT KB SEARCH (user clicked KB button)
        // Much more aggressive than automatic search since user has explicit intent
        // - 10 chunks (comprehensive coverage)
        // - 15% threshold (cast wide net - user knows what they're looking for)
        // - Always return top results even if below threshold
        let kb_chunk_limit = 10;
        let kb_similarity_threshold = 0.15;

        println!(
            "[KB RAG] 🔍 EXPLICIT KB SEARCH ({} chunks, {:.0}% threshold)",
            kb_chunk_limit,
            kb_similarity_threshold * 100.0
        );

        // Perform semantic search in KB
        // Explicit KB search: exclude system docs - user wants THEIR documents only
        match kb_rag::search_kb_chunks(&db_pool, &user_id, &query, kb_chunk_limit, false).await {
            Ok(kb_results) => {
                // For EXPLICIT search: be more permissive
                // 1. First, get all chunks above threshold
                let mut relevant_chunks: Vec<_> = kb_results
                    .iter()
                    .filter(|r| r.similarity_score > kb_similarity_threshold)
                    .cloned()
                    .collect();

                // 2. If none meet threshold, take top 5 best matches anyway (user explicitly requested)
                if relevant_chunks.is_empty() && !kb_results.is_empty() {
                    println!("[KB RAG] ⚠️ No chunks above {:.0}% threshold - returning top 5 best matches", kb_similarity_threshold * 100.0);
                    relevant_chunks = kb_results.into_iter().take(5).collect();
                }

                if !relevant_chunks.is_empty() {
                    println!(
                        "[KB RAG] ✅ Found {} relevant chunks (best: {:.1}%, worst: {:.1}%)",
                        relevant_chunks.len(),
                        relevant_chunks.first().map(|r| r.similarity_score * 100.0).unwrap_or(0.0),
                        relevant_chunks.last().map(|r| r.similarity_score * 100.0).unwrap_or(0.0)
                    );

                    // Build KB context section with emphatic instructions
                    kb_context.push_str("\n\n╔═══════════════════════════════════════════════════════════╗\n");
                    kb_context.push_str("║  🔍 EXPLICIT KNOWLEDGE BASE SEARCH - USER REQUESTED       ║\n");
                    kb_context.push_str("╚═══════════════════════════════════════════════════════════╝\n\n");
                    kb_context.push_str("⚠️ CRITICAL INSTRUCTION: The user clicked the KB button to explicitly search their indexed documents.\n");
                    kb_context.push_str("You MUST use the information below to answer the question.\n");
                    kb_context.push_str("DO NOT suggest web search - the answer is in the KB context below.\n\n");
                    kb_context.push_str("=== RETRIEVED DOCUMENTS ===\n\n");

                    for (idx, result) in relevant_chunks.iter().enumerate() {
                        kb_context.push_str(&format!(
                            "📄 Document {}: {} (similarity: {:.1}%)\n{}\n\n",
                            idx + 1,
                            result.file_name,
                            result.similarity_score * 100.0,
                            result.content
                        ));
                    }

                    kb_context.push_str("=== END OF KB DOCUMENTS ===\n\n");
                    kb_context.push_str("✅ Answer the question using ONLY the information above from the user's Knowledge Base.\n");
                } else {
                    println!("[KB RAG] ⚠️ No documents found in knowledge base");
                }
            }
            Err(e) => {
                eprintln!("[KB RAG] ⚠️ Knowledge base search failed: {}", e);
                // Continue without KB context - don't fail the entire request
            }
        }

        println!("[⏱️ PERF] KB RAG search: {:.3}s", kb_start.elapsed().as_secs_f32());
    }

    // Look up the user's display name from their first onboarding memory so the
    // LLM can address them by name and use the name in MEMORY_EXTRACT lines.
    let user_display_name = memory::get_user_display_name(&db_pool, &user_id).await;

    // Close database connection
    db_pool.close().await;

    // STEP 5: Build prompt using conversation engine
    let engine = ConversationEngine::new();

    // Convert conversation history to conversation engine format
    let engine_history: Option<Vec<conversation_engine::ConversationTurn>> = conversation_history.as_ref().map(|hist| {
        hist.iter().map(|turn| {
            conversation_engine::ConversationTurn {
                role: turn.role.clone(),
                content: turn.content.clone(),
            }
        }).collect()
    });

    // Convert recalled memories to conversation engine format
    // Include semantic results AND one-hop graph-linked memories only.
    // entity_matched_memories are for contradiction/duplicate detection in the background
    // task and should NOT be injected into the prompt — they failed the hybrid search threshold.
    let engine_memories: Vec<conversation_engine::Memory> = recalled_memories
        .iter()
        .chain(linked_memories.iter())
        .map(|mem| conversation_engine::Memory {
            id: mem.id,
            content: mem.content.clone(),
            original_text: mem.original_text.clone(),
            title: mem.title.clone(),  // Already Option<String>, no need to wrap in Some()
            similarity: mem.similarity,
            created_at: mem.created_at,
        })
        .collect();

    // Build full prompt (user_display_name was fetched before the pool was closed)
    let mut full_prompt = engine.build_prompt(
        &message,
        engine_history.as_deref(),
        Some(&engine_memories),
        is_api_model,
        user_display_name.as_deref(),
    );

    // Prepend KB context if available
    if !kb_context.is_empty() {
        full_prompt = format!("{}{}", kb_context, full_prompt);
        println!("[KB RAG] Added {} chars of KB context to prompt", kb_context.len());
    }

    // STEP 6: Call LLM based on backend (use forced_backend for Child mode)
    // All API backends use SSE streaming so the frontend can display tokens as they arrive.
    // Local GGUF models are blocking and cannot stream, so they continue to return all at once.
    let _api_start = std::time::Instant::now();

    // Paired-call channel: after the main local model call completes, the loaded model session
    // is sent here so the background task can reuse it for Call 2 (relationship classification)
    // without a second disk load. None for API backends.
    let mut local_session_rx: Option<tokio::sync::oneshot::Receiver<llm::local_models::LocalModelSession>> = None;

    let reply_text = if forced_backend.to_lowercase().contains("anthropic") || forced_backend.to_lowercase().contains("claude") {
        // Use Anthropic with streaming
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY not set".to_string())?;

        let model_name = if forced_backend.contains("haiku") {
            "claude-haiku-4-5-20251001"
        } else if forced_backend.contains("opus") {
            "claude-opus-4-7"
        } else {
            "claude-sonnet-4-6" // Default: Sonnet 4.6 (best balance of speed and capability)
        };

        println!("[⏱️ PERF] Calling Anthropic API ({}) with streaming...", model_name);
        let messages = vec![llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];

        let app_handle = app.clone();
        let response = llm::anthropic::send_message_streaming(
            &api_key,
            model_name,
            messages,
            None, // system prompt is in the message
            Some(4096),
            None,
            move |token| { app_handle.emit("stream-token", token).ok(); },
        ).await.map_err(|e| e.to_string())?;

        response.content

    } else if forced_backend.to_lowercase().contains("openai") || forced_backend.to_lowercase().contains("gpt") {
        // Use OpenAI with streaming (including Child mode)
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY not set".to_string())?;

        let model_name = "gpt-4o-mini";

        println!("[⏱️ PERF] Calling OpenAI API ({}) with streaming...", model_name);
        let messages = vec![llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];

        let app_handle = app.clone();
        let response = llm::openai::send_message_streaming(
            &api_key,
            model_name,
            messages,
            Some(4096),
            None,
            "https://api.openai.com/v1/chat/completions",
            move |token| { app_handle.emit("stream-token", token).ok(); },
        ).await.map_err(|e| e.to_string())?;

        response.content

    } else if forced_backend.to_lowercase().contains("xai") || forced_backend.to_lowercase().contains("grok") {
        // Use xAI (Grok) with streaming - OpenAI-compatible format
        let api_key = std::env::var("XAI_API_KEY")
            .map_err(|_| "XAI_API_KEY not set. Get your API key from https://console.x.ai/".to_string())?;

        let model_name = if forced_backend.contains("vision") {
            "grok-2-vision-1212"
        } else {
            "grok-3"
        };

        println!("[⏱️ PERF] Calling xAI API ({}) with streaming...", model_name);
        let messages = vec![llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];

        let app_handle = app.clone();
        let response = llm::openai::send_message_streaming(
            &api_key,
            model_name,
            messages,
            Some(4096),
            None,
            "https://api.x.ai/v1/chat/completions",
            move |token| { app_handle.emit("stream-token", token).ok(); },
        ).await.map_err(|e| e.to_string())?;

        response.content

    } else if forced_backend.to_lowercase().contains("local") || forced_backend.ends_with(".gguf") {
        // Use local GGUF model

        // Determine model path
        let model_path = if forced_backend.ends_with(".gguf") {
            // Explicit path provided
            forced_backend.clone()
        } else {
            // Use default model from environment or fallback
            std::env::var("LOCAL_MODEL_PATH")
                .unwrap_or_else(|_| "models/user/Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string())
        };

        let messages = vec![llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];

        // Paired-call: load model once, generate main response, pass session to background
        // task via channel so Call 2 reuses the already-loaded model.
        let (session_tx, session_rx) = tokio::sync::oneshot::channel::<llm::local_models::LocalModelSession>();
        local_session_rx = Some(session_rx);

        let model_path_clone = model_path.clone();
        let response = tokio::task::spawn_blocking(move || {
            let session = llm::local_models::LocalModelSession::load(&model_path_clone)?;
            let response = session.generate(messages, Some(4096), None, None)?;
            // Send session to background task — if the receiver was already dropped, ignore.
            let _ = session_tx.send(session);
            Ok::<_, llm::LLMError>(response)
        })
        .await
        .map_err(|e| format!("Failed to run local model task: {}", e))?
        .map_err(|e| e.to_string())?;

        response.content

    } else {
        return Err(format!(
            "Unsupported backend: {}. Use 'anthropic', 'openai', 'xai' (grok), 'local', or provide a .gguf file path",
            forced_backend
        ));
    };

    println!("[RUST] ✅ Pure Rust conversation complete");

    // Extract recalled memory IDs for later use
    let _recalled_memory_ids: Vec<i32> = recalled_memories.iter().map(|m| m.id).collect();

    // STEP 6.5: Check if LLM requested a web search
    let web_search_detected = reply_text.contains("WEB_SEARCH_NEEDED:");
    let web_search_query = if web_search_detected {
        println!("[RUST] LLM detected need for web search");

        // Extract the suggested search query
        if let Some(marker_pos) = reply_text.find("WEB_SEARCH_NEEDED:") {
            let after_marker = &reply_text[marker_pos + 18..]; // Skip "WEB_SEARCH_NEEDED:"
            let query_end = after_marker.find('\n').unwrap_or(after_marker.len());
            let query = after_marker[..query_end].trim().to_string();
            println!("[RUST] Suggested search query: {}", query);
            Some(query)
        } else {
            None
        }
    } else {
        None
    };

    // Parse MEMORY_EXTRACT facts from the LLM response — fires for any message type,
    // no is_question gate. Both API and local models use the same MEMORY_EXTRACT marker.
    let msg_lower = message.to_lowercase();
    let is_api = ConversationEngine::is_api_model(&forced_backend);

    let mut extracted_facts: Vec<String> = Vec::new();

    for line in reply_text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("MEMORY_EXTRACT:") {
            let fact = trimmed["MEMORY_EXTRACT:".len()..].trim().to_string();
            if !fact.is_empty() {
                extracted_facts.push(fact);
            }
        }
    }

    // Stopwords excluded from both safety filters — too common to serve as grounding evidence.
    let filter_stopwords: std::collections::HashSet<&str> = [
        "about", "some", "what", "your", "their", "that", "this", "with", "from",
        "have", "been", "they", "them", "will", "would", "could", "should", "does",
        "just", "more", "also", "when", "where", "there", "here", "very", "much",
        "good", "well", "like", "make", "time", "know", "want", "need", "back",
        "into", "over", "then", "than", "even", "only", "such", "each", "both",
    ].iter().copied().collect();

    // Build a shared set of content words from the user's message (>3 chars, not stopwords).
    // Used by both safety filters below.
    let msg_content_words: std::collections::HashSet<&str> = msg_lower
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| w.len() > 3 && !filter_stopwords.contains(w))
        .collect();

    if !extracted_facts.is_empty() {
        // Safety filter 1 — rephrasing guard: only applied to longer messages (≥8 content
        // words) where the heuristic is reliable. On shorter messages it produces false
        // rejections — "I've been feeling burnt out lately" → "Albert has been feeling burnt
        // out lately" shares 80% of words yet is a correct extraction. Threshold raised to
        // 75% (was 50%) for the same reason.
        if msg_content_words.len() >= 8 {
            extracted_facts.retain(|fact| {
                let fact_lower = fact.to_lowercase();
                let overlap = msg_content_words.iter()
                    .filter(|w| fact_lower.contains(*w))
                    .count();
                let keep = (overlap as f32 / msg_content_words.len() as f32) < 0.75;
                if !keep {
                    println!("[RUST] ⚠️ Discarding MEMORY_EXTRACT that rephrases the message: {}", &fact[..fact.len().min(80)]);
                }
                keep
            });
        }

        // Safety filter 2 — hallucination guard: the extracted fact must share at least one
        // content word with the user's message. A model that emits MEMORY_EXTRACT for
        // "How are you today?" and returns "Albert has been thinking about educational
        // pursuits" is hallucinating — nothing in the message grounds that claim.
        // This catches weak local models that fire on chitchat and invent the content.
        if !msg_content_words.is_empty() {
            extracted_facts.retain(|fact| {
                let fact_lower = fact.to_lowercase();
                let grounded = msg_content_words.iter().any(|w| fact_lower.contains(*w));
                if !grounded {
                    println!("[RUST] ⚠️ Discarding MEMORY_EXTRACT: fact shares no words with message — likely hallucination: {}", &fact[..fact.len().min(80)]);
                }
                grounded
            });
        }

        // Safety filter 3 — meta-question guard: local models (especially Qwen) sometimes
        // emit MEMORY_EXTRACT for questions, e.g. "Albert asked about the capital of France."
        // The rephrasing guard misses this on short questions (<8 content words) because the
        // threshold requires a longer message to be reliable. Explicitly reject facts that
        // describe the user asking rather than stating a personal fact.
        let meta_question_patterns = [
            "asked about", "asked what", "asked how", "asked why",
            "asked when", "asked where", "asked if", "asked whether",
            "wants to know", "inquired about", "is wondering",
            "wondered about", "is curious about", "was asking",
        ];
        extracted_facts.retain(|fact| {
            let fact_lower = fact.to_lowercase();
            let is_meta = meta_question_patterns.iter().any(|p| fact_lower.contains(p));
            if is_meta {
                println!("[RUST] ⚠️ Discarding meta-question MEMORY_EXTRACT: {}", &fact[..fact.len().min(80)]);
            }
            !is_meta
        });
    }

    if !extracted_facts.is_empty() {
        println!("[RUST] 💡 LLM extracted {} fact(s) from message", extracted_facts.len());
    }

    // Replace the generic "User" placeholder with the person's actual name.
    // Post-processing is more reliable than prompting — the LLM consistently
    // writes "User has a dog" as a placeholder regardless of prompt instructions.
    if let Some(ref name) = user_display_name {
        extracted_facts = extracted_facts
            .into_iter()
            .map(|fact| {
                fact.replace("User's ", &format!("{}'s ", name))
                    .replace("User ", &format!("{} ", name))
            })
            .collect();
    }

    // STEP 8: Convert memory::Memory to lib.rs Memory for response
    // Include linked memories in UI display so user can see what was pulled in via graph traversal
    let response_memories: Vec<Memory> = recalled_memories
        .iter()
        .chain(linked_memories.iter())
        .map(|mem| Memory {
            id: mem.id,
            title: mem.title.clone(),
            content: mem.content.clone(),
            source_type: mem.source_type.clone(),
            session_id: mem.session_id.clone(),
            user_id: mem.user_id.clone(),
            namespace: mem.namespace.clone(),
            is_syncable: Some(mem.is_syncable),
            is_shareable: Some(mem.is_shareable),
            created_at: mem.created_at.to_rfc3339(),
            updated_at: mem.updated_at.map(|dt| dt.to_rfc3339()),
            similarity: mem.similarity,
            event_type: mem.event_type.clone(),
            event_date: mem.event_date.map(|dt| dt.to_string()),
            link_count: Some(mem.link_count),
            is_ephemeral: Some(mem.is_ephemeral),
            expires_at: mem.expires_at.map(|dt| dt.to_string()),
            entities_detected: mem.entities_detected.clone(),
            original_text: mem.original_text.clone(),
        })
        .collect();

    // Strip WEB_SEARCH_NEEDED marker (and everything after it) from displayed text.
    // The marker is for internal detection only - users should see clean prose up to that point.
    let reply_text = if web_search_detected {
        if let Some(pos) = reply_text.find("WEB_SEARCH_NEEDED:") {
            reply_text[..pos].trim().to_string()
        } else {
            reply_text
        }
    } else {
        reply_text
    };

    // Strip <think>...</think> blocks produced by reasoning models (DeepSeek R1, Qwen3).
    // Three cases:
    //   1. Full block:   <think>...</think>\nresponse  → keep response
    //   2. No open tag:  reasoning...</think>\nresponse → keep response (DeepSeek: open tag
    //                    is in the injected prefix, not the generated output)
    //   3. Unclosed:     <think>reasoning...            → strip to end
    let reply_text = {
        let mut text = reply_text;
        loop {
            let think_start = text.find("<think>");
            let think_end   = text.find("</think>");
            match (think_start, think_end) {
                (Some(start), Some(end)) if start < end => {
                    // Case 1: properly wrapped block
                    let after = text[end + "</think>".len()..].trim_start().to_string();
                    println!("[RUST] Stripped <think> block from response");
                    text = after;
                }
                (None, Some(end)) => {
                    // Case 2: no opening tag — reasoning was injected as prefix
                    let after = text[end + "</think>".len()..].trim_start().to_string();
                    println!("[RUST] Stripped leading reasoning block (no open tag) from response");
                    text = after;
                }
                (Some(start), None) => {
                    // Case 3: unclosed <think> — strip from here to end
                    text = text[..start].trim_end().to_string();
                    println!("[RUST] Stripped unclosed <think> block from response");
                    break;
                }
                _ => break,
            }
        }
        text
    };

    // Strip MEMORY_EXTRACT lines and model meta-commentary from displayed response.
    // Weak local models (3B) sometimes append "Note: The assistant's response..." paragraphs
    // that expose internal prompt instructions. Truncate at the first such paragraph break.
    let reply_text = {
        // Truncate at "Note: The assistant" / "Note: The MEMORY_EXTRACT" meta-commentary.
        let truncated = if let Some(pos) = reply_text.find("\n\nNote: ") {
            &reply_text[..pos]
        } else if let Some(pos) = reply_text.find("\nNote: The assistant") {
            &reply_text[..pos]
        } else if let Some(pos) = reply_text.find("\nNote: The MEMORY") {
            &reply_text[..pos]
        } else {
            &reply_text
        };

        let filtered: Vec<String> = truncated
            .lines()
            .filter_map(|line| {
                let t = line.trim_start();
                if t.contains("MEMORY_EXTRACT:") { return None; }
                // Handle "PART N — ..." lines from local models following the two-part format.
                // If content follows the separator, keep it. If it's just a header, drop the line.
                if t.starts_with("PART ") && t.len() > 5 {
                    if t.chars().nth(5).map_or(false, |c| c.is_ascii_digit()) {
                        for sep in &[" \u{2014} ", " \u{2013} ", " - "] {
                            if let Some(pos) = t.find(sep) {
                                let content = t[pos + sep.len()..].trim_start();
                                return if content.is_empty() { None } else { Some(content.to_string()) };
                            }
                        }
                        return None;
                    }
                }
                Some(line.to_string())
            })
            .collect();
        let joined = filtered.join("\n");
        let trimmed = joined.trim().to_string();
        if trimmed != reply_text.trim() {
            println!("[RUST] Stripped MEMORY_EXTRACT line(s) and/or meta-commentary from displayed response");
        }
        trimmed
    };

    // Strip leading "{name}, " or "{name}: " added by weak local models that address
    // the user by name at the start of every single response despite being told not to.
    let reply_text = if let Some(ref name) = user_display_name {
        let prefix_comma = format!("{}, ", name);
        let prefix_colon = format!("{}: ", name);
        if reply_text.starts_with(&prefix_comma) {
            reply_text[prefix_comma.len()..].trim_start().to_string()
        } else if reply_text.starts_with(&prefix_colon) {
            reply_text[prefix_colon.len()..].trim_start().to_string()
        } else {
            reply_text
        }
    } else {
        reply_text
    };

    // Prepend Sovereign mode warning if present (with proper spacing)
    // Add medical disclaimer for HIPAA mode if health-related content detected
    let mut final_reply_text = if let Some(warning) = warning_prefix {
        format!("{}\n\n{}", warning, reply_text)
    } else {
        reply_text
    };

    // HIPAA Mode: Auto-add medical disclaimer if health-related terms detected
    if containment_mode.to_lowercase() == "hipaa" {
        let health_keywords = ["symptom", "treatment", "medication", "diagnosis", "disease",
                               "condition", "health", "medical", "doctor", "patient", "therapy"];
        let lower_reply = final_reply_text.to_lowercase();

        if health_keywords.iter().any(|keyword| lower_reply.contains(keyword)) {
            let disclaimer = "\n\n⚕️ AI-generated. Not a substitute for clinical judgment or current clinical guidelines.";
            final_reply_text.push_str(disclaimer);
        }
    }

    // STEP 9: RETURN RESPONSE IMMEDIATELY (before memory processing)
    let immediate_response = ReplyResponse {
        reply_text: final_reply_text.clone(),
        recalled_memories: Some(response_memories.clone()),
        model_backend: Some(forced_backend.clone()),
        containment_mode: Some(containment_mode.clone()),
        schema: None,
        blocked: Some(false),
        web_search_needed: web_search_query.as_ref().map(|_| true),
        web_search_query: web_search_query.clone(),
        original_query: Some(query.clone()),
    };

    // STEP 10: LOG EXCHANGE TO CONVERSATION HISTORY (non-blocking, skipped in HIPAA mode)
    if containment_mode.to_lowercase() != "hipaa" {
        let ch_session = session_id.clone();
        let ch_user = user_id.clone();
        let ch_message = query.clone();  // Store clean question, not file dump
        let ch_reply = final_reply_text.clone();
        let ch_backend = forced_backend.clone();
        let ch_mode = containment_mode.clone();
        tokio::spawn(async move {
            { let db_url = crate::db::get_db_url();
                match sqlx::SqlitePool::connect(&db_url).await {
                    Ok(pool) => {
                        if let Err(e) = conversation_history::log_exchange(
                            &pool, &ch_session, &ch_user, &ch_message,
                            &ch_reply, &ch_backend, &ch_mode,
                        ).await {
                            eprintln!("[ConvHistory] ⚠️ Failed to log exchange: {}", e);
                        }
                    }
                    Err(e) => eprintln!("[ConvHistory] ⚠️ DB pool error: {}", e),
                }
            }
        });
    }

    // STEP 11: BACKGROUND MEMORY PROCESSING (async, non-blocking)
    // This runs AFTER user has received their conversational response

    // HIPAA Mode: Disable memory extraction and storage entirely for compliance
    let hipaa_ephemeral_enforcement = containment_mode.to_lowercase() == "hipaa";
    let effective_skip_memory = skip_memory_storage.unwrap_or(false) || hipaa_ephemeral_enforcement;

    if hipaa_ephemeral_enforcement {
        println!("[HIPAA] 🔒 Ephemeral mode enforced - memory extraction and storage disabled");
    }

    let is_explicit_remember = query.trim().to_lowercase().starts_with("remember:");

    // Explicit "Remember:" commands: if the LLM didn't extract a fact, pull the content directly.
    if is_explicit_remember && extracted_facts.is_empty() {
        let remember_content = query.trim()["remember:".len()..].trim().to_string();
        if !remember_content.is_empty() {
            println!("[RUST] 📌 Explicit Remember: command — using content as extracted fact");
            extracted_facts.push(remember_content);
        }
    }

    // Gate 1: Reject pure trivial content (single-word acks, filler phrases, <3 words).
    // Everything else goes to the LLM — it decides what's actually worth storing.
    // Explicit "Remember:" commands always bypass this check.
    let memory_gate_passed = is_explicit_remember || engine.is_memory_worthy(&query);

    // MEMORY_EXTRACT text (if any) is carried into the background task — NOT stored immediately.
    // Storage happens only after contradiction detection completes.
    let bg_extracted_text: Option<String> = extracted_facts.into_iter().next();

    if !effective_skip_memory && (memory_gate_passed || bg_extracted_text.is_some()) {
        // Clone data needed for background task
        let bg_message = query.clone();  // Store clean question in memories, not file dump
        let bg_user_id = user_id.clone();
        let bg_session_id = session_id.clone();
        let bg_forced_backend = forced_backend.clone();
        let bg_containment_mode = containment_mode.clone();
        let bg_app = app.clone();
        let bg_is_api = is_api;
        let bg_is_explicit_remember = is_explicit_remember;
        // Pre-loaded model session for Call 2 (local models only — None for API backends).
        let bg_local_session = local_session_rx;

        // Spawn background task for memory processing
        tokio::spawn(async move {
            // Use extracted fact as content when MEMORY_EXTRACT fired; raw message otherwise.
            // The extracted fact is a clean, focused statement — better for storage and search.
            let factual_content = bg_extracted_text.clone().unwrap_or_else(|| bg_message.clone());

            // Generate embedding FIRST (needed for duplicate check and storage)
            let factual_clone = factual_content.clone();
            let message_embedding = match tokio::task::spawn_blocking(move || {
                llm::local_embeddings::generate_local_embedding(&factual_clone)
            })
            .await {
                Ok(Ok(embedding)) => embedding,
                Ok(Err(e)) => {
                    println!("[RUST BACKGROUND] ⚠️ Failed to generate embedding: {}", e);
                    return;
                }
                Err(e) => {
                    println!("[RUST BACKGROUND] ⚠️ Embedding task panicked: {}", e);
                    return;
                }
            };

            // Reconnect to database for memory storage
            let db_url = crate::db::get_db_url();

            let db_pool = match sqlx::SqlitePool::connect(&db_url).await {
                Ok(pool) => pool,
                Err(e) => {
                    println!("[RUST BACKGROUND] ⚠️ Failed to connect to database: {}", e);
                    return;
                }
            };

            // Do a BROADER search for relationship detection (more memories, lower threshold)
            // This ensures we catch relationships even if the memory wasn't in the top 5 for conversation

            let relationship_search_results = match memory::hybrid_search(
                &db_pool,
                message_embedding.clone(),
                query_entities.clone(),
                Some(&bg_user_id),
                None,  // All sessions
                None,  // All namespaces
                15,    // Search more memories (15 instead of 5)
            ).await {
                Ok(results) => results,
                Err(e) => {
                    println!("[RUST BACKGROUND] ⚠️ Relationship search failed: {}", e);
                    db_pool.close().await;
                    return;
                }
            };

            // Filter to >35% similarity — matches the hybrid search floor. The LLM handles
            // false candidates correctly (returns NONE), so a conservative pre-filter here
            // only causes missed relationships like niece/nephews at the same event (41%).
            let mut similar_memories: Vec<(i32, String, Option<String>, f32)> = relationship_search_results
                .into_iter()
                .filter(|m| m.similarity.unwrap_or(0.0) >= 0.35)
                .map(|m| (m.id, m.content.clone(), m.title.clone(), m.similarity.unwrap_or(0.0) as f32))
                .collect();

            // Sort by similarity (most relevant first) for relationship classification
            // We want the MOST similar memories, not the most recent ones
            similar_memories.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

            // API models handle larger candidate sets well; local 7B models degrade with more context
            let relationship_candidate_limit = if bg_is_api { 10 } else { 6 };
            similar_memories.truncate(relationship_candidate_limit);

            if !similar_memories.is_empty() {
                println!("[RUST BACKGROUND] Found {} similar memories (>35% similarity) for relationship classification", similar_memories.len());
            }

            // DUPLICATE CHECK: Check if this is a duplicate
            // First check hybrid score (>98%), then pure cosine (>93%)
            // Hybrid score of 1.0 = exact match even if pure cosine is slightly lower due to entity boosting

            let mut is_duplicate = false;
            for (mem_id, _mem_content, _mem_title, hybrid_score) in &similar_memories {
                // Check 1: Very high hybrid score indicates duplicate (entity + semantic match)
                if *hybrid_score > 0.98 {
                    println!("[RUST BACKGROUND] 🔄 DUPLICATE DETECTED: Memory {} has {:.1}% hybrid similarity",
                             mem_id, hybrid_score * 100.0);
                    println!("[RUST BACKGROUND] Skipping memory storage for duplicate");
                    is_duplicate = true;
                    break;
                }

                // Check 2: Pure cosine similarity (lowered threshold to 0.93 to catch near-duplicates)
                if let Ok(Some(candidate_mem)) = memory::get_memory(&db_pool, *mem_id).await {
                    if let Some(ref candidate_embedding_vec) = candidate_mem.embedding {
                        let pure_similarity = crate::llm::local_embeddings::cosine_similarity(
                            &message_embedding,
                            candidate_embedding_vec
                        );

                        if pure_similarity > 0.93 {
                            println!("[RUST BACKGROUND] 🔄 DUPLICATE DETECTED: Memory {} has {:.1}% pure cosine similarity",
                                     mem_id, pure_similarity * 100.0);
                            println!("[RUST BACKGROUND] Skipping memory storage for duplicate");
                            is_duplicate = true;
                            break;
                        }
                    }
                }
            }

            if is_duplicate {
                db_pool.close().await;
                return;
            }

            // Local models: only proceed when MEMORY_EXTRACT fired — that's the storage
            // decision for local. No extracted text means nothing to store.
            // API models: always proceed to Call 2, which makes the should_remember decision.
            if bg_extracted_text.is_none() && !bg_is_api {
                db_pool.close().await;
                return;
            }

            println!("[RUST BACKGROUND] ✅ Proceeding with relationship classification");

            // Paired-call: receive the pre-loaded model session from the main call.
            // By the time we reach here, the session was already sent (before the background
            // task was spawned), so this await completes instantly.
            let local_session = if let Some(rx) = bg_local_session {
                match rx.await {
                    Ok(session) => {
                        println!("[RUST BACKGROUND] ✅ Received pre-loaded model session for Call 2");
                        Some(session)
                    }
                    Err(_) => {
                        println!("[RUST BACKGROUND] ⚠️ Model session unavailable — will load fresh for Call 2");
                        None
                    }
                }
            } else {
                None
            };

            // Local: MEMORY_EXTRACT fired → relationships + title only (should_remember=true).
            // API: full should_remember + relationship classification.
            let (should_remember, llm_title, llm_relationships) = if !bg_is_api {
                match ask_llm_for_relationships(&factual_content, &similar_memories, &bg_forced_backend, local_session).await {
                    Ok((title, rels)) => {
                        println!("[RUST BACKGROUND] ✅ Relationship classifier: {} relationships", rels.len());
                        (true, title, rels)
                    }
                    Err(e) => {
                        println!("[RUST BACKGROUND] ⚠️ Relationship classifier failed: {} — storing without links", e);
                        let fallback = generate_title_from_content(&factual_content);
                        (true, Some(fallback), Vec::new())
                    }
                }
            } else {
                // API: ask_llm_about_memory_with_relationships (should_remember + relationships)
                match ask_llm_about_memory_with_relationships(
                    &bg_message,
                    "Background memory processing",
                    &similar_memories,
                    &bg_forced_backend,
                ).await {
                    Ok(result) => result,
                    Err(e) => {
                        println!("[RUST BACKGROUND] ⚠️ LLM call failed: {}", e);
                        if !bg_forced_backend.contains("local") && !bg_forced_backend.ends_with(".gguf") {
                            match ask_llm_about_memory_with_relationships(
                                &bg_message,
                                "Background memory processing",
                                &similar_memories,
                                "local",
                            ).await {
                                Ok(result) => result,
                                Err(e2) => {
                                    println!("[RUST BACKGROUND] ⚠️ Fallback also failed: {}", e2);
                                    (false, None, Vec::new())
                                }
                            }
                        } else {
                            (false, None, Vec::new())
                        }
                    }
                }
            };

            // Explicit "Remember:" command overrides LLM decision — user is the authority
            let (should_remember, llm_title) = if bg_is_explicit_remember && !should_remember {
                println!("[RUST BACKGROUND] ✅ Explicit 'Remember:' command — overriding LLM decision to store");
                let fallback_title = factual_content.chars().take(60).collect::<String>();
                let fallback_title = if factual_content.len() > 60 {
                    format!("{}…", fallback_title)
                } else {
                    fallback_title
                };
                (true, Some(fallback_title))
            } else {
                (should_remember, llm_title)
            };

            if !should_remember {
                db_pool.close().await;
                return;
            }

            println!("[RUST BACKGROUND] ✅ LLM decided to remember: {:?}", llm_title);

            // CHECK FOR CONTRADICTIONS (BEFORE STORAGE!)
            // If LLM detected contradiction with sufficient confidence, emit event to frontend
            // Require >= 0.65 to avoid false positives from local models
            let contradiction_detected = llm_relationships.iter()
                .any(|rel| rel.relationship_type == "contradicts" && rel.confidence.unwrap_or(0.0) >= 0.65);

            if contradiction_detected {
                println!("[RUST BACKGROUND] ⚠️ CONTRADICTION DETECTED - emitting event to frontend (NOT storing yet)");

                // Find the contradicting relationship
                if let Some(contradiction) = llm_relationships.iter()
                    .find(|rel| rel.relationship_type == "contradicts") {

                    // Fetch the conflicting memory details
                    if let Ok(Some(conflicting_memory)) = memory::get_memory(&db_pool, contradiction.memory_id).await {
                        // Prepare payload for frontend (memoryA = OLD, memoryB = NEW)
                        let payload = serde_json::json!({
                            "memoryA": {
                                "id": conflicting_memory.id,
                                "content": conflicting_memory.content,
                                "title": conflicting_memory.title,
                                "created_at": conflicting_memory.created_at.to_rfc3339(),
                            },
                            "memoryB": {
                                "content": factual_content.clone(),
                                "title": llm_title.clone(),
                                "created_at": chrono::Utc::now().to_rfc3339(),
                            },
                            "reason": contradiction.reason,
                            "confidence": contradiction.confidence.unwrap_or(0.75),
                            "pending_memory": {
                                "content": factual_content.clone(),
                                "title": llm_title.clone(),
                                "embedding": message_embedding.clone(),
                            },
                            "relationships": llm_relationships,
                            "user_id": bg_user_id.clone(),
                            "session_id": bg_session_id.clone(),
                        });

                        // Emit event to frontend to show modal
                        if let Err(e) = bg_app.emit("contradiction-detected", payload) {
                            println!("[RUST BACKGROUND] ⚠️ Failed to emit contradiction event: {}", e);
                        } else {
                            println!("[RUST BACKGROUND] 🔔 Contradiction event emitted - modal should appear");
                        }

                        // Don't store memory yet - wait for user decision via resolve_memory_conflict_v2
                        db_pool.close().await;
                        return;
                    }
                }
            }

            // HIPAA mode: Enable ephemeral memory (8-hour expiration)
            let _is_ephemeral = bg_containment_mode == "hipaa";

            // Extract entities using NLP enhancer (no tags - entities handle everything)
            // Note: We use LLM-generated title, NLP for entities/events
            let message_for_nlp = factual_content.clone();
            let (entities, event_type, event_date, namespace) = match tokio::task::spawn_blocking(move || {
                let enhancer = nlp_enhancer::NLPEnhancer::new();

                // Generate namespace and detect events
                let enhancement = enhancer.enhance(&message_for_nlp);

                // Extract entities (includes both proper nouns AND common nouns)
                let ents = enhancer.extract_entities(&message_for_nlp);
                let entities_json = serde_json::json!(ents.iter().map(|e| {
                    serde_json::json!({
                        "word": e.word,
                        "label": e.label,
                        "score": e.score,
                        "start": e.start,
                        "end": e.end
                    })
                }).collect::<Vec<_>>());

                (entities_json, enhancement.event_type, enhancement.event_date, enhancement.namespace)
            })
            .await {
                Ok(result) => result,
                Err(e) => {
                    println!("[RUST BACKGROUND] ⚠️ Failed to enhance memory: {}", e);
                    db_pool.close().await;
                    return;
                }
            };

            // Store memory in database (with LLM-generated title, NLP entities, and events!)
            let memory_id = match memory::insert_memory(
                &db_pool,
                llm_title.as_deref(),              // title (LLM-generated!)
                &factual_content,                  // content (factual statements only!)
                Some("conversation"),              // source_type
                Some(&bg_session_id),              // session_id
                Some(message_embedding),           // embedding
                None,                              // parent_scroll_id
                None,                              // chunk_index
                Some(&bg_user_id),                 // user_id
                &namespace,                        // namespace (NLP-detected)
                true,                              // is_syncable
                false,                             // is_shareable
                Some(entities),                    // entities_detected (includes proper + common nouns!)
                event_type.as_deref(),             // event_type (auto-detected!)
                event_date,                        // event_date (auto-extracted!)
                Some(&bg_message),                 // original_text (FULL original user message for context!)
            ).await {
                Ok(id) => {
                    println!("[RUST BACKGROUND] ✅ Memory stored with ID: {}", id);
                    id
                }
                Err(e) => {
                    println!("[RUST BACKGROUND] ⚠️ Failed to store memory: {}", e);
                    db_pool.close().await;
                    return;
                }
            };

            // Store LLM-classified relationships (if any)
            if !llm_relationships.is_empty() {
                println!("[RUST BACKGROUND] Creating {} LLM-classified relationships...", llm_relationships.len());

                for rel in &llm_relationships {
                    // Skip "none" relationships
                    if rel.relationship_type == "none" {
                        continue;
                    }

                    // Create relationship in database
                    match memory::create_memory_link(
                        &db_pool,
                        memory_id,
                        rel.memory_id,
                        &rel.relationship_type,
                        rel.confidence.unwrap_or(0.75),
                        Some(&rel.reason),
                        "llm",
                    ).await {
                        Ok(link_id) => {
                            println!("[RUST BACKGROUND] ✅ Created {} relationship (link #{}): {} -> {}",
                                     rel.relationship_type, link_id, memory_id, rel.memory_id);
                        }
                        Err(e) => {
                            println!("[RUST BACKGROUND] ⚠️ Failed to create relationship: {}", e);
                        }
                    }
                }

                // Update timestamps to trigger sync
                let relationship_count = llm_relationships.iter()
                    .filter(|r| r.relationship_type != "none" && r.memory_id > 0)
                    .count();

                if relationship_count > 0 {
                    let memory_ids: Vec<i32> = llm_relationships.iter()
                        .filter(|r| r.relationship_type != "none" && r.memory_id > 0)
                        .map(|r| r.memory_id)
                        .chain(std::iter::once(memory_id))
                        .collect();

                    for mem_id in memory_ids {
                        let _ = sqlx::query("UPDATE memories SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?")
                            .bind(mem_id)
                            .execute(&db_pool)
                            .await;
                    }
                    println!("[RUST BACKGROUND] ✅ Updated {} memory timestamps to trigger relationship sync", relationship_count + 1);
                }
            }

            // Relationship classification complete
            if !llm_relationships.is_empty() {
                println!("[RUST BACKGROUND] ✅ Classified {} relationships", llm_relationships.len());
            }

            db_pool.close().await;
            println!("[RUST BACKGROUND] 🏁 Memory processing complete");
        });
    }

    // Return immediate response to user
    Ok(immediate_response)
}

// The legacy rule-based relationship detection code has been removed.
// Memory processing now happens entirely in the background task above.

#[tauri::command]
async fn run_ensemble(
    _app: tauri::AppHandle,
    message: String,
    models: Vec<String>,
    user_id: String,
    _session_id: String,
    containment_mode: String,
    kb_enabled: Option<bool>,
    user_query: Option<String>,
) -> Result<serde_json::Value, String> {
    println!("[Ensemble] Running multi-model ensemble with {} models", models.len());
    println!("[Ensemble] Containment mode: {}", containment_mode);

    if models.len() < 2 {
        return Err("Need at least 2 models for ensemble".to_string());
    }

    // ENSEMBLE MODE: Lightweight research tool
    // - Includes: memory context, KB context, web search
    // - Excludes: containment checks (UI blocks child mode), memory storage, relationship detection

    // Canonical API model names — update here to change across all ensemble phases
    const ANTHROPIC_MODEL: &str = "claude-sonnet-4-6";
    const OPENAI_MODEL: &str = "gpt-4o";
    const XAI_MODEL: &str = "grok-3";

    // Determine coordinator model upfront: Anthropic > xAI > OpenAI > first local model
    let coordinator_model = models.iter()
        .find(|m| m.to_lowercase().contains("anthropic"))
        .or_else(|| models.iter().find(|m| m.to_lowercase().contains("xai") || m.to_lowercase().contains("grok")))
        .or_else(|| models.iter().find(|m| m.to_lowercase().contains("openai") || m.to_lowercase().contains("gpt")))
        .unwrap_or(&models[0])
        .clone();

    // Get database connection
    let database_url = crate::db::get_db_url();
    let db_pool = sqlx::SqlitePool::connect(&database_url)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    // Use clean user_query for memory/KB search when a file is attached (message may contain file dump)
    let search_query = user_query.as_deref().unwrap_or(&message).to_string();

    // Lightweight context gathering (no containment, no storage, no relationships)

    // Extract entities from question for memory search (use spawn_blocking for CPU-intensive work)
    let message_for_nlp = search_query.clone();
    let query_entities = tokio::task::spawn_blocking(move || {
        let enhancer = nlp_enhancer::NLPEnhancer::new();
        enhancer.extract_entities(&message_for_nlp)
    })
    .await
    .map_err(|e| format!("Entity extraction failed: {}", e))?;

    // Convert entities to strings for hybrid search
    let query_entity_strings: Vec<String> = query_entities.iter().map(|e| e.word.clone()).collect();

    // Generate embedding for hybrid search
    let message_for_embedding = search_query.clone();
    let query_embedding = tokio::task::spawn_blocking(move || {
        llm::local_embeddings::generate_local_embedding(&message_for_embedding)
    })
    .await
    .map_err(|e| format!("Failed to run embedding task: {}", e))?
    .map_err(|e| format!("Failed to generate embedding: {}", e))?;

    // Search for relevant memories using hybrid search
    let recalled_memories = memory::hybrid_search(
        &db_pool,
        query_embedding,
        query_entity_strings,
        Some(&user_id),
        None, // No session filter - search all sessions
        None, // No namespace filter
        5     // Top 5 relevant memories
    )
    .await
    .unwrap_or_else(|e| {
        println!("[Ensemble] Memory search failed: {}", e);
        Vec::new()
    });
    // Build context string from memories
    let mut context = String::new();

    if !recalled_memories.is_empty() {
        context.push_str("\n\n[RELEVANT CONTEXT FROM YOUR MEMORIES]\n");
        for (idx, mem) in recalled_memories.iter().enumerate() {
            context.push_str(&format!("{}. {}\n", idx + 1, mem.content));
        }
    }

    // KB context (if enabled)
    if kb_enabled.unwrap_or(false) {
        match kb_rag::search_kb_chunks(&db_pool, &user_id, &search_query, 8, false).await {
            Ok(kb_results) => {
                let relevant_chunks: Vec<_> = kb_results.iter()
                    .filter(|r| r.similarity_score > 0.15)
                    .cloned()
                    .collect();
                let to_use = if relevant_chunks.is_empty() {
                    kb_results.into_iter().take(5).collect::<Vec<_>>()
                } else {
                    relevant_chunks
                };
                if !to_use.is_empty() {
                    context.push_str("\n\n[KNOWLEDGE BASE DOCUMENTS]\n");
                    context.push_str("The following documents were retrieved from the user's Knowledge Base. Use this information in your answer.\n\n");
                    for (idx, chunk) in to_use.iter().enumerate() {
                        context.push_str(&format!(
                            "📄 Document {}: {} (relevance: {:.1}%)\n{}\n\n",
                            idx + 1,
                            chunk.file_name,
                            chunk.similarity_score * 100.0,
                            chunk.content
                        ));
                    }
                    context.push_str("[END KNOWLEDGE BASE DOCUMENTS]\n");
                    println!("[Ensemble] Added {} KB chunks to context", to_use.len());
                } else {
                    println!("[Ensemble] KB search returned no results");
                }
            }
            Err(e) => println!("[Ensemble] KB search failed (non-fatal): {}", e),
        }
    }

    // Phase 0: Assess if question needs web search
    println!("[Ensemble] Phase 0: Assessing if question needs web search");

    let assessment_prompt = format!(
        "Question: \"{}\"\n\n\
        Does this question require CURRENT, UP-TO-DATE information that may have changed recently?\n\n\
        Examples that need web search:\n\
        - Latest version of software/library (\"What is the latest version of React?\")\n\
        - Current events, news, or recent developments\n\
        - Current prices, statistics, or numbers that change over time\n\
        - Recent releases, updates, or announcements\n\n\
        Examples that DON'T need web search:\n\
        - Conceptual questions (\"How does async/await work?\")\n\
        - Historical facts with fixed dates (\"When was Python created?\")\n\
        - How-to questions about established technology\n\
        - Opinion or advice questions\n\n\
        Respond with ONLY:\n\
        - 'NO' if the question can be answered with existing knowledge\n\
        - 'YES: <brief search query>' if current information is needed\n\n\
        Example: 'YES: React latest stable version 2026'",
        message
    );

    let engine = conversation_engine::ConversationEngine::new();
    let assessment_full_prompt = engine.build_prompt(&assessment_prompt, None, None, true, None);

    // Use coordinator to assess
    let needs_search = if coordinator_model.to_lowercase().contains("anthropic") {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
        if !api_key.is_empty() {
            let messages = vec![llm::Message {
                role: "user".to_string(),
                content: assessment_full_prompt.clone(),
            }];
            llm::anthropic::send_message(&api_key, ANTHROPIC_MODEL, messages, None, Some(256), None)
                .await
                .map(|r| r.content)
                .ok()
        } else {
            None
        }
    } else if coordinator_model.to_lowercase().contains("openai") {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        if !api_key.is_empty() {
            let messages = vec![llm::Message {
                role: "user".to_string(),
                content: assessment_full_prompt.clone(),
            }];
            llm::openai::send_message(&api_key, OPENAI_MODEL, messages, Some(256), None)
                .await
                .map(|r| r.content)
                .ok()
        } else {
            None
        }
    } else {
        None
    };

    // If assessment says yes, do web search
    let mut search_results_text = String::new();
    let mut search_results_json = serde_json::Value::Null;

    if let Some(assessment) = needs_search {
        if assessment.to_uppercase().starts_with("YES") {
            println!("[Ensemble] 🔍 Question needs current information");

            // Extract search query from assessment (format: "YES: query here")
            // Take text after "YES:" but stop at first newline or period (to avoid extra explanation)
            let search_query = if assessment.contains(':') {
                let after_colon = assessment.split(':').nth(1).unwrap_or(&message);
                // Stop at first newline, period, or "because" to get just the query
                let query = after_colon
                    .split('\n').next().unwrap_or(after_colon)
                    .split('.').next().unwrap_or(after_colon)
                    .split(" because").next().unwrap_or(after_colon)
                    .split(" The ").next().unwrap_or(after_colon)
                    .trim();
                query
            } else {
                &message
            };

            println!("[Ensemble] 🔍 Searching web for: {}", search_query);

            // Trigger web search with 5-second timeout
            match tokio::time::timeout(
                std::time::Duration::from_secs(5),
                crate::web_search::search_duckduckgo(search_query, 3)
            ).await {
                Ok(Ok(search_response)) => {
                    println!("[Ensemble] ✅ Got {} search results", search_response.num_results);
                    search_results_text = format!("\n\n[CURRENT INFORMATION FROM WEB SEARCH - Use this to answer the question with up-to-date facts]\nQuery: \"{}\"\n\n", search_query);
                    for (idx, result) in search_response.results.iter().take(3).enumerate() {
                        search_results_text.push_str(&format!(
                            "Source {}: {}\n{}\n{}\n\n",
                            idx + 1,
                            result.title,
                            result.url,
                            result.snippet
                        ));
                    }
                    // Store search results for UI display
                    search_results_json = serde_json::to_value(&search_response).unwrap_or(serde_json::Value::Null);
                }
                Ok(Err(e)) => {
                    println!("[Ensemble] ⚠️ Web search failed: {}", e);
                }
                Err(_) => {
                    println!("[Ensemble] ⚠️ Web search timed out");
                }
            }
        } else {
            println!("[Ensemble] ✅ Question can be answered with existing knowledge");
        }
    }

    // Phase 1: Get responses from all models (direct API calls - no memory storage, no web search capability)
    println!("[Ensemble] Phase 1: Collecting responses from {} models", models.len());
    let mut individual_responses = Vec::new();

    // Build complete prompt with context + question + search results
    let mut full_question = String::new();

    // Add memories and KB context first
    if !context.is_empty() {
        full_question.push_str(&context);
        full_question.push_str("\n\n");
    }

    // Add web search results if available
    if !search_results_text.is_empty() {
        full_question.push_str(&search_results_text);
        full_question.push_str("\n\n");
    }

    // Add the actual question
    full_question.push_str(&format!("Question: {}", message));

    // Call each model directly (no send_message_with_memory complexity)
    for model_backend in &models {
        println!("[Ensemble] Querying model: {}", model_backend);

        let response_result = if model_backend.to_lowercase().contains("anthropic") {
            let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
            if api_key.is_empty() {
                Err("ANTHROPIC_API_KEY not set".to_string())
            } else {
                let messages = vec![llm::Message {
                    role: "user".to_string(),
                    content: full_question.clone(),
                }];
                llm::anthropic::send_message(&api_key, ANTHROPIC_MODEL, messages, None, Some(4096), None)
                    .await
                    .map(|r| r.content)
                    .map_err(|e| e.to_string())
            }
        } else if model_backend.to_lowercase().contains("openai") || model_backend.to_lowercase().contains("gpt") {
            let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
            if api_key.is_empty() {
                Err("OPENAI_API_KEY not set".to_string())
            } else {
                let messages = vec![llm::Message {
                    role: "user".to_string(),
                    content: full_question.clone(),
                }];
                llm::openai::send_message(&api_key, OPENAI_MODEL, messages, Some(4096), None)
                    .await
                    .map(|r| r.content)
                    .map_err(|e| e.to_string())
            }
        } else if model_backend.to_lowercase().contains("xai") || model_backend.to_lowercase().contains("grok") {
            let api_key = std::env::var("XAI_API_KEY").unwrap_or_default();
            if api_key.is_empty() {
                Err("XAI_API_KEY not set".to_string())
            } else {
                let messages = vec![llm::Message {
                    role: "user".to_string(),
                    content: full_question.clone(),
                }];
                llm::xai::send_message(&api_key, XAI_MODEL, messages, Some(4096), None)
                    .await
                    .map(|r| r.content)
                    .map_err(|e| e.to_string())
            }
        } else {
            // Local model - use llama.cpp
            if !std::path::Path::new(model_backend).exists() {
                Err(format!("Local model file not found: {}", model_backend))
            } else {
                let model_path = model_backend.clone();
                let question = full_question.clone();

                // Call local model in blocking task (CPU-bound inference)
                let result = tokio::task::spawn_blocking(move || {
                    let messages = vec![llm::Message {
                        role: "user".to_string(),
                        content: question,
                    }];

                    llm::local_models::generate_with_local_model(
                        &model_path,
                        messages,
                        Some(512),  // Reasonable token limit for ensemble responses
                        None,       // Default temperature
                    )
                })
                .await
                .map_err(|e| format!("Local model task failed: {}", e))?;

                result.map(|r| r.content).map_err(|e| e.to_string())
            }
        };

        match response_result {
            Ok(response) => {
                individual_responses.push(serde_json::json!({
                    "model": model_backend,
                    "response": response,
                    "success": true
                }));
            }
            Err(e) => {
                println!("[Ensemble] Model {} failed: {}", model_backend, e);
                individual_responses.push(serde_json::json!({
                    "model": model_backend,
                    "response": format!("Error: {}", e),
                    "success": false
                }));
            }
        }
    }

    // Filter successful responses
    let successful_responses: Vec<&serde_json::Value> = individual_responses
        .iter()
        .filter(|r| r["success"].as_bool().unwrap_or(false))
        .collect();

    if successful_responses.len() < 2 {
        return Err("Not enough successful responses for ensemble".to_string());
    }

    // Phase 2: Synthesis - Coordinator combines all responses
    println!("[Ensemble] Phase 2: Synthesizing best answer");
    println!("[Ensemble] Using coordinator model: {}", coordinator_model);

    // Build synthesis prompt with all responses
    let mut synthesis_prompt = format!("Original Question: {}\n\n", message);

    // Note if web search was performed
    if !search_results_text.is_empty() {
        synthesis_prompt.push_str("[Note: Models were provided with current web search results]\n\n");
    }

    synthesis_prompt.push_str("AGENT RESPONSES:\n");
    for (idx, response) in successful_responses.iter().enumerate() {
        let model = response["model"].as_str().unwrap_or("unknown");
        let answer = response["response"].as_str().unwrap_or("");
        synthesis_prompt.push_str(&format!("\nModel {} ({}):\n{}\n", idx + 1, model, answer));
    }

    synthesis_prompt.push_str("\n\nYou are the ensemble coordinator. Your job is to EVALUATE the responses and produce the best answer, NOT to average them.\n\n");
    synthesis_prompt.push_str("PROCESS:\n");
    synthesis_prompt.push_str("1. Identify where models AGREE (high confidence), DISAGREE (choose the better-supported position), or where one adds unique important detail\n");
    synthesis_prompt.push_str("2. Multi-model agreement = more reliable. When models disagree on specific facts (dates, APIs, library names, etc.) with no clear winner: state it's uncertain or omit if not essential\n");
    synthesis_prompt.push_str("3. Do NOT introduce new specific facts (library names, file paths, versions, API endpoints, product names) that don't appear in ANY response\n");
    synthesis_prompt.push_str("4. Prefer conservative, privacy-preserving, local-first approaches when multiple options exist\n");
    synthesis_prompt.push_str("5. Be concise but accurate - prioritize correctness over brevity\n");
    synthesis_prompt.push_str("6. CRITICAL - Time-sensitive and version-specific claims:\n");
    synthesis_prompt.push_str("   - If the question asks about future dates (e.g., '2026') or versions beyond your training data, treat ALL specific version numbers, release years, and named future features as POTENTIALLY SPECULATIVE\n");
    synthesis_prompt.push_str("   - Be especially cautious about: specific version numbers (e.g., 'WebAssembly 3.0'), exact future release years, named standards or products that may not exist yet\n");
    synthesis_prompt.push_str("   - Do NOT treat multi-model consensus on these future details as proof they are correct - models can share the same hallucination\n");
    synthesis_prompt.push_str("   - Either OMIT speculative version/date claims from your answer, OR explicitly mark them as 'models suggest... but this may be speculative'\n");
    synthesis_prompt.push_str("   - You SHOULD still provide timeless, practical guidance (e.g., 'when to use X vs Y') even if specific future details are uncertain\n\n");
    synthesis_prompt.push_str("OUTPUT FORMAT (required):\n\n");
    synthesis_prompt.push_str("**CONSENSUS & UNCERTAINTY** (2-4 brief bullets):\n");
    synthesis_prompt.push_str("- Strong consensus (verified or timeless): [facts all models agree on that are well-supported]\n");
    synthesis_prompt.push_str("- Uncertain/speculative claims: [version numbers, future dates, or details that may be hallucinated - mark as speculative]\n");
    synthesis_prompt.push_str("- Key disagreements: [where models differ on approach, opinion, or trade-offs]\n\n");
    synthesis_prompt.push_str("**SYNTHESIZED ANSWER:**\n");
    synthesis_prompt.push_str("[Your unified answer here, incorporating consensus and resolving disagreements]\n\n");
    synthesis_prompt.push_str("Keep the consensus section brief (2-3 bullets total). Do not copy verbatim - synthesize into a unified voice.\n\n");
    synthesis_prompt.push_str("Provide your synthesis directly without any system markers or requests:");

    // For synthesis, we skip ALL pre-checks and memory search - just get the LLM response
    // Build the prompt directly without memory context
    let engine = conversation_engine::ConversationEngine::new();
    let full_prompt = engine.build_prompt(&synthesis_prompt, None, None, true, None);

    // Call LLM directly without all the overhead
    let reply_text = if coordinator_model.to_lowercase().contains("anthropic") {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY not set".to_string())?;
        let model_name = ANTHROPIC_MODEL;
        let messages = vec![llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];
        let response = llm::anthropic::send_message(&api_key, model_name, messages, None, Some(4096), None)
            .await
            .map_err(|e| e.to_string())?;
        response.content
    } else if coordinator_model.to_lowercase().contains("xai") || coordinator_model.to_lowercase().contains("grok") {
        let api_key = std::env::var("XAI_API_KEY")
            .map_err(|_| "XAI_API_KEY not set".to_string())?;
        let model_name = XAI_MODEL;
        let messages = vec![llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];
        let response = llm::xai::send_message(&api_key, model_name, messages, Some(4096), None)
            .await
            .map_err(|e| e.to_string())?;
        response.content
    } else if coordinator_model.to_lowercase().contains("openai") || coordinator_model.to_lowercase().contains("gpt") {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY not set".to_string())?;
        let model_name = OPENAI_MODEL;
        let messages = vec![llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];
        let response = llm::openai::send_message(&api_key, model_name, messages, Some(4096), None)
            .await
            .map_err(|e| e.to_string())?;
        response.content
    } else {
        // Local model coordinator — enables fully offline ensemble
        println!("[Ensemble] Using local model as coordinator: {}", coordinator_model);
        let model_path = if coordinator_model.ends_with(".gguf") {
            coordinator_model.clone()
        } else {
            std::env::var("LOCAL_MODEL_PATH")
                .unwrap_or_else(|_| "models/user/Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string())
        };
        let messages = vec![llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];
        let model_path_clone = model_path.clone();
        let response = tokio::task::spawn_blocking(move || {
            llm::local_models::generate_with_local_model(&model_path_clone, messages, Some(2048), None)
        })
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
        response.content
    };

    let synthesized_response = reply_text;

    // Clean up database connection
    db_pool.close().await;

    println!("[Ensemble] Ensemble complete!");

    Ok(serde_json::json!({
        "individual_responses": individual_responses,
        "synthesized_response": synthesized_response,
        "coordinator_model": coordinator_model,
        "models_used": models.len(),
        "successful_models": successful_responses.len(),
        "search_results": search_results_json
    }))
}

/// List memories - Pure Rust implementation with direct DB access
// list_memories → commands/memory.rs
/// Insert a memory - Pure Rust implementation with local embedding generation
#[allow(clippy::too_many_arguments)]
pub(crate) async fn insert_memory(
    title: Option<String>,
    content: String,
    source_type: Option<String>,
    session_id: Option<String>,
    user_id: Option<String>,
    _tags: Option<Vec<String>>,
    namespace: String,
    is_syncable: bool,
    is_shareable: bool,
    is_ephemeral: Option<bool>,
    ephemeral_hours: Option<f64>,
) -> Result<i32, String> {
    println!("[Rust] insert_memory called - content length: {}", content.len());

    // Extract entities using Candle BERT NER (pure Rust ML)
    println!("[Rust] Extracting entities with Candle BERT NER...");
    let entities = match commands::nlp::extract_entities(content.clone()).await {
        Ok(ents) => {
            println!("[Rust] ✅ NER extraction successful - {} entities", ents.as_array().map(|a| a.len()).unwrap_or(0));
            ents
        }
        Err(e) => {
            eprintln!("[Rust] ⚠️ NER extraction failed: {}", e);
            serde_json::json!([])  // Fallback to empty array
        }
    };

    // Generate embedding using local candle model
    println!("[Rust] Generating local embedding...");
    let content_clone = content.clone();
    let embedding = tokio::task::spawn_blocking(move || {
        llm::local_embeddings::generate_local_embedding(&content_clone)
    })
    .await
    .map_err(|e| format!("Failed to run embedding task: {}", e))?
    .map_err(|e| format!("Failed to generate embedding: {}", e))?;

    println!("[Rust] Generated 384-dim embedding");

    // Calculate ephemeral expiry time if needed
    let expires_at = if is_ephemeral.unwrap_or(false) {
        let hours = ephemeral_hours.unwrap_or(24.0);
        let duration = chrono::Duration::milliseconds((hours * 3600.0 * 1000.0) as i64);
        Some(chrono::Utc::now() + duration)
    } else {
        None
    };

    // Connect to database
    let db_url = crate::db::get_db_url();

    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    // Convert embedding to pgvector::Vector
    let embedding_vec = embedding.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();

    // Insert memory into database (WITH entities cached!)
    let memory_id = sqlx::query_scalar::<_, i32>(
        "INSERT INTO memories (
            title, content, source_type, session_id, embedding,
            user_id, namespace, is_syncable, is_shareable,
            is_ephemeral, expires_at, entities_detected, event_type, event_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id"
    )
    .bind(title.as_deref())
    .bind(&content)
    .bind(source_type.as_deref())
    .bind(session_id.as_deref())
    .bind(&embedding_vec)
    .bind(user_id.as_deref())
    .bind(&namespace)
    .bind(is_syncable)
    .bind(is_shareable)
    .bind(is_ephemeral.unwrap_or(false))
    .bind(expires_at)
    .bind(&entities)
    .bind(None::<String>)              // event_type (test memory - no event)
    .bind(None::<chrono::DateTime<chrono::Utc>>) // event_date (test memory - no event)
    .fetch_one(&pool)
    .await
    .map_err(|e| format!("Failed to insert memory: {}", e))?;

    pool.close().await;

    println!("[Rust] Successfully inserted memory with ID: {}", memory_id);

    Ok(memory_id)
}

// update_memory → commands/memory.rs
// list_conversation_sessions, get_conversation_messages, search_conversations,
// delete_conversation_session, store_message_feedback → commands/conversation.rs

// ============================================================================

// delete_memory → commands/memory.rs
// get_memory → commands/memory.rs
// store_onboarding_response → commands/onboarding.rs
// complete_onboarding      → commands/onboarding.rs

/// Helper: Detect if query is asking about Zynkbot
/// Simple rule: If query mentions "zynkbot", search system docs
/// Otherwise, search user's personal memories
fn is_query_about_zynkbot(query: &str) -> bool {
    let q = query.to_lowercase();
    q.contains("zynkbot") || q.contains("zynk bot")
        || q.contains("zynksync") || q.contains("zynk sync")
        || q.contains("zynklink") || q.contains("zynk link")
        || q.contains("zchat")
        || q.contains("ensemble mode") || q.contains("ensemble")
        || q.contains("containment mode") || q.contains("snap-in") || q.contains("snapin")
        || q.contains("memory manager")
        || q.contains("knowledge base")
}

// seed_system_memories → commands/onboarding.rs
// index_system_documentation → commands/onboarding.rs

// get_memory_links → commands/memory.rs
// get_memory_graph → commands/memory.rs
// create_memory_link → commands/memory.rs
// delete_memory_link → commands/memory.rs
// ============================================================================
// CONTAINMENT & SAFETY COMMANDS (Rust-based, uses Candle)
// ============================================================================

/// Check if text passes containment filtering using toxic-bert + OpenAI API
/// Returns: Ok(None) if allowed, Ok(Some(message)) if blocked/warned
// check_containment, set_containment_mode, get_containment_mode,
// initialize_safety_models → commands/safety.rs

// ============================================================================
// WHISPER: LOCAL SPEECH-TO-TEXT (Pure Rust)
// ============================================================================

/// Transcribe audio file to text using local Whisper model (candle)
/// TEMPORARILY DISABLED: Whisper conflicts with llama-cpp-2 (GGML symbol collision)
/// Re-enable when: (1) whisper.cpp/llama.cpp resolve conflicts, or (2) use dynamic linking
///
/// # Arguments
/// * `audio_data` - WAV audio file as byte array (16kHz mono preferred)
///
/// # Returns
/// * Error message explaining feature is disabled
#[tauri::command]
async fn transcribe_audio(_audio_data: Vec<u8>) -> Result<String, String> {
    Err("Voice transcription temporarily disabled due to library conflicts. Will be re-enabled when whisper.cpp and llama.cpp resolve GGML symbol conflicts.".to_string())
}

// ============================================================================
// NEW: RUST-BASED NLP COMMANDS (Phases 1-3)
// ============================================================================

// check_question_worthiness → commands/safety.rs

// extract_entities → commands/nlp.rs
// extract_facts_from_question → commands/nlp.rs
// ============================================================================
// CONVERSATION ENGINE COMMANDS (Phase 5)
// ============================================================================

// check_memory_worthiness → commands/nlp.rs

// build_conversation_prompt → commands/conversation.rs
// ============================================================================
// NEW: EINSTEIN DEMO & CONTRADICTION DETECTION
// ============================================================================

// apply_einstein_seed → commands/onboarding.rs
// load_small_einstein_demo → commands/onboarding.rs

/// Load Einstein demo memories - LEGACY, archived in archive/einstein_demo_loader.rs
/// Kept here as dead code only for reference; apply_einstein_seed is the active path.
#[allow(dead_code)]
async fn load_einstein_demo_legacy(user_id: String) -> Result<serde_json::Value, String> {
    println!("[Rust] load_einstein_demo_legacy called for user: {}", user_id);

    // Einstein memories data - ALL 59 memories (FIRST-PERSON for original_text)
    // These will be converted to third-person for the content field
    let einstein_memories = vec![
        // Early Life (1879-1900)
        ("My Birth and Early Childhood", "I was born on March 14, 1879, in Ulm, Germany. My parents were Hermann Einstein and Pauline Koch.", "biography", vec!["early_life", "birth", "family"]),
        ("My Slow Speech Development", "I didn't speak until I was 3 years old, which worried my parents. I developed the habit of repeating sentences to myself before saying them out loud.", "biography", vec!["childhood", "development"]),
        ("The Compass That Changed Everything", "When I was 5, my father showed me a compass. The invisible force that made the needle move fascinated me and sparked my lifelong interest in physics.", "biography", vec!["childhood", "physics_inspiration"]),
        ("Learning the Violin", "My mother Pauline was a talented pianist. She made sure I learned violin from age 6. I've played it throughout my life and named my violin 'Lina'.", "personal", vec!["music", "violin", "hobbies"]),
        ("My Swiss Education", "After initially failing the entrance exam, I was admitted to the Swiss Federal Polytechnic School (ETH Zurich) in 1896.", "education", vec!["university", "switzerland"]),

        // Patent Office Years (1902-1909)
        ("Working at the Patent Office", "From 1902-1909, I worked as a patent examiner in Bern, Switzerland. This job gave me time to think deeply about physics.", "career", vec!["employment", "bern", "patent_office"]),
        ("My Marriage to Mileva", "I married Mileva Marić, a fellow physics student, in January 1903. We had three children together.", "personal", vec!["marriage", "family", "mileva"]),
        ("My Miracle Year - 1905", "In 1905, I published four groundbreaking papers in Annalen der Physik that revolutionized physics. This became known as my 'miracle year'.", "science", vec!["1905", "publications", "breakthrough"]),

        // Special Relativity
        ("My Special Relativity Theory", "My 1905 paper 'On the Electrodynamics of Moving Bodies' introduced special relativity. I showed that time and space are relative, not absolute.", "science", vec!["relativity", "1905", "theory"]),
        ("My Famous E=mc² Equation", "I derived the equation E=mc², showing that energy and mass are interchangeable. This became the most famous equation in physics.", "science", vec!["energy", "mass", "formula", "famous"]),
        ("Time Dilation Discovery", "I discovered that time passes slower for objects moving at high speeds relative to a stationary observer. This was a consequence of special relativity.", "science", vec!["relativity", "time", "physics"]),
        ("The Constancy of Light Speed", "I postulated that the speed of light in a vacuum is constant (299,792,458 m/s) regardless of the motion of the light source or observer.", "science", vec!["light", "constant", "speed"]),

        // General Relativity
        ("Developing General Relativity", "Between 1907-1915, I developed general relativity, describing gravity as the curvature of spacetime caused by mass and energy.", "science", vec!["gravity", "relativity", "spacetime"]),
        ("My Equivalence Principle", "I realized that gravitational acceleration is indistinguishable from acceleration due to mechanical forces. This became the equivalence principle.", "science", vec!["gravity", "principle", "acceleration"]),
        ("Spacetime Curvature Concept", "I described gravity not as a force, but as the curvature of four-dimensional spacetime by massive objects.", "science", vec!["spacetime", "curvature", "gravity"]),
        ("The 1919 Eclipse That Made Me Famous", "Arthur Eddington's 1919 solar eclipse expedition confirmed my prediction that massive objects bend light. I became world-famous overnight.", "science", vec!["eclipse", "verification", "famous", "eddington"]),
        ("Predicting Gravitational Waves", "In 1916, I predicted gravitational waves - ripples in spacetime caused by accelerating masses. They were finally detected in 2015, long after my death.", "science", vec!["waves", "prediction", "spacetime"]),
        ("Black Holes From My Equations", "My general relativity equations predicted the existence of black holes - regions where spacetime curvature becomes so extreme that nothing can escape.", "science", vec!["black_holes", "prediction", "gravity"]),

        // Quantum Physics
        ("The Photoelectric Effect Paper", "My 1905 paper on the photoelectric effect proposed that light consists of discrete quanta (photons). This work earned me the 1921 Nobel Prize.", "science", vec!["quantum", "light", "photons", "nobel"]),
        ("Receiving the Nobel Prize", "I won the 1921 Nobel Prize in Physics for my discovery of the law of the photoelectric effect, not for relativity.", "achievements", vec!["nobel", "prize", "1921", "photoelectric"]),
        ("My Skepticism About Quantum Mechanics", "Despite helping found quantum mechanics, I famously said 'God does not play dice' and remained skeptical of its probabilistic nature.", "philosophy", vec!["quantum", "skepticism", "god", "dice"]),
        ("The EPR Paradox", "In 1935, Podolsky, Rosen, and I published the EPR paradox paper, arguing that quantum mechanics was incomplete.", "science", vec!["quantum", "paradox", "epr", "1935"]),
        ("Spooky Action at a Distance", "I called quantum entanglement 'spooky action at a distance' and believed hidden variables must explain quantum correlations.", "philosophy", vec!["entanglement", "spooky", "quantum"]),
        ("My Debates with Niels Bohr", "Niels Bohr and I engaged in famous debates about quantum mechanics from the 1920s-1950s. We disagreed fundamentally on its interpretation.", "philosophy", vec!["bohr", "debates", "quantum", "philosophy"]),

        // Personal Life
        ("My Divorce and Remarriage", "I divorced Mileva in 1919 and married my cousin Elsa Löwenthal the same year. Elsa died in 1936.", "personal", vec!["marriage", "divorce", "elsa", "family"]),
        ("My Three Children", "I had three children: Lieserl (whose fate remains unknown), Hans Albert (who became a professor), and Eduard (who struggled with schizophrenia).", "personal", vec!["children", "family", "sons"]),
        ("My Love of Sailing", "I love sailing and have owned several sailboats. I sailed on lakes near Berlin and later in the US, despite never learning to swim.", "personal", vec!["sailing", "hobbies", "boats"]),
        ("Playing My Violin 'Lina'", "I named my violin 'Lina' and often perform in string quartets. Music helps me think about physics problems.", "personal", vec!["violin", "music", "hobbies", "lina"]),
        ("My Disheveled Appearance", "I famously avoid wearing socks and let my hair grow wild. I don't want to waste time on mundane matters.", "personal", vec!["appearance", "hair", "socks", "quirks"]),
        ("My Pipe Smoking Habit", "I'm a passionate pipe smoker. Smoking contributes to a somewhat calm and objective judgment in all human affairs.", "personal", vec!["smoking", "pipe", "habits"]),

        // Political Views
        ("My Pacifism", "I'm a committed pacifist and have been throughout most of my life, opposing militarism and nationalism.", "politics", vec!["pacifism", "peace", "antimilitarism"]),
        ("Escaping Nazi Germany", "As a Jew, I fled Nazi Germany in 1933 when Hitler came to power. My property was confiscated and my citizenship revoked.", "politics", vec!["nazis", "germany", "escape", "1933"]),
        ("My Letter to President Roosevelt", "In 1939, I signed a letter to President Roosevelt warning that Nazi Germany might develop atomic weapons. This helped start the Manhattan Project.", "politics", vec!["roosevelt", "atomic", "letter", "1939", "manhattan"]),
        ("My Deep Regret About the Atomic Bomb", "After Hiroshima, I deeply regretted my role in the atomic bomb's development. Had I known, I would have become a watchmaker.", "politics", vec!["atomic", "regret", "hiroshima", "bomb"]),
        ("My Support for Zionism", "I support Zionism and was offered the presidency of Israel in 1952, which I declined.", "politics", vec!["zionism", "israel", "presidency", "1952"]),
        ("My Civil Rights Activism", "I speak out against racism in America. I've called racism 'a disease of white people' and support the civil rights movement.", "politics", vec!["racism", "civil_rights", "activism", "america"]),
        ("My Socialist Views", "In 1949, I published 'Why Socialism?' arguing for a planned economy to overcome the 'economic anarchy' of capitalism.", "politics", vec!["socialism", "economics", "capitalism", "1949"]),

        // Princeton Years (1933-1955)
        ("My Years at Princeton", "From 1933 until now, I work at the Institute for Advanced Study in Princeton, New Jersey.", "career", vec!["princeton", "institute", "professor"]),
        ("My Quest for a Unified Field Theory", "I've spent the last 30 years trying to develop a unified field theory combining gravity and electromagnetism, but haven't succeeded yet.", "science", vec!["unified_field", "quest", "failure", "gravity"]),
        ("Becoming an American Citizen", "I became a US citizen in 1940, though I kept my Swiss citizenship. I've never returned to Germany after 1933.", "personal", vec!["citizenship", "america", "usa", "1940"]),
        ("FBI Surveillance of Me", "J. Edgar Hoover's FBI keeps a 1,427-page file on me, monitoring me as a potential subversive due to my political views.", "politics", vec!["fbi", "hoover", "surveillance", "subversive"]),

        // Famous Quotes
        ("Imagination vs Knowledge", "I believe imagination is more important than knowledge. Knowledge is limited. Imagination encircles the world.", "philosophy", vec!["quotes", "imagination", "knowledge"]),
        ("My Definition of Insanity", "I define insanity as doing the same thing over and over again and expecting different results.", "philosophy", vec!["quotes", "insanity", "definition"]),
        ("Simplicity in Science", "I believe everything should be made as simple as possible, but not simpler.", "philosophy", vec!["quotes", "simplicity", "science"]),
        ("The Mystery of the Universe", "I think the most incomprehensible thing about the universe is that it is comprehensible.", "philosophy", vec!["quotes", "universe", "comprehension"]),
        ("True Education", "Education is what remains after one has forgotten what one has learned in school.", "philosophy", vec!["quotes", "education", "learning"]),

        // Scientific Philosophy
        ("My Belief in Determinism", "I believe in a deterministic universe. God does not play dice with the universe.", "philosophy", vec!["determinism", "god", "dice", "belief"]),
        ("My View of God", "I believe in 'Spinoza's God' - a pantheistic view of God as synonymous with nature and the laws of physics, not a personal deity.", "philosophy", vec!["god", "spinoza", "pantheism", "religion"]),
        ("Science and Religion Together", "I believe science without religion is lame, religion without science is blind.", "philosophy", vec!["science", "religion", "quotes", "philosophy"]),
        ("My Cosmic Religious Feeling", "I experience a 'cosmic religious feeling' when contemplating the harmony of natural law.", "philosophy", vec!["cosmic", "religion", "harmony", "nature"]),

        // Additional Scientific Work
        ("My Brownian Motion Paper", "My 1905 paper on Brownian motion provided empirical evidence for the existence of atoms and molecules.", "science", vec!["1905", "atoms", "brownian", "molecules"]),
        ("Bose-Einstein Statistics", "I extended Satyendra Nath Bose's work on photon statistics to massive particles, predicting Bose-Einstein condensates.", "science", vec!["quantum", "statistics", "bose", "condensate"]),
        ("The Cosmological Constant Blunder", "I introduced the cosmological constant in 1917 to allow a static universe. I later called it my 'biggest blunder' after Hubble's discovery.", "science", vec!["cosmology", "constant", "blunder", "universe"]),
        ("Predicting Gravitational Lensing", "I predicted that massive objects would bend light paths, creating gravitational lensing effects observable in astronomy.", "science", vec!["lensing", "gravity", "light", "astronomy"]),

        // More Personal Details
        ("My Swiss Citizenship", "I became a Swiss citizen in 1901 and have maintained it for life, even after becoming American in 1940.", "personal", vec!["switzerland", "citizenship", "1901"]),
        ("Patent Work Influenced My Thinking", "My patent examination work involved evaluating electromagnetic devices, which influenced my thinking about light and relativity.", "career", vec!["patents", "electromagnetic", "work"]),
        ("The Olympia Academy", "My friends and I formed the 'Olympia Academy' in Bern to discuss philosophy, science, and literature.", "personal", vec!["olympia", "academy", "philosophy", "bern"]),
        ("My Berlin Years", "I was a professor at the Prussian Academy of Sciences in Berlin from 1914-1932 during my most productive years.", "career", vec!["berlin", "professor", "prussian", "academy"]),
        ("My Stance on World War I", "During WWI, I was one of few German intellectuals to publicly oppose the war, signing anti-war manifestos.", "politics", vec!["wwi", "pacifism", "war", "manifesto"]),
    ];

    // Connect to database
    let db_url = crate::db::get_db_url();

    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    // PAUSE SYNC DURING EINSTEIN LOAD (avoids contradiction pre-check overhead)
    // Check if sync is currently running
    let sync_was_running = {
        let global_service = ZYNKSYNC_SERVICE.lock().await;
        if let Some(service) = global_service.as_ref() {
            service.is_auto_sync_enabled().await
        } else {
            false
        }
    };

    if sync_was_running {
        println!("[Einstein] 🛑 Pausing sync during demo load (will resume after)...");
        let _ = stop_zynksync().await; // Stop sync temporarily
    } else {
        println!("[Einstein] ℹ️ Sync not running, skipping pause");
    }

    // Clear existing demo memories ONLY (both full and small demos)
    // This preserves the user's actual personal memories
    println!("[Rust] Cleaning up old Einstein demo memories...");
    let deleted_count = sqlx::query(
        "DELETE FROM memories
         WHERE session_id IN ('einstein-demo-session', 'einstein-demo-small-session')
         OR (namespace = 'einstein' AND source_type = 'demo_data')"
    )
    .execute(&pool)
    .await
    .map_err(|e| format!("Failed to clear existing demo memories: {}", e))?
    .rows_affected();

    println!("[Rust] ✅ Deleted {} old demo memories", deleted_count);

    let total_memories = einstein_memories.len();

    println!("[Rust] Loading {} Einstein memories using BATCH processing...", total_memories);
    println!("[Rust] This should take ~30-60 seconds (much faster than before!)");

    // Extract first-person content (simulates Einstein speaking in conversation)
    // For demo: both content and original_text are first-person (simulates standalone conversation turns)
    let all_first_person: Vec<String> = einstein_memories
        .iter()
        .map(|(_, content, _, _)| content.to_string())
        .collect();

    println!("[Rust] Generating embeddings for all {} memories from first-person text...", total_memories);

    // Generate embeddings from first-person text (simulates real user memory creation)
    let all_contents_for_embeddings = all_first_person.clone();
    let embeddings = tokio::task::spawn_blocking(move || {
        llm::local_embeddings::generate_local_embeddings_batch(all_contents_for_embeddings, Some(32))
    })
    .await
    .map_err(|e| format!("Failed to run batch embedding task: {}", e))?
    .map_err(|e| format!("Failed to generate batch embeddings: {}", e))?;

    println!("[Rust] ✅ Generated all {} embeddings from first-person text!", embeddings.len());

    // Generate ALL entity extractions from FIRST-PERSON text (for consistency with embeddings)
    println!("[Rust] Extracting entities for all {} memories from FIRST-PERSON text...", total_memories);
    let all_contents_for_entities = all_first_person.clone();
    let all_entities = tokio::task::spawn_blocking(move || {
        use nlp_enhancer::NLPEnhancer;
        let enhancer = NLPEnhancer::new();

        // NEW: Use batch extraction method (10-20x faster!)
        let all_entities_results = enhancer.extract_entities_batch(&all_contents_for_entities);

        // Convert to JSON format
        all_entities_results.into_iter().map(|entities| {
            serde_json::json!(entities.iter().map(|e| {
                serde_json::json!({
                    "word": e.word,
                    "label": e.label,
                    "score": e.score,
                    "start": e.start,
                    "end": e.end
                })
            }).collect::<Vec<_>>())
        }).collect::<Vec<_>>()
    })
    .await
    .map_err(|e| format!("Failed to run batch entity extraction task: {}", e))?;

    println!("[Rust] ✅ Extracted entities for all {} memories!", all_entities.len());

    // Detect events for all memories using BATCH processing from FIRST-PERSON text
    println!("[Rust] Detecting events for all {} memories from FIRST-PERSON text...", total_memories);
    let all_contents_for_events = all_first_person.clone();
    let all_events = tokio::task::spawn_blocking(move || {
        use nlp_enhancer::NLPEnhancer;
        let enhancer = NLPEnhancer::new();

        // Detect events from third-person text (normalized content)
        all_contents_for_events
            .iter()
            .map(|content| enhancer.detect_event(content))
            .collect::<Vec<_>>()
    })
    .await
    .map_err(|e| format!("Failed to run batch event detection task: {}", e))?;

    println!("[Rust] ✅ Detected events for all {} memories!", all_events.len());
    println!("[Rust] Now inserting into database (simulating real conversation)...");
    println!("[Rust]   - content: First-person (simulates Einstein speaking)");
    println!("[Rust]   - original_text: First-person (same - each is standalone)");
    println!("[Rust]   - embeddings: Generated from first-person");

    // Insert each memory (simulating standalone conversation turns with Einstein)
    let mut memory_ids = Vec::new();
    let session_id = "einstein-demo-session";

    for (idx, ((((title, first_person_content, namespace, tags), embedding), entities), (event_type, event_date))) in
        einstein_memories.into_iter()
            .zip(embeddings.into_iter())
            .zip(all_entities.into_iter())
            .zip(all_events.into_iter())
            .enumerate() {
        let progress = idx + 1;
        let percent = (progress * 100) / total_memories;

        // Log progress every 10 memories or at start/end
        if progress % 10 == 0 || progress == 1 || progress == total_memories {
            let event_str = event_type.as_deref().unwrap_or("none");
            println!("[Rust] Inserting {}/{} ({}%) - {} (event: {})", progress, total_memories, percent, title, event_str);
        }

        let embedding_vec = embedding.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
        let _tags_vec: Vec<String> = tags.iter().map(|s| s.to_string()).collect();

        // Insert memory (simulating Einstein speaking in conversation)
        let memory_id = sqlx::query_scalar::<_, i32>(
            "INSERT INTO memories (
                title, content, original_text, namespace, user_id, session_id,
                embedding, source_type, is_syncable, is_shareable,
                entities_detected, event_type, event_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id"
        )
        .bind(title)
        .bind(&first_person_content)         // First-person (simulates Einstein speaking)
        .bind(&first_person_content)         // Same - each memory is standalone
        .bind(namespace)
        .bind(&user_id)
        .bind(session_id)
        .bind(&embedding_vec)
        .bind("demo_data")
        .bind(true)
        .bind(false)
        .bind(&entities)
        .bind(event_type.as_deref())
        .bind(event_date)
        .fetch_one(&pool)
        .await
        .map_err(|e| format!("Failed to insert memory '{}': {}", title, e))?;

        memory_ids.push(memory_id);
    }

    println!("[Rust] ✅ Completed loading all {} Einstein memories!", total_memories);
    println!("[Rust] ✅ Demo simulates standalone conversation turns with Einstein");

    // STEP 2: Generate relationships between all Einstein memories using auto-detection
    println!("[Rust] Now generating relationships between {} memories...", memory_ids.len());
    println!("[Rust] Using local embedding similarity to classify relationships (no API calls required)...");
    println!("[Rust] Progress will be logged every 10 memories processed.");

    // Query back all Einstein memories we just inserted
    let all_memories = if memory_ids.is_empty() {
        Vec::new()
    } else {
        let in_clause = memory_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        let sql = format!("SELECT id, content, title FROM memories WHERE id IN ({})", in_clause);
        let mut q = sqlx::query_as::<_, (i32, String, Option<String>)>(&sql);
        for id in &memory_ids { q = q.bind(id); }
        q.fetch_all(&pool).await.map_err(|e| format!("Failed to fetch inserted memories: {}", e))?
    };

    println!("[Rust] Fetched {} memories for relationship detection", all_memories.len());
    println!("[Rust] ⚡ OPTIMIZED MODE: Loading embeddings AND entities from database (no regeneration!)");

    // Load all embeddings from database in one query (OPTIMIZATION!)
    let embeddings_query = if memory_ids.is_empty() {
        Vec::new()
    } else {
        let in_clause = memory_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        let sql = format!("SELECT id, embedding FROM memories WHERE id IN ({})", in_clause);
        let mut q = sqlx::query_as::<_, (i32, Vec<u8>)>(&sql);
        for id in &memory_ids { q = q.bind(id); }
        q.fetch_all(&pool).await.map_err(|e| format!("Failed to fetch embeddings: {}", e))?
    };

    // Create a map of memory_id -> embedding for fast lookup
    let embeddings_map: std::collections::HashMap<i32, Vec<f32>> = embeddings_query
        .into_iter()
        .map(|(id, blob)| (id, blob_to_f32(&blob)))
        .collect();

    println!("[Rust] ✅ Loaded {} embeddings from database (0 regenerations!)", embeddings_map.len());

    // Load all entities from database in one query (OPTIMIZATION! - avoids 500+ re-extractions)
    let entities_query = if memory_ids.is_empty() {
        Vec::new()
    } else {
        let in_clause = memory_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        let sql = format!("SELECT id, entities_detected FROM memories WHERE id IN ({})", in_clause);
        let mut q = sqlx::query_as::<_, (i32, Option<serde_json::Value>)>(&sql);
        for id in &memory_ids { q = q.bind(id); }
        q.fetch_all(&pool).await.map_err(|e| format!("Failed to fetch entities: {}", e))?
    };

    // Create a map of memory_id -> entity words for fast lookup
    // IMPORTANT: Include memories with EMPTY entity arrays to prevent fallback extraction
    let entities_map: std::collections::HashMap<i32, Vec<String>> = entities_query
        .into_iter()
        .filter_map(|(id, entities_json)| {
            if let Some(json) = entities_json {
                if let Some(arr) = json.as_array() {
                    let words: Vec<String> = arr
                        .iter()
                        .filter_map(|ent| ent.get("word").and_then(|w| w.as_str()).map(|s| s.to_string()))
                        .collect();
                    // Return empty Vec for memories with no entities (prevents fallback extraction)
                    Some((id, words))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    println!("[Rust] ✅ Loaded {} entity sets from database (no re-extraction!)", entities_map.len());

    let mut total_relationships_created = 0;

    // For each memory, detect relationships using CACHED embeddings
    for (idx, (memory_id, _content, _title)) in all_memories.iter().enumerate() {
        if (idx + 1) % 10 == 0 || idx == 0 {
            println!("[Rust] Processing memory {}/{} for relationships...", idx + 1, all_memories.len());
        }

        let current_embedding = match embeddings_map.get(memory_id) {
            Some(emb) => emb,
            None => {
                println!("[Rust] ⚠️ Missing embedding for memory {}, skipping", memory_id);
                continue;
            }
        };

        // Calculate similarities with all other memories using cached embeddings
        let mut similarities: Vec<(i32, f32)> = Vec::new();
        for (other_id, other_embedding) in &embeddings_map {
            if other_id == memory_id {
                continue; // Skip self
            }

            // Calculate cosine similarity
            let similarity = crate::llm::local_embeddings::cosine_similarity(
                current_embedding,
                other_embedding,
            );

            // Lowered from 0.35 to 0.25 to create bridge relationships between topic clusters
            // This prevents isolated clusters by allowing cross-topic connections
            if similarity > 0.25 {
                similarities.push((*other_id, similarity));
            }
        }

        // Sort by similarity descending and take top 10 (increased from 5 for better graph connectivity)
        // Combined with lower threshold (0.25), this creates cross-topic bridges without being too slow
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(10);

        // Classify relationships using local embedding similarity (same signal the main pipeline uses)
        let current_entities = entities_map.get(memory_id);

        for (target_id, similarity_score) in similarities {
            // Compute entity overlap fraction (both sets may be empty for abstract memories)
            let target_entities = entities_map.get(&target_id);
            let entity_overlap: f32 = match (current_entities, target_entities) {
                (Some(a), Some(b)) if !a.is_empty() && !b.is_empty() => {
                    let shared = a.iter().filter(|e| b.contains(e)).count();
                    shared as f32 / a.len().max(b.len()) as f32
                }
                _ => 0.0,
            };

            // Map similarity + entity overlap to relationship type (no API, no rules engine)
            let (relation_type, notes) = if similarity_score > 0.75 && entity_overlap > 0.25 {
                ("elaborates", format!(
                    "Very high semantic similarity ({:.0}%) with shared entities — adds detail to the same topic",
                    similarity_score * 100.0
                ))
            } else if similarity_score > 0.60 {
                ("supports", format!(
                    "High semantic similarity ({:.0}%) — reinforces or agrees with existing memory",
                    similarity_score * 100.0
                ))
            } else {
                ("reminds_of", format!(
                    "Moderate semantic similarity ({:.0}%) — loosely connected topics",
                    similarity_score * 100.0
                ))
            };

            // Skip if link already exists in either direction
            let existing_link = sqlx::query_scalar::<_, i32>(
                "SELECT id FROM memory_links
                 WHERE (source_memory_id = ? AND target_memory_id = ?)
                    OR (source_memory_id = ? AND target_memory_id = ?)
                 LIMIT 1"
            )
            .bind(*memory_id)
            .bind(target_id)
            .fetch_optional(&pool)
            .await;

            if let Ok(Some(_)) = existing_link {
                continue;
            }

            match memory::create_memory_link(
                &pool,
                *memory_id,
                target_id,
                relation_type,
                similarity_score,
                Some(&notes),
                "system",
            ).await {
                Ok(_) => total_relationships_created += 1,
                Err(e) => println!("[Rust] ⚠️ Failed to create relationship: {}", e),
            }
        }
    }

    pool.close().await;

    println!("[Rust] ✅ Successfully loaded {} Einstein memories with {} relationships!", memory_ids.len(), total_relationships_created);

    // RESUME SYNC AFTER EINSTEIN LOAD (if it was running before)
    if sync_was_running {
        println!("[Einstein] ✅ Resuming sync now that Einstein load is complete...");
        let _ = start_zynksync(None).await; // Restart sync with default 60s interval
        println!("[Einstein] 🔄 Sync will now propagate Einstein memories in the background");
    } else {
        println!("[Einstein] ℹ️ Sync was not running before, not restarting");
    }

    Ok(serde_json::json!({
        "success": true,
        "loaded_count": memory_ids.len(),
        "relationships_created": total_relationships_created,
        "message": format!("Loaded {} Einstein memories with {} relationships", memory_ids.len(), total_relationships_created)
    }))
}

/// Pre-check memory for contradictions before saving (SMART VERSION with entity extraction)
/// Returns list of potentially contradicting memories
/// This version extracts entities from both user input and existing memories,
/// then checks for actual factual contradictions, not just high similarity.
#[tauri::command]
async fn pre_check_memory(
    content: String,
    user_id: String,
    exclude_memory_id: i32,
) -> Result<serde_json::Value, String> {
    println!("[Rust] 🔍 Smart duplicate check - extracting keywords from input...");

    // Extract keywords (ALL nouns) for contradiction matching - not just named entities
    // This ensures we can match "dog" in both "I have a dog named Max" and "My dog's name is Wendy"
    let content_for_keywords = content.clone();
    let query_entities = tokio::task::spawn_blocking(move || {
        use nlp_enhancer::NLPEnhancer;
        let enhancer = NLPEnhancer::new();
        enhancer.extract_keywords(&content_for_keywords)
    })
    .await
    .map_err(|e| format!("Failed to extract keywords: {}", e))?;

    println!("[Rust] Extracted {} keywords for duplicate check: {:?}", query_entities.len(), query_entities);

    // Generate embedding for new content
    let content_clone = content.clone();
    let query_embedding = tokio::task::spawn_blocking(move || {
        llm::local_embeddings::generate_local_embedding(&content_clone)
    })
    .await
    .map_err(|e| format!("Failed to run embedding task: {}", e))?
    .map_err(|e| format!("Failed to generate embedding: {}", e))?;

    // Connect to database
    let db_url = crate::db::get_db_url();

    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    // Clone query_entities AND query_embedding before passing to hybrid_search
    let _query_entities_clone = query_entities.clone();
    let query_embedding_clone = query_embedding.clone();

    // Use HYBRID SEARCH (same as main conversation flow) to find memories with:
    // 1. Semantic similarity OR
    // 2. Shared keywords (ALL nouns, not just named entities - important for catching contradictions!)
    let similar_memories = memory::hybrid_search(
        &pool,
        query_embedding,
        query_entities,  // This moves query_entities
        Some(&user_id),
        None,
        None,
        10,  // Get top 10 candidates for quality filtering
    )
    .await
    .map_err(|e| format!("Failed to search for similar memories: {}", e))?;

    // Don't close pool yet - we need it for duplicate detection

    // QUALITY FILTER: Only check memories with similarity > 35% for contradictions
    // Using 35% to match hybrid_search threshold (was 40% for vector_search)
    // EXCLUDE the newly stored memory to avoid checking it against itself
    let candidates: Vec<memory::Memory> = similar_memories
        .into_iter()
        .filter(|m| m.similarity.unwrap_or(0.0) > 0.35 && m.id != exclude_memory_id)
        .collect();

    println!("[Rust] Found {} candidates (>35%) for duplicate check", candidates.len());

    if candidates.is_empty() {
        return Ok(serde_json::json!({
            "has_duplicate": false,
            "has_contradiction": false,  // Kept for backward compatibility
            "count": 0
        }));
    }

    // DUPLICATE DETECTION: Check pure semantic similarity (not hybrid score)
    // Hybrid search combines entity + semantic (60%/40%), which can score identical memories at only 0.6
    // For duplicate detection, we need PURE cosine similarity from embeddings
    println!("[Rust] 🔍 Checking for duplicates using pure cosine similarity...");
    for candidate in &candidates {
        // Get candidate's embedding from database
        let candidate_embedding_query = sqlx::query_as::<_, (Vec<u8>,)>(
            "SELECT embedding FROM memories WHERE id = ?"
        )
        .bind(candidate.id)
        .fetch_one(&pool)
        .await;

        if let Ok((candidate_blob,)) = candidate_embedding_query {
            let candidate_embedding_vec = blob_to_f32(&candidate_blob);

            // Calculate PURE cosine similarity (not hybrid score)
            let pure_similarity = crate::llm::local_embeddings::cosine_similarity(
                &query_embedding_clone,
                &candidate_embedding_vec
            );

            let hybrid_score = candidate.similarity.unwrap_or(0.0);
            println!("[Rust]   Memory {} - Pure cosine: {:.3}, Hybrid score: {:.3}",
                     candidate.id, pure_similarity, hybrid_score);

            // Duplicate threshold: Check hybrid first (>98%), then pure cosine (>93%)
            // Hybrid score 1.0 = exact match even if pure cosine slightly lower due to entity boosting
            // Lowered pure cosine from 0.95 to 0.93 to catch near-duplicates
            if hybrid_score > 0.98 || pure_similarity > 0.93 {
                println!("[Rust] 🔄 DUPLICATE DETECTED: Memory {} - Hybrid: {:.1}%, Pure cosine: {:.1}%",
                         candidate.id, hybrid_score * 100.0, pure_similarity * 100.0);
                return Ok(serde_json::json!({
                    "has_duplicate": true,
                    "duplicate_memory_id": candidate.id,
                    "duplicate_content": candidate.content,
                    "duplicate_title": candidate.title,
                    "similarity": pure_similarity,
                    "message": "This memory appears to be a duplicate of an existing memory."
                }));
            }
        }
    }

    // Pure cosine similarity check above is sufficient for duplicate detection
    // Legacy hybrid score check removed - it was causing false positives

    // ============================================================================
    // CONTRADICTION DETECTION DISABLED - Trust LLM for contradiction detection
    // ============================================================================
    // The LLM relationship classifier handles contradiction detection via the second API call.
    // This pre_check_memory function now ONLY checks for duplicates (>95% similarity).
    // If no duplicates found, return early and let the LLM handle relationship classification.
    // ============================================================================

    println!("[Rust] ✅ No duplicates detected - trusting LLM for contradiction detection");
    return Ok(serde_json::json!({
        "has_duplicate": false,
        "has_contradiction": false,
        "message": "No duplicates detected. LLM will handle contradiction detection."
    }));

    // LEGACY CONTRADICTION DETECTION CODE (DISABLED)
    // Preserved for reference but not executed - LLM handles this now
    // ============================================================================
    #[allow(unreachable_code)]
    {
    // Store first candidate similarity for later use
    let first_similarity = candidates.first().map(|c| c.similarity.unwrap_or(0.0)).unwrap_or(0.0);

    // OPTIMIZED CHECK: Use already-extracted entities, use CACHED entities from database
    // This is 10x faster than re-extracting entities for all candidates!
    let content_for_check = content.clone();
    let detected_contradiction = tokio::task::spawn_blocking(move || {
        // Convert to set of entity words for fast comparison (already extracted above)
        let new_entity_words: std::collections::HashSet<String> = _query_entities_clone
            .into_iter()
            .map(|e| e.to_lowercase())
            .collect();

        println!("[Rust] New input has {} keywords for duplicate check: {:?}", new_entity_words.len(), new_entity_words);

        // NLP enhancer for keyword extraction
        use nlp_enhancer::NLPEnhancer;
        let enhancer = NLPEnhancer::new();

        // Check each candidate for contradiction by extracting keywords on-the-fly
        // (Not using cached entities since they're named entities only, we need ALL nouns)
        for candidate in &candidates {
            // Extract keywords from candidate content on-the-fly
            let cached_keywords = enhancer.extract_keywords(&candidate.content);
            let cached_entity_set: std::collections::HashSet<String> = cached_keywords.into_iter().collect();

            // Find shared keywords between new input and candidate memory
            let shared_keywords_raw: Vec<String> = new_entity_words
                .intersection(&cached_entity_set)
                .cloned()
                .collect();

            println!("[Rust] Memory {} shares {} keywords: {:?}", candidate.id, shared_keywords_raw.len(), shared_keywords_raw);

            // Keywords are already filtered by extract_keywords (no stop words, 3+ chars)
            // But double-check to be safe
            let stop_words: std::collections::HashSet<&str> = vec![
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
                "been", "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "should", "could", "may", "might", "must", "can", "i", "you",
                "he", "she", "it", "we", "they", "my", "your", "his", "her", "its",
                "our", "their", "this", "that", "these", "those", "am", "me", "him",
                "us", "them", "what", "which", "who", "when", "where", "why", "how"
            ].into_iter().collect();

            // Filter shared keywords - remove any stop words and short keywords
            let shared_entities: Vec<String> = shared_keywords_raw
                .into_iter()
                .filter(|kw| {
                    let kw_lower = kw.to_lowercase();
                    !stop_words.contains(kw_lower.as_str()) && kw.len() >= 3
                })
                .collect();

            if !shared_entities.is_empty() {
                println!("[Rust] After filtering stop words: {} meaningful entities: {:?}", shared_entities.len(), shared_entities);
            }

            let new_lower = content_for_check.to_lowercase();
            let existing_lower = candidate.content.to_lowercase();

            // Semantic pattern grouping - check if both texts match patterns from same group
            // Groups semantically equivalent patterns together so "named X" matches "name is Y"
            let pattern_groups = vec![
                // Name patterns - "dog named Wendy" vs "dog's name is Max"
                vec!["named ", "name is ", "called ", "goes by ", "known as "],

                // Identity patterns - "I am X" vs "I'm Y"
                vec!["i am ", "i'm ", "my name is ", "i call myself "],

                // Location patterns - "lives in NYC" vs "from Boston" vs "resides in LA"
                vec!["lives in ", "live in ", "from ", "resides in ", "based in ", "located in "],

                // Work patterns - "works at Google" vs "employed by Microsoft"
                vec!["works at ", "work at ", "employed by ", "employed at ", "works for ", "job at ", "teaches at ", "studies at "],

                // Relationship patterns - "married to X" vs "spouse is Y"
                vec!["married to ", "spouse is ", "wife is ", "husband is ", "partner is ", "wed to "],

                // Age patterns - "I'm 30" vs "I am 25 years old"
                vec!["i'm ", "i am ", "age is ", "years old"],

                // Possession patterns - "I own X" vs "I have Y"
                vec!["i own ", "i have ", "my car is ", "my pet is ", "owns ", "has "],
            ];

            // Check if both texts match patterns from the same group
            let mut has_pattern_match = false;
            for pattern_group in &pattern_groups {
                let new_has_pattern = pattern_group.iter().any(|p| new_lower.contains(p));
                let existing_has_pattern = pattern_group.iter().any(|p| existing_lower.contains(p));
                if new_has_pattern && existing_has_pattern {
                    has_pattern_match = true;
                    break;
                }
            }

            // SMART GATING to balance precision and recall:
            // 1. If both match semantic patterns (e.g., "name is") AND share 1+ entity → check for contradiction
            //    (catches "dog named Max" vs "dog's name is Wendy" even with only 1 shared entity)
            // 2. Else require 2+ meaningful shared entities OR 85%+ similarity
            //    (prevents false positives like "My dog" vs "My regret")
            let pattern_match_gate = has_pattern_match && !shared_entities.is_empty();
            let strict_gate = shared_entities.len() >= 2 ||
                             (!shared_entities.is_empty() && candidate.similarity.unwrap_or(0.0) > 0.85);

            if pattern_match_gate || strict_gate {

                // Pattern 1: Check for negation patterns (e.g., "I have X" vs "I don't have X")
                let negation_words = vec!["not", "never", "no", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"];
                let new_has_negation = negation_words.iter().any(|neg| new_lower.contains(neg));
                let existing_has_negation = negation_words.iter().any(|neg| existing_lower.contains(neg));

                // If one has negation and other doesn't, likely contradictory
                if new_has_negation != existing_has_negation {
                    let reason = format!("Makes opposing claims about: {}", shared_entities.join(", "));
                    println!("[Rust] ⚠️ CONTRADICTION DETECTED (negation): {}", reason);
                    return Some((candidate.id, candidate.content.clone(), candidate.title.clone(), reason));
                }

                // Pattern 2: Semantic pattern grouping for substitution detection
                // Check each pattern group (already defined above)

                // Stop words to filter out pattern values (possessive pronouns, articles, etc.)
                // Prevents false positives like "named my" (violin) vs "named max" (dog)
                let pattern_stop_words: std::collections::HashSet<&str> = vec![
                    "the", "a", "an", "my", "your", "his", "her", "its", "our", "their",
                    "this", "that", "these", "those", "i", "me", "you", "he", "she", "it", "we", "they"
                ].into_iter().collect();

                for pattern_group in pattern_groups {
                    // Try to find a pattern from this group in each text
                    let mut new_pattern_match: Option<(&str, String)> = None;
                    let mut existing_pattern_match: Option<(&str, String)> = None;

                    // Find matching pattern in new text
                    for pattern in &pattern_group {
                        if new_lower.contains(pattern) {
                            if let Some(after) = new_lower.split(pattern).nth(1) {
                                let value = after.split_whitespace().next().unwrap_or("");
                                // Skip if stop word, empty, or too short
                                if !value.is_empty() && value.len() > 1 && !pattern_stop_words.contains(value) {
                                    new_pattern_match = Some((pattern, value.to_string()));
                                    break;
                                }
                            }
                        }
                    }

                    // Find matching pattern in existing text
                    for pattern in &pattern_group {
                        if existing_lower.contains(pattern) {
                            if let Some(after) = existing_lower.split(pattern).nth(1) {
                                let value = after.split_whitespace().next().unwrap_or("");
                                // Skip if stop word, empty, or too short
                                if !value.is_empty() && value.len() > 1 && !pattern_stop_words.contains(value) {
                                    existing_pattern_match = Some((pattern, value.to_string()));
                                    break;
                                }
                            }
                        }
                    }

                    // If both texts have a pattern from this group with different values, it's a contradiction
                    if let (Some((new_pat, new_val)), Some((existing_pat, existing_val))) =
                        (new_pattern_match, existing_pattern_match) {
                        if new_val != existing_val {
                            let reason = format!("Claims '{}{}' but memory says '{}{}'",
                                new_pat.trim(), new_val, existing_pat.trim(), existing_val);
                            println!("[Rust] ⚠️ CONTRADICTION DETECTED (substitution): {}", reason);
                            return Some((candidate.id, candidate.content.clone(), candidate.title.clone(), reason));
                        }
                    }
                }
            }
        }
        None
    })
    .await
    .map_err(|e| format!("Failed to run duplicate check: {}", e))?;

    // If contradiction detected, return structured format for frontend
    if let Some((mem_id, mem_content, mem_title, reason)) = detected_contradiction {
        Ok(serde_json::json!({
            "has_contradiction": true,
            "new_statement": content,
            "new_memory_id": null,
            "new_memory_title": "New Statement",
            "conflicting_memory": {
                "id": mem_id,
                "title": mem_title.unwrap_or_else(|| "Untitled".to_string()),
                "content": mem_content,
            },
            "reason": reason,
            "similarity": first_similarity
        }))
    } else {
        println!("[Rust] ✅ No entity-level contradictions detected");
        Ok(serde_json::json!({
            "has_contradiction": false,
            "count": 0
        }))
    }
    } // End of #[allow(unreachable_code)] block - LEGACY contradiction detection code preserved but disabled
}

/// Resolve conflict between memories
/// User chooses which memory to keep or merge
/// TODO: This should be refactored to check for conflicts BEFORE storing (see Option A in docs)
/// Currently, the new memory is already stored when this is called, leading to cleanup issues
#[tauri::command]
async fn resolve_conflict(
    _new_content: String,  // Content of the new statement (unused - kept for API compatibility)
    existing_memory_id: i32,
    resolution: String,  // "memoryA" (keep new), "memoryB" (keep existing), "both" (keep both)
    user_id: String,  // User ID
    explanation: Option<String>,  // User's explanation if "both" selected
    new_memory_id: Option<i32>,  // ID of the already-stored new memory (for deletion if discarded)
) -> Result<serde_json::Value, String> {
    println!("[Rust] resolve_conflict called - resolution: {}, explanation: {:?}", resolution, explanation);

    // Connect to database
    let db_url = crate::db::get_db_url();

    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    match resolution.as_str() {
        "memoryA" => {
            // User chose OLD memory is correct (memoryA = existing) - delete the new one
            // SWAPPED: memoryA is now OLD memory to match user expectations
            println!("[Rust] Resolution: Keep OLD memory #{}, delete new", existing_memory_id);

            // CRITICAL: The new memory was already stored - delete it now
            if let Some(new_id) = new_memory_id {
                println!("[Rust] Deleting already-stored new memory #{}", new_id);

                let result = sqlx::query("DELETE FROM memories WHERE id = ?")
                    .bind(new_id)
                    .execute(&pool)
                    .await
                    .map_err(|e| format!("Failed to delete new memory: {}", e))?;

                println!("[Rust] ✅ Deleted {} row(s) - new memory discarded", result.rows_affected());
            } else {
                println!("[Rust] ✅ No new memory was stored (conflict detected before storage)");
            }
        }
        "memoryB" => {
            // User chose NEW memory is correct (memoryB = new) - delete the old one
            // SWAPPED: memoryB is now NEW memory to match user expectations
            println!("[Rust] Resolution: Keep NEW memory, delete OLD #{}", existing_memory_id);

            // Delete the existing (OLD) memory
            let result = sqlx::query("DELETE FROM memories WHERE id = ?")
                .bind(existing_memory_id)
                .execute(&pool)
                .await
                .map_err(|e| format!("Failed to delete existing memory: {}", e))?;

            println!("[Rust] ✅ Deleted {} row(s) - old memory discarded", result.rows_affected());

            // TODO: Propagate deletion to other devices
            // This requires access to ZYNKSYNC_SERVICE which we don't have here
            // For now, the next auto-sync will clean it up via content-hash comparison
            println!("[Rust] ⚠️ Deletion will sync on next auto-sync cycle (immediate propagation not yet implemented in resolve_conflict)");
        }
        "both" => {
            // User says both are correct - use already-stored new memory
            println!("[Rust] Resolution: Keep BOTH memories with explanation");

            // Use the new memory that was already stored (passed as parameter)
            let actual_new_memory_id = new_memory_id
                .ok_or_else(|| "New memory ID required for 'both' resolution".to_string())?;

            println!("[Rust] Using already-stored new memory with ID: {}", actual_new_memory_id);

            // If user provided explanation, store it as a separate memory linking them
            if let Some(explanation_text) = explanation {
                let explanation_id = memory::insert_memory(
                    &pool,
                    Some("Explanation of conflicting memories"),  // title
                    &explanation_text,
                    Some("user"),
                    None,  // session_id
                    None,  // embedding
                    None,  // parent_scroll_id
                    None,  // chunk_index
                    Some(&user_id),
                    "default",  // namespace
                    true,  // is_syncable
                    false,  // is_shareable
                    None,  // entities_detected
                    None,  // event_type
                    None,  // event_date
                    None,  // original_text
                )
                .await
                .map_err(|e| format!("Failed to store explanation: {}", e))?;

                println!("[Rust] Stored explanation memory with ID: {}", explanation_id);

                // Create 'elaborates' link from explanation to both memories
                memory::create_memory_link(
                    &pool,
                    explanation_id,
                    actual_new_memory_id,
                    "elaborates",
                    0.95,
                    Some("Explanation elaborates on relationship between conflicting memories"),
                    "user"
                )
                .await
                .map_err(|e| format!("Failed to create explanation->new link: {}", e))?;

                memory::create_memory_link(
                    &pool,
                    explanation_id,
                    existing_memory_id,
                    "elaborates",
                    0.95,
                    Some("Explanation elaborates on relationship between conflicting memories"),
                    "user"
                )
                .await
                .map_err(|e| format!("Failed to create explanation->existing link: {}", e))?;

                println!("[Rust] ✅ Created elaboration links to memories {} and {}", actual_new_memory_id, existing_memory_id);

                // Update both memories' timestamps to trigger sync
                sqlx::query("UPDATE memories SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ? OR id = ?")
                    .bind(actual_new_memory_id)
                    .bind(existing_memory_id)
                    .execute(&pool)
                    .await
                    .map_err(|e| format!("Failed to update timestamps: {}", e))?;

                println!("[Rust] ✅ Updated timestamps to trigger relationship sync");
            } else {
                // No explanation - just create an 'elaborates' link between the two memories
                memory::create_memory_link(
                    &pool,
                    actual_new_memory_id,
                    existing_memory_id,
                    "elaborates",
                    0.85,
                    Some("User confirmed both memories are correct and related"),
                    "user"
                )
                .await
                .map_err(|e| format!("Failed to create elaborates link: {}", e))?;

                println!("[Rust] ✅ Created elaborates link between memories {} and {}", actual_new_memory_id, existing_memory_id);

                // Update both memories' timestamps to trigger sync
                sqlx::query("UPDATE memories SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ? OR id = ?")
                    .bind(actual_new_memory_id)
                    .bind(existing_memory_id)
                    .execute(&pool)
                    .await
                    .map_err(|e| format!("Failed to update timestamps: {}", e))?;

                println!("[Rust] ✅ Updated timestamps to trigger relationship sync");
            }
        }
        _ => {
            pool.close().await;
            return Err(format!("Unknown resolution strategy: {}", resolution));
        }
    }

    pool.close().await;

    Ok(serde_json::json!({
        "success": true,
        "resolution": resolution,
        "message": format!("Conflict resolved with strategy: {}", resolution)
    }))
}

// Helper struct for pending memory data
#[derive(serde::Deserialize, Clone)]
struct PendingMemory {
    content: String,
    title: Option<String>,
    embedding: Vec<f32>,
}

/// NEW: Resolve memory conflict (v2) - For refactored flow where memory isn't stored yet
/// This is called when contradiction is detected BEFORE storage
#[tauri::command]
async fn resolve_memory_conflict_v2(
    pending_memory_json: String,  // JSON with content, title, embedding, etc.
    conflicting_memory_id: i32,
    decision: String,  // "keep_old" | "keep_new" | "keep_both" (with or without explanation)
    explanation: Option<String>,
    relationships_json: String,  // JSON array of relationships from LLM
    user_id: String,
    session_id: String,
) -> Result<serde_json::Value, String> {
    println!("[Rust] resolve_memory_conflict_v2 called - decision: {}, explanation: {:?}", decision, explanation);

    let pending: PendingMemory = serde_json::from_str(&pending_memory_json)
        .map_err(|e| format!("Failed to parse pending memory: {}", e))?;

    // Parse relationships
    let relationships: Vec<RelationshipClassification> = serde_json::from_str(&relationships_json)
        .map_err(|e| format!("Failed to parse relationships: {}", e))?;

    // Connect to database
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url).await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    match decision.as_str() {
        "keep_old" => {
            // Discard new — it was never stored, so nothing to delete.
            println!("[Rust] Resolution: Keep OLD memory #{}, discard new", conflicting_memory_id);
            Ok(serde_json::json!({
                "success": true,
                "action": "kept_old",
                "message": "Kept existing memory, new statement discarded"
            }))
        }

        "keep_new" => {
            println!("[Rust] Resolution: Keep NEW memory, delete OLD #{}", conflicting_memory_id);

            sqlx::query("DELETE FROM memories WHERE id = ?")
                .bind(conflicting_memory_id)
                .execute(&pool)
                .await
                .map_err(|e| format!("Failed to delete old memory: {}", e))?;

            println!("[Rust] ✅ Deleted old memory #{}", conflicting_memory_id);

            let new_memory_id = store_pending_memory(&pool, &pending, &user_id, &session_id).await?;

            for rel in relationships.iter().filter(|r| r.memory_id != conflicting_memory_id && r.relationship_type != "none") {
                let _ = memory::create_memory_link(
                    &pool,
                    new_memory_id,
                    rel.memory_id,
                    &rel.relationship_type,
                    rel.confidence.unwrap_or(0.75),
                    Some(&rel.reason),
                    "llm",
                ).await;
            }

            println!("[Rust] ✅ Stored new memory #{} and created relationships", new_memory_id);

            Ok(serde_json::json!({
                "success": true,
                "action": "kept_new",
                "new_memory_id": new_memory_id,
                "message": "Old memory deleted, new memory stored"
            }))
        }

        "not_a_contradiction" => {
            println!("[Rust] Resolution: Not a contradiction — storing new memory, no contradiction link");

            let new_memory_id = store_pending_memory(&pool, &pending, &user_id, &session_id).await?;

            for rel in relationships.iter().filter(|r| r.relationship_type != "contradicts" && r.relationship_type != "none") {
                let _ = memory::create_memory_link(
                    &pool,
                    new_memory_id,
                    rel.memory_id,
                    &rel.relationship_type,
                    rel.confidence.unwrap_or(0.75),
                    Some(&rel.reason),
                    "llm",
                ).await;
            }

            println!("[Rust] ✅ Stored new memory #{} without contradiction link", new_memory_id);

            Ok(serde_json::json!({
                "success": true,
                "action": "not_a_contradiction",
                "new_memory_id": new_memory_id,
                "message": "Both memories kept, contradiction edge removed"
            }))
        }

        "keep_both" => {
            println!("[Rust] Resolution: Accept contradiction — keeping both memories");

            let new_memory_id = store_pending_memory(&pool, &pending, &user_id, &session_id).await?;

            memory::create_memory_link(
                &pool,
                new_memory_id,
                conflicting_memory_id,
                "contradicts",
                0.95,
                Some("user_acknowledged"),
                "user",
            ).await
            .map_err(|e| format!("Failed to create contradiction link: {}", e))?;

            println!("[Rust] ✅ Created contradicts relationship: {} <-> {}", new_memory_id, conflicting_memory_id);

            // Create other non-contradiction relationships
            for rel in relationships.iter().filter(|r| r.relationship_type != "contradicts" && r.relationship_type != "none") {
                let _ = memory::create_memory_link(
                    &pool,
                    new_memory_id,
                    rel.memory_id,
                    &rel.relationship_type,
                    rel.confidence.unwrap_or(0.75),
                    Some(&rel.reason),
                    "llm",
                ).await;
            }

            Ok(serde_json::json!({
                "success": true,
                "action": "kept_both",
                "new_memory_id": new_memory_id,
                "message": "Both memories stored with contradiction relationship"
            }))
        }

        "both_with_explanation" => {
            println!("[Rust] Resolution: Resolve with explanation");

            let new_memory_id = store_pending_memory(&pool, &pending, &user_id, &session_id).await?;

            // Create CONTRADICTS relationship
            memory::create_memory_link(
                &pool,
                new_memory_id,
                conflicting_memory_id,
                "contradicts",
                0.95,
                Some("resolved:explained"),
                "user",
            ).await
            .map_err(|e| format!("Failed to create contradiction link: {}", e))?;

            println!("[Rust] ✅ Created contradicts relationship: {} <-> {}", new_memory_id, conflicting_memory_id);

            // Store explanation memory with resolves edges to both memories
            if let Some(expl) = explanation {
                if !expl.trim().is_empty() {
                    let expl_clone = expl.clone();
                    let expl_embedding = tokio::task::spawn_blocking(move || {
                        crate::llm::local_embeddings::generate_local_embedding(&expl_clone)
                    }).await
                    .map_err(|e| format!("Failed to generate embedding: {}", e))?
                    .map_err(|e| format!("Failed to generate embedding: {}", e))?;

                    // Inherit entities from both conflicting memories
                    let combined_entities = {
                        let mut seen_words = std::collections::HashSet::new();
                        let mut merged: Vec<serde_json::Value> = Vec::new();
                        for mem_id in [new_memory_id, conflicting_memory_id] {
                            if let Ok(Some(mem)) = memory::get_memory(&pool, mem_id).await {
                                if let Some(serde_json::Value::Array(ents)) = mem.entities_detected {
                                    for ent in ents {
                                        let word = ent.get("word")
                                            .and_then(|w| w.as_str())
                                            .unwrap_or("")
                                            .to_lowercase();
                                        if !word.is_empty() && seen_words.insert(word) {
                                            merged.push(ent);
                                        }
                                    }
                                }
                            }
                        }
                        Some(serde_json::Value::Array(merged))
                    };

                    println!("[Rust] Attempting to store explanation memory: '{}'", expl);
                    let explanation_result = memory::insert_memory(
                        &pool,
                        Some("Resolution of contradictory memories"),
                        &expl,
                        Some("user_explanation"),
                        Some(&session_id),
                        Some(expl_embedding),
                        None, None,
                        Some(&user_id),
                        "personal",
                        true, false,
                        combined_entities, None, None,
                        Some(&expl),
                    ).await;
                    let explanation_id = match explanation_result {
                        Ok(id) => id,
                        Err(e) => {
                            println!("[Rust] ❌ Failed to store explanation memory: {}", e);
                            return Err(format!("Failed to store explanation: {}", e));
                        }
                    };

                    // resolves edges from explanation to both memories
                    let _ = memory::create_memory_link(&pool, explanation_id, new_memory_id, "resolves", 0.95, Some("Resolves contradiction"), "user").await;
                    let _ = memory::create_memory_link(&pool, explanation_id, conflicting_memory_id, "resolves", 0.95, Some("Resolves contradiction"), "user").await;

                    println!("[Rust] ✅ Stored resolution memory #{} with resolves edges", explanation_id);
                }
            }

            // Create other non-contradiction relationships
            for rel in relationships.iter().filter(|r| r.relationship_type != "contradicts" && r.relationship_type != "none") {
                let _ = memory::create_memory_link(
                    &pool,
                    new_memory_id,
                    rel.memory_id,
                    &rel.relationship_type,
                    rel.confidence.unwrap_or(0.75),
                    Some(&rel.reason),
                    "llm",
                ).await;
            }

            Ok(serde_json::json!({
                "success": true,
                "action": "resolved_with_explanation",
                "new_memory_id": new_memory_id,
                "message": "Both memories stored, contradiction resolved with explanation"
            }))
        }

        _ => Err(format!("Invalid decision: '{}'. Must be one of: keep_old, keep_new, not_a_contradiction, keep_both, both_with_explanation", decision))
    }
}

/// Helper: Store pending memory with NLP enhancement
async fn store_pending_memory(
    pool: &sqlx::SqlitePool,
    pending: &PendingMemory,
    user_id: &str,
    session_id: &str,
) -> Result<i32, String> {
    let content_clone = pending.content.clone();
    let (entities, event_type, event_date, namespace) = tokio::task::spawn_blocking(move || {
        let enhancer = nlp_enhancer::NLPEnhancer::new();
        let enhancement = enhancer.enhance(&content_clone);
        let ents = enhancer.extract_entities(&content_clone);
        let entities_json = serde_json::json!(ents.iter().map(|e| {
            serde_json::json!({
                "word": e.word,
                "label": e.label,
                "score": e.score,
                "start": e.start,
                "end": e.end
            })
        }).collect::<Vec<_>>());
        (entities_json, enhancement.event_type, enhancement.event_date, enhancement.namespace)
    })
    .await
    .map_err(|e| format!("Failed to enhance memory: {}", e))?;

    let memory_id = memory::insert_memory(
        pool,
        pending.title.as_deref(),
        &pending.content,
        Some("conversation"),
        Some(session_id),
        Some(pending.embedding.clone()),
        None, None,
        Some(user_id),
        &namespace,
        true,  // is_syncable
        false, // is_shareable
        Some(entities),
        event_type.as_deref(),
        event_date,
        Some(&pending.content),
    ).await
    .map_err(|e| format!("Failed to insert memory: {}", e))?;

    Ok(memory_id)
}

// ============================================================================
// TAURI APP SETUP
// ============================================================================

// cleanup_expired_memories → commands/memory.rs
// ============================================================================
// ZChat Commands (Device-to-Device Messaging)
// ============================================================================

/// Send a chat message to a paired device
#[tauri::command]
async fn zchat_send_message(
    to_device_id: String,
    message_text: String,
) -> Result<zchat::SendMessageResponse, String> {
    // Get database pool (from Flask for now, will use Tauri state later)
    let pool = get_db_pool().await?;

    // Get current device ID and user ID
    let device_id_str = user_identity::get_device_id()?;
    let user_id_str = user_identity::get_user_id()?;

    let device_id = uuid::Uuid::parse_str(&device_id_str)
        .map_err(|e| format!("Invalid device ID: {}", e))?;
    let user_id = uuid::Uuid::parse_str(&user_id_str)
        .map_err(|e| format!("Invalid user ID: {}", e))?;
    let to_device_uuid = uuid::Uuid::parse_str(&to_device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    // Store message locally
    let response = zchat::send_message(&pool, device_id, to_device_uuid, message_text, user_id).await?;

    // Try to deliver immediately via ZynkLink (async, don't block on failure)
    let user_id_clone = user_id_str.clone();
    let device_id_clone = device_id_str.clone();
    let to_device_clone = to_device_id.clone();
    tokio::spawn(async move {
        let db_url = crate::db::get_db_url();

        let pool = match sqlx::SqlitePool::connect(&db_url).await {
            Ok(p) => p,
            Err(e) => {
                println!("[ZChat] Failed to connect to database for delivery: {}", e);
                return;
            }
        };

        match zynklink::deliver_zchat_to_peer(&pool, &user_id_clone, &device_id_clone, &to_device_clone).await {
            Ok(count) if count > 0 => println!("[ZChat] Delivered {} message(s) to ZynkLink peer", count),
            Ok(_) => println!("[ZChat] No messages to deliver"),
            Err(e) => println!("[ZChat] Failed to deliver (will retry on next send): {}", e),
        }
    });

    Ok(response)
}

/// Get chat messages with a specific device
#[tauri::command]
async fn zchat_get_messages(
    device_id: String,
    since: Option<String>,
) -> Result<zchat::GetMessagesResponse, String> {
    let pool = get_db_pool().await?;

    let current_device_id_str = user_identity::get_device_id()?;
    let current_device_id = uuid::Uuid::parse_str(&current_device_id_str)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    let device_uuid = uuid::Uuid::parse_str(&device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    zchat::get_messages(&pool, current_device_id, device_uuid, since).await
}

/// Receive chat messages from remote device (called during sync)
#[tauri::command]
async fn zchat_deliver_messages(
    messages: Vec<zchat::DeliverMessageData>,
) -> Result<zchat::DeliverMessagesResponse, String> {
    let pool = get_db_pool().await?;
    zchat::deliver_messages(&pool, messages).await
}

/// Mark messages as delivered
#[tauri::command]
async fn zchat_mark_delivered(message_ids: Vec<String>) -> Result<usize, String> {
    let pool = get_db_pool().await?;

    let uuids: Result<Vec<uuid::Uuid>, _> = message_ids
        .iter()
        .map(|id| uuid::Uuid::parse_str(id))
        .collect();

    let uuids = uuids.map_err(|e| format!("Invalid message ID: {}", e))?;
    zchat::mark_delivered(&pool, uuids).await
}

/// Mark messages as read
#[tauri::command]
async fn zchat_mark_read(message_ids: Vec<String>) -> Result<usize, String> {
    let pool = get_db_pool().await?;

    let uuids: Result<Vec<uuid::Uuid>, _> = message_ids
        .iter()
        .map(|id| uuid::Uuid::parse_str(id))
        .collect();

    let uuids = uuids.map_err(|e| format!("Invalid message ID: {}", e))?;
    zchat::mark_read(&pool, uuids).await
}

/// Get unread message count from a specific device
#[tauri::command]
async fn zchat_get_unread_count(from_device_id: String) -> Result<i64, String> {
    let pool = get_db_pool().await?;

    let current_device_id_str = user_identity::get_device_id()?;
    let current_device_id = uuid::Uuid::parse_str(&current_device_id_str)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    let from_device_uuid = uuid::Uuid::parse_str(&from_device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    zchat::get_unread_count(&pool, current_device_id, from_device_uuid).await
}

/// Mark all unread messages from a device as read
#[tauri::command]
async fn zchat_mark_all_read_from_device(from_device_id: String) -> Result<usize, String> {
    let pool = get_db_pool().await?;

    let current_device_id_str = user_identity::get_device_id()?;
    let current_device_id = uuid::Uuid::parse_str(&current_device_id_str)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    let from_device_uuid = uuid::Uuid::parse_str(&from_device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    // Get all unread message IDs from this device
    let message_ids = sqlx::query_scalar::<_, uuid::Uuid>(
        r#"
        SELECT id
        FROM zchat_messages
        WHERE to_device_id = ? AND from_device_id = ? AND read_at IS NULL
        "#,
    )
    .bind(current_device_id)
    .bind(from_device_uuid)
    .fetch_all(&pool)
    .await
    .map_err(|e| format!("Failed to fetch unread messages: {}", e))?;

    if message_ids.is_empty() {
        return Ok(0);
    }

    zchat::mark_read(&pool, message_ids).await
}

#[tauri::command]
async fn zchat_clear_history(with_device_id: String) -> Result<(), String> {
    let pool = get_db_pool().await?;
    // Device IDs are stored as binary UUID blobs — must bind as Uuid, not &str
    let device_uuid = uuid::Uuid::parse_str(&with_device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;
    sqlx::query(
        "DELETE FROM zchat_messages WHERE from_device_id = ? OR to_device_id = ?"
    )
    .bind(device_uuid)
    .bind(device_uuid)
    .execute(&pool)
    .await
    .map_err(|e| format!("Failed to clear chat history: {}", e))?;
    Ok(())
}

/// Get undelivered messages to a specific device (for sync)
#[tauri::command]
async fn zchat_get_undelivered_messages(
    to_device_id: String,
) -> Result<Vec<zchat::DeliverMessageData>, String> {
    let pool = get_db_pool().await?;

    let from_device_id_str = user_identity::get_device_id()?;
    let from_device_id = uuid::Uuid::parse_str(&from_device_id_str)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    let to_device_uuid = uuid::Uuid::parse_str(&to_device_id)
        .map_err(|e| format!("Invalid device ID: {}", e))?;

    zchat::get_undelivered_messages(&pool, from_device_id, to_device_uuid).await
}

// ============================================================================
// WEB SEARCH COMMANDS
// ============================================================================

/// Search DuckDuckGo for information
#[tauri::command]
async fn search_web(query: String, max_results: Option<usize>) -> Result<web_search::SearchResponse, String> {
    println!("[WebSearch] Searching for: {}", query);
    let max = max_results.unwrap_or(5);
    web_search::search_duckduckgo(&query, max).await
}

// Helper function to get database pool (placeholder - will be improved)
async fn get_db_pool() -> Result<sqlx::SqlitePool, String> {
    sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Database connection failed: {}", e))
}

// get_memory_contradictions → commands/memory.rs
// get_namespaces → commands/memory.rs
// update_memory_link → commands/memory.rs
// get_full_memory_graph → commands/memory.rs
// ============================================================================
// KNOWLEDGE BASE COMMANDS
// ============================================================================

/// Scan knowledge base directory and return list of files
#[tauri::command]
async fn scan_knowledge_base(directory: String) -> Result<Vec<knowledge_base::KnowledgeBaseFile>, String> {
    knowledge_base::scan_knowledge_base_directory(&directory)
}

/// Search knowledge base for query terms
#[tauri::command]
async fn search_knowledge_base(
    directory: String,
    query: String,
) -> Result<Vec<knowledge_base::KnowledgeBaseSearchResult>, String> {
    knowledge_base::search_knowledge_base(&directory, &query)
}

/// Read a specific knowledge base file
#[tauri::command]
async fn read_knowledge_base_file(file_path: String) -> Result<String, String> {
    knowledge_base::read_knowledge_base_file(&file_path)
}

// ============================================================================
// KNOWLEDGE BASE RAG COMMANDS (Phase 2: Vector Search & Chunking)
// ============================================================================

/// Get the KB folder path for a user
#[tauri::command]
async fn get_kb_folder_path(user_id: String) -> Result<String, String> {
    let path = kb_rag::get_kb_folder_path(&user_id)?;
    Ok(path.to_string_lossy().to_string())
}

/// Open KB folder in file explorer (cross-platform)
#[tauri::command]
async fn open_kb_folder_in_explorer(user_id: String) -> Result<(), String> {
    let path = kb_rag::get_kb_folder_path(&user_id)?;
    let path_str = path.to_string_lossy().to_string();

    // Cross-platform folder opening
    #[cfg(target_os = "windows")]
    let command = "explorer";

    #[cfg(target_os = "macos")]
    let command = "open";

    #[cfg(target_os = "linux")]
    let command = "xdg-open";

    std::process::Command::new(command)
        .arg(&path_str)
        .spawn()
        .map_err(|e| format!("Failed to open folder with {}: {}", command, e))?;

    println!("[KB] Opened folder: {}", path_str);
    Ok(())
}

/// Open an external file in the default system editor
#[tauri::command]
async fn open_external_file(path: String) -> Result<(), String> {
    // Get the project root directory using Cargo manifest directory
    // During development: Use CARGO_MANIFEST_DIR (set by cargo)
    // In production: Navigate up from executable location
    let project_root = if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        // Development mode: CARGO_MANIFEST_DIR points to zynkbot_rust/src-tauri
        // Project root is 2 levels up
        std::path::PathBuf::from(manifest_dir)
            .parent()
            .and_then(|p| p.parent())
            .ok_or_else(|| "Failed to navigate from CARGO_MANIFEST_DIR".to_string())?
            .to_path_buf()
    } else {
        // Production mode: Navigate up from executable
        let exe_path = std::env::current_exe()
            .map_err(|e| format!("Failed to get executable path: {}", e))?;
        exe_path
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .ok_or_else(|| "Failed to determine project root from executable".to_string())?
            .to_path_buf()
    };

    let full_path = project_root.join(&path);

    if !full_path.exists() {
        return Err(format!("File not found: {}\nProject root: {}\nRelative path: {}",
            full_path.display(), project_root.display(), path));
    }

    let path_str = full_path.to_string_lossy().to_string();

    // Cross-platform file opening
    #[cfg(target_os = "windows")]
    let command = "cmd";
    #[cfg(target_os = "windows")]
    let args = vec!["/C", "start", "", &path_str];

    #[cfg(target_os = "macos")]
    let command = "open";
    #[cfg(target_os = "macos")]
    let args = vec![&path_str];

    #[cfg(target_os = "linux")]
    let command = "xdg-open";
    #[cfg(target_os = "linux")]
    let args = vec![&path_str];

    std::process::Command::new(command)
        .args(&args)
        .spawn()
        .map_err(|e| format!("Failed to open file: {}", e))?;

    println!("[External] Opened file: {}", path_str);
    Ok(())
}

/// Open an external folder in the system file explorer
#[tauri::command]
async fn open_external_folder(path: String) -> Result<(), String> {
    // Get the project root directory using Cargo manifest directory
    // During development: Use CARGO_MANIFEST_DIR (set by cargo)
    // In production: Navigate up from executable location
    let project_root = if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        // Development mode: CARGO_MANIFEST_DIR points to zynkbot_rust/src-tauri
        // Project root is 2 levels up
        std::path::PathBuf::from(manifest_dir)
            .parent()
            .and_then(|p| p.parent())
            .ok_or_else(|| "Failed to navigate from CARGO_MANIFEST_DIR".to_string())?
            .to_path_buf()
    } else {
        // Production mode: Navigate up from executable
        let exe_path = std::env::current_exe()
            .map_err(|e| format!("Failed to get executable path: {}", e))?;
        exe_path
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .ok_or_else(|| "Failed to determine project root from executable".to_string())?
            .to_path_buf()
    };

    let full_path = project_root.join(&path);

    if !full_path.exists() {
        return Err(format!("Folder not found: {}\nProject root: {}\nRelative path: {}",
            full_path.display(), project_root.display(), path));
    }

    let path_str = full_path.to_string_lossy().to_string();

    // Cross-platform folder opening
    #[cfg(target_os = "windows")]
    let command = "explorer";

    #[cfg(target_os = "macos")]
    let command = "open";

    #[cfg(target_os = "linux")]
    let command = "xdg-open";

    std::process::Command::new(command)
        .arg(&path_str)
        .spawn()
        .map_err(|e| format!("Failed to open folder: {}", e))?;

    println!("[External] Opened folder: {}", path_str);
    Ok(())
}

/// Index a document: chunk it, generate embeddings, store in vector DB
#[tauri::command]
async fn index_kb_document(
    app: tauri::AppHandle,
    user_id: String,
    file_path: String,
) -> Result<i32, String> {
    // Connect to database
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    let progress_cb: Box<dyn Fn(usize, usize) + Send + Sync> = Box::new(move |current, total| {
        let _ = app.emit("kb:indexing_progress", serde_json::json!({ "current": current, "total": total }));
    });

    kb_rag::index_document(&pool, &user_id, &file_path, Some(progress_cb)).await
}

/// List all indexed KB documents for a user
#[tauri::command]
async fn list_kb_documents(
    user_id: String,
) -> Result<Vec<kb_rag::KBDocument>, String> {
    // Connect to database
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    kb_rag::list_kb_documents(&pool, &user_id).await
}

/// Remove a document from the index
#[tauri::command]
async fn remove_kb_document(
    user_id: String,
    file_path: String,
) -> Result<(), String> {
    // Connect to database
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    kb_rag::remove_document_index(&pool, &user_id, &file_path).await
}

/// Clear all indexed documents for a user
#[tauri::command]
async fn clear_all_kb_documents(
    user_id: String,
) -> Result<i64, String> {
    // Connect to database
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    kb_rag::clear_all_documents(&pool, &user_id).await
}

/// Semantic search in knowledge base
#[tauri::command]
async fn search_kb(
    user_id: String,
    query: String,
    top_k: Option<usize>,
) -> Result<Vec<kb_rag::KBSearchResult>, String> {
    // Connect to database
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    let k = top_k.unwrap_or(10);
    // User-initiated search: exclude system docs
    kb_rag::search_kb_chunks(&pool, &user_id, &query, k, false).await
}

// ============================================================================
// SNAP-IN COMMANDS (Experimental: Professional & Personal Workspaces)
// ============================================================================

#[tauri::command]
async fn index_snapin_notes(
    patient_name: String,
    session_title: String,
    notes_content: String,
    user_id: String,
) -> Result<String, String> {
    // Sanitize patient name for file path (lowercase, replace spaces with underscores)
    let safe_patient = patient_name
        .to_lowercase()
        .replace(" ", "_")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect::<String>();

    // Sanitize session title
    let safe_session = session_title
        .replace(" ", "_")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .collect::<String>();

    // Create virtual file path for logical organization
    let file_path = format!(
        "snap_ins/therapist/{}/{}.txt",
        safe_patient,
        safe_session
    );

    println!("[Snap-in] Indexing notes at path: {}", file_path);

    // Connect to database
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    // Use the existing KB RAG system to index the notes as a text document
    kb_rag::index_text_as_document(
        &pool,
        &user_id,
        &file_path,
        &notes_content
    ).await?;

    println!("[Snap-in] Successfully indexed notes for {}", patient_name);

    Ok(format!(
        "✅ Session notes indexed for {}\nStored at: {}",
        patient_name,
        file_path
    ))
}

// ============================================================================
// ZYNKSYNC COMMANDS (Phase 9: Device-to-Device Sync)
// ============================================================================

use std::sync::Arc;
use tokio::sync::Mutex;
use zynksync::{ZynkSyncService, PeerDevice, SyncResult};

// Global ZynkSync service instance
pub(crate) static ZYNKSYNC_SERVICE: once_cell::sync::Lazy<Arc<Mutex<Option<Arc<ZynkSyncService>>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(None)));

// Global app handle for emitting events from HTTP server
pub static APP_HANDLE: once_cell::sync::Lazy<Arc<std::sync::Mutex<Option<tauri::AppHandle>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(std::sync::Mutex::new(None)));

/// Cancellation registry for in-flight ZynkLink downloads, keyed by `relative_path`.
/// `download_to_custom_location` and `download_to_knowledge_base` register an
/// AtomicBool flag at start, poll it between streamed chunks, and remove it on
/// completion (or cancellation, or error). The `cancel_zynklink_download` Tauri
/// command flips the flag for a given key, causing the matching streaming loop
/// to break early, delete its .part file, and return Err("Cancelled by user").
static DOWNLOAD_CANCELS: once_cell::sync::Lazy<
    std::sync::Mutex<std::collections::HashMap<String, Arc<std::sync::atomic::AtomicBool>>>,
> = once_cell::sync::Lazy::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Auto-start HTTP server on app launch (shared by ZynkSync + ZynkLink)
/// This runs automatically in setup() - NOT a Tauri command
async fn auto_start_http_server() -> Result<(), String> {
    println!("[HTTP Server] Auto-starting shared server...");
    println!("[HTTP Server] Current directory: {:?}", std::env::current_dir());

    // Initialize user identity early
    match user_identity::get_identity() {
        Ok(identity) => {
            println!("[HTTP Server] ✅ Identity: user_id={}, device_id={}",
                identity.user_id, identity.device_id);
        }
        Err(e) => {
            println!("[HTTP Server] ⚠️ Warning: Could not initialize identity: {}", e);
        }
    }

    // Get database connection
    let db_url = crate::db::get_db_url();

    println!("[HTTP Server] Connecting to database...");
    let db_pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;
    println!("[HTTP Server] ✅ Database connected");

    // Get persistent device ID
    let device_id = user_identity::get_device_id()
        .map_err(|e| format!("Failed to get device ID: {}", e))?;

    let device_name = hostname::get()
        .map_err(|e| format!("Failed to get hostname: {}", e))?
        .to_string_lossy()
        .to_string();

    println!("[HTTP Server] Device: {} ({})", device_name, device_id);

    // Load or generate this device's TLS certificate for encrypted LAN communication
    let data_dir = crate::db::get_app_data_dir();
    let (cert_pem, key_pem, cert_der) = match crate::tls::load_or_generate_cert(&data_dir) {
        Ok(v) => v,
        Err(e) => {
            println!("[HTTP Server] ❌ TLS certificate error: {}", e);
            return Err(format!("Failed to initialize TLS: {}", e));
        }
    };

    // Create ZynkSync service (but don't start auto-sync yet)
    // Use 60 second interval to match UI expectations
    let service = Arc::new(ZynkSyncService::new(
        device_id.clone(),
        device_name,
        db_pool,
        Some(60), // 60 second sync interval
        cert_pem,
        key_pem,
        cert_der,
    ));

    // Start HTTP server on port 57963
    println!("[HTTP Server] Starting HTTP server on port 57963...");
    match service.clone().start_http_server().await {
        Ok(port) => {
            println!("[HTTP Server] ✅ HTTP server started on port {}", port);
        }
        Err(e) => {
            println!("[HTTP Server] ❌ HTTP server failed: {}", e);
            return Err(format!("Failed to start HTTP server: {}", e));
        }
    };

    // Store service globally
    {
        let mut global_service = ZYNKSYNC_SERVICE.lock().await;
        *global_service = Some(service.clone());
    }

    println!("[HTTP Server] ✅ Server ready - ZynkSync and ZynkLink can now be used");

    // Check if sync was previously enabled and auto-start if so
    if load_sync_state().await {
        println!("[ZynkSync] Restoring previous sync state (was enabled)...");

        // Load devices from database
        if let Err(e) = service.load_devices().await {
            println!("[ZynkSync] ⚠️ Failed to load devices: {}", e);
        }
        if let Err(e) = service.rebuild_http_client().await {
            println!("[ZynkSync] ⚠️ Failed to rebuild HTTP client with peer certs: {}", e);
        }

        // Generate pairing code
        match service.generate_pairing_code().await {
            Ok(code) => {
                println!("[ZynkSync] ✅ Pairing code: {} (expires in 10 minutes)", code);
            }
            Err(e) => {
                println!("[ZynkSync] ⚠️ Failed to generate pairing code: {}", e);
            }
        }

        // Start auto-sync loop in background
        let service_clone = service.clone();
        tokio::spawn(async move {
            service_clone.start_auto_sync().await;
        });

        // Start message delivery loop in background
        let service_clone = service.clone();
        tokio::spawn(async move {
            service_clone.start_message_delivery_loop().await;
        });

        // Start heartbeat loop in background
        let service_clone = service.clone();
        tokio::spawn(async move {
            service_clone.start_heartbeat_loop().await;
        });

        println!("[ZynkSync] ✅ Auto-sync restored automatically");
    } else {
        println!("[ZynkSync] Sync was disabled - not auto-starting (click 'Start Sync' to enable)");
    }

    // Keep the Tokio runtime alive forever by blocking here
    // This prevents the runtime from dropping and killing the HTTP server task
    println!("[HTTP Server] Keeping runtime alive...");
    std::future::pending::<()>().await;

    Ok(())
}

/// Start ZynkSync auto-sync loop (HTTP server already running from app launch)
/// Note: sync_interval is configured at app launch (60 seconds default)
#[tauri::command]
async fn start_zynksync(_sync_interval_secs: Option<u64>) -> Result<String, String> {
    println!("[ZynkSync] Starting auto-sync loop...");

    // Get the HTTP server service (should already be running from app launch)
    let service = {
        let global_service = ZYNKSYNC_SERVICE.lock().await;
        match global_service.as_ref() {
            Some(svc) => {
                // Check if auto-sync already running
                if svc.is_auto_sync_enabled().await {
                    println!("[ZynkSync] ⚠️ Auto-sync already running");
                    return Err("ZynkSync auto-sync is already running. Stop it first.".to_string());
                }
                Arc::clone(svc)
            }
            None => {
                println!("[ZynkSync] ❌ HTTP server not running");
                return Err("HTTP server not started. Please restart the app.".to_string());
            }
        }
    };

    // Load manually added devices from database
    if let Err(e) = service.load_devices().await {
        println!("[ZynkSync] ⚠️ Failed to load devices: {}", e);
    }
    if let Err(e) = service.rebuild_http_client().await {
        println!("[ZynkSync] ⚠️ Failed to rebuild HTTP client with peer certs: {}", e);
    }

    // Generate pairing code for this device
    println!("[ZynkSync] Generating pairing code...");
    match service.generate_pairing_code().await {
        Ok(code) => {
            println!("[ZynkSync] ✅ Pairing code: {} (expires in 10 minutes)", code);
        }
        Err(e) => {
            println!("[ZynkSync] ⚠️ Failed to generate pairing code: {}", e);
        }
    }

    // Start auto-sync loop in background
    let service_clone = Arc::clone(&service);
    tokio::spawn(async move {
        service_clone.start_auto_sync().await;
    });

    // Start message delivery loop in background
    let service_clone = Arc::clone(&service);
    tokio::spawn(async move {
        service_clone.start_message_delivery_loop().await;
    });

    // Start heartbeat loop in background
    let service_clone = Arc::clone(&service);
    tokio::spawn(async move {
        service_clone.start_heartbeat_loop().await;
    });

    println!("[ZynkSync] ✅ Auto-sync started successfully");

    // Save sync state to file (so we remember on restart)
    if let Err(e) = save_sync_state(true).await {
        eprintln!("[ZynkSync] ⚠️ Failed to save sync state: {}", e);
    }

    // Return device ID for display
    let device_id = user_identity::get_device_id()
        .map_err(|e| format!("Failed to get device ID: {}", e))?;
    Ok(device_id)
}

/// Save sync state to file
async fn save_sync_state(enabled: bool) -> Result<(), String> {
    let state_dir = dirs::data_dir()
        .ok_or("Failed to get data directory".to_string())?
        .join("zynkbot");

    std::fs::create_dir_all(&state_dir)
        .map_err(|e| format!("Failed to create state directory: {}", e))?;

    let state_file = state_dir.join("sync_state.json");
    let state = serde_json::json!({ "sync_enabled": enabled });

    std::fs::write(&state_file, state.to_string())
        .map_err(|e| format!("Failed to write sync state: {}", e))?;

    Ok(())
}

/// Load sync state from file
async fn load_sync_state() -> bool {
    let state_dir = match dirs::data_dir() {
        Some(dir) => dir.join("zynkbot"),
        None => return false,
    };

    let state_file = state_dir.join("sync_state.json");

    match std::fs::read_to_string(&state_file) {
        Ok(content) => {
            if let Ok(state) = serde_json::from_str::<serde_json::Value>(&content) {
                state.get("sync_enabled").and_then(|v| v.as_bool()).unwrap_or(false)
            } else {
                false
            }
        }
        Err(_) => false,
    }
}

/// Stop ZynkSync auto-sync loop (HTTP server keeps running for ZynkLink)
#[tauri::command]
async fn stop_zynksync() -> Result<(), String> {
    println!("[ZynkSync] Stopping auto-sync...");

    let global_service = ZYNKSYNC_SERVICE.lock().await;
    if let Some(service) = global_service.as_ref() {
        // Check if auto-sync is actually running
        if !service.is_auto_sync_enabled().await {
            println!("[ZynkSync] ⚠️ Auto-sync not running");
            return Err("ZynkSync auto-sync is not running".to_string());
        }

        service.stop_auto_sync().await;
        println!("[ZynkSync] ✅ Auto-sync stopped (HTTP server still running for ZynkLink)");

        // Save sync state to file (so we remember on restart)
        if let Err(e) = save_sync_state(false).await {
            eprintln!("[ZynkSync] ⚠️ Failed to save sync state: {}", e);
        }

        Ok(())
    } else {
        Err("HTTP server not started. Please restart the app.".to_string())
    }
}

/// Check if ZynkSync auto-sync is running
#[tauri::command]
async fn get_zynksync_status() -> Result<bool, String> {
    let global_service = ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => Ok(service.is_auto_sync_enabled().await),
        None => Ok(false), // HTTP server not started yet
    }
}

/// Get list of discovered peer devices
#[tauri::command]
async fn get_zynksync_peers() -> Result<Vec<PeerDevice>, String> {
    let global_service = ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => Ok(service.get_peers().await),
        None => Ok(Vec::new()),
    }
}

/// Manually trigger sync to a specific peer
#[tauri::command]
async fn sync_to_peer(peer_id: String, user_id: Option<String>) -> Result<SyncResult, String> {
    let global_service = ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => service.sync_to_peer(&peer_id, user_id.as_deref()).await,
        None => Err("ZynkSync not started".to_string()),
    }
}

/// Receive memories from a peer device (called by Flask endpoint)
#[tauri::command]
async fn receive_sync_memories(memories: Vec<zynksync::SyncMemory>) -> Result<usize, String> {
    let global_service = ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => service.receive_from_peer(memories).await,
        None => Err("ZynkSync not started".to_string()),
    }
}

/// Request pairing with a peer (generates 6-digit code)
#[tauri::command]
async fn request_device_pairing(peer_id: String) -> Result<String, String> {
    let global_service = ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => service.request_pairing(&peer_id).await,
        None => Err("ZynkSync not started".to_string()),
    }
}

/// Verify pairing code and authorize peer
#[tauri::command]
async fn verify_pairing_code(peer_id: String, code: String) -> Result<(), String> {
    let global_service = ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => service.verify_pairing_code(&peer_id, &code).await,
        None => Err("ZynkSync not started".to_string()),
    }
}

/// Unpair from a device
#[tauri::command]
async fn unpair_device(peer_id: String) -> Result<(), String> {
    let global_service = ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => service.unpair_device(&peer_id).await,
        None => Err("ZynkSync not started".to_string()),
    }
}

/// Add a device manually by IP address and pairing code
/// Returns peer device info including the host's user_id for identity sync
#[tauri::command]
async fn add_zynksync_device(host_ip: String, pairing_code: String) -> Result<PeerDevice, String> {
    let service = {
        let global_service = ZYNKSYNC_SERVICE.lock().await;
        match global_service.as_ref() {
            Some(s) => std::sync::Arc::clone(s),
            None => return Err("ZynkSync not started".to_string()),
        }
    };
    let peer = service.add_device(&host_ip, &pairing_code).await?;
    // Rebuild HTTP client so the newly-pinned peer cert takes effect immediately.
    if let Err(e) = service.rebuild_http_client().await {
        eprintln!("[ZynkSync] Warning: failed to rebuild HTTP client after pairing: {}", e);
    }
    Ok(peer)
}

/// Remove a manually added device
#[tauri::command]
async fn remove_zynksync_device(device_id: String) -> Result<(), String> {
    let global_service = ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => service.remove_device(&device_id).await,
        None => Err("ZynkSync not started".to_string()),
    }
}

/// Get this device's pairing code for sharing
#[tauri::command]
async fn get_zynksync_pairing_code() -> Result<String, String> {
    let global_service = ZYNKSYNC_SERVICE.lock().await;
    match global_service.as_ref() {
        Some(service) => service.get_pairing_code().await,
        None => Err("ZynkSync not started".to_string()),
    }
}

/// Check sync status with all peers and emit event if user action needed
/// Returns info about whether local device is more recent than peers
#[tauri::command]
async fn check_sync_status_with_peers(app: tauri::AppHandle, user_id: String) -> Result<serde_json::Value, String> {
    let global_service = ZYNKSYNC_SERVICE.lock().await;
    let service = match global_service.as_ref() {
        Some(svc) => svc,
        None => return Err("ZynkSync not started".to_string()),
    };

    let peers = service.get_peers().await;
    let paired_peers: Vec<_> = peers.iter().filter(|p| p.paired).collect();

    if paired_peers.is_empty() {
        return Ok(serde_json::json!({
            "needs_prompt": false,
            "reason": "no_peers"
        }));
    }

    // Get local inventory
    let local_inventory = service.get_local_inventory_public(&user_id).await?;

    // Check each peer
    let mut local_is_more_recent = false;
    let mut peers_with_different_counts = Vec::new();

    for peer in paired_peers {
        // Get remote inventory
        match service.get_remote_inventory_public(&peer.url, &user_id).await {
            Ok(remote_inventory) => {
                // Compare timestamps
                let is_more_recent = match (&local_inventory.latest_activity, &remote_inventory.latest_activity) {
                    (Some(local_time), Some(remote_time)) => local_time > remote_time,
                    (Some(_), None) => true,
                    _ => false,
                };

                if is_more_recent && local_inventory.memory_count != remote_inventory.memory_count {
                    local_is_more_recent = true;
                    peers_with_different_counts.push(serde_json::json!({
                        "device_id": peer.device_id,
                        "device_name": peer.device_name,
                        "local_count": local_inventory.memory_count,
                        "remote_count": remote_inventory.memory_count,
                        "local_time": local_inventory.latest_activity,
                        "remote_time": remote_inventory.latest_activity,
                    }));
                }
            }
            Err(e) => {
                println!("[ZynkSync] Warning: Could not get inventory from {}: {}", peer.device_name, e);
            }
        }
    }

    if local_is_more_recent && !peers_with_different_counts.is_empty() {
        // Emit event to frontend
        let _ = app.emit("sync_prompt_needed", serde_json::json!({
            "local_memory_count": local_inventory.memory_count,
            "peers": peers_with_different_counts,
        }));

        Ok(serde_json::json!({
            "needs_prompt": true,
            "local_memory_count": local_inventory.memory_count,
            "peers": peers_with_different_counts,
        }))
    } else {
        Ok(serde_json::json!({
            "needs_prompt": false,
            "reason": "no_difference"
        }))
    }
}

/// Force all paired peers to sync to this device's state (including deletions)
/// This is called when user confirms they want to update other devices
#[tauri::command]
async fn broadcast_sync_to_all_peers(user_id: String) -> Result<Vec<SyncResult>, String> {
    let global_service = ZYNKSYNC_SERVICE.lock().await;
    let service = match global_service.as_ref() {
        Some(svc) => Arc::clone(svc),
        None => return Err("ZynkSync not started".to_string()),
    };

    drop(global_service); // Release lock before async operations

    let peers = service.get_peers().await;
    let paired_peers: Vec<_> = peers.into_iter().filter(|p| p.paired).collect();

    if paired_peers.is_empty() {
        return Ok(Vec::new());
    }

    println!("[ZynkSync] Broadcasting sync to {} paired devices", paired_peers.len());

    let mut results = Vec::new();
    for peer in paired_peers {
        match service.sync_bidirectional(&peer.device_id, &user_id).await {
            Ok(result) => {
                println!("[ZynkSync] ✓ Synced with {}: sent={}, received={}",
                    peer.device_name, result.memories_sent, result.memories_received);
                results.push(result);
            }
            Err(e) => {
                println!("[ZynkSync] ✗ Failed to sync with {}: {}", peer.device_name, e);
                results.push(SyncResult {
                    peer_device_id: peer.device_id,
                    peer_device_name: peer.device_name,
                    memories_sent: 0,
                    memories_received: 0,
                    conflicts_resolved: 0,
                    success: false,
                    error: Some(e),
                });
            }
        }
    }

    Ok(results)
}

/// Get the local IP address of this device
#[tauri::command]
async fn get_local_ip() -> Result<String, String> {
    use std::net::UdpSocket;

    // Connect to a public DNS server to determine local IP
    // This doesn't actually send data, just determines routing
    match UdpSocket::bind("0.0.0.0:0") {
        Ok(socket) => {
            match socket.connect("8.8.8.8:80") {
                Ok(_) => {
                    match socket.local_addr() {
                        Ok(addr) => Ok(addr.ip().to_string()),
                        Err(e) => Err(format!("Failed to get local address: {}", e))
                    }
                }
                Err(e) => Err(format!("Failed to connect: {}", e))
            }
        }
        Err(e) => Err(format!("Failed to bind socket: {}", e))
    }
}

/// Clear all memories for a specific user (for demo/testing purposes)
///
/// Parameters:
/// - user_id: The user ID whose memories to clear
/// - propagate: Whether to propagate deletions to paired devices (default: true)
///              Set to false during identity adoption to prevent deleting memories on other devices
#[tauri::command]
async fn clear_all_memories(user_id: String, propagate: Option<bool>) -> Result<serde_json::Value, String> {
    let should_propagate = propagate.unwrap_or(true);

    println!("[Memory] Clearing all memories for user: {} (propagate: {})", user_id, should_propagate);

    let db_url = crate::db::get_db_url();

    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    // Get content hashes BEFORE deletion (only if we'll propagate)
    let content_hashes = if should_propagate {
        let contents: Vec<String> = sqlx::query_scalar::<_, String>(
            "SELECT content FROM memories WHERE user_id = ?"
        )
        .bind(&user_id)
        .fetch_all(&pool)
        .await
        .map_err(|e| format!("Failed to get memory contents: {}", e))?;
        use sha2::{Digest, Sha256};
        contents.iter().map(|c| format!("{:x}", Sha256::digest(c.as_bytes()))).collect()
    } else {
        Vec::new()
    };

    if should_propagate && !content_hashes.is_empty() {
        println!("[Memory] Found {} memories to delete and propagate", content_hashes.len());
    } else if !should_propagate {
        println!("[Memory] Deletion propagation disabled (identity adoption cleanup)");
    }

    // Delete from local database
    let result = sqlx::query("DELETE FROM memories WHERE user_id = ?")
        .bind(&user_id)
        .execute(&pool)
        .await
        .map_err(|e| format!("Failed to clear memories: {}", e))?;

    let deleted_count = result.rows_affected();
    println!("[Memory] Deleted {} memories from local database", deleted_count);

    // Propagate deletions to paired devices via ZynkSync (only if enabled)
    if should_propagate && !content_hashes.is_empty() {
        println!("[Memory] Propagating {} deletions to paired devices...", content_hashes.len());

        let service = {
            let global_service = ZYNKSYNC_SERVICE.lock().await;
            global_service.as_ref().cloned()
        };

        if let Some(service) = service {
            let mut propagated = 0;
            for hash in content_hashes {
                match service.propagate_deletion_by_hash(hash).await {
                    Ok(count) => propagated += count,
                    Err(e) => eprintln!("[Memory] Failed to propagate deletion: {}", e),
                }
            }
            println!("[Memory] ✓ Propagated deletions to {} device(s)", propagated);
        } else {
            println!("[Memory] ⚠️ ZynkSync not initialized - deletions not propagated");
        }
    }

    // Clear the user profile (name, age) so a demo persona (e.g. Einstein) doesn't
    // persist across sessions after the user resets their memory.
    let profile_path = crate::db::get_user_profile_path();
    if profile_path.exists() {
        std::fs::remove_file(&profile_path).ok();
        println!("[Memory] Deleted user_profile.json");
    }

    Ok(serde_json::json!({
        "success": true,
        "deleted_count": deleted_count
    }))
}

/// Get current user and device identity
#[tauri::command]
async fn get_user_identity() -> Result<user_identity::UserIdentity, String> {
    user_identity::get_identity()
}

/// Migrate all memories from old user_id to new user_id
/// Used during identity adoption to preserve memories instead of deleting them
#[tauri::command]
async fn migrate_user_memories(old_user_id: String, new_user_id: String) -> Result<i64, String> {
    println!("[Memory] Migrating memories from {} to {}", old_user_id, new_user_id);

    let db_url = crate::db::get_db_url();

    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    // Update user_id for all memories
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

/// Clear all conversation history for a user (sessions + messages via CASCADE)
#[tauri::command]
async fn clear_conversation_history(user_id: String) -> Result<serde_json::Value, String> {
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    let result = sqlx::query("DELETE FROM conversation_sessions WHERE user_id = ?")
        .bind(&user_id)
        .execute(&pool)
        .await
        .map_err(|e| format!("Failed to clear conversation history: {}", e))?;

    let deleted = result.rows_affected();
    println!("[ConvHistory] Cleared {} sessions for user {}", deleted, &user_id[..8.min(user_id.len())]);

    Ok(serde_json::json!({ "deleted_count": deleted }))
}

/// Read a text file from disk and return its contents as a string.
/// Used by the file-attachment feature in the chat UI.
#[tauri::command]
async fn read_text_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read file '{}': {}", path, e))
}

/// Set user_id manually (for device linking via sync code)
#[tauri::command]
async fn set_user_identity(user_id: String) -> Result<(), String> {
    user_identity::set_user_id(&user_id)
}

/// Reset user identity - creates completely new user_id and device_id
#[tauri::command]
async fn reset_user_identity() -> Result<user_identity::UserIdentity, String> {
    // Call reset_all to create new IDs
    let (new_user_id, new_device_id) = user_identity::reset_all_identity()?;

    println!("[Identity] Reset complete - New user_id: {}, New device_id: {}", new_user_id, new_device_id);

    // Return the new identity
    user_identity::get_identity()
}

/// Generate a 6-digit one-time sync code (expires in 5 minutes)
#[tauri::command]
async fn generate_sync_code() -> Result<serde_json::Value, String> {
    let user_id = user_identity::get_user_id()?;
    let device_id = user_identity::get_device_id()?;

    let db_url = crate::db::get_db_url();

    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    let code = sync_codes::generate_sync_code(&user_id, &device_id, &pool).await?;

    Ok(serde_json::json!({
        "code": code,
        "expires_in": 300  // 5 minutes
    }))
}

/// Verify a sync code and return user_id without consuming it
#[tauri::command]
async fn verify_sync_code_info(code: String) -> Result<sync_codes::SyncCodeInfo, String> {
    let db_url = crate::db::get_db_url();

    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    sync_codes::get_code_info(&code, &pool).await
}

/// Sync with remote device using code (Device B verifying with Device A)
#[tauri::command]
async fn sync_with_code(code: String, device_ip: String) -> Result<serde_json::Value, String> {
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

// =============================================================================
// ZynkLink Commands - File Sharing with Code-Based Pairing
// =============================================================================

/// Generate a ZynkLink code for file sharing
#[tauri::command]
async fn generate_zynklink_code() -> Result<serde_json::Value, String> {
    println!("[ZynkLink] Generate code command invoked");

    let user_id = user_identity::get_user_id()
        .map_err(|e| { println!("[ZynkLink] Failed to get user_id: {}", e); e })?;
    let device_id = user_identity::get_device_id()
        .map_err(|e| { println!("[ZynkLink] Failed to get device_id: {}", e); e })?;

    println!("[ZynkLink] user_id: {}..., device_id: {}...", &user_id[..8], &device_id[..8]);

    // Use the shared service pool so this command uses the same properly-configured
    // connection (WAL mode, busy_timeout=15s) as the Axum HTTP server.
    let pool = {
        let guard = ZYNKSYNC_SERVICE.lock().await;
        match guard.as_ref() {
            Some(service) => service.get_db_pool(),
            None => {
                println!("[ZynkLink] Service not started, falling back to create_pool");
                drop(guard);
                crate::db::create_pool().await.map_err(|e| format!("Failed to connect to database: {}", e))?
            }
        }
    };

    let device_name = hostname::get()
        .map_err(|e| format!("Failed to get hostname: {}", e))?
        .to_string_lossy()
        .to_string();

    sqlx::query(
        "INSERT INTO zynk_devices (device_id, device_name, owner_user_id, is_paired, port, created_at, last_seen_at)
         VALUES (?, ?, ?, true, 57963, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
         ON CONFLICT (device_id) DO UPDATE
         SET owner_user_id = ?, last_seen_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
    )
    .bind(&device_id)
    .bind(&device_name)
    .bind(&user_id)
    .bind(&user_id)
    .execute(&pool)
    .await
    .map_err(|e| { println!("[ZynkLink] Failed to ensure device entry: {}", e); format!("Failed to ensure device entry: {}", e) })?;

    println!("[ZynkLink] Generating code...");
    let code = zynklink::generate_zynklink_code(&pool, &user_id, &device_id).await
        .map_err(|e| { println!("[ZynkLink] Code generation failed: {}", e); e })?;

    println!("[ZynkLink] Code generated successfully: {}", code);

    Ok(serde_json::json!({
        "success": true,
        "code": code
    }))
}

/// Link with remote device using ZynkLink code (Device B linking with Device A)
#[tauri::command]
async fn link_with_zynklink_code(app: tauri::AppHandle, code: String, device_ip: String) -> Result<serde_json::Value, String> {
    println!("[ZynkLink] Device B: Linking with code {} to remote device {}", code, device_ip);

    let user_id = user_identity::get_user_id()?;
    let device_id = user_identity::get_device_id()?;

    // Step 1: Verify code on REMOTE device (Device A) via HTTPS.
    // Peer cert is not yet pinned — TOFU accepted; the code exchange provides authentication.
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("Failed to build link client: {}", e))?;
    let verify_url = format!("https://{}:57963/api/zynklink/verify-code", device_ip);

    println!("[ZynkLink] Device B: Verifying code with remote device...");

    let response = client
        .post(&verify_url)
        .json(&serde_json::json!({
            "code": code
        }))
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

    println!("[ZynkLink] Device B: Code verified! Remote user_id: {}...", &verify_data.user_id[..8]);

    // Step 2: Notify remote device to accept the pairing and create local pairing
    let accept_url = format!("https://{}:57963/api/zynklink/accept-code", device_ip);

    // Get our local IP to send to the remote device
    let local_ip = {
        use std::net::UdpSocket;
        match UdpSocket::bind("0.0.0.0:0") {
            Ok(socket) => {
                match socket.connect("8.8.8.8:80") {
                    Ok(_) => {
                        socket.local_addr()
                            .map(|addr| addr.ip().to_string())
                            .unwrap_or_else(|_| "unknown".to_string())
                    }
                    Err(_) => "unknown".to_string()
                }
            }
            Err(_) => "unknown".to_string()
        }
    };

    println!("[ZynkLink] Device B: Requesting pairing acceptance (our IP: {})...", local_ip);

    let accept_response = client
        .post(&accept_url)
        .json(&serde_json::json!({
            "code": code,
            "acceptor_user_id": user_id,
            "acceptor_device_id": device_id,
            "acceptor_device_ip": local_ip
        }))
        .send()
        .await
        .map_err(|e| format!("Failed to create pairing: {}", e))?;

    if !accept_response.status().is_success() {
        let error_text = accept_response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("Pairing acceptance failed: {}", error_text));
    }

    // Parse response to get Device A's IP address
    let accept_data: serde_json::Value = accept_response
        .json()
        .await
        .map_err(|e| format!("Failed to parse accept response: {}", e))?;

    let creator_device_ip = accept_data.get("creator_device_ip")
        .and_then(|ip| ip.as_str())
        .unwrap_or(&device_ip); // Fallback to manual IP if not provided

    println!("[ZynkLink] Device B: Pairing created on remote device! Creator IP: {}", creator_device_ip);

    // Step 3: Create local pairing record using the shared service pool
    let pool = {
        let guard = ZYNKSYNC_SERVICE.lock().await;
        match guard.as_ref() {
            Some(service) => service.get_db_pool(),
            None => {
                drop(guard);
                crate::db::create_pool().await.map_err(|e| format!("Failed to connect to database: {}", e))?
            }
        }
    };

    // Ensure both devices exist in zynk_devices (required for foreign key constraint)
    println!("[ZynkLink] Device B: Ensuring local device is registered with our IP {}...", local_ip);
    let device_name = hostname::get()
        .map_err(|e| format!("Failed to get hostname: {}", e))?
        .to_string_lossy()
        .to_string();

    sqlx::query(
        "INSERT INTO zynk_devices (device_id, device_name, device_ip, owner_user_id, is_paired, port, created_at, last_seen_at)
         VALUES (?, ?, ?, ?, true, 57963, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
         ON CONFLICT (device_id) DO UPDATE
         SET device_ip = ?, owner_user_id = ?, last_seen_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
    )
    .bind(&device_id)
    .bind(&device_name)
    .bind(&local_ip)
    .bind(&user_id)
    .bind(&local_ip)
    .bind(&user_id)
    .execute(&pool)
    .await
    .map_err(|e| format!("Failed to ensure local device entry: {}", e))?;

    println!("[ZynkLink] Device B: Ensuring remote device is registered with IP {}...", creator_device_ip);
    sqlx::query(
        "INSERT INTO zynk_devices (device_id, device_name, device_ip, owner_user_id, is_paired, port, created_at, last_seen_at)
         VALUES (?, ?, ?, ?, true, 57963, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
         ON CONFLICT (device_id) DO UPDATE
         SET device_ip = ?, owner_user_id = ?, last_seen_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
    )
    .bind(&verify_data.device_id)
    .bind(&format!("Remote Device {}", &verify_data.device_id[..8]))
    .bind(creator_device_ip)
    .bind(&verify_data.user_id)
    .bind(creator_device_ip)
    .bind(&verify_data.user_id)
    .execute(&pool)
    .await
    .map_err(|e| format!("Failed to ensure remote device entry: {}", e))?;

    println!("[ZynkLink] Device B: Both devices registered successfully");

    // Create bidirectional pairing (user1 < user2 for consistency)
    let (user1_id, user2_id, device1_id, device2_id) = if user_id.as_str() < verify_data.user_id.as_str() {
        (user_id.clone(), verify_data.user_id.clone(), device_id.clone(), verify_data.device_id.clone())
    } else {
        (verify_data.user_id.clone(), user_id.clone(), verify_data.device_id.clone(), device_id.clone())
    };

    sqlx::query(
        "INSERT INTO zynklink_pairings (user1_id, user2_id, device1_id, device2_id, is_active)
         VALUES (?, ?, ?, ?, 1)
         ON CONFLICT (user1_id, user2_id) DO UPDATE SET is_active = 1, linked_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
    )
    .bind(&user1_id)
    .bind(&user2_id)
    .bind(&device1_id)
    .bind(&device2_id)
    .execute(&pool)
    .await
    .map_err(|e| format!("Failed to create local pairing: {}", e))?;

    println!("[ZynkLink] Device B: Local pairing record created!");

    // Emit event to refresh UI immediately
    println!("[ZynkLink] Device B: Emitting zynklink-pairing-updated event");
    let _ = app.emit("zynklink-pairing-updated", serde_json::json!({
        "remote_user_id": verify_data.user_id,
        "remote_device_id": verify_data.device_id
    }));

    Ok(serde_json::json!({
        "success": true,
        "message": format!("Successfully linked for file sharing with user {}", verify_data.user_id),
        "remote_user_id": verify_data.user_id,
        "remote_device_id": verify_data.device_id
    }))
}

/// List all ZynkLink pairings
#[tauri::command]
async fn list_zynklink_pairings() -> Result<serde_json::Value, String> {
    let user_id = user_identity::get_user_id()?;
    let device_id = user_identity::get_device_id()?;

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    zynklink::list_zynklink_pairings(&pool, &user_id, &device_id).await
}

/// Revoke a ZynkLink pairing
#[tauri::command]
async fn revoke_zynklink_pairing(linked_user_id: String) -> Result<serde_json::Value, String> {
    let user_id = user_identity::get_user_id()?;

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    zynklink::revoke_zynklink_pairing(&pool, &user_id, &linked_user_id).await
}

/// Pause or resume a ZynkLink pairing (session-only; clears on restart)
#[tauri::command]
async fn toggle_zynklink_pause(linked_device_id: String, paused: bool) -> Result<serde_json::Value, String> {
    zynklink::set_link_paused(&linked_device_id, paused).await;
    Ok(serde_json::json!({ "success": true, "paused": paused }))
}

/// Share a directory
#[tauri::command]
async fn share_directory(local_path: String, share_name: String, is_readable: bool, is_writable: bool) -> Result<serde_json::Value, String> {
    let user_id = user_identity::get_user_id()?;
    let device_id = user_identity::get_device_id()?;

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    // Ensure device exists in zynk_devices (required for foreign key constraint)
    // ZynkLink can work independently of ZynkSync, so we create a self-entry if needed
    let device_name = hostname::get()
        .map_err(|e| format!("Failed to get hostname: {}", e))?
        .to_string_lossy()
        .to_string();

    sqlx::query(
        "INSERT INTO zynk_devices (device_id, device_name, owner_user_id, is_paired, port, created_at, last_seen_at)
         VALUES (?, ?, ?, true, 57963, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
         ON CONFLICT (device_id) DO UPDATE
         SET owner_user_id = ?, last_seen_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
    )
    .bind(&device_id)
    .bind(&device_name)
    .bind(&user_id)
    .execute(&pool)
    .await
    .map_err(|e| format!("Failed to ensure device entry: {}", e))?;

    let request = zynklink::ShareDirectoryRequest {
        local_path,
        share_name,
        is_readable,
        is_writable,
    };

    let response = zynklink::share_directory(&pool, &device_id, request).await?;
    Ok(serde_json::to_value(response).map_err(|e| format!("Serialization error: {}", e))?)
}

/// Unshare a directory
#[tauri::command]
async fn unshare_directory(share_id: i32) -> Result<serde_json::Value, String> {
    let device_id = user_identity::get_device_id()?;

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    zynklink::unshare_directory(&pool, &device_id, share_id).await
}

/// List my shared directories
#[tauri::command]
async fn list_my_shared_directories() -> Result<serde_json::Value, String> {
    let device_id = user_identity::get_device_id()?;

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let response = zynklink::list_my_shared_directories(&pool, &device_id).await?;
    Ok(serde_json::to_value(response).map_err(|e| format!("Serialization error: {}", e))?)
}

/// List remote shared directories (from paired users)
/// Fetches directories via HTTP from remote devices
#[tauri::command]
async fn list_remote_directories() -> Result<serde_json::Value, String> {
    let user_id = user_identity::get_user_id()?;
    let device_id = user_identity::get_device_id()?;

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    // Get paired device IDs and their IPs
    let paired_devices = sqlx::query_as::<_, (String, String, Option<String>)>(
        "SELECT
            CASE WHEN device1_id = ? THEN device2_id ELSE device1_id END as device_id,
            CASE WHEN user1_id = ? THEN user2_id ELSE user1_id END as user_id,
            zd.device_ip
         FROM zynklink_pairings zp
         LEFT JOIN zynk_devices zd ON (CASE WHEN device1_id = ? THEN device2_id ELSE device1_id END) = zd.device_id
         WHERE (user1_id = ? OR user2_id = ?) AND is_active = 1"
    )
    .bind(&device_id)
    .bind(&user_id)
    .bind(&device_id)
    .bind(&user_id)
    .bind(&user_id)
    .fetch_all(&pool)
    .await
    .map_err(|e| format!("Failed to get paired devices: {}", e))?;


    let mut all_directories = Vec::new();
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    // Fetch directories from each remote device via HTTPS
    for (remote_device_id, _remote_user_id, device_ip_opt) in paired_devices {
        if let Some(device_ip) = device_ip_opt {
            let url = format!("https://{}:57963/api/zynklink/directories", device_ip);

            match client
                .post(&url)
                .json(&serde_json::json!({
                    "device_id": remote_device_id,
                    "requester_user_id": user_id
                }))
                .send()
                .await
            {
                Ok(response) => {
                    if response.status().is_success() {
                        // Update last_seen_at on successful connection
                        let _ = sqlx::query(
                            "UPDATE zynk_devices SET last_seen_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE device_id = ?"
                        )
                        .bind(&remote_device_id)
                        .execute(&pool)
                        .await;

                        match response.json::<serde_json::Value>().await {
                            Ok(data) => {
                                if let Some(dirs) = data.get("shared_directories").and_then(|d| d.as_array()) {
                                    for dir in dirs {
                                        all_directories.push(dir.clone());
                                    }
                                }
                            }
                            Err(e) => {
                                println!("[ZynkLink] Failed to parse response from {}: {}", device_ip, e);
                            }
                        }
                    } else {
                        println!("[ZynkLink] HTTP error from {}: {}", device_ip, response.status());
                    }
                }
                Err(_e) => {} // Device offline - silent, polled every 5s
            }
        } else {
            // No IP stored for this device yet
        }
    }


    Ok(serde_json::json!({
        "shared_directories": all_directories
    }))
}

/// Scan a shared directory and index files
#[tauri::command]
async fn scan_shared_directory(share_id: i32, max_files: Option<usize>) -> Result<serde_json::Value, String> {
    let device_id = user_identity::get_device_id()?;

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    zynklink::scan_directory(&pool, &device_id, share_id, max_files).await
}

/// List files in a shared directory
/// If the directory belongs to a remote device, fetches via HTTP
#[tauri::command]
async fn list_shared_files(share_id: i32, device_id: String) -> Result<serde_json::Value, String> {
    let local_device_id = user_identity::get_device_id()?;
    let local_user_id = user_identity::get_user_id()?;

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    // If this is a local share (device_id matches our device), query the local database
    if device_id == local_device_id {
        println!("[ZynkLink] Listing files for local share {} (device {}...)", share_id, &device_id[..8]);
        let response = zynklink::list_files(&pool, share_id).await?;
        return Ok(serde_json::to_value(response).map_err(|e| format!("Serialization error: {}", e))?);
    }

    // If this is a remote share, fetch via HTTP
    // Look up the IP address of the remote device
    let device_ip_opt = sqlx::query_as::<_, (Option<String>,)>(
        "SELECT device_ip FROM zynk_devices WHERE device_id = ?"
    )
    .bind(&device_id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| format!("Failed to get device IP: {}", e))?
    .and_then(|r| r.0);

    if let Some(device_ip) = device_ip_opt {
        println!("[ZynkLink] Fetching files for remote share {} from device {}... at {}", share_id, &device_id[..8], device_ip);

        let client = reqwest::Client::builder()
            .danger_accept_invalid_certs(true)
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        let url = format!("https://{}:57963/api/zynklink/files", device_ip);

        match client
            .post(&url)
            .json(&serde_json::json!({
                "share_id": share_id,
                "requester_user_id": local_user_id
            }))
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    match response.json::<serde_json::Value>().await {
                        Ok(data) => {
                            println!("[ZynkLink] Fetched {} files from remote share",
                                data.get("files").and_then(|f| f.as_array()).map(|a| a.len()).unwrap_or(0));
                            return Ok(data);
                        }
                        Err(e) => {
                            return Err(format!("Failed to parse response: {}", e));
                        }
                    }
                } else {
                    return Err(format!("HTTP error: {}", response.status()));
                }
            }
            Err(e) => {
                return Err(format!("Failed to connect to remote device: {}", e));
            }
        }
    } else {
        return Err("No IP address available for remote device".to_string());
    }
}

/// Get file path for download
#[tauri::command]
async fn get_shared_file_path(share_id: i32, relative_path: String) -> Result<String, String> {
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let path = zynklink::get_file_path(&pool, share_id, &relative_path).await?;
    Ok(path.to_string_lossy().to_string())
}

/// Cancel an in-flight ZynkLink download.
/// Flips the AtomicBool flag in DOWNLOAD_CANCELS for the given relative_path so
/// the streaming loop in download_to_custom_location / download_to_knowledge_base
/// sees it on its next chunk-poll, breaks out, deletes its .part file, and
/// returns Err("Cancelled by user") to the Tauri caller.
#[tauri::command]
fn cancel_zynklink_download(relative_path: String) -> Result<(), String> {
    let registry = DOWNLOAD_CANCELS.lock()
        .map_err(|e| format!("Cancel registry lock poisoned: {}", e))?;
    if let Some(flag) = registry.get(&relative_path) {
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
        println!("[ZynkLink] Cancel requested for: {}", relative_path);
        Ok(())
    } else {
        Err(format!("No active download found for: {}", relative_path))
    }
}

/// Download a file to the Knowledge Base folder
#[tauri::command]
async fn download_to_knowledge_base(
    app: tauri::AppHandle,
    share_id: i32,
    relative_path: String,
    device_id: String,
    user_id: String
) -> Result<String, String> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    println!("[KB Download] Starting download - share_id: {}, path: {}, device: {}...",
        share_id, relative_path, &device_id[..8]);

    let local_device_id = user_identity::get_device_id()?;
    let local_user_id = user_identity::get_user_id()?;
    println!("[KB Download] Local device: {}...", &local_device_id[..8]);

    // Get user's knowledge base folder path (cross-platform)
    // Linux: ~/.config/Zynkbot/KnowledgeBase/{user_id}/
    // macOS: ~/Library/Application Support/Zynkbot/KnowledgeBase/{user_id}/
    // Windows: %APPDATA%\Zynkbot\KnowledgeBase\{user_id}\
    let kb_path = kb_rag::get_kb_folder_path(&user_id)?;

    println!("[KB Download] Target directory: {}", kb_path.display());

    // Get filename from relative_path
    let filename = std::path::Path::new(&relative_path)
        .file_name()
        .ok_or("Invalid filename")?
        .to_string_lossy()
        .to_string();

    let destination_path = kb_path.join(&filename);
    let destination_path_str = destination_path.to_string_lossy().to_string();
    println!("[KB Download] Destination: {}", destination_path.display());

    // Connect to database
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    // Check if this is a local or remote share
    if device_id == local_device_id {
        println!("[KB Download] Local file - copying from local share");
        // Local file - just copy it
        let source_path = zynklink::get_file_path(&pool, share_id, &relative_path).await?;
        println!("[KB Download] Source path: {}", source_path.display());

        tokio::fs::copy(&source_path, &destination_path)
            .await
            .map_err(|e| format!("Failed to copy file: {}", e))?;

        println!("[KB Download] ✓ Local file copied successfully");
    } else {
        println!("[KB Download] Remote file - downloading from device {}...", &device_id[..8]);
        // Remote file - stream over HTTP. Previous version buffered the entire
        // response body in RAM via response.bytes(), which allocated multi-GB on
        // the receiver for large files.

        let device_ip_opt = sqlx::query_as::<_, (Option<String>,)>(
            "SELECT device_ip FROM zynk_devices WHERE device_id = ?"
        )
        .bind(&device_id)
        .fetch_optional(&pool)
        .await
        .map_err(|e| format!("Failed to get device IP: {}", e))?
        .and_then(|r| r.0);

        let device_ip = device_ip_opt.ok_or("No IP address available for remote device")?;
        println!("[KB Download] Downloading from {}", device_ip);

        let client = reqwest::Client::builder()
            .danger_accept_invalid_certs(true)
            .build()
            .map_err(|e| format!("Failed to build download client: {}", e))?;
        let url = format!("https://{}:57963/api/zynklink/download", device_ip);

        let response = client
            .post(&url)
            .json(&serde_json::json!({
                "share_id": share_id,
                "relative_path": relative_path,
                "requester_user_id": local_user_id
            }))
            .send()
            .await
            .map_err(|e| format!("Failed to connect to remote device: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("HTTP error: {}", response.status()));
        }

        let total_bytes = response.content_length();

        // Register with the cancellation registry; same RAII pattern as the
        // custom-location download above.
        let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let mut registry = DOWNLOAD_CANCELS.lock()
                .map_err(|e| format!("Cancel registry lock poisoned: {}", e))?;
            registry.insert(relative_path.clone(), cancel_flag.clone());
        }
        struct CancelGuard(String);
        impl Drop for CancelGuard {
            fn drop(&mut self) {
                if let Ok(mut r) = DOWNLOAD_CANCELS.lock() {
                    r.remove(&self.0);
                }
            }
        }
        let _cancel_guard = CancelGuard(relative_path.clone());

        // Stream to .part file, rename on success.
        let temp_path = format!("{}.part", destination_path_str);
        let mut file = tokio::fs::File::create(&temp_path).await
            .map_err(|e| format!("Failed to create destination file: {}", e))?;

        let _ = app.emit("zynklink:download:start", serde_json::json!({
            "share_id": share_id,
            "relative_path": &relative_path,
            "destination": &destination_path_str,
            "total_bytes": total_bytes,
        }));

        let mut bytes_written: u64 = 0;
        let mut last_event_at: u64 = 0;
        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            if cancel_flag.load(std::sync::atomic::Ordering::Relaxed) {
                drop(file);
                let _ = tokio::fs::remove_file(&temp_path).await;
                let _ = app.emit("zynklink:download:cancelled", serde_json::json!({
                    "share_id": share_id,
                    "relative_path": &relative_path,
                    "bytes_written": bytes_written,
                }));
                return Err("Cancelled by user".to_string());
            }

            let chunk = chunk_result
                .map_err(|e| format!("Network read error mid-transfer: {}", e))?;
            file.write_all(&chunk).await
                .map_err(|e| format!("Failed to write chunk to disk: {}", e))?;
            bytes_written += chunk.len() as u64;

            if bytes_written - last_event_at >= 262_144 {
                let _ = app.emit("zynklink:download:progress", serde_json::json!({
                    "share_id": share_id,
                    "relative_path": &relative_path,
                    "bytes_written": bytes_written,
                    "total_bytes": total_bytes,
                }));
                last_event_at = bytes_written;
            }
        }

        file.flush().await
            .map_err(|e| format!("Failed to flush destination file: {}", e))?;
        drop(file);

        tokio::fs::rename(&temp_path, &destination_path).await
            .map_err(|e| format!("Failed to finalize download (rename .part): {}", e))?;

        let _ = app.emit("zynklink:download:complete", serde_json::json!({
            "share_id": share_id,
            "relative_path": &relative_path,
            "destination": &destination_path_str,
            "total_bytes": bytes_written,
        }));

        println!("[KB Download] ✓ Remote file streamed to disk ({} bytes)", bytes_written);
    }

    // Index the file into the knowledge base
    // Use filename as virtual path so it appears at KB root level in UI
    println!("[KB Download] Indexing file into knowledge base...");

    // Read file content for indexing
    let file_content = tokio::fs::read_to_string(&destination_path)
        .await
        .map_err(|e| format!("Failed to read file for indexing: {}", e))?;

    // Use just the filename as the virtual path (not full absolute path)
    // This makes it appear at the root level in the folder tree UI
    match kb_rag::index_text_as_document(&pool, &user_id, &filename, &file_content).await {
        Ok(doc_id) => {
            println!("[KB Download] ✓ File indexed successfully (doc_id: {})", doc_id);
        }
        Err(e) => {
            println!("[KB Download] ⚠️  Warning: File saved but indexing failed: {}", e);
            // Don't fail the whole operation if indexing fails - file is still saved
        }
    }

    println!("[KB Download] ✓ Complete: {}", destination_path.display());
    Ok(destination_path.to_string_lossy().to_string())
}

/// Download a file to a custom location (with file picker)
#[tauri::command]
async fn download_to_custom_location(
    app: tauri::AppHandle,
    share_id: i32,
    relative_path: String,
    device_id: String,
    destination_path: String
) -> Result<String, String> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    let local_device_id = user_identity::get_device_id()?;
    let local_user_id = user_identity::get_user_id()?;

    // Check if this is a local or remote share
    if device_id == local_device_id {
        // Local file - just copy it
        let db_url = crate::db::get_db_url();
        let pool = sqlx::SqlitePool::connect(&db_url)
            .await
            .map_err(|e| format!("Failed to connect to database: {}", e))?;

        let source_path = zynklink::get_file_path(&pool, share_id, &relative_path).await?;
        tokio::fs::copy(&source_path, &destination_path)
            .await
            .map_err(|e| format!("Failed to copy file: {}", e))?;
    } else {
        // Remote file - download via HTTP, streamed in 64KB chunks. Previous code
        // buffered the entire response body in RAM via response.bytes(), then wrote
        // it all at once — allocating multi-GB on the receiver for large files.
        let db_url = crate::db::get_db_url();
        let pool = sqlx::SqlitePool::connect(&db_url)
            .await
            .map_err(|e| format!("Failed to connect to database: {}", e))?;

        // Get device IP
        let device_ip_opt = sqlx::query_as::<_, (Option<String>,)>(
            "SELECT device_ip FROM zynk_devices WHERE device_id = ?"
        )
        .bind(&device_id)
        .fetch_optional(&pool)
        .await
        .map_err(|e| format!("Failed to get device IP: {}", e))?
        .and_then(|r| r.0);

        let device_ip = device_ip_opt.ok_or("No IP address available for remote device")?;

        let client = reqwest::Client::builder()
            .danger_accept_invalid_certs(true)
            .build()
            .map_err(|e| format!("Failed to build download client: {}", e))?;
        let url = format!("https://{}:57963/api/zynklink/download", device_ip);

        let response = client
            .post(&url)
            .json(&serde_json::json!({
                "share_id": share_id,
                "relative_path": relative_path,
                "requester_user_id": local_user_id
            }))
            .send()
            .await
            .map_err(|e| format!("Failed to connect to remote device: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("HTTP error: {}", response.status()));
        }

        let total_bytes = response.content_length();

        // Register this download in the cancellation registry so the user can
        // cancel via the Tauri command. The flag is removed on every exit path
        // (success, error, cancellation) by the CancelGuard drop impl below.
        let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let mut registry = DOWNLOAD_CANCELS.lock()
                .map_err(|e| format!("Cancel registry lock poisoned: {}", e))?;
            registry.insert(relative_path.clone(), cancel_flag.clone());
        }
        // RAII guard ensures the registry entry goes away on any return path.
        struct CancelGuard(String);
        impl Drop for CancelGuard {
            fn drop(&mut self) {
                if let Ok(mut r) = DOWNLOAD_CANCELS.lock() {
                    r.remove(&self.0);
                }
            }
        }
        let _cancel_guard = CancelGuard(relative_path.clone());

        // Write to a .part file first; rename to the final destination only after the
        // stream completes cleanly. Avoids leaving a corrupt destination if the
        // transfer fails or is interrupted partway.
        let temp_path = format!("{}.part", destination_path);
        let mut file = tokio::fs::File::create(&temp_path).await
            .map_err(|e| format!("Failed to create destination file: {}", e))?;

        let _ = app.emit("zynklink:download:start", serde_json::json!({
            "share_id": share_id,
            "relative_path": &relative_path,
            "destination": &destination_path,
            "total_bytes": total_bytes,
        }));

        let mut bytes_written: u64 = 0;
        let mut last_event_at: u64 = 0;
        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            // Check for cancellation before processing each chunk.
            if cancel_flag.load(std::sync::atomic::Ordering::Relaxed) {
                drop(file);
                let _ = tokio::fs::remove_file(&temp_path).await;
                let _ = app.emit("zynklink:download:cancelled", serde_json::json!({
                    "share_id": share_id,
                    "relative_path": &relative_path,
                    "bytes_written": bytes_written,
                }));
                return Err("Cancelled by user".to_string());
            }

            let chunk = chunk_result
                .map_err(|e| format!("Network read error mid-transfer: {}", e))?;
            file.write_all(&chunk).await
                .map_err(|e| format!("Failed to write chunk to disk: {}", e))?;
            bytes_written += chunk.len() as u64;

            // Throttle progress events to ~once per 256KB so we don't flood the
            // Tauri event channel on fast LAN transfers.
            if bytes_written - last_event_at >= 262_144 {
                let _ = app.emit("zynklink:download:progress", serde_json::json!({
                    "share_id": share_id,
                    "relative_path": &relative_path,
                    "bytes_written": bytes_written,
                    "total_bytes": total_bytes,
                }));
                last_event_at = bytes_written;
            }
        }

        file.flush().await
            .map_err(|e| format!("Failed to flush destination file: {}", e))?;
        drop(file);

        tokio::fs::rename(&temp_path, &destination_path).await
            .map_err(|e| format!("Failed to finalize download (rename .part): {}", e))?;

        let _ = app.emit("zynklink:download:complete", serde_json::json!({
            "share_id": share_id,
            "relative_path": &relative_path,
            "destination": &destination_path,
            "total_bytes": bytes_written,
        }));
    }

    Ok(destination_path)
}

/// Execute web search with DuckDuckGo and fetch page content
#[tauri::command]
async fn execute_web_search(query: String, max_results: usize, fetch_top_n: usize) -> Result<serde_json::Value, String> {
    println!("[WebSearch] Executing search for: {}", query);

    // Use the web_search module to search and fetch content
    let search_response = web_search::search_with_content(&query, max_results, fetch_top_n).await?;

    // Convert to JSON
    let json_response = serde_json::to_value(&search_response)
        .map_err(|e| format!("Failed to serialize search results: {}", e))?;

    println!("[WebSearch] Search complete - found {} results", search_response.num_results);

    Ok(json_response)
}

/// Returns true if the message appears to be asking about time-sensitive information
/// that a local model cannot reliably answer from training data alone.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Disable GTK overlay scrollbars on Linux — they cause a ghost line flash
    // when modals open because GTK renders them at the OS level, not via CSS.
    #[cfg(target_os = "linux")]
    std::env::set_var("GTK_OVERLAY_SCROLLING", "0");

    tauri::Builder::default()
        .setup(|app| {
            // Load .env file from project root (cross-platform)
            // Try multiple paths to handle different execution contexts
            let env_paths = [
                "../../.env",           // From target/debug/
                "../../../.env",        // Alternative path
                ".env",                 // Current directory
                "../../../../../../.env" // From deep nested paths
            ];

            for path in &env_paths {
                if std::path::Path::new(path).exists() {
                    println!("[Dotenv] Loading .env from: {}", path);
                    dotenv::from_path(path).ok();
                    break;
                }
            }

            // Fallback: Search upwards from current directory
            {  // Load dotenv for API keys
                let mut current_dir = std::env::current_dir().unwrap_or_default();
                for _ in 0..5 {
                    let env_file = current_dir.join(".env");
                    if env_file.exists() {
                        println!("[Dotenv] Found .env at: {:?}", env_file);
                        dotenv::from_path(env_file).ok();
                        break;
                    }
                    if !current_dir.pop() {
                        break;
                    }
                }
            }

            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // Initialize dialog plugin for directory selection (Knowledge Base)
            app.handle().plugin(tauri_plugin_dialog::init())?;

            // Initialize shell plugin for opening folders/URLs
            app.handle().plugin(tauri_plugin_shell::init())?;

            // Store app handle globally for HTTP server event emission
            {
                let mut global_app_handle = APP_HANDLE.lock()
                    .map_err(|e| format!("APP_HANDLE mutex poisoned: {}", e))?;
                *global_app_handle = Some(app.handle().clone());
                println!("[Tauri] ✅ App handle stored globally for event emission");
            }

            println!("[Tauri] Pure Rust backend - NO Flask dependency");
            println!("[Tauri] All operations: local embeddings, DB access, LLM calls");

            // Warmup BERT NER on startup so first duplicate check is fast (3-5s instead of 19s)
            std::thread::spawn(|| {
                println!("[Startup] 🔥 Pre-loading BERT NER model...");
                let start = std::time::Instant::now();
                let enhancer = nlp_enhancer::NLPEnhancer::new();
                let duration = start.elapsed();

                // Test it to ensure it's fully loaded
                let _ = enhancer.extract_entities("Warmup test");

                println!("[Startup] ✅ BERT NER ready ({:.2}s) - First duplicate check will be fast!", duration.as_secs_f32());
            });

            // Auto-initialize system memories, KB docs, and conversation history tables
            std::thread::spawn(|| {
                println!("[System] 🌱 Checking for system initialization...");

                // Create a new Tokio runtime for this thread
                let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
                rt.block_on(async {
                    // Run migrations and set up all tables (idempotent, sqlx tracks versions)
                    match crate::db::create_pool().await {
                        Ok(pool) => {
                            println!("[System] ✅ Database migrations complete");
                            match conversation_history::ensure_tables(&pool).await {
                                Ok(_) => println!("[System] ✅ Conversation history tables ready"),
                                Err(e) => eprintln!("[System] ⚠️ Conversation history table setup error: {}", e),
                            }
                        }
                        Err(e) => eprintln!("[System] ⚠️ DB migration error: {}", e),
                    }

                    // Seed system memories (idempotent - only creates if missing)
                    match commands::onboarding::seed_system_memories().await {
                        Ok(msg) => println!("[System] ✅ {}", msg),
                        Err(e) => eprintln!("[System] ⚠️ System memory seed error: {}", e),
                    }

                    // Index system documentation (idempotent - only indexes if missing)
                    match commands::onboarding::index_system_documentation().await {
                        Ok(msg) => println!("[System KB] ✅ {}", msg),
                        Err(e) => eprintln!("[System KB] ⚠️ System doc indexing error: {}", e),
                    }

                    println!("[System] 🎉 First-time initialization complete (if needed)");
                });
            });

            // Auto-start HTTP server for ZynkSync and ZynkLink
            // This allows both systems to work independently:
            // - ZynkSync: Memory syncing between your devices (same user_id)
            // - ZynkLink: File sharing with other users (different user_id)
            // The server handles both on port 57963 with separate endpoints
            std::thread::spawn(move || {
                println!("[HTTP Server] Thread spawned, creating Tokio runtime...");
                match tokio::runtime::Runtime::new() {
                    Ok(runtime) => {
                        println!("[HTTP Server] ✅ Tokio runtime created");
                        runtime.block_on(async {
                            println!("[HTTP Server] Starting shared server for ZynkSync + ZynkLink...");
                            match auto_start_http_server().await {
                                Ok(_) => println!("[HTTP Server] ✅ Server started - ZynkSync and ZynkLink ready"),
                                Err(e) => eprintln!("[HTTP Server] ⚠️ Failed to start: {}", e),
                            }
                        });
                    }
                    Err(e) => {
                        eprintln!("[HTTP Server] ❌ Failed to create Tokio runtime: {}", e);
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Model and API key commands
            commands::models::get_models,
            commands::models::open_models_folder,
            commands::models::get_api_keys,
            commands::models::set_api_key,
            commands::models::remove_api_key,
            send_message_with_memory,
            run_ensemble,
            commands::memory::list_memories,
            commands::memory::update_memory,
            commands::memory::delete_memory,
            commands::memory::get_memory,
            // Onboarding
            commands::onboarding::store_onboarding_response,
            commands::onboarding::complete_onboarding,
            commands::onboarding::seed_system_memories,
            commands::onboarding::index_system_documentation,
            // Memory links/relationships (Pure Rust)
            commands::memory::get_memory_links,
            commands::memory::get_memory_graph,
            commands::memory::create_memory_link,
            commands::memory::delete_memory_link,
            commands::memory::update_memory_link,
            commands::memory::get_memory_contradictions,
            commands::memory::get_full_memory_graph,
            commands::memory::get_namespaces,
            // NEW: Einstein demo & contradiction detection (Pure Rust)
            commands::onboarding::apply_einstein_seed,
            commands::onboarding::load_small_einstein_demo,
            pre_check_memory,
            resolve_conflict,
            resolve_memory_conflict_v2,  // NEW: Refactored contradiction resolution
            // Safety and containment commands
            commands::safety::check_containment,
            commands::safety::set_containment_mode,
            commands::safety::get_containment_mode,
            commands::safety::initialize_safety_models,
            // Whisper: Local speech-to-text
            transcribe_audio,
            // NEW: Rust-based NLP commands (Phases 1-3)
            commands::safety::check_question_worthiness,
            commands::nlp::extract_entities,
            commands::nlp::extract_facts_from_question,
            // NEW: Rust-based conversation engine commands (Phase 5)
            commands::nlp::check_memory_worthiness,
            commands::conversation::build_conversation_prompt,
            // NEW: Ephemeral memory cleanup (Phase 6 - HIPAA)
            commands::memory::cleanup_expired_memories,
            // NEW: ZChat commands (Device-to-Device Messaging)
            zchat_send_message,
            zchat_get_messages,
            zchat_deliver_messages,
            zchat_mark_delivered,
            zchat_mark_read,
            zchat_get_unread_count,
            zchat_mark_all_read_from_device,
            zchat_clear_history,
            zchat_get_undelivered_messages,
            // NEW: ZynkSync commands (Phase 9)
            start_zynksync,
            stop_zynksync,
            get_zynksync_status,
            get_zynksync_peers,
            sync_to_peer,
            receive_sync_memories,
            request_device_pairing,
            verify_pairing_code,
            unpair_device,
            add_zynksync_device,
            remove_zynksync_device,
            get_zynksync_pairing_code,
            check_sync_status_with_peers,
            broadcast_sync_to_all_peers,
            get_local_ip,
            // Memory management
            clear_all_memories,
            clear_conversation_history,
            read_text_file,
            // User identity
            get_user_identity,
            set_user_identity,
            reset_user_identity,
            migrate_user_memories,
            // Sync codes
            generate_sync_code,
            verify_sync_code_info,
            sync_with_code,
            // ZynkLink - File Sharing
            generate_zynklink_code,
            link_with_zynklink_code,
            list_zynklink_pairings,
            revoke_zynklink_pairing,
            toggle_zynklink_pause,
            share_directory,
            unshare_directory,
            list_my_shared_directories,
            list_remote_directories,
            scan_shared_directory,
            list_shared_files,
            get_shared_file_path,
            download_to_knowledge_base,
            download_to_custom_location,
            cancel_zynklink_download,
            // External file/folder opening
            open_external_file,
            open_external_folder,
            // Knowledge Base (Phase 1)
            scan_knowledge_base,
            search_knowledge_base,
            read_knowledge_base_file,
            // Knowledge Base (Phase 2 - RAG)
            get_kb_folder_path,
            open_kb_folder_in_explorer,
            index_kb_document,
            list_kb_documents,
            remove_kb_document,
            clear_all_kb_documents,
            search_kb,
            // Snap-in commands
            index_snapin_notes,
            // Web search
            search_web,
            execute_web_search,
            // Conversation History
            commands::conversation::list_conversation_sessions,
            commands::conversation::get_conversation_messages,
            commands::conversation::search_conversations,
            commands::conversation::delete_conversation_session,
            // Feedback / training data collection
            commands::conversation::store_message_feedback,
        ])
        .on_window_event(|_window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                println!("[Cleanup] Window closing - sending goodbye to peers...");

                // Send goodbye to all paired peers so they mark us offline immediately
                std::thread::spawn(|| {
                    let rt = match tokio::runtime::Runtime::new() {
                    Ok(rt) => rt,
                    Err(e) => { eprintln!("[Cleanup] Failed to create runtime: {}", e); return; }
                };
                    rt.block_on(async {
                        let guard = ZYNKSYNC_SERVICE.lock().await;
                        if let Some(service) = guard.as_ref() {
                            service.send_goodbye_to_peers().await;
                        }
                    });
                }).join().ok();

                println!("[Cleanup] Window closing - cleaning up child processes...");

                // Kill React dev server and npm processes (Linux/macOS)
                #[cfg(target_os = "linux")]
                {
                    // Kill React dev server
                    let _ = std::process::Command::new("pkill")
                        .args(&["-9", "-f", "react-scripts start"])
                        .output();

                    // Kill npm tauri processes
                    let _ = std::process::Command::new("pkill")
                        .args(&["-9", "-f", "npm run tauri"])
                        .output();

                    // Kill anything on port 3000
                    if let Ok(output) = std::process::Command::new("lsof")
                        .args(&["-ti", ":3000"])
                        .output()
                    {
                        if let Ok(pids) = String::from_utf8(output.stdout) {
                            for pid in pids.lines() {
                                let _ = std::process::Command::new("kill")
                                    .args(&["-9", pid])
                                    .output();
                            }
                        }
                    }

                    println!("[Cleanup] ✓ Cleaned up dev server and ports");
                }

                // Kill processes on Windows
                #[cfg(target_os = "windows")]
                {
                    // Kill node processes
                    let _ = std::process::Command::new("taskkill")
                        .args(["/F", "/IM", "node.exe"])
                        .output();

                    // Kill anything on port 3000
                    let _ = std::process::Command::new("cmd")
                        .args(["/C", "for /f \"tokens=5\" %a in ('netstat -aon ^| find \":3000\" ^| find \"LISTENING\"') do taskkill /F /PID %a"])
                        .output();

                    println!("[Cleanup] ✓ Cleaned up dev server and ports");
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

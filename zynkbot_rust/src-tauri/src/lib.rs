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
// use chrono::Utc;  // Unused - commented out

/// Normalize common voice-transcription misspellings of the brand name.
/// Whisper and other STT engines frequently transcribe "Zynkbot" as Zincbot,
/// Zinkbot, Sinkbot, Zyncbot, or "Zinc bot". Fix these before any processing.
pub(crate) fn normalize_brand_names(text: String) -> String {
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

pub fn blob_to_f32(blob: &[u8]) -> Vec<f32> {
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
pub struct ReplyResponse {
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
pub struct ConversationTurn {
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

pub(crate) fn generate_title_from_content(content: &str) -> String {
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
pub(crate) async fn ask_llm_for_relationships(
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
            #[cfg(debug_assertions)]
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
        "grok-4.3",
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
pub struct RelationshipClassification {
    #[serde(deserialize_with = "deserialize_memory_id")]
    memory_id: i32,
    relationship_type: String,  // "contradicts", "supports", "elaborates", "caused_by", "reminds_of", "none"
    reason: String,
    confidence: Option<f32>,  // 0.0-1.0 confidence score
}

/// Enhanced memory decision that also classifies relationships with similar memories
/// Returns (should_remember, optional_title, relationship_classifications)
pub(crate) async fn ask_llm_about_memory_with_relationships(
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
            #[cfg(debug_assertions)]
            println!("[Memory Decision] Attempted to parse: {}", json_str);
            #[cfg(debug_assertions)]
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
    .bind(None::<String>)
    .bind(None::<chrono::DateTime<chrono::Utc>>)
    .fetch_one(&pool)
    .await
    .map_err(|e| format!("Failed to insert memory: {}", e))?;

    pool.close().await;

    println!("[Rust] Successfully inserted memory with ID: {}", memory_id);

    Ok(memory_id)
}

pub(crate) fn is_query_about_zynkbot(query: &str) -> bool {
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
        let _ = commands::zynksync::stop_zynksync().await; // Stop sync temporarily
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
        let _ = commands::zynksync::start_zynksync(None).await; // Restart sync with default 60s interval
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

// Helper struct for pending memory data
#[derive(serde::Deserialize, Clone)]
pub struct PendingMemory {
    content: String,
    title: Option<String>,
    embedding: Vec<f32>,
}

/// Helper: Store pending memory with NLP enhancement
pub async fn store_pending_memory(
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
// ============================================================================
// WEB SEARCH COMMANDS
// ============================================================================



// get_memory_contradictions → commands/memory.rs
// get_namespaces → commands/memory.rs
// update_memory_link → commands/memory.rs
// get_full_memory_graph → commands/memory.rs
// ============================================================================
// ZYNKSYNC COMMANDS (Phase 9: Device-to-Device Sync)
// ============================================================================

use std::sync::Arc;
use tokio::sync::Mutex;
use zynksync::ZynkSyncService;

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

/// Save sync state to file
pub async fn save_sync_state(enabled: bool) -> Result<(), String> {
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
            // Ensure model directories exist in user data dir (for installed binary)
            let models_dir = crate::db::get_app_data_dir();
            std::fs::create_dir_all(models_dir.join("models/system")).ok();
            std::fs::create_dir_all(models_dir.join("models/user")).ok();

            // Load .env — check user data dir first (installed binary), then dev paths
            let data_dir_env = crate::db::get_app_data_dir().join(".env");
            if data_dir_env.exists() {
                println!("[Dotenv] Loading .env from: {}", data_dir_env.display());
                dotenv::from_path(&data_dir_env).ok();
            } else {
                // Dev mode fallbacks
                let env_paths = [
                    "../../.env",
                    "../../../.env",
                    ".env",
                    "../../../../../../.env",
                ];
                for path in &env_paths {
                    if std::path::Path::new(path).exists() {
                        println!("[Dotenv] Loading .env from: {}", path);
                        dotenv::from_path(path).ok();
                        break;
                    }
                }
                // Last resort: search upward
                let mut current_dir = std::env::current_dir().unwrap_or_default();
                for _ in 0..5 {
                    let env_file = current_dir.join(".env");
                    if env_file.exists() {
                        println!("[Dotenv] Found .env at: {:?}", env_file);
                        dotenv::from_path(env_file).ok();
                        break;
                    }
                    if !current_dir.pop() { break; }
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
            commands::chat::send_message_with_memory,
            commands::chat::run_ensemble,
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
            commands::memory::pre_check_memory,
            commands::memory::resolve_conflict,
            commands::memory::resolve_memory_conflict_v2,
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
            // ZChat commands (Device-to-Device Messaging)
            commands::zchat::zchat_send_message,
            commands::zchat::zchat_get_messages,
            commands::zchat::zchat_deliver_messages,
            commands::zchat::zchat_mark_delivered,
            commands::zchat::zchat_mark_read,
            commands::zchat::zchat_get_unread_count,
            commands::zchat::zchat_mark_all_read_from_device,
            commands::zchat::zchat_clear_history,
            commands::zchat::zchat_get_undelivered_messages,
            // ZynkSync commands
            commands::zynksync::start_zynksync,
            commands::zynksync::stop_zynksync,
            commands::zynksync::get_zynksync_status,
            commands::zynksync::get_zynksync_peers,
            commands::zynksync::sync_to_peer,
            commands::zynksync::receive_sync_memories,
            commands::zynksync::request_device_pairing,
            commands::zynksync::verify_pairing_code,
            commands::zynksync::unpair_device,
            commands::zynksync::add_zynksync_device,
            commands::zynksync::remove_zynksync_device,
            commands::zynksync::get_zynksync_pairing_code,
            commands::zynksync::check_sync_status_with_peers,
            commands::zynksync::broadcast_sync_to_all_peers,
            commands::zynksync::get_local_ip,
            // Memory management
            commands::zynksync::clear_all_memories,
            commands::conversation::clear_conversation_history,
            commands::utils::read_text_file,
            commands::utils::read_file_base64,
            // User identity
            commands::user_identity::get_user_identity,
            commands::user_identity::set_user_identity,
            commands::user_identity::reset_user_identity,
            commands::user_identity::migrate_user_memories,
            // Sync codes
            commands::sync_codes::generate_sync_code,
            commands::sync_codes::verify_sync_code_info,
            commands::sync_codes::sync_with_code,
            // ZynkLink - File Sharing
            commands::zynklink::generate_zynklink_code,
            commands::zynklink::link_with_zynklink_code,
            commands::zynklink::list_zynklink_pairings,
            commands::zynklink::revoke_zynklink_pairing,
            commands::zynklink::toggle_zynklink_pause,
            commands::zynklink::share_directory,
            commands::zynklink::unshare_directory,
            commands::zynklink::list_my_shared_directories,
            commands::zynklink::list_remote_directories,
            commands::zynklink::scan_shared_directory,
            commands::zynklink::list_shared_files,
            commands::zynklink::get_shared_file_path,
            commands::zynklink::download_to_knowledge_base,
            commands::zynklink::download_to_custom_location,
            commands::zynklink::cancel_zynklink_download,
            // External file/folder opening + Knowledge Base
            commands::knowledge_base::open_external_file,
            commands::knowledge_base::open_external_folder,
            commands::knowledge_base::scan_knowledge_base,
            commands::knowledge_base::search_knowledge_base,
            commands::knowledge_base::read_knowledge_base_file,
            commands::knowledge_base::get_kb_folder_path,
            commands::knowledge_base::open_kb_folder_in_explorer,
            commands::knowledge_base::index_kb_document,
            commands::knowledge_base::list_kb_documents,
            commands::knowledge_base::remove_kb_document,
            commands::knowledge_base::clear_all_kb_documents,
            commands::knowledge_base::search_kb,
            commands::knowledge_base::index_snapin_notes,
            // Web search
            commands::web_search::search_web,
            commands::web_search::execute_web_search,
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

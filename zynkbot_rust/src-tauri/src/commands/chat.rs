use crate::ConversationTurn;
use crate::ReplyResponse;
use crate::Memory;

/// Tauri command: cancel the in-flight LLM generation.
/// Sets the global GENERATION_CANCELLED flag; the streaming loops in llm/openai.rs
/// and llm/anthropic.rs check this flag at the top of each iteration and break out.
#[tauri::command]
pub fn cancel_generation() {
    crate::GENERATION_CANCELLED.store(true, std::sync::atomic::Ordering::SeqCst);
    println!("[Rust] 🛑 Generation cancellation requested");
}

/// Returns the ZynkSync mTLS-capable HTTP client if `base_url` points to a paired peer,
/// otherwise returns a plain client that accepts self-signed certificates (for LAN Ollama).
async fn build_custom_client(base_url: &str) -> reqwest::Client {
    let guard = crate::ZYNKSYNC_SERVICE.lock().await;
    if let Some(service) = guard.as_ref() {
        if let Some(client) = service.get_peer_client_for_url(base_url).await {
            println!("[mTLS] Using ZynkSync mTLS client for custom endpoint: {}", base_url);
            return client;
        }
    }
    crate::llm::openai::default_client(true)
}

/// Can the "local" backend (or this .gguf path) actually answer on this device?
/// On Android too since 2026-10-22 (the engine builds for the phone; a model file in
/// the phone's models folder is all it takes).
fn local_backend_usable(backend: &str) -> bool {
    if backend.ends_with(".gguf") {
        return std::path::Path::new(backend).is_file();
    }
    crate::llm::local_models::resolve_default_model_path().is_ok()
}

/// Order: custom endpoint, a downloaded local model, then any API provider with a key
/// (same provider order as the hands-free path). None only when nothing can answer.
fn get_best_available_backend() -> Option<String> {
    let custom_url = std::env::var("CUSTOM_API_URL").unwrap_or_default();
    let custom_model = std::env::var("CUSTOM_MODEL").unwrap_or_default();
    if !custom_url.is_empty() && !custom_model.is_empty() {
        return Some("custom".to_string());
    }
    if !cfg!(target_os = "android") {
        let models_dir = crate::db::get_models_dir().join("user");
        if let Ok(entries) = std::fs::read_dir(&models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file()
                    && path.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false)
                {
                    return Some(path.to_string_lossy().to_string());
                }
            }
        }
    }
    for (var, name) in [
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("OPENAI_API_KEY", "openai"),
        ("XAI_API_KEY", "xai"),
        ("MISTRAL_API_KEY", "mistral"),
    ] {
        if !std::env::var(var).unwrap_or_default().is_empty() {
            return Some(name.to_string());
        }
    }
    None
}

#[cfg(test)]
mod backend_fallback_tests {
    use super::local_backend_usable;

    #[test]
    fn a_gguf_path_that_does_not_exist_is_not_usable() {
        assert!(!local_backend_usable("/definitely/not/here/model.gguf"));
    }

    #[cfg(not(target_os = "android"))]
    #[test]
    fn a_gguf_file_that_exists_is_usable_on_desktop() {
        let dir = std::env::temp_dir().join(format!("zb_gguf_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("tiny.gguf");
        std::fs::write(&file, b"not a real model").unwrap();
        assert!(local_backend_usable(file.to_str().unwrap()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn format_api_error(backend: &str, raw: &str) -> String {
    let provider = if backend.contains("anthropic") || backend.contains("claude") { "Anthropic" }
        else if backend.contains("openai") || backend.contains("gpt") { "OpenAI" }
        else if backend.contains("xai") || backend.contains("grok") { "xAI" }
        else if backend.contains("mistral") { "Mistral" }
        else { "the provider" };

    if raw.contains("401") || raw.contains("nauthorized") || raw.contains("invalid_api_key") || raw.contains("invalid api key") {
        format!("❌ {} rejected the API key. Check your key in Settings → API Keys.", provider)
    } else if raw.contains("403") || raw.contains("model_not_found") || raw.contains("not have access") || raw.contains("does not have") {
        format!("❌ Your {} account doesn't have access to this model. Try selecting a different model in Settings → API Keys.", provider)
    } else if raw.contains("429") || raw.contains("rate_limit") || raw.contains("quota") {
        format!("❌ {} rate limit reached. Wait a moment and try again.", provider)
    } else if raw.contains("529") || raw.contains("overloaded") {
        format!("❌ {} is overloaded right now. Wait a moment and try again.", provider)
    } else if raw.contains("empty reply") {
        format!("❌ {} returned an empty reply. Try again.", provider)
    } else if raw.contains("connect") || raw.contains("timeout") || raw.contains("dns") || raw.contains("network") {
        format!("❌ Can't reach {}. Check your internet connection.", provider)
    } else {
        format!("❌ {} error: {}", provider, raw)
    }
}

/// Tauri command: builds a WebView-emitting sink and delegates to the
/// runtime-agnostic core below. Keep this thin — all logic lives in `generate_reply`.
#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn send_message_with_memory(
    app: tauri::AppHandle,
    message: String,
    user_id: String,
    session_id: String,
    backend: String,
    containment_mode: String,
    conversation_history: Option<Vec<ConversationTurn>>,
    skip_containment: Option<bool>,
    skip_memory_storage: Option<bool>,
    kb_enabled: Option<bool>,
    user_query: Option<String>,
    image_data: Option<Vec<crate::llm::ImageAttachment>>,
) -> Result<ReplyResponse, String> {
    let sink: std::sync::Arc<dyn crate::response_sink::ResponseSink> =
        std::sync::Arc::new(crate::response_sink::TauriSink::new(app));
    generate_reply(
        sink, message, user_id, session_id, backend, containment_mode,
        conversation_history, skip_containment, skip_memory_storage,
        kb_enabled, user_query, image_data,
        false, // hands_free: this is the in-app path
    )
    .await
}

/// Runtime-agnostic chat core: produces a reply, streaming tokens and side-channel
/// events to `sink`, with no dependency on Tauri's `AppHandle`. The Tauri command
/// above passes a `TauriSink`; the planned native Android assistant path passes a
/// JNI-backed sink so this can run and speak a reply with no WebView present.
/// No Flask dependency — handles everything in Rust.
#[allow(clippy::too_many_arguments)]
pub async fn generate_reply(
    sink: std::sync::Arc<dyn crate::response_sink::ResponseSink>,
    message: String,
    user_id: String,
    session_id: String,
    backend: String,
    containment_mode: String,
    conversation_history: Option<Vec<ConversationTurn>>,
    skip_containment: Option<bool>,
    skip_memory_storage: Option<bool>,
    kb_enabled: Option<bool>,
    user_query: Option<String>,
    image_data: Option<Vec<crate::llm::ImageAttachment>>,
    hands_free: bool,
) -> Result<ReplyResponse, String> {
    use crate::conversation_engine::ConversationEngine;

    // Reset cancellation flag at the start of every new request so stale
    // cancellations from a previous stopped generation don't leak into this one.
    crate::GENERATION_CANCELLED.store(false, std::sync::atomic::Ordering::SeqCst);

    // Normalize brand name misspellings from voice transcription before any processing
    let message = crate::normalize_brand_names(message);

    let request_start = std::time::Instant::now();
    println!("[⏱️ PERF] Request start");
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  💬 NEW CHAT REQUEST                                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    {
        // A picked photo once arrived as megabytes of text and went into the log whole
        // (2026-10-25): show the first 1,500 characters and the size.
        let n = message.chars().count();
        if n > 1500 { println!("▶ {} … ({} characters)", message.chars().take(1500).collect::<String>(), n); } else { println!("▶ {}", message); }
    }
    println!();
    println!("[RUST] Backend: {}", backend);

    // Child mode: Force OpenAI backend for all responses
    let mut forced_backend = backend.clone();
    if containment_mode.to_lowercase() == "child" {
        println!("[RUST] 🧒 Child mode detected - forcing OpenAI backend");
        forced_backend = "openai".to_string();
    }

    // If the selected backend has no key, fall back to custom endpoint or local model
    if containment_mode.to_lowercase() != "child" {
        let has_key = match forced_backend.as_str() {
            b if b.contains("anthropic") || b.contains("claude") => {
                !std::env::var("ANTHROPIC_API_KEY").unwrap_or_default().is_empty()
            }
            b if b.contains("openai") || b.contains("gpt") => {
                !std::env::var("OPENAI_API_KEY").unwrap_or_default().is_empty()
            }
            b if b.contains("xai") || b.contains("grok") => {
                !std::env::var("XAI_API_KEY").unwrap_or_default().is_empty()
            }
            b if b.contains("mistral") => {
                !std::env::var("MISTRAL_API_KEY").unwrap_or_default().is_empty()
            }
            // "local" is what a fresh install stores before anything is chosen. A phone
            // has no local models at all, and a desktop may have nothing downloaded, so
            // treat a local backend that cannot load like a provider with no key and fall
            // through to whatever can answer, as the hands-free path already does
            // (android_jni::resolve_voice_backend). A lone phone with a typed-in OpenAI
            // key answered "Hey Zynk" but showed "Local models not supported on Android"
            // for every typed message (tester, Pixel 7a, 2026-09-22).
            b if b.contains("local") || b.ends_with(".gguf") => local_backend_usable(b),
            _ => true, // custom — assume present until actually tried
        };

        if !has_key {
            println!("[RUST] ⚠️ No API key for '{}' — checking fallbacks", forced_backend);
            if let Some(fallback) = get_best_available_backend() {
                println!("[RUST] 🔄 Falling back to: {}", fallback);
                forced_backend = fallback;
            } else {
                return Err(
                    "No API key is configured for this provider, and no local models or custom endpoint are available. Add an API key in Settings → API Keys.".to_string()
                );
            }
        }
    }

    // When a file is attached, `message` contains the full dump (file content + question).
    // `user_query` carries only the clean question — use it for safety checks, memory search,
    // KB search, and conversation history. Falls back to `message` when no file is attached.
    let query = user_query.unwrap_or_else(|| message.clone());

    // Put the thread in History now, not after the reply: a thread whose first
    // answer was still streaming was invisible there (2026-09-14). HIPAA keeps no
    // history at all. Hands-free turns wait for the reply as before: the model may
    // judge the transcript a TV line (NO_QUERY), and such a turn must never open or
    // name a thread. A failure here must not cost the user the reply.
    if containment_mode.to_lowercase() != "hipaa" && !skip_containment.unwrap_or(false) && !hands_free {
        match sqlx::SqlitePool::connect(&crate::db::get_db_url()).await {
            Ok(pool) => {
                if let Err(e) = crate::conversation_history::open_session(
                    &pool, &session_id, &user_id, &query, &forced_backend, &containment_mode,
                ).await {
                    eprintln!("[ConvHistory] ⚠️ Could not open thread {}: {}", &session_id.chars().take(8).collect::<String>(), e);
                }
            }
            Err(e) => eprintln!("[ConvHistory] ⚠️ DB pool error: {}", e),
        }
    }

    // Weather (2026-10-25, Matt): a weather question is answered from Open-Meteo here,
    // before containment and before any model call. If the lookup fails (offline,
    // unknown place) the question goes on to the model as usual.
    if image_data.is_none() && explicit_remember(&query).is_none() && !skip_containment.unwrap_or(false) {
        if let Some(ask) = crate::weather::detect(&query) {
            match crate::weather::answer(&ask, &user_id).await {
                Ok(text) => {
                    println!("[Weather] answered from Open-Meteo ({:?}, place {:?})", ask.when, ask.place);
                    sink.token(&text);
                    if containment_mode.to_lowercase() != "hipaa" {
                        if let Ok(pool) = sqlx::SqlitePool::connect(&crate::db::get_db_url()).await {
                            if let Err(e) = crate::conversation_history::log_exchange(&pool, &session_id, &user_id, &query, &text, "weather", &containment_mode, !hands_free, if hands_free { "hands_free" } else { "typed" }).await {
                                eprintln!("[ConvHistory] ⚠️ Failed to log weather exchange: {}", e);
                            }
                        }
                    }
                    return Ok(ReplyResponse {
                        reply_text: text,
                        recalled_memories: None,
                        model_backend: Some("weather".to_string()),
                        containment_mode: Some(containment_mode),
                        schema: None,
                        blocked: None,
                        web_search_needed: None,
                        web_search_query: None,
                        original_query: None,
                        pre_search_reply: None,
                        kb_note: None,
                        web_search_results: None,
                    });
                }
                Err(e) => println!("[Weather] {} — asking the model instead", e),
            }
        }
    }

    // STEP 1: Safety check via containment layer (skip for internal operations like web search synthesis)
    let mut warning_prefix: Option<String> = None;

    if !skip_containment.unwrap_or(false) {
        let step_start = std::time::Instant::now();

        // Child mode: Use OpenAI Moderation API for safety check
        if containment_mode.to_lowercase() == "child" {
            println!("[RUST] 🧒 Child mode - checking with OpenAI Moderation API...");
            let layer = crate::containment::ContainmentLayer::new(&containment_mode)?;

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
                        pre_search_reply: None,
                        kb_note: None,
                        web_search_results: None,
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
                        pre_search_reply: None,
                        kb_note: None,
                        web_search_results: None,
                    });
                }
            }
            println!("[⏱️ PERF] OpenAI moderation check: {:.3}s", step_start.elapsed().as_secs_f32());
        } else {
            // Non-child modes: Use standard containment check
            let safety_check = crate::commands::safety::check_containment(query.clone(), containment_mode.clone()).await;
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
                            pre_search_reply: None,
                        kb_note: None,
                        web_search_results: None,
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
        let enhancer = crate::nlp_enhancer::NLPEnhancer::new();
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
        crate::llm::local_embeddings::generate_local_embedding(&query_text_for_embedding)
    })
    .await
    .map_err(|e| format!("Failed to run embedding task: {}", e))?
    .map_err(|e| format!("Failed to generate embedding: {}", e))?;

    // HYBRID SEARCH: Use hybrid_search for weighted entity + semantic scoring
    // Search limit matches build_memory_context caps: API gets more headroom, local stays lean
    // The web-search answer pass carries the fetched pages; for a local model it does not
    // also carry recalled memories (18 were recalled for "the weather in Morocco" and read
    // at 20 tokens/s on a warm phone, 2026-10-22).
    let search_pass = message.starts_with("Answer the user's question using the web search results");
    let memory_search_limit = if search_pass && !ConversationEngine::is_api_model(&forced_backend) { 0 } else { if ConversationEngine::is_api_model(&forced_backend) { 15 } else { 10 } };

    // Smart namespace filtering: search ONLY system memories if query is about Zynkbot
    let is_zynkbot_query = crate::is_query_about_zynkbot(&query);
    let (search_user_id, namespace_filter) = if is_zynkbot_query {
        ("system", Some("_zynkbot"))  // Search system user's memories in _zynkbot namespace
    } else {
        (user_id.as_str(), None)  // Search user's memories, filter system memories manually below
    };

    let recalled_memories = crate::memory::hybrid_search(
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
    let recalled_memories: Vec<crate::memory::Memory> = if is_zynkbot_query {
        recalled_memories  // Keep all (should only be system memories anyway)
    } else {
        recalled_memories.into_iter()
            .filter(|m| m.namespace != "_zynkbot")
            .collect()
    };

    // ONE-HOP GRAPH TRAVERSAL: For each recalled memory, pull in all directly
    // linked memories regardless of relation type (related, contradicts, updates,
    // resolves), skipping memories a newer one has replaced. Links are sorted by confidence so the
    // strongest associations are included first. Capped at 8 hop-memories total to
    // bound prompt size; build_memory_context enforces the final per-model limit.
    const MAX_HOP_MEMORIES: usize = 8;
    let mut linked_memories: Vec<crate::memory::Memory> = Vec::new();
    if !is_zynkbot_query {
        let already_included_ids: std::collections::HashSet<i32> = recalled_memories.iter()
            .map(|m| m.id)
            .collect();
        let mut already_included_content: std::collections::HashSet<String> = recalled_memories.iter()
            .map(|m| m.content.clone())
            .collect();

        'outer: for mem in &recalled_memories {
            if let Ok(mut links) = crate::memory::get_memory_links(&db_pool, mem.id).await {
                // Strongest links first so the cap preserves the most meaningful ones
                links.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
                for link in links {
                    if linked_memories.len() >= MAX_HOP_MEMORIES {
                        break 'outer;
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
                    if let Ok(Some(linked_mem)) = crate::memory::get_memory(&db_pool, linked_id).await {
                        if linked_mem.temporal_status.as_deref() == Some("superseded") {
                            continue;
                        }
                        if already_included_content.contains(&linked_mem.content) {
                            continue;
                        }
                        println!("[RUST] 🔗 Graph traversal: memory #{} → linked #{} ({:?}) via '{}' (conf {:.2})",
                            mem.id, linked_id, linked_mem.title, link.relation_type, link.confidence);
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
    let kb_enabled = kb_enabled.unwrap_or(false);
    let mut kb_context = String::new();
    // What the search actually found. Drives the prompt wording (kb_prompt.rs),
    // the note shown in the chat, and whether memory extraction runs: an answer
    // given when the KB had no real match is not allowed to become a stored fact.
    let mut kb_grounded = true;
    let mut kb_note: Option<String> = None;

    if kb_enabled {
        let kb_start = std::time::Instant::now();

        // EXPLICIT KB SEARCH (user clicked KB button)
        // - 10 chunks (comprehensive coverage)
        // - 15% threshold (cast wide net - user knows what they're looking for)
        // - below the threshold, the 5 best chunks are still shown but labelled weak
        // On-device model (2026-10-22): four chunks, not ten. Ten chunks were 5,649
        // characters, and with the rest of the prompt that was 4,212 tokens in a 4K
        // window: "Prompt too long" instead of an answer.
        let on_device = forced_backend.ends_with(".gguf") || forced_backend.to_lowercase().contains("local");
        let kb_chunk_limit = if on_device { 4 } else { 10 };
        let kb_similarity_threshold = 0.15;
        let kb_weak_limit = if on_device { 2 } else { 5 };

        println!(
            "[KB RAG] 🔍 EXPLICIT KB SEARCH ({} chunks, {:.0}% threshold)",
            kb_chunk_limit,
            kb_similarity_threshold * 100.0
        );

        // Explicit KB search: exclude system docs - user wants THEIR documents only
        match crate::kb_rag::search_kb_chunks(&db_pool, &user_id, &query, kb_chunk_limit, false).await {
            Ok(kb_results) => {
                let built = crate::kb_prompt::build(&kb_results, kb_similarity_threshold, kb_weak_limit);
                println!(
                    "[KB RAG] outcome {:?}: {} chunks returned (best: {:.1}%)",
                    built.outcome,
                    kb_results.len(),
                    kb_results.first().map(|r| r.similarity_score * 100.0).unwrap_or(0.0)
                );
                kb_grounded = built.grounded();
                kb_note = built.note;
                kb_context = built.context;
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
    let user_display_name = crate::memory::get_user_display_name(&db_pool, &user_id).await;

    // Close database connection
    db_pool.close().await;

    // STEP 5: Build prompt using conversation engine
    let engine = ConversationEngine::new()
        .with_on_device(forced_backend.ends_with(".gguf") || forced_backend.to_lowercase().contains("local"));

    // Convert conversation history to conversation engine format
    let engine_history: Option<Vec<crate::conversation_engine::ConversationTurn>> = conversation_history.as_ref().map(|hist| {
        hist.iter().map(|turn| {
            crate::conversation_engine::ConversationTurn {
                role: turn.role.clone(),
                content: turn.content.clone(),
            }
        }).collect()
    });

    // Convert recalled memories to conversation engine format
    // Include semantic results AND one-hop graph-linked memories only.
    // entity_matched_memories are for contradiction/duplicate detection in the background
    // task and should NOT be injected into the prompt — they failed the hybrid search threshold.
    let engine_memories: Vec<crate::conversation_engine::Memory> = recalled_memories
        .iter()
        .chain(linked_memories.iter())
        .map(|mem| crate::conversation_engine::Memory {
            id: mem.id,
            content: mem.content.clone(),
            original_text: mem.original_text.clone(),
            title: mem.title.clone(),  // Already Option<String>, no need to wrap in Some()
            similarity: mem.similarity,
            created_at: mem.created_at,
            event_date: mem.event_date,
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

    // A question about Zynkbot itself: consult the built-in user guides, indexed under the
    // `_system` knowledge-base user, the way the KB button consults the user's own
    // documents (Matt, 2026-10-26). Not when the KB button is on, which asks for the
    // user's documents. Fewer passages on the phone, where the prompt is tight.
    if is_zynkbot_query && !kb_enabled {
        let guide_limit = if is_api_model { 5 } else { 3 };
        match crate::kb_rag::search_kb_chunks(&db_pool, "_system", &query, guide_limit, false).await {
            Ok(hits) => match crate::kb_prompt::build_guides(&hits, 0.30, guide_limit) {
                Some(block) => {
                    println!("[System KB] {} guide passages added for a Zynkbot question (best {:.0}%)",
                        hits.iter().filter(|h| h.similarity_score > 0.30).count().min(guide_limit),
                        hits.first().map(|h| h.similarity_score * 100.0).unwrap_or(0.0));
                    full_prompt = format!("{}{}", block, full_prompt);
                }
                None => println!("[System KB] no guide passage matched the Zynkbot question (best {:.0}%)",
                    hits.first().map(|h| h.similarity_score * 100.0).unwrap_or(0.0)),
            },
            Err(e) => eprintln!("[System KB] guide search failed (non-fatal): {}", e),
        }
    }

    // STEP 6: Call LLM based on backend (use forced_backend for Child mode)
    // All API backends use SSE streaming so the frontend can display tokens as they arrive.
    // Local GGUF models are blocking and cannot stream, so they continue to return all at once.
    let _api_start = std::time::Instant::now();


    let reply_text = if forced_backend.to_lowercase().contains("anthropic") || forced_backend.to_lowercase().contains("claude") {
        // Use Anthropic with streaming
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY not set".to_string())?;

        let model_name_owned = std::env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-5".to_string());
        let model_name = model_name_owned.as_str();

        let sink_c = sink.clone();

        if let Some(ref imgs) = image_data {
            println!("[⏱️ PERF] Calling Anthropic API ({}) with vision streaming...", model_name);
            let response = crate::llm::anthropic::send_vision_streaming(
                &api_key,
                model_name,
                &full_prompt,
                imgs,
                None,
                Some(4096),
                move |token| { sink_c.token(&token); },
            ).await.map_err(|e| e.to_string())?;
            response.content
        } else {
            println!("[⏱️ PERF] Calling Anthropic API ({}) with streaming...", model_name);
            let messages = vec![crate::llm::Message {
                role: "user".to_string(),
                content: full_prompt,
            }];
            let response = crate::llm::anthropic::send_message_streaming(
                &api_key,
                model_name,
                messages,
                None,
                Some(4096),
                None,
                move |token| { sink_c.token(&token); },
            ).await.map_err(|e| format_api_error(&forced_backend, &e.to_string()))?;
            response.content
        }

    } else if forced_backend.to_lowercase().contains("openai") || forced_backend.to_lowercase().contains("gpt") {
        // Use OpenAI with streaming (including Child mode)
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY not set".to_string())?;

        let openai_model_owned = std::env::var("OPENAI_MODEL")
            .unwrap_or_else(|_| "gpt-5.5".to_string());
        let model_name = openai_model_owned.as_str();

        let sink_c = sink.clone();

        if let Some(ref imgs) = image_data {
            println!("[⏱️ PERF] Calling OpenAI API ({}) with vision streaming...", model_name);
            let response = crate::llm::openai::send_vision_streaming(
                &api_key,
                model_name,
                &full_prompt,
                imgs,
                "https://api.openai.com/v1/chat/completions",
                move |token| { sink_c.token(&token); },
            ).await.map_err(|e| e.to_string())?;
            response.content
        } else {
            println!("[⏱️ PERF] Calling OpenAI API ({}) with streaming...", model_name);
            let mut messages = Vec::new();
            if containment_mode.to_lowercase() == "child" {
                messages.push(crate::llm::Message {
                    role: "system".to_string(),
                    content: crate::containment::CHILD_MODE_SYSTEM_PROMPT.to_string(),
                });
                println!("[RUST] 🧒 Child mode - injecting child safety system prompt");
            }
            messages.push(crate::llm::Message {
                role: "user".to_string(),
                content: full_prompt,
            });
            let response = crate::llm::openai::send_message_streaming(
                &api_key,
                model_name,
                messages,
                Some(4096),
                None,
                "https://api.openai.com/v1/chat/completions",
                crate::llm::openai::default_client(false),
                move |token| { sink_c.token(&token); },
            ).await.map_err(|e| format_api_error(&forced_backend, &e.to_string()))?;
            response.content
        }

    } else if forced_backend.to_lowercase().contains("xai") || forced_backend.to_lowercase().contains("grok") {
        // Use xAI (Grok) with streaming - OpenAI-compatible format
        let api_key = std::env::var("XAI_API_KEY")
            .map_err(|_| "XAI_API_KEY not set. Get your API key from https://console.x.ai/".to_string())?;

        let xai_model_owned = std::env::var("XAI_MODEL")
            .unwrap_or_else(|_| "grok-4.5".to_string());
        let xai_model = xai_model_owned.as_str();
        let sink_c = sink.clone();

        if let Some(ref imgs) = image_data {
            println!("[⏱️ PERF] Calling xAI API ({}) with vision streaming...", xai_model);
            let response = crate::llm::openai::send_vision_streaming(
                &api_key,
                xai_model,
                &full_prompt,
                imgs,
                "https://api.x.ai/v1/chat/completions",
                move |token| { sink_c.token(&token); },
            ).await.map_err(|e| e.to_string())?;
            response.content
        } else {
            println!("[⏱️ PERF] Calling xAI API ({}) with streaming...", xai_model);
            let messages = vec![crate::llm::Message {
                role: "user".to_string(),
                content: full_prompt,
            }];
            let response = crate::llm::openai::send_message_streaming(
                &api_key,
                xai_model,
                messages,
                Some(4096),
                None,
                "https://api.x.ai/v1/chat/completions",
                crate::llm::openai::default_client(false),
                move |token| { sink_c.token(&token); },
            ).await.map_err(|e| format_api_error(&forced_backend, &e.to_string()))?;
            response.content
        }

    } else if forced_backend.to_lowercase().contains("mistral") {
        let api_key = std::env::var("MISTRAL_API_KEY")
            .map_err(|_| "MISTRAL_API_KEY not set".to_string())?;
        let mistral_model_owned = std::env::var("MISTRAL_MODEL")
            .unwrap_or_else(|_| "mistral-large-latest".to_string());
        let mistral_model = mistral_model_owned.as_str();
        let sink_c = sink.clone();

        if let Some(ref imgs) = image_data {
            println!("[⏱️ PERF] Calling Mistral API ({}) with vision streaming...", mistral_model);
            let response = crate::llm::openai::send_vision_streaming(
                &api_key,
                mistral_model,
                &full_prompt,
                imgs,
                "https://api.mistral.ai/v1/chat/completions",
                move |token| { sink_c.token(&token); },
            ).await.map_err(|e| e.to_string())?;
            response.content
        } else {
            println!("[⏱️ PERF] Calling Mistral API ({}) with streaming...", mistral_model);
            let messages = vec![crate::llm::Message {
                role: "user".to_string(),
                content: full_prompt,
            }];
            let response = crate::llm::openai::send_message_streaming(
                &api_key,
                mistral_model,
                messages,
                Some(4096),
                None,
                "https://api.mistral.ai/v1/chat/completions",
                crate::llm::openai::default_client(false),
                move |token| { sink_c.token(&token); },
            ).await.map_err(|e| format_api_error(&forced_backend, &e.to_string()))?;
            response.content
        }

    } else if forced_backend.to_lowercase() == "custom" {
        // Custom / Ollama — OpenAI-compatible endpoint at user-supplied URL
        let base_url = std::env::var("CUSTOM_API_URL")
            .map_err(|_| "Custom endpoint not configured. Add it in API Settings.".to_string())?;
        let model_name = std::env::var("CUSTOM_MODEL")
            .map_err(|_| "No model selected for custom endpoint. Configure it in API Settings.".to_string())?;
        let api_key = std::env::var("CUSTOM_API_KEY").unwrap_or_default();
        let api_url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

        if image_data.is_some() {
            return Err("Image attachments are not supported with custom/Ollama endpoints.".to_string());
        }

        let sink_c = sink.clone();
        let messages = vec![crate::llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];

        println!("[⏱️ PERF] Calling custom endpoint ({}) model: {}", api_url, model_name);
        let custom_client = build_custom_client(&base_url).await;
        let response = crate::llm::openai::send_message_streaming(
            &api_key,
            &model_name,
            messages,
            Some(4096),
            None,
            &api_url,
            custom_client,
            move |token| { sink_c.token(&token); },
        ).await.map_err(|e| format!("Custom endpoint error: {} — is Ollama running?", e))?;
        response.content

    } else if forced_backend.to_lowercase().contains("local") || forced_backend.ends_with(".gguf") {
        if image_data.is_some() {
            return Err("Image attachments are not supported with local models. Please switch to a cloud model (Claude, GPT-4o, or Grok) to use vision.".to_string());
        }
        // Use local GGUF model

        // Determine model path
        let model_path = if forced_backend.ends_with(".gguf") {
            // Explicit path provided
            forced_backend.clone()
        } else {
            // LOCAL_MODEL_PATH override, else whichever model is actually installed
            crate::llm::local_models::resolve_default_model_path()
                .map_err(|e| e.to_string())?
        };

        let messages = vec![crate::llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];

        let model_path_clone = model_path.clone();
        let inference_start = std::time::Instant::now();
        // The user's turn comes first: background memory calls step aside while this
        // guard is held (local_models "foreground first", 2026-10-22).
        let foreground = crate::llm::local_models::ForegroundGuard::new();
        let sink_local = sink.clone();
        let response = tokio::task::spawn_blocking(move || {
            let _hold = foreground;
            crate::llm::local_models::with_cached_session(&model_path_clone, |session| {
                // Streams like the cloud paths do (2026-10-22): each piece of text goes to the
                // page as it is written.
                let mut emit = |t: &str| sink_local.token(t);
                session.generate_streaming(messages, Some(4096), None, None, Some(&mut emit))
            })
        })
        .await
        .map_err(|e| format!("Failed to run local model task: {}", e))?
        .map_err(|e| e.to_string())?;
        println!("[⏱️ PERF] Local model inference: {:.2}s", inference_start.elapsed().as_secs_f32());

        response.content

    } else {
        return Err(format!(
            "Unsupported backend: {}. Use 'anthropic', 'openai', 'xai' (grok), 'local', or provide a .gguf file path",
            forced_backend
        ));
    };

    // An empty reply is a failure, not an answer. On 2026-10-21 Anthropic closed four
    // streams with no text (an error event the client then ignored) and the page
    // showed an empty bubble each time. The page renders an Err as an error message.
    if reply_text.trim().is_empty() {
        println!("[RUST] ✗ {} returned an empty reply", forced_backend);
        return Err(format_api_error(&forced_backend, "empty reply"));
    }

    println!("[RUST] ✅ Pure Rust conversation complete");

    // Extract recalled memory IDs for later use
    let _recalled_memory_ids: Vec<i32> = recalled_memories.iter().map(|m| m.id).collect();

    // STEP 6.5: Check if LLM requested a web search. Tolerant of how small models write
    // the marker: Qwen3-4B wrote `Web search needed: "current weather in Lima, Peru"` on
    // 2026-10-22 and the search never ran.
    let web_search_marker = find_web_search_marker(&reply_text);
    let web_search_detected = web_search_marker.is_some();
    let web_search_query = if let Some((_, after_pos)) = web_search_marker {
        println!("[RUST] LLM detected need for web search");
        let after_marker = &reply_text[after_pos..];
        let query_end = after_marker.find('\n').unwrap_or(after_marker.len());
        let query = after_marker[..query_end].trim().trim_matches(|c: char| matches!(c, '"' | '\'' | '“' | '”' | '[' | ']' | '*')).trim().to_string();
        println!("[RUST] Suggested search query: {}", query);
        if query.is_empty() { None } else { Some(query) }
    } else {
        None
    };

    // Hands-free always searches, never asks first (Matt, 2026-09-20, GitHub #26):
    // "want me to search?" has no way to be answered by voice — there is no button off
    // screen, and a spoken "yes" just looked like a new, unrelated question, which the
    // model sometimes "answered" by inventing a search for the word "yes" itself. The
    // synthesis below is a second, memory-free turn; its reply is what the user hears.
    if hands_free && web_search_query.is_some() {
        let search_query = web_search_query.clone().unwrap_or_default();
        sink.event("web-search", serde_json::json!({ "query": search_query }));
        println!("[RUST] Hands-free auto search: {}", search_query);
        let mut context = format!("Here are the web search results for \"{}\":\n\n", search_query);
        // The sources go back with the reply so the chat can show the same "View n
        // search sources" drop-down a typed search gets; page bodies stay out of it.
        let mut web_sources: Option<serde_json::Value> = None;
        match crate::web_search::search_with_content(&search_query, 5, 3).await {
            Ok(results) if !results.results.is_empty() => {
                for (i, r) in results.results.iter().enumerate() {
                    context.push_str(&format!("Source {}: {}\nURL: {}\n", i + 1, r.title, r.url));
                    if !r.snippet.is_empty() { context.push_str(&format!("Summary: {}\n", r.snippet)); }
                    // A local model reads at ~40 tokens/s: 800 characters a page made the second
                    // pass a 1,850-token, 64 s read on 2026-10-22. Half that for local models.
                    let body_chars = if ConversationEngine::is_api_model(&forced_backend) { 800 } else { 300 };
                    if let Some(body) = r.content.as_deref() { context.push_str(&format!("Content: {}\n", body.chars().take(body_chars).collect::<String>())); }
                    context.push('\n');
                }
                web_sources = Some(serde_json::json!({
                    "query": search_query,
                    "results": results.results.iter().map(|r| serde_json::json!({
                        "title": r.title, "url": r.url, "snippet": r.snippet,
                    })).collect::<Vec<_>>(),
                    "num_results": results.results.len(),
                }));
            }
            Ok(_) => context.push_str("No results found.\n"),
            Err(e) => context.push_str(&format!("The search failed: {}\n", e)),
        }
        let prompt = format!(
            "Answer the user's question using the web search results below. Be direct and concise — do not explain your reasoning process, do not narrate what you are doing, just answer.\n\nQuestion: \"{}\"\n\n{}\n\nAnswer directly based on the search results. If the results don't contain enough information, say so briefly.",
            query, context);
        // What the model said before it decided to search is often worth keeping (a
        // partial answer, a caveat). It stays as its own turn, on screen and in the
        // history, and the searched answer follows it (Matt, 2026-09-20). The page
        // does the same for a screen-on search.
        let first_answer = strip_web_search_marker(&reply_text);
        let first_answer = if first_answer.is_empty() { None } else { Some(first_answer) };
        if let Some(first) = first_answer.clone() {
            if containment_mode.to_lowercase() != "hipaa" {
                let (s, u, m, b, c) = (session_id.clone(), user_id.clone(), query.clone(), forced_backend.clone(), containment_mode.clone());
                tokio::spawn(async move {
                    if let Ok(pool) = sqlx::SqlitePool::connect(&crate::db::get_db_url()).await {
                        if let Err(e) = crate::conversation_history::log_exchange(&pool, &s, &u, &m, &first, &b, &c, true, "hands_free").await {
                            eprintln!("[ConvHistory] ⚠️ Failed to log pre-search answer: {}", e);
                        }
                    }
                });
            }
        }
        let synthesized = Box::pin(generate_reply(
            sink.clone(), prompt, user_id.clone(), session_id.clone(), forced_backend.clone(), containment_mode.clone(),
            None, Some(true), Some(true), Some(kb_enabled), Some(query.clone()), None, true,
        )).await?;
        return Ok(ReplyResponse { original_query: Some(query.clone()), pre_search_reply: first_answer, web_search_results: web_sources, ..synthesized });
    }

    // Parse MEMORY_EXTRACT facts from the LLM response — fires for any message type,
    // no is_question gate. Both API and local models use the same MEMORY_EXTRACT marker.

    let mut extracted_facts: Vec<String> = Vec::new();

    // Tolerant parse (KI-049): small local models wrap the marker in markdown, echo the
    // prompt's own heading as the label, or put the label on one line and the fact on
    // the next. Lines consumed here are also hidden from the displayed reply below, so
    // parser and display stay in step.
    let reply_lines: Vec<&str> = reply_text.lines().collect();
    let mut consumed_extract_lines: std::collections::HashSet<usize> = std::collections::HashSet::new();
    // Call 1 of the memory pipeline (2026-10-24): cloud and custom replies end with
    // "MEMORY: yes|no" (conversation_engine); "REMEMBER:" is the older spelling. Taken from
    // the last such line, hidden from the display, and only when the word stands alone
    // ("Remember no matter what..." is prose). None when the model wrote none: on the cloud
    // that means nothing is stored, logged below; the on-device model is not asked for a
    // marker and its call 2 decides.
    let mut memory_flag: Option<bool> = None;
    for (idx, line) in reply_lines.iter().enumerate().rev() {
        let t = line.trim().trim_matches(|c: char| matches!(c, '*' | '`' | '_' | '#' | '-' | ' ')).to_lowercase();
        if let Some(rest) = t.strip_prefix("memory").or_else(|| t.strip_prefix("remember")) {
            let word = rest.trim_start_matches(|c: char| matches!(c, ':' | ' ' | '-' | '—'))
                .trim_end_matches(|c: char| matches!(c, '.' | '!' | ' '));
            if word == "yes" { memory_flag = Some(true); consumed_extract_lines.insert(idx); break; }
            if word == "no" { memory_flag = Some(false); consumed_extract_lines.insert(idx); break; }
        }
    }
    // The marker glued to the end of the last sentence instead of on its own line
    // ("...enjoy the game. MEMORY: no"): a Pixel hands-free reply showed and spoke it
    // (Matt, 2026-10-25). Taken from the last non-empty line and cut off it for display.
    let mut inline_marker_line: Option<(usize, String)> = None;
    if memory_flag.is_none() {
        if let Some((idx, line)) = reply_lines.iter().enumerate().rev().find(|(_, l)| !l.trim().is_empty()) {
            if let Some((flag, rest)) = split_inline_memory_marker(line) {
                memory_flag = Some(flag);
                inline_marker_line = Some((idx, rest));
            }
        }
    }
    if let Some(flag) = memory_flag { println!("[RUST] MEMORY line: {}", if flag { "yes" } else { "no" }); }
    let mut heading_without_fact = false;
    let mut li = 0;
    while li < reply_lines.len() {
        if let Some(rem) = extract_marker_remainder(reply_lines[li]) {
            consumed_extract_lines.insert(li);
            let mut fact = rem;
            if fact.is_empty() {
                // Label alone on its line: the fact is the next non-empty line, unless that
                // line is another label or the PART 2 heading.
                let mut lj = li + 1;
                while lj < reply_lines.len() && reply_lines[lj].trim().is_empty() { lj += 1; }
                if lj < reply_lines.len()
                    && extract_marker_remainder(reply_lines[lj]).is_none()
                    && !reply_lines[lj].trim_start().to_uppercase().starts_with("PART ")
                {
                    fact = reply_lines[lj].trim().trim_matches(|c: char| matches!(c, '*' | '`' | '_')).trim().to_string();
                    consumed_extract_lines.insert(lj);
                    li = lj;
                } else {
                    heading_without_fact = true;
                }
            }
            if !fact.is_empty() {
                extracted_facts.push(fact);
            }
        }
        li += 1;
    }
    if extracted_facts.is_empty() && heading_without_fact {
        println!("[RUST] ⚠️ Reply had a fact-extraction heading but no fact after it — nothing stored (model formatting, KI-049)");
    }

    // An old-style MEMORY_EXTRACT line (a model keeping the June habit) counts as a yes;
    // its text is not stored, call 2 extracts the facts like for any other message.
    let legacy_fact_line = !extracted_facts.is_empty();
    if legacy_fact_line { println!("[RUST] Reply carried a MEMORY_EXTRACT line (older prompt habit); treated as MEMORY: yes"); }

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

    // Strip the WEB_SEARCH_NEEDED marker line from the displayed text; the marker is for
    // internal detection only. Text on either side of it is kept: until 2026-10-20 everything
    // after the marker was dropped, and GPT had started writing the marker first and the
    // useful answer (a drafted email, a partial answer) after it, so the answer the user
    // watched stream in vanished when the stream ended (Matt, 2026-10-20).
    let reply_text = if web_search_detected { strip_web_search_marker(&reply_text) } else { reply_text };

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
            .enumerate()
            .filter_map(|(idx, line)| {
                // Lines the tolerant parser consumed (marker, echoed heading, or the fact
                // line under a bare label) never reach the display (KI-049).
                if consumed_extract_lines.contains(&idx) { return None; }
                if let Some((mi, ref rest)) = inline_marker_line { if mi == idx { return Some(rest.clone()); } }
                let t = line.trim_start();
                if t.contains("MEMORY_EXTRACT:") { return None; }
                // Handle "PART N — ..." lines from local models following the two-part format.
                // If content follows the separator, keep it. If it's just a header, drop the line.
                // Case-insensitive and colon-tolerant: Mistral 7B spoke "Part 2: This is a test
                // question…" through the phone's TTS on 2026-09-13.
                if t.to_uppercase().starts_with("PART ") && t.len() > 5 {
                    if t.chars().nth(5).map_or(false, |c| c.is_ascii_digit()) {
                        for sep in &[" \u{2014} ", " \u{2013} ", " - ", ": ", ":"] {
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
    // A NO_QUERY verdict (hands-free fragment the model judged not a request) gets
    // no warning: there is nothing to warn about, and the prefix would hide the
    // verdict from every `starts_with("NO_QUERY")` check — the Pixel read the warning
    // aloud over a flagged TV line on 2026-10-20 16:18.
    let mut final_reply_text = match warning_prefix {
        Some(warning) if !reply_text.trim().starts_with("NO_QUERY") => format!("{}\n\n{}", warning, reply_text),
        _ => reply_text,
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

    println!("[⏱️ PERF] Total request time: {:.2}s", request_start.elapsed().as_secs_f32());

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
        pre_search_reply: None,
        kb_note: kb_note.clone(),
        web_search_results: None,
    };

    // STEP 10: LOG EXCHANGE TO CONVERSATION HISTORY (non-blocking, skipped in HIPAA mode,
    // and skipped when the model answered NO_QUERY to a hands-free fragment — see
    // android_jni::HANDS_FREE_NOTE — so a TV line never becomes part of the thread)
    let is_no_query = final_reply_text.trim().starts_with("NO_QUERY");
    if containment_mode.to_lowercase() != "hipaa" && !is_no_query {
        let ch_session = session_id.clone();
        let ch_user = user_id.clone();
        let ch_message = query.clone();  // Store clean question, not file dump
        // A reply that was only the WEB_SEARCH_NEEDED marker is empty once the marker is
        // stripped; an empty assistant turn sat in History as a blank bubble (2026-09-20).
        let ch_reply = match (&web_search_query, final_reply_text.trim().is_empty()) {
            (Some(q), true) => format!("Web search needed: \"{}\"", q),
            _ => final_reply_text.clone(),
        };
        let ch_backend = forced_backend.clone();
        let ch_mode = containment_mode.clone();
        // Any real exchange may name the thread, hands-free included. The old rule
        // (hands-free never names) left a thread whose first three questions were
        // spoken titled by a later typed message, and the user could not find it
        // (2026-09-08). Fragments and NO_QUERY turns never reach this point anyway.
        let ch_name_thread = true;
        let ch_hands_free = hands_free;
        tokio::spawn(async move {
            { let db_url = crate::db::get_db_url();
                match sqlx::SqlitePool::connect(&db_url).await {
                    Ok(pool) => {
                        if let Err(e) = crate::conversation_history::log_exchange(
                            &pool, &ch_session, &ch_user, &ch_message,
                            &ch_reply, &ch_backend, &ch_mode, ch_name_thread,
                            if ch_hands_free { "hands_free" } else { "typed" },
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
    let explicit_remember_content = explicit_remember(&query);
    let is_explicit_remember = explicit_remember_content.is_some();
    // A KB-button answer that the search could not ground (nothing found, or
    // weak matches only) is not trusted enough to learn from: a tester's
    // invented answer was stored as a memory and then "confirmed" by a later
    // memory (GitHub #17/#22). An explicit "Remember:" still stores verbatim.
    let kb_ungrounded_reply = kb_enabled && !kb_grounded && !is_explicit_remember;
    let effective_skip_memory = skip_memory_storage.unwrap_or(false)
        || hipaa_ephemeral_enforcement
        || kb_ungrounded_reply;

    if hipaa_ephemeral_enforcement {
        println!("[HIPAA] 🔒 Ephemeral mode enforced - memory extraction and storage disabled");
    }
    if kb_ungrounded_reply {
        println!("[KB RAG] memory extraction skipped: the knowledge base had no real match for this question");
    }

    // Gate 1: pure trivia (single-word acks, filler phrases) never reaches a model.
    // Explicit "Remember:" bypasses it.
    // A request to translate, reprint, rewrite or proofread carries someone else's words;
    // nothing in it is learned about the user (Matt, 2026-10-26). "Remember:" still stores.
    let rewrite_request = !is_explicit_remember && crate::memory_extras::is_rewrite_request(&query);
    if rewrite_request { println!("[RUST] Translate/rewrite request: no memory extraction for this message"); }
    let memory_gate_passed = is_explicit_remember || (engine.is_memory_worthy(&query) && !rewrite_request);
    let on_device_flow = forced_backend.ends_with(".gguf") || forced_backend.to_lowercase().contains("local");
    // Call 1's answer. Cloud and custom: the MEMORY line, or an old-style fact line, says
    // yes; no line at all means no memory. On-device: no marker is asked for, so every
    // non-trivial message goes to call 2, which decides (an explicit "no" is honoured).
    let remember_content: Option<String> = explicit_remember_content.filter(|t| !t.is_empty());
    let call1_yes = if is_explicit_remember { true }
        else if on_device_flow { memory_flag != Some(false) }
        else { memory_flag == Some(true) || legacy_fact_line };
    if !is_explicit_remember && !on_device_flow && memory_flag.is_none() && !legacy_fact_line && memory_gate_passed {
        println!("[RUST] Reply carried no MEMORY line; nothing stored (call 1 of the memory pipeline)");
    }
    if !effective_skip_memory && memory_gate_passed && call1_yes {
        let bg_message = query.clone();  // Store clean question in memories, not file dump
        let bg_user_id = user_id.clone();
        let bg_session_id = session_id.clone();
        let bg_forced_backend = forced_backend.clone();
        // A memory extracted from a "Hey Zynk" turn is marked so it can be reviewed or
        // removed as a group: a fluent TV line slipped past the fragment gate and was
        // stored as the user's own biography (2026-09-08).
        let bg_source_type: &'static str = if hands_free { "hands_free" } else { "conversation" };
        let bg_containment_mode = containment_mode.clone();
        let bg_sink = sink.clone();
        let bg_is_explicit_remember = is_explicit_remember;
        let bg_user_name = user_display_name.clone();
        // The last exchange, so call 2 can resolve "she" and "it"; facts come from the message only.
        let bg_recent_turns = recent_turns_text(conversation_history.as_deref());
        // The memory job queue (memory_jobs.rs, 2026-10-24): one row per message, one
        // worker, oldest first. A spawned task per message raced the next question for the
        // phone's one model and gave up after three interruptions.
        let job = crate::memory_jobs::Job {
            id: 0, session_id: bg_session_id, user_id: bg_user_id, message: bg_message, backend: bg_forced_backend,
            source_type: bg_source_type.to_string(), containment_mode: bg_containment_mode,
            explicit_remember: bg_is_explicit_remember, remember_text: remember_content, user_name: bg_user_name,
            recent_turns: bg_recent_turns, query_entities: query_entities.clone(), attempts: 0,
        };
        crate::memory_jobs::ensure_worker(bg_sink.clone());
        match sqlx::SqlitePool::connect(&crate::db::get_db_url()).await {
            Ok(pool) => {
                match crate::memory_jobs::enqueue(&pool, &job).await {
                    Ok(id) => println!("[RUST] Memory job {} queued", id),
                    Err(e) => println!("[RUST] ⚠️ Could not queue the memory job: {}", e),
                }
                pool.close().await;
            }
            Err(e) => println!("[RUST] ⚠️ Could not open the database for the memory job: {}", e),
        }
    }
 else {
        // Nothing was extracted for this exchange. Record why, so "103 sessions with no
        // memories" can be told apart into gate-refused vs. deliberately skipped.
        let reason = if effective_skip_memory { "skipped" } else if !memory_gate_passed { "gate_skipped" } else { "model_declined" };
        let sid = session_id.clone();
        tokio::spawn(async move {
            if let Ok(pool) = sqlx::SqlitePool::connect(&crate::db::get_db_url()).await {
                crate::conversation_history::set_extraction_outcome(&pool, &sid, reason).await;
                pool.close().await;
            }
        });
    }

    // Return immediate response to user
    Ok(immediate_response)
}

// The legacy rule-based relationship detection code has been removed.
// Memory processing now happens entirely in the background task above.

#[tauri::command]
pub async fn run_ensemble(
    _app: tauri::AppHandle,
    message: String,
    models: Vec<String>,
    user_id: String,
    session_id: String,
    containment_mode: String,
    kb_enabled: Option<bool>,
    user_query: Option<String>,
    image_data: Option<Vec<crate::llm::ImageAttachment>>,
) -> Result<serde_json::Value, String> {
    // Accepted for wire-contract parity with the frontend; ensemble does not yet
    // thread a session through. Do not rename to `_session_id` — see tests/command_contract.rs
    let _ = session_id;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  🔀 ENSEMBLE REQUEST                                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    {
        // A picked photo once arrived as megabytes of text and went into the log whole
        // (2026-10-25): show the first 1,500 characters and the size.
        let n = message.chars().count();
        if n > 1500 { println!("▶ {} … ({} characters)", message.chars().take(1500).collect::<String>(), n); } else { println!("▶ {}", message); }
    }
    println!("  Models: {}", models.join(", "));
    println!();
    println!("[Ensemble] Running multi-model ensemble with {} models", models.len());
    println!("[Ensemble] Containment mode: {}", containment_mode);

    if models.len() < 2 {
        return Err("Need at least 2 models for ensemble".to_string());
    }

    // ENSEMBLE MODE: Lightweight research tool
    // - Includes: memory context, KB context, web search
    // - Excludes: containment checks (UI blocks child mode), memory storage, relationship detection

    // Model names read from user preferences (set in API Settings)
    let anthropic_model = std::env::var("ANTHROPIC_MODEL")
        .unwrap_or_else(|_| "claude-sonnet-5".to_string());
    let openai_model = std::env::var("OPENAI_MODEL")
        .unwrap_or_else(|_| "gpt-5.5".to_string());
    let xai_model = std::env::var("XAI_MODEL")
        .unwrap_or_else(|_| "grok-4.5".to_string());
    let mistral_model = std::env::var("MISTRAL_MODEL")
        .unwrap_or_else(|_| "mistral-large-latest".to_string());

    // Determine coordinator model upfront: Anthropic > xAI > OpenAI > Mistral > custom > first local model
    let coordinator_model = models.iter()
        .find(|m| m.to_lowercase().contains("anthropic"))
        .or_else(|| models.iter().find(|m| m.to_lowercase().contains("xai") || m.to_lowercase().contains("grok")))
        .or_else(|| models.iter().find(|m| m.to_lowercase().contains("openai") || m.to_lowercase().contains("gpt")))
        .or_else(|| models.iter().find(|m| m.to_lowercase().contains("mistral")))
        .or_else(|| models.iter().find(|m| m.to_lowercase() == "custom"))
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
        let enhancer = crate::nlp_enhancer::NLPEnhancer::new();
        enhancer.extract_entities(&message_for_nlp)
    })
    .await
    .map_err(|e| format!("Entity extraction failed: {}", e))?;

    // Convert entities to strings for hybrid search
    let query_entity_strings: Vec<String> = query_entities.iter().map(|e| e.word.clone()).collect();

    // Generate embedding for hybrid search
    let message_for_embedding = search_query.clone();
    let query_embedding = tokio::task::spawn_blocking(move || {
        crate::llm::local_embeddings::generate_local_embedding(&message_for_embedding)
    })
    .await
    .map_err(|e| format!("Failed to run embedding task: {}", e))?
    .map_err(|e| format!("Failed to generate embedding: {}", e))?;

    // Search for relevant memories using hybrid search
    let recalled_memories = crate::memory::hybrid_search(
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
    // Apply a higher similarity threshold for ensemble to reduce noise
    let recalled_memories: Vec<_> = recalled_memories.into_iter()
        .filter(|m| m.similarity.unwrap_or(0.0) >= 0.50)
        .collect();
    println!("[Ensemble] {} memories above 0.50 threshold", recalled_memories.len());

    // Build context string from memories
    let mut context = String::new();

    if !recalled_memories.is_empty() {
        context.push_str("\n\n[BACKGROUND CONTEXT]\nThe following facts about the user may be relevant. Use this as silent background to inform your answer. Do not enumerate, list, or explicitly reference these items unless directly asked.\n\n");
        for mem in recalled_memories.iter() {
            context.push_str(&format!("- {}\n", mem.content));
        }
        context.push_str("[END BACKGROUND CONTEXT]\n");
    }

    // KB context (if enabled) — same wording rules as the single-model path
    if kb_enabled.unwrap_or(false) {
        match crate::kb_rag::search_kb_chunks(&db_pool, &user_id, &search_query, 8, false).await {
            Ok(kb_results) => {
                let built = crate::kb_prompt::build(&kb_results, 0.15, 5);
                println!("[Ensemble] KB outcome {:?} ({} chunks returned)", built.outcome, kb_results.len());
                context.push_str(&built.context);
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

    let engine = crate::conversation_engine::ConversationEngine::new();
    let assessment_full_prompt = engine.build_prompt(&assessment_prompt, None, None, true, None);

    // Use coordinator to assess
    let needs_search = if coordinator_model.to_lowercase().contains("anthropic") {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
        if !api_key.is_empty() {
            let messages = vec![crate::llm::Message { role: "user".to_string(), content: assessment_full_prompt.clone() }];
            crate::llm::anthropic::send_message(&api_key, &anthropic_model, messages, None, Some(256), None)
                .await.map(|r| r.content).ok()
        } else { None }
    } else if coordinator_model.to_lowercase().contains("openai") || coordinator_model.to_lowercase().contains("gpt") {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        if !api_key.is_empty() {
            let messages = vec![crate::llm::Message { role: "user".to_string(), content: assessment_full_prompt.clone() }];
            crate::llm::openai::send_message(&api_key, &openai_model, messages, Some(256), None)
                .await.map(|r| r.content).ok()
        } else { None }
    } else if coordinator_model.to_lowercase().contains("xai") || coordinator_model.to_lowercase().contains("grok") {
        let api_key = std::env::var("XAI_API_KEY").unwrap_or_default();
        if !api_key.is_empty() {
            let messages = vec![crate::llm::Message { role: "user".to_string(), content: assessment_full_prompt.clone() }];
            crate::llm::xai::send_message(&api_key, &xai_model, messages, Some(256), None)
                .await.map(|r| r.content).ok()
        } else { None }
    } else if coordinator_model.to_lowercase().contains("mistral") {
        let api_key = std::env::var("MISTRAL_API_KEY").unwrap_or_default();
        if !api_key.is_empty() {
            let messages = vec![crate::llm::Message { role: "user".to_string(), content: assessment_full_prompt.clone() }];
            crate::llm::openai::send_message_streaming(&api_key, &mistral_model, messages, Some(256), None, "https://api.mistral.ai/v1/chat/completions", crate::llm::openai::default_client(false), |_| {})
                .await.map(|r| r.content).ok()
        } else { None }
    } else if coordinator_model.to_lowercase() == "custom" {
        let base_url = std::env::var("CUSTOM_API_URL").unwrap_or_default();
        let model_name = std::env::var("CUSTOM_MODEL").unwrap_or_default();
        let api_key = std::env::var("CUSTOM_API_KEY").unwrap_or_default();
        if !base_url.is_empty() && !model_name.is_empty() {
            let api_url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
            let messages = vec![crate::llm::Message { role: "user".to_string(), content: assessment_full_prompt.clone() }];
            let custom_client = build_custom_client(&base_url).await;
            crate::llm::openai::send_message_streaming(&api_key, &model_name, messages, Some(256), None, &api_url, custom_client, |_| {})
                .await.map(|r| r.content).ok()
        } else { None }
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

    // Call all models in parallel
    let model_futures: Vec<_> = models.iter().map(|model_backend| {
        let model_backend = model_backend.clone();
        let full_question = full_question.clone();
        let anthropic_model = anthropic_model.clone();
        let openai_model = openai_model.clone();
        let xai_model = xai_model.clone();
        let mistral_model = mistral_model.clone();
        let image_data = image_data.clone();

        async move {
            println!("[Ensemble] Querying model: {}", model_backend);
            let result: Result<String, String> = async {
                if model_backend.to_lowercase().contains("anthropic") {
                    let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
                    if api_key.is_empty() { return Err("ANTHROPIC_API_KEY not set".to_string()); }
                    if let Some(ref imgs) = image_data {
                        crate::llm::anthropic::send_vision_streaming(
                            &api_key, &anthropic_model, &full_question, imgs, None, Some(4096), |_| {}
                        ).await.map(|r| r.content).map_err(|e| e.to_string())
                    } else {
                        let messages = vec![crate::llm::Message { role: "user".to_string(), content: full_question.clone() }];
                        crate::llm::anthropic::send_message(&api_key, &anthropic_model, messages, None, Some(4096), None)
                            .await.map(|r| r.content).map_err(|e| e.to_string())
                    }
                } else if model_backend.to_lowercase().contains("openai") || model_backend.to_lowercase().contains("gpt") {
                    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
                    if api_key.is_empty() { return Err("OPENAI_API_KEY not set".to_string()); }
                    if let Some(ref imgs) = image_data {
                        crate::llm::openai::send_vision_streaming(
                            &api_key, &openai_model, &full_question, imgs,
                            "https://api.openai.com/v1/chat/completions", |_| {}
                        ).await.map(|r| r.content).map_err(|e| e.to_string())
                    } else {
                        let messages = vec![crate::llm::Message { role: "user".to_string(), content: full_question.clone() }];
                        crate::llm::openai::send_message(&api_key, &openai_model, messages, Some(4096), None)
                            .await.map(|r| r.content).map_err(|e| e.to_string())
                    }
                } else if model_backend.to_lowercase().contains("xai") || model_backend.to_lowercase().contains("grok") {
                    let api_key = std::env::var("XAI_API_KEY").unwrap_or_default();
                    if api_key.is_empty() { return Err("XAI_API_KEY not set".to_string()); }
                    if let Some(ref imgs) = image_data {
                        crate::llm::openai::send_vision_streaming(
                            &api_key, &xai_model, &full_question, imgs,
                            "https://api.x.ai/v1/chat/completions", |_| {}
                        ).await.map(|r| r.content).map_err(|e| e.to_string())
                    } else {
                        let messages = vec![crate::llm::Message { role: "user".to_string(), content: full_question.clone() }];
                        crate::llm::xai::send_message(&api_key, &xai_model, messages, Some(4096), None)
                            .await.map(|r| r.content).map_err(|e| e.to_string())
                    }
                } else if model_backend.to_lowercase().contains("mistral") {
                    let api_key = std::env::var("MISTRAL_API_KEY").unwrap_or_default();
                    if api_key.is_empty() { return Err("MISTRAL_API_KEY not set".to_string()); }
                    if let Some(ref imgs) = image_data {
                        crate::llm::openai::send_vision_streaming(
                            &api_key, &mistral_model, &full_question, imgs,
                            "https://api.mistral.ai/v1/chat/completions", |_| {}
                        ).await.map(|r| r.content).map_err(|e| e.to_string())
                    } else {
                        let messages = vec![crate::llm::Message { role: "user".to_string(), content: full_question.clone() }];
                        crate::llm::openai::send_message_streaming(&api_key, &mistral_model, messages, Some(4096), None, "https://api.mistral.ai/v1/chat/completions", crate::llm::openai::default_client(false), |_| {})
                            .await.map(|r| r.content).map_err(|e| e.to_string())
                    }
                } else if model_backend.to_lowercase() == "custom" {
                    let base_url = std::env::var("CUSTOM_API_URL")
                        .map_err(|_| "Custom endpoint not configured. Add it in API Settings.".to_string())?;
                    let model_name = std::env::var("CUSTOM_MODEL")
                        .map_err(|_| "No model selected for custom endpoint.".to_string())?;
                    let api_key = std::env::var("CUSTOM_API_KEY").unwrap_or_default();
                    let api_url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
                    let messages = vec![crate::llm::Message { role: "user".to_string(), content: full_question.clone() }];
                    let custom_client = build_custom_client(&base_url).await;
                    crate::llm::openai::send_message_streaming(
                        &api_key, &model_name, messages, Some(2048), None, &api_url, custom_client, |_| {}
                    ).await.map(|r| r.content).map_err(|e| format!("Custom endpoint error: {} — is Ollama running?", e))
                } else {
                    if image_data.is_some() {
                        return Err("Image attachments are not supported with local models.".to_string());
                    }
                    if !std::path::Path::new(&model_backend).exists() {
                        return Err(format!("Local model file not found: {}", model_backend));
                    }
                    let model_path = model_backend.clone();
                    let question = full_question.clone();
                    let result = tokio::task::spawn_blocking(move || {
                        let messages = vec![crate::llm::Message { role: "user".to_string(), content: question }];
                        crate::llm::local_models::generate_with_local_model(&model_path, messages, Some(512), None)
                    }).await.map_err(|e| format!("Local model task failed: {}", e))?;
                    result.map(|r| r.content).map_err(|e| e.to_string())
                }
            }.await;
            (model_backend, result)
        }
    }).collect();

    let phase1_results = futures::future::join_all(model_futures).await;

    let individual_responses: Vec<serde_json::Value> = phase1_results.into_iter().map(|(model, result)| {
        match result {
            Ok(response) => serde_json::json!({
                "model": model,
                "response": response,
                "success": true
            }),
            Err(e) => {
                println!("[Ensemble] Model {} failed: {}", model, e);
                serde_json::json!({
                    "model": model,
                    "response": format!("Error: {}", e),
                    "success": false
                })
            }
        }
    }).collect();

    // Filter successful responses
    let successful_responses: Vec<&serde_json::Value> = individual_responses
        .iter()
        .filter(|r| r["success"].as_bool().unwrap_or(false))
        .collect();

    if successful_responses.is_empty() {
        return Err("All models failed to respond. Check your API keys and connections.".to_string());
    }
    if successful_responses.len() < 2 {
        // Only 1 model responded — log which ones failed and continue with synthesis
        let failed: Vec<&str> = individual_responses.iter()
            .filter(|r| !r["success"].as_bool().unwrap_or(false))
            .map(|r| r["model"].as_str().unwrap_or("unknown"))
            .collect();
        println!("[Ensemble] Warning: only 1 model succeeded. Failed: {:?}. Proceeding with single response.", failed);
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
    let engine = crate::conversation_engine::ConversationEngine::new();
    let full_prompt = engine.build_prompt(&synthesis_prompt, None, None, true, None);

    // Call LLM directly without all the overhead
    let reply_text = if coordinator_model.to_lowercase().contains("anthropic") {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY not set".to_string())?;
        let messages = vec![crate::llm::Message { role: "user".to_string(), content: full_prompt }];
        crate::llm::anthropic::send_message_streaming(&api_key, &anthropic_model, messages, None, Some(4096), None, |_| {})
            .await.map_err(|e| e.to_string())?.content
    } else if coordinator_model.to_lowercase().contains("xai") || coordinator_model.to_lowercase().contains("grok") {
        let api_key = std::env::var("XAI_API_KEY")
            .map_err(|_| "XAI_API_KEY not set".to_string())?;
        let messages = vec![crate::llm::Message { role: "user".to_string(), content: full_prompt }];
        let response = crate::llm::xai::send_message(&api_key, &xai_model, messages, Some(4096), None)
            .await.map_err(|e| e.to_string())?;
        response.content
    } else if coordinator_model.to_lowercase().contains("openai") || coordinator_model.to_lowercase().contains("gpt") {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY not set".to_string())?;
        let messages = vec![crate::llm::Message { role: "user".to_string(), content: full_prompt }];
        let response = crate::llm::openai::send_message(&api_key, &openai_model, messages, Some(4096), None)
            .await.map_err(|e| e.to_string())?;
        response.content
    } else if coordinator_model.to_lowercase().contains("mistral") {
        let api_key = std::env::var("MISTRAL_API_KEY")
            .map_err(|_| "MISTRAL_API_KEY not set".to_string())?;
        let messages = vec![crate::llm::Message { role: "user".to_string(), content: full_prompt }];
        crate::llm::openai::send_message_streaming(&api_key, &mistral_model, messages, Some(4096), None, "https://api.mistral.ai/v1/chat/completions", crate::llm::openai::default_client(false), |_| {})
            .await.map_err(|e| e.to_string())?.content
    } else if coordinator_model.to_lowercase() == "custom" {
        let base_url = std::env::var("CUSTOM_API_URL")
            .map_err(|_| "Custom endpoint not configured".to_string())?;
        let model_name = std::env::var("CUSTOM_MODEL")
            .map_err(|_| "No model selected for custom endpoint".to_string())?;
        let api_key = std::env::var("CUSTOM_API_KEY").unwrap_or_default();
        let api_url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
        let messages = vec![crate::llm::Message { role: "user".to_string(), content: full_prompt }];
        let custom_client = build_custom_client(&base_url).await;
        crate::llm::openai::send_message_streaming(&api_key, &model_name, messages, Some(4096), None, &api_url, custom_client, |_| {})
            .await.map_err(|e| e.to_string())?.content
    } else {
        // Local model coordinator — enables fully offline ensemble
        println!("[Ensemble] Using local model as coordinator: {}", coordinator_model);
        let model_path = if coordinator_model.ends_with(".gguf") {
            coordinator_model.clone()
        } else {
            crate::llm::local_models::resolve_default_model_path()
                .map_err(|e| e.to_string())?
        };
        let messages = vec![crate::llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];
        let model_path_clone = model_path.clone();
        let response = tokio::task::spawn_blocking(move || {
            crate::llm::local_models::generate_with_local_model(&model_path_clone, messages, Some(2048), None)
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

/// The explicit "Remember:" command. Returns the text after the keyword,
/// trimmed, or `None` when the message is not a Remember command.
///
/// Accepted forms, all case-insensitive:
/// - typed: `Remember: ...`
/// - spoken: `remember colon ...` — dictation never produces punctuation, so
///   the user says the word "colon". The offline Vosk model does not know
///   that word well and has transcribed it as "cohen" and "call and" in real
///   use (OnePlus, 2026-09-07: "remember cohen the gate code is..." was stored
///   as a memory about someone named Cohen). The spelling variants below are
///   the ones seen or phonetically adjacent; each must be followed by a space
///   or end of text so "remember colonial history" is not a command.
/// - punctuation right after the keyword is ignored ("Remember colon, the...").
pub fn explicit_remember(query: &str) -> Option<String> {
    const SPOKEN_COLON: [&str; 8] = [
        "colon", "colin", "collin", "cohen", "colan", "call and", "call in", "call on",
    ];
    let trimmed = query.trim();
    let lower = trimmed.to_ascii_lowercase();

    // Typed form: "remember:" with anything after.
    if let Some(rest) = lower.strip_prefix("remember:") {
        let rest_start = trimmed.len() - rest.len();
        return Some(trimmed[rest_start..].trim().to_string());
    }

    // Spoken form: "remember" + one of the colon spellings + separator. Dictation
    // often puts a comma after "remember" ("remember, colon the ferry runs hourly"),
    // and that comma made the whole command invisible: the request went to the model
    // as chat and the extractor stored its own paraphrase instead of the words said
    // (Matt, 2026-10-07, "User noted that the ferry runs hourly in summer").
    let after_remember = match lower.strip_prefix("remember") {
        Some(r) if r.starts_with(char::is_whitespace) || r.starts_with([',', '.', ';']) => {
            r.trim_start_matches(|c: char| c.is_whitespace() || c == ',' || c == '.' || c == ';')
        }
        _ => return None,
    };
    for variant in SPOKEN_COLON {
        if let Some(rest) = after_remember.strip_prefix(variant) {
            // A space or sentence punctuation may follow; an apostrophe may not,
            // so "remember cohen's birthday" stays an ordinary sentence.
            const SEPARATORS: [char; 6] = [',', '.', ';', ':', '!', '?'];
            let separated = rest.is_empty()
                || rest.starts_with(char::is_whitespace)
                || rest.starts_with(SEPARATORS);
            if !separated {
                continue;
            }
            let rest_start = trimmed.len() - rest.len();
            let content = trimmed[rest_start..]
                .trim_start_matches(|c: char| SEPARATORS.contains(&c) || c.is_whitespace())
                .trim_end()
                .to_string();
            return Some(content);
        }
    }
    None
}

/// "…enjoy the game. MEMORY: no" → (false, "…enjoy the game."). The marker must be the
/// last thing on the line, preceded by real text; a line that is only the marker is the
/// own-line case handled before this. Case-insensitive; markdown and end punctuation
/// around the marker are ignored.
pub(crate) fn split_inline_memory_marker(line: &str) -> Option<(bool, String)> {
    let lower = line.to_lowercase();
    let key_pos = ["memory", "remember"].iter()
        .filter_map(|k| lower.rfind(k).map(|p| (p, k.len())))
        .max_by_key(|(p, _)| *p)?;
    let (pos, klen) = key_pos;
    // A separator is required: "I remember no such thing" is prose, "MEMORY: no" is the marker.
    let tail = lower[pos + klen..].trim_start_matches(|c: char| matches!(c, ' ' | '*' | '_' | '`'));
    if !tail.starts_with(|c: char| matches!(c, ':' | '-' | '—')) { return None; }
    let after = tail
        .trim_start_matches(|c: char| matches!(c, ':' | ' ' | '-' | '—' | '*' | '_' | '`'))
        .trim_end_matches(|c: char| matches!(c, '.' | '!' | ' ' | '*' | '_' | '`' | ')' | ']'));
    let flag = match after { "yes" => true, "no" => false, _ => return None };
    let before = line[..pos].trim_end_matches(|c: char| matches!(c, '*' | '_' | '`' | '(' | '[' | ' ' | '-' | '—' | '|'));
    let before = before.trim_end();
    if before.is_empty() { return None; }
    Some((flag, before.to_string()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn inline_memory_marker_is_cut_off_the_last_line() {
        use super::split_inline_memory_marker as s;
        assert_eq!(s("Enjoy the game tonight. MEMORY: no"), Some((false, "Enjoy the game tonight.".to_string())));
        assert_eq!(s("Got it, I'll keep that in mind. **MEMORY: yes**"), Some((true, "Got it, I'll keep that in mind.".to_string())));
        assert_eq!(s("Sure thing — Memory: No."), Some((false, "Sure thing".to_string())));
        assert_eq!(s("MEMORY: no"), None, "the own-line case is not this helper's");
        assert_eq!(s("I remember no such thing."), None);
        assert_eq!(s("I remember no."), None);
        assert_eq!(s("Working memory: yesterday's list is gone."), None);
    }

    #[test]
    fn web_search_marker_keeps_text_on_both_sides() {
        use super::strip_web_search_marker as strip;
        assert_eq!(strip("WEB_SEARCH_NEEDED: weather in Rutland"), "");
        assert_eq!(strip("Web search needed: \"current weather in Lima, Peru\""), "");
        assert_eq!(strip("**WEB_SEARCH_NEEDED**: news today"), "");
        assert_eq!(super::find_web_search_marker("Web search needed: \"weather\"").map(|(a, b)| (a, &"Web search needed: \"weather\""[b..])), Some((0, "\"weather\"")));
        assert_eq!(strip("WEB_SEARCH_NEEDED: weather\n"), "");
        assert_eq!(strip("Here is what I know.\n\nWEB_SEARCH_NEEDED: q"), "Here is what I know.");
        assert_eq!(strip("WEB_SEARCH_NEEDED: q\n\nMeanwhile, a draft:\nDear Senator"), "Meanwhile, a draft:\nDear Senator");
        assert_eq!(strip("First part.\nWEB_SEARCH_NEEDED: q\nSecond part."), "First part.\n\nSecond part.");
        assert_eq!(strip("No marker at all"), "No marker at all");
    }

    use super::explicit_remember;

    #[test]
    fn a_dictated_comma_after_remember_still_makes_it_a_command() {
        assert_eq!(explicit_remember("remember, colon the ferry runs hourly in summer").as_deref(), Some("the ferry runs hourly in summer"));
        assert_eq!(explicit_remember("Remember. colon, the spare key is with Priya").as_deref(), Some("the spare key is with Priya"));
        assert_eq!(explicit_remember("remember colon the library card expires in march").as_deref(), Some("the library card expires in march"));
        assert!(explicit_remember("remember, we talked about colons in grammar").is_none());
    }

    #[test]
    fn remember_keyword_is_case_insensitive() {
        assert_eq!(explicit_remember("Remember: I park on level 3").as_deref(), Some("I park on level 3"));
        assert_eq!(explicit_remember("remember: I park on level 3").as_deref(), Some("I park on level 3"));
        assert_eq!(explicit_remember("REMEMBER: I park on level 3").as_deref(), Some("I park on level 3"));
        assert_eq!(explicit_remember("  ReMeMbEr:   spaced out  ").as_deref(), Some("spaced out"));
    }

    #[test]
    fn spoken_remember_colon_from_dictation_works_too() {
        assert_eq!(explicit_remember("remember colon i park on level three").as_deref(), Some("i park on level three"));
        assert_eq!(explicit_remember("Remember colon the gate code is four two one").as_deref(), Some("the gate code is four two one"));
        assert_eq!(explicit_remember("remember colon").as_deref(), Some(""));
    }

    #[test]
    fn vosk_misspellings_of_colon_are_accepted() {
        // Both transcripts are verbatim from the OnePlus log, 2026-09-07.
        assert_eq!(explicit_remember("remember cohen the gate code is for to one").as_deref(), Some("the gate code is for to one"));
        assert_eq!(explicit_remember("remember call and i park on level three").as_deref(), Some("i park on level three"));
        assert_eq!(explicit_remember("remember colin my locker is twelve").as_deref(), Some("my locker is twelve"));
    }

    #[test]
    fn punctuation_after_the_keyword_is_ignored() {
        assert_eq!(explicit_remember("Remember colon, the gate code is 9247.").as_deref(), Some("the gate code is 9247."));
        assert_eq!(explicit_remember("Remember:  the cat is grey").as_deref(), Some("the cat is grey"));
    }

    #[test]
    fn remember_without_a_colon_is_an_ordinary_message() {
        assert!(explicit_remember("remember my birthday").is_none());
        assert!(explicit_remember("Do you remember: the cat?").is_none());
        assert!(explicit_remember("remember colonial history for me").is_none());
        assert!(explicit_remember("remember cohen's birthday is in may").is_none());
        assert!(explicit_remember("remembering colon cancer risks").is_none());
        assert!(explicit_remember("i remember colon the gate").is_none());
        assert!(explicit_remember("").is_none());
    }

    #[test]
    fn remember_with_nothing_after_it_is_empty_not_none() {
        assert_eq!(explicit_remember("Remember:").as_deref(), Some(""));
    }
}

/// Recognise a fact-extraction marker line tolerantly (KI-049). Returns the text after the
/// marker; empty when the label stands alone and the fact follows on the next line.
/// Accepts the canonical `MEMORY_EXTRACT:` and what small local models actually write:
/// The reply without its `WEB_SEARCH_NEEDED: query` line. The marker and the rest of its
/// line go; text before it and text after it are both kept, joined by a blank line.
/// A reply that was only the marker becomes empty.
/// Where the web-search marker starts and where its query begins, if a reply has one.
/// Accepts `WEB_SEARCH_NEEDED:` and the ways small models write it: any case, spaces or
/// underscores, markdown around it, a colon or dash after it (2026-10-22).
fn find_web_search_marker(text: &str) -> Option<(usize, usize)> {
    let lower = text.to_ascii_lowercase();
    let start = ["web_search_needed", "web search needed", "web-search-needed"]
        .iter().filter_map(|m| lower.find(m).map(|i| (i, m.len()))).min_by_key(|(i, _)| *i)?;
    let (pos, len) = start;
    let mut after = pos + len;
    let bytes = text.as_bytes();
    while after < bytes.len() && matches!(bytes[after], b' ' | b':' | b'-' | b'*' | b'_' | b'`') { after += 1; }
    // Markdown wrappers before the marker belong to it too.
    let mut begin = pos;
    while begin > 0 && matches!(bytes[begin - 1], b'*' | b'_' | b'`' | b'#' | b'>' | b' ') && bytes[begin - 1] != b'\n' { begin -= 1; }
    Some((begin, after))
}

fn strip_web_search_marker(text: &str) -> String {
    let Some((pos, _)) = find_web_search_marker(text) else { return text.trim().to_string() };
    let before = text[..pos].trim();
    let rest = &text[pos..];
    let after = match rest.find('\n') { Some(i) => rest[i + 1..].trim(), None => "" };
    match (before.is_empty(), after.is_empty()) {
        (true, true) => String::new(),
        (false, true) => before.to_string(),
        (true, false) => after.to_string(),
        (false, false) => format!("{}\n\n{}", before, after),
    }
}

/// markdown wrappers (`**MEMORY_EXTRACT:**`, `- MEMORY_EXTRACT:`), a space for the
/// underscore, the prompt's own headings echoed as labels ("FACT EXTRACTION:",
/// "PERSONAL FACT EXTRACTION:", "PART 1 — FACT EXTRACTION:"), and a bare heading with no
/// colon. 2026-09-12, llama3.2:3b wrote "FACT EXTRACTION:" on one line and
/// "Albert has a dog named Mike." on the next, and the strict parser stored nothing, silently.
fn extract_marker_remainder(line: &str) -> Option<String> {
    // Drop markdown emphasis/code wrappers anywhere, then leading list/heading decoration.
    let cleaned: String = line.chars().filter(|c| !matches!(c, '*' | '`')).collect();
    let mut t: &str = cleaned.trim().trim_start_matches(|c: char| matches!(c, '-' | '•' | '#' | '>' | '_' | ' ' | '\t'));
    // The local prompt's "PART 1 — " prefix.
    if t.get(..6).map_or(false, |h| h.eq_ignore_ascii_case("part 1")) {
        t = t[6..].trim_start_matches(|c: char| matches!(c, ' ' | '\u{2014}' | '\u{2013}' | '-' | ':'));
    }
    let upper = t.to_uppercase();
    for marker in ["MEMORY_EXTRACT", "MEMORY EXTRACT", "PERSONAL FACT EXTRACTION", "FACT EXTRACTION"] {
        if !upper.starts_with(marker) { continue; }
        let rest = match t.get(marker.len()..) { Some(r) => r, None => return None };
        let rest_t = rest.trim_start();
        if let Some(after) = rest_t.strip_prefix(':') {
            return Some(after.trim().trim_matches(|c: char| matches!(c, '_' | ':')).trim().to_string());
        }
        // No colon: only a bare heading counts (a sentence that merely mentions the
        // marker must not be mistaken for one).
        if rest_t.is_empty() { return Some(String::new()); }
        return None;
    }
    None
}

/// Everything the background memory task needs from the turn that spawned it.
#[derive(Clone)]
pub(crate) struct MemoryBg {
    pub(crate) bg_message: String,
    pub(crate) bg_user_id: String,
    pub(crate) bg_session_id: String,
    pub(crate) bg_forced_backend: String,
    pub(crate) bg_source_type: &'static str,
    pub(crate) bg_containment_mode: String,
    pub(crate) bg_sink: std::sync::Arc<dyn crate::response_sink::ResponseSink>,
    pub(crate) bg_is_explicit_remember: bool,
    pub(crate) bg_user_name: Option<String>,
    /// The last exchange or two, for call 2's pronouns (2026-10-24).
    pub(crate) bg_recent_turns: Option<String>,
    pub(crate) query_entities: Vec<String>,
    /// Memories already stored from this same message: never call 3 candidates (they are
    /// linked to each other by the worker; the phone called a sibling "updates" and raised
    /// a card about it, 2026-10-24).
    pub(crate) bg_sibling_ids: Vec<i32>,
    /// A name correction in the message (wrong, right): memories holding the wrong name
    /// are put in front of the relationship classifier (2026-10-25).
    pub(crate) bg_correction: Option<(String, String)>,
}

/// The last two turns as "User: … / Zynkbot: …", each cut to 300 characters.
fn recent_turns_text(history: Option<&[ConversationTurn]>) -> Option<String> {
    let h = history?;
    let start = h.len().saturating_sub(2);
    let lines: Vec<String> = h[start..].iter().map(|t| {
        let who = if t.role.eq_ignore_ascii_case("user") { "User" } else { "Zynkbot" };
        format!("{}: {}", who, crate::conversation_engine::cut_chars(&t.content, 300))
    }).collect();
    if lines.is_empty() { None } else { Some(lines.join("\n")) }
}

/// Call 2 of the memory pipeline for one message: the facts, each with title and date, and
/// on the cloud also tags, tone and named things (memory_prompts.rs). A local model is held
/// to the JSON schema and gets the slim form; a cloud model answers free-form and is parsed
/// leniently. Ok(empty) means nothing to store; Err carries "interrupted" when the user's
/// next message took the model, so the job queue can wait instead of giving up.
pub(crate) async fn extract_facts(bg: &MemoryBg) -> Result<Vec<crate::memory_prompts::ExtractedFact>, String> {
    let subject = bg.bg_user_name.clone().unwrap_or_else(|| "User".to_string());
    let today = chrono::Local::now().date_naive();
    let on_device = bg.bg_forced_backend.ends_with(".gguf") || bg.bg_forced_backend.to_lowercase().contains("local");
    let prompt = crate::memory_prompts::extraction_prompt_with(&bg.bg_message, &subject, bg.bg_recent_turns.as_deref(), today, on_device);
    let schema = if on_device { crate::memory_prompts::EXTRACTION_SCHEMA_SLIM } else { crate::memory_prompts::EXTRACTION_SCHEMA };
    let text = crate::call_backend_with_schema(&prompt, &bg.bg_forced_backend, schema).await
        .map_err(|e| { println!("[RUST BACKGROUND] Call 2 failed: {}", e); e })?;
    let facts = crate::memory_prompts::parse_extraction(&text, &bg.bg_message);
    let facts = drop_near_duplicates(facts);
    println!("[RUST BACKGROUND] Call 2: {} fact(s)", facts.len());
    Ok(facts)
}

/// Two facts from one message that mean the same thing keep only the first: the word rule
/// in the parser catches restatements, this catches rewordings ("I'm at 0.4" / "the reading
/// came back point four") by embedding similarity. The threshold sits under the storage
/// path's cross-memory duplicate check, which these pairs slipped past on 2026-10-24.
pub(crate) const SAME_MESSAGE_DUPLICATE_SIMILARITY: f32 = 0.86;
fn drop_near_duplicates(facts: Vec<crate::memory_prompts::ExtractedFact>) -> Vec<crate::memory_prompts::ExtractedFact> {
    if facts.len() < 2 { return facts; }
    let mut kept: Vec<(crate::memory_prompts::ExtractedFact, Option<Vec<f32>>)> = Vec::new();
    for f in facts {
        let emb = crate::llm::local_embeddings::generate_local_embedding(&f.fact).ok();
        let dup = match &emb {
            Some(e) => kept.iter().find(|(k, ke)| ke.as_ref().map_or(false, |ke| crate::llm::local_embeddings::cosine_similarity(e, ke) >= SAME_MESSAGE_DUPLICATE_SIMILARITY)).map(|(k, _)| k.fact.clone()),
            None => None,
        };
        if let Some(d) = dup { println!("[RUST BACKGROUND] Call 2: dropped a near-duplicate of \"{}\": {}", d, f.fact); continue; }
        kept.push((f, emb));
    }
    kept.into_iter().map(|(f, _)| f).collect()
}

/// Call 2's fields where it gave them, call 3's otherwise.
fn merge_extras(a: crate::memory_extras::MemoryExtras, b: crate::memory_extras::MemoryExtras) -> crate::memory_extras::MemoryExtras {
    crate::memory_extras::MemoryExtras {
        event_date: a.event_date.or(b.event_date),
        namespace: a.namespace.or(b.namespace),
        tags: a.tags.or(b.tags),
        tone: a.tone.or(b.tone),
        entities: a.entities.or(b.entities),
        passing: a.passing.or(b.passing),
    }
}

/// The background memory pipeline for one candidate fact (or, with None, for the raw
/// message): embedding, duplicate and contradiction checks, the memory decision,
/// relationships, storage. Was the body of the spawned task in send_message_with_memory
/// until 2026-10-22; lifted out so the local-model flow can run it once per fact.
/// Returns the id of the memory it stored, None when nothing was stored (duplicate, held for a
/// contradiction card, or an error), so the caller can link the facts of one message.
pub(crate) async fn run_memory_background(bg: MemoryBg, fact: crate::memory_prompts::ExtractedFact) -> Option<i32> {
    let MemoryBg { bg_message, bg_user_id, bg_session_id, bg_forced_backend, bg_source_type, bg_containment_mode,
        bg_sink, bg_is_explicit_remember, bg_user_name: _, bg_recent_turns: _, query_entities, bg_sibling_ids, bg_correction } = bg;
            // The stored text is the fact from call 2 (or the words after "Remember:"), never
            // the raw message: until 2026-10-24 a cloud reply with no fact line stored the
            // user's message verbatim (219 of them in September).
            let factual_content = fact.fact.clone();

            // Generate embedding FIRST (needed for duplicate check and storage)
            let factual_clone = factual_content.clone();
            let message_embedding = match tokio::task::spawn_blocking(move || {
                crate::llm::local_embeddings::generate_local_embedding(&factual_clone)
            })
            .await {
                Ok(Ok(embedding)) => embedding,
                Ok(Err(e)) => {
                    println!("[RUST BACKGROUND] ⚠️ Failed to generate embedding: {}", e);
                    return None;
                }
                Err(e) => {
                    println!("[RUST BACKGROUND] ⚠️ Embedding task panicked: {}", e);
                    return None;
                }
            };

            // Pending indicator for the Memory Manager spinner: every call here has a fact.
            let has_memory_candidate = true;
            let pending_id: u64 = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            if has_memory_candidate {
                bg_sink.event("memory-processing-started", serde_json::json!({ "pending_id": pending_id }));
            }

            // Reconnect to database for memory storage
            let db_url = crate::db::get_db_url();

            let db_pool = match sqlx::SqlitePool::connect(&db_url).await {
                Ok(pool) => pool,
                Err(e) => {
                    println!("[RUST BACKGROUND] ⚠️ Failed to connect to database: {}", e);
                    return None;
                }
            };

            // Do a BROADER search for relationship detection (more memories, lower threshold)
            // This ensures we catch relationships even if the memory wasn't in the top 5 for conversation

            let relationship_search_results = match crate::memory::hybrid_search(
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
                    return None;
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
            // Cloud 10; the phone 4 (6 until 2026-10-24), the same number as its recall, since
            // every candidate is prompt the phone reads at ~40 tokens/s.
            let relationship_candidate_limit = if crate::conversation_engine::ConversationEngine::is_api_model(&bg_forced_backend) { 10 } else { 4 };
            similar_memories.retain(|(id, _, _, _)| !bg_sibling_ids.contains(id));
            // A name correction: every memory that holds the wrong name goes first, whatever
            // its similarity, so the classifier sees the pair that conflicts (2026-10-25).
            if let Some((wrong, _)) = &bg_correction {
                let held: Vec<(i32, String, Option<String>)> = sqlx::query_as(
                    "SELECT id, content, title FROM memories WHERE user_id = ? AND (temporal_status IS NULL OR temporal_status != 'superseded') AND (expires_at IS NULL OR substr(expires_at, 1, 10) >= date('now')) AND content LIKE ? ORDER BY created_at DESC LIMIT 10")
                    .bind(&bg_user_id).bind(format!("%{}%", wrong)).fetch_all(&db_pool).await.unwrap_or_default();
                let mut forced: Vec<(i32, String, Option<String>, f32)> = held.into_iter().filter(|(id, _, _)| !bg_sibling_ids.contains(id)).map(|(id, content, title)| (id, content, title, 1.0)).collect();
                println!("[RUST BACKGROUND] Name correction {} -> {}: {} memories hold the old name", wrong, bg_correction.as_ref().map(|c| c.1.as_str()).unwrap_or(""), forced.len());
                similar_memories.retain(|(id, _, _, _)| !forced.iter().any(|(f, _, _, _)| f == id));
                forced.extend(similar_memories.drain(..));
                similar_memories = forced;
            }
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
                if let Ok(Some(candidate_mem)) = crate::memory::get_memory(&db_pool, *mem_id).await {
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
                if has_memory_candidate {
                    bg_sink.event("memory-processing-complete", serde_json::json!({ "pending_id": pending_id, "status": "skipped" }));
                }
                crate::conversation_history::set_extraction_outcome(&db_pool, &bg_session_id, "duplicate").await;
                db_pool.close().await;
                return None;
            }

            println!("[RUST BACKGROUND] ✅ Proceeding with relationship classification");

            // Call 3: how the fact relates to the similar memories. Title, date, tags, tone and
            // named things come from call 2 when it gave them; call 3's own are the fallback
            // ("Remember:" skips call 2, so its title comes from here). On-device links go
            // through the floor and the dropped-types list (lib.rs LOCAL_LINK_CONFIDENCE_FLOOR /
            // LOCAL_DROPPED_LINK_TYPES, memory bench 2026-10-22/23) and are tagged "local_llm".
            let bg_on_device = bg_forced_backend.ends_with(".gguf") || bg_forced_backend.to_lowercase().contains("local");
            let link_created_by = if bg_on_device { "local_llm" } else { "llm" };
            let (should_remember, llm_title, llm_relationships, llm_extras) = if similar_memories.is_empty() && fact.title.is_some() {
                println!("[RUST BACKGROUND] No similar memories; call 3 skipped");
                (true, fact.title.clone(), Vec::new(), fact.extras.clone())
            } else if bg_on_device && fact.title.is_some() {
                // The phone: relationships only (ask_llm_for_links_only); title and date came
                // from call 2, tags/tone/entities from the name finder below.
                match crate::ask_llm_for_links_only(&factual_content, &similar_memories, &bg_forced_backend).await {
                    Ok(rels) => {
                        let before = rels.len();
                        let rels = crate::filter_local_links(rels);
                        if rels.len() < before { println!("[RUST BACKGROUND] Dropped {} link(s) from the local model (floor / dropped types)", before - rels.len()); }
                        println!("[RUST BACKGROUND] ✅ Links (phone): {} relationships", rels.len());
                        (true, fact.title.clone(), rels, fact.extras.clone())
                    }
                    Err(e) => {
                        println!("[RUST BACKGROUND] ⚠️ Link call failed: {} — storing without links", e);
                        (true, fact.title.clone(), Vec::new(), fact.extras.clone())
                    }
                }
            } else {
                match crate::ask_llm_for_relationships(&factual_content, &similar_memories, &bg_forced_backend).await {
                    Ok((title3, rels, extras3)) => {
                        let before = rels.len();
                        let rels = if bg_on_device { crate::filter_local_links(rels) } else { rels };
                        if rels.len() < before { println!("[RUST BACKGROUND] Dropped {} link(s) from the local model (floor / dropped types)", before - rels.len()); }
                        println!("[RUST BACKGROUND] ✅ Relationship classifier: {} relationships", rels.len());
                        (true, fact.title.clone().or(title3), rels, merge_extras(fact.extras.clone(), extras3))
                    }
                    Err(e) => {
                        println!("[RUST BACKGROUND] ⚠️ Relationship classifier failed: {} — storing without links", e);
                        let fallback = fact.title.clone().unwrap_or_else(|| crate::generate_title_from_content(&factual_content));
                        (true, Some(fallback), Vec::new(), fact.extras.clone())
                    }
                }
            };

            // Explicit "Remember:" command overrides LLM decision — user is the authority.
            // The title comes from the user's own words too: the classifier sees the raw
            // query, and on a dictated "remember cohen the gate code..." it titled the
            // memory "Gate code for Cohen" (2026-09-07). The stored content is already
            // the text after the keyword, so the title is cut from that.
            let (should_remember, llm_title) = if bg_is_explicit_remember {
                if !should_remember {
                    println!("[RUST BACKGROUND] ✅ Explicit 'Remember:' command — overriding LLM decision to store");
                }
                let fallback_title = factual_content.chars().take(60).collect::<String>();
                let fallback_title = if factual_content.chars().count() > 60 {
                    format!("{}…", fallback_title)
                } else {
                    fallback_title
                };
                (true, Some(fallback_title))
            } else {
                (should_remember, llm_title)
            };

            if !should_remember {
                bg_sink.event("memory-processing-complete", serde_json::json!({ "pending_id": pending_id, "status": "skipped" }));
                crate::conversation_history::set_extraction_outcome(&db_pool, &bg_session_id, "model_declined").await;
                db_pool.close().await;
                return None;
            }

            println!("[RUST BACKGROUND] ✅ LLM decided to remember: {:?}", llm_title);

            // CHECK FOR CONTRADICTIONS (BEFORE STORAGE!)
            // If LLM detected contradiction with sufficient confidence, emit event to frontend
            // Require >= 0.65 to avoid false positives from local models
            let contradiction_detected = llm_relationships.iter()
                .any(|rel| rel.relationship_type == "contradicts" && rel.confidence.unwrap_or(0.0) >= 0.65);

            // An explicit "Remember:" is stored no matter what. The contradiction modal
            // needs the page in front to be answered; hands-free there is no page, so
            // holding the memory for it meant the fact the user just dictated was never
            // saved while the reply said it was (OnePlus, 2026-09-07). The user has
            // already stated the new fact; the older memory stays and can be resolved
            // or deleted in Memory Manager.
            if contradiction_detected && bg_is_explicit_remember {
                println!("[RUST BACKGROUND] ⚠️ Contradiction noted, but this is an explicit Remember — storing anyway");
            }
            // On-device model (2026-10-23): store first, ask after. Its background pass can
            // take minutes on a phone, the card then arrives when the user has moved on, and
            // a memory held for an unanswered card was simply lost (the kitchen-scale test).
            // So the new memory is stored like any other, and the card is raised afterwards
            // about the already-stored pair (resolve_stored_conflict). Cloud models keep the
            // ask-first card: theirs arrives in seconds.
            let ask_after_store = contradiction_detected && !bg_is_explicit_remember && bg_on_device;
            if ask_after_store {
                println!("[RUST BACKGROUND] ⚠️ Contradiction from the on-device model — storing first, asking after");
            }
            if contradiction_detected && !bg_is_explicit_remember && !bg_on_device {
                println!("[RUST BACKGROUND] ⚠️ CONTRADICTION DETECTED - emitting event to frontend (NOT storing yet)");

                // Find the contradicting relationship
                if let Some(contradiction) = llm_relationships.iter()
                    .find(|rel| rel.relationship_type == "contradicts") {

                    // Fetch the conflicting memory details
                    if let Ok(Some(conflicting_memory)) = crate::memory::get_memory(&db_pool, contradiction.memory_id).await {
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
                                "original_text": bg_message.clone(),
                            },
                            "relationships": llm_relationships,
                            "user_id": bg_user_id.clone(),
                            "session_id": bg_session_id.clone(),
                        });

                        // Emit event to frontend to show modal
                        // Written down first: the window may be gone by now (migration 0024).
                        let mut payload = payload;
                        match crate::memory::insert_memory_card(&db_pool, &bg_user_id, &bg_session_id, &payload).await {
                            Ok(card_id) => { payload["card_id"] = serde_json::json!(card_id); }
                            Err(e) => eprintln!("[RUST BACKGROUND] card not saved: {}", e),
                        }
                        bg_sink.event("contradiction-detected", payload);
                        bg_sink.card_notice("Zynkbot has a question about a memory", "Tap to answer it.");
                        println!("[RUST BACKGROUND] 🔔 Contradiction event emitted - modal should appear");

                        // Don't store memory yet - wait for user decision via resolve_memory_conflict_v2
                        bg_sink.event("memory-processing-complete", serde_json::json!({ "pending_id": pending_id, "status": "contradiction" }));
                        crate::conversation_history::set_extraction_outcome(&db_pool, &bg_session_id, "contradiction").await;
                        db_pool.close().await;
                        return None;
                    }
                }
            }

            // HIPAA mode: Enable ephemeral memory (8-hour expiration)
            let _is_ephemeral = bg_containment_mode == "hipaa";

            // Extract entities using NLP enhancer (no tags - entities handle everything)
            // Note: We use LLM-generated title, NLP for entities/events
            let message_for_nlp = factual_content.clone();
            let (entities, event_type, event_date, namespace) = match tokio::task::spawn_blocking(move || {
                let enhancer = crate::nlp_enhancer::NLPEnhancer::new();

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
                    return None;
                }
            };

            // The decision call's own fields win over the NLP guesses when present and
            // valid (2026-09-07): a real event date, a namespace from the known list,
            // tags, tone, and named entities. NLP stays as the fallback.
            let today = chrono::Utc::now().date_naive();
            let namespace = crate::memory_extras::resolve_namespace(llm_extras.namespace.as_deref(), &namespace);
            let event_date = crate::memory_extras::validate_event_date(llm_extras.event_date.as_deref(), &bg_message, today)
                .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc())
                .or(event_date);
            let tags = crate::memory_extras::clean_tags(llm_extras.tags.as_deref());
            let (sentiment_label, sentiment_score) = crate::memory_extras::tone_to_sentiment(llm_extras.tone.as_deref());
            let llm_entities = crate::memory_extras::clean_entities(llm_extras.entities.as_deref());

            // Store memory in database (with LLM-generated title, NLP entities, and events!)
            let memory_id = match crate::memory::insert_memory(
                &db_pool,
                llm_title.as_deref(),              // title (LLM-generated!)
                &factual_content,                  // content (factual statements only!)
                Some(bg_source_type),              // source_type ("hands_free" for wake-word turns)
                Some(&bg_session_id),              // session_id
                Some(message_embedding),           // embedding
                None,                              // parent_scroll_id
                None,                              // chunk_index
                Some(&bg_user_id),                 // user_id
                &namespace,                        // namespace (NLP-detected)
                true,                              // is_syncable
                false,                             // is_shareable
                Some(entities.clone()),            // entities_detected (includes proper + common nouns!)
                event_type.as_deref(),             // event_type (auto-detected!)
                event_date,                        // event_date (auto-extracted!)
                Some(&bg_message),                 // original_text (FULL original user message for context!)
            ).await {
                Ok(id) => {
                    println!("[RUST BACKGROUND] ✅ Memory stored with ID: {}", id);
                    // A passing fact (an errand, a parking level, this week's trip) leaves recall
                    // two weeks after its day and stays in Memory Manager with a Keep button
                    // (Matt, 2026-10-26). Lasting facts are untouched.
                    if llm_extras.passing == Some(true) {
                        let until = crate::memory_extras::passing_expiry(event_date, chrono::Utc::now());
                        match sqlx::query("UPDATE memories SET is_ephemeral = 1, expires_at = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?").bind(until).bind(id).execute(&db_pool).await {
                            Ok(_) => println!("[RUST BACKGROUND] Passing fact: memory #{} leaves recall after {}", id, until.format("%Y-%m-%d")),
                            Err(e) => println!("[RUST BACKGROUND] ⚠️ Could not mark memory #{} as passing: {}", id, e),
                        }
                    }
                    bg_sink.event("memory-processing-complete", serde_json::json!({ "pending_id": pending_id, "status": "stored", "memory_id": id }));
                    crate::conversation_history::set_extraction_outcome(&db_pool, &bg_session_id, "stored").await;
                    id
                }
                Err(e) => {
                    println!("[RUST BACKGROUND] ⚠️ Failed to store memory: {}", e);
                    bg_sink.event("memory-processing-complete", serde_json::json!({ "pending_id": pending_id, "status": "error" }));
                    crate::conversation_history::set_extraction_outcome(&db_pool, &bg_session_id, "error").await;
                    db_pool.close().await;
                    return None;
                }
            };

            let _ = sqlx::query("UPDATE memories SET provenance_json = ? WHERE id = ?")
                .bind(serde_json::json!({ "annotated": "at write", "by": bg_forced_backend.clone(), "requested": bg_is_explicit_remember }).to_string())
                .bind(memory_id).execute(&db_pool).await;
            if let Err(e) = crate::memory::set_memory_extras(&db_pool, memory_id, &tags, sentiment_label, sentiment_score).await {
                println!("[RUST BACKGROUND] ⚠️ Could not save tags/tone: {}", e);
            }
            // Entities: the model's list when it gave one, else the name-finder's proper nouns.
            let entity_rows: Vec<crate::memory_extras::EntityOut> = if !llm_entities.is_empty() {
                llm_entities
            } else {
                entities.as_array().map(|arr| arr.iter().filter_map(|e| {
                    let word = e.get("word")?.as_str()?.trim().to_string();
                    let label = e.get("label")?.as_str().unwrap_or("").to_string();
                    let score = e.get("score").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    if word.chars().count() < 3 || word.starts_with("##") || score < 0.6 { return None; }
                    let kind = if label.contains("PER") { "person" } else if label.contains("LOC") { "place" } else if label.contains("ORG") { "org" } else if word.chars().next().map_or(false, |c| c.is_uppercase()) { "thing" } else { return None; };
                    Some(crate::memory_extras::EntityOut { name: word, kind: kind.to_string() })
                }).collect()).unwrap_or_default()
            };
            match crate::memory::insert_memory_entities(&db_pool, memory_id, &entity_rows).await {
                Ok(n) if n > 0 => println!("[RUST BACKGROUND] {} entit{} recorded", n, if n == 1 { "y" } else { "ies" }),
                Ok(_) => {}
                Err(e) => println!("[RUST BACKGROUND] ⚠️ Could not save entities: {}", e),
            }

            // Store LLM-classified relationships (if any)
            if !llm_relationships.is_empty() {
                println!("[RUST BACKGROUND] Creating {} LLM-classified relationships...", llm_relationships.len());

                for rel in &llm_relationships {
                    // Skip "none" relationships
                    if rel.relationship_type == "none" {
                        continue;
                    }

                    // Create relationship in database
                    match crate::memory::create_memory_link(
                        &db_pool,
                        memory_id,
                        rel.memory_id,
                        &rel.relationship_type,
                        rel.confidence.unwrap_or(0.75),
                        Some(&rel.reason),
                        link_created_by,
                    ).await {
                        Ok(link_id) => {
                            println!("[RUST BACKGROUND] ✅ Created {} relationship (link #{}): {} -> {}",
                                     rel.relationship_type, link_id, memory_id, rel.memory_id);
                        }
                        Err(e) => {
                            println!("[RUST BACKGROUND] ⚠️ Failed to create relationship: {}", e);
                        }
                    }

                    // "updates": the new memory replaces the older state of the same thing.
                    // The older memory leaves recall and stays as history; Memory Manager
                    // shows it as replaced with a way back (migration 0022, 2026-10-23).
                    if rel.relationship_type == "updates" {
                        match crate::memory::set_temporal_status(&db_pool, rel.memory_id, Some("superseded")).await {
                            Ok(()) => {
                                println!("[RUST BACKGROUND] 🔁 Memory #{} replaces #{}; the older one is kept as history", memory_id, rel.memory_id);
                                bg_sink.event("memory-superseded", serde_json::json!({ "new_memory_id": memory_id, "old_memory_id": rel.memory_id }));
                            }
                            Err(e) => println!("[RUST BACKGROUND] ⚠️ Could not mark memory #{} as replaced: {}", rel.memory_id, e),
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
                        let _ = sqlx::query("UPDATE memories SET updated_at = datetime('now') WHERE id = ?")
                            .bind(mem_id)
                            .execute(&db_pool)
                            .await;
                    }
                    println!("[RUST BACKGROUND] ✅ Updated {} memory timestamps to trigger relationship sync", relationship_count + 1);
                }
            }

            // The on-device model's card, raised about the stored pair (see ask_after_store).
            if ask_after_store {
                if let Some(contradiction) = llm_relationships.iter()
                    .find(|rel| rel.relationship_type == "contradicts") {
                    if let Ok(Some(conflicting_memory)) = crate::memory::get_memory(&db_pool, contradiction.memory_id).await {
                        let payload = serde_json::json!({
                            "stored_memory_id": memory_id,
                            "memoryA": {
                                "id": conflicting_memory.id,
                                "content": conflicting_memory.content,
                                "title": conflicting_memory.title,
                                "created_at": conflicting_memory.created_at.to_rfc3339(),
                            },
                            "memoryB": {
                                "id": memory_id,
                                "content": factual_content.clone(),
                                "title": llm_title.clone(),
                                "created_at": chrono::Utc::now().to_rfc3339(),
                            },
                            "reason": contradiction.reason,
                            "confidence": contradiction.confidence.unwrap_or(0.75),
                            "relationships": llm_relationships,
                            "user_id": bg_user_id.clone(),
                            "session_id": bg_session_id.clone(),
                        });
                        // Written down first: the window may be gone by now (migration 0024).
                        let mut payload = payload;
                        match crate::memory::insert_memory_card(&db_pool, &bg_user_id, &bg_session_id, &payload).await {
                            Ok(card_id) => { payload["card_id"] = serde_json::json!(card_id); }
                            Err(e) => eprintln!("[RUST BACKGROUND] card not saved: {}", e),
                        }
                        bg_sink.event("contradiction-detected", payload);
                        bg_sink.card_notice("Zynkbot has a question about a memory", "Tap to answer it.");
                        println!("[RUST BACKGROUND] 🔔 Card raised about stored memory #{} and #{}", memory_id, conflicting_memory.id);
                    }
                }
            }

            // Relationship classification complete
            if !llm_relationships.is_empty() {
                println!("[RUST BACKGROUND] ✅ Classified {} relationships", llm_relationships.len());
            }

            db_pool.close().await;
            println!("[RUST BACKGROUND] 🏁 Memory processing complete");
            Some(memory_id)
}



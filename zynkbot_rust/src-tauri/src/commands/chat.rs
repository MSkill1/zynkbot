use crate::ConversationTurn;
use crate::ReplyResponse;
use crate::Memory;
use tauri::Emitter;

/// No Flask dependency - handles everything in Rust
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
    _kb_enabled: Option<bool>,
    user_query: Option<String>,
    image_data: Option<crate::llm::ImageAttachment>,
) -> Result<ReplyResponse, String> {
    use crate::conversation_engine::ConversationEngine;

    // Normalize brand name misspellings from voice transcription before any processing
    let message = crate::normalize_brand_names(message);

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
    let memory_search_limit = if ConversationEngine::is_api_model(&forced_backend) { 15 } else { 10 };

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

    // ONE-HOP GRAPH TRAVERSAL: For each recalled memory, pull in directly linked
    // memories via "elaborates", "contradicts", or "resolves" relationships.
    // Capped at 3 additional memories to prevent prompt bloat.
    let mut linked_memories: Vec<crate::memory::Memory> = Vec::new();
    if !is_zynkbot_query {
        let already_included_ids: std::collections::HashSet<i32> = recalled_memories.iter()
            .map(|m| m.id)
            .collect();
        let mut already_included_content: std::collections::HashSet<String> = recalled_memories.iter()
            .map(|m| m.content.clone())
            .collect();

        for mem in &recalled_memories {
            if let Ok(links) = crate::memory::get_memory_links(&db_pool, mem.id).await {
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
                    if let Ok(Some(linked_mem)) = crate::memory::get_memory(&db_pool, linked_id).await {
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
        match crate::kb_rag::search_kb_chunks(&db_pool, &user_id, &query, kb_chunk_limit, false).await {
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
    let user_display_name = crate::memory::get_user_display_name(&db_pool, &user_id).await;

    // Close database connection
    db_pool.close().await;

    // STEP 5: Build prompt using conversation engine
    let engine = ConversationEngine::new();

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
    let mut local_session_rx: Option<tokio::sync::oneshot::Receiver<crate::llm::local_models::LocalModelSession>> = None;

    let reply_text = if forced_backend.to_lowercase().contains("anthropic") || forced_backend.to_lowercase().contains("claude") {
        // Use Anthropic with streaming
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY not set".to_string())?;

        let model_name = if forced_backend.contains("haiku") {
            "claude-haiku-4-5-20251001"
        } else if forced_backend.contains("opus") {
            "claude-opus-4-7"
        } else {
            "claude-sonnet-4-6"
        };

        let app_handle = app.clone();

        if let Some(ref img) = image_data {
            println!("[⏱️ PERF] Calling Anthropic API ({}) with vision streaming...", model_name);
            let response = crate::llm::anthropic::send_vision_streaming(
                &api_key,
                model_name,
                &full_prompt,
                img,
                None,
                Some(4096),
                move |token| { app_handle.emit("stream-token", token).ok(); },
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
                move |token| { app_handle.emit("stream-token", token).ok(); },
            ).await.map_err(|e| e.to_string())?;
            response.content
        }

    } else if forced_backend.to_lowercase().contains("openai") || forced_backend.to_lowercase().contains("gpt") {
        // Use OpenAI with streaming (including Child mode)
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY not set".to_string())?;

        let model_name = if image_data.is_some() { "gpt-4o" } else { "gpt-4o-mini" };

        let app_handle = app.clone();

        if let Some(ref img) = image_data {
            println!("[⏱️ PERF] Calling OpenAI API ({}) with vision streaming...", model_name);
            let response = crate::llm::openai::send_vision_streaming(
                &api_key,
                model_name,
                &full_prompt,
                img,
                "https://api.openai.com/v1/chat/completions",
                move |token| { app_handle.emit("stream-token", token).ok(); },
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
                move |token| { app_handle.emit("stream-token", token).ok(); },
            ).await.map_err(|e| e.to_string())?;
            response.content
        }

    } else if forced_backend.to_lowercase().contains("xai") || forced_backend.to_lowercase().contains("grok") {
        // Use xAI (Grok) with streaming - OpenAI-compatible format
        let api_key = std::env::var("XAI_API_KEY")
            .map_err(|_| "XAI_API_KEY not set. Get your API key from https://console.x.ai/".to_string())?;

        let app_handle = app.clone();

        if let Some(ref img) = image_data {
            println!("[⏱️ PERF] Calling xAI API (grok-4.3) with vision streaming...");
            let response = crate::llm::openai::send_vision_streaming(
                &api_key,
                "grok-4.3",
                &full_prompt,
                img,
                "https://api.x.ai/v1/chat/completions",
                move |token| { app_handle.emit("stream-token", token).ok(); },
            ).await.map_err(|e| e.to_string())?;
            response.content
        } else {
            let model_name = "grok-4.3";
            println!("[⏱️ PERF] Calling xAI API ({}) with streaming...", model_name);
            let messages = vec![crate::llm::Message {
                role: "user".to_string(),
                content: full_prompt,
            }];
            let response = crate::llm::openai::send_message_streaming(
                &api_key,
                model_name,
                messages,
                Some(4096),
                None,
                "https://api.x.ai/v1/chat/completions",
                move |token| { app_handle.emit("stream-token", token).ok(); },
            ).await.map_err(|e| e.to_string())?;
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

        let app_handle = app.clone();
        let messages = vec![crate::llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];

        println!("[⏱️ PERF] Calling custom endpoint ({}) model: {}", api_url, model_name);
        let response = crate::llm::openai::send_message_streaming(
            &api_key,
            &model_name,
            messages,
            Some(4096),
            None,
            &api_url,
            move |token| { app_handle.emit("stream-token", token).ok(); },
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
            // Use default model from environment or fallback
            std::env::var("LOCAL_MODEL_PATH")
                .unwrap_or_else(|_| "models/user/Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string())
        };

        let messages = vec![crate::llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];

        // Paired-call: load model once, generate main response, pass session to background
        // task via channel so Call 2 reuses the already-loaded model.
        let (session_tx, session_rx) = tokio::sync::oneshot::channel::<crate::llm::local_models::LocalModelSession>();
        local_session_rx = Some(session_rx);

        let model_path_clone = model_path.clone();
        let response = tokio::task::spawn_blocking(move || {
            let session = crate::llm::local_models::LocalModelSession::load(&model_path_clone)?;
            let response = session.generate(messages, Some(4096), None, None)?;
            // Send session to background task — if the receiver was already dropped, ignore.
            let _ = session_tx.send(session);
            Ok::<_, crate::llm::LLMError>(response)
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
                        if let Err(e) = crate::conversation_history::log_exchange(
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

    // Explicit "Remember:" commands always store verbatim — override any LLM MEMORY_EXTRACT.
    if is_explicit_remember {
        let remember_content = query.trim()["remember:".len()..].trim().to_string();
        if !remember_content.is_empty() {
            println!("[RUST] 📌 Explicit Remember: command — storing verbatim content");
            extracted_facts = vec![remember_content];
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
                crate::llm::local_embeddings::generate_local_embedding(&factual_clone)
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
                match crate::ask_llm_for_relationships(&factual_content, &similar_memories, &bg_forced_backend, local_session).await {
                    Ok((title, rels)) => {
                        println!("[RUST BACKGROUND] ✅ Relationship classifier: {} relationships", rels.len());
                        (true, title, rels)
                    }
                    Err(e) => {
                        println!("[RUST BACKGROUND] ⚠️ Relationship classifier failed: {} — storing without links", e);
                        let fallback = crate::generate_title_from_content(&factual_content);
                        (true, Some(fallback), Vec::new())
                    }
                }
            } else {
                // API: ask_llm_about_memory_with_relationships (should_remember + relationships)
                match crate::ask_llm_about_memory_with_relationships(
                    &bg_message,
                    "Background memory processing",
                    &similar_memories,
                    &bg_forced_backend,
                ).await {
                    Ok(result) => result,
                    Err(e) => {
                        println!("[RUST BACKGROUND] ⚠️ LLM call failed: {}", e);
                        if !bg_forced_backend.contains("local") && !bg_forced_backend.ends_with(".gguf") {
                            match crate::ask_llm_about_memory_with_relationships(
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
                    return;
                }
            };

            // Store memory in database (with LLM-generated title, NLP entities, and events!)
            let memory_id = match crate::memory::insert_memory(
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
                    match crate::memory::create_memory_link(
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
                        let _ = sqlx::query("UPDATE memories SET updated_at = datetime('now') WHERE id = ?")
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
pub async fn run_ensemble(
    _app: tauri::AppHandle,
    message: String,
    models: Vec<String>,
    user_id: String,
    _session_id: String,
    containment_mode: String,
    kb_enabled: Option<bool>,
    user_query: Option<String>,
    image_data: Option<crate::llm::ImageAttachment>,
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
    const XAI_MODEL: &str = "grok-4.3";

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
        match crate::kb_rag::search_kb_chunks(&db_pool, &user_id, &search_query, 8, false).await {
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

    let engine = crate::conversation_engine::ConversationEngine::new();
    let assessment_full_prompt = engine.build_prompt(&assessment_prompt, None, None, true, None);

    // Use coordinator to assess
    let needs_search = if coordinator_model.to_lowercase().contains("anthropic") {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
        if !api_key.is_empty() {
            let messages = vec![crate::llm::Message {
                role: "user".to_string(),
                content: assessment_full_prompt.clone(),
            }];
            crate::llm::anthropic::send_message(&api_key, ANTHROPIC_MODEL, messages, None, Some(256), None)
                .await
                .map(|r| r.content)
                .ok()
        } else {
            None
        }
    } else if coordinator_model.to_lowercase().contains("openai") {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        if !api_key.is_empty() {
            let messages = vec![crate::llm::Message {
                role: "user".to_string(),
                content: assessment_full_prompt.clone(),
            }];
            crate::llm::openai::send_message(&api_key, OPENAI_MODEL, messages, Some(256), None)
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
            } else if let Some(ref img) = image_data {
                crate::llm::anthropic::send_vision_streaming(
                    &api_key, ANTHROPIC_MODEL, &full_question, img, None, Some(4096), |_| {}
                ).await.map(|r| r.content).map_err(|e| e.to_string())
            } else {
                let messages = vec![crate::llm::Message { role: "user".to_string(), content: full_question.clone() }];
                crate::llm::anthropic::send_message(&api_key, ANTHROPIC_MODEL, messages, None, Some(4096), None)
                    .await.map(|r| r.content).map_err(|e| e.to_string())
            }
        } else if model_backend.to_lowercase().contains("openai") || model_backend.to_lowercase().contains("gpt") {
            let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
            if api_key.is_empty() {
                Err("OPENAI_API_KEY not set".to_string())
            } else if let Some(ref img) = image_data {
                crate::llm::openai::send_vision_streaming(
                    &api_key, "gpt-4o", &full_question, img,
                    "https://api.openai.com/v1/chat/completions", |_| {}
                ).await.map(|r| r.content).map_err(|e| e.to_string())
            } else {
                let messages = vec![crate::llm::Message { role: "user".to_string(), content: full_question.clone() }];
                crate::llm::openai::send_message(&api_key, OPENAI_MODEL, messages, Some(4096), None)
                    .await.map(|r| r.content).map_err(|e| e.to_string())
            }
        } else if model_backend.to_lowercase().contains("xai") || model_backend.to_lowercase().contains("grok") {
            let api_key = std::env::var("XAI_API_KEY").unwrap_or_default();
            if api_key.is_empty() {
                Err("XAI_API_KEY not set".to_string())
            } else if let Some(ref img) = image_data {
                crate::llm::openai::send_vision_streaming(
                    &api_key, "grok-4.3", &full_question, img,
                    "https://api.x.ai/v1/chat/completions", |_| {}
                ).await.map(|r| r.content).map_err(|e| e.to_string())
            } else {
                let messages = vec![crate::llm::Message { role: "user".to_string(), content: full_question.clone() }];
                crate::llm::xai::send_message(&api_key, XAI_MODEL, messages, Some(4096), None)
                    .await.map(|r| r.content).map_err(|e| e.to_string())
            }
        } else {
            // Local model - use llama.cpp
            if image_data.is_some() {
                return Err("Image attachments are not supported with local models.".to_string());
            }
            if !std::path::Path::new(model_backend).exists() {
                Err(format!("Local model file not found: {}", model_backend))
            } else {
                let model_path = model_backend.clone();
                let question = full_question.clone();

                // Call local model in blocking task (CPU-bound inference)
                let result = tokio::task::spawn_blocking(move || {
                    let messages = vec![crate::llm::Message {
                        role: "user".to_string(),
                        content: question,
                    }];

                    crate::llm::local_models::generate_with_local_model(
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
    let engine = crate::conversation_engine::ConversationEngine::new();
    let full_prompt = engine.build_prompt(&synthesis_prompt, None, None, true, None);

    // Call LLM directly without all the overhead
    let reply_text = if coordinator_model.to_lowercase().contains("anthropic") {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY not set".to_string())?;
        let model_name = ANTHROPIC_MODEL;
        let messages = vec![crate::llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];
        let response = crate::llm::anthropic::send_message(&api_key, model_name, messages, None, Some(4096), None)
            .await
            .map_err(|e| e.to_string())?;
        response.content
    } else if coordinator_model.to_lowercase().contains("xai") || coordinator_model.to_lowercase().contains("grok") {
        let api_key = std::env::var("XAI_API_KEY")
            .map_err(|_| "XAI_API_KEY not set".to_string())?;
        let model_name = XAI_MODEL;
        let messages = vec![crate::llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];
        let response = crate::llm::xai::send_message(&api_key, model_name, messages, Some(4096), None)
            .await
            .map_err(|e| e.to_string())?;
        response.content
    } else if coordinator_model.to_lowercase().contains("openai") || coordinator_model.to_lowercase().contains("gpt") {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY not set".to_string())?;
        let model_name = OPENAI_MODEL;
        let messages = vec![crate::llm::Message {
            role: "user".to_string(),
            content: full_prompt,
        }];
        let response = crate::llm::openai::send_message(&api_key, model_name, messages, Some(4096), None)
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

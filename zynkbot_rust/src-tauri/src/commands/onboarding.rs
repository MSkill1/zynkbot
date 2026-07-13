use crate::{insert_memory, extract_names_from_response, read_user_profile, write_user_profile, kb_rag, nlp_enhancer};

/// Store an onboarding response as a memory
#[tauri::command]
pub async fn store_onboarding_response(
    user_id: String,
    question_id: String,
    question: String,
    answer: String,
) -> Result<i32, String> {
    println!("[Rust] store_onboarding_response called - question_id: {}", question_id);

    let tags = match question_id.as_str() {
        "name_age" => vec!["onboarding".to_string(), "identity".to_string(), "personal".to_string()],
        "family" => vec!["onboarding".to_string(), "relationships".to_string(), "family".to_string()],
        "work" => vec!["onboarding".to_string(), "work".to_string(), "career".to_string()],
        "interests" => vec!["onboarding".to_string(), "interests".to_string(), "hobbies".to_string()],
        "goals" => vec!["onboarding".to_string(), "goals".to_string(), "aspirations".to_string()],
        "purpose" => vec!["onboarding".to_string(), "purpose".to_string(), "intentions".to_string()],
        _ => vec!["onboarding".to_string()],
    };

    let title = Some(format!("Onboarding: {}", question.split('.').next().unwrap_or(&question).trim()));

    if question_id == "name_age" {
        let (full_name, preferred_name, age) = extract_names_from_response(&answer);
        let mut profile = read_user_profile();
        if let Some(ref name) = full_name {
            profile["full_name"] = serde_json::Value::String(name.clone());
        }
        if let Some(ref name) = preferred_name {
            profile["preferred_name"] = serde_json::Value::String(name.clone());
        }
        if let Some(a) = age {
            profile["age"] = serde_json::Value::Number(serde_json::Number::from(a));
        }
        match write_user_profile(&profile) {
            Ok(_) => println!("[Onboarding] Saved name={:?} preferred={:?} age={:?}", full_name, preferred_name, age),
            Err(e) => eprintln!("[Onboarding] Failed to save user profile: {}", e),
        }
    }

    insert_memory(
        title,
        answer,
        Some("onboarding".to_string()),
        None,
        Some(user_id),
        Some(tags),
        "onboarding".to_string(),
        true,
        false,
        None,
        None,
    ).await
}

/// Complete onboarding and return a summary
#[tauri::command]
pub async fn complete_onboarding(
    user_id: String,
    responses: serde_json::Value,
) -> Result<String, String> {
    println!("[Rust] complete_onboarding called for user: {}", user_id);

    let mut summary_parts = Vec::new();

    if let Some(name_age) = responses.get("name_age").and_then(|v| v.as_str()) {
        summary_parts.push(format!("Your identity: {}", name_age));
    }
    if let Some(family) = responses.get("family").and_then(|v| v.as_str()) {
        summary_parts.push(format!("Your family: {}", family));
    }
    if let Some(relationships) = responses.get("relationships").and_then(|v| v.as_str()) {
        summary_parts.push(format!("Important relationships: {}", relationships));
    }
    if let Some(interests) = responses.get("interests").and_then(|v| v.as_str()) {
        summary_parts.push(format!("What you care about: {}", interests));
    }
    if let Some(goals) = responses.get("goals").and_then(|v| v.as_str()) {
        summary_parts.push(format!("Your goals: {}", goals));
    }
    if let Some(purpose) = responses.get("purpose").and_then(|v| v.as_str()) {
        summary_parts.push(format!("How I can help: {}", purpose));
    }

    let summary = summary_parts.join("\n\n");

    println!("[Rust] ✅ Onboarding complete - stored {} responses", summary_parts.len());

    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        println!("[Onboarding] 🔗 Starting relationship detection for onboarding memories...");

        let db_url = crate::db::get_db_url();
        let db_pool = match sqlx::SqlitePool::connect(&db_url).await {
            Ok(pool) => pool,
            Err(e) => {
                eprintln!("[Onboarding] ⚠️ Failed to connect to database: {}", e);
                return;
            }
        };

        let onboarding_memories: Vec<crate::memory::Memory> = match sqlx::query_as::<_, crate::memory::Memory>(
            "SELECT * FROM memories WHERE user_id = ? AND namespace = 'onboarding' ORDER BY created_at DESC LIMIT 10"
        )
        .bind(&user_id)
        .fetch_all(&db_pool)
        .await {
            Ok(memories) => memories,
            Err(e) => {
                eprintln!("[Onboarding] ⚠️ Failed to fetch onboarding memories: {}", e);
                db_pool.close().await;
                return;
            }
        };

        println!("[Onboarding] Found {} onboarding memories to analyze", onboarding_memories.len());

        let embedding_rows: Vec<(i32, Option<Vec<u8>>)> = sqlx::query_as(
            "SELECT id, embedding FROM memories WHERE user_id = ? AND namespace = 'onboarding'"
        )
        .bind(&user_id)
        .fetch_all(&db_pool)
        .await
        .unwrap_or_default();

        let embedding_map: std::collections::HashMap<i32, Vec<f32>> = embedding_rows
            .into_iter()
            .filter_map(|(id, blob)| blob.map(|b| (id, crate::memory::blob_to_f32_pub(&b))))
            .collect();

        for memory in &onboarding_memories {
            if memory.link_count > 0 {
                continue;
            }

            if let Some(embedding_vec) = embedding_map.get(&memory.id) {
                let embedding_vec = embedding_vec.clone();
                let similar_memories = match crate::memory::vector_search(
                    &db_pool,
                    embedding_vec.clone(),
                    Some(&user_id),
                    None,
                    None,
                    10
                ).await {
                    Ok(mems) => mems,
                    Err(e) => {
                        eprintln!("[Onboarding] ⚠️ Vector search failed for memory {}: {}", memory.id, e);
                        continue;
                    }
                };

                let candidates: Vec<&crate::memory::Memory> = similar_memories.iter()
                    .filter(|m| m.id != memory.id && m.similarity.unwrap_or(0.0) > 0.5)
                    .collect();

                if candidates.is_empty() {
                    continue;
                }

                println!("[Onboarding] Memory {} has {} relationship candidates (taking top 3)", memory.id, candidates.len());

                let mut created_count = 0;
                for candidate in candidates.iter().take(3) {
                    let link_result = sqlx::query(
                        "INSERT INTO memory_links (from_memory_id, to_memory_id, relationship_type, strength, created_at)
                         VALUES (?, ?, ?, ?, datetime('now'))
                         ON CONFLICT (from_memory_id, to_memory_id) DO NOTHING"
                    )
                    .bind(memory.id)
                    .bind(candidate.id)
                    .bind("related")
                    .bind(candidate.similarity.unwrap_or(0.7))
                    .execute(&db_pool)
                    .await;

                    if let Ok(result) = link_result {
                        if result.rows_affected() > 0 {
                            created_count += 1;
                            println!("[Onboarding] ✅ Created relationship: {} -> {} (similarity: {:.2})",
                                memory.id, candidate.id, candidate.similarity.unwrap_or(0.0));
                        }
                    }
                }

                if created_count > 0 {
                    let _ = sqlx::query(
                        "UPDATE memories SET link_count = link_count + ? WHERE id = ?"
                    )
                    .bind(created_count as i32)
                    .bind(memory.id)
                    .execute(&db_pool)
                    .await;

                    println!("[Onboarding] ✅ Updated link_count for memory {} (+{})", memory.id, created_count);
                }
            } else {
                println!("[Onboarding] ⚠️ Memory {} has no embedding, skipping", memory.id);
            }
        }

        db_pool.close().await;
        println!("[Onboarding] 🏁 Relationship detection complete");
    });

    Ok(summary)
}

/// Seed system memories about Zynkbot (called on first run)
#[tauri::command]
pub async fn seed_system_memories() -> Result<String, String> {
    println!("[System] 🌱 Seeding system memories...");

    let db_url = crate::db::get_db_url();
    let db_pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    let existing_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memories WHERE namespace = '_zynkbot'"
    )
    .fetch_one(&db_pool)
    .await
    .map_err(|e| format!("Failed to check existing system memories: {}", e))?;

    const EXPECTED_COUNT: i64 = 12;
    if existing_count >= EXPECTED_COUNT {
        println!("[System] ✅ System memories already exist ({}), skipping seed", existing_count);
        db_pool.close().await;
        return Ok(format!("System memories already exist ({} memories)", existing_count));
    }

    if existing_count > 0 {
        println!("[System] 🔄 System memories outdated ({}/{}), re-seeding...", existing_count, EXPECTED_COUNT);
        sqlx::query("DELETE FROM memories WHERE namespace = '_zynkbot'")
            .execute(&db_pool)
            .await
            .map_err(|e| format!("Failed to clear stale system memories: {}", e))?;
    }

    let system_memories = vec![
        (
            "Zynkbot Identity",
            "I am Zynkbot, a local-first AI companion with persistent memory. Unlike cloud-based chatbots, I run entirely on your device and store all conversations as memories that you can view, edit, or delete at any time. I believe in user privacy, data ownership, and transparent memory management."
        ),
        (
            "Memory System",
            "I use a hybrid memory system combining semantic vector search with entity-based matching, stored entirely in a local SQLite database. Every conversation is stored as a memory with embeddings, allowing me to recall relevant context from past discussions. I automatically detect relationships between memories and can identify contradictions, prompting you to resolve conflicts."
        ),
        (
            "Core Features",
            "My main features include: Memory Manager for viewing and editing all stored memories, Ensemble Mode for getting consensus from multiple AI models, Knowledge Base for document-based RAG retrieval, ZynkSync for syncing memories across your devices, and automatic contradiction detection to maintain consistency."
        ),
        (
            "Model Support",
            "I support multiple AI models including local models via llama.cpp (Llama, Qwen, Mistral running offline) and API models (Anthropic Claude, OpenAI GPT, xAI Grok). You can switch between models or use Ensemble Mode to get multiple perspectives on the same question."
        ),
        (
            "Privacy and Control",
            "Everything I know about you is stored locally in a SQLite database on your device — no cloud, no server, no account required. You have complete control: view all memories in Memory Manager, edit any memory to correct mistakes, delete memories you don't want me to remember, and export/backup your data at any time. I only know what you explicitly share with me."
        ),
        (
            "Ensemble Mode",
            "Ensemble Mode runs your question through multiple AI models simultaneously and synthesizes their responses. It detects when a question needs current information (like 'latest version of React') and automatically performs web searches. The coordinator identifies consensus, uncertainties, and disagreements, providing evidence-based answers while exposing where models differ."
        ),
        (
            "ZynkSync — Device Memory Sync",
            "ZynkSync syncs your memories across paired devices over your local network. Your laptop and desktop can share the same AI memory context automatically. Everything syncs over your home WiFi or office LAN — no cloud, no account, no data leaving your network. Pair devices through Settings → ZynkSync."
        ),
        (
            "ZChat — Device Messaging",
            "ZChat lets you send messages directly to other Zynkbot users you've linked to via ZynkLink. Exchange a pairing code with another user, then open the ZynkLink panel and click their name to start a chat. Messages travel directly between devices over your local network and are stored locally — no external servers involved."
        ),
        (
            "ZynkLink — File and Model Sharing",
            "ZynkLink lets you share files and AI models directly between paired devices without using the internet. Download a large language model once and share it to all your devices over your local network. Files transfer peer-to-peer with no cloud storage required. You can also download shared files directly into your Knowledge Base. Access it through Settings → ZynkLink."
        ),
        (
            "Containment Modes",
            "Zynkbot has five user-controlled containment modes: Guardian (default — blocks harmful content while allowing thoughtful conversation), Child (enhanced safety for minors with semantic filtering), Sovereign (model responses unfiltered, with warnings for potentially harmful content), Witness (no filtering — for research, debugging, and academic work), and HIPAA (healthcare compliance mode that prevents storage of protected health information and blocks diagnostic or dosing advice). You choose the mode — not a corporation. Switch modes in Settings."
        ),
        (
            "Snap-Ins",
            "Snap-ins are experimental professional tool modules with isolated data storage and specialized interfaces. The Therapist snap-in is a proof of concept featuring patient session tracking and structured note entry with HIPAA-mode integration. Snap-ins are designed for extensibility and can be built for medical, legal, research, or any professional workflow. Data stored in a snap-in is kept separate from your main memory. Access them through Settings → Snap-Ins."
        ),
        (
            "Voice Input",
            "Zynkbot is designed to support voice input using Whisper, an on-device speech recognition model that transcribes locally with no audio sent to any external server. Voice input is currently disabled pending resolution of a build conflict between Whisper and the local LLM runtime. It will be re-enabled in a future release."
        ),
    ];

    println!("[System] Creating {} system memories...", system_memories.len());

    let mut created_count = 0;
    for (title, content) in &system_memories {
        match insert_memory(
            Some(title.to_string()),
            content.to_string(),
            Some("system".to_string()),
            None,
            Some("system".to_string()),
            Some(vec!["system".to_string(), "zynkbot".to_string(), "identity".to_string()]),
            "_zynkbot".to_string(),
            false,
            false,
            None,
            None,
        ).await {
            Ok(memory_id) => {
                println!("[System] ✅ Created system memory: {} (id: {})", title, memory_id);
                created_count += 1;
            }
            Err(e) => {
                eprintln!("[System] ⚠️ Failed to create system memory '{}': {}", title, e);
            }
        }
    }

    db_pool.close().await;

    println!("[System] 🏁 System memory seeding complete - created {}/{} memories", created_count, system_memories.len());

    Ok(format!("Created {} system memories", created_count))
}

/// Index system documentation into Knowledge Base
#[tauri::command]
pub async fn index_system_documentation() -> Result<String, String> {
    println!("[System KB] Starting system documentation indexing...");

    let db_url = crate::db::get_db_url();
    let db_pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    let existing_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM kb_documents WHERE user_id = '_system'"
    )
    .fetch_one(&db_pool)
    .await
    .map_err(|e| format!("Failed to check existing system docs: {}", e))?;

    if existing_count > 0 {
        println!("[System KB] ✅ System documentation already indexed ({}), skipping", existing_count);
        db_pool.close().await;
        return Ok(format!("System documentation already indexed ({} documents)", existing_count));
    }

    let current_exe = std::env::current_exe()
        .map_err(|e| format!("Failed to get executable path: {}", e))?;

    let mut kb_path = current_exe
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .ok_or("Failed to determine project root")?
        .join("system_docs")
        .join("_system");

    if !kb_path.exists() {
        println!("[System KB] Dev path not found, trying current directory...");
        kb_path = std::env::current_dir()
            .map_err(|e| format!("Failed to get current dir: {}", e))?
            .join("system_docs")
            .join("_system");
    }

    if !kb_path.exists() {
        println!("[System KB] Trying ../system_docs/_system from current dir...");
        let current = std::env::current_dir()
            .map_err(|e| format!("Failed to get current dir: {}", e))?;
        if let Some(parent) = current.parent() {
            kb_path = parent.join("system_docs").join("_system");
        }
    }

    if !kb_path.exists() {
        eprintln!("[System KB] ⚠️ System documentation directory not found — skipping (installed binary without bundled docs)");
        db_pool.close().await;
        return Ok("System documentation not available in this installation".to_string());
    }

    println!("[System KB] Found system docs directory: {:?}", kb_path);

    let system_docs = vec![
        "01_zynkbot_overview.md",
        "02_memory_system.md",
        "03_features.md",
        "04_technical_architecture.md",
        "05_installation_setup.md",
        "06_faq.md",
    ];

    let mut indexed_count = 0;
    let mut errors = Vec::new();

    for doc_file in &system_docs {
        let doc_path = kb_path.join(doc_file);
        let doc_path_str = doc_path.to_str()
            .ok_or_else(|| format!("Invalid path for {}", doc_file))?;

        println!("[System KB] Indexing: {}", doc_file);

        match kb_rag::index_document(&db_pool, "_system", doc_path_str, None).await {
            Ok(doc_id) => {
                println!("[System KB] ✅ Indexed {} (doc_id: {})", doc_file, doc_id);
                indexed_count += 1;
            }
            Err(e) => {
                let error = format!("Failed to index {}: {}", doc_file, e);
                eprintln!("[System KB] ⚠️ {}", error);
                errors.push(error);
            }
        }
    }

    db_pool.close().await;

    let result = if errors.is_empty() {
        format!("✅ Indexed {}/{} system documentation files", indexed_count, system_docs.len())
    } else {
        format!("⚠️ Indexed {}/{} files. Errors: {}", indexed_count, system_docs.len(), errors.join("; "))
    };

    println!("[System KB] 🏁 {}", result);
    Ok(result)
}

/// Apply the pre-computed Einstein seed (first install + reinstall)
#[tauri::command]
pub async fn apply_einstein_seed(user_id: String) -> Result<serde_json::Value, String> {
    println!("[Einstein] Applying pre-computed Einstein seed for user: {}", user_id);

    let seed_sql = include_str!("../../seeds/einstein_seed.sql");
    let sql = seed_sql
        .replace("ZYNKBOT_SEED_USER_ID", &user_id)
        .replace("'::vector", "'")
        .replace("'::jsonb", "'");

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let statements: Vec<&str> = sql.split(";\n").collect();
    let mut memories_loaded = 0;
    let mut relationships_created = 0;

    for stmt in &statements {
        let sql_lines: String = stmt.lines()
            .filter(|l| !l.trim_start().starts_with("--"))
            .collect::<Vec<_>>()
            .join("\n");
        let trimmed = sql_lines.trim();
        if trimmed.is_empty() {
            continue;
        }
        let full_stmt = format!("{};", trimmed);
        match sqlx::query(&full_stmt).execute(&pool).await {
            Ok(result) => {
                let rows = result.rows_affected();
                if full_stmt.contains("INSERT INTO memories") { memories_loaded += rows; }
                else if full_stmt.contains("INSERT INTO memory_links") { relationships_created += rows; }
            }
            Err(e) => println!("[Einstein] ⚠️ Statement error (skipping): {}", e),
        }
    }

    let emb_rows: Vec<(i32, Option<Vec<u8>>)> = sqlx::query_as::<_, (i32, Option<Vec<u8>>)>(
        "SELECT id, embedding FROM memories WHERE session_id = 'einstein-demo-session' AND user_id = ?"
    )
    .bind(&user_id)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();

    let mut converted = 0usize;
    for (id, raw) in emb_rows {
        if let Some(bytes) = raw {
            if let Ok(text) = std::str::from_utf8(&bytes) {
                if text.starts_with('[') {
                    if let Ok(floats) = serde_json::from_str::<Vec<f32>>(text) {
                        let blob: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
                        let _ = sqlx::query("UPDATE memories SET embedding = ? WHERE id = ?")
                            .bind(&blob)
                            .bind(id)
                            .execute(&pool)
                            .await;
                        converted += 1;
                    }
                }
            }
        }
    }
    if converted > 0 {
        println!("[Einstein] Converted {} text embeddings to BLOB format", converted);
    }

    pool.close().await;

    let profile_path = crate::db::get_user_profile_path();
    let profile = serde_json::json!({
        "full_name": "Albert Einstein",
        "preferred_name": "Albert",
        "demo_persona": true
    });
    if let Ok(json) = serde_json::to_string_pretty(&profile) {
        std::fs::write(&profile_path, json).ok();
        println!("[Einstein] Wrote user_profile.json with Einstein persona");
    }

    println!("[Einstein] ✅ Applied seed: {} memories, {} relationships", memories_loaded, relationships_created);

    Ok(serde_json::json!({
        "success": true,
        "loaded_count": memories_loaded,
        "relationships_created": relationships_created,
        "message": format!("Einstein restored: {} memories, {} relationships", memories_loaded, relationships_created)
    }))
}

/// Load small Einstein demo (10 memories) for fast testing
#[tauri::command]
pub async fn load_small_einstein_demo(user_id: String) -> Result<serde_json::Value, String> {
    println!("[Rust] load_small_einstein_demo called for user: {} (10 memories only)", user_id);

    let einstein_memories = vec![
        ("My Birth", "I was born on March 14, 1879, in Ulm, Germany. My parents were Hermann Einstein and Pauline Koch.", "biography", vec!["birth", "family"]),
        ("The Compass", "When I was 5, my father showed me a compass. The invisible force fascinated me.", "biography", vec!["childhood", "physics"]),
        ("Learning Violin", "My mother made sure I learned violin from age 6. I named my violin 'Lina'.", "personal", vec!["music", "violin"]),
        ("Patent Office", "From 1902-1909, I worked as a patent examiner in Bern, Switzerland.", "career", vec!["employment", "patent_office"]),
        ("Marriage to Mileva", "I married Mileva Marić in January 1903. We had three children together.", "personal", vec!["marriage", "family"]),
        ("My Miracle Year", "In 1905, I published four groundbreaking papers that revolutionized physics.", "science", vec!["1905", "breakthrough"]),
        ("E=mc² Equation", "I derived E=mc², showing that energy and mass are interchangeable.", "science", vec!["energy", "formula"]),
        ("General Relativity", "Between 1907-1915, I developed general relativity, describing gravity as spacetime curvature.", "science", vec!["gravity", "relativity"]),
        ("I had three children", "Lieserl (whose fate remains unknown), Hans Albert (who became a professor), and Eduard (who struggled with schizophrenia).", "personal", vec!["children", "family"]),
        ("Zionism Support", "I support Zionism and was offered the presidency of Israel in 1952, which I declined.", "politics", vec!["israel", "zionism"]),
    ];

    println!("[Rust] Connecting to database...");
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let session_id = "einstein-demo-small-session";
    sqlx::query("DELETE FROM memories WHERE session_id = ?")
        .bind(session_id)
        .execute(&pool)
        .await
        .map_err(|e| format!("Failed to delete old memories: {}", e))?;

    let total_memories = einstein_memories.len();
    println!("[Rust] Generating embeddings for {} memories (batch)...", total_memories);

    let contents: Vec<String> = einstein_memories
        .iter()
        .map(|(_, content, _, _)| content.to_string())
        .collect();

    let contents_for_embeddings = contents.clone();
    let embeddings = tokio::task::spawn_blocking(move || {
        crate::llm::local_embeddings::generate_local_embeddings_batch(contents_for_embeddings, None)
    })
    .await
    .map_err(|e| format!("Failed to run batch embedding task: {}", e))?
    .map_err(|e| format!("Batch embedding generation failed: {}", e))?;

    println!("[Rust] ✅ Generated {} embeddings!", embeddings.len());

    println!("[Rust] Extracting entities for all memories (batch)...");
    let all_contents_for_entities = contents;
    let all_entities = tokio::task::spawn_blocking(move || {
        let enhancer = nlp_enhancer::NLPEnhancer::new();
        enhancer.extract_entities_batch(&all_contents_for_entities)
            .into_iter()
            .map(|entities| {
                serde_json::json!(entities.iter().map(|e| {
                    serde_json::json!({
                        "word": e.word,
                        "label": e.label,
                        "score": e.score,
                        "start": e.start,
                        "end": e.end
                    })
                }).collect::<Vec<_>>())
            })
            .collect::<Vec<_>>()
    })
    .await
    .map_err(|e| format!("Failed to extract entities: {}", e))?;

    println!("[Rust] ✅ Extracted entities!");

    println!("[Rust] Detecting events for all memories (batch)...");
    let all_contents_for_events: Vec<String> = einstein_memories
        .iter()
        .map(|(_, content, _, _)| content.to_string())
        .collect();

    let all_events = tokio::task::spawn_blocking(move || {
        let enhancer = nlp_enhancer::NLPEnhancer::new();
        all_contents_for_events
            .iter()
            .map(|content| enhancer.detect_event(content))
            .collect::<Vec<_>>()
    })
    .await
    .map_err(|e| format!("Failed to detect events: {}", e))?;

    println!("[Rust] ✅ Detected events! Inserting into database...");

    let mut memory_ids = Vec::new();
    for (idx, ((((title, content, namespace, tags), embedding), entities), (event_type, event_date))) in
        einstein_memories.into_iter()
            .zip(embeddings.into_iter())
            .zip(all_entities.into_iter())
            .zip(all_events.into_iter())
            .enumerate() {

        println!("[Rust] Inserting {}/{} - {} (event: {:?})",
            idx + 1, total_memories, title, event_type);

        let embedding_vec = embedding.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
        let _tags_vec: Vec<String> = tags.iter().map(|s| s.to_string()).collect();

        let memory_id = sqlx::query_scalar::<_, i32>(
            "INSERT INTO memories (
                title, content, namespace, user_id, session_id,
                embedding, source_type, is_syncable, is_shareable,
                entities_detected, event_type, event_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id"
        )
        .bind(title)
        .bind(content)
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
        .map_err(|e| format!("Failed to insert memory: {}", e))?;

        memory_ids.push(memory_id);
    }

    println!("[Rust] ✅ Loaded {} small Einstein demo memories!", total_memories);

    Ok(serde_json::json!({
        "success": true,
        "message": format!("Successfully loaded {} Einstein demo memories (small test set)", total_memories),
        "memory_ids": memory_ids
    }))
}

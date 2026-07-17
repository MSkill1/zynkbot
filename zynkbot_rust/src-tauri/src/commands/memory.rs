use crate::Memory;
use crate::memory;

/// List memories with optional filters
#[tauri::command]
pub async fn list_memories(
    user_id: Option<String>,
    session_id: Option<String>,
    namespace: Option<String>,
    event_type: Option<String>,
    date_from: Option<String>,
    date_to: Option<String>,
) -> Result<Vec<Memory>, String> {
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let mut sql = String::from(
        "SELECT
            id, title, content, source_type, session_id, created_at, updated_at,
            parent_scroll_id, chunk_index, user_id, namespace,
            is_syncable, is_shareable,
            event_type, event_date,
            link_count, is_ephemeral, expires_at,
            entities_detected, original_text,
            NULL as similarity
        FROM memories
        WHERE TRUE
        AND namespace != '_zynkbot'
        AND user_id != 'system'"
    );

    if user_id.is_some() { sql.push_str(" AND user_id = ?"); }
    if session_id.is_some() { sql.push_str(" AND session_id = ?"); }
    if namespace.is_some() { sql.push_str(" AND namespace = ?"); }
    if event_type.is_some() { sql.push_str(" AND event_type = ?"); }
    if date_from.is_some() { sql.push_str(" AND created_at >= ?"); }
    if date_to.is_some() { sql.push_str(" AND created_at <= ?"); }
    sql.push_str(" ORDER BY datetime(created_at) DESC");

    let mut query = sqlx::query_as::<_, memory::Memory>(&sql);
    if let Some(uid) = user_id.as_ref() { query = query.bind(uid); }
    if let Some(sid) = session_id.as_ref() { query = query.bind(sid); }
    if let Some(ns) = namespace.as_ref() { query = query.bind(ns); }
    if let Some(event) = event_type.as_ref() { query = query.bind(event); }
    if let Some(from) = date_from.as_ref() { query = query.bind(from); }
    if let Some(to) = date_to.as_ref() { query = query.bind(to); }

    let db_memories = query
        .fetch_all(&pool)
        .await
        .map_err(|e| format!("Failed to query memories: {}", e))?;

    pool.close().await;

    let memories: Vec<Memory> = db_memories
        .iter()
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
            similarity: None,
            event_type: mem.event_type.clone(),
            event_date: mem.event_date.map(|dt| dt.to_string()),
            link_count: Some(mem.link_count),
            is_ephemeral: Some(mem.is_ephemeral),
            expires_at: mem.expires_at.map(|dt| dt.to_string()),
            entities_detected: mem.entities_detected.clone(),
            original_text: mem.original_text.clone(),
        })
        .collect();

    Ok(memories)
}

/// Update a memory — regenerates embedding if content changed
#[tauri::command]
pub async fn update_memory(
    memory_id: i32,
    title: Option<String>,
    content: Option<String>,
    namespace: Option<String>,
) -> Result<bool, String> {
    println!("[Rust] update_memory called for ID: {}", memory_id);

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let namespace_check: Option<String> = sqlx::query_scalar(
        "SELECT namespace FROM memories WHERE id = ?"
    )
    .bind(memory_id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| format!("Failed to check memory namespace: {}", e))?;

    if namespace_check == Some("_zynkbot".to_string()) {
        pool.close().await;
        return Err("Cannot edit system memories. These are core Zynkbot identity memories.".to_string());
    }

    let embedding_vec = if let Some(ref new_content) = content {
        println!("[Rust] Content updated, regenerating embedding...");
        let content_clone = new_content.clone();
        let embedding = tokio::task::spawn_blocking(move || {
            crate::llm::local_embeddings::generate_local_embedding(&content_clone)
        })
        .await
        .map_err(|e| format!("Failed to run embedding task: {}", e))?
        .map_err(|e| format!("Failed to generate embedding: {}", e))?;

        Some(embedding.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
    } else {
        None
    };

    let result = if let Some(emb) = embedding_vec {
        sqlx::query(
            "UPDATE memories
            SET title = COALESCE(?, title),
                content = COALESCE(?, content),
                original_text = COALESCE(?, original_text),
                namespace = COALESCE(?, namespace),
                embedding = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?"
        )
        .bind(title.as_deref())
        .bind(content.as_deref())
        .bind(content.as_deref())
        .bind(namespace.as_deref())
        .bind(&emb)
        .bind(memory_id)
        .execute(&pool)
        .await
    } else {
        sqlx::query(
            "UPDATE memories
            SET title = COALESCE(?, title),
                content = COALESCE(?, content),
                original_text = COALESCE(?, original_text),
                namespace = COALESCE(?, namespace),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?"
        )
        .bind(title.as_deref())
        .bind(content.as_deref())
        .bind(content.as_deref())
        .bind(namespace.as_deref())
        .bind(memory_id)
        .execute(&pool)
        .await
    };

    pool.close().await;

    match result {
        Ok(r) => {
            println!("[Rust] Updated {} row(s)", r.rows_affected());
            Ok(r.rows_affected() > 0)
        }
        Err(e) => Err(format!("Failed to update memory: {}", e)),
    }
}

/// Delete a memory and propagate deletion to synced devices
#[tauri::command]
pub async fn delete_memory(memory_id: i32) -> Result<bool, String> {
    println!("[Rust] delete_memory called for ID: {}", memory_id);

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let namespace: Option<String> = sqlx::query_scalar(
        "SELECT namespace FROM memories WHERE id = ?"
    )
    .bind(memory_id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| format!("Failed to check memory namespace: {}", e))?;

    if namespace == Some("_zynkbot".to_string()) {
        pool.close().await;
        return Err("Cannot delete system memories. These are core Zynkbot identity memories.".to_string());
    }

    let content_hash: Option<String> = sqlx::query_scalar::<_, String>(
        "SELECT content FROM memories WHERE id = ?"
    )
    .bind(memory_id)
    .fetch_optional(&pool)
    .await
    .ok()
    .flatten()
    .map(|content| {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(content.as_bytes()))
    });

    if let Some(ref hash) = content_hash {
        println!("[Rust] Memory content hash: {}", hash);
    }

    let result = sqlx::query("DELETE FROM memories WHERE id = ?")
        .bind(memory_id)
        .execute(&pool)
        .await
        .map_err(|e| format!("Failed to delete memory: {}", e))?;

    pool.close().await;

    let success = result.rows_affected() > 0;
    println!("[Rust] Deleted {} row(s)", result.rows_affected());

    if success {
        if let Some(hash) = content_hash {
            let zynksync_service = crate::ZYNKSYNC_SERVICE.lock().await;
            if let Some(service) = zynksync_service.as_ref() {
                // Always record tombstone so future syncs can't resurrect this memory,
                // regardless of whether auto-sync is currently enabled.
                let _ = service.record_tombstones(&[hash.clone()]).await;
                if service.is_auto_sync_enabled().await {
                    match service.propagate_deletion_by_hash(hash).await {
                        Ok(count) => println!("[Rust] ✓ Deletion synced to {} device(s)", count),
                        Err(e) => eprintln!("[Rust] ⚠ Warning: Failed to sync deletion: {}", e),
                    }
                } else {
                    println!("[Rust] Auto-sync disabled - tombstone recorded, deletion not propagated");
                }
            } else {
                println!("[Rust] ⚠ ZynkSync not running - deletion not propagated to other devices");
            }
        } else {
            println!("[Rust] ⚠ Could not get content hash - deletion not propagated to other devices");
        }
    }

    Ok(success)
}

/// Get a single memory by ID
#[tauri::command]
pub async fn get_memory(memory_id: i32) -> Result<Option<Memory>, String> {
    println!("[Rust] get_memory called for ID: {}", memory_id);

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let db_memory = sqlx::query_as::<_, memory::Memory>(
        "SELECT
            id, title, content, source_type, session_id, created_at, updated_at,
            parent_scroll_id, chunk_index, user_id, namespace,
            is_syncable, is_shareable,
            event_type, event_date,
            link_count, is_ephemeral, expires_at,
            entities_detected, original_text,
            NULL as similarity
        FROM memories
        WHERE id = ?"
    )
    .bind(memory_id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| format!("Failed to query memory: {}", e))?;

    pool.close().await;

    let memory = db_memory.map(|mem| Memory {
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
        similarity: None,
        event_type: mem.event_type.clone(),
        event_date: mem.event_date.map(|dt| dt.to_string()),
        link_count: Some(mem.link_count),
        is_ephemeral: Some(mem.is_ephemeral),
        expires_at: mem.expires_at.map(|dt| dt.to_string()),
        entities_detected: mem.entities_detected.clone(),
        original_text: mem.original_text.clone(),
    });

    Ok(memory)
}

/// Get memory links with full memory details
#[tauri::command]
pub async fn get_memory_links(memory_id: i32) -> Result<serde_json::Value, String> {
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    #[derive(sqlx::FromRow)]
    struct LinkWithMemory {
        id: i32,
        source_memory_id: i32,
        target_memory_id: i32,
        relation_type: String,
        confidence: f32,
        created_at: chrono::DateTime<chrono::Utc>,
        notes: Option<String>,
        created_by: String,
        related_memory_id: i32,
        related_memory_title: Option<String>,
        related_memory_content: String,
    }

    let query = "
        SELECT
            ml.id,
            ml.source_memory_id,
            ml.target_memory_id,
            ml.relation_type,
            ml.confidence,
            ml.created_at,
            ml.notes,
            ml.created_by,
            CASE
                WHEN ml.source_memory_id = ? THEN ml.target_memory_id
                ELSE ml.source_memory_id
            END as related_memory_id,
            CASE
                WHEN ml.source_memory_id = ? THEN m_target.title
                ELSE m_source.title
            END as related_memory_title,
            CASE
                WHEN ml.source_memory_id = ? THEN m_target.content
                ELSE m_source.content
            END as related_memory_content
        FROM memory_links ml
        LEFT JOIN memories m_source ON ml.source_memory_id = m_source.id
        LEFT JOIN memories m_target ON ml.target_memory_id = m_target.id
        WHERE ml.source_memory_id = ? OR ml.target_memory_id = ?
        ORDER BY ml.created_at DESC
    ";

    let rows = sqlx::query_as::<_, LinkWithMemory>(query)
        .bind(memory_id)
        .bind(memory_id)
        .bind(memory_id)
        .bind(memory_id)
        .bind(memory_id)
        .fetch_all(&pool)
        .await
        .map_err(|e| format!("Failed to query memory links: {}", e))?;

    pool.close().await;

    let links: Vec<serde_json::Value> = rows.iter().map(|row| {
        serde_json::json!({
            "id": row.id,
            "source_memory_id": row.source_memory_id,
            "target_memory_id": row.target_memory_id,
            "relation_type": row.relation_type,
            "confidence": row.confidence,
            "created_at": row.created_at.format("%Y-%m-%dT%H:%M:%S%.fZ").to_string(),
            "notes": row.notes,
            "created_by": row.created_by,
            "related_memory_id": row.related_memory_id,
            "related_memory": {
                "id": row.related_memory_id,
                "title": row.related_memory_title,
                "content": row.related_memory_content
            }
        })
    }).collect();

    Ok(serde_json::json!({
        "links": links,
        "count": links.len()
    }))
}

/// Get the graph of related memories to a given depth
#[tauri::command]
pub async fn get_memory_graph(memory_id: i32, depth: Option<i32>) -> Result<serde_json::Value, String> {
    println!("[Rust] get_memory_graph called for memory_id: {}, depth: {:?}", memory_id, depth);

    let depth_param = depth.unwrap_or(1);
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let graph_data = memory::get_memory_graph(&pool, memory_id, depth_param)
        .await
        .map_err(|e| format!("Failed to get memory graph: {}", e))?;

    pool.close().await;

    println!("[Rust] Found {} related memories in graph", graph_data.len());

    let nodes: Vec<serde_json::Value> = graph_data.iter().map(|(rel, mem)| {
        serde_json::json!({
            "memory_id": mem.id,
            "title": mem.title,
            "content": mem.content,
            "namespace": mem.namespace,
            "relation_type": rel.relation_type,
            "confidence": rel.confidence,
            "direction": rel.direction,
        })
    }).collect();

    Ok(serde_json::json!({
        "center_memory_id": memory_id,
        "depth": depth_param,
        "nodes": nodes,
        "count": nodes.len()
    }))
}

/// Create a relationship link between two memories
#[tauri::command]
pub async fn create_memory_link(
    source_memory_id: i32,
    target_memory_id: i32,
    relation_type: String,
    confidence: Option<f64>,
    notes: Option<String>,
) -> Result<serde_json::Value, String> {
    println!("[Rust] create_memory_link called: {} -> {} ({})",
             source_memory_id, target_memory_id, relation_type);

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let link_id = memory::create_memory_link(
        &pool,
        source_memory_id,
        target_memory_id,
        &relation_type,
        confidence.unwrap_or(0.8) as f32,
        notes.as_deref(),
        "user"
    )
    .await
    .map_err(|e| format!("Failed to create memory link: {}", e))?;

    pool.close().await;

    println!("[Rust] Successfully created memory link with ID: {}", link_id);

    Ok(serde_json::json!({
        "success": true,
        "link_id": link_id,
        "source_memory_id": source_memory_id,
        "target_memory_id": target_memory_id,
        "relation_type": relation_type
    }))
}

/// Delete a memory link by ID
#[tauri::command]
pub async fn delete_memory_link(link_id: i32) -> Result<serde_json::Value, String> {
    println!("[Rust] delete_memory_link called for link_id: {}", link_id);

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let rows_affected = memory::delete_memory_link(&pool, link_id)
        .await
        .map_err(|e| format!("Failed to delete memory link: {}", e))?;

    pool.close().await;

    println!("[Rust] Deleted {} link(s)", rows_affected);

    Ok(serde_json::json!({
        "success": rows_affected > 0,
        "rows_affected": rows_affected
    }))
}

/// Get contradicting memories for a given memory
#[tauri::command]
pub async fn get_memory_contradictions(memory_id: i32) -> Result<serde_json::Value, String> {
    println!("[Rust] get_memory_contradictions called for memory_id: {}", memory_id);

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let contradictions = sqlx::query_as::<_, memory::MemoryLink>(
        "SELECT id, source_memory_id, target_memory_id, relation_type,
                confidence, created_at, notes, created_by
         FROM memory_links
         WHERE relation_type = 'contradicts'
         AND (source_memory_id = ? OR target_memory_id = ?)
         ORDER BY confidence DESC"
    )
    .bind(memory_id)
    .bind(memory_id)
    .fetch_all(&pool)
    .await
    .map_err(|e| format!("Failed to query contradictions: {}", e))?;

    pool.close().await;

    println!("[Rust] Found {} contradictions", contradictions.len());

    Ok(serde_json::json!({
        "memory_id": memory_id,
        "contradictions": contradictions,
        "count": contradictions.len()
    }))
}

/// List namespaces with memory counts
#[tauri::command]
pub async fn get_namespaces(user_id: Option<String>) -> Result<serde_json::Value, String> {
    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let query = if let Some(ref uid) = user_id {
        sqlx::query_as::<_, (String, i64)>(
            "SELECT namespace, COUNT(*) as count
             FROM memories
             WHERE user_id = ?
             GROUP BY namespace
             ORDER BY count DESC"
        )
        .bind(uid)
    } else {
        sqlx::query_as::<_, (String, i64)>(
            "SELECT namespace, COUNT(*) as count
             FROM memories
             GROUP BY namespace
             ORDER BY count DESC"
        )
    };

    let namespaces = query
        .fetch_all(&pool)
        .await
        .map_err(|e| format!("Failed to query namespaces: {}", e))?;

    pool.close().await;

    let namespace_list: Vec<serde_json::Value> = namespaces
        .iter()
        .map(|(ns, count)| serde_json::json!({"namespace": ns, "count": count}))
        .collect();

    Ok(serde_json::json!({
        "namespaces": namespace_list,
        "total": namespace_list.len()
    }))
}

/// Update an existing memory link
#[tauri::command]
pub async fn update_memory_link(
    link_id: i32,
    relation_type: Option<String>,
    strength: Option<f32>,
    notes: Option<String>,
) -> Result<serde_json::Value, String> {
    println!("[Rust] update_memory_link called for link_id: {}", link_id);

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let result = sqlx::query(
        "UPDATE memory_links
        SET relation_type = COALESCE(?, relation_type),
            confidence = COALESCE(?, confidence),
            notes = COALESCE(?, notes)
        WHERE id = ?"
    )
    .bind(relation_type.as_deref())
    .bind(strength)
    .bind(notes.as_deref())
    .bind(link_id)
    .execute(&pool)
    .await
    .map_err(|e| format!("Failed to update memory link: {}", e))?;

    pool.close().await;

    println!("[Rust] Updated {} link(s)", result.rows_affected());

    Ok(serde_json::json!({
        "success": result.rows_affected() > 0,
        "rows_affected": result.rows_affected()
    }))
}

/// Get the full memory graph for a user (all nodes and edges)
#[tauri::command]
pub async fn get_full_memory_graph(
    user_id: Option<String>,
    namespace: Option<String>,
) -> Result<serde_json::Value, String> {
    println!("[Rust] get_full_memory_graph called - user_id: {:?}, namespace: {:?}", user_id, namespace);

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let mut sql = String::from(
        "SELECT id, title, content, namespace FROM memories WHERE TRUE"
    );
    if user_id.is_some() { sql.push_str(" AND user_id = ?"); }
    if namespace.is_some() { sql.push_str(" AND namespace = ?"); }

    #[derive(sqlx::FromRow)]
    struct MemoryNode {
        id: i32,
        title: Option<String>,
        content: String,
        namespace: String,
    }

    let mut query = sqlx::query_as::<_, MemoryNode>(&sql);
    if let Some(ref uid) = user_id { query = query.bind(uid); }
    if let Some(ref ns) = namespace { query = query.bind(ns); }

    let memories = query
        .fetch_all(&pool)
        .await
        .map_err(|e| format!("Failed to query memories: {}", e))?;

    let memory_ids: Vec<i32> = memories.iter().map(|m| m.id).collect();

    let links = if memory_ids.is_empty() {
        Vec::new()
    } else {
        let in_clause = memory_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        let sql = format!(
            "SELECT ml.id, ml.source_memory_id, ml.target_memory_id,
                    ml.relation_type, ml.confidence, ml.created_at,
                    ml.notes, ml.created_by
             FROM memory_links ml
             WHERE ml.source_memory_id IN ({}) AND ml.target_memory_id IN ({})",
            in_clause, in_clause
        );
        let mut q = sqlx::query_as::<_, memory::MemoryLink>(&sql);
        for id in &memory_ids { q = q.bind(id); }
        for id in &memory_ids { q = q.bind(id); }
        q.fetch_all(&pool)
            .await
            .map_err(|e| format!("Failed to query memory links: {}", e))?
    };

    pool.close().await;

    println!("[Rust] Found {} memories and {} links", memories.len(), links.len());

    let nodes: Vec<serde_json::Value> = memories
        .iter()
        .map(|m| serde_json::json!({
            "id": m.id,
            "title": m.title,
            "content": m.content,
            "namespace": m.namespace
        }))
        .collect();

    let edges: Vec<serde_json::Value> = links
        .iter()
        .map(|l| serde_json::json!({
            "id": l.id,
            "source_memory_id": l.source_memory_id,
            "target_memory_id": l.target_memory_id,
            "relation_type": l.relation_type,
            "confidence": l.confidence
        }))
        .collect();

    Ok(serde_json::json!({
        "memories": nodes,
        "links": edges,
        "memory_count": nodes.len(),
        "link_count": edges.len()
    }))
}

/// Delete expired ephemeral memories (HIPAA compliance)
#[tauri::command]
pub async fn cleanup_expired_memories() -> Result<serde_json::Value, String> {
    println!("[Rust] cleanup_expired_memories called");

    let db_url = crate::db::get_db_url();
    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let result = sqlx::query(
        "DELETE FROM memories
        WHERE is_ephemeral = 1
        AND expires_at IS NOT NULL
        AND expires_at < CURRENT_TIMESTAMP"
    )
    .execute(&pool)
    .await
    .map_err(|e| format!("Failed to cleanup expired memories: {}", e))?;

    pool.close().await;

    let deleted_count = result.rows_affected();
    println!("[Rust] Cleaned up {} expired memories", deleted_count);

    Ok(serde_json::json!({
        "success": true,
        "deleted_count": deleted_count,
        "message": format!("Deleted {} expired ephemeral memories", deleted_count)
    }))
}

/// Pre-check memory for duplicates using hybrid search + pure cosine similarity
#[tauri::command]
pub async fn pre_check_memory(
    content: String,
    user_id: String,
    exclude_memory_id: i32,
) -> Result<serde_json::Value, String> {
    println!("[Rust] 🔍 Smart duplicate check - extracting keywords from input...");

    let content_for_keywords = content.clone();
    let query_entities = tokio::task::spawn_blocking(move || {
        use crate::nlp_enhancer::NLPEnhancer;
        let enhancer = NLPEnhancer::new();
        enhancer.extract_keywords(&content_for_keywords)
    })
    .await
    .map_err(|e| format!("Failed to extract keywords: {}", e))?;

    println!("[Rust] Extracted {} keywords for duplicate check: {:?}", query_entities.len(), query_entities);

    let content_clone = content.clone();
    let query_embedding = tokio::task::spawn_blocking(move || {
        crate::llm::local_embeddings::generate_local_embedding(&content_clone)
    })
    .await
    .map_err(|e| format!("Failed to run embedding task: {}", e))?
    .map_err(|e| format!("Failed to generate embedding: {}", e))?;

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    let _query_entities_clone = query_entities.clone();
    let query_embedding_clone = query_embedding.clone();

    let similar_memories = memory::hybrid_search(
        &pool,
        query_embedding,
        query_entities,
        Some(&user_id),
        None,
        None,
        10,
    )
    .await
    .map_err(|e| format!("Failed to search for similar memories: {}", e))?;

    let candidates: Vec<crate::memory::Memory> = similar_memories
        .into_iter()
        .filter(|m| m.similarity.unwrap_or(0.0) > 0.35 && m.id != exclude_memory_id)
        .collect();

    println!("[Rust] Found {} candidates (>35%) for duplicate check", candidates.len());

    if candidates.is_empty() {
        return Ok(serde_json::json!({
            "has_duplicate": false,
            "has_contradiction": false,
            "count": 0
        }));
    }

    println!("[Rust] 🔍 Checking for duplicates using pure cosine similarity...");
    for candidate in &candidates {
        let candidate_embedding_query = sqlx::query_as::<_, (Vec<u8>,)>(
            "SELECT embedding FROM memories WHERE id = ?"
        )
        .bind(candidate.id)
        .fetch_one(&pool)
        .await;

        if let Ok((candidate_blob,)) = candidate_embedding_query {
            let candidate_embedding_vec = crate::blob_to_f32(&candidate_blob);

            let pure_similarity = crate::llm::local_embeddings::cosine_similarity(
                &query_embedding_clone,
                &candidate_embedding_vec
            );

            let hybrid_score = candidate.similarity.unwrap_or(0.0);
            println!("[Rust]   Memory {} - Pure cosine: {:.3}, Hybrid score: {:.3}",
                     candidate.id, pure_similarity, hybrid_score);

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

    println!("[Rust] ✅ No duplicates detected - trusting LLM for contradiction detection");
    Ok(serde_json::json!({
        "has_duplicate": false,
        "has_contradiction": false,
        "message": "No duplicates detected. LLM will handle contradiction detection."
    }))
}

/// Resolve conflict between memories — user chooses which to keep or merge
#[tauri::command]
pub async fn resolve_conflict(
    _new_content: String,
    existing_memory_id: i32,
    resolution: String,
    user_id: String,
    explanation: Option<String>,
    new_memory_id: Option<i32>,
) -> Result<serde_json::Value, String> {
    println!("[Rust] resolve_conflict called - resolution: {}, explanation: {:?}", resolution, explanation);

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    match resolution.as_str() {
        "memoryA" => {
            println!("[Rust] Resolution: Keep OLD memory #{}, delete new", existing_memory_id);

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
            println!("[Rust] Resolution: Keep NEW memory, delete OLD #{}", existing_memory_id);

            let result = sqlx::query("DELETE FROM memories WHERE id = ?")
                .bind(existing_memory_id)
                .execute(&pool)
                .await
                .map_err(|e| format!("Failed to delete existing memory: {}", e))?;

            println!("[Rust] ✅ Deleted {} row(s) - old memory discarded", result.rows_affected());
            println!("[Rust] ⚠️ Deletion will sync on next auto-sync cycle (immediate propagation not yet implemented in resolve_conflict)");
        }
        "both" => {
            println!("[Rust] Resolution: Keep BOTH memories with explanation");

            let actual_new_memory_id = new_memory_id
                .ok_or_else(|| "New memory ID required for 'both' resolution".to_string())?;

            println!("[Rust] Using already-stored new memory with ID: {}", actual_new_memory_id);

            if let Some(explanation_text) = explanation {
                let explanation_id = memory::insert_memory(
                    &pool,
                    Some("Explanation of conflicting memories"),
                    &explanation_text,
                    Some("user"),
                    None,
                    None,
                    None,
                    None,
                    Some(&user_id),
                    "default",
                    true,
                    false,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .map_err(|e| format!("Failed to store explanation: {}", e))?;

                println!("[Rust] Stored explanation memory with ID: {}", explanation_id);

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

                sqlx::query("UPDATE memories SET updated_at = datetime('now') WHERE id = ? OR id = ?")
                    .bind(actual_new_memory_id)
                    .bind(existing_memory_id)
                    .execute(&pool)
                    .await
                    .map_err(|e| format!("Failed to update timestamps: {}", e))?;

                println!("[Rust] ✅ Updated timestamps to trigger relationship sync");
            } else {
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

                sqlx::query("UPDATE memories SET updated_at = datetime('now') WHERE id = ? OR id = ?")
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

/// NEW: Resolve memory conflict (v2) - For refactored flow where memory isn't stored yet
#[tauri::command]
pub async fn resolve_memory_conflict_v2(
    pending_memory_json: String,
    conflicting_memory_id: i32,
    decision: String,
    explanation: Option<String>,
    relationships_json: String,
    user_id: String,
    session_id: String,
) -> Result<serde_json::Value, String> {
    println!("[Rust] resolve_memory_conflict_v2 called - decision: {}, explanation: {:?}", decision, explanation);

    let pending: crate::PendingMemory = serde_json::from_str(&pending_memory_json)
        .map_err(|e| format!("Failed to parse pending memory: {}", e))?;

    let relationships: Vec<crate::RelationshipClassification> = serde_json::from_str(&relationships_json)
        .map_err(|e| format!("Failed to parse relationships: {}", e))?;

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url()).await
        .map_err(|e| format!("Failed to connect to database: {}", e))?;

    match decision.as_str() {
        "keep_old" => {
            println!("[Rust] Resolution: Keep OLD memory #{}, discard new", conflicting_memory_id);
            Ok(serde_json::json!({
                "success": true,
                "action": "kept_old",
                "message": "Kept existing memory, new statement discarded"
            }))
        }

        "keep_new" => {
            println!("[Rust] Resolution: Keep NEW memory, delete OLD #{}", conflicting_memory_id);

            // Compute hash before deleting so we can propagate to sync peers
            let content_hash: Option<String> = sqlx::query_scalar::<_, String>(
                "SELECT content FROM memories WHERE id = ?"
            )
            .bind(conflicting_memory_id)
            .fetch_optional(&pool)
            .await
            .ok()
            .flatten()
            .map(|content| {
                use sha2::{Digest, Sha256};
                format!("{:x}", Sha256::digest(content.as_bytes()))
            });

            sqlx::query("DELETE FROM memories WHERE id = ?")
                .bind(conflicting_memory_id)
                .execute(&pool)
                .await
                .map_err(|e| format!("Failed to delete old memory: {}", e))?;

            println!("[Rust] ✅ Deleted old memory #{}", conflicting_memory_id);

            // Propagate deletion to sync peers so they don't restore the old memory
            if let Some(hash) = content_hash {
                let zynksync_service = crate::ZYNKSYNC_SERVICE.lock().await;
                if let Some(service) = zynksync_service.as_ref() {
                    if service.is_auto_sync_enabled().await {
                        match service.propagate_deletion_by_hash(hash).await {
                            Ok(count) => println!("[Rust] ✓ Deletion propagated to {} device(s)", count),
                            Err(e) => eprintln!("[Rust] ⚠ Warning: Failed to propagate deletion: {}", e),
                        }
                    }
                }
            }

            let new_memory_id = crate::store_pending_memory(&pool, &pending, &user_id, &session_id).await?;

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

            let new_memory_id = crate::store_pending_memory(&pool, &pending, &user_id, &session_id).await?;

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

            let new_memory_id = crate::store_pending_memory(&pool, &pending, &user_id, &session_id).await?;

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

            let new_memory_id = crate::store_pending_memory(&pool, &pending, &user_id, &session_id).await?;

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

            if let Some(expl) = explanation {
                if !expl.trim().is_empty() {
                    let expl_clone = expl.clone();
                    let expl_embedding = tokio::task::spawn_blocking(move || {
                        crate::llm::local_embeddings::generate_local_embedding(&expl_clone)
                    }).await
                    .map_err(|e| format!("Failed to generate embedding: {}", e))?
                    .map_err(|e| format!("Failed to generate embedding: {}", e))?;

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

                    let _ = memory::create_memory_link(&pool, explanation_id, new_memory_id, "resolves", 0.95, Some("Resolves contradiction"), "user").await;
                    let _ = memory::create_memory_link(&pool, explanation_id, conflicting_memory_id, "resolves", 0.95, Some("Resolves contradiction"), "user").await;

                    println!("[Rust] ✅ Stored resolution memory #{} with resolves edges", explanation_id);
                }
            }

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

use serde::Deserialize;

#[derive(Deserialize)]
struct PersonaImportPackage {
    format: String,
    collection: PersonaCollection,
    memories: Vec<PersonaImportMemory>,
}

#[derive(Deserialize)]
struct PersonaCollection {
    collection_id: String,
    namespace: String,
}

#[derive(Deserialize)]
struct PersonaImportMemory {
    memory_id: String,
    content: String,
    #[serde(rename = "type")]
    memory_type: String,
    temporal_status: String,
    placement: String,
    sources: serde_json::Value,
}

/// Validate and atomically import a reviewed private-persona memory collection.
/// Embeddings are generated locally and are not trusted from the package.
#[tauri::command]
pub async fn import_persona_memory_collection(
    user_id: String,
    package_json: String,
    dry_run: Option<bool>,
) -> Result<serde_json::Value, String> {
    let package: PersonaImportPackage = serde_json::from_str(&package_json)
        .map_err(|e| format!("Invalid persona memory package: {}", e))?;
    if package.format != "zynkbot-persona-memory-import" {
        return Err("Unsupported persona memory package format".to_string());
    }
    if package.collection.collection_id.trim().is_empty()
        || package.collection.namespace.trim().is_empty()
    {
        return Err("Collection ID and namespace are required".to_string());
    }

    let database_memories: Vec<&PersonaImportMemory> = package
        .memories
        .iter()
        .filter(|memory| memory.placement != "persona")
        .collect();
    if database_memories.iter().any(|memory| {
        memory.memory_id.trim().is_empty()
            || memory.content.trim().is_empty()
            || !matches!(memory.placement.as_str(), "retrieved" | "pinned")
    }) {
        return Err("Package contains an invalid database memory".to_string());
    }
    let mut ids = std::collections::HashSet::new();
    if database_memories
        .iter()
        .any(|memory| !ids.insert(memory.memory_id.as_str()))
    {
        return Err("Package contains duplicate memory IDs".to_string());
    }

    let pool = crate::db::create_pool()
        .await
        .map_err(|e| format!("Failed to open memory database: {}", e))?;
    let existing: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memories WHERE user_id = ? AND collection_id = ?",
    )
    .bind(&user_id)
    .bind(&package.collection.collection_id)
    .fetch_one(&pool)
    .await
    .map_err(|e| format!("Failed to inspect existing collection: {}", e))?;

    if dry_run.unwrap_or(false) {
        pool.close().await;
        return Ok(serde_json::json!({
            "valid": true,
            "collection_id": package.collection.collection_id,
            "database_memories": database_memories.len(),
            "persona_rules_skipped": package.memories.len() - database_memories.len(),
            "existing_collection_rows": existing,
        }));
    }

    let texts: Vec<String> = database_memories
        .iter()
        .map(|memory| memory.content.clone())
        .collect();
    let embeddings = tokio::task::spawn_blocking(move || {
        crate::llm::local_embeddings::generate_local_embeddings_batch(texts, Some(32))
    })
    .await
    .map_err(|e| format!("Embedding task failed: {}", e))?
    .map_err(|e| format!("Failed to generate local embeddings: {}", e))?;
    if embeddings.len() != database_memories.len() {
        return Err("Embedding count did not match memory count".to_string());
    }

    let mut transaction = pool
        .begin()
        .await
        .map_err(|e| format!("Failed to begin import transaction: {}", e))?;
    let mut inserted = 0_u64;
    let mut skipped = 0_u64;
    for (memory, embedding) in database_memories.iter().zip(embeddings) {
        let embedding_blob: Vec<u8> = embedding
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let provenance = serde_json::to_string(&memory.sources)
            .map_err(|e| format!("Failed to serialize memory provenance: {}", e))?;
        let result = sqlx::query(
            "INSERT OR IGNORE INTO memories (
                title, content, source_type, user_id, namespace, embedding,
                is_syncable, is_shareable, original_text, collection_id,
                memory_placement, external_id, temporal_status, provenance_json
             ) VALUES (?, ?, 'persona_import', ?, ?, ?, 1, 0, ?, ?, ?, ?, ?, ?)",
        )
        .bind(format!("Imported {}", memory.memory_type))
        .bind(&memory.content)
        .bind(&user_id)
        .bind(&package.collection.namespace)
        .bind(embedding_blob)
        .bind(&memory.content)
        .bind(&package.collection.collection_id)
        .bind(&memory.placement)
        .bind(&memory.memory_id)
        .bind(&memory.temporal_status)
        .bind(provenance)
        .execute(&mut *transaction)
        .await
        .map_err(|e| format!("Failed to import memory {}: {}", memory.memory_id, e))?;
        if result.rows_affected() == 1 {
            inserted += 1;
        } else {
            skipped += 1;
        }
    }
    transaction
        .commit()
        .await
        .map_err(|e| format!("Failed to commit persona memory import: {}", e))?;
    pool.close().await;
    Ok(serde_json::json!({
        "success": true,
        "collection_id": package.collection.collection_id,
        "inserted": inserted,
        "skipped_existing": skipped,
        "persona_rules_skipped": package.memories.len() - database_memories.len(),
    }))
}

/// Delete one imported collection as a rollback boundary.
#[tauri::command]
pub async fn delete_persona_memory_collection(
    user_id: String,
    collection_id: String,
) -> Result<serde_json::Value, String> {
    if collection_id.trim().is_empty() {
        return Err("Collection ID is required".to_string());
    }
    let pool = crate::db::create_pool()
        .await
        .map_err(|e| format!("Failed to open memory database: {}", e))?;
    let result = sqlx::query(
        "DELETE FROM memories WHERE user_id = ? AND collection_id = ?",
    )
    .bind(&user_id)
    .bind(&collection_id)
    .execute(&pool)
    .await
    .map_err(|e| format!("Failed to delete persona memory collection: {}", e))?;
    pool.close().await;
    Ok(serde_json::json!({
        "success": true,
        "collection_id": collection_id,
        "deleted": result.rows_affected(),
    }))
}

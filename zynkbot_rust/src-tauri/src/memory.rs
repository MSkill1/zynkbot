use std::cmp::Ordering;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use chrono::{DateTime, Utc};

// ============================================================================
// STRUCTS
// ============================================================================

/// Memory relationship/link between two memories
#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
#[allow(dead_code)]
pub struct MemoryLink {
    pub id: i32,
    pub source_memory_id: i32,
    pub target_memory_id: i32,
    pub relation_type: String,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
    pub notes: Option<String>,
    pub created_by: String,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct Memory {
    pub id: i32,
    pub title: Option<String>,
    pub content: String,
    pub source_type: Option<String>,
    pub session_id: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
    #[sqlx(skip)]
    pub embedding: Option<Vec<f32>>,
    pub parent_scroll_id: Option<i32>,
    pub chunk_index: Option<i32>,
    pub user_id: Option<String>,
    pub namespace: String,
    pub is_syncable: bool,
    pub is_shareable: bool,
    pub event_type: Option<String>,
    pub event_date: Option<DateTime<Utc>>,
    pub link_count: i32,
    pub is_ephemeral: bool,
    pub expires_at: Option<DateTime<Utc>>,
    pub entities_detected: Option<serde_json::Value>,
    pub original_text: Option<String>,
    #[sqlx(default)]
    pub similarity: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct MemoryRelationship {
    pub memory_id: i32,
    pub related_memory_id: i32,
    pub relation_type: String,
    pub confidence: f32,
    pub direction: String,
    pub created_at: DateTime<Utc>,
    pub notes: Option<String>,
    pub created_by: String,
}

// ============================================================================
// INTERNAL HELPERS
// ============================================================================

/// Row type for fetching just id + embedding blob (used by vector/hybrid search)
#[derive(sqlx::FromRow)]
struct EmbeddingRow {
    id: i32,
    embedding: Option<Vec<u8>>,
}

/// Row type for fetching id + embedding + entities (used by hybrid search)
#[derive(sqlx::FromRow)]
struct HybridRow {
    id: i32,
    embedding: Option<Vec<u8>>,
    entities_detected: Option<String>,
}

fn blob_to_f32(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

pub fn blob_to_f32_pub(blob: &[u8]) -> Vec<f32> {
    blob_to_f32(blob)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Return the id of any existing memory for `user_id` whose embedding has cosine similarity
/// greater than `threshold` to `query_embedding`, or None. Replacement for the pgvector
/// `<=>` operator query — SQLite has no native vector ops, so we score in Rust.
#[allow(dead_code)]
pub async fn find_near_duplicate(
    pool: &SqlitePool,
    user_id: &str,
    query_embedding: &[f32],
    threshold: f32,
) -> Option<i32> {
    let rows: Vec<(i32, Option<Vec<u8>>)> = sqlx::query_as(
        "SELECT id, embedding FROM memories WHERE user_id = ? AND embedding IS NOT NULL",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await
    .ok()?;

    for (id, blob) in rows {
        if let Some(b) = blob {
            if cosine_similarity(query_embedding, &blob_to_f32(&b)) > threshold {
                return Some(id);
            }
        }
    }
    None
}

/// Return the user's preferred name from user_profile.json, falling back to full_name.
/// Returns None if onboarding hasn't occurred yet.
pub async fn get_user_display_name(_pool: &SqlitePool, _user_id: &str) -> Option<String> {
    let path = crate::db::get_user_profile_path();
    let profile: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}));

    let preferred = profile.get("preferred_name").and_then(|v| v.as_str()).unwrap_or("");
    if !preferred.is_empty() {
        return Some(preferred.to_string());
    }

    let full = profile.get("full_name").and_then(|v| v.as_str()).unwrap_or("");
    if !full.is_empty() {
        Some(full.to_string())
    } else {
        None
    }
}

/// Fraction of query_entities that fuzzy-match any stored entity word.
fn entity_overlap_score(entities_json: &Option<String>, query_entities: &[String]) -> f64 {
    if query_entities.is_empty() {
        return 0.0;
    }
    let stored: Vec<String> = entities_json
        .as_ref()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .and_then(|v| v.as_array().cloned())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    e.get("word")
                        .and_then(|w| w.as_str())
                        .map(|s| s.to_lowercase())
                })
                .collect()
        })
        .unwrap_or_default();

    let matches = query_entities
        .iter()
        .filter(|qe| {
            stored
                .iter()
                .any(|se| se.contains(qe.as_str()) || qe.contains(se.as_str()))
        })
        .count();

    (matches as f64) / (query_entities.len() as f64)
}

/// Fetch full Memory rows for a list of IDs (similarity field left as None, set by caller).
async fn fetch_memories_by_ids(
    pool: &SqlitePool,
    ids: &[i32],
) -> Result<Vec<Memory>, sqlx::Error> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let sql = format!(
        "SELECT id, title, content, source_type, session_id, created_at, updated_at,
                parent_scroll_id, chunk_index, user_id, namespace,
                is_syncable, is_shareable, event_type, event_date,
                link_count, is_ephemeral, expires_at, entities_detected, original_text
         FROM memories WHERE id IN ({})",
        placeholders
    );
    let mut query = sqlx::query_as::<_, Memory>(&sql);
    for id in ids {
        query = query.bind(id);
    }
    query.fetch_all(pool).await
}

// ============================================================================
// VECTOR SEARCH
// ============================================================================

/// Search for similar memories using vector similarity (Rust-side cosine computation).
/// Returns only memories with similarity >= 0.50.
#[allow(dead_code)]
pub async fn vector_search(
    pool: &SqlitePool,
    query_embedding: Vec<f32>,
    user_id: Option<&str>,
    session_id: Option<&str>,
    namespace: Option<&str>,
    limit: i32,
) -> Result<Vec<Memory>, sqlx::Error> {
    const MIN_SIMILARITY: f64 = 0.50;

    println!(
        "[VectorSearch] MIN_SIMILARITY={} (Rust-side cosine)",
        MIN_SIMILARITY
    );

    // Step 1: fetch (id, embedding_blob) for all candidates matching filters
    let mut sql =
        String::from("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL");
    if user_id.is_some() {
        sql.push_str(" AND user_id = ?");
    }
    if session_id.is_some() {
        sql.push_str(" AND session_id = ?");
    }
    if namespace.is_some() {
        sql.push_str(" AND namespace = ?");
    }

    let mut q = sqlx::query_as::<_, EmbeddingRow>(&sql);
    if let Some(uid) = user_id {
        q = q.bind(uid);
    }
    if let Some(sid) = session_id {
        q = q.bind(sid);
    }
    if let Some(ns) = namespace {
        q = q.bind(ns);
    }
    let rows = q.fetch_all(pool).await?;

    // Step 2: score in Rust
    let mut scored: Vec<(i32, f64)> = rows
        .iter()
        .filter_map(|r| {
            let blob = r.embedding.as_ref()?;
            let sim = cosine_similarity(&query_embedding, &blob_to_f32(blob)) as f64;
            if sim >= MIN_SIMILARITY { Some((r.id, sim)) } else { None }
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored.truncate(limit as usize);

    // Step 3: fetch full rows for top IDs
    let id_scores: HashMap<i32, f64> = scored.iter().cloned().collect();
    let ids: Vec<i32> = scored.iter().map(|(id, _)| *id).collect();

    let mut memories = fetch_memories_by_ids(pool, &ids).await?;
    for m in &mut memories {
        m.similarity = id_scores.get(&m.id).copied();
    }
    memories.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(Ordering::Equal));

    println!("[VectorSearch] Returned {} memories:", memories.len());
    for (i, m) in memories.iter().enumerate() {
        if let Some(sim) = m.similarity {
            println!(
                "  [{}] Memory {} - Similarity: {:.3} ({}%)",
                i + 1,
                m.id,
                sim,
                (sim * 100.0) as i32
            );
        }
    }

    Ok(memories)
}

// ============================================================================
// HYBRID SEARCH
// ============================================================================

/// Hybrid search combining entity matching + semantic similarity (Rust-side).
///
/// Scoring: entity_overlap * 0.6 + semantic_similarity * 0.4 when entities match,
/// pure semantic similarity otherwise.  Threshold: 0.35.
///
/// Vector search is a linear scan: all candidate rows are fetched from SQLite and
/// cosine similarity is computed in Rust. This is correct and fast at typical personal
/// memory counts. At very large scale (hundreds of thousands of memories), sqlite-vec
/// would replace this with an indexed ANN search — see ROADMAP.md.
#[allow(dead_code)]
pub async fn hybrid_search(
    pool: &SqlitePool,
    query_embedding: Vec<f32>,
    query_entities: Vec<String>,
    user_id: Option<&str>,
    session_id: Option<&str>,
    namespace: Option<&str>,
    limit: i32,
) -> Result<Vec<Memory>, sqlx::Error> {
    const MIN_SIMILARITY: f64 = 0.35;

    let query_entities_lower: Vec<String> =
        query_entities.iter().map(|e| e.to_lowercase()).collect();

    // Step 1: fetch candidates that have an embedding or entity data
    let mut sql = String::from(
        "SELECT id, embedding, entities_detected FROM memories \
         WHERE (embedding IS NOT NULL OR (entities_detected IS NOT NULL AND entities_detected != '[]'))",
    );
    // Exclude system memories EXCEPT when explicitly searching the system user.
    // Zynkbot system-doc queries pass user_id = "system"; without this guard the
    // blanket `user_id != 'system'` contradicted the caller's `user_id = 'system'`
    // filter and returned 0 rows for every Zynkbot question.
    if user_id != Some("system") {
        sql.push_str(" AND user_id != 'system'");
    }
    if user_id.is_some() {
        sql.push_str(" AND user_id = ?");
    }
    if session_id.is_some() {
        sql.push_str(" AND session_id = ?");
    }
    if namespace.is_some() {
        sql.push_str(" AND namespace = ?");
    }

    let mut q = sqlx::query_as::<_, HybridRow>(&sql);
    if let Some(uid) = user_id {
        q = q.bind(uid);
    }
    if let Some(sid) = session_id {
        q = q.bind(sid);
    }
    if let Some(ns) = namespace {
        q = q.bind(ns);
    }
    let rows = q.fetch_all(pool).await?;

    // Step 2: score in Rust
    let mut scored: Vec<(i32, f64)> = rows
        .iter()
        .filter_map(|r| {
            let semantic = r
                .embedding
                .as_ref()
                .map(|b| cosine_similarity(&query_embedding, &blob_to_f32(b)) as f64)
                .unwrap_or(0.0);

            let entity =
                entity_overlap_score(&r.entities_detected, &query_entities_lower);

            let score = if entity > 0.0 {
                (entity * 0.6) + (semantic * 0.4)
            } else {
                semantic
            };

            if score >= MIN_SIMILARITY || entity > 0.0 {
                Some((r.id, score))
            } else {
                None
            }
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored.truncate(limit as usize);

    // Step 3: fetch full rows for top IDs
    let id_scores: HashMap<i32, f64> = scored.iter().cloned().collect();
    let ids: Vec<i32> = scored.iter().map(|(id, _)| *id).collect();

    let mut memories = fetch_memories_by_ids(pool, &ids).await?;
    for m in &mut memories {
        m.similarity = id_scores.get(&m.id).copied();
    }
    memories.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(Ordering::Equal));

    println!("[HybridSearch] Returned {} memories:", memories.len());
    for (i, m) in memories.iter().enumerate() {
        if let Some(sim) = m.similarity {
            println!(
                "  [{}] Memory {} - Score: {:.3} ({}%) - {}",
                i + 1,
                m.id,
                sim,
                (sim * 100.0) as i32,
                m.title.as_ref().unwrap_or(&"(no title)".to_string())
            );
        }
    }

    Ok(memories)
}

// ============================================================================
// CRUD — list / get / delete
// ============================================================================

/// List all memories (no vector search, just filtering).
#[allow(dead_code)]
pub async fn list_memories(
    pool: &SqlitePool,
    user_id: Option<&str>,
    session_id: Option<&str>,
    namespace: Option<&str>,
) -> Result<Vec<Memory>, sqlx::Error> {
    let mut sql = String::from(
        "SELECT id, title, content, source_type, session_id, created_at, updated_at,
                parent_scroll_id, chunk_index, user_id, namespace,
                is_syncable, is_shareable, event_type, event_date,
                link_count, is_ephemeral, expires_at, entities_detected, original_text
         FROM memories WHERE user_id != 'system'",
    );

    if user_id.is_some() {
        sql.push_str(" AND user_id = ?");
    }
    if session_id.is_some() {
        sql.push_str(" AND session_id = ?");
    }
    if namespace.is_some() {
        sql.push_str(" AND namespace = ?");
    }
    sql.push_str(" ORDER BY datetime(created_at) DESC");

    let mut query = sqlx::query_as::<_, Memory>(&sql);
    if let Some(uid) = user_id {
        query = query.bind(uid);
    }
    if let Some(sid) = session_id {
        query = query.bind(sid);
    }
    if let Some(ns) = namespace {
        query = query.bind(ns);
    }

    query.fetch_all(pool).await
}

/// Get a single memory by ID.
pub async fn get_memory(
    pool: &SqlitePool,
    memory_id: i32,
) -> Result<Option<Memory>, sqlx::Error> {
    sqlx::query_as::<_, Memory>(
        "SELECT id, title, content, source_type, session_id, created_at, updated_at,
                parent_scroll_id, chunk_index, user_id, namespace,
                is_syncable, is_shareable, event_type, event_date,
                link_count, is_ephemeral, expires_at, entities_detected, original_text
         FROM memories WHERE id = ?",
    )
    .bind(memory_id)
    .fetch_optional(pool)
    .await
}

/// Delete a memory by ID.
#[allow(dead_code)]
pub async fn delete_memory(
    pool: &SqlitePool,
    memory_id: i32,
) -> Result<u64, sqlx::Error> {
    let result = sqlx::query("DELETE FROM memories WHERE id = ?")
        .bind(memory_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}

// ============================================================================
// INSERT / UPDATE
// ============================================================================

/// Insert a new memory into the database.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub async fn insert_memory(
    pool: &SqlitePool,
    title: Option<&str>,
    content: &str,
    source_type: Option<&str>,
    session_id: Option<&str>,
    embedding: Option<Vec<f32>>,
    parent_scroll_id: Option<i32>,
    chunk_index: Option<i32>,
    user_id: Option<&str>,
    namespace: &str,
    is_syncable: bool,
    is_shareable: bool,
    entities_detected: Option<serde_json::Value>,
    event_type: Option<&str>,
    event_date: Option<DateTime<Utc>>,
    original_text: Option<&str>,
) -> Result<i32, sqlx::Error> {
    let embedding_vec: Option<Vec<u8>> = embedding
        .map(|v| v.iter().flat_map(|f| f.to_le_bytes()).collect());

    let result = sqlx::query_scalar::<_, i32>(
        "INSERT INTO memories (
            title, content, source_type, session_id, embedding,
            parent_scroll_id, chunk_index, user_id, namespace,
            is_syncable, is_shareable, entities_detected, event_type, event_date,
            original_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id",
    )
    .bind(title)
    .bind(content)
    .bind(source_type)
    .bind(session_id)
    .bind(embedding_vec)
    .bind(parent_scroll_id)
    .bind(chunk_index)
    .bind(user_id)
    .bind(namespace)
    .bind(is_syncable)
    .bind(is_shareable)
    .bind(entities_detected)
    .bind(event_type)
    .bind(event_date)
    .bind(original_text)
    .fetch_one(pool)
    .await?;

    Ok(result)
}

/// Update an existing memory's title and/or content.
#[allow(dead_code)]
pub async fn update_memory(
    pool: &SqlitePool,
    memory_id: i32,
    title: Option<&str>,
    content: Option<&str>,
) -> Result<u64, sqlx::Error> {
    let result = sqlx::query(
        "UPDATE memories
         SET title = COALESCE(?, title),
             content = COALESCE(?, content)
         WHERE id = ?",
    )
    .bind(title)
    .bind(content)
    .bind(memory_id)
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

// ============================================================================
// MEMORY LINKS — Semantic Relationship Graph
// ============================================================================

/// Get all links for a specific memory (bidirectional, via memory_relationships view).
pub async fn get_memory_relationships(
    pool: &SqlitePool,
    memory_id: i32,
) -> Result<Vec<MemoryRelationship>, sqlx::Error> {
    sqlx::query_as::<_, MemoryRelationship>(
        "SELECT memory_id, related_memory_id, relation_type,
                confidence, direction, created_at, notes, created_by
         FROM memory_relationships
         WHERE memory_id = ?
         ORDER BY datetime(created_at) DESC",
    )
    .bind(memory_id)
    .fetch_all(pool)
    .await
}

/// Get all direct links for a memory (raw memory_links table, both directions).
#[allow(dead_code)]
pub async fn get_memory_links(
    pool: &SqlitePool,
    memory_id: i32,
) -> Result<Vec<MemoryLink>, sqlx::Error> {
    sqlx::query_as::<_, MemoryLink>(
        "SELECT id, source_memory_id, target_memory_id, relation_type,
                confidence, created_at, notes, created_by
         FROM memory_links
         WHERE source_memory_id = ? OR target_memory_id = ?
         ORDER BY datetime(created_at) DESC",
    )
    .bind(memory_id)
    .bind(memory_id)
    .fetch_all(pool)
    .await
}

/// Create a new memory link (semantic relationship).
#[allow(dead_code)]
pub async fn create_memory_link(
    pool: &SqlitePool,
    source_memory_id: i32,
    target_memory_id: i32,
    relation_type: &str,
    confidence: f32,
    notes: Option<&str>,
    created_by: &str,
) -> Result<i32, sqlx::Error> {
    let valid_types = [
        "supports", "contradicts", "elaborates", "reminds_of",
        "caused_by", "quotes", "resolves", "mentions",
    ];
    if !valid_types.contains(&relation_type) {
        return Err(sqlx::Error::Protocol(format!(
            "Invalid relation_type: {}. Must be one of: {:?}",
            relation_type, valid_types
        )));
    }
    if !(0.0..=1.0).contains(&confidence) {
        return Err(sqlx::Error::Protocol(format!(
            "Invalid confidence: {}. Must be between 0.0 and 1.0",
            confidence
        )));
    }

    let link_id = sqlx::query_scalar::<_, i32>(
        "INSERT INTO memory_links
         (source_memory_id, target_memory_id, relation_type, confidence, notes, created_by)
         VALUES (?, ?, ?, ?, ?, ?)
         RETURNING id",
    )
    .bind(source_memory_id)
    .bind(target_memory_id)
    .bind(relation_type)
    .bind(confidence)
    .bind(notes)
    .bind(created_by)
    .fetch_one(pool)
    .await?;

    sqlx::query(
        "UPDATE memories SET updated_at = datetime('now') WHERE id = ?",
    )
    .bind(source_memory_id)
    .execute(pool)
    .await?;

    Ok(link_id)
}

/// Delete a memory link by ID.
#[allow(dead_code)]
pub async fn delete_memory_link(
    pool: &SqlitePool,
    link_id: i32,
) -> Result<u64, sqlx::Error> {
    let result = sqlx::query("DELETE FROM memory_links WHERE id = ?")
        .bind(link_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}

/// Get full memory graph for a memory (includes content of related memories).
/// Currently supports depth=1 (direct connections only).
#[allow(dead_code)]
pub async fn get_memory_graph(
    pool: &SqlitePool,
    memory_id: i32,
    _depth: i32,
) -> Result<Vec<(MemoryRelationship, Memory)>, sqlx::Error> {
    let relationships = get_memory_relationships(pool, memory_id).await?;
    let mut graph_data = Vec::new();
    for rel in relationships {
        if let Some(related_memory) = get_memory(pool, rel.related_memory_id).await? {
            graph_data.push((rel, related_memory));
        }
    }
    Ok(graph_data)
}

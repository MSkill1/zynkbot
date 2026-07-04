// Knowledge Base RAG (Retrieval Augmented Generation) System
// Handles document chunking, embedding generation, vector storage, and semantic search

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc};
use sqlx::{SqlitePool, Row};

fn blob_to_f32(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KBDocument {
    pub id: i32,
    pub user_id: String,
    pub file_path: String,
    pub file_name: String,
    pub file_size: i64,
    pub last_modified: DateTime<Utc>,
    pub indexed_at: DateTime<Utc>,
    pub chunk_count: i32,
    pub status: String,  // 'indexed', 'indexing', 'needs_reindex', 'error'
    pub error_message: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KBSearchResult {
    pub chunk_id: i32,
    pub document_id: i32,
    pub file_name: String,
    pub file_path: String,
    pub chunk_index: i32,
    pub content: String,
    pub similarity_score: f32,
}

// ============================================================================
// KB FOLDER MANAGEMENT
// ============================================================================

/// Get the knowledge base folder path for a user
/// Creates the folder if it doesn't exist
///
/// Uses project-relative path for portability across machines:
/// - Path: {project_root}/knowledge_base/{user_id}/
/// - Example: C:\Users\{username}\zynkbot\knowledge_base\{user_id}\
///
/// This ensures KB files stay within the project directory and can be
/// backed up/moved with the project.
pub fn get_kb_folder_path(user_id: &str) -> Result<PathBuf, String> {
    // Get project root (2 levels up from src-tauri working directory)
    let project_root = std::env::current_dir()
        .map_err(|e| format!("Failed to get current directory: {}", e))?
        .parent()  // Go from src-tauri to zynkbot_rust
        .and_then(|p| p.parent())  // Go from zynkbot_rust to project root
        .ok_or("Cannot determine project root")?
        .to_path_buf();

    let kb_folder = project_root
        .join("knowledge_base")
        .join(user_id);

    // Create folder if it doesn't exist
    if !kb_folder.exists() {
        fs::create_dir_all(&kb_folder)
            .map_err(|e| format!("Failed to create KB folder: {}", e))?;

        // On first use, copy the sample document so devs have something to index immediately
        let sample_src = project_root
            .join("knowledge_base")
            .join("sample_knowledge_base_document.txt");
        if sample_src.exists() {
            let sample_dst = kb_folder.join("sample_knowledge_base_document.txt");
            if let Err(e) = fs::copy(&sample_src, &sample_dst) {
                println!("[KB] Warning: could not copy sample document: {}", e);
            } else {
                println!("[KB] Copied sample document to new KB folder");
            }
        }
    }

    Ok(kb_folder)
}

// ============================================================================
// DOCUMENT CHUNKING
// ============================================================================

/// Chunk document into pieces with sentence-boundary awareness
/// - target_tokens: Approximate target size per chunk (default: 500)
/// - overlap_tokens: Number of tokens to overlap between chunks (default: 50)
///
///   Returns vector of text chunks
pub fn chunk_document(content: &str, target_tokens: usize, overlap_tokens: usize) -> Vec<String> {
    // Simple token estimation: ~4 characters per token
    let chars_per_token = 4;
    let target_chars = target_tokens * chars_per_token;
    let overlap_chars = overlap_tokens * chars_per_token;

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < content.len() {
        // Find a safe end position
        let mut end = (start + target_chars).min(content.len());

        // Ensure we're on a character boundary
        while end < content.len() && !content.is_char_boundary(end) {
            end += 1;
        }

        // If not at end of document, try to break at sentence boundary
        if end < content.len() {
            let mut search_start = (end.saturating_sub(target_chars / 4)).max(start);

            // Ensure search_start is on a character boundary
            while search_start < content.len() && !content.is_char_boundary(search_start) {
                search_start += 1;
            }

            // Use char_indices to find sentence boundaries safely
            if let Some((idx, _)) = content[search_start..end]
                .char_indices()
                .rev()
                .find(|(_, c)| *c == '.' || *c == '!' || *c == '?')
            {
                // Move to the character AFTER the punctuation
                let boundary_pos = search_start + idx;
                end = content[boundary_pos..]
                    .char_indices()
                    .nth(1)
                    .map(|(i, _)| boundary_pos + i)
                    .unwrap_or(content.len());

                // Skip whitespace using char_indices
                while end < content.len() {
                    if let Some(c) = content[end..].chars().next() {
                        if c.is_whitespace() {
                            end += c.len_utf8();
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        // Extract chunk (now safe)
        let chunk = content[start..end].trim().to_string();
        if !chunk.is_empty() {
            chunks.push(chunk);
        }

        // If we've reached the end of content, we're done
        if end >= content.len() {
            break;
        }

        // Move start position with overlap, ensuring character boundary
        start = end.saturating_sub(overlap_chars);
        while start > 0 && start < content.len() && !content.is_char_boundary(start) {
            start += 1;
        }

        // Ensure we make progress
        if start >= content.len() {
            break;
        }
    }

    chunks
}

/// Estimate token count for text (simple approximation)
pub fn estimate_token_count(text: &str) -> i32 {
    // Simple heuristic: ~4 characters per token
    (text.len() / 4) as i32
}

// ============================================================================
// DOCUMENT INDEXING
// ============================================================================

/// Index a single document: chunk it, generate embeddings, store in database
/// This is the main indexing function
pub async fn index_document(
    pool: &SqlitePool,
    user_id: &str,
    file_path: &str,
    on_progress: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,
) -> Result<i32, String> {
    println!("[KB RAG] Starting indexing: {}", file_path);

    // Read file
    println!("[KB RAG] Reading file...");
    let content = fs::read_to_string(file_path)
        .map_err(|e| format!("Failed to read file: {}", e))?;
    println!("[KB RAG] File read successfully, length: {} bytes", content.len());

    // Get file metadata
    println!("[KB RAG] Getting file metadata...");
    let metadata = fs::metadata(file_path)
        .map_err(|e| format!("Failed to read metadata: {}", e))?;

    let file_size = metadata.len() as i64;
    let modified_time: DateTime<Utc> = metadata.modified()
        .map_err(|e| format!("Failed to get modified time: {}", e))?
        .into();

    let file_name = Path::new(file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("Invalid file name")?
        .to_string();

    // Chunk the document
    println!("[KB RAG] Starting chunking...");
    // all-MiniLM-L6-v2 has a 256-token limit (~1000 chars). 128 tokens * 4 chars = ~512 chars per chunk.
    let chunks = chunk_document(&content, 128, 25);
    let chunk_count = chunks.len();

    println!("[KB RAG] Created {} chunks", chunk_count);

    // Generate embeddings BEFORE opening the DB transaction. Embedding generation
    // loads the BERT model (seconds on first call) and runs CPU/GPU inference — none
    // of which depends on DB state. Doing it inside the transaction held the SQLite
    // write lock long enough to starve other writers (e.g. ZynkSync pairing code
    // generation), causing `database is locked` errors at startup.
    println!("[KB RAG] Generating embeddings for {} chunks (batch)...", chunk_count);
    let chunk_texts: Vec<String> = chunks.iter().map(|s| s.to_string()).collect();

    // Report chunk count so the UI can show "0 / N" immediately
    if let Some(ref cb) = on_progress {
        cb(0, chunk_count);
    }

    // Process in sub-batches so the caller can emit progress per batch.
    // generate_local_embeddings_batch already batches internally, but calling it
    // once for the whole document gives no intermediate feedback.
    const EMBED_BATCH: usize = 32;
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunk_count);
    for (batch_idx, batch_texts) in chunk_texts.chunks(EMBED_BATCH).enumerate() {
        let batch_vec = batch_texts.to_vec();
        let batch_embeddings = tokio::task::spawn_blocking(move || {
            crate::llm::local_embeddings::generate_local_embeddings_batch(batch_vec, Some(EMBED_BATCH))
        })
        .await
        .map_err(|e| format!("Failed to run embedding task: {}", e))?
        .map_err(|e| format!("Failed to generate embeddings: {}", e))?;

        embeddings.extend(batch_embeddings);
        let completed = std::cmp::min((batch_idx + 1) * EMBED_BATCH, chunk_count);
        if let Some(ref cb) = on_progress {
            cb(completed, chunk_count);
        }
    }
    println!("[KB RAG] ✓ All {} embeddings generated", chunk_count);

    let chunk_count = chunk_count as i32;

    // Start transaction
    let mut tx = pool.begin()
        .await
        .map_err(|e| format!("Failed to start transaction: {}", e))?;

    // Insert or update document record
    let doc_id = sqlx::query_scalar::<_, i32>(
        r#"
        INSERT INTO kb_documents (user_id, file_path, file_name, file_size, last_modified, chunk_count, status)
        VALUES (?, ?, ?, ?, ?, ?, 'indexing')
        ON CONFLICT (user_id, file_path)
        DO UPDATE SET
            file_size = EXCLUDED.file_size,
            last_modified = EXCLUDED.last_modified,
            chunk_count = EXCLUDED.chunk_count,
            status = 'indexing',
            error_message = NULL
        RETURNING id
        "#
    )
    .bind(user_id)
    .bind(file_path)
    .bind(&file_name)
    .bind(file_size)
    .bind(modified_time)
    .bind(chunk_count)
    .fetch_one(&mut *tx)
    .await
    .map_err(|e| format!("Failed to insert document: {}", e))?;

    // Delete old chunks if re-indexing
    sqlx::query("DELETE FROM kb_chunks WHERE document_id = ?")
        .bind(doc_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("Failed to delete old chunks: {}", e))?;

    // Insert chunks with pre-generated embeddings
    for (idx, chunk_text) in chunks.iter().enumerate() {
        let token_count = estimate_token_count(chunk_text);

        let embedding_blob: Vec<u8> = embeddings[idx].iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        sqlx::query(
            "INSERT INTO kb_chunks (document_id, chunk_index, content, embedding, token_count)
             VALUES (?, ?, ?, ?, ?)"
        )
        .bind(doc_id)
        .bind(idx as i32)
        .bind(chunk_text)
        .bind(embedding_blob)
        .bind(token_count)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("Failed to insert chunk {}: {}", idx, e))?;

        if (idx + 1) % 10 == 0 {
            println!("[KB RAG] Indexed {}/{} chunks", idx + 1, chunk_count);
        }
    }

    // Mark document as indexed
    sqlx::query("UPDATE kb_documents SET status = 'indexed', indexed_at = datetime('now') WHERE id = ?")
        .bind(doc_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("Failed to update status: {}", e))?;

    // Commit transaction
    tx.commit()
        .await
        .map_err(|e| format!("Failed to commit transaction: {}", e))?;

    println!("[KB RAG] Indexing complete: {} (document_id: {})", file_name, doc_id);

    Ok(doc_id)
}

/// Index text content directly (for snap-ins, notes, etc.)
/// Creates a virtual document with the given file_path (for organization)
pub async fn index_text_as_document(
    pool: &SqlitePool,
    user_id: &str,
    virtual_file_path: &str,
    content: &str,
) -> Result<i32, String> {
    println!("[KB RAG] Indexing text as document: {}", virtual_file_path);

    // Extract file name from virtual path
    let file_name = Path::new(virtual_file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("untitled.txt")
        .to_string();

    let file_size = content.len() as i64;
    let modified_time: DateTime<Utc> = Utc::now();

    // Chunk the document
    println!("[KB RAG] Starting chunking...");
    // all-MiniLM-L6-v2 has a 256-token limit (~1000 chars). 128 tokens * 4 chars = ~512 chars per chunk.
    let chunks = chunk_document(content, 128, 25);
    let chunk_count = chunks.len() as i32;

    println!("[KB RAG] Created {} chunks", chunk_count);

    // Generate embeddings BEFORE opening the DB transaction — see index_document for rationale.
    println!("[KB RAG] Generating embeddings for {} chunks (batch)...", chunk_count);
    let chunk_texts: Vec<String> = chunks.iter().map(|s| s.to_string()).collect();
    let embeddings = crate::llm::local_embeddings::generate_local_embeddings_batch(chunk_texts, None)
        .map_err(|e| format!("Failed to generate embeddings: {}", e))?;
    println!("[KB RAG] ✓ All {} embeddings generated", chunk_count);

    // Start transaction
    let mut tx = pool.begin()
        .await
        .map_err(|e| format!("Failed to start transaction: {}", e))?;

    // Insert or update document record
    let doc_id = sqlx::query_scalar::<_, i32>(
        r#"
        INSERT INTO kb_documents (user_id, file_path, file_name, file_size, last_modified, chunk_count, status)
        VALUES (?, ?, ?, ?, ?, ?, 'indexing')
        ON CONFLICT (user_id, file_path)
        DO UPDATE SET
            file_size = EXCLUDED.file_size,
            last_modified = EXCLUDED.last_modified,
            chunk_count = EXCLUDED.chunk_count,
            status = 'indexing',
            error_message = NULL
        RETURNING id
        "#
    )
    .bind(user_id)
    .bind(virtual_file_path)
    .bind(&file_name)
    .bind(file_size)
    .bind(modified_time)
    .bind(chunk_count)
    .fetch_one(&mut *tx)
    .await
    .map_err(|e| format!("Failed to insert document: {}", e))?;

    // Delete old chunks if re-indexing
    sqlx::query("DELETE FROM kb_chunks WHERE document_id = ?")
        .bind(doc_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("Failed to delete old chunks: {}", e))?;

    // Insert chunks with pre-generated embeddings
    for (idx, chunk_text) in chunks.iter().enumerate() {
        let token_count = estimate_token_count(chunk_text);

        let embedding_blob: Vec<u8> = embeddings[idx].iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        sqlx::query(
            "INSERT INTO kb_chunks (document_id, chunk_index, content, embedding, token_count)
             VALUES (?, ?, ?, ?, ?)"
        )
        .bind(doc_id)
        .bind(idx as i32)
        .bind(chunk_text)
        .bind(embedding_blob)
        .bind(token_count)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("Failed to insert chunk {}: {}", idx, e))?;

        if (idx + 1) % 10 == 0 {
            println!("[KB RAG] Indexed {}/{} chunks", idx + 1, chunk_count);
        }
    }

    // Mark document as successfully indexed
    sqlx::query("UPDATE kb_documents SET status = 'indexed', indexed_at = datetime('now') WHERE id = ?")
        .bind(doc_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("Failed to update document status: {}", e))?;

    // Commit transaction
    tx.commit()
        .await
        .map_err(|e| format!("Failed to commit transaction: {}", e))?;

    println!("[KB RAG] ✅ Successfully indexed text document with {} chunks", chunk_count);

    Ok(doc_id)
}

/// Remove document and all its chunks from index
pub async fn remove_document_index(
    pool: &SqlitePool,
    user_id: &str,
    file_path: &str,
) -> Result<(), String> {
    // Delete chunks explicitly — CASCADE DELETE requires PRAGMA foreign_keys=ON
    // which may not be set on caller-created pools.
    let doc_id: Option<i32> = sqlx::query_scalar(
        "SELECT id FROM kb_documents WHERE user_id = ? AND file_path = ?"
    )
    .bind(user_id)
    .bind(file_path)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("Failed to find document: {}", e))?;

    if let Some(id) = doc_id {
        sqlx::query("DELETE FROM kb_chunks WHERE document_id = ?")
            .bind(id)
            .execute(pool)
            .await
            .map_err(|e| format!("Failed to delete chunks: {}", e))?;

        sqlx::query("DELETE FROM kb_documents WHERE id = ?")
            .bind(id)
            .execute(pool)
            .await
            .map_err(|e| format!("Failed to delete document: {}", e))?;
    }

    Ok(())
}

// ============================================================================
// SEMANTIC SEARCH
// ============================================================================

/// Search knowledge base chunks using semantic similarity
/// Returns top K most relevant chunks with their source document info
pub async fn search_kb_chunks(
    pool: &SqlitePool,
    user_id: &str,
    query: &str,
    top_k: usize,
    include_system_docs: bool,
) -> Result<Vec<KBSearchResult>, String> {
    println!("[KB RAG] Searching: {}", query);

    // Extract potential filenames and file references from query
    // Strategy 1: Direct filename mentions (e.g., "requirements.txt", "config.json")
    // Strategy 2: Multi-word file references (e.g., "database queries", "installation guide")
    // Strategy 3: Single significant words (e.g., "Grok" matches "Grok Review.txt")
    let query_lower = query.to_lowercase();

    // Get words with extensions
    let mut potential_filenames: Vec<String> = query_lower
        .split_whitespace()
        .filter(|word| word.contains('.') && word.len() > 3)
        .map(|s| s.to_string())
        .collect();

    // Add single significant words (4+ chars, capitalized in original query)
    for word in query.split_whitespace() {
        if word.len() >= 4 && word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
            potential_filenames.push(word.to_lowercase());
        }
    }

    // Also extract significant multi-word phrases that might be filenames
    // Look for consecutive capitalized/quoted words or phrases between quotes
    let query_words: Vec<&str> = query_lower.split_whitespace().collect();
    for i in 0..query_words.len().saturating_sub(1) {
        let two_word = format!("{} {}", query_words[i], query_words[i + 1]);
        // Skip common stop word combinations
        if !two_word.contains("the ") && !two_word.contains(" the")
            && !two_word.contains("in ") && !two_word.contains(" in")
            && two_word.len() > 8 {  // Reasonable length for a file reference
            potential_filenames.push(two_word);
        }
    }

    if !potential_filenames.is_empty() {
        println!("[KB RAG] Detected potential file references in query: {:?}", potential_filenames);
    }

    // Generate query embedding
    let query_embedding = crate::llm::local_embeddings::generate_local_embedding(query)
        .map_err(|e| format!("Failed to generate query embedding: {}", e))?;

    // Fetch all candidate chunks with their embeddings (Rust-side cosine similarity)
    let user_filter = if include_system_docs {
        "(d.user_id = ? OR d.user_id = '_system')"
    } else {
        "d.user_id = ?"
    };

    let query_sql = format!(
        "SELECT c.id as chunk_id, c.document_id, c.chunk_index, c.content,
                d.file_name, d.file_path, c.embedding
         FROM kb_chunks c
         JOIN kb_documents d ON c.document_id = d.id
         WHERE {} AND d.status = 'indexed' AND d.file_path NOT LIKE 'snap_ins/%'
           AND c.embedding IS NOT NULL",
        user_filter
    );

    let rows = sqlx::query(&query_sql)
        .bind(user_id)
        .fetch_all(pool)
        .await
        .map_err(|e| format!("Search query failed: {}", e))?;

    let mut search_results: Vec<KBSearchResult> = rows
        .into_iter()
        .filter_map(|r| {
            let embedding_blob: Option<Vec<u8>> = r.try_get("embedding").ok().flatten();
            let blob = embedding_blob?;
            let stored = blob_to_f32(&blob);
            let mut similarity_score = cosine_similarity(&query_embedding, &stored);

            let file_name: String = r.try_get("file_name").unwrap_or_default();

            // Filename-aware boosting: if query mentions this file, boost significantly
            if !potential_filenames.is_empty() {
                let file_name_lower = file_name.to_lowercase();
                let file_name_no_ext = file_name_lower
                    .trim_end_matches(".txt")
                    .trim_end_matches(".md")
                    .trim_end_matches(".pdf")
                    .trim_end_matches(".json")
                    .trim_end_matches(".csv");

                for filename_ref in &potential_filenames {
                    // Check if filename reference appears in the file name
                    if file_name_lower.contains(filename_ref.as_str())
                        || filename_ref.contains(&file_name_lower)
                        || file_name_no_ext.contains(filename_ref.as_str())
                        || filename_ref.contains(file_name_no_ext) {
                        // Add 0.5 boost (50%) to ensure filename matches pass threshold
                        similarity_score += 0.5;
                        break;
                    }
                }
            }

            Some(KBSearchResult {
                chunk_id: r.try_get("chunk_id").unwrap_or(0),
                document_id: r.try_get("document_id").unwrap_or(0),
                file_name,
                file_path: r.try_get("file_path").unwrap_or_default(),
                chunk_index: r.try_get("chunk_index").unwrap_or(0),
                content: r.try_get("content").unwrap_or_default(),
                similarity_score,
            })
        })
        .collect();

    // Re-sort after boosting, then limit to top_k
    search_results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap_or(std::cmp::Ordering::Equal));
    search_results.truncate(top_k);

    println!("[KB RAG] Found {} relevant chunks", search_results.len());

    Ok(search_results)
}

/// Get all indexed documents for a user
pub async fn list_kb_documents(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<Vec<KBDocument>, String> {
    let rows = sqlx::query(
        r#"
        SELECT id, user_id, file_path, file_name, file_size, last_modified, indexed_at, chunk_count, status, error_message
        FROM kb_documents
        WHERE user_id = ? AND user_id != '_system'
        ORDER BY file_name ASC
        "#
    )
    .bind(user_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to list documents: {}", e))?;

    let records: Vec<KBDocument> = rows
        .into_iter()
        .map(|r| KBDocument {
            id: r.try_get("id").unwrap_or(0),
            user_id: r.try_get("user_id").unwrap_or_default(),
            file_path: r.try_get("file_path").unwrap_or_default(),
            file_name: r.try_get("file_name").unwrap_or_default(),
            file_size: r.try_get("file_size").unwrap_or(0),
            last_modified: r.try_get("last_modified").unwrap_or_default(),
            indexed_at: r.try_get("indexed_at").unwrap_or_default(),
            chunk_count: r.try_get("chunk_count").unwrap_or(0),
            status: r.try_get("status").unwrap_or_default(),
            error_message: r.try_get("error_message").ok(),
        })
        .collect();

    Ok(records)
}

/// Clear all indexed documents for a user
pub async fn clear_all_documents(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<i64, String> {
    let result = sqlx::query("DELETE FROM kb_documents WHERE user_id = ?")
        .bind(user_id)
        .execute(pool)
    .await
    .map_err(|e| format!("Failed to clear documents: {}", e))?;

    Ok(result.rows_affected() as i64)
}

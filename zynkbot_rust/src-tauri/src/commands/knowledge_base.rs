use crate::knowledge_base;
use crate::kb_rag;
use tauri::Emitter;

/// Scan knowledge base directory and return list of files
#[tauri::command]
pub async fn scan_knowledge_base(directory: String) -> Result<Vec<knowledge_base::KnowledgeBaseFile>, String> {
    knowledge_base::scan_knowledge_base_directory(&directory)
}

/// Search knowledge base for query terms
#[tauri::command]
pub async fn search_knowledge_base(
    directory: String,
    query: String,
) -> Result<Vec<knowledge_base::KnowledgeBaseSearchResult>, String> {
    knowledge_base::search_knowledge_base(&directory, &query)
}

/// Read a specific knowledge base file
#[tauri::command]
pub async fn read_knowledge_base_file(file_path: String) -> Result<String, String> {
    knowledge_base::read_knowledge_base_file(&file_path)
}

/// Get the KB folder path for a user
#[tauri::command]
pub async fn get_kb_folder_path(user_id: String) -> Result<String, String> {
    let path = kb_rag::get_kb_folder_path(&user_id)?;
    Ok(path.to_string_lossy().to_string())
}

/// Open KB folder in file explorer (cross-platform)
#[tauri::command]
pub async fn open_kb_folder_in_explorer(user_id: String) -> Result<(), String> {
    let path = kb_rag::get_kb_folder_path(&user_id)?;
    let path_str = path.to_string_lossy().to_string();

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
pub async fn open_external_file(path: String) -> Result<(), String> {
    let candidate = std::path::PathBuf::from(&path);
    let full_path = if candidate.is_absolute() {
        candidate
    } else {
        crate::db::get_app_data_dir().join(&path)
    };

    if !full_path.exists() {
        return Err(format!("File not found: {}", full_path.display()));
    }

    let path_str = full_path.to_string_lossy().to_string();

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
pub async fn open_external_folder(path: String) -> Result<(), String> {
    let candidate = std::path::PathBuf::from(&path);
    let full_path = if candidate.is_absolute() {
        candidate
    } else {
        crate::db::get_app_data_dir().join(&path)
    };

    if !full_path.exists() {
        return Err(format!("Folder not found: {}", full_path.display()));
    }

    let path_str = full_path.to_string_lossy().to_string();

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
pub async fn index_kb_document(
    app: tauri::AppHandle,
    user_id: String,
    file_path: String,
) -> Result<i32, String> {
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    let progress_cb: Box<dyn Fn(usize, usize) + Send + Sync> = Box::new(move |current, total| {
        let _ = app.emit("kb:indexing_progress", serde_json::json!({ "current": current, "total": total }));
    });

    kb_rag::index_document(&pool, &user_id, &file_path, Some(progress_cb)).await
}

/// List all indexed KB documents for a user
#[tauri::command]
pub async fn list_kb_documents(user_id: String) -> Result<Vec<kb_rag::KBDocument>, String> {
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    kb_rag::list_kb_documents(&pool, &user_id).await
}

/// Remove a document from the index
#[tauri::command]
pub async fn remove_kb_document(user_id: String, file_path: String) -> Result<(), String> {
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    kb_rag::remove_document_index(&pool, &user_id, &file_path).await
}

/// Clear all indexed documents for a user
#[tauri::command]
pub async fn clear_all_kb_documents(user_id: String) -> Result<i64, String> {
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    kb_rag::clear_all_documents(&pool, &user_id).await
}

/// Semantic search in knowledge base
#[tauri::command]
pub async fn search_kb(
    user_id: String,
    query: String,
    top_k: Option<usize>,
) -> Result<Vec<kb_rag::KBSearchResult>, String> {
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    let k = top_k.unwrap_or(10);
    kb_rag::search_kb_chunks(&pool, &user_id, &query, k, false).await
}

#[tauri::command]
pub async fn index_snapin_notes(
    patient_name: String,
    session_title: String,
    notes_content: String,
    user_id: String,
) -> Result<String, String> {
    let safe_patient = patient_name
        .to_lowercase()
        .replace(" ", "_")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect::<String>();

    let safe_session = session_title
        .replace(" ", "_")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .collect::<String>();

    let file_path = format!(
        "snap_ins/therapist/{}/{}.txt",
        safe_patient,
        safe_session
    );

    println!("[Snap-in] Indexing notes at path: {}", file_path);

    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

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

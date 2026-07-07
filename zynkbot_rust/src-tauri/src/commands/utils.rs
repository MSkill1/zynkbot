#[tauri::command]
pub async fn read_text_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read file '{}': {}", path, e))
}

#[tauri::command]
pub async fn read_file_base64(path: String) -> Result<String, String> {
    let bytes = std::fs::read(&path)
        .map_err(|e| format!("Failed to read file '{}': {}", path, e))?;
    Ok(base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes))
}

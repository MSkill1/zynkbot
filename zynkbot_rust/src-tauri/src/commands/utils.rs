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

/// Text of an attached file for the chat: a PDF through the same extractor the knowledge
/// base uses, anything else as UTF-8. Refuses binary (a photo Android could not name was
/// read as text and became 8 million tokens, 2026-10-25). No size cap on text (Matt).
#[tauri::command]
pub async fn read_attachment_text(path: String) -> Result<String, String> {
    let bytes = std::fs::read(&path).map_err(|e| format!("Could not read {}: {}", path, e))?;
    let lower = path.to_lowercase();
    if lower.ends_with(".docx") || (bytes.starts_with(b"PK\x03\x04") && crate::word_docs::docx_text(&bytes).is_ok()) {
        return crate::word_docs::docx_text(&bytes);
    }
    if lower.ends_with(".doc") || bytes.starts_with(&[0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]) {
        return crate::word_docs::doc_text(&bytes);
    }
    if lower.ends_with(".pdf") || bytes.starts_with(b"%PDF") {
        let text = pdf_extract::extract_text_from_mem(&bytes).map_err(|e| format!("Could not read the PDF text: {}", e))?;
        if text.trim().is_empty() { return Err("This PDF has no text to read; it may be a scanned image.".to_string()); }
        return Ok(text);
    }
    let sample = &bytes[..bytes.len().min(4096)];
    let nul = sample.contains(&0u8);
    let bad = String::from_utf8_lossy(sample).matches('\u{FFFD}').count();
    if nul || bad * 10 > sample.len().max(1) { return Err("Not a text file.".to_string()); }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

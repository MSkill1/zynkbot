use crate::web_search;

#[tauri::command]
pub async fn search_web(query: String, max_results: Option<usize>) -> Result<web_search::SearchResponse, String> {
    println!("[WebSearch] Searching for: {}", query);
    let max = max_results.unwrap_or(5);
    web_search::search(&query, max, None).await
}

#[tauri::command]
pub async fn execute_web_search(app: tauri::AppHandle, query: String, max_results: usize, fetch_top_n: usize, backend: Option<String>) -> Result<serde_json::Value, String> {
    use tauri::Emitter;
    println!("[WebSearch] Executing search for: {}", query);
    let _ = app.emit("chat-status", "Searching the web…");

    // A cloud model's own search streams its answer to the screen as it is written.
    let stream_app = app.clone();
    let search_response = web_search::search_with_content_streaming(
        &query, max_results, fetch_top_n, backend.as_deref(),
        &move |t: &str| { let _ = stream_app.emit("stream-token", t); },
    ).await?;

    let json_response = serde_json::to_value(&search_response)
        .map_err(|e| format!("Failed to serialize search results: {}", e))?;

    println!("[WebSearch] Search complete - found {} results", search_response.num_results);

    Ok(json_response)
}

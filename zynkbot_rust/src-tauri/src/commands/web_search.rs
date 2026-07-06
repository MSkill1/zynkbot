use crate::web_search;

#[tauri::command]
pub async fn search_web(query: String, max_results: Option<usize>) -> Result<web_search::SearchResponse, String> {
    println!("[WebSearch] Searching for: {}", query);
    let max = max_results.unwrap_or(5);
    web_search::search_duckduckgo(&query, max).await
}

#[tauri::command]
pub async fn execute_web_search(query: String, max_results: usize, fetch_top_n: usize) -> Result<serde_json::Value, String> {
    println!("[WebSearch] Executing search for: {}", query);

    let search_response = web_search::search_with_content(&query, max_results, fetch_top_n).await?;

    let json_response = serde_json::to_value(&search_response)
        .map_err(|e| format!("Failed to serialize search results: {}", e))?;

    println!("[WebSearch] Search complete - found {} results", search_response.num_results);

    Ok(json_response)
}

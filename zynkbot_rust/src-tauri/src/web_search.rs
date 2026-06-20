/// DuckDuckGo Web Search Module
/// Provides web search functionality using DuckDuckGo Lite
use serde::{Deserialize, Serialize};
use scraper::{Html, Selector};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>, // Actual page content (fetched and extracted)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    pub query: String,
    pub results: Vec<SearchResult>,
    pub num_results: usize,
}

/// Extract clean text from HTML
fn extract_text_from_html(html: &str) -> String {
    let document = Html::parse_document(html);

    // Remove script and style tags
    let mut text = String::new();

    // Try to find main content areas first (common patterns)
    let main_selectors = [
        "main", "article", "#content", ".content", "#main", ".main",
        ".post-content", ".entry-content", "[role='main']"
    ];

    let mut found_main = false;
    for selector_str in &main_selectors {
        if let Ok(selector) = Selector::parse(selector_str) {
            for element in document.select(&selector) {
                let element_text: String = element.text().collect();
                if element_text.trim().len() > 100 {  // Only use if substantial content
                    text.push_str(&element_text);
                    text.push('\n');
                    found_main = true;
                }
            }
        }
        if found_main {
            break;
        }
    }

    // If no main content found, fall back to body
    if !found_main {
        if let Ok(body_selector) = Selector::parse("body") {
            for element in document.select(&body_selector) {
                text.push_str(&element.text().collect::<String>());
            }
        }
    }

    // Clean up whitespace
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(5000)  // Limit to ~5000 chars to avoid overwhelming the LLM
        .collect()
}

/// Fetch page content from a URL and extract text
async fn fetch_page_content(url: &str) -> Result<String, String> {
    println!("[WebFetch] Fetching content from: {}", url);

    let client = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0")
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Failed to fetch page: {}", e))?;

    let html = response
        .text()
        .await
        .map_err(|e| format!("Failed to read page: {}", e))?;

    let text = extract_text_from_html(&html);

    println!("[WebFetch] Extracted {} chars from page", text.len());

    Ok(text)
}

/// Search DuckDuckGo and fetch content from top results
pub async fn search_with_content(query: &str, max_results: usize, fetch_top_n: usize) -> Result<SearchResponse, String> {
    // First, get search results
    let mut response = search_duckduckgo(query, max_results).await?;

    // Fetch content from top N results
    let fetch_count = fetch_top_n.min(response.results.len());
    println!("[WebSearch] Fetching content from top {} results", fetch_count);

    for i in 0..fetch_count {
        let url = &response.results[i].url;

        // Try to fetch content (ignore failures for individual pages)
        match fetch_page_content(url).await {
            Ok(content) => {
                println!("[WebFetch] ✅ Successfully fetched {} chars from result #{}", content.len(), i + 1);
                response.results[i].content = Some(content);
            }
            Err(e) => {
                println!("[WebFetch] ⚠️ Failed to fetch result #{}: {}", i + 1, e);
                // Continue with other results
            }
        }
    }

    Ok(response)
}

/// Search DuckDuckGo Lite and return results
pub async fn search_duckduckgo(query: &str, max_results: usize) -> Result<SearchResponse, String> {
    println!("[WebSearch] Searching DuckDuckGo Lite for: {}", query);

    // DDG Lite has lighter bot detection than the main HTML endpoint
    let url = format!("https://lite.duckduckgo.com/lite/?q={}", urlencoding::encode(query));

    let client = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0")
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let response = client
        .get(&url)
        .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        .header("Accept-Language", "en-US,en;q=0.5")
        .send()
        .await
        .map_err(|e| format!("Failed to fetch search results: {}", e))?;

    let html = response
        .text()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?;

    let document = Html::parse_document(&html);

    // DDG Lite uses a table layout: a.result-link for title+URL, td.result-snippet for snippet
    let link_selector = Selector::parse("a.result-link")
        .map_err(|e| format!("Failed to parse link selector: {}", e))?;
    let snippet_selector = Selector::parse("td.result-snippet")
        .map_err(|e| format!("Failed to parse snippet selector: {}", e))?;

    let links: Vec<_> = document.select(&link_selector).collect();
    let snippets: Vec<_> = document.select(&snippet_selector).collect();

    let mut results = Vec::new();

    for (i, link_el) in links.iter().enumerate() {
        if results.len() >= max_results {
            break;
        }

        let title = link_el.text().collect::<String>().trim().to_string();
        if title.is_empty() {
            continue;
        }

        let raw_href = link_el.value().attr("href").unwrap_or("").to_string();

        // DDG redirect format: //duckduckgo.com/l/?uddg=ENCODED_URL&rut=...
        let url = if let Some(uddg_start) = raw_href.find("uddg=") {
            let encoded = raw_href[uddg_start + 5..].split('&').next().unwrap_or("");
            urlencoding::decode(encoded)
                .map(|s| s.to_string())
                .unwrap_or_else(|_| format!("https:{}", raw_href))
        } else if raw_href.starts_with("//") {
            format!("https:{}", raw_href)
        } else {
            raw_href
        };

        if url.is_empty() {
            continue;
        }

        let snippet = snippets.get(i)
            .map(|e| e.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        results.push(SearchResult {
            title,
            url,
            snippet,
            content: None,
        });
    }

    println!("[WebSearch] Found {} results", results.len());

    Ok(SearchResponse {
        query: query.to_string(),
        results: results.clone(),
        num_results: results.len(),
    })
}

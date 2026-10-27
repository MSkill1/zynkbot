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
    /// Which engine answered: duckduckgo, brave, searxng, anthropic or openai (2026-10-27).
    #[serde(default)]
    pub engine: String,
    /// An answer written by the provider's own search tool, with `results` as its
    /// sources. None from the engines that return results only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
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
pub async fn search_with_content(query: &str, max_results: usize, fetch_top_n: usize, backend: Option<&str>) -> Result<SearchResponse, String> {
    // First, get search results
    let mut response = search(query, max_results, backend).await?;
    // A provider's own search has read the pages already and written the answer.
    if response.answer.is_some() {
        return Ok(response);
    }

    // Fetch content from top N results. One after another used to mean up to
    // fetch_count * 10s of the total wait, all of it silent on a hands-free question
    // (Mike, 2026-09-20). Fetched together instead: the wait is now the slowest one
    // page, not the sum of all of them.
    let fetch_count = fetch_top_n.min(response.results.len());
    println!("[WebSearch] Fetching content from top {} results (in parallel)", fetch_count);

    let fetches = response.results[..fetch_count].iter().map(|r| {
        let url = r.url.clone();
        async move { let result = fetch_page_content(&url).await; (url, result) }
    });
    for (i, (url, result)) in futures::future::join_all(fetches).await.into_iter().enumerate() {
        match result {
            Ok(content) => {
                println!("[WebFetch] ✅ Successfully fetched {} chars from result #{}", content.len(), i + 1);
                response.results[i].content = Some(content);
            }
            Err(e) => {
                println!("[WebFetch] ⚠️ Failed to fetch result #{} ({}): {}", i + 1, url, e);
                // Continue with other results
            }
        }
    }

    Ok(response)
}

/// DuckDuckGo answers a burst of searches from one address with a "select all squares
/// containing a duck" page (HTTP 202) instead of results. Until 2026-10-27 that page parsed
/// as zero results, and the user and the model were told "no results found" for queries
/// with thousands; a credit-union thread on the Pixel lost four searches in a row that way.
pub fn is_bot_challenge(status: u16, html: &str) -> bool {
    status == 202 || html.contains("bots use DuckDuckGo") || html.contains("complete the following challenge")
}

/// Search DuckDuckGo Lite and return results
pub async fn search_duckduckgo(query: &str, max_results: usize) -> Result<SearchResponse, String> {
    // Pacing (2026-10-27): DuckDuckGo blocks an address after a burst, so searches are a
    // few seconds apart, and once it has blocked us no request goes out until the block
    // is likely over; the user gets the message at once instead of a wait and a guess.
    match ddg_wait(std::time::Instant::now()) {
        Err(left) => return Err(format!("DuckDuckGo blocked this network after several quick searches; about {} more minute(s) to wait.", left.as_secs() / 60 + 1)),
        Ok(gap) if !gap.is_zero() => tokio::time::sleep(gap).await,
        Ok(_) => {}
    }
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

    let status = response.status().as_u16();
    let html = response
        .text()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?;
    if is_bot_challenge(status, &html) {
        println!("[WebSearch] DuckDuckGo answered with its human check (HTTP {}), not results", status);
        ddg_mark_blocked();
        return Err("DuckDuckGo is asking this network to prove it is human, which it does after several quick searches. Searching is blocked for a few minutes; try again later.".to_string());
    }

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

    ddg_mark_searched();
    Ok(SearchResponse {
        query: query.to_string(),
        results: results.clone(),
        num_results: results.len(),
        engine: "duckduckgo".to_string(),
        answer: None,
    })
}

#[cfg(test)]
mod challenge_tests {
    use super::is_bot_challenge;

    #[test]
    fn the_human_check_page_is_not_an_empty_result() {
        assert!(is_bot_challenge(202, "<html><body>anything</body></html>"));
        assert!(is_bot_challenge(200, "<p>Unfortunately, bots use DuckDuckGo too.</p><p>Please complete the following challenge</p>"));
        assert!(!is_bot_challenge(200, "<table><tr><td><a class=\"result-link\" href=\"//duckduckgo.com/l/?uddg=x\">Banks in Rutland</a></td></tr></table>"));
        assert!(!is_bot_challenge(200, "<html><body>No results.</body></html>"));
    }
}

// ============================================================================
// WHICH ENGINE (2026-10-27)
// ============================================================================
//
// A cloud model searches through its own provider's search tool, on the user's own key:
// Anthropic's and OpenAI's run the searches on their index and hand back the answer with
// its sources. A local model has nowhere like that, so it uses, in order, whatever is set
// up: a Brave Search key, a SearXNG server address, and failing both DuckDuckGo, scraped
// and paced. WEB_SEARCH_ENGINE=duckduckgo|brave|searxng forces one, for testing.

#[derive(Debug, Clone, PartialEq)]
pub enum Engine {
    Anthropic,
    OpenAI,
    Brave,
    SearxNG(String),
    DuckDuckGo,
}

pub fn engine_for(backend: Option<&str>) -> Engine {
    engine_for_with(backend, |k| std::env::var(k).ok())
}

/// The choice itself, with the settings passed in so it can be tested without the
/// process environment.
pub fn engine_for_with(backend: Option<&str>, lookup: impl Fn(&str) -> Option<String>) -> Engine {
    let setting = |k: &str| lookup(k).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    let forced = setting("WEB_SEARCH_ENGINE").map(|v| v.to_lowercase());
    let backend = backend.unwrap_or("").to_lowercase();
    if forced.is_none() {
        if backend == "anthropic" && setting("ANTHROPIC_API_KEY").is_some() { return Engine::Anthropic; }
        if backend == "openai" && setting("OPENAI_API_KEY").is_some() { return Engine::OpenAI; }
    }
    match forced.as_deref() {
        Some("duckduckgo") => return Engine::DuckDuckGo,
        Some("brave") if setting("BRAVE_SEARCH_API_KEY").is_some() => return Engine::Brave,
        Some("searxng") => if let Some(u) = setting("SEARXNG_URL") { return Engine::SearxNG(u) },
        _ => {}
    }
    if setting("BRAVE_SEARCH_API_KEY").is_some() { return Engine::Brave; }
    if let Some(u) = setting("SEARXNG_URL") { return Engine::SearxNG(u); }
    Engine::DuckDuckGo
}

/// Search through the engine chosen for this backend.
pub async fn search(query: &str, max_results: usize, backend: Option<&str>) -> Result<SearchResponse, String> {
    let engine = engine_for(backend);
    println!("[WebSearch] engine for {:?}: {:?}", backend.unwrap_or("-"), engine);
    match engine {
        Engine::Anthropic => {
            let key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| "No Anthropic key".to_string())?;
            let model = std::env::var("ANTHROPIC_MODEL").ok().filter(|m| !m.trim().is_empty()).unwrap_or_else(|| "claude-sonnet-5".to_string());
            let a = crate::llm::anthropic::search_and_answer(&key, &model, query, 3).await.map_err(|e| format!("Anthropic web search: {}", e))?;
            Ok(from_answer(query, a, "anthropic"))
        }
        Engine::OpenAI => {
            let key = std::env::var("OPENAI_API_KEY").map_err(|_| "No OpenAI key".to_string())?;
            let model = std::env::var("OPENAI_MODEL").ok().filter(|m| !m.trim().is_empty()).unwrap_or_else(|| "gpt-5.5".to_string());
            let a = crate::llm::openai::search_and_answer(&key, &model, query).await.map_err(|e| format!("OpenAI web search: {}", e))?;
            Ok(from_answer(query, a, "openai"))
        }
        Engine::Brave => {
            let key = std::env::var("BRAVE_SEARCH_API_KEY").map_err(|_| "No Brave key".to_string())?;
            search_brave(query, max_results, key.trim()).await
        }
        Engine::SearxNG(url) => search_searxng(query, max_results, &url).await,
        Engine::DuckDuckGo => search_duckduckgo(query, max_results).await,
    }
}

fn from_answer(query: &str, a: crate::llm::WebAnswer, engine: &str) -> SearchResponse {
    let results: Vec<SearchResult> = a.sources.into_iter()
        .map(|s| SearchResult { title: s.title, url: s.url, snippet: s.snippet, content: None })
        .collect();
    SearchResponse { query: query.to_string(), num_results: results.len(), results, engine: engine.to_string(), answer: Some(a.answer) }
}

// ---- Brave Search API -------------------------------------------------------

pub async fn search_brave(query: &str, max_results: usize, key: &str) -> Result<SearchResponse, String> {
    let url = format!("https://api.search.brave.com/res/v1/web/search?q={}&count={}", urlencoding::encode(query), max_results.clamp(1, 20));
    println!("[WebSearch] Brave Search for: {}", query);
    let response = reqwest::Client::new()
        .get(&url)
        .header("Accept", "application/json")
        .header("X-Subscription-Token", key)
        .send()
        .await
        .map_err(|e| format!("Brave Search: {}", e))?;
    let status = response.status();
    let text = response.text().await.map_err(|e| format!("Brave Search: {}", e))?;
    if !status.is_success() {
        let short: String = text.chars().take(200).collect();
        return Err(match status.as_u16() {
            401 | 403 => "Brave Search refused the key; check it in Keys and servers.".to_string(),
            429 => "Brave Search: this key's quota is used up for now.".to_string(),
            _ => format!("Brave Search answered {}: {}", status, short),
        });
    }
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("Brave Search: unreadable answer: {}", e))?;
    Ok(brave_results(query, &v, max_results))
}

/// Results out of Brave's JSON: `web.results[].{title,url,description}`.
pub fn brave_results(query: &str, v: &serde_json::Value, max_results: usize) -> SearchResponse {
    let results: Vec<SearchResult> = v["web"]["results"].as_array().map(|a| a.as_slice()).unwrap_or(&[]).iter()
        .filter_map(|r| {
            let url = r["url"].as_str()?.to_string();
            Some(SearchResult { title: r["title"].as_str().unwrap_or("").to_string(), url, snippet: r["description"].as_str().unwrap_or("").to_string(), content: None })
        })
        .take(max_results)
        .collect();
    SearchResponse { query: query.to_string(), num_results: results.len(), results, engine: "brave".to_string(), answer: None }
}

// ---- SearXNG ------------------------------------------------------------------

pub async fn search_searxng(query: &str, max_results: usize, base_url: &str) -> Result<SearchResponse, String> {
    let base = base_url.trim().trim_end_matches('/');
    let url = format!("{}/search?q={}&format=json", base, urlencoding::encode(query));
    println!("[WebSearch] SearXNG ({}) for: {}", base, query);
    let response = reqwest::Client::builder()
        .user_agent("Zynkbot")
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(|e| format!("Search server: {}", e))?
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Search server {}: {}", base, e))?;
    let status = response.status();
    let text = response.text().await.map_err(|e| format!("Search server: {}", e))?;
    if status.as_u16() == 403 {
        return Err(format!("The search server {} refuses JSON output. Its settings.yml needs `json` under search → formats.", base));
    }
    if !status.is_success() {
        return Err(format!("The search server {} answered {}.", base, status));
    }
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|_| format!("The search server {} did not answer with JSON; is the address a SearXNG instance?", base))?;
    Ok(searxng_results(query, &v, max_results))
}

/// Results out of SearXNG's JSON: `results[].{title,url,content}`.
pub fn searxng_results(query: &str, v: &serde_json::Value, max_results: usize) -> SearchResponse {
    let results: Vec<SearchResult> = v["results"].as_array().map(|a| a.as_slice()).unwrap_or(&[]).iter()
        .filter_map(|r| {
            let url = r["url"].as_str()?.to_string();
            Some(SearchResult { title: r["title"].as_str().unwrap_or("").to_string(), url, snippet: r["content"].as_str().unwrap_or("").to_string(), content: None })
        })
        .take(max_results)
        .collect();
    SearchResponse { query: query.to_string(), num_results: results.len(), results, engine: "searxng".to_string(), answer: None }
}

// ---- DuckDuckGo pacing ----------------------------------------------------------

/// (when the last DuckDuckGo search went out, until when DuckDuckGo is known to block us)
static DDG_STATE: std::sync::Mutex<(Option<std::time::Instant>, Option<std::time::Instant>)> = std::sync::Mutex::new((None, None));
const DDG_GAP: std::time::Duration = std::time::Duration::from_secs(3);
const DDG_BLOCK: std::time::Duration = std::time::Duration::from_secs(5 * 60);

/// Ok(how long to wait before the next search), or Err(how long the block has left).
fn ddg_wait(now: std::time::Instant) -> Result<std::time::Duration, std::time::Duration> {
    let state = DDG_STATE.lock().unwrap_or_else(|e| e.into_inner());
    ddg_wait_from(state.0, state.1, now)
}

fn ddg_wait_from(last: Option<std::time::Instant>, blocked_until: Option<std::time::Instant>, now: std::time::Instant) -> Result<std::time::Duration, std::time::Duration> {
    if let Some(until) = blocked_until {
        if until > now { return Err(until - now); }
    }
    Ok(last.map(|l| DDG_GAP.saturating_sub(now.saturating_duration_since(l))).unwrap_or_default())
}

fn ddg_mark_searched() {
    let mut state = DDG_STATE.lock().unwrap_or_else(|e| e.into_inner());
    state.0 = Some(std::time::Instant::now());
    state.1 = None;
}

fn ddg_mark_blocked() {
    let mut state = DDG_STATE.lock().unwrap_or_else(|e| e.into_inner());
    state.0 = Some(std::time::Instant::now());
    state.1 = Some(std::time::Instant::now() + DDG_BLOCK);
}

#[cfg(test)]
mod engine_tests {
    use super::*;
    use std::collections::HashMap;

    fn settings(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn a_cloud_model_searches_through_its_provider_and_a_local_one_through_what_is_set_up() {
        let s = settings(&[("ANTHROPIC_API_KEY", "sk-ant"), ("OPENAI_API_KEY", "sk"), ("BRAVE_SEARCH_API_KEY", "b"), ("SEARXNG_URL", "https://search.example.org/")]);
        let look = |k: &str| s.get(k).cloned();
        assert_eq!(engine_for_with(Some("anthropic"), look), Engine::Anthropic);
        assert_eq!(engine_for_with(Some("openai"), look), Engine::OpenAI);
        assert_eq!(engine_for_with(Some("custom"), look), Engine::Brave, "a local model takes Brave first");
        let no_brave = settings(&[("ANTHROPIC_API_KEY", "sk-ant"), ("SEARXNG_URL", "https://search.example.org/")]);
        let look2 = |k: &str| no_brave.get(k).cloned();
        assert_eq!(engine_for_with(Some("local"), look2), Engine::SearxNG("https://search.example.org/".into()));
        assert_eq!(engine_for_with(Some("xai"), look2), Engine::SearxNG("https://search.example.org/".into()), "xAI has no search tool here; it searches like a local model");
        let nothing = settings(&[("ANTHROPIC_API_KEY", "")]);
        let look3 = |k: &str| nothing.get(k).cloned();
        assert_eq!(engine_for_with(Some("anthropic"), look3), Engine::DuckDuckGo, "an empty key is no key");
        assert_eq!(engine_for_with(None, look3), Engine::DuckDuckGo);
    }

    #[test]
    fn the_engine_can_be_forced_for_testing() {
        let s = settings(&[("ANTHROPIC_API_KEY", "sk-ant"), ("BRAVE_SEARCH_API_KEY", "b"), ("WEB_SEARCH_ENGINE", "duckduckgo")]);
        assert_eq!(engine_for_with(Some("anthropic"), |k| s.get(k).cloned()), Engine::DuckDuckGo);
        let s2 = settings(&[("ANTHROPIC_API_KEY", "sk-ant"), ("WEB_SEARCH_ENGINE", "brave")]);
        assert_eq!(engine_for_with(Some("anthropic"), |k| s2.get(k).cloned()), Engine::DuckDuckGo, "forcing an engine that is not set up falls through");
    }

    #[test]
    fn brave_and_searxng_results_are_read() {
        let b = serde_json::json!({"web": {"results": [
            {"title": "Heritage Family Credit Union", "url": "https://www.hfcuvt.com/", "description": "Serving Rutland County since 1956."},
            {"title": "no url"}, {"title": "Second", "url": "https://example.org/2", "description": "two"}
        ]}});
        let r = brave_results("q", &b, 5);
        assert_eq!((r.engine.as_str(), r.num_results, r.results[0].snippet.as_str()), ("brave", 2, "Serving Rutland County since 1956."));
        let x = serde_json::json!({"query": "q", "results": [
            {"title": "VSECU", "url": "https://www.vsecu.com/", "content": "Vermont State Employees Credit Union", "engine": "brave"},
            {"title": "Green Mountain CU", "url": "https://gmcu.org/", "content": "..."}
        ]});
        let r = searxng_results("q", &x, 1);
        assert_eq!((r.engine.as_str(), r.num_results, r.results[0].title.as_str()), ("searxng", 1, "VSECU"));
        assert!(r.answer.is_none());
    }

    #[test]
    fn duckduckgo_searches_are_spaced_and_a_block_is_remembered() {
        let t0 = std::time::Instant::now();
        assert_eq!(ddg_wait_from(None, None, t0), Ok(std::time::Duration::ZERO));
        let w = ddg_wait_from(Some(t0), None, t0 + std::time::Duration::from_secs(1)).unwrap();
        assert!(w > std::time::Duration::from_millis(1900) && w <= std::time::Duration::from_secs(2));
        assert_eq!(ddg_wait_from(Some(t0), None, t0 + std::time::Duration::from_secs(10)), Ok(std::time::Duration::ZERO));
        let left = ddg_wait_from(Some(t0), Some(t0 + DDG_BLOCK), t0 + std::time::Duration::from_secs(60)).unwrap_err();
        assert_eq!(left, DDG_BLOCK - std::time::Duration::from_secs(60));
        assert!(ddg_wait_from(Some(t0), Some(t0 + DDG_BLOCK), t0 + DDG_BLOCK + std::time::Duration::from_secs(1)).is_ok());
    }
}

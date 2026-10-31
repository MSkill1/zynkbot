use super::{LLMError, LLMResponse, Message, Usage};
use futures::StreamExt;
use reqwest;
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";

/// Anthropic API request structure
#[derive(Debug, Serialize)]
#[allow(dead_code)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

// --- Streaming SSE types ---

#[derive(Debug, Deserialize)]
struct AnthropicSSEEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    delta: Option<AnthropicStreamDelta>,
    #[serde(default)]
    usage: Option<AnthropicStreamUsage>,
    /// Present on `{"type":"error",...}` events, which Anthropic sends inside an
    /// otherwise successful stream for overload and rate-limit conditions.
    #[serde(default)]
    error: Option<AnthropicStreamError>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamError {
    #[serde(rename = "type", default)]
    error_type: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamDelta {
    #[serde(rename = "type", default)]
    delta_type: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamUsage {
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    output_tokens: Option<u32>,
}

/// Anthropic API response structure
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<ContentBlock>,
    usage: UsageInfo,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    // Only some block types carry `text` — thinking, redacted_thinking and tool_use
    // blocks do not. Without a default, one such block made serde reject the whole
    // response with "error decoding response body", which is what dropped Anthropic
    // out of ensemble Phase 1 while the streaming coordinator path still worked.
    // The consumer below already filters to content_type == "text"; it just never
    // got the chance to run. The streaming structs in this file are defensive the
    // same way.
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct UsageInfo {
    input_tokens: u32,
    output_tokens: u32,
}

/// Send a message to Claude via Anthropic API
/// This replaces Python's anthropic.Anthropic().messages.create()
#[allow(dead_code)]
pub async fn send_message(
    api_key: &str,
    model: &str,
    messages: Vec<Message>,
    system_prompt: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<LLMResponse, LLMError> {
    let client = reqwest::Client::new();

    let request_body = AnthropicRequest {
        model: model.to_string(),
        max_tokens: max_tokens.unwrap_or(4096),
        messages,
        system: system_prompt,
        temperature,
        stream: None,
    };

    println!("[Rust Anthropic] Sending request to Claude: {}", model);

    let response = client
        .post(ANTHROPIC_API_URL)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&request_body)
        .send()
        .await
        .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(LLMError::APIError(format!(
            "Status {}: {}",
            status, error_text
        )));
    }

    let anthropic_response: AnthropicResponse = response
        .json()
        .await
        .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

    // Extract text from content blocks
    let content = anthropic_response
        .content
        .into_iter()
        .filter(|block| block.content_type == "text")
        .map(|block| block.text)
        .collect::<Vec<String>>()
        .join("\n");

    println!("[Rust Anthropic] ✅ Received response from Claude");

    Ok(LLMResponse {
        content,
        model: anthropic_response.model,
        usage: Some(Usage {
            input_tokens: anthropic_response.usage.input_tokens,
            output_tokens: anthropic_response.usage.output_tokens,
        }),
    })
}

/// Send a message to Claude via Anthropic API with SSE token streaming.
/// Calls `on_token` for each text delta as it arrives, then returns the
/// complete LLMResponse (with accumulated content) when the stream closes.
/// True for the errors Anthropic asks callers to retry after a short wait:
/// HTTP 529 / `overloaded_error` and HTTP 429 / `rate_limit_error`.
pub(crate) fn is_retryable(err: &LLMError) -> bool {
    let msg = err.to_string();
    msg.contains("overloaded") || msg.contains("529") || msg.contains("rate_limit") || msg.contains("429")
}

/// Streams a reply, retrying once after a short pause if Anthropic reports overload
/// or a rate limit before any text has been emitted. A retry after text has reached
/// the page would repeat that text, so it is not attempted.
#[allow(dead_code)]
pub async fn send_message_streaming<F>(
    api_key: &str,
    model: &str,
    messages: Vec<Message>,
    system_prompt: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    on_token: F,
) -> Result<LLMResponse, LLMError>
where
    F: Fn(String),
{
    let emitted = std::sync::atomic::AtomicBool::new(false);
    let first = send_message_streaming_once(
        api_key, model, messages.clone(), system_prompt.clone(), max_tokens, temperature,
        |t| { emitted.store(true, std::sync::atomic::Ordering::SeqCst); on_token(t) },
    ).await;
    match first {
        Err(ref e) if is_retryable(e) && !emitted.load(std::sync::atomic::Ordering::SeqCst) => {
            println!("[Rust Anthropic] Retrying once in 3 s after: {}", e);
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            send_message_streaming_once(
                api_key, model, messages, system_prompt, max_tokens, temperature, on_token,
            ).await
        }
        other => other,
    }
}

async fn send_message_streaming_once<F>(
    api_key: &str,
    model: &str,
    messages: Vec<Message>,
    system_prompt: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    on_token: F,
) -> Result<LLMResponse, LLMError>
where
    F: Fn(String),
{
    let client = reqwest::Client::new();

    let request_body = AnthropicRequest {
        model: model.to_string(),
        max_tokens: max_tokens.unwrap_or(4096),
        messages,
        system: system_prompt,
        temperature,
        stream: Some(true),
    };

    println!("[Rust Anthropic] Starting streaming request to Claude: {}", model);

    let response = client
        .post(ANTHROPIC_API_URL)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&request_body)
        .send()
        .await
        .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(LLMError::APIError(format!(
            "Status {}: {}",
            status, error_text
        )));
    }

    let mut byte_stream = response.bytes_stream();
    let mut line_buffer = String::new();
    let mut full_text = String::new();
    let mut input_tokens: u32 = 0;
    let mut output_tokens: u32 = 0;

    while let Some(chunk_result) = byte_stream.next().await {
        if crate::GENERATION_CANCELLED.load(std::sync::atomic::Ordering::SeqCst) {
            println!("[Rust Anthropic] 🛑 Cancelled mid-stream");
            break;
        }
        let bytes = chunk_result.map_err(|e| LLMError::RequestFailed(e.to_string()))?;
        let text = String::from_utf8_lossy(&bytes);
        line_buffer.push_str(&text);

        // Process all complete lines in the buffer
        while let Some(newline_pos) = line_buffer.find('\n') {
            let line = line_buffer[..newline_pos].trim().to_string();
            line_buffer = line_buffer[newline_pos + 1..].to_string();

            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }

            let event: AnthropicSSEEvent = match serde_json::from_str(data) {
                Ok(e) => e,
                Err(_) => continue, // Skip malformed lines
            };

            match event.event_type.as_str() {
                "content_block_delta" => {
                    if let Some(delta) = event.delta {
                        if delta.delta_type.as_deref() == Some("text_delta") {
                            if let Some(text) = delta.text {
                                full_text.push_str(&text);
                                on_token(text);
                            }
                        }
                    }
                }
                "message_start" => {
                    if let Some(usage) = event.usage {
                        input_tokens = usage.input_tokens.unwrap_or(0);
                    }
                }
                "message_delta" => {
                    if let Some(usage) = event.usage {
                        if let Some(tokens) = usage.output_tokens {
                            output_tokens = tokens;
                        }
                    }
                }
                // Until 2026-10-21 this fell into the catch-all below: four replies
                // that day ended as "Streaming complete (0 tokens)" with the page
                // showing an empty bubble, because the overload error never surfaced.
                "error" => {
                    let (kind, msg) = event.error
                        .map(|e| (e.error_type.unwrap_or_default(), e.message.unwrap_or_default()))
                        .unwrap_or_default();
                    println!("[Rust Anthropic] ✗ Error event in stream: {} — {}", kind, msg);
                    return Err(LLMError::APIError(format!("{}: {}", kind, msg)));
                }
                "ping" | "content_block_start" | "content_block_stop" | "message_stop" => {}
                other => println!("[Rust Anthropic] Unrecognised stream event: {}", other),
            }
        }
    }

    println!("[Rust Anthropic] ✅ Streaming complete ({} tokens)", output_tokens);

    Ok(LLMResponse {
        content: full_text,
        model: model.to_string(),
        usage: Some(Usage {
            input_tokens,
            output_tokens,
        }),
    })
}

/// Send a vision request (text + one or more images) to Claude with streaming.
/// Builds the multipart content array required by Anthropic's vision API.
pub async fn send_vision_streaming<F>(
    api_key: &str,
    model: &str,
    text: &str,
    images: &[super::ImageAttachment],
    system_prompt: Option<String>,
    max_tokens: Option<u32>,
    on_token: F,
) -> Result<LLMResponse, LLMError>
where
    F: Fn(String),
{
    let client = reqwest::Client::new();

    let mut content_blocks: Vec<serde_json::Value> = images.iter().map(|img| serde_json::json!({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": img.mime_type,
            "data": img.base64
        }
    })).collect();
    content_blocks.push(serde_json::json!({ "type": "text", "text": text }));
    let content = serde_json::Value::Array(content_blocks);

    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": max_tokens.unwrap_or(4096),
        "stream": true,
        "messages": [{"role": "user", "content": content}]
    });

    if let Some(sys) = system_prompt {
        body["system"] = serde_json::json!(sys);
    }

    println!("[Rust Anthropic] Starting vision streaming request to Claude: {}", model);

    let response = client
        .post(ANTHROPIC_API_URL)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        return Err(LLMError::APIError(format!("Status {}: {}", status, error_text)));
    }

    let mut byte_stream = response.bytes_stream();
    let mut line_buffer = String::new();
    let mut full_text = String::new();
    let mut input_tokens: u32 = 0;
    let mut output_tokens: u32 = 0;

    while let Some(chunk_result) = byte_stream.next().await {
        if crate::GENERATION_CANCELLED.load(std::sync::atomic::Ordering::SeqCst) {
            println!("[Rust Anthropic] 🛑 Cancelled mid-stream");
            break;
        }
        let bytes = chunk_result.map_err(|e| LLMError::RequestFailed(e.to_string()))?;
        let text = String::from_utf8_lossy(&bytes);
        line_buffer.push_str(&text);

        while let Some(newline_pos) = line_buffer.find('\n') {
            let line = line_buffer[..newline_pos].trim().to_string();
            line_buffer = line_buffer[newline_pos + 1..].to_string();

            if !line.starts_with("data: ") { continue; }
            let data = &line[6..];
            if data == "[DONE]" { break; }

            let event: AnthropicSSEEvent = match serde_json::from_str(data) {
                Ok(e) => e,
                Err(_) => continue,
            };

            match event.event_type.as_str() {
                "content_block_delta" => {
                    if let Some(delta) = event.delta {
                        if delta.delta_type.as_deref() == Some("text_delta") {
                            if let Some(text) = delta.text {
                                full_text.push_str(&text);
                                on_token(text);
                            }
                        }
                    }
                }
                "message_start" => {
                    if let Some(usage) = event.usage {
                        input_tokens = usage.input_tokens.unwrap_or(0);
                    }
                }
                "message_delta" => {
                    if let Some(usage) = event.usage {
                        if let Some(tokens) = usage.output_tokens {
                            output_tokens = tokens;
                        }
                    }
                }
                // Until 2026-10-21 this fell into the catch-all below: four replies
                // that day ended as "Streaming complete (0 tokens)" with the page
                // showing an empty bubble, because the overload error never surfaced.
                "error" => {
                    let (kind, msg) = event.error
                        .map(|e| (e.error_type.unwrap_or_default(), e.message.unwrap_or_default()))
                        .unwrap_or_default();
                    println!("[Rust Anthropic] ✗ Error event in stream: {} — {}", kind, msg);
                    return Err(LLMError::APIError(format!("{}: {}", kind, msg)));
                }
                "ping" | "content_block_start" | "content_block_stop" | "message_stop" => {}
                other => println!("[Rust Anthropic] Unrecognised stream event: {}", other),
            }
        }
    }

    println!("[Rust Anthropic] ✅ Vision streaming complete ({} tokens)", output_tokens);

    Ok(LLMResponse {
        content: full_text,
        model: model.to_string(),
        usage: Some(Usage { input_tokens, output_tokens }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_event_in_stream_parses() {
        let e: AnthropicSSEEvent = serde_json::from_str(
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        ).unwrap();
        assert_eq!(e.event_type, "error");
        let err = e.error.unwrap();
        assert_eq!(err.error_type.as_deref(), Some("overloaded_error"));
        assert_eq!(err.message.as_deref(), Some("Overloaded"));
        assert!(is_retryable(&LLMError::APIError("overloaded_error: Overloaded".into())));
        assert!(is_retryable(&LLMError::APIError("Status 529 Overloaded".into())));
        assert!(!is_retryable(&LLMError::APIError("invalid_request_error: bad".into())));
    }

    /// A response mixing a thinking block with a text block must parse, and only the
    /// text must survive. Guards the ensemble Phase 1 failure documented above.
    #[test]
    fn response_with_non_text_block_parses() {
        let body = r#"{
            "id": "msg_123",
            "model": "claude-opus-5",
            "content": [
                {"type": "thinking", "thinking": "considering the question"},
                {"type": "text", "text": "Here is the answer."}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;

        let parsed: AnthropicResponse =
            serde_json::from_str(body).expect("response with a thinking block must deserialize");

        let content = parsed
            .content
            .into_iter()
            .filter(|b| b.content_type == "text")
            .map(|b| b.text)
            .collect::<Vec<String>>()
            .join("\n");

        assert_eq!(content, "Here is the answer.");
    }

    /// The ordinary all-text case must keep working unchanged.
    #[test]
    fn response_with_only_text_blocks_parses() {
        let body = r#"{
            "id": "msg_456",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "Plain answer."}],
            "usage": {"input_tokens": 3, "output_tokens": 2}
        }"#;
        let parsed: AnthropicResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.content.len(), 1);
        assert_eq!(parsed.content[0].text, "Plain answer.");
    }
}

// ============================================================================
// THE WEB THROUGH ANTHROPIC'S OWN SEARCH TOOL (2026-10-27)
// ============================================================================

/// One request with the `web_search` server tool: the user's query as the instruction,
/// at most `max_searches` searches, Claude's answer with its sources back. Anthropic runs
/// the searches on its own index and bills the user's key per search plus tokens. The
/// page text itself comes back encrypted, so the answer is Claude's, not ours.
pub async fn search_and_answer(api_key: &str, model: &str, query: &str, max_searches: u32, on_token: super::OnToken<'_>) -> Result<super::WebAnswer, LLMError> {
    let body = serde_json::json!({
        "model": model,
        "max_tokens": 1500,
        // Streamed (2026-10-28): the answer's words reach the screen as they are written
        // instead of all at once after the searches.
        "stream": true,
        "system": "You answer from web search results. Search for what the user asks, then answer directly and concisely from what you find. Do not narrate the searching or describe your process.",
        "messages": [{"role": "user", "content": format!("Search the web for: {}", query)}],
        // The fetch tool beside the search tool (2026-10-27): Claude can find a page by name
        // and then read it whole, or read an address the user gave. No charge beyond tokens.
        "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": max_searches},
            {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 3, "max_content_tokens": 20000},
        ],
    });
    println!("[Rust Anthropic] Web search through Claude ({}): {}", model, query);
    let response = reqwest::Client::new()
        .post(ANTHROPIC_API_URL)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| LLMError::RequestFailed(e.to_string()))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.map_err(|e| LLMError::RequestFailed(e.to_string()))?;
        return Err(LLMError::APIError(format!("Status {}: {}", status, text)));
    }
    let mut blocks: Vec<serde_json::Value> = Vec::new();
    let mut searches = 0u64;
    super::read_sse(response, |ev| {
        if ev["type"].as_str() == Some("message_delta") {
            searches = ev["usage"]["server_tool_use"]["web_search_requests"].as_u64().unwrap_or(searches);
        }
        fold_stream_event(&mut blocks, ev, on_token)
    }).await?;
    let answer = parse_web_answer(&serde_json::json!({ "content": blocks }))?;
    println!("[Rust Anthropic] ✅ Web answer: {} chars, {} sources, {} searches", answer.answer.len(), answer.sources.len(), searches);
    Ok(answer)
}

/// One streamed event folded into the content blocks a non-streamed response would have
/// held, so `parse_web_answer` reads both the same way. Text deltas also go to `on_token`.
pub fn fold_stream_event(blocks: &mut Vec<serde_json::Value>, ev: &serde_json::Value, on_token: super::OnToken<'_>) -> Result<(), LLMError> {
    match ev["type"].as_str().unwrap_or("") {
        "content_block_start" => {
            let mut b = ev["content_block"].clone();
            if b["type"].as_str() == Some("text") && b["text"].is_null() { b["text"] = serde_json::json!(""); }
            blocks.push(b);
        }
        "content_block_delta" => {
            let idx = ev["index"].as_u64().unwrap_or(0) as usize;
            let d = &ev["delta"];
            if let Some(block) = blocks.get_mut(idx) {
                match d["type"].as_str().unwrap_or("") {
                    "text_delta" => {
                        let t = d["text"].as_str().unwrap_or("");
                        let so_far = block["text"].as_str().unwrap_or("").to_string();
                        block["text"] = serde_json::json!(format!("{}{}", so_far, t));
                        if !t.is_empty() { on_token(t); }
                    }
                    "citations_delta" => {
                        if !block["citations"].is_array() { block["citations"] = serde_json::json!([]); }
                        if let Some(arr) = block["citations"].as_array_mut() { arr.push(d["citation"].clone()); }
                    }
                    _ => {}
                }
            }
        }
        "error" => {
            let msg = ev["error"]["message"].as_str().unwrap_or("the stream reported an error");
            return Err(LLMError::APIError(msg.to_string()));
        }
        _ => {}
    }
    Ok(())
}

/// The answer text and the sources out of a Messages response that used the web search
/// and web fetch tools: text blocks joined, every `web_search_result` and every fetched
/// page once, each citation's text as the snippet of its page.
pub fn parse_web_answer(v: &serde_json::Value) -> Result<super::WebAnswer, LLMError> {
    let mut answer = String::new();
    let mut sources: Vec<super::WebSource> = Vec::new();
    let mut snippets: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for block in v["content"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        match block["type"].as_str().unwrap_or("") {
            "text" => {
                if let Some(t) = block["text"].as_str() { answer.push_str(t); }
                for c in block["citations"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                    if let (Some(url), Some(cited)) = (c["url"].as_str(), c["cited_text"].as_str()) {
                        snippets.entry(url.to_string()).or_insert_with(|| cited.trim().to_string());
                    }
                }
            }
            "web_fetch_tool_result" => {
                let c = &block["content"];
                if c["type"].as_str() == Some("web_fetch_result") {
                    let url = c["url"].as_str().unwrap_or("").to_string();
                    if !url.is_empty() && !sources.iter().any(|s| s.url == url) {
                        let title = c["content"]["title"].as_str().unwrap_or("").to_string();
                        sources.push(super::WebSource { title, url, snippet: String::new() });
                    }
                }
                // A fetch that failed is not an error for the whole answer: Claude says so in its text.
            }
            "web_search_tool_result" => {
                if let Some(items) = block["content"].as_array() {
                    for item in items {
                        if item["type"].as_str() != Some("web_search_result") { continue; }
                        let url = item["url"].as_str().unwrap_or("").to_string();
                        if url.is_empty() || sources.iter().any(|s| s.url == url) { continue; }
                        sources.push(super::WebSource { title: item["title"].as_str().unwrap_or("").to_string(), url, snippet: String::new() });
                    }
                } else if let Some(code) = block["content"]["error_code"].as_str() {
                    return Err(LLMError::APIError(format!("Anthropic web search: {}", code)));
                }
            }
            _ => {}
        }
    }
    for s in &mut sources {
        if let Some(sn) = snippets.get(&s.url) { s.snippet = sn.clone(); }
    }
    let answer = answer.trim().to_string();
    if answer.is_empty() && sources.is_empty() {
        return Err(LLMError::InvalidResponse("the web search returned neither an answer nor sources".to_string()));
    }
    Ok(super::WebAnswer { answer, sources })
}

#[cfg(test)]
mod web_answer_tests {
    use super::parse_web_answer;

    #[test]
    fn answer_sources_and_snippets_come_out_of_the_response() {
        let v: serde_json::Value = serde_json::json!({
            "content": [
                {"type": "text", "text": "I'll search for that."},
                {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "heritage family credit union rutland"}},
                {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                    {"type": "web_search_result", "url": "https://www.hfcuvt.com/", "title": "Heritage Family Credit Union", "encrypted_content": "x", "page_age": "2026"},
                    {"type": "web_search_result", "url": "https://www.hfcuvt.com/business", "title": "Business banking", "encrypted_content": "y"},
                    {"type": "web_search_result", "url": "https://www.hfcuvt.com/", "title": "dup", "encrypted_content": "z"}
                ]},
                {"type": "text", "text": "Heritage Family Credit Union is at hfcuvt.com. ", "citations": [
                    {"type": "web_search_result_location", "url": "https://www.hfcuvt.com/", "title": "Heritage Family Credit Union", "encrypted_index": "e", "cited_text": "Heritage Family Credit Union serves Rutland County..."}
                ]},
                {"type": "text", "text": "Its business page lists checking accounts."}
            ],
            "usage": {"server_tool_use": {"web_search_requests": 1}}
        });
        let a = parse_web_answer(&v).unwrap();
        assert_eq!(a.answer, "I'll search for that.Heritage Family Credit Union is at hfcuvt.com. Its business page lists checking accounts.");
        assert_eq!(a.sources.len(), 2);
        assert_eq!(a.sources[0].snippet, "Heritage Family Credit Union serves Rutland County...");
        assert_eq!(a.sources[1].title, "Business banking");
    }

    #[test]
    fn a_fetched_page_is_a_source_too() {
        let v = serde_json::json!({"content": [
            {"type": "server_tool_use", "id": "f1", "name": "web_fetch", "input": {"url": "https://containai.ai/"}},
            {"type": "web_fetch_tool_result", "tool_use_id": "f1", "content": {"type": "web_fetch_result", "url": "https://containai.ai/",
                "content": {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "Zynkbot..."}, "title": "Contain AI"}, "retrieved_at": "2026-10-27T15:00:00Z"}},
            {"type": "web_fetch_tool_result", "tool_use_id": "f2", "content": {"type": "web_fetch_tool_result_error", "error_code": "url_not_accessible"}},
            {"type": "text", "text": "The site describes Zynkbot. The second page could not be read."}
        ]});
        let a = parse_web_answer(&v).unwrap();
        assert_eq!(a.sources.len(), 1);
        assert_eq!((a.sources[0].title.as_str(), a.sources[0].url.as_str()), ("Contain AI", "https://containai.ai/"));
        assert!(a.answer.starts_with("The site describes"));
    }

    #[test]
    fn a_tool_error_is_an_error_not_an_empty_answer() {
        let v = serde_json::json!({"content": [
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": {"type": "web_search_tool_result_error", "error_code": "too_many_requests"}}
        ]});
        assert!(parse_web_answer(&v).unwrap_err().to_string().contains("too_many_requests"));
        assert!(parse_web_answer(&serde_json::json!({"content": []})).is_err());
    }
}

#[cfg(test)]
mod web_stream_tests {
    use super::{fold_stream_event, parse_web_answer};
    use std::sync::Mutex;

    #[test]
    fn streamed_events_fold_into_the_same_answer_and_words_go_out_as_they_come() {
        let events = vec![
            serde_json::json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me look. "}}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({"type": "content_block_start", "index": 1, "content_block": {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}}}),
            serde_json::json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"query\":"}}),
            serde_json::json!({"type": "content_block_start", "index": 2, "content_block": {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "url": "https://www.hfcuvt.com/", "title": "Heritage Family Credit Union", "encrypted_content": "x"}]}}),
            serde_json::json!({"type": "content_block_start", "index": 3, "content_block": {"type": "text", "text": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 3, "delta": {"type": "text_delta", "text": "Heritage is at "}}),
            serde_json::json!({"type": "content_block_delta", "index": 3, "delta": {"type": "citations_delta", "citation": {"type": "web_search_result_location", "url": "https://www.hfcuvt.com/", "cited_text": "Serves Rutland County"}}}),
            serde_json::json!({"type": "content_block_delta", "index": 3, "delta": {"type": "text_delta", "text": "hfcuvt.com."}}),
        ];
        let words = Mutex::new(Vec::<String>::new());
        let on_token = |t: &str| words.lock().unwrap().push(t.to_string());
        let mut blocks = Vec::new();
        for ev in &events { fold_stream_event(&mut blocks, ev, &on_token).unwrap(); }
        assert_eq!(*words.lock().unwrap(), vec!["Let me look. ", "Heritage is at ", "hfcuvt.com."]);
        let a = parse_web_answer(&serde_json::json!({"content": blocks})).unwrap();
        assert_eq!(a.answer, "Let me look. Heritage is at hfcuvt.com.");
        assert_eq!(a.sources.len(), 1);
        assert_eq!(a.sources[0].snippet, "Serves Rutland County");
    }

    #[test]
    fn an_error_event_stops_the_stream() {
        let mut blocks = Vec::new();
        let ev = serde_json::json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}});
        assert!(fold_stream_event(&mut blocks, &ev, &|_| {}).unwrap_err().to_string().contains("Overloaded"));
    }
}

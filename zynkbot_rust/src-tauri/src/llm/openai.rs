use super::{LLMError, LLMResponse, Message, Usage};
use futures::StreamExt;
use reqwest;
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// OpenAI's "-pro" models (gpt-5.x-pro, o1-pro) are only served by the Responses API;
/// `/v1/chat/completions` answers 404 "This is not a chat model". Older builds offered
/// them in the picker, so a stored pro id is mapped to its chat sibling here instead of
/// failing the whole ensemble turn. Only applied when talking to api.openai.com — xAI
/// and Ollama-compatible endpoints reuse this module and own their model names.
fn chat_model_for<'a>(model: &'a str, api_url: &str) -> std::borrow::Cow<'a, str> {
    if api_url.starts_with("https://api.openai.com/") {
        if let Some(base) = model.strip_suffix("-pro") {
            println!("[Rust OpenAI] {} is not a chat model; using {} instead", model, base);
            return std::borrow::Cow::Owned(base.to_string());
        }
    }
    std::borrow::Cow::Borrowed(model)
}

/// OpenAI API request structure
#[derive(Debug, Serialize)]
#[allow(dead_code)]
struct OpenAIRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

/// How hard a reasoning model thinks before it answers (2026-11-17): "low" for gpt-5.x
/// and the o-series on api.openai.com, and for Grok 4.5 and later on api.x.ai (xAI's
/// reasoning guide: grok-4.5, 4.6 and 4.7 take low, medium or high and cannot switch
/// reasoning off). On the Pixel (2026-11-16) gpt-5.4 took 67 s to its first word with
/// no level set. Nothing is sent for gpt-4.1, gpt-4o, older Grok ids, Mistral or a
/// custom endpoint, which may reject the field. ZYNK_CLOUD_EFFORT overrides the level.
pub(crate) fn reasoning_effort_for(model: &str, api_url: &str) -> Option<String> {
    let m = model.to_lowercase();
    let supported = if api_url.starts_with("https://api.openai.com/") {
        m.starts_with("gpt-5") || (m.starts_with('o') && m[1..].starts_with(|c: char| c.is_ascii_digit()))
    } else if api_url.starts_with("https://api.x.ai/") {
        m.strip_prefix("grok-")
            .and_then(|v| v.split(|c: char| !(c.is_ascii_digit() || c == '.')).next())
            .and_then(|v| v.parse::<f32>().ok())
            .map_or(false, |v| v >= 4.5)
    } else {
        false
    };
    if !supported { return None; }
    Some(std::env::var("ZYNK_CLOUD_EFFORT").ok().map(|v| v.trim().to_lowercase()).filter(|v| !v.is_empty()).unwrap_or_else(|| "low".to_string()))
}

/// A stream that ended for length with no text: the model spent the whole allowance
/// reasoning. Named for what happened instead of "empty reply".
fn starved_of_text(text: &str, finish_reason: Option<&str>) -> Result<(), LLMError> {
    if text.trim().is_empty() && finish_reason == Some("length") {
        println!("[Rust OpenAI] ✗ The model spent its whole reply allowance reasoning and wrote no text");
        return Err(LLMError::APIError("the model spent its whole reply allowance reasoning and wrote no text".to_string()));
    }
    Ok(())
}

// --- Streaming SSE types (OpenAI-compatible, also used by xAI) ---

/// An OpenAI-compatible stream can carry `{"error": {...}}` in place of a chunk.
/// Returns the error's message when `data` is such an object.
pub(crate) fn stream_error_message(data: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(data).ok()?;
    let err = v.get("error")?;
    if err.is_null() { return None; }
    Some(
        err.get("message").and_then(|m| m.as_str()).map(|s| s.to_string())
            .or_else(|| err.as_str().map(|s| s.to_string()))
            .unwrap_or_else(|| err.to_string()),
    )
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIStreamChunk {
    pub choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIStreamChoice {
    pub delta: OpenAIStreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIStreamDelta {
    #[serde(default)]
    pub content: Option<String>,
}

/// OpenAI API response structure
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIResponse {
    id: String,
    model: String,
    choices: Vec<Choice>,
    usage: UsageInfo,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Choice {
    message: MessageContent,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MessageContent {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct UsageInfo {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// Send a message to GPT via OpenAI API
/// This replaces Python's openai.ChatCompletion.create()
#[allow(dead_code)]
pub async fn send_message(
    api_key: &str,
    model: &str,
    messages: Vec<Message>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<LLMResponse, LLMError> {
    let client = reqwest::Client::new();
    let model = chat_model_for(model, OPENAI_API_URL);

    let request_body = OpenAIRequest {
        model: model.to_string(),
        messages,
        max_completion_tokens: max_tokens,
        temperature,
        stream: None,
        reasoning_effort: reasoning_effort_for(&model, OPENAI_API_URL),
    };

    println!("[Rust OpenAI] Sending request to GPT: {}", model);

    let response = client
        .post(OPENAI_API_URL)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
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

    let openai_response: OpenAIResponse = response
        .json()
        .await
        .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

    // Extract content from first choice
    let content = openai_response
        .choices
        .first()
        .map(|choice| choice.message.content.clone())
        .ok_or_else(|| LLMError::InvalidResponse("No choices in response".to_string()))?;

    println!("[Rust OpenAI] ✅ Received response from GPT");

    Ok(LLMResponse {
        content,
        model: openai_response.model,
        usage: Some(Usage {
            input_tokens: openai_response.usage.prompt_tokens,
            output_tokens: openai_response.usage.completion_tokens,
        }),
    })
}

/// Build a plain reqwest client for external API calls.
/// Pass `true` for LAN/self-signed endpoints (Ollama), `false` for public HTTPS APIs.
pub fn default_client(accept_invalid_certs: bool) -> reqwest::Client {
    reqwest::Client::builder()
        .danger_accept_invalid_certs(accept_invalid_certs)
        .build()
        .unwrap_or_default()
}

/// Send a message via OpenAI API with SSE token streaming.
/// Also used by xAI (Grok) since it uses the OpenAI-compatible format.
/// `api_url` lets callers override the endpoint (e.g. xAI uses a different base URL).
/// `client` should be built by the caller — use `default_client(false)` for public APIs,
/// `default_client(true)` for LAN endpoints, or the ZynkSync mTLS client for peer URLs.
#[allow(dead_code)]
pub async fn send_message_streaming<F>(
    api_key: &str,
    model: &str,
    messages: Vec<Message>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    api_url: &str,
    client: reqwest::Client,
    on_token: F,
) -> Result<LLMResponse, LLMError>
where
    F: Fn(String),
{
    let model = chat_model_for(model, api_url);

    let request_body = OpenAIRequest {
        model: model.to_string(),
        messages,
        max_completion_tokens: max_tokens,
        temperature,
        stream: Some(true),
        reasoning_effort: reasoning_effort_for(&model, api_url),
    };

    println!("[Rust OpenAI] Starting streaming request to: {}", model);

    let response = client
        .post(api_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
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
    let mut output_tokens: u32 = 0;
    let mut finish_reason: Option<String> = None;

    'outer: while let Some(chunk_result) = byte_stream.next().await {
        if crate::GENERATION_CANCELLED.load(std::sync::atomic::Ordering::SeqCst) {
            println!("[Rust OpenAI] 🛑 Cancelled mid-stream");
            break 'outer;
        }
        let bytes = chunk_result.map_err(|e| LLMError::RequestFailed(e.to_string()))?;
        let text = String::from_utf8_lossy(&bytes);
        line_buffer.push_str(&text);

        while let Some(newline_pos) = line_buffer.find('\n') {
            let line = line_buffer[..newline_pos].trim().to_string();
            line_buffer = line_buffer[newline_pos + 1..].to_string();

            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];
            if data == "[DONE]" {
                break 'outer;
            }

            let chunk: OpenAIStreamChunk = match serde_json::from_str(data) {
                Ok(c) => c,
                Err(_) => {
                    // An error object in place of a chunk used to be skipped like any
                    // malformed line, leaving an empty reply (2026-10-21).
                    if let Some(msg) = stream_error_message(data) {
                        println!("[Rust OpenAI] ✗ Error in stream: {}", msg);
                        return Err(LLMError::APIError(msg));
                    }
                    continue;
                }
            };

            for choice in chunk.choices {
                if let Some(token) = choice.delta.content {
                    full_text.push_str(&token);
                    on_token(token);
                }
                if choice.finish_reason.is_some() { finish_reason = choice.finish_reason.clone(); }
                if choice.finish_reason.as_deref() == Some("stop") {
                    output_tokens = full_text.split_whitespace().count() as u32; // approximate
                }
            }
        }
    }

    println!("[Rust OpenAI] ✅ Streaming complete");
    starved_of_text(&full_text, finish_reason.as_deref())?;

    Ok(LLMResponse {
        content: full_text,
        model: model.to_string(),
        usage: Some(Usage {
            input_tokens: 0, // OpenAI doesn't provide input tokens in stream chunks
            output_tokens,
        }),
    })
}

/// Send a vision request (text + one or more images) via OpenAI-compatible API with streaming.
/// Used for both OpenAI (GPT-4o) and xAI (Grok Vision).
pub async fn send_vision_streaming<F>(
    api_key: &str,
    model: &str,
    text: &str,
    images: &[super::ImageAttachment],
    api_url: &str,
    on_token: F,
) -> Result<LLMResponse, LLMError>
where
    F: Fn(String),
{
    let client = reqwest::Client::new();

    let mut content_blocks: Vec<serde_json::Value> = images.iter().map(|img| serde_json::json!({
        "type": "image_url",
        "image_url": { "url": format!("data:{};base64,{}", img.mime_type, img.base64) }
    })).collect();
    content_blocks.push(serde_json::json!({ "type": "text", "text": text }));

    let mut body = serde_json::json!({
        "model": chat_model_for(model, api_url).as_ref(),
        "stream": true,
        "max_completion_tokens": 8192,
        "messages": [{ "role": "user", "content": content_blocks }]
    });
    if let Some(level) = reasoning_effort_for(model, api_url) {
        body["reasoning_effort"] = serde_json::json!(level);
    }

    println!("[Rust OpenAI] Starting vision streaming request to: {}", model);

    let response = client
        .post(api_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
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
    let mut output_tokens: u32 = 0;
    let mut finish_reason: Option<String> = None;

    'outer: while let Some(chunk_result) = byte_stream.next().await {
        if crate::GENERATION_CANCELLED.load(std::sync::atomic::Ordering::SeqCst) {
            println!("[Rust OpenAI] 🛑 Cancelled mid-stream");
            break 'outer;
        }
        let bytes = chunk_result.map_err(|e| LLMError::RequestFailed(e.to_string()))?;
        let text = String::from_utf8_lossy(&bytes);
        line_buffer.push_str(&text);

        while let Some(newline_pos) = line_buffer.find('\n') {
            let line = line_buffer[..newline_pos].trim().to_string();
            line_buffer = line_buffer[newline_pos + 1..].to_string();

            if !line.starts_with("data: ") { continue; }
            let data = &line[6..];
            if data == "[DONE]" { break 'outer; }

            let chunk: OpenAIStreamChunk = match serde_json::from_str(data) {
                Ok(c) => c,
                Err(_) => {
                    // An error object in place of a chunk used to be skipped like any
                    // malformed line, leaving an empty reply (2026-10-21).
                    if let Some(msg) = stream_error_message(data) {
                        println!("[Rust OpenAI] ✗ Error in stream: {}", msg);
                        return Err(LLMError::APIError(msg));
                    }
                    continue;
                }
            };

            for choice in chunk.choices {
                if let Some(token) = choice.delta.content {
                    full_text.push_str(&token);
                    on_token(token);
                }
                if choice.finish_reason.is_some() { finish_reason = choice.finish_reason.clone(); }
                if choice.finish_reason.as_deref() == Some("stop") {
                    output_tokens = full_text.split_whitespace().count() as u32;
                }
            }
        }
    }

    println!("[Rust OpenAI] ✅ Vision streaming complete");
    starved_of_text(&full_text, finish_reason.as_deref())?;

    Ok(LLMResponse {
        content: full_text,
        model: model.to_string(),
        usage: Some(Usage { input_tokens: 0, output_tokens }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_effort_goes_to_the_models_that_take_it() {
        let oa = "https://api.openai.com/v1/chat/completions";
        let xai = "https://api.x.ai/v1/chat/completions";
        for m in ["gpt-5.5", "gpt-5.4-mini", "gpt-5", "o3", "o4-mini", "o1"] { assert_eq!(reasoning_effort_for(m, oa).as_deref(), Some("low"), "{m}"); }
        for m in ["gpt-4.1", "gpt-4o", "gpt-4o-mini"] { assert_eq!(reasoning_effort_for(m, oa), None, "{m}"); }
        for m in ["grok-4.5", "grok-4.6", "grok-4.7", "grok-5"] { assert_eq!(reasoning_effort_for(m, xai).as_deref(), Some("low"), "{m}"); }
        for m in ["grok-4.3", "grok-4.20-0309-reasoning", "grok-3-mini"] { assert_eq!(reasoning_effort_for(m, xai), None, "{m}"); }
        assert_eq!(reasoning_effort_for("gpt-5.5", "https://api.mistral.ai/v1/chat/completions"), None);
        assert_eq!(reasoning_effort_for("gpt-5.5", "http://localhost:11434/v1/chat/completions"), None);
    }

    #[test]
    fn a_stream_that_only_reasoned_is_an_error_and_a_short_answer_is_not() {
        assert!(starved_of_text("", Some("length")).is_err());
        assert!(starved_of_text("Yes.", Some("length")).is_ok());
        assert!(starved_of_text("", Some("stop")).is_ok());
        assert!(starved_of_text("", None).is_ok());
    }

    #[test]
    fn error_object_in_stream_is_reported() {
        assert_eq!(
            stream_error_message(r#"{"error":{"message":"The server is overloaded","type":"server_error"}}"#).as_deref(),
            Some("The server is overloaded"),
        );
        assert!(stream_error_message(r#"{"id":"x","choices":[]}"#).is_none());
        assert!(stream_error_message("not json").is_none());
    }

    #[tokio::test]
    async fn test_message_format() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        }];

        let request = OpenAIRequest {
            model: "gpt-4".to_string(),
            messages,
            max_completion_tokens: Some(100),
            reasoning_effort: None,
            temperature: None,
            stream: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("gpt-4"));
    }
}

// ============================================================================
// THE WEB THROUGH OPENAI'S OWN SEARCH TOOL (2026-10-27)
// ============================================================================

const OPENAI_RESPONSES_URL: &str = "https://api.openai.com/v1/responses";

/// One Responses API request with the `web_search` tool: the user's query as the
/// instruction, the model's answer with its cited pages back. Billed to the user's key.
pub async fn search_and_answer(api_key: &str, model: &str, query: &str, on_token: super::OnToken<'_>) -> Result<super::WebAnswer, LLMError> {
    let body = serde_json::json!({
        "model": model,
        "tools": [{"type": "web_search"}],
        "stream": true,
        "input": format!("Search the web for: {}\n\nAnswer directly and concisely from what you find. Do not narrate the searching or describe your process.", query),
    });
    println!("[Rust OpenAI] Web search through GPT ({}): {}", model, query);
    stream_responses(OPENAI_RESPONSES_URL, api_key, &body, on_token).await
}

/// A streamed Responses API request (OpenAI and xAI speak the same events): the answer's
/// words go to `on_token` as they arrive, and the finished response the stream ends with
/// is read as a non-streamed one would be.
pub async fn stream_responses(url: &str, api_key: &str, body: &serde_json::Value, on_token: super::OnToken<'_>) -> Result<super::WebAnswer, LLMError> {
    let response = reqwest::Client::new()
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(body)
        .send()
        .await
        .map_err(|e| LLMError::RequestFailed(e.to_string()))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.map_err(|e| LLMError::RequestFailed(e.to_string()))?;
        return Err(LLMError::APIError(format!("Status {}: {}", status, text)));
    }
    let mut finished: Option<serde_json::Value> = None;
    super::read_sse(response, |ev| fold_responses_event(&mut finished, ev, on_token)).await?;
    let finished = finished.ok_or_else(|| LLMError::InvalidResponse("the stream ended without a finished response".to_string()))?;
    let answer = parse_web_answer(&finished)?;
    println!("[Rust Responses] ✅ Web answer: {} chars, {} sources", answer.answer.len(), answer.sources.len());
    Ok(answer)
}

/// One Responses stream event: text deltas go out, the finished response is kept.
pub fn fold_responses_event(finished: &mut Option<serde_json::Value>, ev: &serde_json::Value, on_token: super::OnToken<'_>) -> Result<(), LLMError> {
    match ev["type"].as_str().unwrap_or("") {
        "response.output_text.delta" => {
            if let Some(t) = ev["delta"].as_str() { if !t.is_empty() { on_token(t); } }
        }
        "response.completed" | "response.incomplete" => { *finished = Some(ev["response"].clone()); }
        "response.failed" => {
            let msg = ev["response"]["error"]["message"].as_str().unwrap_or("the search failed");
            return Err(LLMError::APIError(msg.to_string()));
        }
        "error" => {
            let msg = ev["message"].as_str().or_else(|| ev["error"]["message"].as_str()).unwrap_or("the stream reported an error");
            return Err(LLMError::APIError(msg.to_string()));
        }
        _ => {}
    }
    Ok(())
}

/// The answer text and the cited pages out of a Responses API result: every
/// `output_text` joined, every `url_citation` once.
pub fn parse_web_answer(v: &serde_json::Value) -> Result<super::WebAnswer, LLMError> {
    let mut answer = String::new();
    let mut sources: Vec<super::WebSource> = Vec::new();
    for item in v["output"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        if item["type"].as_str() != Some("message") { continue; }
        for part in item["content"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
            if part["type"].as_str() != Some("output_text") { continue; }
            if let Some(t) = part["text"].as_str() { answer.push_str(t); }
            for a in part["annotations"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                if a["type"].as_str() != Some("url_citation") { continue; }
                let url = without_openai_tag(a["url"].as_str().unwrap_or(""));
                if url.is_empty() || sources.iter().any(|s| s.url == url) { continue; }
                sources.push(super::WebSource { title: a["title"].as_str().unwrap_or("").to_string(), url, snippet: String::new() });
            }
        }
    }
    if let Some(msg) = v["error"]["message"].as_str() {
        return Err(LLMError::APIError(msg.to_string()));
    }
    let answer = answer.trim().to_string();
    if answer.is_empty() {
        return Err(LLMError::InvalidResponse("the web search returned no answer".to_string()));
    }
    Ok(super::WebAnswer { answer, sources })
}

/// OpenAI appends `utm_source=openai` to every cited address; the page is the same without it.
fn without_openai_tag(url: &str) -> String {
    let u = url.replace("?utm_source=openai&", "?").replace("&utm_source=openai", "").replace("?utm_source=openai", "");
    u
}

#[cfg(test)]
mod web_answer_tests {
    use super::{parse_web_answer, without_openai_tag};

    #[test]
    fn the_tracking_tag_is_dropped_from_cited_addresses() {
        assert_eq!(without_openai_tag("https://a.org/p.html?utm_source=openai"), "https://a.org/p.html");
        assert_eq!(without_openai_tag("https://a.org/p?x=1&utm_source=openai"), "https://a.org/p?x=1");
        assert_eq!(without_openai_tag("https://a.org/p?utm_source=openai&x=1"), "https://a.org/p?x=1");
        assert_eq!(without_openai_tag("https://a.org/p"), "https://a.org/p");
    }

    #[test]
    fn answer_and_cited_pages_come_out_of_the_response() {
        let v = serde_json::json!({"output": [
            {"type": "web_search_call", "id": "ws1", "status": "completed", "action": {"type": "search", "query": "vsecu business checking"}},
            {"type": "message", "id": "m1", "role": "assistant", "content": [
                {"type": "output_text", "text": "VSECU offers business checking with no monthly fee.", "annotations": [
                    {"type": "url_citation", "url": "https://www.vsecu.com/business?utm_source=openai", "title": "Business | VSECU", "start_index": 0, "end_index": 10},
                    {"type": "url_citation", "url": "https://www.vsecu.com/business", "title": "dup", "start_index": 0, "end_index": 10}
                ]}
            ]}
        ]});
        let a = parse_web_answer(&v).unwrap();
        assert_eq!(a.answer, "VSECU offers business checking with no monthly fee.");
        assert_eq!(a.sources.len(), 1);
        assert_eq!(a.sources[0].title, "Business | VSECU");
    }

    #[test]
    fn no_text_is_an_error() {
        assert!(parse_web_answer(&serde_json::json!({"output": []})).is_err());
    }
}

#[cfg(test)]
mod web_stream_tests {
    use super::{fold_responses_event, parse_web_answer};
    use std::sync::Mutex;

    #[test]
    fn deltas_go_out_and_the_finished_response_is_kept() {
        let finished_resp = serde_json::json!({"output": [{"type": "message", "content": [
            {"type": "output_text", "text": "VSECU has no fee.", "annotations": [
                {"type": "url_citation", "url": "https://www.vsecu.com/b?utm_source=openai", "title": "VSECU"}]}]}]});
        let events = vec![
            serde_json::json!({"type": "response.created", "response": {}}),
            serde_json::json!({"type": "response.web_search_call.searching"}),
            serde_json::json!({"type": "response.output_text.delta", "delta": "VSECU "}),
            serde_json::json!({"type": "response.output_text.delta", "delta": "has no fee."}),
            serde_json::json!({"type": "response.completed", "response": finished_resp}),
        ];
        let words = Mutex::new(Vec::<String>::new());
        let on_token = |t: &str| words.lock().unwrap().push(t.to_string());
        let mut finished = None;
        for ev in &events { fold_responses_event(&mut finished, ev, &on_token).unwrap(); }
        assert_eq!(*words.lock().unwrap(), vec!["VSECU ", "has no fee."]);
        let a = parse_web_answer(&finished.unwrap()).unwrap();
        assert_eq!(a.answer, "VSECU has no fee.");
        assert_eq!(a.sources[0].url, "https://www.vsecu.com/b");
    }

    #[test]
    fn a_failed_response_is_an_error() {
        let mut finished = None;
        let ev = serde_json::json!({"type": "response.failed", "response": {"error": {"message": "rate limited"}}});
        assert!(fold_responses_event(&mut finished, &ev, &|_| {}).unwrap_err().to_string().contains("rate limited"));
    }
}

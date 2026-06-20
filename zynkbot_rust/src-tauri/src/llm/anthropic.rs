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
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamDelta {
    #[serde(rename = "type")]
    delta_type: String,
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
                        if delta.delta_type == "text_delta" {
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
                _ => {}
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


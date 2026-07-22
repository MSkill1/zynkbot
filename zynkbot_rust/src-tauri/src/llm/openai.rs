use super::{LLMError, LLMResponse, Message, Usage};
use futures::StreamExt;
use reqwest;
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// OpenAI API request structure
#[derive(Debug, Serialize)]
#[allow(dead_code)]
struct OpenAIRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

// --- Streaming SSE types (OpenAI-compatible, also used by xAI) ---

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

    let request_body = OpenAIRequest {
        model: model.to_string(),
        messages,
        max_tokens,
        temperature,
        stream: None,
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

/// Send a message via OpenAI API with SSE token streaming.
/// Also used by xAI (Grok) since it uses the OpenAI-compatible format.
/// `api_url` lets callers override the endpoint (e.g. xAI uses a different base URL).
#[allow(dead_code)]
pub async fn send_message_streaming<F>(
    api_key: &str,
    model: &str,
    messages: Vec<Message>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    api_url: &str,
    accept_invalid_certs: bool,
    on_token: F,
) -> Result<LLMResponse, LLMError>
where
    F: Fn(String),
{
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(accept_invalid_certs)
        .build()
        .unwrap_or_default();

    let request_body = OpenAIRequest {
        model: model.to_string(),
        messages,
        max_tokens,
        temperature,
        stream: Some(true),
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

    'outer: while let Some(chunk_result) = byte_stream.next().await {
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
                Err(_) => continue,
            };

            for choice in chunk.choices {
                if let Some(token) = choice.delta.content {
                    full_text.push_str(&token);
                    on_token(token);
                }
                if choice.finish_reason.as_deref() == Some("stop") {
                    output_tokens = full_text.split_whitespace().count() as u32; // approximate
                }
            }
        }
    }

    println!("[Rust OpenAI] ✅ Streaming complete");

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

    let body = serde_json::json!({
        "model": model,
        "stream": true,
        "max_tokens": 4096,
        "messages": [{ "role": "user", "content": content_blocks }]
    });

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

    'outer: while let Some(chunk_result) = byte_stream.next().await {
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
                Err(_) => continue,
            };

            for choice in chunk.choices {
                if let Some(token) = choice.delta.content {
                    full_text.push_str(&token);
                    on_token(token);
                }
                if choice.finish_reason.as_deref() == Some("stop") {
                    output_tokens = full_text.split_whitespace().count() as u32;
                }
            }
        }
    }

    println!("[Rust OpenAI] ✅ Vision streaming complete");

    Ok(LLMResponse {
        content: full_text,
        model: model.to_string(),
        usage: Some(Usage { input_tokens: 0, output_tokens }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_message_format() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        }];

        let request = OpenAIRequest {
            model: "gpt-4".to_string(),
            messages,
            max_tokens: Some(100),
            temperature: None,
            stream: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("gpt-4"));
    }
}

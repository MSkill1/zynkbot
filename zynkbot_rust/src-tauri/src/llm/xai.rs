use super::{LLMError, LLMResponse, Message, Usage};
use reqwest;
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
const XAI_API_URL: &str = "https://api.x.ai/v1/chat/completions";

/// xAI API request structure (OpenAI-compatible)
#[derive(Debug, Serialize)]
#[allow(dead_code)]
struct XAIRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

/// xAI API response structure (OpenAI-compatible)
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct XAIResponse {
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

/// Send a message to Grok via xAI API
/// xAI API is OpenAI-compatible, available with X Premium+ subscription
#[allow(dead_code)]
pub async fn send_message(
    api_key: &str,
    model: &str,
    messages: Vec<Message>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<LLMResponse, LLMError> {
    let client = reqwest::Client::new();

    let request_body = XAIRequest {
        model: model.to_string(),
        messages,
        max_tokens,
        temperature,
    };

    println!("[Rust xAI] Sending request to Grok: {}", model);

    let response = client
        .post(XAI_API_URL)
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

    let xai_response: XAIResponse = response
        .json()
        .await
        .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

    // Extract content from first choice
    let content = xai_response
        .choices
        .first()
        .map(|choice| choice.message.content.clone())
        .ok_or_else(|| LLMError::InvalidResponse("No choices in response".to_string()))?;

    println!("[Rust xAI] ✅ Received response from Grok");

    Ok(LLMResponse {
        content,
        model: xai_response.model,
        usage: Some(Usage {
            input_tokens: xai_response.usage.prompt_tokens,
            output_tokens: xai_response.usage.completion_tokens,
        }),
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

        let request = XAIRequest {
            model: "grok-3".to_string(),
            messages,
            max_tokens: Some(100),
            temperature: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("grok-3"));
    }
}

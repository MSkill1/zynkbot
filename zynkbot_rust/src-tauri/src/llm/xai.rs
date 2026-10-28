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

// ============================================================================
// THE WEB THROUGH xAI'S OWN SEARCH TOOL (2026-10-27)
// ============================================================================

const XAI_RESPONSES_URL: &str = "https://api.x.ai/v1/responses";

/// One Responses API request with xAI's `web_search` tool: Grok's answer with the pages
/// it cited. The answer carries inline "[[1]](url)" links; the annotation titles are the
/// citation numbers, so the page's host stands in as the title.
pub async fn search_and_answer(api_key: &str, model: &str, query: &str, on_token: super::OnToken<'_>) -> Result<super::WebAnswer, super::LLMError> {
    let body = serde_json::json!({
        "model": model,
        "tools": [{"type": "web_search"}],
        "stream": true,
        "input": [{"role": "user", "content": format!("Search the web for: {}\n\nAnswer directly and concisely from what you find. Do not narrate the searching or describe your process.", query)}],
    });
    println!("[Rust xAI] Web search through Grok ({}): {}", model, query);
    let mut answer = super::openai::stream_responses(XAI_RESPONSES_URL, api_key, &body, on_token).await?;
    for s in &mut answer.sources {
        if s.title.trim().is_empty() || s.title.trim().chars().all(|c| c.is_ascii_digit()) {
            s.title = host_of(&s.url);
        }
    }
    Ok(answer)
}

/// "https://www.hfcuvt.com/business/x.html" -> "hfcuvt.com"
pub fn host_of(url: &str) -> String {
    let rest = url.split("://").nth(1).unwrap_or(url);
    let host = rest.split('/').next().unwrap_or(rest);
    host.trim_start_matches("www.").to_string()
}

#[cfg(test)]
mod web_answer_tests {
    #[test]
    fn hosts_stand_in_for_numbered_titles() {
        assert_eq!(super::host_of("https://www.hfcuvt.com/business/x.html"), "hfcuvt.com");
        assert_eq!(super::host_of("http://x.ai/company"), "x.ai");
        assert_eq!(super::host_of("nonsense"), "nonsense");
    }
}

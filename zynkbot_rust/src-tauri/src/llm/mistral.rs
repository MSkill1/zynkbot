//! Mistral's own web search (2026-10-27): the Conversations API with the `web_search`
//! connector, no agent needed. Chat Completions cannot do it; this endpoint returns the
//! answer as chunks of text and `tool_reference` chunks naming the cited pages.
use super::{LLMError, WebAnswer, WebSource};

const MISTRAL_CONVERSATIONS_URL: &str = "https://api.mistral.ai/v1/conversations";

pub async fn search_and_answer(api_key: &str, model: &str, query: &str) -> Result<WebAnswer, LLMError> {
    match request(api_key, model, query).await {
        // The small "ministral" models refuse the connector; a model that takes it answers instead.
        Err(LLMError::APIError(msg)) if msg.contains("connector") && model != "mistral-small-latest" => {
            println!("[Rust Mistral] {} cannot search ({}); using mistral-small-latest for the search", model, msg.chars().take(80).collect::<String>());
            request(api_key, "mistral-small-latest", query).await
        }
        other => other,
    }
}

async fn request(api_key: &str, model: &str, query: &str) -> Result<WebAnswer, LLMError> {
    let body = serde_json::json!({
        "model": model,
        "tools": [{"type": "web_search"}],
        "inputs": format!("Search the web for: {}\n\nAnswer directly and concisely from what you find. Do not narrate the searching or describe your process.", query),
        "store": false,
    });
    println!("[Rust Mistral] Web search through Mistral ({}): {}", model, query);
    let response = reqwest::Client::new()
        .post(MISTRAL_CONVERSATIONS_URL)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| LLMError::RequestFailed(e.to_string()))?;
    let status = response.status();
    let text = response.text().await.map_err(|e| LLMError::RequestFailed(e.to_string()))?;
    if !status.is_success() {
        return Err(LLMError::APIError(format!("Status {}: {}", status, text)));
    }
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| LLMError::InvalidResponse(e.to_string()))?;
    let answer = parse_web_answer(&value)?;
    println!("[Rust Mistral] ✅ Web answer: {} chars, {} sources", answer.answer.len(), answer.sources.len());
    Ok(answer)
}

/// The answer and the cited pages out of a Conversations result: every `message.output`
/// entry's text chunks joined, every `tool_reference` chunk once.
pub fn parse_web_answer(v: &serde_json::Value) -> Result<WebAnswer, LLMError> {
    let mut answer = String::new();
    let mut sources: Vec<WebSource> = Vec::new();
    for entry in v["outputs"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        if entry["type"].as_str() != Some("message.output") { continue; }
        match &entry["content"] {
            serde_json::Value::String(t) => answer.push_str(t),
            serde_json::Value::Array(chunks) => {
                for c in chunks {
                    match c["type"].as_str().unwrap_or("") {
                        "text" => if let Some(t) = c["text"].as_str() { answer.push_str(t); },
                        "tool_reference" => {
                            let url = c["url"].as_str().unwrap_or("").to_string();
                            if url.is_empty() || sources.iter().any(|s| s.url == url) { continue; }
                            sources.push(WebSource { title: c["title"].as_str().unwrap_or("").to_string(), url, snippet: String::new() });
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    let answer = answer.trim().to_string();
    if answer.is_empty() {
        return Err(LLMError::InvalidResponse("the web search returned no answer".to_string()));
    }
    Ok(WebAnswer { answer, sources })
}

#[cfg(test)]
mod web_answer_tests {
    use super::parse_web_answer;

    #[test]
    fn text_chunks_and_tool_references_come_out_of_the_outputs() {
        let v = serde_json::json!({"outputs": [
            {"type": "tool.execution", "name": "web_search", "id": "t1"},
            {"type": "message.output", "role": "assistant", "content": [
                {"type": "text", "text": "Green Mountain Credit Union has a Rutland branch "},
                {"type": "tool_reference", "tool": "web_search", "title": "Green Mountain Credit Union", "url": "https://gmcu.org/", "source": "brave"},
                {"type": "text", "text": "on Cold River Road."},
                {"type": "tool_reference", "tool": "web_search", "title": "dup", "url": "https://gmcu.org/", "source": "brave"}
            ]}
        ]});
        let a = parse_web_answer(&v).unwrap();
        assert_eq!(a.answer, "Green Mountain Credit Union has a Rutland branch on Cold River Road.");
        assert_eq!(a.sources.len(), 1);
        assert_eq!(a.sources[0].title, "Green Mountain Credit Union");
    }

    #[test]
    fn a_plain_string_answer_is_read_and_nothing_is_an_error() {
        let v = serde_json::json!({"outputs": [{"type": "message.output", "content": "Plain answer."}]});
        assert_eq!(parse_web_answer(&v).unwrap().answer, "Plain answer.");
        assert!(parse_web_answer(&serde_json::json!({"outputs": []})).is_err());
    }
}

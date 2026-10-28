// LLM integration module
// This module handles communication with various LLM providers

pub mod anthropic;
pub mod openai;
pub mod xai;  // xAI (Grok) API - OpenAI-compatible
pub mod mistral;  // Mistral: its own web search through the Conversations API (2026-10-27)
pub mod local_embeddings;

// On-device GGUF inference on every platform, Android included (2026-10-22, branch
// local-model). The Android stub that returned "Local models not supported on Android"
// from every call lived here from 2026-09 until then.
pub mod local_models;

// pub mod whisper;    // TEMPORARILY DISABLED: Conflicts with llama-cpp-2 (GGML symbols)

use serde::{Deserialize, Serialize};

/// Common message format for all LLM providers
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Image attachment for vision-capable models
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageAttachment {
    pub base64: String,
    pub mime_type: String,
}

/// Common response format from LLMs
#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct LLMResponse {
    pub content: String,
    pub model: String,
    pub usage: Option<Usage>,
}

/// Token usage information
#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Error type for LLM operations
#[derive(Debug)]
#[allow(dead_code)]
pub enum LLMError {
    RequestFailed(String),
    InvalidResponse(String),
    APIError(String),
}

impl std::fmt::Display for LLMError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LLMError::RequestFailed(msg) => write!(f, "Request failed: {}", msg),
            LLMError::InvalidResponse(msg) => write!(f, "Invalid response: {}", msg),
            LLMError::APIError(msg) => write!(f, "API error: {}", msg),
        }
    }
}

impl std::error::Error for LLMError {}

/// A page a provider's own web search drew on (2026-10-27).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct WebSource {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// What a provider's own web search tool returned: the model's answer and its sources.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WebAnswer {
    pub answer: String,
    pub sources: Vec<WebSource>,
}

/// Where a provider's web answer is sent word by word while it is written (2026-10-28).
pub type OnToken<'a> = &'a (dyn Fn(&str) + Send + Sync);

/// Reads a server-sent-events response and hands every `data:` line that holds JSON to
/// `on_event`, as it arrives. The three web search endpoints stream the same way.
pub async fn read_sse(
    response: reqwest::Response,
    mut on_event: impl FnMut(&serde_json::Value) -> Result<(), LLMError>,
) -> Result<(), LLMError> {
    use futures::StreamExt;
    let mut stream = response.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| LLMError::RequestFailed(e.to_string()))?;
        buf.extend_from_slice(&chunk);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let Some(data) = line.trim().strip_prefix("data:") else { continue };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" { continue; }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                on_event(&v)?;
            }
        }
    }
    Ok(())
}

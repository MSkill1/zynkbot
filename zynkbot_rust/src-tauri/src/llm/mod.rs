// LLM integration module
// This module handles communication with various LLM providers

pub mod anthropic;
pub mod openai;
pub mod xai;  // xAI (Grok) API - OpenAI-compatible
pub mod local_embeddings;

#[cfg(not(target_os = "android"))]
pub mod local_models;

// Android stub: local GGUF LLM inference is Phase 2. Avoids scattering #[cfg]
// throughout chat.rs and lib.rs while keeping the type system consistent.
#[cfg(target_os = "android")]
pub mod local_models {
    use super::{LLMError, LLMResponse, Message};

    pub struct LocalModelSession;

    impl LocalModelSession {
        pub fn load(_path: &str) -> Result<Self, LLMError> {
            Err(LLMError::APIError("Local models not supported on Android".to_string()))
        }
        pub fn generate(
            &self,
            _messages: Vec<Message>,
            _max_tokens: Option<u32>,
            _temperature: Option<f32>,
            _json_schema: Option<&str>,
        ) -> Result<LLMResponse, LLMError> {
            Err(LLMError::APIError("Local models not supported on Android".to_string()))
        }
    }

    pub fn generate_with_local_model(
        _model_path: &str,
        _messages: Vec<Message>,
        _max_tokens: Option<u32>,
        _temperature: Option<f32>,
    ) -> Result<LLMResponse, LLMError> {
        Err(LLMError::APIError("Local models not supported on Android".to_string()))
    }

    pub fn generate_with_local_model_constrained(
        _model_path: &str,
        _messages: Vec<Message>,
        _json_schema: &str,
    ) -> Result<LLMResponse, LLMError> {
        Err(LLMError::APIError("Local models not supported on Android".to_string()))
    }
}

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

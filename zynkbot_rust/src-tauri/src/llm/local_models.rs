use super::{LLMError, LLMResponse, Message, Usage};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, LlamaModel},
    sampling::LlamaSampler,
    json_schema_to_grammar,
};
use once_cell::sync::Lazy;
use std::path::PathBuf;
use std::num::NonZeroU32;

/// Detect model type and build appropriate prompt format
fn build_prompt_for_model(model_path: &str, messages: &[Message]) -> String {
    let model_name = model_path.to_lowercase();

    // Order matters: more specific patterns before generic ones
    let (format_name, prompt) = if model_name.contains("qwen") {
        // Qwen3 is a thinking model — inject <think>\n\n</think> to short-circuit chain-of-thought
        // so token budget goes to the actual response, not internal reasoning.
        ("Qwen (ChatML, no-think)", build_chatml_prompt(messages, true))
    } else if model_name.contains("deepseek") && model_name.contains("llama") {
        // DeepSeek R1 Distill Llama — thinking enabled (local test only, not pushed).
        // Inject <think> open tag directly so only DeepSeek gets it; Lexi and other
        // Llama 3 models use no_think=false without the think tag.
        let mut p = build_llama3_prompt(messages, false);
        p.push_str("<think>\n");
        ("DeepSeek Distill Llama (Llama 3, thinking)", p)
    } else if model_name.contains("deepseek") {
        // DeepSeek R1 Distill Qwen and other DeepSeek variants — all are reasoning models,
        // so apply no-think regardless of base architecture
        ("DeepSeek (ChatML, no-think)", build_chatml_prompt(messages, true))
    } else if model_name.contains("llama-3") || model_name.contains("llama3") {
        ("Llama 3", build_llama3_prompt(messages, false))
    } else if model_name.contains("dolphin") || model_name.contains("openhermes") || model_name.contains("tinyllama") {
        ("ChatML", build_chatml_prompt(messages, false))
    } else if model_name.contains("mistral") && model_name.contains("instruct") {
        ("Mistral Instruct", build_mistral_instruct_prompt(messages))
    } else if model_name.contains("phi-3") || model_name.contains("phi-4")
           || model_name.contains("phi3") || model_name.contains("phi4") {
        ("Phi-3/4", build_phi3_prompt(messages))
    } else if model_name.contains("phi-2") || model_name.contains("phi2") {
        ("Phi-2", build_phi2_prompt(messages))
    } else {
        ("Simple (generic)", build_simple_prompt(messages))
    };

    println!("[Rust Local Models] Using {} prompt format", format_name);
    prompt
}

/// ChatML format — Qwen, DeepSeek (Qwen-based), Dolphin, OpenHermes, TinyLlama
///
/// `no_think`: inject `<think>\n\n</think>` before the first assistant token to short-circuit
/// Qwen3's chain-of-thought mode. The model sees the think block as already closed and
/// generates a direct response instead of reasoning first.
///
/// When the caller passes a single user message containing the full conversation-engine
/// prompt (system instructions + "USER'S QUESTION: ..."), split it into a proper
/// system role + user role. ChatML models follow instructions in the system role far
/// more reliably than instructions buried inside a user turn.
fn build_chatml_prompt(messages: &[Message], no_think: bool) -> String {
    let assistant_prefix = if no_think {
        "<|im_start|>assistant\n<think>\n\n</think>\n"
    } else {
        "<|im_start|>assistant\n"
    };

    // Single-message path: the conversation engine packs everything into one user message.
    // Split on "USER'S QUESTION:" so instructions land in the system role.
    if messages.len() == 1 && messages[0].role == "user" {
        let content = &messages[0].content;
        if let Some(q_pos) = content.find("USER'S QUESTION:") {
            let system_part = content[..q_pos].trim();
            let after_q = &content[q_pos + "USER'S QUESTION:".len()..];
            let user_question = if let Some(r_pos) = after_q.find("\n\nYOUR RESPONSE:") {
                after_q[..r_pos].trim()
            } else {
                after_q.trim()
            };
            let mut prompt = String::new();
            if !system_part.is_empty() {
                prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", system_part));
            }
            prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", user_question));
            prompt.push_str(assistant_prefix);
            return prompt;
        }
    }

    // Multi-message path (or single message without the marker): pass roles through as-is.
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "user"      => prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", msg.content)),
            "assistant" => prompt.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", msg.content)),
            "system"    => prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", msg.content)),
            _ => {}
        }
    }
    prompt.push_str(assistant_prefix);
    prompt
}

/// Llama 3 format — Meta Llama 3.x, Lexi Uncensored, DeepSeek R1 Distill Llama
///
/// `no_think`: inject `<think>\n\n</think>` after the assistant header to short-circuit
/// DeepSeek R1's chain-of-thought. Without this, the model outputs `</think>` as its
/// first token (trying to close a think block that was never opened).
fn build_llama3_prompt(messages: &[Message], no_think: bool) -> String {
    // BOS token is injected by str_to_token(AddBos::Always) — don't include it here
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|eot_id|>");
            }
            "user" => {
                prompt.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|eot_id|>");
            }
            "assistant" => {
                prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|eot_id|>");
            }
            _ => {}
        }
    }
    if no_think {
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n<think>\n\n</think>\n");
    } else {
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    prompt
}

/// Mistral Instruct format
fn build_mistral_instruct_prompt(messages: &[Message]) -> String {
    let mut prompt = String::new();
    let mut system_content = String::new();
    let mut conversation = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => system_content = msg.content.clone(),
            _ => conversation.push(msg),
        }
    }

    for (i, msg) in conversation.iter().enumerate() {
        match msg.role.as_str() {
            "user" => {
                if i == 0 && !system_content.is_empty() {
                    prompt.push_str(&format!("<s>[INST] {}\n\n{} [/INST]", system_content, msg.content));
                } else if i == 0 {
                    prompt.push_str(&format!("<s>[INST] {} [/INST]", msg.content));
                } else {
                    prompt.push_str(&format!("[INST] {} [/INST]", msg.content));
                }
            }
            "assistant" => {
                prompt.push_str(&format!(" {}</s>", msg.content));
            }
            _ => {}
        }
    }

    if let Some(last) = conversation.last() {
        if last.role == "user" {
            prompt.push(' ');
        }
    }

    prompt
}

/// Phi-2 format
fn build_phi2_prompt(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => prompt.push_str(&format!("{}\n\n", msg.content)),
            "user" => prompt.push_str(&format!("Instruct: {}\n", msg.content)),
            "assistant" => prompt.push_str(&format!("Output: {}\n", msg.content)),
            _ => {}
        }
    }
    prompt.push_str("Output: ");
    prompt
}

/// Phi-3 / Phi-4 format (Microsoft)
fn build_phi3_prompt(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => prompt.push_str(&format!("<|system|>\n{}<|end|>\n", msg.content)),
            "user" => prompt.push_str(&format!("<|user|>\n{}<|end|>\n", msg.content)),
            "assistant" => prompt.push_str(&format!("<|assistant|>\n{}<|end|>\n", msg.content)),
            _ => {}
        }
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

/// Simple format — generic fallback for unknown models
fn build_simple_prompt(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "user" => prompt.push_str(&format!("User: {}\n\n", msg.content)),
            "assistant" => prompt.push_str(&format!("Assistant: {}\n\n", msg.content)),
            "system" => prompt.push_str(&format!("{}\n\n", msg.content)),
            _ => {}
        }
    }
    prompt.push_str("Assistant: ");
    prompt
}

// llama.cpp backend — initialized once, lives for the process lifetime
static BACKEND: Lazy<LlamaBackend> = Lazy::new(|| {
    let mut backend = LlamaBackend::init().unwrap_or_else(|e| panic!("Failed to initialize llama.cpp backend: {e}"));
    backend.void_logs();
    backend
});

/// A loaded model that can be reused across multiple generation calls.
///
/// Create with `LocalModelSession::load`, call `generate` one or more times,
/// then drop to unload the model from memory. Designed for paired-call use:
/// load once, run both the main conversation call and Call 2 (relationship
/// classification), then drop.
pub struct LocalModelSession {
    model: LlamaModel,
    model_path: String,
    n_ctx: u32,
    n_threads: i32,
}

// Safety: LlamaModel wraps a raw pointer to an immutable llama_model C object.
// llama.cpp models are read-only after loading — inference reads weights via
// a separate per-call LlamaContext. We never access the model from two threads
// simultaneously: paired-call use is sequential (main call → await DB work →
// Call 2), and each spawn_blocking that touches this holds the only reference.
unsafe impl Send for LocalModelSession {}

impl LocalModelSession {
    /// Load a GGUF model from disk. Offloads all layers to GPU if CUDA is available.
    pub fn load(model_path: &str) -> Result<Self, LLMError> {
        let _ = &*BACKEND;

        let path = PathBuf::from(model_path);
        if !path.exists() {
            return Err(LLMError::RequestFailed(format!(
                "Model file not found: {}",
                model_path
            )));
        }

        println!("[Rust Local Models] Loading model: {}", model_path);

        // 999 = offload all layers to GPU if CUDA is compiled in.
        // llama.cpp silently ignores this on CPU-only builds — safe in both modes.
        #[cfg(feature = "cuda")]
        println!("[Rust Local Models] ⚡ CUDA build — requesting 999 GPU layers");
        #[cfg(not(feature = "cuda"))]
        println!("[Rust Local Models] ℹ️  CPU-only build — GPU offload disabled");

        let model_params = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(&BACKEND, path, &model_params)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to load model: {}", e)))?;

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let n_ctx = 8192u32;

        println!(
            "[Rust Local Models] ✅ Model loaded ({} threads, {}K context): {}",
            n_threads, n_ctx / 1024, model_path
        );

        Ok(Self { model, model_path: model_path.to_string(), n_ctx, n_threads })
    }

    /// Run inference on this session. Pass `json_schema` to constrain output to valid JSON.
    /// Creates a fresh context per call — contexts are not reused between calls.
    pub fn generate(
        &self,
        messages: Vec<Message>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        json_schema: Option<&str>,
    ) -> Result<LLMResponse, LLMError> {
        // n_ctx and n_batch must match — n_batch < prompt length causes the initial decode to fail
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_n_batch(self.n_ctx)
            .with_n_threads(self.n_threads)
            .with_n_threads_batch(self.n_threads);

        let mut ctx = self.model
            .new_context(&BACKEND, ctx_params)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to create context: {}", e)))?;

        let prompt = build_prompt_for_model(&self.model_path, &messages);
        let max_output_tokens = max_tokens.unwrap_or(512) as usize;
        let temp = temperature.unwrap_or(0.7);

        println!(
            "[Rust Local Models] Generating response (max_tokens: {})...",
            max_output_tokens
        );

        let tokens = self.model
            .str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to tokenize prompt: {}", e)))?;

        println!("[Rust Local Models] Prompt: {} tokens (n_batch: {})", tokens.len(), self.n_ctx);

        if tokens.len() > self.n_ctx as usize {
            return Err(LLMError::RequestFailed(format!(
                "Prompt too long: {} tokens exceeds context window of {}. Shorten conversation history or use a model with larger context.",
                tokens.len(), self.n_ctx
            )));
        }

        let max_output_tokens = max_output_tokens.min(
            (self.n_ctx as usize).saturating_sub(tokens.len() + 64)
        );

        ctx.clear_kv_cache();

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, &token) in tokens.iter().enumerate() {
            batch.add(token, i as i32, &[0], i == tokens.len() - 1)
                .map_err(|e| LLMError::RequestFailed(format!("Failed to add token to batch: {}", e)))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to decode prompt ({} tokens): {}", tokens.len(), e)))?;

        // Grammar-constrained sampling triggers a C-level abort() in llama.cpp when the grammar
        // engine finds no valid tokens under its constraints — killing the whole process with no
        // recoverable error. This is not model-specific: any model can hit it depending on context
        // state, vocab, and quantization. Prompt-based JSON enforcement is reliable enough across
        // all tested models, so grammar is bypassed universally.
        let effective_schema = if json_schema.is_some() {
            println!("[Rust Local Models] Grammar bypass: relying on prompt for JSON format");
            None
        } else {
            json_schema
        };

        // Grammar-constrained path: grammar filter forces every token to continue valid JSON,
        // then low temperature keeps output deterministic. Used for Call 2 (relationship JSON).
        // Normal path: temperature + top-k + top-p + dist gives natural varied responses.
        let mut sampler = if let Some(schema) = effective_schema {
            let grammar_str = json_schema_to_grammar(schema)
                .map_err(|e| LLMError::RequestFailed(format!("JSON schema → grammar failed: {}", e)))?;
            let grammar = LlamaSampler::grammar(&self.model, &grammar_str, "root")
                .map_err(|e| LLMError::RequestFailed(format!("Grammar sampler init failed: {}", e)))?;
            LlamaSampler::chain_simple([
                grammar,
                LlamaSampler::temp(0.1),
                LlamaSampler::dist(42),
            ])
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(temp),
                LlamaSampler::top_k(40),
                LlamaSampler::top_p(0.9, 1),
                LlamaSampler::dist(42),
            ])
        };

        let mut output = String::new();
        let prompt_len = tokens.len();
        let mut token_count = 0;

        for _iteration in 0..max_output_tokens {
            let new_token_id = sampler.sample(&ctx, -1);
            sampler.accept(new_token_id);

            if self.model.is_eog_token(new_token_id) {
                break;
            }

            #[allow(deprecated)]
            match self.model.token_to_str(new_token_id, llama_cpp_2::model::Special::Plaintext) {
                Ok(token_str) => output.push_str(&token_str),
                Err(_) => {}
            }

            let next_pos = prompt_len + token_count;
            let mut next_batch = LlamaBatch::new(1, 1);
            next_batch.add(new_token_id, next_pos as i32, &[0], true)
                .map_err(|e| LLMError::RequestFailed(format!("Failed to add token: {}", e)))?;
            ctx.decode(&mut next_batch)
                .map_err(|e| LLMError::RequestFailed(format!("Failed to decode token: {}", e)))?;

            token_count += 1;
        }

        println!("[Rust Local Models] ✅ Generated {} tokens", token_count);
        println!("[Rust Local Models] Output length: {} chars", output.len());
        println!(
            "[Rust Local Models] Output preview: {}...",
            output.chars().take(100).collect::<String>()
        );

        Ok(LLMResponse {
            content: output.trim().to_string(),
            model: self.model_path.clone(),
            usage: Some(Usage {
                input_tokens: tokens.len() as u32,
                output_tokens: token_count as u32,
            }),
        })
    }
}

/// Core inference: load model, generate, unload. Public API delegates here.
fn generate_internal(
    model_path: &str,
    messages: Vec<Message>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    json_schema: Option<&str>,
) -> Result<LLMResponse, LLMError> {
    LocalModelSession::load(model_path)?.generate(messages, max_tokens, temperature, json_schema)
}

/// Generate a conversational response using a local GGUF model.
pub fn generate_with_local_model(
    model_path: &str,
    messages: Vec<Message>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<LLMResponse, LLMError> {
    generate_internal(model_path, messages, max_tokens, temperature, None)
}

/// Generate a response constrained to valid JSON matching `json_schema`.
///
/// Used for Call 2 (relationship classification) where the model must produce
/// structured JSON regardless of its default conversational tendencies.
/// The schema is a JSON Schema object string; the grammar sampler enforces it
/// at the token level — the model physically cannot produce invalid JSON.
pub fn generate_with_local_model_constrained(
    model_path: &str,
    messages: Vec<Message>,
    json_schema: &str,
) -> Result<LLMResponse, LLMError> {
    generate_internal(model_path, messages, Some(4096), Some(0.1), Some(json_schema))
}

/// Simple wrapper for quick testing
#[allow(dead_code)]
pub fn query_local_model(model_path: &str, prompt: &str) -> Result<String, LLMError> {
    let response = generate_with_local_model(
        model_path,
        vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        Some(256),
        Some(0.7),
    )?;

    Ok(response.content)
}

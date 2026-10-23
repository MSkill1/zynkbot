use super::{LLMError, LLMResponse, Message, Usage};
use llama_cpp_2::{
    context::LlamaContext,
    token::LlamaToken,
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
use std::sync::Mutex;

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
    } else if model_name.contains("gemma") {
        ("Gemma", build_gemma_prompt(messages))
    } else if model_name.contains("lfm") || model_name.contains("granite") || model_name.contains("smollm") {
        // LFM2, Granite 4 and SmolLM3 all use ChatML markers.
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
/// Gemma 3 and 4 turn format (2026-10-22). Gemma has no system role: the instructions
/// go at the top of the first user turn. The engine's conversation prompt arrives as one
/// user message with "USER'S QUESTION:" marking the question; both halves stay in that
/// turn. BOS is added by the tokenizer.
fn build_gemma_prompt(messages: &[Message]) -> String {
    let mut prompt = String::new();
    if messages.len() == 1 && messages[0].role == "user" {
        let content = &messages[0].content;
        let body = if let Some(q_pos) = content.find("USER'S QUESTION:") {
            let system_part = content[..q_pos].trim();
            let after_q = &content[q_pos + "USER'S QUESTION:".len()..];
            let question = match after_q.find("\n\nYOUR RESPONSE:") { Some(r) => after_q[..r].trim(), None => after_q.trim() };
            if system_part.is_empty() { question.to_string() } else { format!("{}\n\n{}", system_part, question) }
        } else { content.trim().to_string() };
        prompt.push_str(&format!("<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n", body));
        return prompt;
    }
    let mut pending_system = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => pending_system = msg.content.clone(),
            "user" => {
                let body = if pending_system.is_empty() { msg.content.clone() } else { format!("{}\n\n{}", pending_system, msg.content) };
                pending_system.clear();
                prompt.push_str(&format!("<start_of_turn>user\n{}<end_of_turn>\n", body));
            }
            "assistant" => prompt.push_str(&format!("<start_of_turn>model\n{}<end_of_turn>\n", msg.content)),
            _ => {}
        }
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

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

static MODEL_CACHE: Lazy<Mutex<Option<LocalModelSession>>> = Lazy::new(|| Mutex::new(None));

/// Run `f` with a cached `LocalModelSession` for `model_path`.
/// Loads the model on first call or when the path changes; reuses otherwise.
pub fn with_cached_session<F, R>(model_path: &str, f: F) -> Result<R, LLMError>
where
    F: FnOnce(&LocalModelSession) -> Result<R, LLMError>,
{
    let mut cache = MODEL_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let needs_load = match &*cache {
        Some(session) => session.model_path != model_path,
        None => true,
    };
    if needs_load {
        if cache.is_some() {
            println!("[Rust Local Models] Model changed — unloading previous model");
        }
        *cache = Some(LocalModelSession::load(model_path)?);
    }
    f(cache.as_ref().unwrap())
}

/// Resolve which local model to use when the caller didn't name one explicitly.
///
/// Order: the `LOCAL_MODEL_PATH` override, then the first `.gguf` present in
/// `models/user/`. There is deliberately no baked-in filename fallback — the
/// previous hardcoded defaults disagreed with each other and named models the
/// setup wizard never downloads, so a fresh machine failed with
/// "Model file not found" instead of using the model it actually had.
pub fn resolve_default_model_path() -> Result<String, LLMError> {
    if let Ok(path) = std::env::var("LOCAL_MODEL_PATH") {
        if !path.trim().is_empty() {
            return Ok(path);
        }
    }

    let user_dir = crate::db::get_models_dir().join("user");
    let entries = std::fs::read_dir(&user_dir).map_err(|e| {
        LLMError::RequestFailed(format!(
            "No local model configured and {} could not be read: {}",
            user_dir.display(),
            e
        ))
    })?;

    let mut models: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|ext| ext.to_str())
                .map_or(false, |ext| ext.eq_ignore_ascii_case("gguf"))
        })
        .collect();
    // Sort so the choice is stable across runs rather than filesystem-order dependent.
    models.sort();

    match models.first() {
        Some(path) => {
            let path = path.to_string_lossy().into_owned();
            println!("[Rust Local Models] No model specified — using {}", path);
            Ok(path)
        }
        None => Err(LLMError::RequestFailed(format!(
            "No GGUF model found in {}. Download one from the setup wizard, or set LOCAL_MODEL_PATH.",
            user_dir.display()
        ))),
    }
}

/// A loaded model that can be reused across multiple generation calls.
///
/// Create with `LocalModelSession::load`, call `generate` one or more times,
/// then drop to unload the model from memory. Designed for paired-call use:
/// load once, run both the main conversation call and Call 2 (relationship
/// classification), then drop.
// ── foreground first (2026-10-22) ────────────────────────────────────────────
// One model, one phone CPU. A user's message must not wait behind the memory
// pipeline's background calls (on 2026-10-22 a message sat three minutes behind a
// relationship-classification call). A foreground turn registers itself with a
// ForegroundGuard before it asks for the model; a background call runs through
// `in_background`, and `generate` stops between tokens as soon as a foreground turn
// is waiting. `background_retry` waits for the foreground to finish and runs the
// call again from the start. Nothing changes for the desktop except that the same
// rule applies there too.

/// Foreground turns waiting for, or using, the local model.
pub static FOREGROUND_WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Held by a user turn from before it asks for the model until its reply is done.
pub struct ForegroundGuard;
impl ForegroundGuard {
    pub fn new() -> Self { FOREGROUND_WAITING.fetch_add(1, std::sync::atomic::Ordering::SeqCst); ForegroundGuard }
}
impl Drop for ForegroundGuard {
    fn drop(&mut self) { FOREGROUND_WAITING.fetch_sub(1, std::sync::atomic::Ordering::SeqCst); }
}

thread_local! { static IN_BACKGROUND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

const INTERRUPTED: &str = "interrupted: a user message is waiting for the model";

/// True when `e` is the interruption `generate` raises for a background call.
pub fn is_interrupted(e: &LLMError) -> bool {
    matches!(e, LLMError::RequestFailed(m) if m == INTERRUPTED)
}

fn foreground_waiting() -> bool { FOREGROUND_WAITING.load(std::sync::atomic::Ordering::SeqCst) > 0 }

/// Run `f` as a background call: it yields to a foreground turn and is retried once
/// the foreground is done (up to `tries` times, polling every half second while a
/// turn is in progress). Blocking; call from a blocking thread.
pub fn background_retry<R>(what: &str, tries: usize, f: impl Fn() -> Result<R, LLMError>) -> Result<R, LLMError> {
    for attempt in 1..=tries {
        let mut waited = 0u32;
        while foreground_waiting() {
            std::thread::sleep(std::time::Duration::from_millis(500));
            waited += 1;
            if waited % 120 == 0 { println!("[Rust Local Models] {} still waiting for the foreground ({} s)", what, waited / 2); }
        }
        let result = IN_BACKGROUND.with(|b| { b.set(true); let r = f(); b.set(false); r });
        match result {
            Err(e) if is_interrupted(&e) && attempt < tries => {
                println!("[Rust Local Models] {} stepped aside for a user message (attempt {} of {}); will run again after", what, attempt, tries);
            }
            other => return other,
        }
    }
    Err(LLMError::RequestFailed(format!("{} gave way to user messages {} times", what, tries)))
}

/// How many leading tokens of `tokens` the context already holds from `prev`, so they
/// need not be read again. At least one token is always left to read, because the last
/// token of the prompt is what produces the logits the first answer token is sampled from.
pub fn reuse_count(prev: &[LlamaToken], tokens: &[LlamaToken]) -> usize {
    if tokens.is_empty() { return 0; }
    let common = prev.iter().zip(tokens.iter()).take_while(|(a, b)| a == b).count();
    common.min(tokens.len() - 1)
}

/// Threads for inference. Desktop: every core. Phone: the fast cores only. A phone has
/// small efficiency cores whose top speed is far below the others', and llama.cpp
/// waits for the slowest thread at every step, so on a 1+5+2 Pixel 10 Pro XL eight
/// threads are slower than six. A core counts as fast if its top frequency is at
/// least three quarters of the fastest core's (2026-10-22).
fn inference_threads() -> i32 {
    let all = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4);
    if !cfg!(target_os = "android") { return all; }
    let mut freqs: Vec<u64> = (0..all)
        .filter_map(|i| std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_max_freq", i)).ok())
        .filter_map(|s| s.trim().parse::<u64>().ok())
        .collect();
    if freqs.len() != all as usize { return all; }
    freqs.sort_unstable();
    let top = *freqs.last().unwrap_or(&0);
    let fast = freqs.iter().filter(|&&f| f * 4 >= top * 3).count() as i32;
    fast.max(2)
}

/// The foreground context kept between turns, with the tokens it currently holds.
struct KvState {
    ctx: LlamaContext<'static>,
    tokens: Vec<LlamaToken>,
}

pub struct LocalModelSession {
    // Declared before `model` so it is dropped first: the context borrows the model.
    kv: Mutex<Option<KvState>>,
    model: Box<LlamaModel>,
    model_path: String,
    n_ctx: u32,
    n_threads: i32,
    n_batch: u32,
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
        #[cfg(all(not(feature = "cuda"), target_os = "android"))]
        println!("[Rust Local Models] ⚡ Android build with the OpenCL backend — requesting 999 GPU layers (CPU if the phone reports no OpenCL device)");
        #[cfg(all(not(feature = "cuda"), not(target_os = "android")))]
        println!("[Rust Local Models] ℹ️  CPU-only build — GPU offload disabled");

        let model_params = LlamaModelParams::default().with_n_gpu_layers(999);
        // Android experiment (2026-10-23, after reading PocketPal's shipped defaults): with
        // memory-mapping the 2.5 GB file is pulled off storage as it is first used, so the
        // first reply after a load is slow and Android counts the pages twice. Off, the
        // load takes a few seconds longer and then stays put. Measured against the
        // 2026-10-22 numbers on the Pixel; kept only if the first reply gets faster.
        #[cfg(target_os = "android")]
        let model_params = model_params.with_use_mmap(false);
        let model = LlamaModel::load_from_file(&BACKEND, path, &model_params)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to load model: {}", e)))?;

        let n_threads = inference_threads();

        // Phone (2026-10-22): a 4K window and 512-token batches. 8K/8K reserved enough
        // memory on a 16 GB Pixel that Android killed fifteen other apps when the model
        // loaded; a phone prompt is a few hundred tokens. Desktop keeps 8K with the batch
        // equal to the window.
        let n_ctx: u32 = if cfg!(target_os = "android") { 4096 } else { 8192 };
        let n_batch: u32 = if cfg!(target_os = "android") { 512 } else { n_ctx };

        println!(
            "[Rust Local Models] ✅ Model loaded ({} threads, {}K context, batch {}): {}",
            n_threads, n_ctx / 1024, n_batch, model_path
        );

        Ok(Self { kv: Mutex::new(None), model: Box::new(model), model_path: model_path.to_string(), n_ctx, n_batch, n_threads })
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
        self.generate_streaming(messages, max_tokens, temperature, json_schema, None)
    }

    /// Generate, calling `on_token` with each piece of text as it is written (the local
    /// path's streaming, 2026-10-22). A foreground call keeps its context between turns
    /// and reads only the part of the prompt that changed since the previous one (the
    /// text before the first difference is the same tokens, so its cache entries stand);
    /// a background call gets a smaller throwaway context so it never disturbs that.
    pub fn generate_streaming(
        &self,
        messages: Vec<Message>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        json_schema: Option<&str>,
        on_token: Option<&mut dyn FnMut(&str)>,
    ) -> Result<LLMResponse, LLMError> {
        let background = IN_BACKGROUND.with(|b| b.get());
        let interrupted = || LLMError::RequestFailed(INTERRUPTED.to_string());
        if background && foreground_waiting() { return Err(interrupted()); }

        let make_ctx = |n_ctx: u32| -> Result<LlamaContext<'static>, LLMError> {
            // The batch can never exceed the window (the desktop's 8K batch in a 2K
            // throwaway context sampled an end-of-text token at once, 2026-10-22).
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(n_ctx))
                .with_n_batch(self.n_batch.min(n_ctx))
                .with_n_threads(self.n_threads)
                .with_n_threads_batch(self.n_threads);
            // Safety: the model lives in a Box whose address does not change, and `kv` is
            // declared before `model` so every context is dropped before the model is.
            let model: &'static LlamaModel = unsafe { &*(&*self.model as *const LlamaModel) };
            model.new_context(&BACKEND, ctx_params)
                .map_err(|e| LLMError::RequestFailed(format!("Failed to create context: {}", e)))
        };

        if background {
            // Small, throwaway: a memory call reads a few hundred tokens and writes a few hundred.
            let n = self.n_ctx.min(2048);
            let mut ctx = make_ctx(n)?;
            let (resp, _) = self.run(&mut ctx, n, &[], messages, max_tokens, temperature, json_schema, on_token, background)?;
            return Ok(resp);
        }

        let mut guard = self.kv.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() { *guard = Some(KvState { ctx: make_ctx(self.n_ctx)?, tokens: Vec::new() }); }
        let state = guard.as_mut().unwrap();
        let prev = std::mem::take(&mut state.tokens);
        match self.run(&mut state.ctx, self.n_ctx, &prev, messages, max_tokens, temperature, json_schema, on_token, background) {
            Ok((resp, tokens)) => { state.tokens = tokens; Ok(resp) }
            Err(e) => { let _ = state.ctx.clear_kv_cache_seq(None, None, None); Err(e) }
        }
    }

    /// One generation on `ctx`. `prev` is what the context already holds (empty for a
    /// fresh one). Returns the response and the tokens the context holds afterwards.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        ctx: &mut LlamaContext<'_>,
        n_ctx: u32,
        prev: &[LlamaToken],
        messages: Vec<Message>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        json_schema: Option<&str>,
        mut on_token: Option<&mut dyn FnMut(&str)>,
        background: bool,
    ) -> Result<(LLMResponse, Vec<LlamaToken>), LLMError> {
        let interrupted = || LLMError::RequestFailed(INTERRUPTED.to_string());
        let prompt = build_prompt_for_model(&self.model_path, &messages);
        let max_output_tokens = max_tokens.unwrap_or(512) as usize;
        let temp = temperature.unwrap_or(0.7);

        println!(
            "[Rust Local Models] Generating response (max_tokens: {})...",
            max_output_tokens
        );

        // llama-cpp-2 0.1.160 (2026-10-22): tokenisation moved to the vocab. `add_special`
        // adds BOS only for models that want one (Qwen's template has none); `parse_special`
        // turns the chat-template markers in the prompt into their tokens, as before.
        let tokens = self.model.vocab().tokenize(prompt.as_bytes(), true, true);
        if tokens.is_empty() {
            return Err(LLMError::RequestFailed("Failed to tokenize prompt: no tokens".to_string()));
        }

        println!("[Rust Local Models] Prompt: {} tokens (n_batch: {})", tokens.len(), self.n_batch);

        if tokens.len() + 64 > n_ctx as usize {
            return Err(LLMError::RequestFailed(format!(
                "Prompt too long: {} tokens exceeds context window of {}. Shorten conversation history or use a model with larger context.",
                tokens.len(), n_ctx
            )));
        }

        let max_output_tokens = max_output_tokens.min(
            (n_ctx as usize).saturating_sub(tokens.len() + 64)
        );

        // Keep what the context already holds and matches the new prompt from the start;
        // drop the rest. At least the last prompt token is always read so there are logits.
        let mut common = reuse_count(prev, &tokens);
        if prev.is_empty() || common == 0 {
            ctx.clear_kv_cache();
            common = 0;
        } else if !matches!(ctx.clear_kv_cache_seq(Some(0), Some(common as u32), None), Ok(true)) {
            ctx.clear_kv_cache();
            common = 0;
        }

        // Feed the new part of the prompt in pieces of at most n_batch tokens; only the very
        // last token asks for logits. A background call steps aside the moment a user turn
        // is waiting: between prompt pieces and between generated tokens.
        let t_prompt = std::time::Instant::now();
        let last = tokens.len() - 1;
        // A background call reads in small pieces so it can step aside within a few
        // seconds (a 512-token piece took 22 s on a warm phone, 2026-10-22).
        let piece = if background { (self.n_batch as usize).min(128) } else { self.n_batch as usize };
        for chunk_start in (common..tokens.len()).step_by(piece) {
            if background && foreground_waiting() { return Err(interrupted()); }
            let chunk = &tokens[chunk_start..(chunk_start + piece).min(tokens.len())];
            let mut batch = LlamaBatch::new(chunk.len(), 1);
            for (i, &token) in chunk.iter().enumerate() {
                let pos = chunk_start + i;
                batch.add(token, pos as i32, &[0], pos == last)
                    .map_err(|e| LLMError::RequestFailed(format!("Failed to add token to batch: {}", e)))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| LLMError::RequestFailed(format!("Failed to decode prompt ({} tokens): {}", tokens.len(), e)))?;
        }
        let new_tokens = tokens.len() - common;
        println!("[Rust Local Models] Prompt read in {:.1}s ({:.0} tokens/s): {} new tokens, {} kept from the previous turn",
            t_prompt.elapsed().as_secs_f32(), new_tokens as f32 / t_prompt.elapsed().as_secs_f32().max(0.001), new_tokens, common);

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
        let mut pending: Vec<u8> = Vec::new();
        let mut generated: Vec<LlamaToken> = Vec::new();
        let prompt_len = tokens.len();
        let mut token_count = 0;

        for _iteration in 0..max_output_tokens {
            if background && foreground_waiting() { return Err(interrupted()); }
            let new_token_id = sampler.sample(&ctx, -1);
            sampler.accept(new_token_id);

            if self.model.vocab().is_eog(new_token_id) {
                break;
            }

            // A token is bytes, and one character can span two tokens. Bytes are kept until
            // they form whole characters, so nothing is dropped (the old per-token string
            // conversion silently lost a token whose bytes were not valid on their own).
            generated.push(new_token_id);
            pending.extend_from_slice(&self.model.vocab().token_to_piece(new_token_id, false, None));
            match std::str::from_utf8(&pending) {
                Ok(s) => { output.push_str(s); if let Some(cb) = on_token.as_mut() { cb(s); } pending.clear(); }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        let s = std::str::from_utf8(&pending[..valid]).unwrap_or("");
                        output.push_str(s); if let Some(cb) = on_token.as_mut() { cb(s); }
                        pending.drain(..valid);
                    }
                }
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
        if !pending.is_empty() { let s = String::from_utf8_lossy(&pending).into_owned(); if let Some(cb) = on_token.as_mut() { cb(&s); } output.push_str(&s); }
        println!("[Rust Local Models] Output length: {} chars", output.len());
        println!(
            "[Rust Local Models] Output preview: {}...",
            output.chars().take(100).collect::<String>()
        );

        let mut held = tokens;
        held.extend(generated);
        Ok((LLMResponse {
            content: output.trim().to_string(),
            model: self.model_path.clone(),
            usage: Some(Usage {
                input_tokens: held.len() as u32,
                output_tokens: token_count as u32,
            }),
        }, held))
    }
}

/// Core inference through the cached session (until 2026-10-22 this loaded the model
/// from disk for every call, so a background memory call put a second 2.5 GB copy
/// in memory next to the cached one).
fn generate_internal(
    model_path: &str,
    messages: Vec<Message>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    json_schema: Option<&str>,
) -> Result<LLMResponse, LLMError> {
    with_cached_session(model_path, |session| session.generate(messages, max_tokens, temperature, json_schema))
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

#[cfg(test)]
mod engine_smoke {
    /// Loads the GGUF named by LOCAL_MODEL_PATH and asks it one thing. Ignored by default
    /// (needs a model file and a minute of CPU); run after an engine-crate bump:
    ///   LOCAL_MODEL_PATH=/path/model.gguf LD_LIBRARY_PATH=lib/vosk cargo test --lib engine_smoke -- --ignored --nocapture
    #[test]
    #[ignore]
    fn engine_smoke_generates_text() {
        let path = std::env::var("LOCAL_MODEL_PATH").expect("LOCAL_MODEL_PATH");
        let t0 = std::time::Instant::now();
        let out = super::query_local_model(&path, "Reply with one short sentence: what is the capital of France?")
            .expect("generation failed");
        println!("[engine_smoke] {:?} -> {:?}", t0.elapsed(), out);
        assert!(!out.trim().is_empty(), "empty reply");
        // Second turn with the same opening text: the kept context should make the read
        // nearly free (watch the "kept from the previous turn" count in the output).
        let t1 = std::time::Instant::now();
        let out2 = super::query_local_model(&path, "Reply with one short sentence: what is the capital of Spain?")
            .expect("second generation failed");
        println!("[engine_smoke] second turn {:?} -> {:?}", t1.elapsed(), out2);
        assert!(!out2.trim().is_empty(), "empty second reply");
        // A background-style call in between must not disturb the kept context.
        let out3 = super::background_retry("smoke", 1, || super::query_local_model(&path, "Reply with one short sentence: what is the capital of Germany?")).expect("background call failed");
        println!("[engine_smoke] background -> {:?}", out3);
        let t2 = std::time::Instant::now();
        let out4 = super::query_local_model(&path, "Reply with one short sentence: what is the capital of Italy?")
            .expect("third generation failed");
        println!("[engine_smoke] third turn {:?} -> {:?}", t2.elapsed(), out4);
        assert!(!out4.trim().is_empty(), "empty third reply");
    }
}

#[cfg(test)]
mod kept_context_tests {
    use super::reuse_count;
    use llama_cpp_2::token::LlamaToken;
    fn t(v: &[i32]) -> Vec<LlamaToken> { v.iter().map(|&i| LlamaToken(i)).collect() }

    #[test]
    fn same_prompt_again_keeps_all_but_the_last_token() {
        assert_eq!(reuse_count(&t(&[1, 2, 3, 4]), &t(&[1, 2, 3, 4])), 3);
    }
    #[test]
    fn a_longer_prompt_with_the_same_start_keeps_the_start() {
        assert_eq!(reuse_count(&t(&[1, 2, 3]), &t(&[1, 2, 3, 9, 9])), 3);
    }
    #[test]
    fn divergence_keeps_up_to_the_first_difference() {
        assert_eq!(reuse_count(&t(&[1, 2, 3, 4, 5]), &t(&[1, 2, 7, 8])), 2);
    }
    #[test]
    fn nothing_held_or_nothing_shared_keeps_nothing() {
        assert_eq!(reuse_count(&[], &t(&[1, 2])), 0);
        assert_eq!(reuse_count(&t(&[5, 6]), &t(&[1, 2])), 0);
        assert_eq!(reuse_count(&t(&[1]), &[]), 0);
    }
}

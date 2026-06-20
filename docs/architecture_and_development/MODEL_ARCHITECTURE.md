# Zynkbot Model Architecture

## Directory Structure

```
zynkbot_rust/src-tauri/models/
├── system/                          # System models (shipped with app or auto-downloaded)
│   ├── bert-base-NER/              # Entity extraction (dslim/bert-base-NER)
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   ├── all-MiniLM-L6-v2/           # Embeddings (sentence-transformers/all-MiniLM-L6-v2)
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── vocab.txt
│   ├── toxic-bert/                 # Safety filtering (TinyBERT toxicity classifier)
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer.json
│   └── ggml-base.en.bin            # Whisper speech-to-text
│
└── user/                            # User-downloadable chat models
    ├── DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
    ├── Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf
    ├── Qwen3-8B-Q4_K_M.gguf
    └── [user can add more .gguf files here]
```

## LLM Backends

Zynkbot routes conversations to one of four backends based on the model selected in the UI. Any backend that is not `"local"` and does not end with `.gguf` is treated as an API model (`ConversationEngine::is_api_model()`).

### API Models

All three API backends use streaming responses.

| Backend | Source file | Model(s) |
|---------|-------------|----------|
| Anthropic | `llm/anthropic.rs` | `claude-sonnet-4-6` (default), `claude-haiku-4-5-20251001`, `claude-opus-4-7` |
| OpenAI | `llm/openai.rs` | `gpt-4o-mini` |
| xAI (Grok) | `llm/xai.rs` | `grok-3` (default), `grok-2-vision-1212` (vision) |

xAI uses the OpenAI-compatible streaming implementation (`openai::send_message_streaming`) pointed at `https://api.x.ai/v1/chat/completions`.

### Local GGUF

**Source file:** `llm/local_models.rs`  
**Runtime:** `llama_cpp_2` crate (llama.cpp Rust bindings)  
**Context window:** 8192 tokens (set at model load time)  
**Threads:** Set to available CPU parallelism at runtime (`std::thread::available_parallelism()`)  
**GPU:** All layers offloaded via `with_n_gpu_layers(99)`; falls back to CPU automatically

Prompt format is auto-detected from the model filename:

| Filename pattern | Format | Example models |
|-----------------|--------|----------------|
| `qwen` | ChatML | Qwen2.5, Qwen3 |
| `deepseek` + `llama` | Llama 3 | DeepSeek R1 Distill Llama 8B |
| `deepseek` (other) | ChatML | DeepSeek R1 Distill Qwen |
| `llama-3` / `llama3` | Llama 3 | Llama 3.1/3.2/3.3, Lexi Uncensored |
| `dolphin`, `openhermes`, `tinyllama` | ChatML | Dolphin, OpenHermes, TinyLlama |
| `mistral` + `instruct` | Mistral Instruct | Mistral 7B Instruct |
| `phi-3` / `phi-4` / `phi3` / `phi4` | Phi-3/4 | Phi-3 Mini, Phi-4 |
| `phi-2` / `phi2` | Phi-2 | Phi-2 |
| (anything else) | Simple (generic fallback) | Unknown/community models |

Models not matching a known pattern receive a generic `User:` / `Assistant:` format. This works for many models but may produce degraded output compared to the model's native format. If you add a model family that isn't listed here, add a detection case and prompt builder in `local_models.rs`.

**Sampling:** Conversation responses use a `LlamaSampler` chain: temperature (0.7) → top-k (40) → top-p (0.9). Structured calls (relationship classification, memory decisions) use a grammar-constrained chain — see [Relationship Detection Architecture](RELATIONSHIP_DETECTION_ARCHITECTURE.md).

---

## Adaptive Context Limits

`ConversationEngine` applies different limits depending on backend type:

| Setting | API models | Local GGUF |
|---------|-----------|------------|
| Hybrid memory search limit | 15 | 7 |
| Memories included in prompt (max) | 20 | 7 |
| Conversation history included | 40 messages | 8 messages |

One-hop graph-linked memories (`elaborates`, `contradicts`, `resolves`) are added on top of the search results with no cap.

---

## Model Categories

### System Models (Required)
- **Location**: `src-tauri/models/system/`
- **Purpose**: Core functionality (embeddings, NER, safety, STT)
- **Managed by**: Application (auto-download on first run)
- **User control**: No (required for operation)

### User Chat Models (Optional)
- **Location**: `src-tauri/models/user/`
- **Purpose**: LLM backends for conversation
- **Managed by**: User downloads from Hugging Face
- **User control**: Yes (can add/remove models)

## Path Resolution

All model paths use `TAURI_APP_DIR` for consistency:

```rust
// Get the app's resource directory (consistent across dev/production)
let app_dir = std::env::var("TAURI_APP_DIR")
    .unwrap_or_else(|_| std::env::current_dir().unwrap().to_string_lossy().to_string());

let system_models = PathBuf::from(&app_dir).join("models").join("system");
let user_models = PathBuf::from(&app_dir).join("models").join("user");
```


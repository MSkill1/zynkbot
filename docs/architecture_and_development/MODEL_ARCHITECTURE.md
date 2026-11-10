# Zynkbot Model Architecture

*Checked against the code on 2026-11-10 (`v1`, a7599eb).*

## Where the models live

Models are not in the source tree. An installed app keeps them in its data folder
(`db::get_models_dir()`, `src-tauri/src/db.rs`):

| Platform | Folder |
|----------|--------|
| Linux | `~/.local/share/zynkbot/models/` |
| Windows | `%LOCALAPPDATA%\zynkbot\models\` |
| Android | `/data/data/ai.containai.zynkbot/files/zynkbot/models/` |

A development or test build (the executable sits under `target/`) walks up from the
executable and uses the first `models/` folder it finds, which is `src-tauri/models/`.
There is no `TAURI_APP_DIR` lookup.

```
models/
├── system/                          # Required; fetched by the setup wizard on first run
│   ├── all-MiniLM-L6-v2/           # Embeddings, 384 dimensions (llm/local_embeddings.rs)
│   ├── bert-base-NER/              # Named things in a message (nlp_enhancer.rs)
│   └── toxic-bert/                 # Safety filtering (safety_classifier.rs)
│
└── user/                            # Chat models, any .gguf file
    ├── Sophro-4B-memory-v1-Q4_0.gguf          # the Zynkbot model
    ├── gemma-4-12b-it-Q4_K_M.gguf             # optional alternatives (desktop list, 2026-11-09)
    ├── gemma-4-E4B-it-Q4_K_M.gguf
    ├── Qwen3.5-9B-Q4_K_M.gguf
    ├── Ministral-3-8B-Instruct-2512-Q4_K_M.gguf
    └── [any other .gguf the user drops in]
```

The three system models run through Candle, not llama.cpp. `check_needs_setup()` in
`commands/setup_wizard.rs` sends the user to the setup wizard when any of the three
`model.safetensors` files is missing, and `download_system_models()` fetches them from
Hugging Face.

Speech recognition is not in this folder. Hands-free listening uses Vosk, whose model is
bundled with the app (`vosk_desktop.rs`; the Android assets folder), and cloud dictation
uses OpenAI's `gpt-4o-mini-transcribe` (`DICTATION_MODEL` in `lib.rs`). The Whisper module
is disabled (`llm/mod.rs`: its GGML symbols clash with llama.cpp).

### Chat models

`get_models()` in `commands/models.rs` lists every `.gguf` file in `models/user/`, plus the
file named by `LOCAL_MODEL_PATH` when that is set. `download_user_model()` in
`commands/setup_wizard.rs` downloads the files in the tree above by id and verifies each
against a SHA-256 checksum: Sophro (`zb-qwen3-4b`) on phone and desktop, the four
alternatives on the desktop. Measured strengths and failings of each are in
[BENCHMARKS.md](../BENCHMARKS.md), section 5. When a call names no model,
`resolve_default_model_path()` takes `LOCAL_MODEL_PATH`, otherwise the first `.gguf` in
`models/user/` in name order; there is no built-in file name.

## LLM Backends

The model picked in the UI is a backend name. `get_models()` offers a cloud provider when
its API key is set, the custom endpoint when `CUSTOM_API_URL` is set, and each local file.

`ConversationEngine::is_api_model()` is true for the four cloud providers only. `"local"`,
a name ending in `.gguf`, and `"custom"` are not API models and get the smaller limits
below.

### Cloud providers

All four stream their replies.

| Backend | Source file | Default model | Setting that overrides it |
|---------|-------------|---------------|---------------------------|
| Anthropic | `llm/anthropic.rs` | `claude-sonnet-5` | `ANTHROPIC_MODEL` |
| OpenAI | `llm/openai.rs` | `gpt-5.5` | `OPENAI_MODEL` |
| xAI (Grok) | `llm/openai.rs` (chat), `llm/xai.rs` (web search) | `grok-4.5` | `XAI_MODEL` |
| Mistral | `llm/openai.rs` (chat), `llm/mistral.rs` (web search) | `mistral-large-latest` | `MISTRAL_MODEL` |

xAI and Mistral chat use the OpenAI-compatible streaming client
(`openai::send_message_streaming`) pointed at `https://api.x.ai/v1/chat/completions` and
`https://api.mistral.ai/v1/chat/completions`. The models each provider offers are listed
in the API Keys window (`src/components/APIKeyModal.jsx`); the choice is saved as the
setting in the last column. That list changes often and is not repeated here.

### Custom endpoint

`CUSTOM_API_URL` and `CUSTOM_MODEL` name an OpenAI-compatible server such as Ollama. It is
shown as "Custom / Ollama" and treated as a small model: the limits of the right-hand
column below.

### Local GGUF

**Source file:** `llm/local_models.rs`
**Runtime:** `llama_cpp_2` crate (llama.cpp Rust bindings)

| Setting | Desktop | Android |
|---------|---------|---------|
| Context window | 8192 tokens | 4096 tokens |
| Batch size | the context size | 512 |
| Threads | every core | the fast cores only (`inference_threads()`) |
| GPU layers requested | 999 (used only by a CUDA build) | 0 (CPU engine) |

The loaded model is kept between calls (`MODEL_CACHE`). The reply context is kept too,
with the tokens of the previous turn, so only the new part of a prompt is read
(`reuse_count()`). A background call (the memory pipeline) gets a throwaway context of at
most 2048 tokens.

A user's message always goes first. A reply registers a `ForegroundGuard`; a background
call (the memory pipeline) stops between tokens when a reply is waiting, and
`background_retry()` runs it again from the start afterwards.

Prompt format is chosen from the file name. When the name says nothing known, the file's
own `general.architecture` decides (`build_prompt_for_model()`):

| File name contains | Format | Example models |
|--------------------|--------|----------------|
| `qwen` | ChatML, thinking closed | Qwen2.5, Qwen3, Qwen3.5, Sophro |
| `deepseek` + `llama` | Llama 3, thinking open | DeepSeek R1 Distill Llama 8B |
| `deepseek` (other) | ChatML, thinking closed | DeepSeek R1 Distill Qwen |
| `llama-3` / `llama3` | Llama 3 | Llama 3.1/3.2/3.3 |
| `dolphin`, `openhermes`, `tinyllama` | ChatML | Dolphin, OpenHermes, TinyLlama |
| `gemma-4` / `gemma4`, or architecture `gemma4` | Gemma 4 (`<|turn>` markers, thought channel closed) | Gemma 4 E4B, 12B |
| `gemma` (other) | Gemma | Gemma 2, Gemma 3 |
| `lfm`, `granite`, `smollm` | ChatML | LFM2, Granite 4, SmolLM3 |
| `mistral` + `instruct`, or `ministral` | Mistral Instruct | Mistral 7B Instruct, Ministral 3 8B |
| `phi-3` / `phi-4` / `phi3` / `phi4` | Phi-3/4 | Phi-3 Mini, Phi-4 |
| `phi-2` / `phi2` | Phi-2 | Phi-2 |
| (anything else) | Simple (generic fallback) | Unknown models |

Sophro's file name contains none of these words; its architecture (`qwen3`) gives it the
Qwen format. Architectures map as follows: `qwen*` to Qwen, `gemma*` to Gemma, `llama*` to
Llama 3, `phi3*` to Phi-3/4, `mistral*` to Mistral Instruct, `granite*` / `lfm*` /
`smollm*` to ChatML.

A model matching nothing gets a generic `User:` / `Assistant:` format, which often gives
worse output than the model's own. To add a model family, add a detection case and a
prompt builder in `local_models.rs`; never put a model on the download list whose format
the engine does not handle.

**Sampling.** Two sampler chains, both with a fixed seed (42):

| Use | Chain |
|-----|-------|
| Replies | temperature (0.7 unless the caller gives one) → top-k 40 → top-p 0.9 |
| Structured calls | grammar built from a JSON schema → temperature 0.1 |

A reply is limited to 512 new tokens unless the caller gives a number. Gemma 4's thought
text is kept out of the reply and out of the stream (`strip_thought_channel()`).

---

## The model calls behind one message

The memory system makes up to three model calls for a message. All three go to the same
backend the user chose for the conversation; there is no separate memory model.

| Call | What it does | Code | When it runs |
|------|--------------|------|--------------|
| 1 | The reply | `commands/chat.rs`, prompt from `ConversationEngine::build_prompt()` | At once, streamed |
| 2 | The facts worth keeping: one memory per fact, each with a title and date; a cloud model also returns tags, tone and named things | `chat::extract_facts()`, prompt and schemas in `memory_prompts.rs` | Afterwards, from the job queue |
| 3 | How each new fact relates to similar stored memories: `updates`, `contradicts`, `related` or `none` | `ask_llm_for_relationships()` and `ask_llm_for_links_only()` in `lib.rs` | After call 2, once per fact |

Calls 2 and 3 are a row in the `memory_jobs` table (`memory_jobs.rs`). One worker runs
the rows in order while the model is idle, so a slow phone never has two calls competing
and a row survives an interruption or a restart.

How the calls differ by backend:

| | Cloud provider | Local GGUF |
|---|----------------|------------|
| Call 2 output | Free-form JSON, parsed leniently, full schema (`EXTRACTION_SCHEMA`) | Held to `EXTRACTION_SCHEMA_SLIM` by the grammar sampler |
| Call 3 | `ask_llm_for_relationships()`: title, links and extras in one answer | The same, held to `RELATIONSHIP_SCHEMA`. When call 2 already gave a title, the fact is stored first and `ask_llm_for_links_only()` (`LINKS_SCHEMA`) asks for the links afterwards |
| Candidate memories shown to call 3 | 10 | 4 |
| An `updates` answer | Applied: the older memory is marked replaced | Turned into `contradicts`, so the user is asked on a card (`filter_local_links()`) |
| Link recorded as made by | `llm` | `local_llm` |

Call 3 is skipped when call 2 gave a title and no stored memory is similar. "Remember:"
skips call 2 and stores the text as given. Facts from one message are linked `related` to
each other without a model call. One more structured call, `ask_llm_for_extras()`
(`EXTRAS_SCHEMA`), adds date, tags and named things to a memory stored before those fields
existed; the `enrich_memory_backlog` command (`commands/memory_report.rs`) runs it.

The full pipeline is in [Memory Processing Pipeline](MEMORY_PROCESSING_PIPELINE.md); the
link types and the classifier prompt are in
[Relationship Detection Architecture](RELATIONSHIP_DETECTION_ARCHITECTURE.md).

---

## Adaptive Context Limits

The limits depend on the backend (`commands/chat.rs`, `conversation_engine.rs`):

| Setting | Cloud provider | Local GGUF (phone or desktop) | Custom endpoint |
|---------|----------------|-------------------------------|-----------------|
| Memory search limit | 30 | 10 | 10 |
| Memories included in the prompt (max) | 30 | 4, the fact only, cut to 300 characters | 7 |
| Conversation history included | 40 messages | 8 messages, each cut to 400 characters | 8 messages |
| Knowledge Base passages (explicit search) | 10 | 4 | 10 |
| Recalled memories in a web-search answer | yes | none | none |

A local model also gets a shorter instruction text and its own memory heading ("WHAT THE
USER HAS TOLD YOU BEFORE:"); see
[Prompt Construction Pipeline](PROMPT_CONSTRUCTION_PIPELINE.md).

Memories linked to a recalled memory are added on top of the search results: one hop, any
link type (`related`, `contradicts`, `updates`, `resolves`), strongest links first, at
most 8, skipping memories a newer one has replaced. The per-backend maximum above still
applies to the total.

---

## Model Categories

### System Models (Required)
- **Location**: `models/system/` in the app data folder
- **Purpose**: embeddings, named things, safety filtering
- **Managed by**: the setup wizard, on first run
- **User control**: No (required for operation)

### User Chat Models (Optional)
- **Location**: `models/user/` in the app data folder
- **Purpose**: local backends for conversation and for the memory pipeline
- **Managed by**: the app's download list, or the user copying a `.gguf` file in
  (`open_models_folder()` opens the folder)
- **User control**: Yes (can add and remove models)

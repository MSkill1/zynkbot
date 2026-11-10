# Zynkbot: Comprehensive Architecture Documentation

*Checked against the code on 2026-11-10 (`v1`, a7599eb).*

App version at this commit: 0.9.6 (`zynkbot_rust/src-tauri/Cargo.toml`, `tauri.conf.json`, `package.json`).

This is the overview. Each part has a short description here and a link to the document that holds the detail.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Technology Stack](#technology-stack)
3. [Core Architecture](#core-architecture)
4. [Component Deep Dive](#component-deep-dive)
5. [Data Flow](#data-flow)
6. [Security & Privacy](#security--privacy)
7. [Deployment](#deployment)
8. [Development Guide](#development-guide)
9. [References](#references)

---

## System Overview

### What is Zynkbot?

Zynkbot is a conversational assistant that keeps a memory of what the user tells it. The memory lives in a SQLite file on the user's device.

- **One Rust backend, three targets.** The same Rust code builds for Linux, Windows and Android through Tauri 2. The React frontend runs in Tauri's WebView on all three. On Android a second, native path (Kotlin calling Rust over JNI) answers hands-free voice turns without the WebView. See [ANDROID_ARCHITECTURE.md](ANDROID_ARCHITECTURE.md).
- **Chat backends.** Anthropic, OpenAI, xAI, Mistral, a custom OpenAI-compatible endpoint (for example Ollama), and local GGUF files run by llama.cpp. Local GGUF models run on desktop and on Android (CPU on Android). See [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).
- **On-device helper models.** Embeddings (all-MiniLM-L6-v2), entity extraction (bert-base-NER) and the safety classifier (toxic-bert) run on the device with Candle.
- **Memory.** Facts are extracted from the conversation by a model, stored one fact per row, and linked to each other. Recall combines name matching with embedding similarity.
- **Device to device.** ZynkSync (memories, history, links, keys), ZynkLink (files) and ZChat (messages) connect paired devices directly over TLS on port 57963.
- **Cloud backup (optional).** An encrypted backup to the user's own Cloudflare R2 bucket.

### Design Principles

1. **Local first.** Embeddings, entity extraction and safety classification run on the device. The database is a local file.
2. **Rust backend.** The backend is Rust. It links two native libraries: llama.cpp (through `llama-cpp-2`) and, on desktop, Vosk.
3. **Works without a network** when a local GGUF model is selected. Cloud backends, web search, weather, page reading and cloud backup need a connection.
4. **Source available.** Community source licence plus commercial licensing. See [LICENSE](../../LICENSE) and [COMMERCIAL_LICENSE.md](../../COMMERCIAL_LICENSE.md).

---

## Technology Stack

### Frontend

- **Framework**: React 19 (`react ^19.1.1`)
- **Build tool**: Create React App (`react-scripts 5.0.1`)
- **Runtime**: Tauri 2 WebView (`@tauri-apps/api ^2.10.0`, `@tauri-apps/cli ^2.9.4`)
- **State**: React hooks
- **IPC**: Tauri commands and events

### Backend (Rust)

Versions as written in `zynkbot_rust/src-tauri/Cargo.toml`:

```
Rust 1.77.2+ (edition 2021)
├── tauri 2.11.1                  # App framework (desktop and Android)
├── tokio 1                       # Async runtime
├── sqlx 0.8 (sqlite)             # SQLite client, queries built at run time
├── reqwest 0.12                  # HTTP client
├── axum 0.7                      # HTTPS server for device-to-device routes
├── tokio-rustls 0.26             # TLS for that server and its clients
├── candle-core / candle-nn / candle-transformers   # ML framework for the helper models
├── tokenizers 0.20               # BERT tokenization
├── hf-hub 0.4                    # Hugging Face downloads
├── llama-cpp-2 0.1.160           # Local GGUF inference (Android: "dynamic-backends" feature)
├── tokio-tungstenite 0.30        # OpenAI Realtime transcription (live dictation)
├── vosk 0.3                      # Offline speech recognition (Linux and Windows desktop)
├── scraper 0.20                  # HTML parsing for web search and page reading
└── uuid 1.0                      # Ids for devices, messages and sync rows
```

There is no mDNS crate in `Cargo.toml`. Devices are paired by code, not discovered.

### Database

- **SQLite**, embedded, through sqlx. No server process.
- 28 migrations in `zynkbot_rust/src-tauri/migrations/`, applied at startup by `db.rs` (`sqlx::migrate!`).
- Embeddings are stored as BLOBs. There is no vector index. Similarity is a linear scan with cosine similarity written in plain Rust (`memory.rs`, `cosine_similarity`).
- The Knowledge Base also has an FTS5 keyword index (migration `0004_kb_fts5.sql`).
- Every table is described in [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md).

### ML Models

| Model | Purpose | Runs on |
|-------|---------|---------|
| all-MiniLM-L6-v2 | Embeddings (384 dimensions) | Candle |
| dslim/bert-base-NER | Entity extraction | Candle |
| unitary/toxic-bert | Safety classification | Candle |
| Wake-word models (`hey_zynk.onnx`, `embedding_model.onnx`, `melspectrogram.onnx`, `silero_vad.onnx`) | "Hey Zynk" detection on Android | ONNX Runtime, in Kotlin |
| Vosk model | Offline speech recognition | Vosk (desktop: `vosk_desktop.rs`; Android: vosk-android) |

The three Candle models are downloaded by the installers, or at first run by `commands/setup_wizard.rs`. Their sizes are not stated in the code, so they are not given here.

Local chat models offered for download (`commands/setup_wizard.rs`, `download_user_model`; each download is checked against a SHA-256):

| File | Role | Size |
|------|------|------|
| Sophro-4B-memory-v1-Q4_0.gguf | The Zynkbot model, offered at first run | 2.4 GB |
| gemma-4-12b-it-Q4_K_M.gguf | Optional | 7.1 GB |
| gemma-4-E4B-it-Q4_K_M.gguf | Optional | 5.0 GB |
| Qwen3.5-9B-Q4_K_M.gguf | Optional | 5.7 GB |
| Ministral-3-8B-Instruct-2512-Q4_K_M.gguf | Optional | 5.2 GB |

Sizes are the ones shown in `src/components/SetupWizard.jsx` and in [BENCHMARKS.md](../BENCHMARKS.md) section 5. Any other GGUF file placed in the user models folder can be selected. Engine settings, prompt formats per model family and context sizes are in [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

---

## Core Architecture

### High-Level System Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  TAURI APP (Linux, Windows, Android)                             │
│                                                                  │
│  FRONTEND (React, zynkbot_rust/src/)                             │
│   App.jsx                  chat screen, sends every message      │
│   components/ChatMessage.jsx        one message, search prompt   │
│   components/MemoryManager.jsx      memory browser and editor    │
│   components/KnowledgeBaseManager.jsx, KnowledgeBasePanel.jsx    │
│   components/SetupWizard.jsx        first-run model downloads    │
│   components/OnboardingModal.jsx    first-run questions          │
│   components/APIKeyModal.jsx        keys and model choice        │
│   components/ControlsPage.jsx       System Controls page         │
│   components/EnsembleModal.jsx      ensemble mode                │
│   components/ConflictResolutionModal.jsx   contradiction cards   │
│   components/ZynkSyncPanel.jsx, ZynkLinkPanel.jsx, ZChatModal.jsx│
│                         │ Tauri commands and events              │
│                         ↓                                        │
│  RUST BACKEND (zynkbot_rust/src-tauri/src/)                      │
│   lib.rs          module list, command registration, startup,    │
│                   and the relationship calls of the memory       │
│                   pipeline                                       │
│   commands/       Tauri command handlers (20 modules)            │
│     chat.rs       send_message_with_memory, generate_reply,      │
│                   extract_facts, run_memory_background,          │
│                   run_ensemble                                   │
│   conversation_engine.rs   prompt building                       │
│   memory.rs                hybrid search, memory rows, links     │
│   memory_jobs.rs, memory_prompts.rs, memory_extras.rs            │
│   conversation_history.rs  threads and messages                  │
│   containment.rs, safety_classifier.rs                           │
│   knowledge_base.rs, kb_rag.rs, kb_prompt.rs, word_docs.rs       │
│   web_search.rs, web_read.rs, weather.rs, chat_modes.rs          │
│   transport.rs, tls.rs     device identity, mTLS, peer registry  │
│   zynksync.rs, sync_outbox.rs, zynklink.rs, zchat.rs             │
│   android_jni.rs           native hands-free path on Android     │
│   llm/  anthropic.rs, openai.rs, xai.rs, mistral.rs,             │
│         local_models.rs (llama.cpp), local_embeddings.rs         │
│                         │ sqlx                                   │
│                         ↓                                        │
│   SQLite file: zynkbot.db in the app data folder                 │
└──────────────────────────────────────────────────────────────────┘
```

The database path comes from `db.rs`, `get_db_path`: the platform's local data folder plus `zynkbot/zynkbot.db` on desktop (on Linux, `~/.local/share/zynkbot/zynkbot.db`), and the app's own files folder on Android.

`lib.rs` is not a thin dispatcher. Besides registering commands it holds the relationship calls (`ask_llm_for_relationships`, `ask_llm_for_links_only`), `is_query_about_zynkbot` and startup code.

---

## Component Deep Dive

### 1. Reply path

**Location**: `src-tauri/src/commands/chat.rs`. `send_message_with_memory` is the Tauri command. It wraps the window in a `TauriSink` and calls `generate_reply`, which does the work and does not depend on Tauri. On Android, `android_jni.rs` calls the same `generate_reply` for hands-free voice turns, with `hands_free` set and a note appended to the transcript (`HANDS_FREE_NOTE`).

What `generate_reply` does, in order:

1. **Backend choice.** Child mode forces the OpenAI backend. If the selected backend has no key, or a local model cannot be used, `get_best_available_backend` picks another.
2. **History thread.** The thread is opened in History before the reply (not in HIPAA mode, not for hands-free turns).
3. **Weather.** A weather question (`weather::detect`) is answered from Open-Meteo by `weather::answer`, with no model call. If that fails, the message goes to the model as usual.
4. **Safety check.** Child mode calls the OpenAI Moderation API and blocks if the API fails. Other modes go through `commands::safety::check_containment` to `ContainmentLayer::enforce`. In Sovereign mode a flagged message gets a warning in front of the reply. In Guardian and HIPAA mode it is blocked. See [section 9](#9-safety-layer).
5. **Entities and embedding.** BERT NER extracts names from the message (common words are filtered out). `llm::local_embeddings::generate_local_embedding` makes the 384-dimension vector.
6. **Memory search.** `memory::hybrid_search` returns up to 30 memories for a cloud backend and 10 otherwise. A question about Zynkbot itself (`is_query_about_zynkbot`) searches only the built-in `_zynkbot` memories. See [section 3](#3-memory-search).
7. **Linked memories.** Up to 8 memories linked to the results are added, strongest links first. Superseded and completed memories are skipped.
8. **Knowledge Base**, only when the KB button is on. See [section 8](#8-knowledge-base).
9. **Built-in guides.** For a question about Zynkbot with the KB button off, passages from the built-in user guides are searched and added (`kb_prompt::build_guides`).
10. **Prompt.** `ConversationEngine::build_prompt` builds one of three prompt variants: full (cloud), on-device (local GGUF) or slim (custom endpoint). History is 40, 8 or 8 messages and memories are 30, 4 or 7, in that order. Detail in [PROMPT_CONSTRUCTION_PIPELINE.md](PROMPT_CONSTRUCTION_PIPELINE.md).
11. **Blocks in front of the prompt.** A conversation mode block (`chat_modes.rs`: tool, debate, premise, editing), the KB block, the guide passages, and the text of up to two web pages whose addresses appear in the message (`web_read.rs`).
12. **Pictures.** For cloud backends, attached pictures are sent with the prompt, together with the newest three pictures sent earlier in the thread. The custom endpoint and local models get no pictures.
13. **Model call.** The reply streams token by token to the sink for every backend, local models included. An empty reply is an error.
14. **Web search marker.** If the reply contains `WEB_SEARCH_NEEDED:`, the suggested query is returned to the frontend, which shows it in an editable box (`ChatMessage.jsx`) and searches when the user confirms. A hands-free turn runs the search itself without asking. `web_search.rs` uses the cloud provider's own search when one is available, otherwise Brave, SearXNG or DuckDuckGo depending on what is configured.
15. **Clean-up.** The `MEMORY:` line and any `<think>` block are removed from the displayed reply. HIPAA mode adds a medical disclaimer when health terms are detected.
16. **Return.** The reply is returned and the exchange is written to conversation history (not in HIPAA mode).

The memory work in [section 2](#2-memory-pipeline) starts after the reply has been returned.

### 2. Memory pipeline

One user message can lead to three model calls:

1. **Call 1, the reply.** Cloud and custom prompts ask the model to end with `MEMORY: yes` or `MEMORY: no`. The on-device prompt does not ask for this line.
2. **Call 2, the facts.** `chat.rs`, `extract_facts`, with the prompt from `memory_prompts.rs`. It returns up to three facts, each with a title.
3. **Call 3, the relationships.** `lib.rs`, `ask_llm_for_relationships` (or `ask_llm_for_links_only`), run per fact inside `chat.rs`, `run_memory_background`.

Calls 2 and 3 do not run inline. After the reply, `generate_reply` writes a row to the `memory_jobs` table (migration 0023). One worker (`memory_jobs.rs`) takes jobs oldest first, retries a failed job up to four attempts, and picks up unfinished jobs after a restart.

Before a job is queued:

- Nothing is queued in HIPAA mode, when memory storage is switched off for the turn, or when a Knowledge Base reply was not grounded in the documents.
- The message must pass `ConversationEngine::is_memory_worthy` (at least three words, not filler) and must not be a rewrite or translate request.
- A cloud or custom reply must end with `MEMORY: yes`. An on-device reply proceeds unless the model said no.
- A message that starts with "Remember:" skips these checks.

For each fact, `run_memory_background`:

- makes the embedding of the fact,
- searches 15 memories and keeps those scoring 0.35 or more,
- drops the fact as a duplicate when the hybrid score is above 0.98 or the cosine similarity is above 0.93,
- asks call 3 how the fact relates to the similar memories,
- stores the fact and its links.

There are four link types (migration 0022): `related`, `contradicts`, `updates` and `resolves`. An `updates` link marks the older memory as superseded, and superseded memories are left out of recall. A `contradicts` link with confidence 0.65 or more produces a card for the user (`ConflictResolutionModal.jsx`); the card is saved in `memory_cards` (migration 0024) so it survives a restart. With a cloud backend the card is shown before the fact is stored. With a local model the fact is stored first and the question is asked afterwards. If call 3 fails, the fact is stored without links. There is no rule-based contradiction check in the code.

Two more behaviours sit on top of storage:

- **Passing facts.** A fact that call 2 marks as passing (an errand, this week's trip) gets an `expires_at` date 14 days after its day (`memory_extras.rs`, `PASSING_DAYS`) and leaves recall after that.
- **Check-ins.** A passing fact that was an intention can be asked about once, shortly before it leaves recall (`commands/checkin.rs`, migration 0026). The feature is marked experimental in the code and has a switch on the System Controls page.

Full detail: [MEMORY_PROCESSING_PIPELINE.md](MEMORY_PROCESSING_PIPELINE.md). Links and the graph: [MEMORY_RELATIONSHIP_GRAPH.md](MEMORY_RELATIONSHIP_GRAPH.md) and [RELATIONSHIP_DETECTION_ARCHITECTURE.md](RELATIONSHIP_DETECTION_ARCHITECTURE.md).

### 3. Memory search

**Location**: `src-tauri/src/memory.rs`

```rust
pub async fn hybrid_search(
    pool: &SqlitePool,
    query_embedding: Vec<f32>,
    query_entities: Vec<String>,
    hints: &QueryHints,
    user_id: Option<&str>,
    session_id: Option<&str>,
    namespace: Option<&str>,
    limit: i32,
) -> Result<Vec<Memory>, sqlx::Error>
```

- The SQL leaves out superseded and completed memories, expired memories, and system memories unless the search is for the `system` user.
- Every remaining row is scored in Rust. There is no vector index.
- The score comes from `hybrid_score`. When a name in the question matches a name stored with the memory, the base score is 60% entity match plus 40% embedding similarity. When no name matches, the base score is the embedding similarity alone. A namespace match and shared tags add a small boost.
- A row is kept when its score is 0.35 or more (`MIN_SIMILARITY`), or when a name matched.
- Results are sorted by score and cut to `limit`.

How search quality is measured: [BENCHMARKS.md](../BENCHMARKS.md) section 2.

### 4. Ensemble Mode

**Location**: `src-tauri/src/commands/chat.rs`, `run_ensemble`. User-facing description: [ENSEMBLE_MODE.md](../ENSEMBLE_MODE.md).

```rust
pub async fn run_ensemble(
    _app: tauri::AppHandle,
    message: String,
    models: Vec<String>,
    user_id: String,
    session_id: String,
    containment_mode: String,
    kb_enabled: Option<bool>,
    user_query: Option<String>,
    image_data: Option<Vec<crate::llm::ImageAttachment>>,
) -> Result<serde_json::Value, String>
```

- At least two models are required.
- One model is the coordinator. The order of preference is Anthropic, xAI, OpenAI, Mistral, custom endpoint, then the first model listed.
- **Phase 0.** The coordinator is asked whether the question needs a web search (yes or no). A local coordinator is not asked, so no search runs. A search fetches three results with a 5-second timeout.
- **Phase 1.** All models answer in parallel. Each gets the same memory context (top 5 memories scoring 0.50 or more) and, if the KB button is on, the same Knowledge Base passages.
- **Phase 2.** The coordinator is told to evaluate the answers, not average them, and writes the final answer.
- The result holds `individual_responses`, `synthesized_response`, `coordinator_model`, `models_used`, `successful_models` and `search_results`.
- `run_ensemble` runs the containment check of the mode in force on the question (since 2026-11-10) and, in HIPAA mode, removes identifiers from the question and the shared context. It stores no memories. The UI does not offer ensemble in Child mode.

### 5. ZynkSync (device-to-device sync)

**Location**: `src-tauri/src/sync_outbox.rs` (queue, drain, receiver), `src-tauri/src/zynksync.rs` (service, pairing, routes), `src-tauri/src/transport.rs` (identity, mTLS, device registry, shared with ZynkLink and ZChat)

**Purpose**: keep memories, conversation history, memory links and keys the same across a user's devices over the local network, including devices that were off.

**Architecture** (outbox):
```
any write to memories / conversation_sessions / conversation_messages / memory_links / sync_secrets
   ↓ SQLite trigger (migrations 0013, 0014, 0016)
sync_outbox (id, table_name, row_sync_id, op, payload, created_at)
   ↓ every 60 seconds, and within seconds of a local change; per paired peer, from the peer's cursor
drain: collapse to one entry per row, load each live row with every column, at most 300 rows
   ↓ HTTPS POST /api/zynksync/outbox   (mTLS, pinned certificates, x-target-device-id)
receiver: one transaction under sync_suppress
   → upsert by sync_id (newest updated_at wins), delete by sync_id, tombstone for memories
   → unknown sync_id: match by content (memory) / message key / (source, target, relation) (link), adopt the name
   → records sync_inbox_cursor[sender] = batch.through
   ← OutboxReceipt { through, applied, known_before }
sender: sync_outbox_cursor[peer] = through  (or start the peer over if known_before < cursor)
```
`sync_bidirectional` is a drain to the peer followed by `/api/zynksync/outbox/pull`, which asks the peer to drain to us.

**First contact and recovery**: a peer with no cursor row, a peer whose cursor points before the oldest queued row, or a peer that reports knowing less than our cursor says, is sent the live tables in a fixed order, 300 rows per cycle. After that the queue takes over.

**Identity**: every request carries `x-device-id`, `x-device-name` and `x-device-port` headers. `require_verified_device` rejects a request addressed to another device (HTTP 421). `inject_verified_device` corrects a known device's address and port from the connection. Device identity is not part of the cloud backup; a reinstalled device pairs again by code.

**Key points**:
- Port 57963, mutual TLS, certificate pinned on both sides
- 6-digit pairing codes with a 10-minute expiry; a joining device adopts the host's user id and re-keys its existing rows to it
- Keys travel as rows (`sync_secrets`); the newest `updated_at` per key wins
- Outbox rows are kept for at most 7 days (`OUTBOX_RETENTION_DAYS`)
- An offline peer is skipped for two minutes after a refused connection

User-facing description: [NETWORKING_FEATURES.md](../NETWORKING_FEATURES.md).

### 6. ZynkLink (device-to-device file sharing)

**Location**: `src-tauri/src/zynklink.rs`, `src-tauri/src/commands/zynklink.rs`

- Uses the same server and mutual TLS as ZynkSync. Every request goes to `https://<ip>:57963/api/zynklink/...`.
- Has its own pairing codes (`generate_zynklink_code`, `link_with_zynklink_code`).
- A device shares a folder with read and write flags (`share_directory`).
- A paired device can list and browse shared folders, download a file to a chosen location, or download it straight into the Knowledge Base (`download_to_knowledge_base`). Downloads can be paused and cancelled.
- The ZynkLink code computes no file checksum. Integrity rests on the TLS connection.

### 7. ZChat (device-to-device messaging)

**Location**: `src-tauri/src/zchat.rs`, `src-tauri/src/commands/zchat.rs`

- Messages are stored in the local database with UUID ids and `delivered_at` and `read_at` times.
- Sending stores the message and tries to deliver it at once (`deliver_to_peer`). Undelivered messages are retried on the next send and by the ZynkSync loop.

### 8. Knowledge Base

**Location**: `src-tauri/src/knowledge_base.rs` (files and documents), `src-tauri/src/kb_rag.rs` (chunking, indexing, search), `src-tauri/src/kb_prompt.rs` (prompt block), `src-tauri/src/word_docs.rs` (Word files)

1. Supported file types (`SUPPORTED_EXTENSIONS`): txt, md, json, csv, log, rs, js, jsx, ts, tsx, py, java, cpp, c, h, html, css, xml, yaml, yml, toml, pdf, docx, doc.
2. A document is split into chunks of about 128 tokens with a 25-token overlap (`chunk_document(&content, 128, 25)`).
3. Each chunk is embedded with all-MiniLM-L6-v2 and stored in `kb_chunks`. There is no vector index.
4. A search (`search_kb_chunks`) scores chunks by cosine similarity and merges in FTS5 keyword matches.
5. In a normal reply the search runs only when the KB button is on: 10 chunks for cloud and custom backends, 4 for a local model, with a 15% similarity threshold.
6. `kb_prompt::build` writes the block that goes in front of the prompt. It has three outcomes: found, weak matches only, nothing. Only "found" counts as grounded. For the other two the reply carries a note for the UI and no memory is stored from that reply.

Tables: `kb_documents`, `kb_chunks` (see [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md)).

### 9. Safety Layer

**Location**: `src-tauri/src/containment.rs` (modes and rules), `src-tauri/src/safety_classifier.rs` (classifier)

There are six containment modes: witness, sovereign, guardian, child, elder and hipaa.

The classifier is toxic-bert run with Candle:

```rust
pub fn classify_text(text: &str) -> Result<SafetyResult, String>
```

`SafetyResult` holds `category`, `confidence` and `all_scores` for six classes, each scored with a sigmoid. `should_block(text, mode)` turns the result into a decision: witness and sovereign never block; guardian and hipaa block severe toxicity or threats above 0.9. A comment in the code calls the classifier demo-grade.

`ContainmentLayer::enforce` adds to this:

- a pattern check for harmful instruction requests, which can block on its own,
- in Sovereign mode, a warning instead of a block,
- in HIPAA mode, checks for personal health information, diagnosis requests and medication dosing before the general check,
- in Child mode, the OpenAI Moderation API first and the local classifier second. Child mode needs an OpenAI key and blocks when the API cannot be reached.

If the safety check itself fails outside Child mode, `generate_reply` lets the message through.

Detail: [CONTAINMENT_ARCHITECTURE.md](CONTAINMENT_ARCHITECTURE.md) and [HIPAA_SECURITY_INTEGRATION_GUIDE.md](HIPAA_SECURITY_INTEGRATION_GUIDE.md).

### 10. Cloud backup

**Location**: `src-tauri/src/commands/backup.rs`, UI in `src/components/CloudBackupSettings.jsx`

- `backup_memories_to_r2` uploads one encrypted object, `backup.enc`, to the user's Cloudflare R2 bucket.
- Encryption is AES-256-GCM. The key is in a local file, `backup.key`. It is random, or derived from a passphrase with Argon2id.
- The backup holds memories (not the built-in `_zynkbot` ones), memory links, keys and settings, the wake-word voice profile, and conversation threads with their messages.
- `restore_memories_from_r2` restores memories (deleted ones stay deleted), links (matched by memory text), threads, messages, and keys the device does not already have.
- Device identity and pairings are not in the backup.

### 11. Other shipped features

- **Thread seal** (`seal.rs`): the user can export one thread as a file with a SHA-256 digest and an Ed25519 signature made by a key that stays on the device. Nothing is written to the database.
- **Report a problem** (`commands/report.rs`, `build_problem_report`): builds a text report on the device for the user to paste into a GitHub issue. The code sends nothing.
- **Memory report** (`commands/memory_report.rs`): a "what do you know about me" report built from the local memory graph.
- **Voice**: desktop dictation through Vosk (`vosk_desktop.rs`) or OpenAI transcription (`openai_realtime.rs`). The Android wake word, assistant role and hands-free path are in [ANDROID_ARCHITECTURE.md](ANDROID_ARCHITECTURE.md).
- **Einstein demo** (`einstein_demo.rs`): a sample memory set a new user can load.

---

## Data Flow

**Example**: the user types "What's my favorite color?" with a cloud backend selected.

1. `App.jsx` calls `invoke("send_message_with_memory", ...)`.
2. `commands/chat.rs`, `generate_reply`: the History thread is opened, the safety check passes, entities and the embedding are made.
3. `memory::hybrid_search` returns the memory "User's favorite color is blue" among its results.
4. `ConversationEngine::build_prompt` builds the full prompt with history and memories.
5. The model's reply streams to the window. Its last line is `MEMORY: no`, which is removed before display.
6. The exchange is written to conversation history.
7. No memory job is queued: the message is a question and the model answered `MEMORY: no`.

Had the user typed "My favorite color is green now", the reply would end `MEMORY: yes`, a job would be queued, call 2 would extract the fact, and call 3 would most likely link it to the older memory as `updates` or `contradicts`.

---

## Security & Privacy

### What stays on the device

1. Embeddings, entity extraction and safety classification run on the device.
2. The database, the log (`logs/zynkbot.log`) and the user profile are local files.
3. With a local GGUF model selected, the reply and the memory calls run on the device.
4. ZynkSync, ZynkLink and ZChat connect paired devices directly over mutual TLS with pinned certificates. No relay server appears in the code.

### What leaves the device

- With a cloud backend or a custom endpoint, the prompt (the message, recent history, recalled memories, any KB passages) goes to that provider. So do the calls of the memory pipeline.
- Child mode sends each message to the OpenAI Moderation API.
- Web search, weather (Open-Meteo), reading a web page named in a message, and model downloads (Hugging Face) make network requests.
- Live dictation through OpenAI sends audio to OpenAI when the user chooses it.
- Cloud backup uploads the encrypted `backup.enc` to the user's R2 bucket when the user sets it up.

Keys are entered by the user. A search of the Rust and React sources for analytics or telemetry libraries finds none, and "Report a problem" sends nothing by itself. This is a statement about the source at this commit, not an audit of every dependency.

### HIPAA mode

Selected as a containment mode. In `generate_reply`:

- no memory is extracted or stored,
- no conversation history is written,
- messages are checked for personal health information, diagnosis requests and medication dosing before the model is called,
- a medical disclaimer is added to replies that mention health terms.

HIPAA mode does not give memories an 8-hour expiry: it stores none. Detail in [HIPAA_SECURITY_INTEGRATION_GUIDE.md](HIPAA_SECURITY_INTEGRATION_GUIDE.md).

---

## Deployment

### Installation scripts (build from source)

**Windows:**
```
install.bat              (run as Administrator)
START_ZYNKBOT.bat
```

**Linux:**
```bash
./install.sh
./START_ZYNKBOT.sh
```

Both installers list the same eight steps: install system dependencies, install the Rust toolchain, detect a GPU and configure CUDA, configure the environment, install Node dependencies, create the model folders, download the three system models, and offer the local chat models (Sophro first; each download is checked against a SHA-256).

Neither script creates the database. The app creates the SQLite file and applies its migrations at startup.

Guides: [LINUX_INSTALLATION_GUIDE.md](../LINUX_INSTALLATION_GUIDE.md), [WINDOWS_INSTALLATION_GUIDE.md](../WINDOWS_INSTALLATION_GUIDE.md), [INSTALLATION_TROUBLESHOOTING.md](../INSTALLATION_TROUBLESHOOTING.md), [ANDROID_USER_GUIDE.md](../ANDROID_USER_GUIDE.md).

---

## Development Guide

### Prerequisites

- **Rust**: 1.77.2 or newer (`rust-version` in `Cargo.toml`)
- **Node.js**: 18 or newer (`engines` in `package.json`)
- **SQLite**: nothing to install; it is embedded

### Build

**Development:**
```bash
cd zynkbot_rust
npm install
npm run tauri:dev
```

**Production build:**
```bash
cd zynkbot_rust
npm run tauri:build
```

On Windows, build and run in the release profile. Android builds have their own steps. Both are described in `zynkbot_rust/CLAUDE.md` and [ANDROID_ARCHITECTURE.md](ANDROID_ARCHITECTURE.md).

There is no `cargo sqlx prepare` step. The code uses run-time queries (`sqlx::query`), not the compile-time `query!` macros, and there is no `.sqlx` folder.

Tests are described in [TESTING.md](../TESTING.md).

### Project Structure

```
(repository root)
├── zynkbot_rust/                  # The app
│   ├── src/                       # React frontend
│   │   ├── index.js               # Entry point
│   │   ├── App.jsx                # Chat screen
│   │   ├── components/            # 36 components (modals, panels, pages)
│   │   ├── hooks/
│   │   ├── styles/
│   │   └── utils/
│   ├── public/
│   ├── package.json
│   └── src-tauri/                 # Rust backend
│       ├── src/
│       │   ├── lib.rs             # Modules, command registration, startup, relationship calls
│       │   ├── main.rs
│       │   ├── commands/          # backup, chat, checkin, conversation, knowledge_base,
│       │   │                      # memory, memory_report, models, nlp, onboarding, report,
│       │   │                      # safety, setup_wizard, sync_codes, user_identity, utils,
│       │   │                      # web_search, zchat, zynklink, zynksync
│       │   ├── llm/               # anthropic, openai, xai, mistral, local_models,
│       │   │                      # local_embeddings
│       │   ├── conversation_engine.rs   # Prompt building
│       │   ├── conversation_history.rs  # Threads and messages
│       │   ├── memory.rs                # Hybrid search, memory rows, links
│       │   ├── memory_jobs.rs           # Memory job queue and worker
│       │   ├── memory_prompts.rs        # Prompts of the memory pipeline
│       │   ├── memory_extras.rs         # Namespaces, tags, passing facts
│       │   ├── nlp_enhancer.rs          # Titles, tags, events, BERT NER
│       │   ├── containment.rs           # Containment modes
│       │   ├── safety_classifier.rs     # toxic-bert classifier
│       │   ├── knowledge_base.rs, kb_rag.rs, kb_prompt.rs, word_docs.rs
│       │   ├── web_search.rs, web_read.rs, weather.rs, chat_modes.rs
│       │   ├── transport.rs, tls.rs     # Identity, mTLS, peers
│       │   ├── zynksync.rs, sync_outbox.rs, sync_codes.rs
│       │   ├── zynklink.rs, zchat.rs
│       │   ├── android_jni.rs, voice_prefs.rs, wake_profile.rs
│       │   ├── vosk_desktop.rs, openai_realtime.rs
│       │   ├── seal.rs, app_log.rs, env_file.rs, db.rs, user_identity.rs
│       │   ├── response_sink.rs         # Where a reply goes (window, JNI, tests)
│       │   └── memory_bench.rs, recall_bench.rs, reply_bench.rs   # Benches
│       ├── migrations/            # 28 SQL migrations
│       ├── models/
│       │   ├── system/            # all-MiniLM-L6-v2, bert-base-NER, toxic-bert
│       │   └── user/              # Local chat models (.gguf)
│       ├── gen/android/           # Android project, committed, partly hand-written
│       ├── tests/
│       ├── Cargo.toml
│       ├── tauri.conf.json
│       ├── tauri.linux.conf.json
│       └── tauri.windows.conf.json
├── docs/                          # Documentation
│   ├── architecture_and_development/
│   ├── case_studies/
│   ├── snap_ins/
│   ├── troubleshooting/
│   └── ROADMAP.md, FEATURES.md, BENCHMARKS.md, KNOWN_ISSUES.md, ...
├── knowledge_base/                # A sample Knowledge Base document
├── labs/                          # Experiments and benches
├── wake-word-training/
├── assets/
├── install.sh, install.bat        # Installers (Linux, Windows)
├── uninstall.sh, uninstall.bat
├── START_ZYNKBOT.sh, START_ZYNKBOT.bat
├── README.md, CHANGELOG.md, CONTRIBUTING.md, CLA.md, API_COSTS.md, DISCLAIMER.md
└── LICENSE, COMMERCIAL_LICENSE.md, TRADEMARK
```

Installed copies keep models in the app data folder; a development build uses `src-tauri/models/` (`db.rs`, `get_models_dir`).

---

## References

### Related Documentation

In this folder:

- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md): models, backends, local engine settings
- [PROMPT_CONSTRUCTION_PIPELINE.md](PROMPT_CONSTRUCTION_PIPELINE.md): how the prompt is built
- [MEMORY_PROCESSING_PIPELINE.md](MEMORY_PROCESSING_PIPELINE.md): the memory pipeline step by step
- [MEMORY_RELATIONSHIP_GRAPH.md](MEMORY_RELATIONSHIP_GRAPH.md) and [RELATIONSHIP_DETECTION_ARCHITECTURE.md](RELATIONSHIP_DETECTION_ARCHITECTURE.md): links between memories
- [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md): every table
- [CONTAINMENT_ARCHITECTURE.md](CONTAINMENT_ARCHITECTURE.md) and [HIPAA_SECURITY_INTEGRATION_GUIDE.md](HIPAA_SECURITY_INTEGRATION_GUIDE.md): containment modes
- [ANDROID_ARCHITECTURE.md](ANDROID_ARCHITECTURE.md): the Android app

Elsewhere in `docs/`:

- [MODELS.md](../MODELS.md): model information for users
- [NETWORKING_FEATURES.md](../NETWORKING_FEATURES.md): ZynkSync, ZynkLink and ZChat for users
- [ENSEMBLE_MODE.md](../ENSEMBLE_MODE.md)
- [BENCHMARKS.md](../BENCHMARKS.md)
- [ROADMAP.md](../ROADMAP.md)
- [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)

### External Resources

- [Tauri Documentation](https://tauri.app/)
- [Candle ML Framework](https://github.com/huggingface/candle)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Rust Book](https://doc.rust-lang.org/book/)

---

**License**: Zynkbot Community Source License; commercial licensing available, see [COMMERCIAL_LICENSE.md](../../COMMERCIAL_LICENSE.md)
**Contact**: [GitHub Issues](https://github.com/MSkill1/zynkbot/issues)

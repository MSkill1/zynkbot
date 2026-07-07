# Zynkbot: Comprehensive Architecture Documentation

**Last Updated:** 2026-06-17
**Status:** Pure Rust/Tauri Production Implementation
**Version:** 0.9

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

---

## System Overview

### What is Zynkbot?

Zynkbot is a **privacy-first conversational AI system** with persistent semantic memory. It combines:

- **Local-First AI**: All core ML operations run on-device using pure Rust frameworks (Candle)
- **Semantic Memory**: SQLite with in-process vector similarity search (Candle)
- **Hybrid Search**: Entity-based (BERT NER) + semantic embeddings for intelligent memory recall
- **Cross-Platform**: Native desktop app for Windows, Linux, and macOS via Tauri
- **Device Sync**: ZynkSync, ZynkLink, ZChat for device-to-device capabilities
- **Hybrid LLM Support**: Local GGUF models (llama.cpp) + API backends (OpenAI, Anthropic, xAI)

### Design Principles

1. **Privacy First**: All sensitive operations run locally (embeddings, NER, safety classification)
2. **Pure Rust Core**: Zero Python dependencies in production - fully Rust-powered
3. **Desktop-Native**: Tauri 2.x cross-platform desktop app with native OS integration
4. **Offline-Capable**: Full functionality without internet (except API LLMs)
5. **Open Source**: Dual-licensed, see license in documentation
---

## Technology Stack

### Frontend

- **Framework**: React 18
- **Build Tool**: Vite
- **Desktop Runtime**: Tauri 2.9.2
- **State Management**: React Hooks
- **IPC**: Tauri command system

### Backend (Pure Rust)

```
Rust 1.77.2+ (Edition 2021)
├── tauri 2.9.2              # Desktop app framework
├── tokio 1.x                # Async runtime
├── sqlx 0.8                 # SQLite client (async, compile-time checked queries)
├── reqwest 0.12             # HTTP client
├── candle-core/nn/transformers  # Pure Rust ML framework
│   ├── all-MiniLM-L6-v2     # Embeddings (384-dim)
│   └── dslim/bert-base-NER  # Named Entity Recognition
├── tokenizers 0.20          # BERT tokenization
├── llama-cpp-2 0.1          # Local GGUF model inference
├── hf-hub 0.4               # Hugging Face model downloads
├── uuid 1.x                 # UUID generation for message/memory IDs
├── mdns-sd 0.11             # mDNS device discovery (future)
└── scraper 0.20             # Web search HTML parsing
```

### Database

- **SQLite** (embedded, no server process) via sqlx
- Vector similarity search computed in-process in Rust (cosine similarity via Candle)
- JSON stored as TEXT for flexible metadata
- Standard B-tree indexes; no external vector index extension required

### ML Models

| Model | Purpose | Size | Framework |
|-------|---------|------|-----------|
| all-MiniLM-L6-v2 | Embeddings | 80MB | Candle |
| dslim/bert-base-NER | Entity extraction | 260MB | Candle |
| toxic-bert | Safety filtering | 260MB | Candle |
| DeepSeek-R1-Distill-Llama-8B-Q4 | Local chat LLM (optional) | 4.9GB | llama.cpp |
| Qwen3-8B-Q4                      | Local chat LLM (optional) | 4.7GB | llama.cpp |
| Llama-3.1-8B-Lexi-Uncensored-Q4  | Local chat LLM (optional) | 4.7GB | llama.cpp |

---

## Core Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TAURI DESKTOP APP                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  FRONTEND (React)                                     │  │
│  │  • Chat.jsx - Conversation interface                 │  │
│  │  • MemoryManager.jsx - Memory browser & editor       │  │
│  │  • Settings.jsx - Model selection, API keys          │  │
│  │  • KnowledgeBase.jsx - Document upload & search      │  │
│  │  • Onboarding.jsx - First-time setup                 │  │
│  │  • EnsembleModal.jsx - Multi-model consensus         │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │ Tauri IPC Commands                   │
│                     ↓                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  RUST BACKEND (src-tauri/src/)                       │  │
│  │                                                        │  │
│  │  [Command Handlers] (commands/)                       │  │
│  │  • memory.rs      → Memory CRUD, links, graph        │  │
│  │  • onboarding.rs  → Onboarding, Einstein demo        │  │
│  │  • conversation.rs → Session history, prompt builder │  │
│  │  • nlp.rs         → Entity extraction, facts         │  │
│  │  • models.rs      → API keys, model discovery        │  │
│  │  • safety.rs      → Containment modes, classifier    │  │
│  │  [Core Commands] (lib.rs — thin dispatcher)          │  │
│  │  • send_message_with_memory()  → Conversation flow   │  │
│  │  • run_ensemble()              → Multi-model mode    │  │
│  │                                                        │  │
│  │  [Core Services]                                      │  │
│  │  ├── memory.rs - Hybrid search, CRUD operations      │  │
│  │  ├── knowledge_base.rs - KB indexing & document mgmt  │  │
│  │  ├── kb_rag.rs - Knowledge base RAG retrieval         │  │
│  │  ├── safety_classifier.rs - Content filtering        │  │
│  │  ├── nlp_enhancer.rs - Entity extraction (BERT NER)  │  │
│  │  ├── zynksync.rs - Device-to-device memory sync      │  │
│  │  ├── zynklink.rs - Device-to-device file sharing     │  │
│  │  ├── zchat.rs - Device-to-device messaging           │  │
│  │  └── llm/                                             │  │
│  │      ├── local_embeddings.rs (all-MiniLM-L6-v2)      │  │
│  │      ├── local_models.rs (llama.cpp GGUF)            │  │
│  │      ├── anthropic.rs (Claude API)                   │  │
│  │      ├── openai.rs (GPT API)                         │  │
│  │      └── xai.rs (Grok API)                           │  │
│  │                                                        │  │
│  │  [Data Layer]                                         │  │
│  │  └── SQLite (via sqlx, embedded — no server)         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
                ┌───────────────────────────┐
                │  SQLite Database          │
                │  (~/.local/share/zynkbot/ │
                │   zynkbot.db on Linux)    │
                └───────────────────────────┘
```

---

## Component Deep Dive

### 1. Conversation Engine

**Location**: `src-tauri/src/commands/chat.rs::send_message_with_memory()`

**Responsibilities**:
1. Orchestrate conversation flow
2. Build prompts with recalled memories
3. Route requests to appropriate LLM backend
4. Store user messages and responses as memories

**Conversation Flow: Three-Phase Architecture**

Zynkbot uses a **three-phase conversation architecture** to ensure UI responsiveness while maintaining comprehensive memory processing:

1. **Phase 1 (Synchronous)**: Pre-prompt construction and context gathering
2. **Phase 2 (Synchronous)**: LLM inference and immediate response display
3. **Phase 3 (Asynchronous)**: Background memory storage and relationship detection

This ensures the user sees the AI response **immediately after LLM inference**, and all memory processing happens in the background without blocking the UI.

---

### Phase 1: Pre-Prompt Construction (Synchronous)

```
User Input
   ↓
[1] Safety Check
    • Child Mode: OpenAI Moderation API
    • Other Modes: Local toxic-bert classifier
    • Sovereign Mode: Warning prefix (continue with LLM)
    • Guardian/HIPAA: Hard block certain content pre-LLM
   ↓
[2] Entity Extraction + Embedding Generation (Parallel)
    • BERT NER: Extract entities from query
    • Stop word filtering (removes 100+ common words)
    • Embedding: Generate 384-dim vector (all-MiniLM-L6-v2)
   ↓
[3] Hybrid Memory Search
    • Entity-based search: in-process entity overlap (JSON, Rust)
    • Semantic search: in-process cosine similarity (Candle)
    • Weighted scoring: 60% semantic + 40% entity
    • Returns top 5 most relevant memories
    • Smart filtering: System memories only if query about Zynkbot
   ↓
[4] Knowledge Base Search (if enabled via UI button)
    • Generate embedding for query
    • Search indexed documents (in-process cosine similarity)
    • Return top 10 relevant chunks (higher limit reflects explicit user intent to query their KB)
    • Build KB context string
   ↓
[5] Build Prompt
    • System instructions (containment mode/keyword specific)
    • Recalled memories (formatted as context)
    • KB context (if enabled)
    • Conversation history (last N turns)
    • User message
```

**Code Location:** `lib.rs` (Phase 1)

---

### Phase 2: LLM Inference and Display (Synchronous)

Depending on model (local GGUF 5-60s, API 1-30s)

```
[6] LLM Inference
    • Local GGUF: llama.cpp via llama-cpp-2 crate
    • API: OpenAI, Anthropic, or xAI
   
[7] Check if Web Search Needed
    • LLM can request web search via "WEB_SEARCH_NEEDED:" marker
    • Extract search query from response
    • Return web_search_needed flag to frontend
   ↓
[8] RETURN RESPONSE TO FRONTEND → ⚡ DISPLAY IMMEDIATELY
```

**Critical Point:** Response is displayed to the user **immediately after step 8**. Steps 9-21 happen in the background and do NOT block the UI.

**Code Location:** `lib.rs` (Phase 2)

---

### Phase 3: Background Processing (Asynchronous)

**Runs in background Tokio task — does not block the UI.**

**IMPORTANT:** This entire phase runs asynchronously AFTER the response is displayed. The user never waits for these steps.

```
[9] Extract Factual Statements
    • LLM analyzes the full user message + assistant response for personal facts
    • Questions and instructions ARE processed — personal facts are extracted from context
    • Extracted facts are written as declarative statements in third-person + first-person form
   ↓
[10] Generate Embedding for User Message
    • 384-dim vector for semantic search
    • Used for duplicate detection and relationship classification
   ↓
[11] Broader Hybrid Search for Relationships
    • Search 15 memories (vs 5 for conversation)
    • Filter to >35% similarity threshold
    • Sort by relevance (most similar first)
   ↓
[12] Duplicate Detection (Skip Storage if Duplicate)
    • Check 1: Hybrid score >98% (entity + semantic)
    • Check 2: Pure cosine similarity >93%
    • If duplicate found: abort background task, don't store
   ↓
[13] Attribute-Based Contradiction Pre-Check (Rule-Based)
    • Detect location contradictions ("I live in SF" vs "I live in NYC")
    • Detect age contradictions ("I'm 25" vs "I'm 30")
    • Detect employer contradictions
    • Rule-based (doesn't rely on LLM)
   ↓
[14] Ask LLM: Should Remember? + Classify Relationships
    • LLM judges if message is memory-worthy
    • Filters conversational filler ("ok", "thanks", "lol")
    • Generates memory title
    • Classifies relationships with similar memories:
      - "contradicts" (conflicting information)
      - "supports" (reinforcing information)
      - "elaborates" (adds new details)
      - "none" (unrelated)
   ↓
[15] Contradiction Detection → User Resolution Required
    • If contradiction detected (rule-based OR LLM):
      - Emit "contradiction-detected" event to frontend
      - Show ConflictResolutionModal with both memories
      - User chooses: keep old, keep new, not a contradiction, accept contradiction, or resolve with explanation (stores explanation memory with `resolves` edges to both)
      - ABORT background task, wait for user decision via resolve_memory_conflict_v2
    • If no contradiction: proceed to storage
   ↓
[16] NLP Enhancement
    • Entity extraction: BERT NER (persons, orgs, locations, products)
    • Event detection: Extract event type and date
    • Namespace generation: Categorize as personal, work, travel, etc.
   ↓
[17] Memory Storage
    • Store in SQLite memories table:
      - content (factual statement)
      - title (LLM-generated)
      - embedding (384-dim vector, stored as BLOB)
      - entities_detected (JSON TEXT array)
      - namespace (auto-generated or default "personal")
      - event_type, event_date (if detected)
      - is_ephemeral=true if HIPAA mode (8-hour expiration)
   ↓
[18] Relationship Storage
    • Create memory_links table entries:
      - from_memory_id (newly created memory)
      - to_memory_id (existing memory)
      - relationship_type ("contradicts", "supports", "expands")
      - reason (LLM explanation)
      - detection_method ("llm" or "rule-based")
   ↓
[19] Update Timestamps for ZynkSync
    • Update updated_at on all involved memories
    • Triggers ZynkSync to detect changes and sync to paired devices
   ↓
[20] Background Task Complete
    • Close database connection
    • Task terminates silently
```

**Code Location:** `lib.rs` (Phase 3 — Tokio spawn async block)

---

**Key Implementation Details**:

- **Non-Blocking**: Phase 3 runs in `tokio::spawn()` - never blocks UI
- **Duplicate Prevention**: 98% hybrid or 93% cosine similarity threshold prevents near-duplicate storage
- **Two-Tier Contradiction Detection**: Rule-based (fast, deterministic) + LLM-based (semantic, flexible)
- **User Control**: Contradictions pause background processing and require user resolution
- **HIPAA Ephemeral**: 8-hour auto-expiration enabled automatically in HIPAA mode
- **Stop Word Filtering**: 100+ common words removed from entity list to improve search quality
- **Relationship Reuse**: Same hybrid search results used for both conversation context AND relationship detection (efficient)
- **Fallback Logic**: If API LLM fails for memory classification, falls back to local model
- **Smart Namespace Filtering**: Queries about "Zynkbot" search only system memories, not user memories

---

### 2. Memory System

**Location**: `src-tauri/src/memory.rs`

**Core Operations**:

#### Hybrid Search Algorithm

```rust
// Simplified illustration — see memory.rs::hybrid_search() for full implementation
pub async fn hybrid_search(
    pool: &SqlitePool,
    query_embedding: Vec<f32>,
    query_entities: Vec<String>,
    user_id: &str,
    limit: usize,
) -> Result<Vec<Memory>, String> {
    // Step 1: Fetch all candidates that have embedding or entity data from SQLite
    let candidates = sqlx::query!(
        "SELECT id, embedding, entities_detected FROM memories \
         WHERE embedding IS NOT NULL OR (entities_detected IS NOT NULL AND entities_detected != '[]')"
    )
    .fetch_all(pool)
    .await?;

    // Step 2: Score each candidate in Rust (no SQL vector ops)
    let mut scored: Vec<(f64, Uuid)> = candidates.iter()
        .filter_map(|r| {
            let semantic = r.embedding.as_ref()
                .map(|b| cosine_similarity(&query_embedding, &blob_to_f32(b)) as f64)
                .unwrap_or(0.0);
            let entity = entity_overlap_score(&r.entities_detected, &query_entities);
            let score = if !query_entities.is_empty() {
                entity * 0.4 + semantic * 0.6
            } else {
                semantic
            };
            (score >= MIN_SIMILARITY).then(|| (score, r.id))
        })
        .collect();

    // Step 3: Sort by score, fetch full Memory rows for top N
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // ... fetch and return top `limit` memories
}
```

**Features**:
- **Entity-Based**: BERT NER extracts entities, scored in Rust against stored JSON
- **Semantic**: 384-dim cosine similarity computed in Rust via Candle — no SQL vector ops
- **Weighted Scoring**: 60% semantic + 40% entity matching
- **Namespace Filtering**: Personal/work/family/onboarding/system separation

---

### 3. Ensemble Mode (Multi-Model Consensus)

**Location**: `src-tauri/src/commands/chat.rs::run_ensemble()`

**Purpose**: Run multiple AI models simultaneously and synthesize their responses

**3-Phase Process**:

1. **Phase 0: Assessment** - Coordinator determines if question needs web search
2. **Phase 1: Collection** - Each selected model answers independently with memory context
3. **Phase 2: Synthesis** - Coordinator evaluates responses, identifies consensus/uncertainty

**Implementation**:
```rust
pub async fn run_ensemble(
    message: String,
    models: Vec<String>,  // e.g., ["claude", "gpt4", "local"]
    user_id: String,
    session_id: String,
    containment_mode: String
) -> Result<serde_json::Value, String> {
    // Phase 0: Assess if question needs web search
    let needs_search = assess_search_need(&message, &models[0]).await?;
    let search_results = if needs_search {
        search_duckduckgo(&message, 3).await.ok()
    } else {
        None
    };

    // Phase 1: Collect responses from all models (parallel)
    let mut individual_responses = Vec::new();
    for model in &models {
        let response = query_model(model, &message, &context, &search_results).await?;
        individual_responses.push(response);
    }

    // Phase 2: Synthesize best answer
    let coordinator_model = select_coordinator(&models);
    let synthesized = synthesize_responses(
        coordinator_model,
        &message,
        &individual_responses,
        &search_results
    ).await?;

    Ok(json!({
        "individual_responses": individual_responses,
        "synthesized_response": synthesized,
        "coordinator_model": coordinator_model,
        "search_results": search_results
    }))
}
```

**Features**:
- Automatic web search detection (e.g., "latest React version")
- Memory context provided to all models
- DuckDuckGo search integration with 5-second timeout
- Coordinator evaluates (not averages) responses

---

### 4. ZynkSync (Device-to-Device Memory Sync)

**Location**: `src-tauri/src/zynksync.rs`

**Purpose**: Sync memories across user's devices over local network

**Architecture**:
```
Device A (192.168.1.100:57963)
   ↓ HTTP POST /api/zynksync/push-memories
   ↓ {user_id, memories: [...], relationships: [...]}
   ↓
Device B (192.168.1.101:57963)
   → Receives memories
   → Checks for conflicts (last-write-wins by timestamp)
   → Inserts/updates local database
   → Returns {memories_received, conflicts_resolved}
```

**Key Features**:
- **Port 57963** for HTTP communication
- **6-digit pairing codes** (10-minute expiry)
- **Bidirectional pairing** - both devices add each other automatically
- **Selective sync** - only `is_syncable=true` memories
- **Namespace support** - sync specific folders (personal/work)
- **Complete data sync** - embeddings, entities, relationships all synced

---

### 5. ZynkLink (Device-to-Device File Sharing)

**Location**: `src-tauri/src/zynklink.rs`

**Purpose**: Share files between paired devices

**Features**:
- Share local directories with read/write permissions
- Browse files from paired devices
- Download files from remote devices
- Download files directly into RAG vector database (knowledge base)
- SHA256 integrity verification
- HTTP-based file serving

---

### 6. ZChat (Device-to-Device Messaging)

**Location**: `src-tauri/src/zchat.rs`

**Purpose**: Direct messaging between paired devices

**Features**:
- UUID-based message IDs
- Read receipts (delivered_at, read_at)
- Local SQLite storage
- Real-time delivery when devices online

---

### 7. Knowledge Base (RAG)

**Location**: `src-tauri/src/knowledge_base.rs` (indexing & document management), `src-tauri/src/kb_rag.rs` (RAG retrieval)

**Purpose**: Document-based retrieval-augmented generation

**Process**:
1. User uploads TXT/MD file
2. Document chunked into 512-token segments
3. Each chunk embedded with all-MiniLM-L6-v2
4. Stored in `kb_chunks` table with vector index
5. Query time: semantic search retrieves relevant chunks
6. Chunks added to LLM context

**Tables**:
- `kb_documents` - Document metadata
- `kb_chunks` - Text chunks with embeddings

---

### 8. Safety Layer

**Location**: `src-tauri/src/safety_classifier.rs`

**Purpose**: Content filtering using toxic-bert model

**Implementation**:
```rust
pub fn classify_toxicity(text: &str) -> Result<SafetyScore, String> {
    let model = load_toxic_bert_model()?;
    let tokenizer = load_tokenizer()?;

    // Tokenize input
    let tokens = tokenizer.encode(text, true)?;
    let input_ids = tokens.get_ids();

    // Run inference
    let logits = model.forward(&Tensor::new(input_ids, &Device::Cpu)?)?;
    let probabilities = softmax(&logits)?;

    // 6 toxicity categories
    let categories = ["toxic", "severe_toxic", "obscene",
                      "threat", "insult", "identity_hate"];

    Ok(SafetyScore {
        categories: categories.into_iter()
            .zip(probabilities.iter())
            .collect(),
        is_safe: probabilities.iter().all(|&p| p < 0.5)
    })
}
```

---

## Data Flow

### Complete User Interaction Example

**Scenario**: User asks "What's my favorite color?"

```
┌─────────────────────────────────────────────────────────┐
│ [1] USER INPUT (React frontend)                        │
│     Chat.jsx → invoke("send_message_with_memory")      │
└────────────────┬────────────────────────────────────────┘
                 ↓ Tauri IPC
┌─────────────────────────────────────────────────────────┐
│ [2] RUST BACKEND (lib.rs)                              │
│                                                          │
│  Safety Check:                                          │
│    → toxic-bert classifier                             │
│    → Result: Pass (not toxic)                          │
│                                                          │
│  Parallel Processing:                                   │
│    → Thread 1: BERT NER entity extraction              │
│    → Thread 2: all-MiniLM-L6-v2 embedding generation   │
│                                                          │
│  Hybrid Search:                                         │
│    → Entity match: ["favorite", "color"]               │
│    → Semantic search: embedding similarity             │
│    → Result: "User's favorite color is blue"           │
│                                                          │
│  LLM Inference:                                         │
│    → Build prompt with recalled memories               │
│    → Query selected model (local or API)               │
│    → Response: "Your favorite color is blue!"          │
│                                                          │
│  Store (if memory-worthy):                             │
│    → Question not stored (query, not assertion)        │
│                                                          │
│  Return Response                                        │
└────────────────┬────────────────────────────────────────┘
                 ↓ Tauri IPC response
┌─────────────────────────────────────────────────────────┐
│ [3] FRONTEND DISPLAY                                    │
│     Chat.jsx receives response                          │
│     → Appends to conversation                           │
│     → Renders: "Your favorite color is blue!"           │
└─────────────────────────────────────────────────────────┘

```

---

## Security & Privacy

### Privacy-First Architecture

1. **Local-First ML**: Embeddings, NER, and safety classification run on-device
2. **Optional API**: User must explicitly provide API keys
3. **No Telemetry**: Zero analytics, crash reports, or usage tracking
4. **Local Storage**: All data in local SQLite database (single file on disk)
5. **Device-to-Device**: ZynkSync/ZynkLink/ZChat never touch cloud

### HIPAA Mode

**Activation**: User selects specific containment mode

**Behaviors**:
- All memories stored with `expires_at` set to 8 hours from creation (ISO timestamp)
- Background job purges expired memories
- Stricter safety threshold
- No cloud sync (device-local only)

---

## Deployment

### Installation Scripts

**Windows:**
```bash
install.bat              # Automated installation (Run as Administrator)
START_ZYNKBOT.bat   # Launch application
```

**Linux:**
```bash
./install.sh             # Automated installation
./START_ZYNKBOT.sh       # Launch application
```

Both scripts handle:
- Dependency installation (Rust, Node.js, and build tools)
- SQLite database creation (embedded — no separate server process)
- Schema initialization
- System model downloads (embeddings, safety, NER)
- Optional user LLM downloads

---

## Development Guide

### Prerequisites

- **Rust**: 1.77.2+
- **Node.js**: 18+
- **SQLite**: Embedded — no separate install required (created automatically by the app)

### Build Instructions

**Development mode:**
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

**SQLx cache generation:**
```bash
cd zynkbot_rust/src-tauri
DATABASE_URL="sqlite:zynkbot.db" cargo sqlx prepare
```

### Project Structure

```
zynkbot/                       # Repository root
├── zynkbot_rust/              # Main Tauri desktop application
│   ├── src/                   # React frontend
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Chat.jsx
│   │   │   ├── MemoryManager.jsx
│   │   │   ├── Settings.jsx
│   │   │   ├── KnowledgeBase.jsx
│   │   │   ├── Onboarding.jsx
│   │   │   ├── EnsembleModal.jsx
│   │   │   ├── ZynkSyncPanel.jsx
│   │   │   ├── ZynkLinkPanel.jsx
│   │   │   ├── ZChatModal.jsx
│   │   │   └── ConflictResolutionModal.jsx
│   │   └── main.jsx
│   ├── src-tauri/             # Rust backend (Tauri)
│   │   ├── src/
│   │   │   ├── lib.rs         # Tauri entry point: module declarations + invoke_handler
│   │   │   ├── commands/      # Tauri IPC command handlers (modular)
│   │   │   │   ├── memory.rs
│   │   │   │   ├── onboarding.rs
│   │   │   │   ├── conversation.rs
│   │   │   │   ├── nlp.rs
│   │   │   │   ├── models.rs
│   │   │   │   └── safety.rs
│   │   │   ├── memory.rs      # Hybrid search, CRUD operations
│   │   │   ├── knowledge_base.rs  # KB indexing & document management
│   │   │   ├── kb_rag.rs          # Knowledge Base RAG retrieval
│   │   │   ├── containment.rs # Safety/containment modes
│   │   │   ├── safety_classifier.rs  # TinyBERT toxic classification
│   │   │   ├── nlp_enhancer.rs       # BERT NER, entity extraction
│   │   │   ├── conversation_engine.rs # Prompt building
│   │   │   ├── relationship_detector.rs # Memory relationships
│   │   │   ├── zynksync.rs    # Cross-device memory sync
│   │   │   ├── zynklink.rs    # Device-to-device file sharing
│   │   │   ├── zchat.rs       # Device messaging
│   │   │   ├── web_search.rs  # DuckDuckGo integration
│   │   │   └── llm/
│   │   │       ├── local_embeddings.rs  # all-MiniLM-L6-v2
│   │   │       ├── local_models.rs      # llama.cpp GGUF
│   │   │       ├── anthropic.rs         # Claude API
│   │   │       ├── openai.rs            # GPT API
│   │   │       └── xai.rs               # Grok API
│   │   ├── models/
│   │   │   ├── system/            # Auto-downloaded system models
│   │   │   │   ├── all-MiniLM-L6-v2/   # 80MB embeddings
│   │   │   │   ├── toxic-bert/         # 260MB safety classifier
│   │   │   │   └── bert-base-NER/      # 260MB entity extraction
│   │   │   └── user/              # Optional local LLMs (.gguf)
│   │   │       ├── DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf   # 4.9GB (optional)
│   │   │       ├── Qwen3-8B-Q4_K_M.gguf                       # 4.7GB (optional)
│   │   │       └── Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf # 4.7GB (optional)
│   │   ├── Cargo.toml
│   │   ├── tauri.conf.json
│   │   └── .env
│   ├── package.json
│   └── vite.config.js
├── docs/                      # Documentation
│   ├── FEATURES.md
│   ├── NETWORKING_FEATURES.md
│   ├── DIGITAL_RESILIENCE.md
│   ├── PROJECT_VISION.md
│   ├── case_studies/
│   │   ├── conversational_memory.md
│   │   ├── hipaa_compliance.md
│   │   ├── emergency_resilience.md
│   │   └── hvac_field_service.md
│   ├── architecture_and_development/
│   │   └── ARCHITECTURE_COMPREHENSIVE.md
│   └── troubleshooting/
├── scripts/                   # Helper scripts
│   ├── db/                    # Database setup
│   │   ├── complete_fresh_install_schema.sql
│   │   └── database_schema.sql
│   └── tools/                 # Development tools
├── knowledge_base/            # Example KB files (for testing)
├── labs/                      # Experimental features
├── README.md
├── LICENSE                    # AGPL v3
├── COMMERCIAL_LICENSE.md
├── ROADMAP.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── install.bat                # Windows installer
└── install.sh                 # Linux/macOS installer
```

**Key Directories:**

- **`zynkbot_rust/`** - Main Tauri desktop application (React + Rust)
- **`docs/`** - Comprehensive documentation (features, architecture, case studies)
- **`scripts/`** - Database schemas, development tools, diagnostic scripts
- **`knowledge_base/`** - Example documents for Knowledge Base testing
- **`labs/`** - Experimental features not yet in production

---

---

## References

### Related Documentation

- [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) - Database structure
- [LINUX_INSTALLATION.md](LINUX_INSTALLATION.md) - Linux setup guide
- [TAURI_INSTALLATION.md](TAURI_INSTALLATION.md) - Cross-platform installation
- [MODELS.md](MODELS.md) - ML model information
- [ZynkSync Documentation](ZynkSync_Documentation.md) - Device sync
- [ZynkLink Documentation](ZynkLink_Filesystem_Documentation.md) - File sharing
- [ZChat Documentation](ZChat_Documentation.md) - Device messaging

### External Resources

- [Tauri Documentation](https://tauri.app/)
- [Candle ML Framework](https://github.com/huggingface/candle)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Rust Book](https://doc.rust-lang.org/book/)

---

**License**: AGPL-3.0 (open source) / Commercial License available — see [COMMERCIAL_LICENSE.md](../../COMMERCIAL_LICENSE.md)
**Contact**: [GitHub Issues](https://github.com/MSkill1/zynkbot/issues)
**Version**: 1.0.0 (Production Release)

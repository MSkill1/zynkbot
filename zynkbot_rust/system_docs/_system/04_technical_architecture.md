# Zynkbot Technical Architecture

## System Overview

Zynkbot is a Tauri desktop application with a React frontend and a Rust backend. The frontend and backend communicate exclusively via Tauri's IPC layer (typed `invoke` calls). All data is stored locally in a SQLite database file on the user's device.

```
┌─────────────────────────────────────────┐
│          React Frontend                 │
│  (UI Components, State Management)      │
└──────────────┬──────────────────────────┘
               │ IPC (invoke / emit)
┌──────────────▼──────────────────────────┐
│          Tauri Runtime                  │
│  (Commands, Events, Window Management)  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        Rust Backend                     │
│  ┌─────────────────────────────────┐    │
│  │  Memory Pipeline                │    │
│  │  - Hybrid search                │    │
│  │  - LLM memory decision          │    │
│  │  - Contradiction detection      │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │  NLP Engine (nlp_enhancer.rs)   │    │
│  │  - BERT NER entity extraction   │    │
│  │  - Event detection              │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │  Embeddings (local_embeddings)  │    │
│  │  - all-MiniLM-L6-v2 (candle)   │    │
│  │  - 384-dim vectors, on-device   │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │  LLM Interface                  │    │
│  │  - Local GGUF via llama.cpp     │    │
│  │  - API clients (Claude, GPT,    │    │
│  │    Grok)                        │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │  KB RAG (kb_rag.rs)             │    │
│  │  - Chunking + embedding         │    │
│  │  - in-process cosine search     │    │
│  │  - Filename-aware boosting      │    │
│  └─────────────────────────────────┘    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│       SQLite (local file, embedded)     │
│  - Memory storage + vector search       │
│  - KB document chunks                   │
│  - Conversation history                 │
│  - ZynkSync / ZynkLink state            │
└─────────────────────────────────────────┘
```

## Technology Stack

### Frontend
- **React 18** with JSX
- **Tauri API v2** for IPC
- **react-force-graph-2d** for the Relationship Graph visualization

### Backend (Rust)
- **Tauri 2** — desktop app framework and IPC layer
- **tokio** — async runtime
- **sqlx** — SQLite async driver with compile-time query validation
- **candle** — pure-Rust ML framework (embeddings, BERT NER, safety classifier)
- **llama-cpp-2** — local GGUF model inference
- **reqwest** — HTTP client for API backends and ZynkLink/ZynkSync
- **serde/serde_json** — serialization

### Database
- **SQLite** (embedded, no server process) via sqlx
- Vector similarity search computed in-process in Rust (cosine similarity via Candle)
- Standard B-tree indexes; JSON stored as TEXT

### ML Models (all run locally)
- **all-MiniLM-L6-v2** — sentence embeddings (384-dim)
- **dslim/bert-base-NER** — named entity recognition
- **toxic-bert (TinyBERT)** — content safety classification
- **Whisper** — speech-to-text (temporarily disabled, see Features doc)

## Source File Structure

```
src-tauri/src/
├── lib.rs                  # All Tauri commands; main application logic
├── main.rs                 # Tauri app entry point
├── db.rs                   # Database connection pool
├── memory.rs               # Memory CRUD and hybrid search
├── kb_rag.rs               # Knowledge Base RAG pipeline
├── knowledge_base.rs       # KB file scanning and reading
├── conversation_engine.rs  # Memory worthiness heuristic (is_memory_worthy)
├── conversation_history.rs # Conversation session storage and retrieval
├── nlp_enhancer.rs         # Entity extraction, event detection, namespacing
├── llm_fact_extractor.rs   # LLM-based fact extraction from messages
├── question_extractor.rs   # Extracts questions from conversation for context
├── containment.rs          # Safety enforcement (all containment modes)
├── safety_classifier.rs    # TinyBERT toxicity classifier
├── relationship_detector.rs # DEPRECATED — replaced by LLM classifier in lib.rs
├── user_identity.rs        # User ID and device identity management
├── sync_codes.rs           # ZynkSync pairing code generation/validation
├── zynksync.rs             # Memory sync protocol
├── zynklink.rs             # File sharing between paired devices
├── zchat.rs                # Device-to-device messaging
├── web_search.rs           # DuckDuckGo web search integration
├── whisper.rs              # Voice transcription (temporarily disabled)
└── llm/
    ├── mod.rs              # LLM routing and API client logic
    └── local_embeddings.rs # Candle-based embedding generation
```

## Message Processing Flow

1. User sends message → React calls `invoke('send_message_with_memory', {...})`
2. Rust backend:
   - Runs containment check (if mode is active)
   - Generates embedding for the message
   - Detects if query is about Zynkbot itself (keyword check)
   - Performs hybrid search for relevant memories
   - Constructs prompt with memory context
3. Message sent to selected model (local GGUF or API)
4. Response returned to frontend immediately
5. Background pipeline (async, non-blocking):
   - Heuristic gate (is_memory_worthy)
   - Duplicate check (cosine similarity)
   - LLM memory decision + relationship classification
   - Contradiction check → modal if needed
   - NLP enhancement (entities, events, namespace)
   - Memory storage + relationship links

## Database Schema (Key Tables)

- **memories** — all stored memories with embeddings, entities, namespace, relationships
- **memory_links** — semantic relationships between memory pairs
- **kb_documents** — indexed Knowledge Base documents
- **kb_chunks** — document chunks with embeddings for RAG search
- **conversation_sessions** — conversation session metadata
- **conversation_messages** — full message log per session
- **message_feedback** — thumbs up/down ratings on responses
- **zynk_devices** — registered devices for sync and linking
- **zynk_device_pairings** — ZynkSync device pairs
- **zynklink_pairings** — ZynkLink cross-user file sharing relationships
- **zchat_messages** — ZChat message history

## Performance Characteristics

| Operation | Notes |
|---|---|
| Hybrid memory search | Fast; scales with database size and hardware |
| Embedding generation | Runs on CPU via Candle |
| Entity extraction (BERT NER) | Runs on CPU via Candle |
| Relationship detection | Runs in background — does not block UI |
| Local model inference | Hardware dependent (CPU/GPU, model size) |
| API model inference | Network dependent |
| KB semantic search | In-process cosine similarity |

## Security Model

- No cloud by default — everything runs locally
- API keys stored in `.env` file, never transmitted except to the respective provider
- ZynkSync uses certificate-based mutual authentication
- ZynkLink uses the same device certificate infrastructure
- All database queries use sqlx prepared statements (SQL injection protection)
- Tauri IPC uses an explicit allowlist of registered commands

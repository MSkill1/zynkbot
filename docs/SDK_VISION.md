# Zynkbot SDK - Developer Platform

**Status:** Planned for v3.0 (2027)
**Purpose:** Enable third-party developers to build privacy-first AI applications using Zynkbot's core components

---

## Overview

The Zynkbot SDK extracts modular, reusable components from Zynkbot and makes them available as a developer platform. Instead of building AI privacy features from scratch, developers can use battle-tested, well-proven modules.

**Vision:** Establish a privacy-first standard for ethical AI development — the infrastructure layer that makes responsible AI the default choice for developers.

---

## Why a SDK?

### The Problem
Building privacy-first AI is hard:
- ❌ Most developers don't have ML expertise
- ❌ HIPAA compliance requires specialized knowledge
- ❌ Safety filtering is complex and evolving
- ❌ Cross-device sync with privacy is non-trivial
- ❌ Existing solutions (OpenAI, Google) are cloud-first

### The Solution
Zynkbot SDK provides:
- ✅ Battle-tested privacy components
- ✅ Pure Rust implementation (fast, safe, portable)
- ✅ Publicly available, auditable source
- ✅ Commercial licensing available
- ✅ Well-documented, easy to integrate

---

## SDK Modules

Zynkbot's 7,000-line monolithic backend was broken into 22 self-contained modules during the pre-open-source refactor. Each is a candidate for independent SDK extraction. They fall into four dependency layers — core must ship first; feature and network layers depend on it.

### Dependency Layers

| Layer | Modules |
|-------|---------|
| **Core** | Database, TLS/mTLS, User Identity, Sync Codes |
| **AI** | LLM Backends, Local Model Inference, Local Embeddings, Memory System, NLP/Entity Extraction, LLM Fact Extractor |
| **Feature** | Conversation Engine, Conversation History, Containment Layer, Safety Classifier, Knowledge Base/RAG, Web Search, Ensemble Mode, Backup |
| **Network** | ZynkSync, ZynkLink, ZChat |
| **Voice** | Voice/Dictation (pending modularization from App.jsx) |

---

### Core Layer

**Database** (`db.rs`) — SQLite connection pool with WAL mode, foreign keys, 20-connection limit, platform-aware paths (Android-aware). Foundation everything else depends on.

**TLS/mTLS** (`tls.rs`) — Self-signed cert generation, pinned cert verification, optional client auth. Required by all three network modules.

**User Identity** (`user_identity.rs`) — Persistent user/device ID generation and storage, hostname-based device naming. Required for pairing and sync.

**Sync Codes** (`sync_codes.rs`) — One-time 6-digit device pairing codes with 5-minute expiry and single-use enforcement.

---

### AI Layer

**LLM Backends** (`llm/`) — Unified streaming interface for Claude (Anthropic), GPT (OpenAI), Grok (xAI), Ollama/custom endpoints, and local GGUF models. Common `Message`, `LLMResponse`, `Usage` types across all providers.

**Local Model Inference** (`llm/local_models.rs`) — llama-cpp-2 GGUF inference with per-family prompt formatting (ChatML, Llama 3, Mistral, Phi), grammar-constrained JSON output for structured extraction. No Python, no GPU required.

**Local Embeddings** (`llm/local_embeddings.rs`) — Candle all-MiniLM-L6-v2, batched embedding generation, L2-normalized 384-dim output for cosine similarity. Runs fully on-device.

**Memory System** (`memory.rs`) — Hybrid vector + entity search, persistent SQLite storage, relationship graph (`MemoryLink`, `MemoryRelationship`), contradiction detection. The core of Zynkbot's long-term context.

**NLP / Entity Extraction** (`nlp_enhancer.rs`, `question_extractor.rs`) — BERT NER with rule-based fallback, tag/namespace/event detection, fact extraction from natural language queries (possession, family, events, travel, occupation).

**LLM Fact Extractor** (`llm_fact_extractor.rs`) — LLM-delegated fact extraction from Q&A pairs via Claude/GPT-4o/Grok. Currently experimental; designed as a higher-accuracy alternative to pattern-based extraction.

---

### Feature Layer

**Conversation Engine** (`conversation_engine.rs`) — Prompt assembly with memory injection, per-model adaptive context limits (API vs. local GGUF), memory-worthiness filtering.

**Conversation History** (`conversation_history.rs`) — Hash-chain session logging, full-text search across sessions, HIPAA-aware skip logic for ephemeral modes.

**Containment Layer** (`containment.rs`) — 6-mode content safety enforcement: Guardian, Child, HIPAA, Elder, Sovereign, Witness. Gates user input via pattern matching + on-device TinyBERT classifier. Child mode optionally delegates to OpenAI Moderation API.

**Safety Classifier** (`safety_classifier.rs`) — Pure-Rust Candle TinyBERT toxicity classification with configurable thresholds per containment mode. No external API calls required.

**Knowledge Base / RAG** (`knowledge_base.rs`, `kb_rag.rs`) — Document indexing with chunking, hybrid semantic + FTS5 retrieval, filename boosting. Supports PDF, text, and code. Embedding generation via local embeddings module.

**Web Search** (`web_search.rs`) — DuckDuckGo Lite wrapper with page content fetching and HTML text extraction. Privacy-respecting: no API key, no tracking.

**Ensemble Mode** (`commands/chat.rs`) — Multi-model parallel query with synthesis coordination. Coordinator model auto-selected by capability tier; results synthesized into a single response.

**Backup** (`commands/backup.rs`) — Encrypted R2 cloud backup with tombstone-safe restore. Memories + conversation history. Propagates deletes to peers on restore.

---

### Network Layer

**ZynkSync** (`zynksync.rs`) — Cross-device memory sync over local network. mTLS device authentication, TCP sync loop, last-write-wins conflict resolution, namespace filtering. One Tauri dependency (`tauri::Emitter` for UI events — trivial to abstract for standalone use).

**ZynkLink** (`zynklink.rs`) — P2P file sharing between paired devices over mTLS. Directory sharing with read/write access controls, file listing and content serving.

**ZChat** (`zchat.rs`) — Device-to-device messaging with delivery and read tracking. No cloud storage, no relay server. Works on local network only.

---

### Voice Layer

**Voice / Dictation** (pending modularization) — Offline Vosk ASR with OpenAI Whisper fallback, wake-word detection (ONNX), TTS playback, silence-detection auto-send, conversation loop. Currently split between `vosk_desktop.rs` (Rust) and App.jsx (JS). Extraction to a dedicated module is a v1.0 pre-release task.

---

### 1. Containment Layer

**What it does:**
- Consent-based safety framework
- Multiple safety modes (Guardian, Child, HIPAA, etc.)
- Content filtering with user control
- Audit logging for compliance

**Use cases:**
- Child-safe educational apps
- Healthcare applications
- Enterprise AI assistants
- Content moderation systems

**API Example:**
```rust
use zynkbot_sdk::containment::{ContainmentLayer, Mode};

let containment = ContainmentLayer::new(Mode::Guardian)?;

// Check if content is safe
let result = containment.check_content("User message here").await?;
if result.is_blocked {
    println!("Content blocked: {}", result.reason);
} else {
    // Proceed with AI processing
}
```

**License:**
- Free for uses permitted by the community source license
- Paid commercial license required for commercial products and services

---

### 2. Memory System

**What it does:**
- Hybrid semantic + entity search
- Local SQLite storage with in-process vector search (Candle)
- Contradiction and duplicate detection
- Transparent, editable memory

**Use cases:**
- Customer support bots with context
- Personal knowledge management apps
- Research assistants
- Conversational AI with long-term memory

**API Example:**
```rust
use zynkbot_sdk::memory::{MemorySystem, SearchQuery};

let memory = MemorySystem::new(database_url)?;

// Store a memory
memory.store("User prefers dark mode",
    namespace: "preferences",
    is_syncable: true
).await?;

// Hybrid search (entity + semantic)
let results = memory.search(
    SearchQuery::new("What theme does user like?")
).await?;

for result in results {
    println!("Memory: {} (similarity: {})",
        result.content,
        result.score
    );
}
```

**Features:**
- Entity extraction (BERT NER)
- Semantic similarity (embeddings)
- Namespace isolation
- Conflict detection

---

### 3. ZynkSync Protocol

**What it does:**
- Cross-device synchronization
- Local network discovery (mDNS)
- Pairing with 6-digit codes

**API Example:**
```rust
use zynkbot_sdk::zynksync::{SyncService, PairingCode};

let sync = SyncService::new(local_port: 57963)?;

// Start discovery
sync.start_discovery().await?;

// Pair with another device
let code = PairingCode::generate(); // e.g., "123456"
sync.pair_with_code(code, remote_ip).await?;

// Sync memories
sync.sync_namespace("work").await?;
```

**Security:**
- Local network only (no internet exposure)
- Pairing code authentication
- TLS support (planned)
- End-to-end encryption (planned)

---

### 4. HIPAA Framework

**What it does:**
- PHI (Protected Health Information) detection
- Ephemeral memory (auto-expiring data)
- Audit logging
- Medical content disclaimers

**Use cases:**
- Telemedicine applications
- Medical note-taking
- Patient portals
- Healthcare chatbots

**API Example:**
```rust
use zynkbot_sdk::hipaa::{HIPAAMode, PHIDetector};

let hipaa = HIPAAMode::new(
    ephemeral: true,
    audit_logging: true
)?;

// Check for PHI before processing
let check = hipaa.scan_for_phi("Patient John Doe, DOB 1980-05-15")?;
if check.contains_phi {
    println!("PHI detected: {:?}", check.phi_types);
    // Redact or block
} else {
    // Safe to process
}

// Store with auto-expiration
hipaa.store_ephemeral(
    content: "Consultation notes",
    expiry: Duration::from_hours(24)
).await?;
```

**Compliance Notes:**
- ⚠️ Framework only, not certified HIPAA-compliant (see HIPAA case study for compliance-ready architecture guidance)
- ⚠️ Organizations must obtain their own certification
- ⚠️ SDK provides tools, not legal guarantees
- ✅ Audit logs for compliance documentation

---

### 5. Snap-in Architecture

**What it does:**
- Domain-specific workspace framework
- Isolated context and memory
- Custom UI components
- Integration with main app

**Use cases:**
- Professional tools (legal, medical, financial)
- Educational modules
- Specialized assistants
- Plugin marketplace

**API Example:**
```rust
use zynkbot_sdk::snapin::{SnapIn, Workspace};

#[derive(SnapIn)]
struct TherapistJournal {
    workspace: Workspace,
    containment: ContainmentLayer,
}

impl TherapistJournal {
    fn new() -> Result<Self> {
        Ok(Self {
            workspace: Workspace::new("therapist-journal")?,
            containment: ContainmentLayer::new(Mode::HIPAA)?,
        })
    }

    async fn save_session(&self, notes: &str) -> Result<()> {
        // Ephemeral storage, HIPAA-friendly
        self.workspace.store_ephemeral(
            content: notes,
            expiry: Duration::from_days(7)
        ).await
    }
}
```

**Features:**
- Isolated namespaces
- Custom safety modes
- UI component library
- Plugin manifest system

---

## Integration Examples

### Example 1: Child-Safe Educational App

```rust
use zynkbot_sdk::{containment, memory, llm};

struct EducationalAssistant {
    containment: ContainmentLayer,
    memory: MemorySystem,
    llm: LLMBackend,
}

impl EducationalAssistant {
    async fn new() -> Result<Self> {
        Ok(Self {
            // Strict filtering for children
            containment: ContainmentLayer::new(Mode::Child)?,
            // Track learning progress
            memory: MemorySystem::new(db_url)?,
            // Local-first inference
            llm: LLMBackend::local("educational-model.gguf")?,
        })
    }

    async fn answer_question(&self, question: &str) -> Result<String> {
        // Check content safety first
        let safety = self.containment.check_content(question).await?;
        if safety.is_blocked {
            return Ok("I can't answer that. Ask your teacher!".to_string());
        }

        // Retrieve relevant learning context
        let context = self.memory.search(
            SearchQuery::new(question)
                .namespace("learning-progress")
        ).await?;

        // Generate answer with context
        let answer = self.llm.generate(question, context).await?;

        // Store interaction for progress tracking
        self.memory.store(&format!(
            "Student asked: {} | Answered: {}",
            question, answer
        ), namespace: "learning-progress").await?;

        Ok(answer)
    }
}
```

---

### Example 2: Medical Documentation Tool

```rust
use zynkbot_sdk::{hipaa, memory, containment};

struct MedicalNotes {
    hipaa: HIPAAMode,
    memory: MemorySystem,
}

impl MedicalNotes {
    async fn save_consultation(&self, notes: &str) -> Result<()> {
        // Scan for PHI
        let phi_check = self.hipaa.scan_for_phi(notes)?;

        if phi_check.contains_phi {
            log::warn!("PHI detected: {:?}", phi_check.phi_types);
        }

        // Store with ephemeral mode (auto-delete after 30 days)
        self.memory.store_ephemeral(
            content: notes,
            namespace: "consultations",
            expiry: Duration::from_days(30),
            audit_log: true
        ).await?;

        // Generate audit entry
        self.hipaa.log_access(
            action: "save_consultation",
            user: current_user(),
            timestamp: now(),
            phi_detected: phi_check.contains_phi
        ).await?;

        Ok(())
    }
}
```

---

## Licensing Model

### Community Source Tier

**Eligible:**
- Personal projects
- Educational use
- Noncommercial community projects
- Non-profit organizations
- Businesses with <$1M annual revenue

**Terms:**
- Zynkbot Community Source License
- Noncommercial modifications and redistribution permitted under the license
- Internal business use permitted

---

### Commercial Tier

**Eligible:**
- Businesses with >$1M annual revenue
- Proprietary/closed-source applications
- Enterprise deployments
- SaaS products

**Pricing (Estimated):**
- **Starter**: $999/year - Single developer, up to $5M revenue
- **Professional**: $4,999/year - Small team, up to $25M revenue
- **Enterprise**: $24,999/year - Unlimited developers, custom terms

**Includes:**
- Commercial use rights
- No copyleft requirements
- Priority support
- Security updates
- Custom integrations (Enterprise)

**Contact:** matt@containai.ai for licensing

---

## Development Timeline

### Phase 1: Module Extraction (Q1-Q2 2027)
- Extract containment layer to standalone crate
- Extract memory system
- Extract ZynkSync protocol
- Extract HIPAA framework
- Extract Snap-in architecture

### Phase 2: SDK Packaging (Q3 2027)
- Unified SDK package
- Comprehensive documentation
- Code examples and tutorials
- Testing framework
- Developer CLI tools

### Phase 3: Developer Portal (Q4 2027)
- SDK documentation website
- Developer account system
- License management
- Support ticketing
- Community forums

### Phase 4: Marketplace (2028)
- Snap-in discovery platform
- Developer publishing
- Revenue sharing model
- Quality assurance process
- User reviews and ratings

---

## Technical Requirements

### Minimum Requirements
- Rust 1.77.2+
- SQLite (embedded — no separate install required)
- 512 MB RAM minimum
- Linux, Windows, or macOS

### Optional Requirements
- CUDA (for GPU-accelerated ML)
- Docker (for containerized deployment)
- Redis (for caching and queue management)

---

## Support & Community

### Documentation
- SDK Reference: docs.containai.ai/sdk
- Tutorials: docs.containai.ai/tutorials
- Examples: github.com/containai/sdk-examples

### Community
- GitHub Discussions: Community Q&A
- Discord: Real-time chat (developers)
- Stack Overflow: Tag `zynkbot-sdk`

### Commercial Support
- Priority tickets
- Implementation assistance
- Custom feature development
- Security audits

---

## Success Stories (Planned)

Once SDK is released, we'll showcase:
- Healthcare applications built with HIPAA framework
- Educational tools using Child mode
- Enterprise AI assistants
- Privacy-focused consumer apps

**Want to be featured?** Build something awesome and contact us!

---

## Get Early Access

**SDK Beta Program (2027):**
- Early access to modules
- Influence SDK design
- Free commercial license during beta
- Recognition as founding developer

**Sign up:** matt@containai.ai (Subject: SDK Beta)

---

## FAQ

**Q: When will SDK be available?**
A: v3.0 targeted for 2027. Beta program may start earlier.

**Q: Can I use it in a commercial product?**
A: Yes, with a commercial license. Contact matt@containai.ai to discuss terms.

**Q: Is there vendor lock-in?**
A: The source is publicly available, and the community source license permits noncommercial forks and internal modifications. Commercial products and services require a license.

**Q: What languages are supported?**
A: Rust-native, with bindings planned for Python, JavaScript, Go.

**Q: Can I contribute to SDK development?**
A: Yes! See [CONTRIBUTING.md](../CONTRIBUTING.md).

**Q: How is this different from LangChain/LlamaIndex?**
A: Focus on privacy, consent, and safety — not just LLM orchestration.

---

**Project:** Zynkbot SDK
**Vision:** Privacy-first AI development platform
**Website:** https://containai.ai
**GitHub:** https://github.com/MSkill1/zynkbot
**Contact:** matt@containai.ai

**License:** Zynkbot Community Source License + commercial licensing
**Status:** Planning phase, development begins 2027

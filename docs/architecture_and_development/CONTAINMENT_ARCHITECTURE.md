# Containment Architecture: Why Local-First Matters

**Technical overview of Zynkbot's containment system and local-first architecture**

---

## What is Containment?

**Containment** is a pre-processing filter layer that runs **before** user input reaches the LLM. It enforces safety boundaries, content restrictions, and privacy rules based on the active containment mode.

**Key principle:** Block problematic content at the routing layer, not by filtering LLM outputs.

**Why this matters:**
- **Traditional approach:** LLM sees everything, output is filtered → LLM still processed sensitive data
- **Containment approach:** Filter input before LLM → LLM never sees blocked content

This is **architectural safety** vs. **cosmetic filtering**.

---

## Why Local-First Architecture

### The Cloud AI Problem

Most AI assistants (ChatGPT, Claude, Gemini) work like this:

```
User Device                    Cloud Servers
    ↓                               ↓
[Input] ──────────────────→ [LLM Processing]
                                    ↓
[Response] ←──────────────  [Store conversation]
                                    ↓
                            [Train next model version]
```

**Your data:**
- Uploaded to third-party servers
- Used for training (unless you pay for enterprise)
- Subject to terms of service changes
- Accessible to company employees (with restrictions)
- Vulnerable to data breaches
- Requires constant internet connection

### The Local-First Alternative

Zynkbot works like this:

```
User Device (Your Computer)
    ↓
[Containment Layer] ← Safety filter runs locally
    ↓
[Local LLM or API] ← Choice: local inference or API call
    ↓
[SQLite Database] ← All memories stored locally
    ↓
[Response displayed]
```

**Your data:**
- Stays on your device (unless you choose API LLM)
- Never used for training
- You own the database file
- Works offline with local models
- No third-party access

---

## Containment Modes (v0.9 Implementation)

Zynkbot provides 5 containment modes, each enforcing different safety boundaries (fork the repo and build a mode for your use case):

### 1. Guardian Mode (Default)

**Purpose:** Balanced safety for general users

**What it blocks:**
- Toxic content (TinyBERT classifier, local inference)
- Self-harm and violence
- Illegal activities
- Adult content
- Hate speech

**How it works:**
```rust
// lib.rs
let safety_check = check_containment(message.clone(), "guardian").await;
match safety_check {
    Ok(Some(block_message)) => {
        // Content blocked - return block message to user
        return Ok(ReplyResponse {
            reply_text: block_message,
            blocked: Some(true),
            // ...
        });
    }
    Ok(None) => {
        // Content passed - continue to LLM
    }
    Err(e) => {
        // Safety check failed - continue anyway (don't block on errors)
    }
}
```

**Implementation:** `containment.rs` + `safety_classifier.rs` (TinyBERT toxic-bert model, 260MB, local)

**Example:**
- User: "How do I hack into someone's email?"
- Containment: Blocks at routing layer
- LLM: Never sees the question

### 2. Child Mode (Strictest)

**Purpose:** Safe environment for minors

**What it blocks:**
- All Guardian blocks +
- Medical advice
- Legal advice
- Financial advice
- Complex ethical questions
- Any potentially harmful instructions

**How it works:**
```rust
// lib.rs
if containment_mode.to_lowercase() == "child" {
    // Use OpenAI Moderation API for safety check
    let layer = containment::ContainmentLayer::new("child")?;
    match layer.check_openai_moderation(&message).await {
        Ok(Some(block_message)) => {
            // Blocked by OpenAI Moderation
            return Ok(ReplyResponse {
                reply_text: block_message,
                blocked: Some(true),
                // ...
            });
        }
        // ...
    }
}
```

**Implementation:** OpenAI Moderation API (requires internet) + rule-based filters

**Why OpenAI Moderation?** More comprehensive than local TinyBERT for child safety. Trade-off: requires internet, but child safety is worth it.

**Example:**
- User: "Tell me about cryptocurrency investing"
- Containment: Blocks (financial advice for children)
- Blocked message: "I can't provide financial advice. Let's talk about age-appropriate topics!"

### 3. HIPAA Mode (Healthcare)

**Purpose:** Prevent Protected Health Information (PHI) leakage

**What it blocks:**
- PHI patterns (SSN, phone, email, medical IDs via regex - 70-85% accuracy)
- Medical advice that could constitute diagnosis
- Treatment recommendations
- Medication dosing

**Additional features:**
- **Ephemeral memory:** Auto-expires after 8 hours (typical clinical shift)
- **Audit logging:** Daily JSON logs of PHI detections
- **No API LLMs:** Forces local model only (no PHI uploaded)

**Implementation:**
```rust
// containment.rs: phi_detector module
// Regex patterns for:
// - SSN: \b\d{3}-\d{2}-\d{4}\b
// - Phone: \b\d{3}-\d{3}-\d{4}\b
// - Medical Record #: MRN\s*:?\s*\d+
// - Insurance ID, addresses, DOB, etc.
```

**Example:**
- User: "Patient John Doe, SSN 219-90-7812, presented with chest pain"
- Containment: Detects SSN pattern, blocks
- Blocked message: "I detected potential PHI (Social Security Number). HIPAA mode prevents processing this information."

**Limitations:** Currently regex-based (70-85% accurate). Future: AI model for PHI detection (95%+ accuracy). See [hipaa_compliance.md](../case_studies/hipaa_compliance.md).

**HIPAA is one example of domain-specific containment.** The same pattern — custom detection rules, audit logging, ephemeral storage, medical request blocking — can be applied to other regulated industries: legal (attorney-client privilege, client confidentiality), financial services (account numbers, trading strategy), mental health and substance abuse records, and others. HIPAA is the most thoroughly specified compliance framework in US healthcare, making it the natural first implementation. The architecture is designed to be extended.

For the full implementation detail, deployment scenarios, and compliance context, see the [HIPAA Security Integration Guide](HIPAA_SECURITY_INTEGRATION_GUIDE.md).

### 4. Sovereign Mode (Permissive)

**Purpose:** Maximum user control, minimal filtering

**What it blocks:**
- Nothing - all queries allowed

**What it does:**
- Issues warnings for potentially risky content
- Logs the warning and user's decision to proceed
- LLM includes warning prefix in response

**How it works:**
```rust
// lib.rs
if message.starts_with("[WARN_ALLOW]") {
    // Sovereign mode: Extract warning, continue with LLM
    warning_prefix = Some(message.trim_start_matches("[WARN_ALLOW]").trim().to_string());
    // Continue to LLM...
}
```

**Example:**
- User: "How would someone theoretically pick a lock?"
- Containment: Warning issued, but allows query
- LLM response: "⚠️ Warning: This information could be misused. Proceeding because you're in Sovereign Mode. Lock picking theory works by..."

**Use cases:**
- Security research
- Educational purposes
- Users who want full control

### 5. Witness Mode (Development/Testing)

**Purpose:** No containment, only logging

**What it blocks:**
- Nothing

**What it does:**
- Logs all queries and responses
- No safety filtering at all
- For development testing and ethical simulations

**Use cases:**
- Testing LLM responses without safety interference
- Personality simulation research
- Debugging containment logic

---

## Local-First Privacy Benefits

### 1. Data Ownership

**SQLite database file lives on your device:**
```
Windows: %LOCALAPPDATA%\zynkbot\zynkbot.db
Linux:   ~/.local/share/zynkbot/zynkbot.db
```

**What this means:**
- You can back it up to USB drive
- Export to CSV/JSON
- Delete the entire database instantly
- No company controls your data
- No terms of service changes can take away access

### 2. Offline Functionality

**What works without internet:**
- ✅ Local LLM inference (Llama 3.2 3B, Qwen 2.5 7B)
- ✅ Memory search (hybrid entity + semantic)
- ✅ Knowledge Base RAG (indexed documents)
- ✅ Entity extraction (BERT NER, local)
- ✅ Safety classification (TinyBERT, local)
- ✅ Embeddings generation (all-MiniLM-L6-v2, local)
- ✅ ZynkSync device sync (local WiFi/LAN)
- ✅ ZChat messaging (local network)
- ✅ ZynkLink file sharing (local network)

**What requires internet:**
- ⚠️ API LLMs (OpenAI, Anthropic, xAI) - optional
- ⚠️ Voice transcription (Web Speech API) - optional (local transcription on roadmap)
- ⚠️ Web search (DuckDuckGo) - optional

**Why this matters:**
- Works in disaster scenarios (see [emergency_resilience.md](../case_studies/emergency_resilience.md))
- No subscription required for basic functionality
- Privacy-preserving (data never uploaded)

### 3. Transparent Memory

**Memory Manager UI shows exactly what's stored:**
- View all memories
- Edit content directly
- Delete memories permanently 
- See which memories influenced each response
- Filter by namespace (personal, work, etc.)

**Compare to cloud AI:**
- ChatGPT: No visibility into what's stored about you
- Claude: "Memory" feature, but opaque - can't see full data
- Zynkbot: Full transparency - you control every byte

### 4. No Training on Your Data

**Cloud AI providers:**
- May use conversations for training (unless enterprise tier)
- You don't control when this happens
- Can't verify they're not using your data

**Zynkbot:**
- Your conversations NEVER used for training anything
- Local models are frozen (downloaded once, never updated from your data)
- You can verify this (open source, auditable)

### 5. Containment at the Edge

**Why local containment is stronger:**

**Cloud approach:**
```
Your sensitive question → Uploaded to server → LLM processes → Filter output
Problem: Server already saw your sensitive data
```

**Local-first approach:**
```
Your sensitive question → Local containment filter → Blocked immediately
Result: Sensitive data never left your device
```

**Example (HIPAA Mode):**
- Cloud AI: "Patient SSN 219-90-7812" uploaded to server, then filtered → PHI breach already occurred
- Zynkbot: SSN detected locally, blocked before any network transmission → No breach

---

## Comparison: Cloud AI vs. Local-First

| Feature | Cloud AI (ChatGPT, Claude) | Zynkbot (Local-First) |
|---------|----------------------------|------------------------|
| **Data Location** | Third-party servers | Your device (local database) |
| **Training on conversations** | Yes (unless enterprise) | Never |
| **Offline functionality** | ❌ Requires internet | ✅ Full features with local LLM |
| **Memory transparency** | ❌ Opaque | ✅ Full visibility/editing |
| **Containment enforcement** | Output filtering | Pre-LLM input filtering |
| **Data ownership** | Terms of service controlled | You own the database file |
| **Privacy audit** | Trust company claims | Open source, verifiable |
| **Works in emergency** | ❌ No internet = no AI | ✅ Local LLM continues working |
| **Cost** | $20-200/month subscription | Free local models (optional API) |
| **Latency (API)** | 1-10s | 1-10s (same if using API) |

**When cloud AI is better:**
- Largest models (GPT-4, Claude Opus)
- No setup required
- Works on any device immediately

**When local-first is better:**
- Privacy-critical use cases (medical, legal, personal)
- Offline/unreliable internet
- Data sovereignty required
- Long-term cost savings (no subscription)
- Full control over data

---

## Use Cases for Containment + Local-First

### Medical (HIPAA Mode)
- Clinical note-taking with PHI protection
- Patient conversation logging (ephemeral, local-only)
- Medical reference queries without uploading patient context
- See [hipaa_compliance.md](../case_studies/hipaa_compliance.md)

### Personal Privacy (Guardian/Sovereign)
- Therapeutic journaling (stays on your device)
- Sensitive personal questions (financial, relationship, health)
- Creative writing (no risk of IP leakage)
- See [conversational_memory.md](../case_studies/conversational_memory.md)

### Field Work (Offline-First)
- Technician accessing manuals without internet (manuals uploaded to knowledge base)
- Emergency response with disrupted infrastructure
- Rural areas with unreliable connectivity
- See [hvac_field_service.md](../case_studies/hvac_field_service.md)

### Research (Data Sovereignty)
- Academic research notes (no cloud upload)
- Proprietary R&D conversations (IP protection)
- Journalist source protection (local-only)

---

## Implementation Details (v0.9)

### Containment Pipeline

**Code location:** `lib.rs` (`send_message_with_memory` function)

**Flow:**
```
1. User input received
2. Containment mode checked
3. If Child Mode:
   - OpenAI Moderation API called
   - If flagged: return block message, stop
4. If Other Modes:
   - Local TinyBERT classifier (toxic-bert)
   - Rule-based filters (HIPAA PHI patterns, etc.)
   - If flagged: return block message or warning
5. If passed: continue to LLM
```

### Local ML Models

**Storage:** `zynkbot_rust/src-tauri/models/system/`

**Models loaded at startup:**
```rust
// safety_classifier.rs
toxic-bert: 260MB (toxic content detection)

// nlp_enhancer.rs
bert-base-NER: 260MB (entity extraction)

// local_embeddings.rs
all-MiniLM-L6-v2: 80MB (embeddings for semantic search)
```

**Framework:** Candle (pure Rust ML, no Python dependency)

**Inference:** CPU-only (no GPU required for these small models)

### Database Architecture

**SQLite — embedded, no server process:**

**Memories table (simplified):**
```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,           -- UUID
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    namespace TEXT DEFAULT 'personal',
    is_ephemeral INTEGER NOT NULL DEFAULT 0,  -- HIPAA mode
    expires_at TEXT,               -- HIPAA 8-hour expiration (ISO timestamp)
    embedding BLOB,                -- 384-dim vector (binary)
    entities_detected TEXT,        -- JSON
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
```

See [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) for the full schema.

**Fully editable:**
- Standard SQL UPDATE and DELETE operations
- Memory Manager UI provides full CRUD interface
- No cryptographic hashing — just timestamps (hashing under consideration)

**Why SQLite:**
- Embedded — no separate server process or installation
- Single file makes backup trivial (copy the file)
- Standard SQL for querying
- Transparent (inspect with `sqlite3` or any SQLite browser)
- Sufficient for single-user memory workloads; tested to 500k+ rows without performance issues
- Vector search handled in-process by the Rust backend (no extension required)

---

## Limitations and Trade-offs

### Current Limitations (v0.9)

**1. HIPAA PHI Detection (70-85% accuracy)**
- Regex-based pattern matching
- Misses variations ("my social is 219907812" vs "SSN: 219-90-7812")
- Future: AI model for contextual PHI detection (95%+ accuracy)

**2. Local Model Quality**
- Llama 3.2 3B < GPT-4 quality
- Trade-off: Privacy + offline vs. best possible answers
- Solution: Hybrid approach (local + optional API) *API calls do not build user profiles

**3. Setup Complexity**
- Requires model downloads (600MB+)
- Initial model download can be slow on slower connections
- Cloud AI is easier to begin using (just sign up)

**4. Voice Transcription**
- Web Speech API requires internet
- Trade-off: Accuracy vs. privacy
- Future: Local Whisper model (larger download)

### Honest Assessment

**Zynkbot is NOT better than cloud AI for:**
- Users who want zero setup
- Users who prioritize convenience over privacy
- Users who need largest models (distributed compute on roadmap)
- Users who want seamless multi-device sync via cloud

**Zynkbot IS better than cloud AI for:**
- Users who need privacy (medical, legal, personal)
- Users in offline/low-connectivity environments
- Users who want data ownership
- Users who want transparent, editable memory
- Users & organizations who want to avoid subscriptions long-term (pay endless $ to corporations for AI)

---

## Future Enhancements

See [ROADMAP.md](../ROADMAP.md) for full timeline.

**Planned improvements:**
- AI-based PHI detection (95%+ accuracy)
- Local Whisper for offline voice transcription
- Mobile support (Android/iOS via Tauri Mobile)
- Enhanced containment rules (domain-specific filters)
- Opt-in secure backup servers 

**NOT planned:**
- ❌ Central servers for sync
- ❌ Telemetry or analytics
- ❌ Advertising or data sales

---

## Technical References

**Related Documentation:**
- [ARCHITECTURE_COMPREHENSIVE.md](ARCHITECTURE_COMPREHENSIVE.md) - Complete system architecture
- [FEATURES.md](../FEATURES.md) - All v0.9 features with implementation details
- [HIPAA Compliance Case Study](../case_studies/hipaa_compliance.md) - PHI detection and ephemeral memory
- [Digital Resilience](../DIGITAL_RESILIENCE.md) - Offline-first architecture benefits

**Source Code:**
- `src-tauri/src/lib.rs` - Conversation flow with containment (lines 1035-2150)
- `src-tauri/src/containment.rs` - Containment mode enforcement
- `src-tauri/src/safety_classifier.rs` - TinyBERT toxic detection
- `src-tauri/src/memory.rs` - Hybrid search and storage

---

*"Containment means blocking at the routing layer. Local-first means your data never leaves your device. Together, they create AI you actually control."*

**ContainAI** – Building the infrastructure for ethical AI


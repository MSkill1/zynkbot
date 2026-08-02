# Zynkbot Development Roadmap

**Last Updated:** August 2026
**Current Version:** v0.9 (Desktop Production-Ready)

This roadmap outlines planned features and enhancements. Timelines are estimates and subject to change based on community feedback and development priorities.

---

## v1.0 - Desktop Stable Release (Q3 2026)

**Focus:** Polish, stability, documentation

### Technical Debt

- **Memory identity merge on first sync** — When two devices sync for the first time, memories that already existed on the receiving device are not adopted into the synced namespace (KI-011). Fix requires an identity merge step during the first-sync handshake. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) (KI-011).
- ~~**ZynkLink/ZynkSync trust split**~~ ✅ — ZynkLink and ZynkSync now maintain independent trust records (`sync_paired` column). Unlinking and unsyncing are independent operations with no side effects on the other pairing.

### Known Limitations

- **Weak local models may store questions as memories** — The memory gating decision (`should_remember`) is entirely LLM-driven. Strong models (Anthropic Claude, dolphin 8B) correctly return `should_remember=false` for pure questions. Weaker models sometimes return `should_remember=true` for queries like "What's my name?", storing the question itself as a memory. Potential fix: add a lightweight pre-filter that detects obvious pure-question messages before the LLM call, avoiding model-quality dependency for this case.

### Core Improvements

- **Write-time memory consolidation** *(ships before v1.1 modularization — quick fix, high impact)* — Multi-turn conversations on a single topic currently produce many near-duplicate memories with dense `similar_to` cross-links. Real example: a single cooking session generated 12 memories, most re-phrasings of "I want to use up the leftover chicken." Threshold tuning is patchwork — no static bar handles both "three distinct facts in one message" and "twelve rephrasings across a session" correctly.

  **The fix — extend the existing relationship detection call:** Memory storage already makes a second LLM call for relationship classification. That call receives the top-K semantically similar existing memories along with the new candidate. Widen its JSON contract from a single classification to a three-way decision:
  1. **Fresh fact** → store as new memory (current behavior)
  2. **Rephrasing** of an existing memory → skip storing (or overwrite existing text if the new phrasing is clearer)
  3. **Elaboration** of an existing memory → store, but auto-link with the `elaborates` relationship (schema plumbing already exists — see commit `84136f4`, currently dormant)

  **Cost:** Zero new API calls. Same prompt structure, extended output schema. Rust-side JSON parser adapts to the new response shape; storage code branches on the decision.

  **Tradeoff:** Aggressive merging loses "different angle" nuance ("I care about not wasting food" vs "I like using leftover chicken" could be distinct signals). Start conservative — favor `elaborates` over hard-skip — and observe.

  **Optional follow-up (v1.0 or v1.1):** Session-end consolidation pass. When a conversation closes (or on a timer), run one LLM call over memories created in the last N minutes: "Are any of these redundant with each other? Merge if so." Catches within-session redundancy the write-time check missed. One API call per session, only if new memories were added.

- ~~**PDF support in Knowledge Base**~~ ✅
- **Word document (.docx) support in Knowledge Base** — index .docx files the same way PDFs are handled; extract text content for RAG search
- ~~End-to-end encryption for ZynkLink, ZynkSync, and ZChat (LAN traffic)~~ ✅
- Semantic conflict detection during sync (see labs/zynksync_improvements/)
- Performance optimization (memory usage, startup time)
- Cross-platform testing (Windows, Linux, macOS)
- Security audit and hardening
- Model Management UI — download/delete .gguf models from UI with progress indicators
- Auto-update notification — detect when a new version is available and prompt the user to update (git pull + restart)
- Multiple contradiction resolution — when a new memory contradicts more than one existing memory, only the first conflict is surfaced to the user; subsequent conflicting memories are silently skipped. Show a resolution modal for each conflict in sequence
- **Scroll-to-bottom button on Android** — When the conversation is long and the user scrolls up to review context, there is no way to jump back to the latest message without manually scrolling down. Add a floating arrow button (↓) that appears when the user is not at the bottom of the conversation and dismisses automatically when they reach the bottom. Small frontend change; already handled on desktop via standard scroll behavior.
- **Rotating startup tips** — Replace the single static "Remember:" tip with a rotating pool of tips that surface less-discovered features (Ensemble mode, KB search, ZynkLink, containment modes, etc.); one tip selected randomly on each launch
- **Camera/image support in Ensemble mode** — Vision-capable models (Claude, GPT-4o, Grok Vision) support image attachments in regular chat; Ensemble currently doesn't expose the attach button. Route the same `send_vision_streaming` path through Phase 1 so vision-capable models in an ensemble can each analyze the image; Phase 2 synthesis then compares their responses. Non-vision models in the ensemble skip the image or receive a text description.
- **GPU conflict warning scope: desktop only** — The Ensemble modal's "GPU conflict" warning (when selecting Custom/Ollama alongside a local GGUF) is only relevant on desktop, where both share the same CUDA device. On Android, local GGUF inference isn't supported anyway (Ollama is proxied to a paired desktop), so the warning is meaningless. Suppress the warning when `window.AndroidPaths` is present.
- **Zynkbot self-documentation refresh (KB seed docs)** — The KB documents that answer "Zynkbot, how does X work?" queries have drifted from current features. Missing: Android support (APK install, ZynkbotShare folder, Ollama proxying, permission model), Mistral as the 4th API provider, ensemble parallel Phase 1, GPU conflict guard, ZynkLink download manifest rescan, all v0.9.x additions. Audit `docs/FEATURES.md` and any seed KB docs, then update. Verify the "include Zynkbot in your question" flow still surfaces the right docs.
- **ZynkSync TLS handshake log spam from paired devices** — Paired Android clients occasionally spam the server log with `TLS handshake failed ... received fatal alert: HandshakeFailure` from many different ephemeral source ports, while normal sync operations to the same device still succeed. The failures come from parallel/probe connection attempts that don't complete negotiation (likely from short-timer retry loops or overlapping foreground-service triggers on the Android side). Not blocking any functionality — just log noise. Fix has two parts: (1) on the server, downgrade `HandshakeFailure` from known paired IPs to `debug!` level, keep `warn!` for unknown IPs (unknown = possible probe/attacker signal worth surfacing); (2) on the Android client, add a connection-attempt debounce so overlapping sync triggers don't fan out into parallel handshakes.
- **Relationship classifier prompt bias tuning** — The LLM classifier over-picks `contradicts` (treats "buying a new Pixel" + "owns iPhone" as replacement instead of addition) and over-picks `supports` when `caused_by` fits better (a "because X" statement got labeled `supports` instead of `caused_by`). Under-picks `elaborates` in ambiguous cases. Fix is prompt engineering in `lib.rs::ask_llm_about_memory_with_relationships`: add examples that distinguish (a) multi-valued attributes (phones, hobbies, books) from single-valued ones (age, spouse, address), and (b) causal wording ("because", "so that", "as a result of") from mere reinforcement. Also consider passing recent conversation turns as context so the classifier sees the actual multi-device / multi-book / multi-anything framing the user established.
- **Knowledge Base indexing progress indicator** — Large document indexing currently gives no feedback while running; add a progress bar or per-chunk status so it doesn't look frozen
- **Startup date reminder surfacing** — At launch, check for memories with dates falling within the next 7 days and surface them in the greeting ("By the way, your niece's birthday party is this Sunday"). The memory system already extracts dates; this adds a query against stored event dates at startup. No OS integration required — passive, in-app only.
- **Cross-device conversation history sync** — ZynkSync currently syncs the memory graph only; the raw conversation log stays local to each device. Syncing conversation history requires resolving integer primary key collisions across devices (using `entry_hash` for deduplication), careful handling of the `prev_hash` chain, and manual FTS5 index maintenance on the receiving end. Essential for the phone → desktop handoff use case.

### Documentation
- API documentation for contributors
- Video tutorials and demos
- Case studies and use cases
- Deployment guides (self-hosting, enterprise)
- GDPR alignment documentation for European deployments (architecture already well-suited; procedural layer — data export UI, breach notification workflow — to be documented)

### Community
- Contributors guide refinement
- Issue templates and labels
- Discussion forums setup
- First community contributions integrated

**Release Criteria:**
- ✅ All core features stable
- ✅ No critical bugs
- ✅ Documentation complete
- ✅ Community processes established

---

## v1.1 - ZynkSync & ZynkLink Modularization (Q4 2026)

**Focus:** Refactor the sync and linking layers into clean, composable internal modules before adding Android-native inference, Parenting Mode, or SDK surface area

ZynkSync and ZynkLink are currently tightly coupled to the desktop context. Before expanding to mobile-native local inference, Parenting Mode, or the public SDK, these layers need stable internal interface contracts — otherwise each new platform inherits the same coupling. This is a prerequisite milestone, not a feature release.

### Modularization Scope
- **ZynkSync module** — Extract sync protocol, conflict resolution, and namespace filtering into a standalone internal crate with a defined API; decouple from desktop-specific file paths and UI hooks
- **ZynkLink module** — Separate file-sharing transport, peer discovery, and permissions model; formalize the `ShareSource` trait that abstracts over desktop direct-fs and Android SAF (prerequisite for Play Store compliance)
- **Interface contracts** — Document the stable boundary each module exposes; these become the base the SDK Foundation builds on
- **Test coverage** — Unit and integration tests against extracted modules to catch regressions before mobile and SDK surface the same code paths

**Leads to:** v1.2 track — either local instance on Android, Parenting Mode, or SDK Foundation. Direction will be finalized based on community feedback and resource availability after v1.1 ships.

---

## v1.2 - Android + SDK Foundation + Companion Enhancements (Q4 2026)

**Focus:** Three co-primary tracks: Android launch, SDK Foundation groundwork, and companion/networking depth

### Android (Co-primary Track)

The core of Zynkbot is the Rust backend — memory system, ML inference, safety layer, and networking — and it is designed to run on Android. The frontend and database layers are not locked in:

- **Frontend:** Tauri Mobile is the current plan, but not a hard requirement — React Native or a thin native shell are viable alternatives. The Rust backend exposes a clean interface that any frontend can use, and it compiles to any target without Python or C++ dependencies.
- **Database:** SQLite — lightweight, embedded, no server process. Already in use on desktop; the same database layer carries forward to mobile without modification.
- **The Rust backend is the constant.** Everything else adapts to the platform.

#### Platform Support
- Android application (priority — v1.1)
- iOS application (follow-on; separate AppStore submission process)
- Mobile-optimized UI with touch gesture support
- Mobile system integration (notifications, background sync)

#### Mobile-Specific Features
- Offline mode optimization and battery efficiency improvements
- Mobile-friendly Snap-ins and voice input optimization
- Camera/photo integration (OCR, image analysis)
- **sqlite-vec: indexed vector search at scale** — current search is a linear scan (correct and fast at typical usage; degrades at very large memory counts). sqlite-vec adds approximate nearest-neighbor indexing so search stays fast regardless of how many memories accumulate. Applies to both desktop and mobile.
- **Desktop ↔ Mobile sync via ZynkSync** — Conflict resolution improvements for mobile edge cases; bandwidth optimization and background sync scheduling

#### On-Device AI Research (Mobile)

Local model inference on phones is a distinct architectural problem from desktop GGUF/llama.cpp. CPU-only GGUF models miss device NPUs entirely. Investigation needed:

- **Apple (iOS/macOS):** Core ML format — Apple's Neural Engine runs Core ML models natively. Conversion tooling (coremltools) exists for common model families.
- **Qualcomm Snapdragon (Android):** QNN/ONNX format — Snapdragon's Hexagon NPU is the dominant Android AI accelerator. Qualcomm AI Hub provides pre-optimized models for common architectures.
- **Google Tensor (Android):** TFLite / LiteRT — Google's in-house NPU on Pixel devices. TFLite models target the Tensor chip directly.
- **Fallback:** GGUF/llama.cpp CPU inference — correct and functional, just slower and battery-intensive. Suitable as a universal baseline while platform-native paths are evaluated.
- **Memory quality bridging** — Local 7B models can extract facts but may produce imprecise relationship classification JSON. When the phone connects to a larger model (home server or user-approved API call), queued locally-extracted memories should be re-evaluated by the larger model for accuracy before permanent storage. Design the re-check protocol here.

**Goal:** Identify which format(s) to target for v1.1 Android launch; ship CPU-path GGUF as baseline while NPU investigation continues.

---

### SDK Foundation (Co-primary Track)

Early groundwork for the developer platform. Full SDK public release is v3.0; v1.1 establishes the internal architecture so the surface area is stable before exposing it externally.

- **Define clean internal API boundaries** — Identify the Rust modules that become SDK-facing (memory system, containment layer, ZynkSync protocol, safety classifier). Ensure each has a clear interface contract, not just internal use.
- **Snap-in architecture hardening** — Snap-ins are the primary SDK extension point. Finalize the data contract and lifecycle hooks so third-party snap-ins can be built against a stable interface.
- **Documentation-first approach** — Write the SDK developer guide before the public release. Internal use forces discovery of gaps.
- **CLI scaffold for snap-in development** — Basic tooling for creating, testing, and packaging a snap-in locally.

---

### User Profile Enhancements

- **User profile update mechanism** — The onboarding process writes a `user_profile.json` file containing the user's full name, preferred name, and age at the time of onboarding. Currently there is no way to update these values after the fact short of re-running onboarding. Add a simple profile editor (accessible from Settings or Memory Manager) that lets the user update any field. Future profile fields to consider adding as use cases emerge: date of birth (to derive age automatically), timezone, occupation, preferred language, pronouns. The JSON structure is intentionally open-ended so new fields can be added without breaking existing reads.

### Companion Layer Enhancements

- **Push notification reminders** — Full OS-level reminders via `tauri-plugin-notification` (cross-platform: Linux, Windows, macOS). User sets lead time (e.g., 1 day, 1 hour before); reminders fire even when the app is minimized. Requires background scheduler and notification permission handling per platform. Builds on the startup date surfacing added in v1.0.

- **Emotional State Awareness** — Detect user's emotional tone before the main LLM call
  - Lightweight sentiment/distress classification on user input
  - Adjust response framing based on detected state (distress, frustration, neutral, positive)
  - Builds continuity across sessions without the user having to re-explain - elaborates/causes chains
- **Per-User Tone Adaptation** — Learn and match individual communication style over time
  - Store tone preferences derived from feedback and conversation patterns
  - Adjust formality, verbosity, and directness per user
  - Stored locally; never inferred from external data

- **Atomic fact extraction with elaborates-linking (deferred from v0.9)** — Currently the LLM prompt asks for one MEMORY_EXTRACT line per user message, combining all personal facts from that message into a single compound memory. Compound storage relies on semantic similarity to surface relevant fragments — e.g. "User has two nephews John (8) and Jack (9)" should match queries about either name or either age. If user feedback during v0.9 shows retrieval missing on specific sub-facts (e.g. "how old is Jack?" failing to surface the nephew memory), switch to atomic extraction:
  - Change the MEMORY_EXTRACT instruction in `conversation_engine.rs` to emit one line per distinct fact instead of per message
  - Each fact becomes its own memory row with a focused embedding
  - Co-extracted facts from the same message get auto-linked with an `elaborates` relationship (the plumbing for this already exists in `lib.rs` since the SQLite migration — see commit `84136f4`, `if stored_ids.len() > 1` block, currently dormant under compound prompting)
  - Trade-off: richer relationship graph and sharper per-fact retrieval, at the cost of fragmenting the user's original phrasing across multiple memories and a denser `elaborates` edge set in the graph view
  - Defer until retrieval issues are observed in practice; don't pre-optimize

(Slim system prompt for local models — implemented in v0.9; previously this section listed it as deferred. See `conversation_engine.rs::build_prompt` where `is_api_model == false` now branches to a ~350-token slim system prompt that preserves all behaviors but condenses the voice section and MEMORY_EXTRACT examples. Necessary because KB context (~1.4k tokens) + the previous 1.2k system prompt + memory recall would overflow a 4K-window local model.)

### Conversation History Enhancements

**"What Did I Learn?" Digest** — A periodic summary view showing what you got out of your conversations, derived from the semantic memory system.
- Weekly and monthly digest views
- Digest entries link back to source conversations
- Topic grouping and message count per topic

**Thread Branching Chart** — Visual diagram (git-branch style) showing where a conversation went off-topic and how it returned.
- Per-conversation branch view accessible from message view
- Shows turn number where topic shifted, length of each branch, and return point

**Memory ↔ Conversation Linking** — Bidirectional link between the semantic memory system and the raw conversation log.
- In Memory Manager: "Source conversations" link on each memory entry
- In conversation history: annotation on messages showing which memories were extracted

**Resumed Session Summarization** — For very long conversations, auto-summarize earlier turns into a compact brief rather than overflowing the context window.
- Inject the brief as a system-level context note, followed by the most recent N turns verbatim

**Feedback Log Viewer** — Read path for the `message_feedback` table (thumbs up/down ratings already collected).
- `get_feedback_log` Tauri command: JOIN `message_feedback` with `conversation_messages` on `cm.id::TEXT = mf.message_id`, return rated responses with text, model backend, and timestamp
- `FeedbackLogPanel.jsx` modal: summary stats (total rated, 👍 / 👎 counts), list of rated responses with faded text preview and model/date metadata - with user consent, gather data on model preferences and usage
- "Feedback" button next to "History" in the Conversation header

**Export**
- Export session to JSON
- Export session to plain text / Markdown

### ZynkSync Enhancements
- **Namespace Filtering UI** — Checkbox in ZynkSync settings to select which namespaces sync
  - Database already supports `namespace` column and indexes
  - Backend filtering logic needed in `zynksync.rs`; UI controls in ZynkSyncPanel.jsx
  - Use case: keep "work" namespace local, sync "family" namespace
- **is_syncable Checkbox?** — Per-memory control in MemoryManager UI
  - Database already has `is_syncable` column (default true)
  - Add checkbox to MemoryManagerModal.jsx edit form
- **Sync Conflict Viewer** — UI to review past conflict resolutions
- **Selective Device Sync** — Choose which paired devices receive which namespaces

### Security
- ~~**TLS 1.3 Encryption** — Encrypt all ZynkSync/ZynkLink/ZChat traffic~~ ✅
- ~~**mTLS Device Authentication** — Devices present their certificate during TLS handshake; server verifies cert against paired-device DB before accepting protected requests. Pairing-bootstrap routes remain open; all sync, file transfer, messaging, Ollama, and API-key propagation routes require a verified cert.~~ ✅
- **Audit Logging** — Comprehensive exportable logs for all network operations (who synced what, when)
- **Network request limits** — Per-connection body size cap, concurrent-request limit, and hard timeouts to prevent a paired device from exhausting CPU, RAM, or bandwidth on the host
- **OS Keychain / Android Keystore storage for API keys** — Currently stored in a plaintext `.env` file; migrate to the OS credential store (macOS Keychain, Windows Credential Manager, Linux Secret Service, Android Keystore) so keys are protected at rest even if the filesystem is accessible to another process
- **Prompt injection defense** — Memory content injected into prompts should be clearly delimited and labeled as untrusted data; LLM-extracted memories containing instruction-like text must not be able to override system-prompt behavior
- **Memory provenance and confidence** — Store the source conversation, extraction timestamp, model used, and a confidence indicator for each extracted memory; enables tracing incorrect memories back to origin and auditing extraction quality over time

### Ensemble Enhancements
- **User-selectable coordinator model** — Currently auto-selected (Anthropic → xAI → OpenAI → local); allow user to manually designate which model acts as coordinator. Critical: the coordinator's training biases shape how the synthesis frames consensus and uncertainty — two coordinators can reach opposite verdicts from identical responses. For sensitive or contested questions, coordinator selection is not cosmetic.
- **Per-question model presets** — Save favorite model combinations for specific use cases (e.g. "research" preset, "creative" preset)

### ZynkLink Enhancements

- **Android share-sheet target** *(small, high impact)* — Register Zynkbot as a share target in Android's share sheet so users can share any file (photo, document, etc.) directly into ZynkbotShare from any app. Currently, getting a photo from the camera roll into ZynkbotShare requires manually navigating to DCIM/Camera and copying the file — the share sheet eliminates that friction entirely. Implementation: add an `<intent-filter>` for `ACTION_SEND` / `ACTION_SEND_MULTIPLE` with `*/*` MIME type to `AndroidManifest.xml`, handle the incoming `content://` URI in `MainActivity.kt` by copying the file into ZynkbotShare via `ContentResolver`, and surface a brief toast confirming the file is ready to share. Small Kotlin addition, no Rust changes needed.

- **Android Storage Access Framework (SAF) migration** *(Play Store blocker)* — The current Android build relies on the `MANAGE_EXTERNAL_STORAGE` permission to see files placed in `Downloads/ZynkbotShare/` by other apps. That permission is heavily restricted by Google Play — most apps are denied unless they fit a narrow set of use cases (file managers, backups, antivirus). To ship on Play Store, Zynkbot must either (a) get approved for `MANAGE_EXTERNAL_STORAGE` under the "sync/backup" use case, or (b) migrate to SAF, which is Google's blessed API for accessing user-chosen folders without broad file-system permissions. Path (b) is preferred because it's more likely to pass review.

  **Architecture change required:** Introduce a `ShareSource` trait in Rust with two implementations — one that uses `tokio::fs` directly (desktop, older Android, Zynkbot-owned folders) and one that talks to a Kotlin bridge (Android 11+ SAF-granted folders). All existing call sites in `scan_directory`, `handle_zynklink_download`, and the peer-download writer switch from direct `tokio::fs` calls to the trait. Kotlin gains bridge methods for enumerating a granted folder via `DocumentFile` and for streaming file bytes to/from Rust via app-private staging. First-launch flow gains a SAF folder-picker prompt on Android 11+.

  **Scope estimate:** ~200-300 lines of new Rust (trait + impls + wiring), ~150 lines of new Kotlin (grant + list + bidirectional byte streaming), ~50 lines of JS changes. Realistic effort: 2-3 focused days including cross-device testing on Android 8, Android 11+, desktop, and mixed setups. Should be done on its own branch.

  **Interim state:** `MANAGE_EXTERNAL_STORAGE` is declared in the manifest and requested at first launch. This works for GitHub-distributed builds and is documented in KNOWN_ISSUES.md as KI-015. Must be swapped for SAF before any Play Store submission.

- **Federated Knowledge Base Query** *(post-Android-launch)* — Allow a paired user to query another user's KB *in place*, without documents ever transferring. User A sends a question to paired User B's device; B's Zynkbot runs local RAG retrieval against B's own KB and returns only the top-k relevant passages. Source documents never leave B's device.

  **Use case:** Small teams and collaborators where one person maintains a reference library (a lawyer's case files, a lab's papers, a team's docs) and others need to query it without receiving wholesale copies. "Query my documents without possessing my documents" — a posture the cloud providers structurally cannot offer.

  **Architecture notes:** Most plumbing already exists — ZynkSync/ZynkLink pairing and auth, TLS transport, and the local RAG query path. The new piece is a remote-query endpoint: receive a question, run local KB retrieval, return relevant chunks. Tractable addition, not a new subsystem.

  **Required for the privacy claim to be honest:**
  - Returned chunks ARE document contents. This meters disclosure per-query; it does not prevent it. A determined querier could extract a document chunk by chunk. Frame as "granular, logged, revocable sharing" — never as "they can't see your files."
  - Per-document (or per-folder) sharing consent on the owner's side — opt-in which KB content is remotely queryable, default none.
  - Query log visible to the KB owner (who asked what, when).
  - Revocation: unshare instantly, per peer or per document.

  **Why deferred:** Post-launch, post-local-models-on-mobile. Should be informed by real user feedback on whether shared-KB workflows are actually in demand.

- **File Upload** — Send files TO paired devices (not just download); requires write permission
- **Live File Sync** — Auto-sync changed files in shared directories (incremental, conflict resolution)
- **Share Permissions UI** — Read/write/delete permissions, time-limited shares
- **Streaming File Transfer** — Replace in-memory file buffering with chunked streaming for large file transfers
  - **Current behavior:** The sending device reads the entire file into RAM before transmitting; the receiving device accumulates all bytes in memory before writing to disk. This works — a Mixtral 7B model (~4GB) has been transferred successfully — but it requires both devices to have enough free RAM to hold the entire file at once.
  - **The risk:** On a machine where available RAM is close to or less than the file size (e.g. a laptop that already has a model loaded, or a machine with 8GB RAM receiving a 7B model), the transfer will fail with an out-of-memory error rather than a clean message. Larger models (13B, 70B) make this worse.
  - **The fix:** Stream the file in chunks on both ends using HTTP chunked transfer encoding. The server reads and sends a few MB at a time; the receiver appends each chunk directly to disk. Memory usage stays constant regardless of file size.
  - **Resumable transfers** (stretch goal): HTTP Range request support would allow an interrupted download to resume from where it left off rather than restarting. This is a meaningful additional change and can be done independently of the streaming fix.

### ZChat Enhancements
- **Group Chat** — Multi-device group messaging with named groups and history
- **File Attachments** — Send files via ZChat (integrated with ZynkLink)
- **Message Search** — Full-text search across past ZChat messages

### Snap-in Enhancements
- **Therapist snap-in note export** — Export session notes, insights, and conversation excerpts to plain text, Markdown, or PDF; useful for sharing with an actual therapist or keeping an offline record

### Knowledge Base Enhancements
- **GPU/CUDA acceleration for embeddings** — Offload the sentence-transformer embedding model to GPU during document indexing; significantly reduces indexing time for large corpora on machines with a capable GPU; CPU fallback remains for machines without CUDA

### Voice Input (Whisper Re-enablement)
Voice transcription is implemented (`useVoiceInput.js`, `whisper.rs`, `transcribe_audio` Tauri command) but disabled due to a GGML symbol collision between llama.cpp and whisper.cpp that causes a link error at build time. A Candle-based Whisper alternative was investigated but has the same gibberish-on-microphone issue (upstream: candle Issue #2182 — model works on audio files, not live microphone input).

- **Monitor upstream fixes** — Track llama.cpp and candle for resolution of the GGML collision / microphone input issues (build this?)
- **Re-enable voice input** — Once a clean path exists, wire `useVoiceInput.js` back into the UI (currently code-complete; only the UI toggle is disabled)
- **Evaluate alternative approaches** — e.g. OpenAI Whisper API (implemented online fallback), or a standalone Whisper binary called via subprocess to avoid the symbol conflict entirely

---

## v1.3 - Advanced Containment Modes (Q1 2027)

**Focus:** Production-ready containment for specialized use cases

### HIPAA Mode Enhancements
- **AI-Based PHI Detection** — Replace regex with a specialized model
  - Target 95-99% accuracy (current regex: 70-85%)
  - Contextual understanding ("my social is 219907812" caught, not just "SSN: 219-90-7812")
  - Local inference only
- **Audit Cryptography** — Tamper-proof audit logs with digital signatures and append-only structure
- **Role-Based Access Control** — Multi-user HIPAA deployments (physician, nurse, admin roles)
- **BAA Template** — Business Associate Agreement template and compliance documentation

### Child Mode Enhancements
- **Parental Dashboard** — Review child activity, view blocked content attempts, adjust sensitivity
- **Educational Reports** — Learning progress tracking and exportable progress reports
- **Custom Blocklists** — Parent-defined topic blocks and allow-list mode

### Sovereign Mode Enhancements
- **Crypto Integration** — Read-only blockchain queries and transaction explanation (never sign transactions)
- **DeFi Safety** — Phishing pattern detection, suspicious contract warnings, rug pull heuristics
- **Filtering** — Phishing pattern detection, suspicious contract warnings, rug pull heuristics.  Focus on allowing users to use Zynkbot as their primary search engine/filter internet according to user instructions. AI generated content (text, picture/video, audio) detection.

---

## v1.4 - ContainAI Services (Q2 2027)

**Focus:** Opt-in cloud services for users who want them, without compromising the local-first architecture

### Memory Backup Service
- **Cryptographically Encrypted Remote Backup** — Opt-in memory backup (~$2.99/month)
  - Memories encrypted before leaving device (zero-knowledge — ContainAI servers store encrypted blobs only)
  - Restore to any device after hardware failure, loss, or theft
  - Off by default; toggle in Settings → Privacy
  - First commercial offering to fund Zynkbot development

---

## v2.0 - Advanced Memory Features (2027)

**Focus:** Enhanced memory capabilities and intelligence

### Memory Layer Extensions

#### Memory Decay & Re-surfacing
- Priority scoring for memories
- Automatic re-surfacing of relevant old memories
- "You mentioned this 6 months ago..." reminders
- Intelligent forgetting (reduce clutter from trivial memories)

#### Threaded Recall
- Timeline view of related memories
- Story mode: "My journey with X" (e.g., "Career doubts: 2023-2025")
- Automatic thread detection and memory clustering by topic

#### Mode-Based Memory Gates
- Memories accessible only in certain modes
- Child mode hides sensitive content
- Work namespace hidden in personal mode

#### MemoryVault — Tamper-Evident Conversation Log
Hash chain integrity layer on top of the v1.0 conversation history tables. The basic log exists; this adds cryptographic tamper-evidence.

- Populate `entry_hash` and `prev_hash` columns (schema stubs already in v1.0 tables)
- Hash chain verification command — detects if any record has been modified
- Opt-in per user; export with hash chain intact for audit purposes

**Use cases:** Legal defensibility, clinical accountability (therapy, medical practice), compliance audit trail for regulated industries.

#### Advanced Conversation History UI
- **Topic Timeline** — Group conversations by detected subject/theme with swim-lane visualization showing frequency and evolution over time
- **Spatial Canvas View** — Conversations as nodes on a 2D canvas (Obsidian/Miro style), AI auto-clustered by topic. *Highest complexity — design-phase item.*
- **"Past-Self" Queries** — Ask Zynkbot questions about your own history: "What was I asking about in February?"

---

## v2.5 - Pattern Recognition & Self-Reflection (2027)

**Focus:** Help users understand themselves

### Reflective Intelligence Enhancers

#### Self-Analysis Tools
- Emotional tone tracking over time
- Decision-making pattern analysis
- Behavior drift alerts ("You used to do this daily")
- Self-contradiction markers ("This conflicts with April 2026")
- Personal hypocrisy detector 

#### Pattern Drift Analyzer
- Mood timeline visualization
- Value alignment view (actions vs stated principles)
- Habit formation/breaking tracker
- Life event correlation (mood shifts during major events)

#### Intent-Outcome Tracker
- "I said I'd do X" → Did you actually do X?
- Goal consistency measurement and proactive reminders
- Accountability partner mode

#### MirrorPath Snap-in
- Dedicated self-reflection workspace with AI-assisted journaling
- Pattern visualization dashboard; exportable reports (for therapy, coaching)

---

## v3.0 - SDK & Developer Platform (2027)

**Focus:** Enable third-party development

### Zynkbot SDK

#### Core Modules
- **Containment Layer** — Consent-based safety framework with pluggable modes and custom rules
- **Memory System** — Hybrid semantic + entity search, relationship detection, namespace support
- **ZynkSync Protocol** — Cross-device synchronization with conflict resolution framework
- **HIPAA Framework** — PHI detection, memory system disable, audit logging, compliance helpers
- **MemoryVault** — Hash-chained conversation history log (tamper-evident, opt-in, local)
- **GDPR Framework** — Right to erasure, data portability export, breach notification workflow
- **Snap-in Architecture** — Domain-specific workspaces

#### Developer Tools
- SDK documentation and tutorials
- Code examples and templates
- Testing framework and debugging tools
- CLI for Snap-in development

#### Snap-in Marketplace
- Snap-in discovery and installation
- Developer accounts and publishing
- Revenue sharing model with quality assurance and vetting
- User reviews and ratings

### Licensing Model
- ✅ Free for non-commercial use
- 💰 Paid commercial licensing (tiered pricing)
- 🤝 Revenue supports open source development

---

## v3.5 - Professional Snap-ins (2027–2028)

**Focus:** Domain-specific applications built on the SDK

### Healthcare Snap-ins
- **Patient Portal** — PHI-aware conversation logging
- **Symptom Tracker** — Medical history with privacy controls
- **Medication Reminder** — AI-assisted medication management
- **Therapy Journal** — HIPAA-friendly session notes

### Education Snap-ins
- **Homework Helper** — Child-safe tutoring
- **Study Planner** — Adaptive learning schedules
- **Research Assistant** — Citation management, note-taking
- **Language Tutor** — Conversational practice

### Professional Snap-ins
- **Legal Assistant** — Case notes, client tracking
- **Financial Advisor** — Budget tracking, investment notes
- **Project Manager** — Task scaffolding, team collaboration
- **Writing Coach** — Long-form writing assistant

### Personal Snap-ins
- **Fitness Tracker** — Workout logging, nutrition advice
- **Relationship Manager** — Gift ideas, important dates
- **Travel Planner** — Itinerary builder, trip memories

---

## v4.0 - Foundation & Ecosystem (2028)

**Focus:** Long-term sustainability

### ContainAI Foundation

#### Structure
- Nonprofit incorporation (501(c)(3) or equivalent)
- Board of directors establishment
- Governance model (community input)
- Grant programs for aligned developers

#### Revenue Model
- 💰 Zynkbot premium features (Parenting Mode, Pro Snap-ins)
- 💰 SDK commercial licensing
- 🤝 Corporate sponsorships (privacy-aligned companies)
- 🎁 Individual donations and grants

#### Programs
- Security audit funding
- Privacy research grants
- Open source project support
- Educational initiatives

### Partner Integrations

#### Privacy-Aligned Organizations
- **Proton** (VPN, encrypted email, calendar)
- **Signal** (secure messaging)
- **Tutanota** (encrypted email)
- **Jitsi** (video conferencing)
- **Nextcloud** (file storage)

---

## Research & Experimental Features

*Unscheduled — requires research, external dependencies, or community interest*

### Advanced Networking
- **Mesh Networking** — Device-to-device communication without any network infrastructure

  **Why this matters:** Zynkbot's current networking (ZynkSync, ZynkLink, ZChat) requires a shared local network — a home router, office LAN, or mobile hotspot. That covers most use cases. But there are situations where even a hotspot isn't viable: active infrastructure suppression, remote environments with no wireless infrastructure, or contexts where creating a WiFi network draws attention. Bluetooth and WiFi Direct allow two devices to communicate directly without any intermediary — no router, no hotspot, no ISP.

  **Target platform:** Primarily mobile (Android/iOS). Desktop Bluetooth is technically feasible but the compelling use case is phones. This feature becomes meaningful once Mobile is production-ready.

  **Planned capabilities:**
  - Bluetooth pairing for ZynkLink contact exchange (replaces IP-based pairing)
  - ZChat delivery over Bluetooth when no network is available
  - ZynkLink file transfer over Bluetooth or WiFi Direct
  - Mesh relay: messages hop between devices to reach a destination out of direct range

  **Prior art:** Briar (open source) demonstrates this architecture works for secure messaging under network suppression. The goal is to bring the same infrastructure-independence to an AI assistant context — private notes, shared documents, and AI model distribution without touching the internet. Collab?
- **Tor Integration** — Anonymous remote sync over Tor hidden services (experimental, performance trade-off)
- **Sneakernet Mode** — USB-based sync for air-gapped environments; export/import memory snapshots

### AI Enhancements
- **On-Device Fine-Tuning** — LoRA adapters trained on user's conversations (never leaves device)
- **Multimodal Models** — Local CLIP for image embeddings, image memory retrieval, visual QA
- **Hallucination Detection** — Confidence scoring, source verification
- **Explainable AI** — Step-by-step reasoning display

#### Epistemic Humility / Refuse-on-Contradiction

**Status:** Research spike, not scheduled. Post-1.0.

**Concept:** Add a reasoning-integrity layer that detects when a query or its supporting memory context is internally contradictory, or when the available evidence is insufficient, and responds with a structured "I can't answer this confidently — here's the conflict / here's what's missing" instead of collapsing into a hallucinated answer. The goal is a model that knows when *not* to answer.

**Why it fits:** Zynkbot already has the two primitives this needs:
- **Contradiction detection** over the memory graph — already identifies when facts conflict.
- **Ensemble mode** — independently-trained models disagreeing is already a usable uncertainty signal.

This feature would combine those into an explicit epistemic gate: before generating, check whether the prompt + retrieved memory context contains unresolved contradictions or low-confidence/high-disagreement signals; if so, surface the conflict for the user rather than answering over it.

**Possible mechanics (to be designed):**
- Run the check on the prompt + retrieved memory-graph context at generation time.
- On detected contradiction or insufficient evidence, short-circuit to a structured response ("this conflicts with X in your memory" / "not enough stored context to answer this well").
- Optionally tag the relevant memory nodes/edges with an uncertainty score and surface high-uncertainty nodes in the contradiction-resolution UI; use user resolutions to improve future stability.
- Optionally modulate strictness by containment mode (e.g. stricter refusal in higher-assurance modes).

**Constraints / requirements:**
- Must be built from Zynkbot's own primitives in pure Rust — no external runtime, must work on mobile (no Python sidecar).
- Must be validated before shipping: demonstrate it catches a class of hallucination or overconfident answer that the existing contradiction-detection + ensemble path misses. If it doesn't measurably improve on what's already there, it doesn't ship.
- Fully first-principles / own-implementation, so it stays license-clean and commercializable under the existing dual license.

**Why deferred:** Non-blocking, post-launch, and needs measurement before it's worth building. The hard part is evaluation (what does it actually improve?), not the code.

**Effort:** Research spike first, then implement only if the eval shows real benefit. The spike must define a concrete failure class before writing any code — e.g., "user asks a question where their memory graph contains a direct contradiction and the model currently answers confidently rather than flagging it" — and assemble a test set of those cases. Without a clear eval target going in, the spike can't produce a meaningful verdict.

### Personal Ethics Layer
- **Custom Constraint Flags** — User-defined guardrails ("don't help me lie unless in Sovereign Mode")
- **Value Alignment Scoring** — Actions vs stated principles
- **Ethical Dilemma Advisor** — Explore decision consequences

### Crisis Support
- **Crisis Companion Mode** — Grounded language, breathing exercises, resource directory (hotlines, local services)
- **Panic Attack Assistant** — Guided calming techniques

### Privacy Innovations
- **Homomorphic Encryption** — Search encrypted memories without decrypting (research phase)
- **Federated Learning** — Collaborative model improvement without data sharing

### Secure Vault (Passwords & Financial Info)

A local-first, LAN-synced credential and financial-info vault built on top of Zynkbot's existing architecture. Positioned as an alternative to Proton Pass, 1Password, and Bitwarden — but with **no third-party servers involved at any layer**. If your Zynkbot devices sync to each other over your own LAN, your passwords sync too. There's no account, no cloud, no vendor that could be compelled or breached.

**Why this fits Zynkbot's story:**
- The infrastructure is already in place: encrypted SQLite storage, TLS-secured LAN sync, ED25519 device pairing (planned v1.2)
- The philosophical stance ("your data never leaves your devices") already matches the ideal password-manager posture
- Memory-system integration means you could ask "what's my Amazon login?" naturally, with auth gating — something no existing manager offers

**What makes this hard (not the same complexity as adding a Snap-in):**
- **Separate encryption subsystem.** Vaults require master-password derivation (Argon2id), per-entry authenticated encryption (AES-256-GCM or XChaCha20-Poly1305), memory-locked secret handling in Rust (`zeroize` crate), secure clipboard clearing, screen-record protection. Cryptographic mistakes are unforgivable in this domain — the bar is much higher than "another memory type."
- **New threat model.** Today Zynkbot trusts the device. A vault means assuming the device might be compromised while unlocked — biometric locks, session timeouts, panic-wipe, autolock on window blur, per-entry re-auth for high-value items (financial data).
- **Autofill integration.** Android requires Autofill Framework or Accessibility Service integration. iOS requires the AutoFill Credential Provider extension. Desktop requires browser extensions. Each platform is a separate implementation with review-process implications.
- **Regulatory surface.** Storing card numbers touches PCI-DSS considerations if ever distributed commercially. Passwords alone are lower stakes but still subject to breach-notification laws in some jurisdictions.
- **Cross-device consistency guarantees.** Sync conflicts on a memory are annoying; sync conflicts on a password are dangerous (which version is the current one?). Requires strict last-writer-wins with vector clocks or CRDTs, not the memory system's semantic-merge approach.

**Scope (if pursued):**
- Master password + biometric unlock (platform-specific hooks)
- Encrypted vault DB separate from memory DB — different keys, different lifecycle
- Password entries, credit-card entries, secure notes, TOTP generator, form-fill metadata
- Auto-lock policies (timeout, blur, panic-shortcut)
- Autofill on each platform (staged: desktop first, Android second, iOS last)
- Emergency access (Shamir secret sharing to trusted contacts?)
- Export/import in 1Password/Bitwarden/KeePass formats

**Why deferred:** Far future. Warrants a dedicated security audit and a separate design document before any code is written. Should not be attempted before v2.0 at the earliest — the core memory system and cross-device story need to be rock solid first, since vault sync is unforgiving of the bugs a memory system can tolerate.

---

## Platform Expansion

- **iOS** — Tauri Mobile for iPhone/iPad (App Store limitations and background sync challenges to navigate)
- **Web Version** — Browser-based, API-only, limited offline capability; use case: Chromebook, public computers
- **Raspberry Pi** — Headless mode for a dedicated home server; low-power 24/7 family or team instance
- **Enterprise / multi-user deployment** — Zynkbot's desktop version is intentionally single-user and local-first. A separate server-based, centralized-database implementation (with vector search) exists that would suit organizations wanting a shared memory store — a hospital where multiple staff interact with the same patient history, for example, or a team knowledge base. The SQLite migration occurred shortly before open-sourcing, leaving that centralized-database branch nearly complete — finishing it for enterprise use would be a targeted effort, not a ground-up build. If there's enterprise interest, it's a realistic development target.

---

## Long-Term Vision (2028–2030)

**Privacy-first AI at scale**
- Demonstrable alternative to surveillance capitalism at scale
- Privacy-first AI becomes a mainstream expectation, not a niche

**SDK as the Standard for Privacy-First AI**
- Healthcare, education, legal, and finance sectors with compliance frameworks built on transparent architecture
- Active ecosystem of third-party applications

**Self-Sustaining Ecosystem**
- Revenue from premium features, SDK licensing, and donations
- Active developer community
- Partnerships with privacy-focused organizations (EFF, Signal Foundation, etc.)

---

## What We Won't Build

**To maintain focus, we explicitly won't:**
- ❌ Engagement optimization (anti-user)
- ❌ Data selling or sharing (anti-ethics)
- ❌ Proprietary lock-in (anti-open source)
- ❌ Dark patterns or manipulation (anti-transparency)
- ❌ Central servers for sync or storage
- ❌ Telemetry or analytics

---

## Community Involvement

We welcome community input on this roadmap.

**How to contribute:**
- Open GitHub issues for feature requests
- Comment on existing roadmap items
- Submit pull requests for features you want to build
- Join discussions about priorities

**Vote on priorities:**
- GitHub Discussions poll (quarterly)
- Community roadmap input

---

## Versioning Philosophy

- **v0.x** — Pre-1.0 development (current)
- **v1.x** — Stable desktop releases
- **v2.x** — Advanced memory features
- **v3.x** — SDK and developer platform
- **v4.x** — Foundation and ecosystem

Each major version focuses on a specific theme. Minor versions add features within that theme.

---

## Get Involved

- **Developers**: Check [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Users**: Try Zynkbot and give feedback
- **Researchers**: Propose collaborations
- **Organizations**: Explore commercial licensing

**Contact:** matt@containai.ai
**GitHub:** https://github.com/MSkill1/zynkbot

---

**Vision:** Privacy-first AI that empowers, not exploits.
**Mission:** Build tools for conscious growth, not automated distraction.

*This roadmap is a living document. Priorities may shift based on community feedback, technical discoveries, and resource availability.*

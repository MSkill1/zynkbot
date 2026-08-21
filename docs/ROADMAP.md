# Zynkbot Development Roadmap

**Last Updated:** August 2026
**Current Version:** v0.9 (Android internal testing, Desktop production-ready)

This roadmap outlines planned features and enhancements. Timelines are estimates and subject to change based on community feedback and development priorities.

---

## v0.9.5 — "Hey Zynk" Wake-Word + Timer (Priority — build first)

**Status:** Not started. Primary next feature after launch.

**Goal:** Say "Hey Zynk, set a timer for 10 minutes" and have it actually create a working countdown — wake-word detection and the timer action built together, with the timer as the first concrete thing the wake word triggers (rather than building wake-word in the abstract with nothing for it to do yet).

**Why build them together:** Wake-word alone is a demo with no payoff; timers alone don't need wake-word to be useful. Pairing them means the very first wake-word build has an immediate, testable, satisfying result — say the phrase, hear a countdown confirmed — instead of a listening service with nothing to show for it.

### Part 1 — Wake-word detection

1. Integrate an on-device wake-word engine (e.g. Porcupine/Picovoice, or an open alternative) trained on "Hey Zynk" — check current licensing terms for custom wake words before committing (free tiers often limit custom phrases).

2. App-level only — NOT an OS modification. No GrapheneOS/AOSP fork, no system-level audio hook. Works identically on stock Android and GrapheneOS.

3. Implement as an Android background/foreground service:
   - Default OFF — always-listening should be opt-in, not default (matches the project's privacy posture).
   - On detection: brief chime/visual confirmation (so the user knows it heard them), then capture the following speech via the existing voice-input pipeline (reuse, don't rebuild).

4. Battery/lifecycle testing — this is the real risk, not the detection model itself:
   - Service survives being backgrounded, isn't killed by Android's process management.
   - Measure real battery drain over a full day of always-listening.
   - Confirm behavior when the phone is locked.

### Part 2 — Timer action (the first thing the wake word triggers)

5. Add a lightweight intent-check step: after wake-word capture (or from any typed/dictated input), ask the model whether the message is a timer/alarm request and extract the duration as structured output (e.g. `{"is_timer": true, "duration_seconds": 600}`).

6. Android: implement the actual timer via `AlarmManager` (`setExactAndAllowWhileIdle` or `setAlarmClock`) + a completion notification with sound, so it fires even if the app is backgrounded. Requires `SCHEDULE_EXACT_ALARM` permission (Android 12+, explicit grant) — add the request flow with a clear explanation of why it's needed.

7. Zynkbot confirms in chat/voice when the timer is set ("Timer set for 10 minutes") — the confirmation is the "did this actually work" signal for the user.

8. Test background-execution edge cases end-to-end: wake word fires while app is backgrounded → timer gets set → phone is locked → timer still fires correctly.

**Sequencing note:** Ship wake-word + timer as one working slice before expanding to other voice commands. Once this slice works reliably, other intents (reminders, memory creation via wake word, etc.) reuse the same wake-word service and the same intent-parsing pattern — cheap to add after this foundation exists.

**Effort estimate:** Medium overall. Timer logic itself is small (a day or so); wake-word detection integration is well-solved (days); the bulk of real time goes to background-service reliability and battery testing on actual Android hardware — budget more time for testing than for the initial build.

---

## v1.0 — Stable Release: Desktop + Android (Q3 2026)

**Focus:** Ship what's built. Most core features are complete. This version closes the remaining gaps and promotes Android from internal testing to public release.

### What's Complete

- Multi-model chat (Anthropic Claude, OpenAI, xAI Grok, Mistral, local Ollama)
- Semantic memory system with relationship graph, contradiction detection, namespace support
- ZynkSync cross-device sync with mTLS authentication
- ZynkLink peer-to-peer file sharing
- ZChat device-to-device messaging
- Ensemble mode (parallel multi-model queries with synthesis)
- Knowledge Base with RAG (PDF support, full-text search)
- Snap-in architecture (Therapist proof-of-concept)
- Containment modes (Guardian, HIPAA, Child, Sovereign — proof-of-concept layer)
- Conversation history with hash-chain integrity
- Memory Manager UI
- Camera/image input on Android
- Android app on Play Console (internal testing)
- Voice dictation via OpenAI Whisper on Android (`graphene-dictation` branch)

### Remaining for v1.0

- **Cloud backup (R2)** — test `feature/cloud-backup-r2` branch end-to-end, merge to main. Essential for users and for development (delete and restore memories cleanly).
- **Vosk offline dictation** — required before `graphene-dictation` merges to main. OpenAI Whisper is the interim implementation; Vosk replaces it with fully on-device transcription. See Voice Input section below.
- **Play Store public release** — promote from internal testing to production track.
- **Write-time memory consolidation** — multi-turn conversations produce near-duplicate memories. Extend the existing relationship-detection LLM call to return a three-way decision (fresh fact / rephrasing / elaboration) and skip or overwrite redundant memories. Zero new API calls. High impact, low cost.
- **Scroll-to-bottom button on Android** — floating ↓ button when user scrolls up in a long conversation; auto-dismisses at bottom.
- **ZynkSync TLS handshake log spam** — downgrade `HandshakeFailure` from known paired IPs to `debug!`; add connection-attempt debounce on the Android client side.
- **Android push notifications for ZChat** — post a system notification with tone when a ZChat message arrives while the app is backgrounded.
- **ZynkLink mTLS cert exchange** — ZynkLink and ZChat routes currently use request-level auth (`check_zynklink_authorized`) rather than verified TLS certificates. Fix: exchange certs during `handle_zynklink_verify_code` / `handle_zynklink_accept_code`, store in `zynk_devices.tls_cert_der`, widen `rebuild_http_client` filter to include link-only peers, move ZynkLink/ZChat routes behind `require_verified_device`. Known security gap — must ship before v1.0.

### Technical Debt (deferred to v1.1+)
- **Real safety classifier** — current `toxic-bert` false-positives on clinical/grief language. Replacement options: Llama Guard 3 (GGUF via existing llama-cpp-2 bindings), LLM-delegated classification, or trust primary model refusals for adult modes. Thresholds raised to suppress false positives for now.
- **Memory identity merge on first sync** — memories on the receiving device are not adopted into the synced namespace on first sync (KI-011). Requires identity merge step during first-sync handshake.
- **Rotating startup tips** — replace static tip with a pool that surfaces less-discovered features.
- **Zynkbot self-documentation refresh** — KB seed docs have drifted from current features; missing Android support, Mistral, ensemble parallel mode, v0.9.x additions.
- **Relationship classifier prompt bias tuning** — over-picks `contradicts` for multi-valued attributes (phones, hobbies), under-picks `elaborates`. Fix is prompt engineering + examples in `lib.rs::ask_llm_about_memory_with_relationships`.
- **Knowledge Base indexing progress indicator** — large document indexing gives no feedback; add progress bar.
- **Startup date reminder surfacing** — check for memories with dates in the next 7 days at launch and surface in the greeting. Memory system already extracts dates; this is a query + greeting injection.
- **Cross-device conversation history sync** — ZynkSync syncs the memory graph only; raw conversation log stays local. Requires `entry_hash` deduplication, `prev_hash` chain handling, and FTS5 index maintenance on receive.
- **Multiple contradiction resolution** — only the first conflict is surfaced when a new memory contradicts multiple existing memories. Show resolution modal for each conflict in sequence.
- **Word document (.docx) support in Knowledge Base**
- **GPU conflict warning scope: desktop only** — suppress the Ensemble GPU conflict warning when `window.AndroidPaths` is present.
- **Model Management UI** — download/delete .gguf models from UI with progress indicators.
- **Auto-update notification** — detect new version and prompt to update.

---

## v1.1 — Parenting Mode (Q4 2026)

**Focus:** First domain-specific feature expansion. Zynkbot as a family companion — safe AI interactions for children, family file sharing, and a parental review layer.

**Scope of Child Mode:** Content filtering at the Zynkbot layer — controlling what the AI will discuss, not device-level lockdown. Device-level MDM (factory reset protection, app blocking) is a separate product track requiring Android Enterprise Device Owner provisioning and is out of scope for this release. What Parenting Mode delivers is a private, safe AI companion that a child can trust to grow with them, while giving parents visibility into AI interactions and control over content boundaries.

### Child Mode Enhancements

- **Content filtering overhaul** — move beyond the current proof-of-concept safety layer to a purpose-built child-appropriate classifier. Replace or supplement `toxic-bert` with a model that understands context: clinical/educational discussion of sensitive topics is allowed; glorification is not. Tuned specifically for child-appropriate interaction.
- **Age-appropriate response calibration** — system prompt and memory extraction adapt to the child's configured age range (e.g. 8–12, 13–17). Language complexity, topic framing, and example selection shift accordingly.
- **Parental Dashboard** — review Zynkbot interactions, view content that was filtered, adjust sensitivity settings. Access gated behind the parent's Zynkbot instance via ZynkSync pairing.
- **Parent/child ZynkSync pairing flow** — dedicated pairing UX for parent-child device pairs, distinct from the peer-to-peer sync flow. Parent receives a summary digest rather than full memory access, preserving the child's private relationship with their Zynkbot.
- **Educational Reports** — learning progress tracking and exportable summaries.
- **Custom topic controls** — parent-defined allow/block list for subjects; allow-list mode for younger children.

### Family File Sharing

- **Family ZynkLink group** — extend ZynkLink to support named family groups. Files (homework, photos, documents) shared into the family group are accessible to all paired family members.
- **Homework flow** — child shares a document from their device; parent's Zynkbot can see it, comment on it, or run KB search against it.
- **Family ZChat** — group chat across all family-paired devices; message history visible to all members.

### Notes on Scope

The existing containment modes (HIPAA, Guardian, Sovereign, Child) are proofs of concept demonstrating the safety-filter architecture. They are not production-grade for their respective domains. Parenting Mode is the first mode that will be developed to a production standard, as the first paid offering. The others will follow in subsequent releases as their respective use cases are validated.

---

## v1.2 — ZynkSync & ZynkLink Modularization (Q4 2026)

**Focus:** Refactor the sync and linking layers into clean, composable internal modules before adding Android-native inference, industry snap-ins, or SDK surface area.

ZynkSync and ZynkLink are currently tightly coupled to the desktop context. Before expanding to mobile-native local inference, industry-specific snap-ins, or the public SDK, these layers need stable internal interface contracts — otherwise each new platform inherits the same coupling. This is a prerequisite milestone, not a feature release.

### Modularization Scope

- **ZynkSync module** — Extract sync protocol, conflict resolution, and namespace filtering into a standalone internal crate with a defined API; decouple from desktop-specific file paths and UI hooks.
- **ZynkLink module** — Separate file-sharing transport, peer discovery, and permissions model; formalize the `ShareSource` trait that abstracts over desktop direct-fs and Android SAF (prerequisite for Play Store compliance).
- **Interface contracts** — Document the stable boundary each module exposes; these become the base the SDK Foundation builds on.
- **Test coverage** — Unit and integration tests against extracted modules.

### ZynkLink Enhancements

- **Android share-sheet target** *(small, high impact)* — Register Zynkbot as a share target in Android's share sheet so users can share any file directly into ZynkbotShare from any app. Add `<intent-filter>` for `ACTION_SEND` / `ACTION_SEND_MULTIPLE` with `*/*` MIME type; handle incoming `content://` URI in `MainActivity.kt` by copying to ZynkbotShare. Small Kotlin addition, no Rust changes.

- **Android Storage Access Framework (SAF) migration** *(Play Store blocker)* — The current build relies on `MANAGE_EXTERNAL_STORAGE` to see files placed in `Downloads/ZynkbotShare/` by other apps. Google Play restricts this permission heavily. Migration to SAF is the path to Play Store compliance.

  **Architecture change:** Introduce a `ShareSource` trait in Rust with two implementations — `tokio::fs` direct (desktop, Zynkbot-owned folders) and a Kotlin bridge (Android 11+ SAF-granted folders). All call sites in `scan_directory`, `handle_zynklink_download`, and the peer-download writer switch to the trait. First-launch flow gains a SAF folder-picker prompt on Android 11+.

  **Scope:** ~200–300 lines Rust, ~150 lines Kotlin, ~50 lines JS. 2–3 focused days including cross-device testing. Own branch.

  **Interim:** `MANAGE_EXTERNAL_STORAGE` is declared and requested at first launch. Works for GitHub-distributed builds; documented as KI-015. Must be swapped for SAF before any Play Store submission.

- **File Upload** — Send files TO paired devices (not just download); requires write permission.
- **Streaming File Transfer** — Replace in-memory file buffering with chunked streaming. Current behavior loads the entire file into RAM on both ends before transfer; works but fails on large models when available RAM is close to file size. Fix: HTTP chunked transfer encoding, constant memory regardless of file size. Resumable transfers (HTTP Range) as stretch goal.
- **Federated Knowledge Base Query** *(post-Android-launch)* — Allow a paired user to query another user's KB in place, without documents transferring. Source documents never leave the owner's device; only top-k relevant passages are returned. Requires per-document sharing consent, query log visible to KB owner, and revocation per-peer or per-document.

---

## v1.3 — Android-Native AI + SDK Foundation (Q4 2026)

**Focus:** Three co-primary tracks: on-device AI research for Android, SDK Foundation groundwork, and companion/networking depth.

### Android On-Device AI Research

Local model inference on phones is a distinct architectural problem from desktop GGUF/llama.cpp. CPU-only GGUF models miss device NPUs entirely. Investigation needed:

- **Apple (iOS/macOS):** Core ML — Apple's Neural Engine runs Core ML models natively. Conversion tooling (coremltools) exists for common model families.
- **Qualcomm Snapdragon (Android):** QNN/ONNX — Snapdragon's Hexagon NPU is the dominant Android AI accelerator. Qualcomm AI Hub provides pre-optimized models.
- **Google Tensor (Android):** TFLite / LiteRT — Google's in-house NPU on Pixel devices.
- **Fallback:** GGUF/llama.cpp CPU inference — correct and functional, slower and battery-intensive. Universal baseline while platform-native paths are evaluated.
- **Memory quality bridging** — When the phone connects to a larger model (home server or user-approved API), queued locally-extracted memories should be re-evaluated by the larger model before permanent storage.

**Goal:** Identify which format(s) to target; ship CPU-path GGUF as baseline while NPU investigation continues.

### sqlite-vec: Indexed Vector Search

Current memory search is a linear scan — correct and fast at typical usage; degrades at very large memory counts. sqlite-vec adds approximate nearest-neighbor indexing so search stays fast regardless of memory count. Applies to both desktop and mobile.

### SDK Foundation

Early groundwork for the developer platform. Full public SDK is v3.0; v1.3 establishes internal architecture so the surface area is stable before external exposure.

- **Define clean internal API boundaries** — identify Rust modules that become SDK-facing (memory system, containment layer, ZynkSync protocol, safety classifier). Ensure each has a clear interface contract.
- **Snap-in architecture hardening** — finalize data contract and lifecycle hooks so third-party snap-ins can be built against a stable interface.
- **Documentation-first approach** — write the SDK developer guide before public release.
- **CLI scaffold for snap-in development** — basic tooling for creating, testing, and packaging a snap-in locally.

### User Profile & Companion Enhancements

- **User profile update mechanism** — add a profile editor (Settings or Memory Manager) to update name, age, and other onboarding fields after the fact. Future fields: date of birth (derive age automatically), timezone, occupation, preferred language, pronouns.
- **Push notification reminders** — full OS-level reminders via `tauri-plugin-notification`. User sets lead time; reminders fire even when the app is minimized. Requires background scheduler and per-platform notification permission handling. Builds on startup date surfacing (v1.0).
- **Emotional State Awareness** — lightweight sentiment/distress classification on user input before the main LLM call. Adjust response framing based on detected state. Builds continuity across sessions via elaborates/causes chains.
- **Per-User Tone Adaptation** — learn and match individual communication style over time. Store tone preferences derived from feedback and conversation patterns. Adjust formality, verbosity, and directness per user. Stored locally.
- **Atomic fact extraction with elaborates-linking** *(deferred from v0.9)* — switch from one MEMORY_EXTRACT per message to one per distinct fact, each with focused embeddings. Co-extracted facts auto-link with `elaborates`. Defer until retrieval issues are observed in practice.
- **Camera/image support in Ensemble mode** — route image attachments through Phase 1 so vision-capable models each analyze the image; non-vision models skip or receive a text description.

### Conversation History Enhancements

- **"What Did I Learn?" Digest** — periodic summary view of what you got out of your conversations. Weekly and monthly views; digest entries link back to source conversations; topic grouping.
- **Thread Branching Chart** — visual diagram (git-branch style) showing where a conversation went off-topic and how it returned.
- **Memory ↔ Conversation Linking** — bidirectional link between semantic memory and raw conversation log. In Memory Manager: "Source conversations" on each memory. In history: annotation showing which memories were extracted.
- **Resumed Session Summarization** — auto-summarize earlier turns into a compact brief for very long conversations; inject as system-level context followed by the most recent N turns verbatim.
- **Feedback Log Viewer** — read path for the `message_feedback` table. `get_feedback_log` command: JOIN with `conversation_messages`, return rated responses with text, model, and timestamp. `FeedbackLogPanel.jsx`: summary stats, list of rated responses. "Feedback" button next to "History."
- **Export** — session to JSON and plain text/Markdown.

### ZynkSync Enhancements

- **Namespace Filtering UI** — checkbox in ZynkSync settings to select which namespaces sync. Database and backend filtering already support this; UI controls in ZynkSyncPanel.jsx needed.
- **is_syncable Checkbox** — per-memory control in Memory Manager UI. Database has `is_syncable` column (default true); add checkbox to edit form.
- **Sync Conflict Viewer** — UI to review past conflict resolutions.
- **Selective Device Sync** — choose which paired devices receive which namespaces.

### ZChat Enhancements

- **Group Chat** — multi-device group messaging with named groups and history.
- **File Attachments** — send files via ZChat (integrated with ZynkLink).
- **Message Search** — full-text search across past ZChat messages.

### Ensemble Enhancements

- **User-selectable coordinator model** — currently auto-selected (Anthropic → xAI → OpenAI → local); allow manual designation. Critical: coordinator's training biases shape synthesis framing — two coordinators can reach opposite verdicts from identical responses.
- **Per-question model presets** — save favorite model combinations for specific use cases.

### Security

- **ZynkLink/ZChat mTLS Device Authentication** — exchange certificates during ZynkLink pairing; move file-transfer and messaging routes behind verified-client-certificate middleware.
- **Audit Logging** — exportable logs for all network operations.
- **Network request limits** — per-connection body size cap, concurrent-request limit, and hard timeouts to prevent a paired device from exhausting host resources.
- **OS Keychain / Android Keystore for API keys** — migrate from plaintext `.env` to the OS credential store (macOS Keychain, Windows Credential Manager, Linux Secret Service, Android Keystore).
- **Prompt injection defense** — clearly delimit and label memory content as untrusted data in prompts; LLM-extracted memories containing instruction-like text must not override system-prompt behavior.
- **Memory provenance and confidence** — store source conversation, extraction timestamp, model used, and confidence indicator per memory; enables tracing incorrect memories and auditing extraction quality.

### Knowledge Base Enhancements

- **GPU/CUDA acceleration for embeddings** — offload sentence-transformer embedding model to GPU during document indexing; CPU fallback remains.
- **Word document (.docx) support** *(carried from v1.0 deferred list)*

### Snap-in Enhancements

- **Therapist snap-in note export** — export session notes, insights, and conversation excerpts to plain text, Markdown, or PDF.

---

## v1.4 — Advanced Containment Modes (Q1 2027)

**Focus:** Production-ready containment for specialized use cases. The modes introduced as proofs-of-concept in v0.9 graduate to their full implementation.

### HIPAA Mode

- **AI-Based PHI Detection** — replace regex with a specialized model. Target 95–99% accuracy (current regex: 70–85%). Contextual understanding ("my social is 219907812" caught, not just "SSN: 219-90-7812"). Local inference only.
- **Audit Cryptography** — tamper-proof audit logs with digital signatures and append-only structure.
- **Role-Based Access Control** — multi-user HIPAA deployments (physician, nurse, admin roles).
- **BAA Template** — Business Associate Agreement template and compliance documentation.

### Guardian Mode Enhancements

- **Sovereign Mode: Crypto Integration** — read-only blockchain queries and transaction explanation (never sign transactions).
- **DeFi Safety** — phishing pattern detection, suspicious contract warnings, rug pull heuristics.
- **Filtering** — AI-generated content detection (text, image, audio); focus on allowing Zynkbot to function as a user-controlled content filter.

### Parenting Mode MDM Track *(if validated by v1.1 adoption)*

Device-level parental controls — factory reset protection, app-level restrictions — require Android Enterprise Device Owner provisioning. This is architecturally distinct from the Zynkbot-layer content filtering shipped in v1.1. If family package adoption warrants it, this track investigates Device Owner provisioning feasibility, GrapheneOS compatibility, and Play Store policy implications before committing to an implementation. Not scheduled; gated on v1.1 validation.

---

## v1.5 — ContainAI Services (Q2 2027)

**Focus:** Opt-in services for users who want them, without compromising the local-first architecture.

*Note: Cloud memory backup ships in v1.0. This version covers additional cloud service offerings.*

### Extended Cloud Services

- **Encrypted memory backup tiers** — extended retention, cross-device restore history, priority restore bandwidth.
- **Family sync service** — cloud relay for family-paired devices that aren't on the same LAN.

---

## v2.0 — Advanced Memory Features (2027)

**Focus:** Enhanced memory capabilities and intelligence.

### Memory Decay & Re-surfacing

- Priority scoring for memories
- Automatic re-surfacing of relevant old memories
- "You mentioned this 6 months ago..." reminders
- Intelligent forgetting (reduce clutter from trivial memories)

### Threaded Recall

- Timeline view of related memories
- Story mode: "My journey with X" (e.g., "Career doubts: 2023–2025")
- Automatic thread detection and memory clustering by topic

### Mode-Based Memory Gates

- Memories accessible only in certain modes
- Child mode hides sensitive content
- Work namespace hidden in personal mode

### MemoryVault — Tamper-Evident Conversation Log

Hash chain integrity layer on top of the v1.0 conversation history tables. The basic log exists; this adds cryptographic tamper-evidence.

- Populate `entry_hash` and `prev_hash` columns (schema stubs already in v1.0 tables)
- Hash chain verification command — detects if any record has been modified
- Opt-in per user; export with hash chain intact for audit purposes

**Use cases:** Legal defensibility, clinical accountability, compliance audit trail for regulated industries.

### Advanced Conversation History UI

- **Topic Timeline** — group conversations by detected subject/theme with swim-lane visualization showing frequency and evolution over time.
- **Spatial Canvas View** — conversations as nodes on a 2D canvas (Obsidian/Miro style), AI auto-clustered by topic. *Highest complexity — design-phase item.*
- **"Past-Self" Queries** — ask Zynkbot questions about your own history: "What was I asking about in February?"

---

## v2.5 — Pattern Recognition & Self-Reflection (2027)

**Focus:** Help users understand themselves.

### Self-Analysis Tools

- Emotional tone tracking over time
- Decision-making pattern analysis
- Behavior drift alerts ("You used to do this daily")
- Self-contradiction markers ("This conflicts with April 2026")
- Personal hypocrisy detector

### Pattern Drift Analyzer

- Mood timeline visualization
- Value alignment view (actions vs stated principles)
- Habit formation/breaking tracker
- Life event correlation (mood shifts during major events)

### Intent-Outcome Tracker

- "I said I'd do X" → Did you actually do X?
- Goal consistency measurement and proactive reminders
- Accountability partner mode

### MirrorPath Snap-in

- Dedicated self-reflection workspace with AI-assisted journaling
- Pattern visualization dashboard; exportable reports (for therapy, coaching)

---

## v3.0 — SDK & Developer Platform (2027)

**Focus:** Enable third-party development.

### Zynkbot SDK

#### Core Modules
- **Containment Layer** — consent-based safety framework with pluggable modes and custom rules
- **Memory System** — hybrid semantic + entity search, relationship detection, namespace support
- **ZynkSync Protocol** — cross-device synchronization with conflict resolution framework
- **HIPAA Framework** — PHI detection, memory system disable, audit logging, compliance helpers
- **MemoryVault** — hash-chained conversation history log (tamper-evident, opt-in, local)
- **GDPR Framework** — right to erasure, data portability export, breach notification workflow
- **Snap-in Architecture** — domain-specific workspaces

#### Developer Tools
- SDK documentation and tutorials
- Code examples and templates
- Testing framework and debugging tools
- CLI for snap-in development

#### Snap-in Marketplace
- Snap-in discovery and installation
- Developer accounts and publishing
- Revenue sharing model with quality assurance and vetting
- User reviews and ratings

### Licensing Model
- ✅ Free for permitted community uses
- 💰 Paid commercial licensing (tiered pricing)
- 🤝 Revenue supports community source development

---

## v3.5 — Professional Snap-ins (2027–2028)

**Focus:** Domain-specific applications built on the SDK.

### Healthcare
- **Patient Portal** — PHI-aware conversation logging
- **Symptom Tracker** — medical history with privacy controls
- **Medication Reminder** — AI-assisted medication management
- **Therapy Journal** — HIPAA-friendly session notes

### Education
- **Homework Helper** — child-safe tutoring
- **Study Planner** — adaptive learning schedules
- **Research Assistant** — citation management, note-taking
- **Language Tutor** — conversational practice

### Professional
- **Legal Assistant** — case notes, client tracking
- **Financial Advisor** — budget tracking, investment notes
- **Project Manager** — task scaffolding, team collaboration
- **Writing Coach** — long-form writing assistant

### Personal
- **Fitness Tracker** — workout logging, nutrition advice
- **Relationship Manager** — gift ideas, important dates
- **Travel Planner** — itinerary builder, trip memories

---

## v4.0 — Foundation & Ecosystem (2028)

**Focus:** Long-term sustainability.

### ContainAI Foundation

#### Structure
- Nonprofit incorporation (501(c)(3) or equivalent)
- Board of directors establishment
- Governance model (community input)
- Grant programs for aligned developers

#### Revenue Model
- 💰 Zynkbot premium snap-ins
- 💰 SDK commercial licensing
- 🤝 Corporate sponsorships (privacy-aligned companies)
- 🎁 Individual donations and grants

#### Programs
- Security audit funding
- Privacy research grants
- Community project support
- Educational initiatives

### Partner Integrations

- **Proton** (VPN, encrypted email, calendar)
- **Signal** (secure messaging)
- **Tutanota** (encrypted email)
- **Jitsi** (video conferencing)
- **Nextcloud** (file storage)

---

## Research & Experimental Features

*Unscheduled — requires research, external dependencies, or community interest.*

### Advanced Networking

- **Mesh Networking** — device-to-device communication without any network infrastructure.

  **Why this matters:** Zynkbot's current networking requires a shared local network. There are situations where even a hotspot isn't viable: active infrastructure suppression, remote environments, or contexts where creating a WiFi network draws attention. Bluetooth and WiFi Direct allow two devices to communicate directly without any intermediary.

  **Target platform:** Primarily mobile (Android/iOS). Meaningful once Mobile is production-ready.

  **Planned capabilities:**
  - Bluetooth pairing for ZynkLink contact exchange (replaces IP-based pairing)
  - ZChat delivery over Bluetooth when no network is available
  - ZynkLink file transfer over Bluetooth or WiFi Direct
  - Mesh relay: messages hop between devices to reach a destination out of direct range

  **Prior art:** Briar (open source) demonstrates this architecture works for secure messaging under network suppression.

- **Tor Integration** — anonymous remote sync over Tor hidden services (experimental, performance trade-off).
- **Sneakernet Mode** — USB-based sync for air-gapped environments; export/import memory snapshots.

### AI Enhancements

- **On-Device Fine-Tuning** — LoRA adapters trained on user's conversations (never leaves device).
- **Multimodal Models** — local CLIP for image embeddings, image memory retrieval, visual QA.
- **Hallucination Detection** — confidence scoring, source verification.
- **Explainable AI** — step-by-step reasoning display.

#### Epistemic Humility / Refuse-on-Contradiction

**Status:** Research spike, not scheduled. Post-1.0.

**Concept:** A reasoning-integrity layer that detects when a query or its supporting memory context is internally contradictory, or when available evidence is insufficient, and responds with a structured "I can't answer this confidently — here's the conflict / here's what's missing" instead of collapsing into a hallucinated answer.

**Why it fits:** Zynkbot already has the two primitives this needs:
- Contradiction detection over the memory graph.
- Ensemble mode — independently-trained models disagreeing is already a usable uncertainty signal.

**Why deferred:** Non-blocking, post-launch, and needs measurement before it's worth building. A concrete failure class must be defined and a test set assembled before any code is written.

### Personal Ethics Layer

- **Custom Constraint Flags** — user-defined guardrails ("don't help me lie unless in Sovereign Mode").
- **Value Alignment Scoring** — actions vs stated principles.
- **Ethical Dilemma Advisor** — explore decision consequences.

### Crisis Support

- **Crisis Companion Mode** — grounded language, breathing exercises, resource directory.
- **Panic Attack Assistant** — guided calming techniques.

### Privacy Innovations

- **Homomorphic Encryption** — search encrypted memories without decrypting (research phase).
- **Federated Learning** — collaborative model improvement without data sharing.

### Secure Vault (Passwords & Financial Info)

A local-first, LAN-synced credential vault built on Zynkbot's existing architecture. No third-party servers at any layer.

**Why this fits:** The infrastructure is already in place — encrypted SQLite storage, TLS-secured LAN sync, ED25519 device pairing. The philosophical stance already matches the ideal password-manager posture.

**What makes this hard:** Separate encryption subsystem (Argon2id, AES-256-GCM, `zeroize`), new threat model (device-unlocked compromise), autofill integration per platform (Android Autofill Framework, iOS Credential Provider, browser extensions), regulatory surface if card numbers are stored, and strict last-writer-wins sync (conflicts on passwords are dangerous, unlike memory conflicts).

**Why deferred:** Far future. Warrants a dedicated security audit and separate design document. Not before v2.0 at the earliest.

---

## Platform Expansion

- **iOS** — Tauri Mobile for iPhone/iPad (App Store limitations and background sync challenges to navigate).
- **Web Version** — browser-based, API-only, limited offline capability; use case: Chromebook, public computers.
- **Raspberry Pi** — headless mode for a dedicated home server; low-power 24/7 family or team instance.
- **Enterprise / multi-user deployment** — a separate server-based, centralized-database implementation exists nearly complete from before the SQLite migration. If enterprise interest emerges, finishing it is a targeted effort, not a ground-up build.

---

## Long-Term Vision (2028–2030)

**Privacy-first AI at scale**
- Demonstrable alternative to surveillance capitalism at scale
- Privacy-first AI becomes a mainstream expectation, not a niche

**SDK as the Standard for Privacy-First AI**
- Healthcare, education, legal, and finance sectors with compliance frameworks built on transparent architecture
- Active ecosystem of third-party applications

**Self-Sustaining Ecosystem**
- Revenue from premium snap-ins, SDK licensing, and donations
- Active developer community
- Partnerships with privacy-focused organizations (EFF, Signal Foundation, etc.)

---

## What We Won't Build

**To maintain focus, we explicitly won't:**
- ❌ Engagement optimization (anti-user)
- ❌ Data selling or sharing (anti-ethics)
- ❌ Proprietary lock-in without source visibility
- ❌ Dark patterns or manipulation (anti-transparency)
- ❌ Central servers for sync or storage
- ❌ Telemetry or analytics

---

## Community Involvement

**How to contribute:**
- Open GitHub issues for feature requests
- Comment on existing roadmap items
- Submit pull requests for features you want to build
- Join discussions about priorities

**Contact:** matt@containai.ai
**GitHub:** https://github.com/MSkill1/zynkbot

---

## Versioning Philosophy

- **v0.x** — Pre-1.0 development (current)
- **v1.x** — Stable releases: core companion + domain snap-ins
- **v2.x** — Advanced memory features
- **v3.x** — SDK and developer platform
- **v4.x** — Foundation and ecosystem

Each major version focuses on a specific theme. Minor versions add features within that theme.

---

*This roadmap is a living document. Priorities may shift based on community feedback, technical discoveries, and resource availability.*

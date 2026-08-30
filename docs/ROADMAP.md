# Zynkbot Development Roadmap

**Last Updated:** August 2026
**Current Version:** v0.9.5-beta1 (Android + Desktop open beta)

This roadmap outlines planned features and enhancements. Timelines are estimates and subject to change based on community feedback and development priorities.

---

## v0.9.5 — Offline Voice + Wake Word + Timer

**Status:** Vosk offline dictation (Part 0) merged to main as v0.9.5-beta1. Parts 1–3 (wake word, timer, voice memory query) not yet started.

**Goal:** Fully offline voice input on all platforms (Linux, Android), "Hey Zynk" wake-word detection, and a working timer/alarm as the first concrete wake-word action. Also includes voice memory query ("What do I know about X?").

### Part 0 — Vosk offline dictation (prerequisite, ships with v0.9.5)

- Replace OpenAI Whisper (API-dependent) with [Vosk](https://alphacephei.com/vosk/) for fully on-device transcription on all platforms.
- **Linux/Desktop:** Use the `vosk` Rust crate (C FFI wrapper). The existing mic button routes through a new `transcribe_audio` Tauri command. **This fixes dictation on Linux** — previously broken because it required the OpenAI API; Vosk works fully offline.
- **Android:** Vosk Java SDK via Kotlin bridge in `MainActivity.kt`, same interface as `AndroidPaths`. Replaces the current Whisper API call on Android.
- Vosk model (~50MB English) shown as a downloadable item in the model management UI alongside the three LLM models. Same model file works on all platforms.
- ⚠️ **Known issue — Windows:** Vosk offline dictation is not yet working on Windows. The `vosk` Rust crate requires a platform-native Vosk shared library (`libvosk.dll`) that is not yet bundled in the Windows build. OpenAI Whisper (cloud) remains the fallback on Windows. Fix tracked for a future patch.
- ⚠️ **Known issue — Memory ID missing on Android:** When opening a memory in the Memory Manager on Android, the memory ID number is not displayed. The ID is shown correctly on desktop. Likely a conditional render or CSS issue in the memory detail view.
- Android Enter key no longer sends — Enter = line break on Android (send via button only), preserving multi-paragraph input. Desktop behavior unchanged.

### Part 1 — Wake-word detection

1. Integrate an on-device wake-word engine (e.g. Porcupine/Picovoice, or an open alternative) trained on "Hey Zynk" — check current licensing terms for custom wake words before committing (free tiers often limit custom phrases).

2. App-level only — NOT an OS modification. No GrapheneOS/AOSP fork, no system-level audio hook. Works identically on stock Android and GrapheneOS.

3. Implement as an Android background/foreground service:
   - Default OFF — always-listening should be opt-in, not default (matches the project's privacy posture).
   - On detection: brief chime/visual confirmation (so the user knows it heard them), then capture the following speech via Vosk.

4. Battery/lifecycle testing — this is the real risk, not the detection model itself:
   - Service survives being backgrounded, isn't killed by Android's process management.
   - Measure real battery drain over a full day of always-listening.
   - Confirm behavior when the phone is locked.

### Part 2 — Timer/alarm (the first wake-word action)

5. Add a lightweight intent-check step: after wake-word capture (or from any typed/dictated input), ask the model whether the message is a timer/alarm request and extract the duration as structured output (e.g. `{"is_timer": true, "duration_seconds": 600}`).

6. Android: implement via `AlarmManager` (`setExactAndAllowWhileIdle` or `setAlarmClock`) + completion notification with sound, fires even when backgrounded. Requires `SCHEDULE_EXACT_ALARM` permission (Android 12+) — add explicit grant request flow.

7. Zynkbot confirms in chat when the timer is set ("Timer set for 10 minutes").

8. Test background-execution edge cases: wake word fires while backgrounded → timer set → phone locked → timer still fires correctly.

### Part 3 — Voice memory query

9. Intent: "Hey Zynk, what do I know about X?" — extract the subject, run a memory search, speak + display the top results.
   - Reuses the same intent-parse step from Part 2.
   - Speak response via TTS (Android TTS API / desktop speech synthesis).
   - Displayed in chat as a normal assistant turn so the answer is browsable.

### Part 4 (future) — "Send" voice command in wake-word flow

After wake-word recording, support "send" or "send that" as a terminal command to submit the transcribed text without the silence-detection auto-send. Requires distinguishing the send command from the message body mid-stream, which adds non-trivial parsing complexity. Deferred until the silence-detection flow is stable and user-tested.

**Sequencing note:** Ship Vosk (Part 0) first — it's a prerequisite for reliable wake-word + dictation. Then wake-word + timer as one working slice. Voice memory query follows using the same infrastructure.

**Effort estimate:** Part 0 (Vosk) is the bulk of the engineering (platform wiring). Parts 1–3 are medium; the real time cost is background-service reliability and battery testing on Android hardware.

### Pre-release gate — Wake-word 5-minute limit investigation

The wake-word service appears to stop detecting after ~5 minutes of inactivity. Currently attributed to "Android hardcoding a security limit" — this is not accurate, since wake-word inference is fully on-device with no remote attack surface. Likely candidates: Android Doze/background-service throttling, or a deliberately-written cooldown in the app's own code. Investigate the actual code path (WakeWordService, PARTIAL_WAKE_LOCK lifecycle, AudioRecord behavior under Doze) before treating it as an OS constraint. May overlap with "Bug D: Android wake word does not trigger with screen off" — diagnose together.

### Pre-release gate — Power audit (Android)

Before v0.9.5 ships, profile the wake word service's battery impact on a real device with the screen on and no active conversation:

- **Tool:** Android Studio Energy Profiler (CPU, radio, and wakelock tabs) while `WakeWordService` is running.
- **Measure:** CPU% from ONNX inference loop (3 models, 12.5 cycles/sec); `AudioRecord` wake-lock hold cost; background baseline vs. service-active delta.
- **Acceptance bar:** Wake word service should account for <5% battery drain per hour of idle listening. If the inference loop is the culprit, quantize or swap to INT8 ONNX models. If `AudioRecord` is the culprit, investigate whether increasing `CHUNK_SAMPLES` (trading latency for CPU budget) reduces drain without meaningfully hurting detection latency.
- **GrapheneOS note:** Run the same audit on GrapheneOS — its background scheduling may throttle the service differently. If GrapheneOS-specific behavior makes the service unreliable or dramatically worse on battery, escalate before deciding whether GrapheneOS remains a supported target.

---

## v0.9.6 — Obsidian Vault Integration

**Status:** Not started.

**Goal:** Export Zynkbot's memory graph as a human-readable, navigable Obsidian vault — making the "closer in spirit to Obsidian than to a chatbot" comparison literal and working.

**Why simpler than any API-based integration:** Obsidian has no API and no server — a vault is just a folder of plain Markdown files on disk, watched and rendered by Obsidian's UI. Zynkbot reads/writes directly into that folder using ordinary filesystem operations, the same pattern already used for the local knowledge base. Nothing talks to Obsidian's internals, so there is no risk of Obsidian updates breaking the integration.

**Design decision:** Zynkbot's SQLite memory graph is the single source of truth. The Obsidian vault is a generated, human-readable projection of it — not a second canonical store. Rich relationship metadata (confidence, type, timestamps) goes in YAML frontmatter; the note body stays clean prose with `[[wikilinks]]` for relationships.

### Scope

**Do first within this release — simpler than Proton Calendar/Drive (no Android intents, no file-format hand-off protocol) and more thematically central to the product's positioning.**

1. **Vault path setting** — user points Zynkbot at their vault folder (or a subfolder, e.g. `Vault/Zynkbot/`).

2. **One-way export (v1):** each memory becomes an individual `.md` file.
   - Filename: memory title, sanitized for filesystem safety.
   - Frontmatter: id, namespace, created date, confidence, source.
   - Body: memory content in prose.
   - Relationships rendered as `[[wikilinks]]` with type noted inline (e.g. "*(contradicts)* [[Lives in New York]]").

3. **Sync trigger:** export on memory creation/update, OR a manual "Export to Obsidian" action — decide based on whether continuous writes while Obsidian has the vault open causes file-lock friction in testing.

4. **Voice/dictation path:** "note this in Obsidian" — dictate → write directly as a new `.md` file. Simpler than any Proton hand-off since there's no OS intent involved, just a file write.

**Effort estimate:** Small for one-way export — mostly formatting/templating work on data Zynkbot already has.

**Two-way sync (v2, later):** watch the vault folder for edits and reflect meaningful changes back into Zynkbot's memory store. More complex (conflict resolution needed) — explicitly deferred past the first version. File as its own future roadmap item.

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
- Offline voice dictation via Vosk on Linux and Android (v0.9.5-beta1)

### Pre-v1.0 Decision — Monetization, distribution, and the zero-trust constraint

This needs to be resolved before Play Store launch. The core tension:

**Zynkbot's identity is zero-trust** — the user's API key goes directly to Anthropic/OpenAI. Matt never touches conversation data. This is not a promise; it is a technical guarantee. A government subpoena to Matt returns nothing. A breach of Matt's infrastructure exposes nothing.

**The distribution problem** is that "bring your own API key" is friction that blocks most mainstream users. The natural solution — a proxy where Matt holds a shared API key and bills users via Stripe — technically puts Matt in the middle of every conversation. Even if the proxy is stateless and never logs, it receives plaintext messages at the network layer before forwarding. That breaks the zero-trust guarantee and contradicts Zynkbot's core positioning. Users would have to trust Matt, not just their chosen AI provider. Given Zynkbot's privacy-first audience (GrapheneOS users, etc.), this reputational cost is real and legal exposure is non-zero.

**Why "opt-in transparency" is not sufficient** — even clear disclosure ("by subscribing, your messages route through Zynkbot's servers") doesn't eliminate the liability. The question isn't whether users consent; it's whether Matt wants to be in a position where he *could* be compelled to produce conversation data.

**Options ranked by zero-trust preservation:**

1. **Guided API key onboarding** *(zero-trust preserved)* — walk users through getting their own key during first-run setup. Zynkbot links to the provider's key page, user pastes it in, done. Reduces friction without introducing a middleman. Can be polished into a one-minute flow. This is the recommended path for v1.0.

2. **Signal-style transparent proxy** *(partial)* — proxy code is open source (Zynkbot already is), stateless with no logging, reproducible server builds, public audit policy. Reduces trust requirement but does not eliminate it — the proxy still receives plaintext. Signal can do this because their E2E encryption means the server never sees message content even in transit. Zynkbot would need a similar cryptographic design to make this claim honestly, which is a significant architecture investment.

3. **Subscription funds user's own key** *(zero-trust preserved, operationally complex)* — user subscribes, Matt's backend programmatically creates a key in the user's name on their provider account (OpenAI supports this via Projects API; Anthropic does not). User's key, user's account, Matt never in the loop. Operationally messy and provider-dependent. Worth revisiting if OpenAI's API key management matures.

4. **Shared proxy with full opt-in disclosure** *(zero-trust broken, maximum friction removed)* — only viable if offered as a clearly-labeled convenience tier alongside the zero-trust self-key path. Never the default. Not recommended for v1.0.

**Recommended resolution for v1.0:** Ship with polished guided key onboarding (option 1). Revisit subscription model post-launch once there is user feedback on how much the key setup actually blocks adoption in practice.

### Remaining for v1.0

- **ZynkSync outbox architecture** — Replace the current state-sync approach (which requires manual wiring for every new feature) with an event-driven outbox pattern. Every write to any memory-related table inserts a record into a `sync_outbox` table (operation, table, row ID, payload, timestamp). A background process drains the outbox to all paired devices. Offline devices accumulate entries until reconnect. Deletes become soft-deletes (`deleted_at` column) so they propagate as outbox events rather than disappearing silently. New features get sync automatically just by writing to the DB — no per-feature sync wiring needed. Conflict resolution: last-write-wins by timestamp covers ~95% of cases. Outbox TTL: prune entries older than N days; devices offline longer do a full re-sync. This permanently fixes the "clear on one device doesn't propagate" class of bugs and is the prerequisite for all future ZynkSync enhancements. Estimated effort: 2–3 days including schema migration and cross-platform testing.

- **Voice/dictation/audio modularization** — Extract the wake word, Vosk dictation, TTS, and conversation loop logic from App.jsx into a dedicated module/hook. Currently entangled with component state in ways that make the feature set hard to extend. Do after the voice feature set is stable (v0.9.5).

- ~~**Cloud backup (R2)**~~ ✅ — merged to main (v0.9.4). Encrypted R2 backup includes memories + conversation history. Tombstone-safe restore propagates to peers.
- ~~**Vosk offline dictation**~~ ✅ — merged to main (v0.9.5-beta1). Linux (cpal + vosk crate) and Android (Kotlin bridge) both ship. Wake-word, timer, and voice memory query deferred to a later v0.9.x release.
- **Play Store public release** — promote from internal testing to production track.
- **Write-time memory consolidation** — multi-turn conversations produce near-duplicate memories. Extend the existing relationship-detection LLM call to return a three-way decision (fresh fact / rephrasing / elaboration) and skip or overwrite redundant memories. Zero new API calls. High impact, low cost.
- **Scroll-to-bottom button on Android** — floating ↓ button when user scrolls up in a long conversation; auto-dismisses at bottom.
- **ZynkSync TLS handshake log spam** — downgrade `HandshakeFailure` from known paired IPs to `debug!`; add connection-attempt debounce on the Android client side.
- **Android push notifications for ZChat** — post a system notification with tone when a ZChat message arrives while the app is backgrounded.
- **ZynkLink mTLS cert exchange** — ZynkLink and ZChat routes currently use request-level auth (`check_zynklink_authorized`) rather than verified TLS certificates. Fix: exchange certs during `handle_zynklink_verify_code` / `handle_zynklink_accept_code`, store in `zynk_devices.tls_cert_der`, widen `rebuild_http_client` filter to include link-only peers, move ZynkLink/ZChat routes behind `require_verified_device`. Known security gap — must ship before v1.0.
- **Photo attachments in chat** — attach images directly in the chat input bar (camera capture or gallery picker on Android; file picker on desktop). Image passed to vision-capable models (Claude, GPT-4o, Gemini) as base64 or URL; non-vision models receive a text notice. Lays groundwork for camera/OCR integration in later versions.
- **Copy full chat history** — single button to copy an entire conversation to the clipboard as plain text. Per-message copy already exists; this covers the whole session. (Reported by galbicka, issue #5.)
- **Save to memory conversationally** — detect "remember this" / "save that" phrasing in user input and route directly to memory storage without a full LLM round-trip. Faster, cheaper, and more intuitive than the current flow.
- **Licensing transition** — relicense from the custom Zynkbot Community Source License to AGPL-3.0 (public/F-Droid distribution) paired with a commercial license for businesses wanting to avoid AGPL copyleft. Prerequisites: (1) dependency license audit (`cargo license` + `npm audit`) to confirm no GPL-incompatible deps before committing to AGPL; (2) CLA in place before any outside PRs are merged. Gate on pre-v1.0 checklist.
- **Pre-submission code audit** *(final gate before v1.0 ships)* — Three passes: (1) **Structural** — dead code, unused imports, TODO/FIXME comments, inconsistent naming; run `cargo clippy` and ESLint clean; (2) **Security** — API key handling, network input validation, anything crossing a trust boundary; (3) **Professionalism** — module-level doc comments on major Rust files, consistent error handling patterns, no commented-out debug blocks. Primary target is App.jsx (most accumulated complexity); Rust side expected to be cleaner. Required before F-Droid submission and any external code review.

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

## v1.1 — Parenting Mode + Proton Orchestration (Q4 2026)

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

### Proton App Orchestration — Calendar + Drive

Zynkbot orchestrates Proton's own official apps via OS-level hand-off (Android intents, standard file formats) rather than integrating with a Proton API — Proton has no stable public API for Calendar/Drive, and unofficial client libraries are too fragile to build on.

- **Calendar:** extract event details via LLM intent parsing (same pattern as timer feature), generate a `.ics` file or fire an intent opening Proton Calendar's "create event" screen pre-filled. One tap to confirm.
- **Drive:** "save this note to Drive" — write into a Drive-synced local folder if one exists, or hand off via Drive's own upload UI.
- Not Proton-exclusive: same approach works with any calendar/drive app that handles standard intents.
- Password manager (Proton Pass) integration explicitly out of scope.

### Notes on Scope

The existing containment modes (HIPAA, Guardian, Sovereign, Child) are proofs of concept demonstrating the safety-filter architecture. They are not production-grade for their respective domains. Parenting Mode is the first mode that will be developed to a production standard, as the first paid offering. The others will follow in subsequent releases as their respective use cases are validated.

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

- **Typed memory classification** *(exploration item)* — Zynkbot's memories are currently flat (a single `content` string with `namespace` and relationship edges). Claude Code's internal agent-memory system uses discrete *types* (user profile, behavioral feedback, project state, reference pointers), each with structured `Why:` and `How to apply:` fields that help the agent decide when to surface a given memory. Worth exploring whether Zynkbot could classify extracted memories into analogous types at write time — e.g. distinguishing a biographical fact ("has two nephews") from a stated preference ("prefers concise replies") from a relationship fact ("niece Emma's birthday is March 4") — and use the type to gate injection: biographical facts injected when the user asks identity questions, preferences injected always, relationship facts injected when the named entity appears in the conversation. Contrast with current behavior: all memories compete on cosine similarity alone, so injection is entirely retrieval-score-driven with no semantic role differentiation. This is an architecture exploration, not a bug fix; defer until retrieval quality issues motivate it.

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
- ~~**ZynkSync mTLS Device Authentication** — Sync-paired devices present their certificate during the TLS handshake; the server verifies it against the paired-device database. Pairing-bootstrap routes remain open; sync, Ollama proxy, and API-key propagation routes require a verified certificate.~~ ✅
- **ZynkLink/ZChat mTLS Device Authentication** — Exchange certificates during ZynkLink pairing and move file-transfer and messaging routes behind verified-client-certificate middleware. These routes currently use their separate pairing records and request-level authorization.
- **Audit Logging** — Comprehensive exportable logs for all network operations (who synced what, when)
- **Network request limits** — Per-connection body size cap, concurrent-request limit, and hard timeouts to prevent a paired device from exhausting CPU, RAM, or bandwidth on the host
- **OS Keychain / Android Keystore storage for API keys** — Currently stored in a plaintext `.env` file; migrate to the OS credential store (macOS Keychain, Windows Credential Manager, Linux Secret Service, Android Keystore) so keys are protected at rest even if the filesystem is accessible to another process
- **Prompt injection defense** — Memory content injected into prompts should be clearly delimited and labeled as untrusted data; LLM-extracted memories containing instruction-like text must not be able to override system-prompt behavior
- **Memory provenance and confidence** — Store the source conversation, extraction timestamp, model used, and a confidence indicator for each extracted memory; enables tracing incorrect memories back to origin and auditing extraction quality over time

### Ensemble Enhancements
- **User-selectable coordinator model** — Currently auto-selected (Anthropic → xAI → OpenAI → local); allow user to manually designate which model acts as coordinator. Critical: the coordinator's training biases shape how the synthesis frames consensus and uncertainty — two coordinators can reach opposite verdicts from identical responses. For sensitive or contested questions, coordinator selection is not cosmetic.
- **Per-question model presets** — Save favorite model combinations for specific use cases (e.g. "research" preset, "creative" preset)

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
- **Calendar integration** — read-only access to the device calendar so Zynkbot can answer "what's on my calendar?" and incorporate upcoming events into memory/context. On Android: `READ_CALENDAR` permission + ContentProvider query via Kotlin bridge. On desktop: OS calendar API (iCal on macOS, EWS/iCal on Linux/Windows). Voice query "Hey Zynk, what's on my calendar?" reuses the wake-word intent framework from v0.9.5. Scope: read-only first; write (add event) is a separate feature.
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

### UI Internationalization (i18n)

- **UI text translation (buildable now):** extract hardcoded frontend strings into an i18next-based system with per-language JSON files. Toggle in settings, auto-detect system locale as default.
- **Voice/dictation language models:** Vosk offers non-English models; on-demand download per language, same pattern as existing model downloads.
- **Full NER/entity-extraction localization (deferred):** bert-base-NER is English-only; a genuinely multilingual pipeline is a real research project. Defer until there is evidence of demand for a specific language.

### Snap-in Enhancements

- **Therapist snap-in note export** — export session notes, insights, and conversation excerpts to plain text, Markdown, or PDF.
- **Google account management snap-in** — orchestrate Google account traffic on the LAN: one device acts as the coordinator, routing which device uses which Google account. Uses Android account management intents and LAN discovery via existing ZynkLink infrastructure. Useful for households with multiple Google accounts across multiple devices.

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
- **Delta sync for large model files** — for syncing GB-scale local LLMs between devices that already have a previous version, transfer only changed portions (rsync/zsync principle) rather than the full file. Only helps on updates between devices that already share a baseline; first-time transfer still requires the full file. Low priority; most relevant to the sneakernet/rural-deployment use case.

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

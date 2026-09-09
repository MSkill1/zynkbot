# Changelog

This file documents notable changes to Zynkbot from the initial public release forward.

For the full commit history, see [GitHub](https://github.com/MSkill1/zynkbot/commits/main).

---

## [Unreleased] — 0.9.6-beta1 (in progress) <!-- draft by Claude 2026-09-09, review wording -->

### Android voice
- Every "Hey Zynk" now runs natively through the Android assistant role: chime, Z overlay, tap-Z-to-cancel, Stop, replies spoken with the built-in voice and joined to the current thread. The old in-app wake path is gone.
- Timer, alarm and stopwatch by voice, handled by the clock app without a model call; stopwatch falls back to opening the clock app.
- Fewer false triggers: silence gate, back-off after repeated fruitless firings, faster close when nobody speaks, a default-deny instruction for TV and fragments, and a personal verifier trained from the owner's own clips (enforcing on the developer's phones; off with clip collection on for testers).
- The Vosk/Whisper selector applies to hands-free too; Whisper falls back to Vosk.
- "Remember colon …" saves a fact word for word by voice; Vosk's misspellings of "colon" are accepted.
- Chime and spoken replies use the media volume.

### Memory
- Memories carry an event date, a category from a fixed list, tags, tone and named entities from the same call that decides whether to remember; older memories are annotated in the background once.
- "About me" report in the Memory Manager, read from the local database; every line opens its memory; tap a tag to filter.
- Memories from hands-free turns are marked and listed separately.
- Explicit Remember is stored even when it contradicts an older memory; a contradiction no longer blocks it.
- One timestamp format everywhere; duplicate conversation rows removed and prevented.

### Knowledge base
- Fabrication guard: the model is told what the search found (real matches, weak matches, nothing) and the reply header says when an answer is not grounded; no memory is extracted from an ungrounded reply.
- Android: Add files through the system picker; "All files access" no longer requested.

### App
- Report a problem: a local text report (version, build, device, masked log tail, optional conversation) from System Controls or under any reply.
- Pinned threads; the current thread is restored after a restart; renamed peers show their new name; a peer's changed address is learned from its own requests; the backup key travels with "Push to all devices".
- 16 KB page-size alignment (Vosk 0.3.75, ONNX Runtime 1.29); full-screen-intent and photo permissions removed; Google Play readiness plan in the roadmap.

---

## [0.9.5-beta1] — 2026-08-23 — Offline Voice Dictation + Stability Fixes

### Highlights
- Offline voice dictation via Vosk ships on Linux (cpal mic capture) and Android (Kotlin bridge + bundled model)
- OpenAI Whisper (cloud) available as an alternative voice engine, selectable in Settings
- Android cold-start black-screen eliminated
- ZynkSync restart crash (port reuse) fixed

### Features
- **Vosk offline dictation (Linux)** — `vosk_desktop.rs` + `cpal` mic capture; transcription runs on-device with no internet required
- **Vosk on Android** — Kotlin bridge in `MainActivity.kt` with bundled `vosk-android` AAR; same command surface as desktop
- **Voice engine selector** — Settings dropdown chooses between Vosk (offline) and OpenAI Whisper (cloud)
- **Cloud backup (R2/S3)** — AES-256-GCM encrypted backup includes memories, conversation history, and embeddings; passphrase-derived key; tombstone-safe restore propagates to sync peers

### Bug Fixes
- **ZynkSync restart crash** — `SO_REUSEADDR` on `TcpListener::bind` (port 57963); rapid stop/restart no longer fails with "address already in use"
- **Android black-screen cold-start** — CSS `html, body, #root { background: #181a20 }` prevents flash before React mounts; React `needsSetup === null` guard shows a loading screen; Android `windowBackground` theme attribute set to match app background

### Internal
- Vosk dependency moved to `[target.'cfg(target_os = "linux")'.dependencies]` — Windows CI no longer tries to link `libvosk.lib`
- `libasound2-dev` added to Linux CI apt-get step (required by `cpal`)
- `build.rs` rpath/link-search block gated to `#[cfg(target_os = "linux")]` — MSVC no longer rejects `-Wl,-rpath` flags
- Release workflow: `prerelease` field set automatically from tag name (`beta`/`alpha`/`rc` suffix → prerelease)

---

## [0.9.3] — 2026-08-01 — Ensemble, Model Picker, and Sync Fixes

### Highlights
- Ensemble Mode overhauled with parallel execution and improved synthesis
- Mistral added as a 4th API provider
- Per-provider model picker — choose which model each provider uses
- Several ZynkSync correctness fixes that were silently breaking contradiction detection and deletion propagation

### Features
- Mistral API support (alongside Anthropic, OpenAI, and xAI)
- Per-provider model picker in Settings — select the specific model for each API provider
- New conversation button
- Ensemble Mode Phase 1 now runs all models in parallel (previously sequential)
- Ensemble Mode Phase 2 synthesis improved: better consensus detection, tighter memory injection, API fallback if local model fails
- GPU conflict guard in Ensemble modal — warns when Custom/Ollama and a local GGUF are both selected (shared CUDA device)
- Live-verified model lists for all four providers
- ZynkLink file visibility and own-share UX polish

### Bug Fixes
- **Contradiction resolution deletion not propagating** — resolving a contradiction by keeping the new memory deleted the old one locally but never notified paired devices; deletion now propagates immediately via ZynkSync
- **Real-time deletion missing tombstone timestamp** — delete-by-hash requests sent to peers were missing `deleted_at`, so the recreation guard on the receiving side could never fire; timestamp now included in all real-time deletion payloads
- **Ensemble local models grayed out on desktop** — production build check incorrectly disabled local GGUF models on all production builds, not just Android; desktop release builds now correctly allow local model selection
- **Memory decision API calls rejected by newer models** — `temperature` parameter sent to all providers was deprecated in newer models (claude-sonnet-5, gpt-5.5, grok-4.5), causing silent 400 errors and fallback failures; removed from all four provider helpers

### Internal
- Android CI: fixed NDK toolchain not on PATH during OpenSSL cross-compilation
- Android CI: fixed `sdkmanager` not found (setup-android action added before NDK install step)

---

## [0.9.2] — 2026-07-25 — Android Phase 1

### Highlights
- Android beta — full-featured app (API models + local ML stack) for Android phones and tablets
- Cross-device sync now works between Android and desktop over encrypted LAN
- Production release signing wired in; APK available on the GitHub release page

### Android
- App identifier set to `ai.containai.zynkbot` (matches Play Console draft)
- Full local ML stack runs on Android via Candle (embeddings, NER, safety classifier)
- ZynkSync, ZynkLink file sharing, and ZChat all functional on Android
- `MANAGE_EXTERNAL_STORAGE` requested at launch on Android 11+ for ZynkbotShare folder visibility
- `WRITE_EXTERNAL_STORAGE` requested on Android ≤ 9 (API 28)
- Foreground service starts correctly on Android 8 (API 26) with version-conditional notification channel
- `ZynkbotShare` folder created at `Downloads/ZynkbotShare/` on first launch

### Bug Fixes
- Fixed ZynkLink pairing showing "Remote Device XXXXXXXX" — acceptor now sends its device name during handshake
- Fixed tablet startup crash caused by permission request flow in `onCreate`
- Fixed Open in Files app crashing on Android 8 (external storage URI format invalid below API 29)
- Fixed `getShareDir()` and `openShareFolder()` crashing the app when called before storage permission granted

### UI
- Per-device expel button (×) in ZynkSync Synced Devices list — removes a device and notifies peers
- ZynkSync pairing code display no longer wraps mid-code on narrow screens

### Internal
- Release APK signed with production RSA-4096 keystore; signing config loaded from `keystore.properties` (gitignored)
- Repo-wide `.gitignore` patterns for `*.jks`, `*.keystore`, `keystore.properties`

---

## [0.9.0] — 2026-07-13 — First Public Release

First public release of Zynkbot as an open source project.

### Highlights
- Local-first AI assistant with persistent semantic memory
- Pure Rust/Tauri desktop app — no Python runtime required
- Supports local GGUF models, OpenAI, Anthropic, and xAI APIs
- Cross-device memory sync (ZynkSync) and peer-to-peer file sharing (ZynkLink)
- Device-to-device messaging (ZChat) with no cloud relay
- Conversation history with search and session resume
- Knowledge Base with RAG — index your own documents for semantic search
- Containment modes: Guardian, Child, HIPAA, Sovereign, Witness
- Multi-model Ensemble Mode with consensus detection
- Runs entirely on your machine — no telemetry, no phone-home

### Features
- Web search result links open in the system browser
- Image attachment support: JPG, PNG, GIF, WebP, BMP — routed to the vision API of the active cloud model
- ZynkSync pause/resume broadcasts to all paired devices instantly
- First-run setup wizard automatically downloads all required AI models
- Contradiction modal resolution propagates memory deletions to sync peers
- Session ID visible alongside User ID and Device ID in identity panel

### Security
- LAN traffic encrypted with TLS 1.3 (self-signed certificates, automatically trusted on pairing)
- Sync endpoints reject requests from unsynced devices
- Unsync propagates automatically to the peer device
- Pairing code rate limiting: 5 attempts per 5-minute window per IP

### Bug Fixes
- Fixed contradiction modal keep new/keep old buttons being swapped
- Fixed `original_text` not preserved when memory stored via contradiction resolution (KI-012)
- Fixed `original_text` not included in ZynkSync payloads (KI-013)
- Fixed contradiction modal crash on first memory conflict detection
- Fixed Anthropic streaming token counter always showing 0
- Fixed child mode system prompt not injected into OpenAI API calls
- Fixed sync never transferring memories when auto-sync was disabled
- Restored `Remember:` command for forcing memory storage
- Fixed `remove_api_key` not finding the `.env` file
- System memories no longer appear in user hybrid search results (KI-003)

### Internal
- `lib.rs` broken into 9 domain command modules for maintainability
- Verbose debug output gated behind `#[cfg(debug_assertions)]`
- CPU-only mode forced for embeddings and safety classifier

See [docs/FEATURES.md](docs/FEATURES.md) for the full feature list.

---

## Contributing to this changelog

If you're submitting a pull request, add a line to `[Unreleased]` describing what changed.
When a version ships, unreleased entries move under a new version heading with a date.

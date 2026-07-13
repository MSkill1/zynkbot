# Changelog

This file documents notable changes to Zynkbot from the initial public release forward.

For the full commit history, see [GitHub](https://github.com/MSkill1/zynkbot/commits/main).

---

## [Unreleased]

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

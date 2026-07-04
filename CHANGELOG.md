# Changelog

This file documents notable changes to Zynkbot from the initial public release forward.

For the full commit history, see [GitHub](https://github.com/MSkill1/zynkbot/commits/main).

---

## [Unreleased]

- Conversation history sync across paired devices via ZynkSync

### Security
- LAN traffic between ZynkSync and ZynkLink devices is now encrypted with TLS 1.3 (self-signed certificates generated on first run, automatically trusted on pairing)
- **Breaking change for existing paired devices**: both devices must be updated to this version simultaneously. A device on the old plain-HTTP build cannot communicate with one on the new TLS build. After updating both devices, re-establish the pairing normally — no data is lost.
- Sync endpoints now reject requests from devices that have been unsynced, preventing re-insertion after removal
- Unsync now propagates automatically to the peer device via push notification
- Fixed: unlink and unsync now fully clean up both sides, including chat history and orphaned device records
- Fixed: auth errors from sync handlers now return proper HTTP 4xx status codes instead of 200 OK

---

## [1.0.0] — Initial Public Release

First public release of Zynkbot as an open source project.

### Highlights

- Local-first AI assistant with persistent semantic memory
- Pure Rust/Tauri desktop app — no Python runtime required
- Supports local models (.gguf), OpenAI, Anthropic, and xAI APIs
- Cross-device memory sync (ZynkSync) and file sharing (ZynkLink)
- Conversation history with search and session resume
- Knowledge Base with RAG — index your own documents for semantic search
- Containment modes: Guardian, Child, HIPAA, Sovereign, Witness
- Runs entirely on your machine — no telemetry, no phone-home

See [docs/FEATURES.md](docs/FEATURES.md) for the full feature list.

---

## Contributing to this changelog

If you're submitting a pull request, add a line to `[Unreleased]` describing what changed.
When a version ships, unreleased entries move under a new version heading with a date.

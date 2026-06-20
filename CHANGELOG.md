# Changelog

This file documents notable changes to Zynkbot from the initial public release forward.

For the full commit history, see [GitHub](https://github.com/MSkill1/zynkbot/commits/main).

---

## [Unreleased]

- Conversation history sync across paired devices via ZynkSync

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

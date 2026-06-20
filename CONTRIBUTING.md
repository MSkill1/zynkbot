# Contributing to Zynkbot

Thank you for your interest in contributing to Zynkbot!

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Reporting Security Vulnerabilities](#reporting-security-vulnerabilities)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Project Philosophy](#project-philosophy)

---

## Code of Conduct

- Be respectful and constructive
- No harassment, personal attacks, or discriminatory language
- Focus on the work, not the person

---

## Reporting Security Vulnerabilities

**Please do not report security vulnerabilities through public GitHub issues.**

Email **matt@containai.ai** with a description of the issue, steps to reproduce, and the affected component. You will receive a response within 72 hours. Confirmed vulnerabilities will be prioritized and credited in the release notes unless you prefer anonymity.

---

## Getting Started

### Ways to Contribute

1. **Report Bugs** — Open an issue with details
2. **Suggest Features** — Open an issue tagged `enhancement`
3. **Improve Documentation** — Fix typos, clarify instructions
4. **Write Code** — Fix bugs or implement features
5. **Review PRs** — Help review other contributions

### Before You Start

- **Read the CLA** — By submitting a PR you agree to the [Contributor License Agreement](CLA.md)
- **Search existing issues** — Your idea might already be discussed
- **Open an issue first** — For large changes, discuss approach before coding

---

## Development Setup

### Prerequisites

- Rust 1.77.2+ (`rustup` recommended)
- Node.js 18+ and npm
- SQLite (handled by installer)

### Setup

```bash
git clone https://github.com/MSkill1/zynkbot.git
cd zynkbot
sudo ./install.sh
cd zynkbot_rust
npm run tauri:dev
```

The installer handles the database schema, environment configuration, and Node dependencies. See [Linux Installation Guide](docs/LINUX_INSTALLATION_GUIDE.md) or [Windows Installation Guide](docs/WINDOWS_INSTALLATION_GUIDE.md) for platform-specific details.

---

## Module Map

Where to find things in the Rust backend:

| What you want to change | File |
|---|---|
| Chat, memory pipeline, conversation flow | `src-tauri/src/lib.rs` |
| Memory CRUD, links, graph, contradictions | `src-tauri/src/commands/memory.rs` |
| Onboarding flow, Einstein demo | `src-tauri/src/commands/onboarding.rs` |
| Conversation history, session feedback | `src-tauri/src/commands/conversation.rs` |
| Entity extraction, NLP utilities | `src-tauri/src/commands/nlp.rs` |
| API key management, model discovery | `src-tauri/src/commands/models.rs` |
| Containment modes, safety classifier | `src-tauri/src/commands/safety.rs` |
| Memory storage engine, hybrid search | `src-tauri/src/memory.rs` |
| Prompt construction, context assembly | `src-tauri/src/conversation_engine.rs` |
| Cross-device memory sync | `src-tauri/src/zynksync.rs` |
| Peer-to-peer file transfer | `src-tauri/src/zynklink.rs` |
| Device-to-device messaging | `src-tauri/src/zchat.rs` |
| Local GGUF model inference | `src-tauri/src/llm/local_models.rs` |
| Safety filtering (toxic-bert) | `src-tauri/src/safety_classifier.rs` |
| Knowledge base RAG | `src-tauri/src/kb_rag.rs` |

---

## Making Changes

### Branch Naming

```
feature/add-safety-classifier-thresholds
bugfix/fix-memory-leak-in-sync
docs/improve-installation-guide
refactor/simplify-containment-layer
```

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

**Examples:**

```
feat(zynksync): add TLS support for device-to-device sync

Implements TLS encryption for ZynkSync communication using
self-signed certificates with pinning.

Closes #123
```

```
fix(memory): prevent duplicate entries in hybrid search

The entity-based search was returning duplicates when multiple
entities matched the same memory. Added deduplication logic.

Fixes #456
```

---

## Pull Request Process

### Before Submitting

- Code compiles without errors (`cargo build`)
- Tests pass (`cargo test`)
- Code is formatted (`cargo fmt`)
- Linter passes (`cargo clippy -- -D warnings`)
- If you added or modified any `sqlx::query!` macros, run `cargo sqlx prepare` from `zynkbot_rust/src-tauri/` with a live database and commit the updated `.sqlx/` cache — without this, offline builds will fail
- Documentation updated if needed
- CHANGELOG.md updated if applicable

### PR Description

Include:
- What does this PR do and why?
- How was it tested and on which platforms?
- Any breaking changes and how users should migrate?
- Before/after screenshots for UI changes

### Review Process

1. At least one maintainer approval required
2. Maintainers may test on different platforms
3. Squash and merge (unless multiple commits are meaningful)

---

## Code Style

### Rust

- Run `cargo fmt` before submitting
- Address all `cargo clippy` warnings
- Avoid `.unwrap()` or `.expect()` without a comment explaining why it's safe
- Use `Result<T, E>` for functions that can fail
- Document public APIs with `///` doc comments

### JavaScript/React

- 2 spaces for indentation
- Semicolons
- Single quotes for strings
- Trailing commas in objects/arrays

---

## Testing

Run the existing tests with:

```bash
cargo test
cargo test -- --nocapture   # with output
cargo test test_name        # specific test
```

Unit tests live in `#[cfg(test)]` modules at the bottom of each source file, following standard Rust convention. If you add a new feature or fix a bug, adding a test for it is appreciated.

---

## Documentation

- Update `docs/` for architecture or feature changes
- Update `CHANGELOG.md` for any user-facing change
- Documentation-only commits use the `docs:` type (see commit format above)

---

## Project Philosophy

Zynkbot is privacy-first and offline-first. Any contribution should be consistent with these principles:

- The core app collects no telemetry, analytics, or usage data of any kind
- The baseline chat experience works entirely offline with a local model — no account or API key required. Some optional features (such as Child mode) require an API key, but the core should never depend on one
- Features should work without an internet connection where possible
- Snap-ins may collect or transmit data only with explicit, informed user consent and must clearly document exactly what is collected and where it goes
- Be explicit about what data, if any, leaves the user's machine

---

## License

By submitting a contribution, you agree to the terms of the [Contributor License Agreement](CLA.md). In short: you keep your copyright, and you grant ContainAI the right to include your contribution under any compatible license (including future versions). See [CLA.md](CLA.md) for full terms and [LICENSE](LICENSE) for the project license.

---

**Thank you for contributing to Zynkbot.**

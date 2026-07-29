# Android Development Plan

> **Status (July 2026):** Phase 1 is complete on the `android-phase1` branch. API-model chat, the full local ML stack (embeddings, NER, safety classifier), ZynkSync, ZynkLink, and ZChat are all functional on Android. The remainder of this document is the original implementation plan — preserved for context. See `ROADMAP.md` for Phase 2 (on-device LLM inference, SAF migration).

## Strategy: Two-Phase Approach

### Phase 1 — API + Local Supporting Models (Early Access)

Ship an Android app that uses API models (Claude, GPT-4, Grok) for LLM inference while running the full local supporting stack on-device. This gets Zynkbot on Android quickly without tackling the hardest technical problem (on-device LLM inference) first.

**What runs locally on Android (via Candle, pure Rust):**
- Embeddings — all-MiniLM-L6-v2 — memory search works correctly
- NER — bert-base-NER — entity extraction works
- Safety classifier — toxic-bert — containment layer works

**What uses API:**
- LLM inference (Claude, GPT-4, Grok — same API choices as desktop)

**Result:** This is not a stripped-down version. Memory system, ZynkSync, ZynkLink, ZChat, and the containment layer all function correctly. The only thing that's API-dependent is the chat model — which is already an option users choose on desktop.

### Phase 2 — On-Device LLM Inference

Add local LLM support via llama.cpp for Android NDK. This is the complex part:
- llama.cpp NDK compilation (CMake + Android toolchain)
- GGUF model loading on mobile (4-8GB models, limited RAM)
- ARM memory management and thermal throttling
- Optional: GPU acceleration via Vulkan/OpenCL

Phase 2 is independent of Phase 1 — Phase 1 ships first and remains the baseline.

---

## Key Technical Facts

**Why the small models work on Android but LLMs don't (yet):**
Candle is a pure Rust ML framework. Rust cross-compiles cleanly to `aarch64-linux-android` without NDK complexity. The BERT-based models (embeddings, NER, safety) use standard transformer operations that Candle implements in pure Rust — no platform-specific acceleration required.

llama.cpp is a C++ library that requires Android NDK compilation, complex CMake setup, and GPU backend integration (Vulkan for Android). That's the hard part deferred to Phase 2.

**Why not ONNX Runtime:**
Earlier exploration found ONNX Runtime introduced cross-compilation and dependency issues. Candle avoids this entirely — it's the framework already in use for all three supporting models.

**Framework:**
Tauri v2 has Android support and can reuse the existing Rust backend (memory system, SQLite, ZynkSync, networking). The React frontend is also reusable. This avoids a full rewrite in React Native.

---

## Framing / Optics

Zynkbot's privacy-first positioning is not compromised by Phase 1:
- The desktop app already ships API models alongside local models — this is not new
- Conversation data is sent to the chosen API provider (Anthropic, OpenAI, xAI), not to Zynkbot servers
- All memory storage, search, and sync remain local
- Local-only users can wait for Phase 2

Suggested positioning: **"Early Access — API models now, on-device models coming."**

---

## Prerequisites Before Starting

- RC11 binary tested and merged to main (current blocker)
- Tauri v2 Android target set up (`cargo tauri android init`)
- Android SDK + NDK installed
- Decision on minimum Android API level (recommend API 26 / Android 8.0+)

---

## Open Questions — Resolved in Phase 1

1. ✅ **Embedding model size on mobile** — ~22MB bundled in the APK; acceptable.
2. ✅ **ZynkSync on mobile** — Works. Memory sync, including edits and deletes, propagates between Android and desktop devices in real time.
3. ✅ **Background sync** — Handled via a foreground service (`SyncForegroundService.kt`); keeps the sync loop alive while the app is backgrounded.
4. ✅ **App distribution** — APK sideload for Phase 1 Early Access; Google Play Store planned for Phase 2.

See `ROADMAP.md` for Phase 2 open questions (on-device LLM inference, SAF folder picker).

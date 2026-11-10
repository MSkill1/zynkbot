# Android Development Plan

> **Historical plan (2025 to August 2026), kept for context.** This is the original Android implementation plan. It does not describe the app as it is today. Phase 1 shipped. Phase 2 shipped in 0.9.6 as an on-device GGUF model running on the CPU; OpenCL was tried and dropped. `ANDROID_ARCHITECTURE.md` describes the app as built. The bracketed notes marked "superseded" or "historical" were added on 2026-11-10 where the plan would mislead a reader about the present. The rest of the body is unchanged.

## Strategy: Two-Phase Approach

### Phase 1 — API + Local Supporting Models (Early Access)

Ship an Android app that uses API models (Claude, GPT-4, Grok) (superseded: Mistral and a custom endpoint are offered too) for LLM inference while running the full local supporting stack on-device. This gets Zynkbot on Android quickly without tackling the hardest technical problem (on-device LLM inference) first.

**What runs locally on Android (via Candle, pure Rust):**
- Embeddings — all-MiniLM-L6-v2 — memory search works correctly
- NER — bert-base-NER — entity extraction works
- Safety classifier — toxic-bert — containment layer works

**What uses API:**
- LLM inference (Claude, GPT-4, Grok — same API choices as desktop)

**Result:** This is not a stripped-down version. Memory system, ZynkSync, ZynkLink, ZChat, and the containment layer all function correctly. The only thing that's API-dependent is the chat model — which is already an option users choose on desktop.

### Phase 2 — On-Device LLM Inference

(superseded: Phase 2 shipped in 0.9.6. The phone runs a GGUF model on the CPU through llama.cpp. An OpenCL backend was built, measured and dropped. There is no Vulkan backend.)

Add local LLM support via llama.cpp for Android NDK. This is the complex part:
- llama.cpp NDK compilation (CMake + Android toolchain)
- GGUF model loading on mobile (4-8GB models, limited RAM)
- ARM memory management and thermal throttling
- Optional: GPU acceleration via Vulkan/OpenCL

Phase 2 is independent of Phase 1 — Phase 1 ships first and remains the baseline.

---

## Key Technical Facts

**Why the small models work on Android but LLMs don't (yet):** (superseded: a local LLM runs on the phone since 0.9.6)
Candle is a pure Rust ML framework. Rust cross-compiles cleanly to `aarch64-linux-android` without NDK complexity. The BERT-based models (embeddings, NER, safety) use standard transformer operations that Candle implements in pure Rust — no platform-specific acceleration required.

llama.cpp is a C++ library that requires Android NDK compilation, complex CMake setup, and GPU backend integration (Vulkan for Android). That's the hard part deferred to Phase 2. (superseded: it shipped on the CPU, with no GPU backend)

**Why not ONNX Runtime:** (superseded in part: the Android app now ships ONNX Runtime, `onnxruntime-android` 1.29.0, for the wake-word models)
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

Suggested positioning: **"Early Access — API models now, on-device models coming."** (superseded: on-device models shipped in 0.9.6)

---

## Prerequisites Before Starting

(historical: this list was written before Phase 1 started)

- RC11 binary tested and merged to main (current blocker)
- Tauri v2 Android target set up (`cargo tauri android init`) (superseded: do not run it again; `gen/android` is committed and hand-edited, see `zynkbot_rust/CLAUDE.md`)
- Android SDK + NDK installed
- `NDK_HOME` exported, pointing at your own NDK — e.g.
  `export NDK_HOME="$ANDROID_HOME/ndk/<version>"`. `cargo tauri android` reads this
  and derives the cross-compilation toolchain from it. The NDK paths are deliberately
  not committed in `zynkbot_rust/.cargo/config.toml`; they used to be hardcoded to one
  machine's absolute path, so Android builds failed anywhere else.
- Decision on minimum Android API level (recommend API 26 / Android 8.0+)

---

## Open Questions — Resolved in Phase 1

1. ✅ **Embedding model size on mobile** — ~22MB bundled in the APK; acceptable. (superseded: the model is not in the APK; the setup wizard downloads it into `models/system`)
2. ✅ **ZynkSync on mobile** — Works. Memory sync, including edits and deletes, propagates between Android and desktop devices in real time. (superseded: not instant; changes leave through an outbox polled every 2 s, and a full sync pass runs every 300 s by default, `zynksync.rs`)
3. ✅ **Background sync** — Handled via a foreground service (`SyncForegroundService.kt`); keeps the sync loop alive while the app is backgrounded.
4. ✅ **App distribution** — APK sideload for Phase 1 Early Access; Google Play Store planned for Phase 2. (superseded: the APK is published on GitHub releases; Google Play is deferred to the 1.0 candidate, `docs/ROADMAP.md`)

See `ROADMAP.md` for Phase 2 open questions (on-device LLM inference, SAF folder picker). (superseded: on-device inference shipped in 0.9.6; `docs/ROADMAP.md` lists what is open now)

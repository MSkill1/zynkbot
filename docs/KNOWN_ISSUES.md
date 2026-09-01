# Known Issues

This file tracks known bugs, edge cases, and rough edges that do not block release but should be fixed in a future update. Contributions welcome — see CONTRIBUTING.md.

---

## Memory Pipeline

### KI-012 — Original text not preserved when memory is stored via contradiction resolution
**Status:** Fixed in this release  
**Affected:** All users — any memory stored after resolving a contradiction modal  
**Description:** The `original_text` field (the verbatim user input) is correctly stored for memories created through the normal path. However, when a contradiction is detected and the user resolves it via the modal, the memory was stored through `store_pending_memory`, which passed `pending.content` (the LLM-extracted fact) as `original_text` instead of the raw user message. Both the Content and Original fields in Memory Manager showed the same extracted text. Fixed by adding `original_text` to `PendingMemory` and threading `bg_message` through the contradiction event payload.  
**Impact:** None — resolved.

---

### KI-013 — Original text not preserved when memory arrives via ZynkSync
**Status:** Fixed in this release  
**Affected:** All users — any memory received from a paired device via ZynkSync  
**Description:** The `original_text` field (the verbatim user input) was not included in the ZynkSync payload. Memories created on one device and synced to another had no `original_text` on the receiving device. Fixed by adding `original_text` to the `SyncMemory` struct, all memory SELECT queries, and the receive INSERT/UPDATE paths in `zynksync.rs`.  
**Impact:** None — resolved. Note: memories synced before this release will still lack `original_text` on the receiving device; only new syncs after upgrading will carry the field.

---

### KI-001 — Double memory on contradiction resolution (edge case)
**Status:** Partially fixed  
**Affected:** All local models  
**Description:** Memory is only stored after the user resolves the contradiction modal — never before. However, in rare cases the background duplicate check may still produce a second copy if the embedding distance between the raw user message and the MEMORY_EXTRACT fact falls between the 0.65 and 0.93 similarity thresholds, causing both to pass the near-duplicate filter.  
**Workaround:** If you see duplicate memories after a contradiction resolution, delete the lower-numbered one — the MEMORY_EXTRACT version is the cleaner fact.  
**Fix target:** Improve near-duplicate search to prefer most-recently-inserted match.

### KI-002 — Contradiction false positive: intention vs. current state
**Status:** Partially mitigated (non-contradiction example added to classifier prompt)  
**Description:** Statements expressing a future intention ("I'm thinking about leaving my job") may occasionally be flagged as contradicting a stored current state ("I work at X"). The classifier prompt includes an example to discourage this, but LLM classification is not deterministic.  
**Workaround:** Select "Not a contradiction" in the modal. No data is lost.

### KI-003 — System memories appearing in user hybrid search
**Status:** Fixed in this release  
**Description:** System memories (user_id = 'system', IDs 1–12) were appearing in user hybrid search results — for example, "Model Support" appeared at ~50% similarity for queries containing common nouns. Fixed by scoping `hybrid_search`, `list_memories`, and the Memory Manager query to exclude `user_id = 'system'` entries.  
**Impact:** None — resolved.

---

## Onboarding

### KI-004 — Onboarding relationship detection skipped on fresh install (fixed)
**Status:** Fixed in this release  
**Description:** `complete_onboarding` reported "no embedding" for all onboarding memories because the `Memory` struct uses `#[sqlx(skip)]` on the embedding field. Embeddings were present in the database but not read by the struct. Fixed by fetching embeddings separately via a raw query.

---

## Local Models

### KI-007 — Uncensored and fine-tuned models may produce lower-quality memory extraction
**Status:** Open / by design  
**Affected:** Uncensored fine-tunes (confirmed: Llama 3.1 8B Lexi Uncensored V2)  
**Description:** Zynkbot's memory pipeline relies on each model following structured extraction instructions precisely — returning only the single new fact introduced in a message. Models fine-tuned for creative or unfiltered output (rather than instruction-following) tend to extract broad context summaries instead of the specific new fact. This causes two downstream problems:

1. **Redundant memories** — the extracted "fact" repeats information already stored from onboarding rather than capturing what's new.
2. **Misdirected contradiction links** — because the extracted fact is a summary of background context rather than the specific claim being corrected, the hybrid search may not surface the most relevant existing memory, causing contradiction relationships to link to the wrong entry.

**Example (Lexi, Q3 test):** User says *"Actually, I've been at Brightline for 4 years, not 3."* Expected extraction: something about tenure correction. Actual extraction: *"Jordan is 31 years old, married to Sarah, and has a 3-year-old daughter named Emma..."* — a family summary unrelated to the correction. The contradiction modal still fired and the correction was stored, but it was linked to the onboarding question memory (Memory 104) rather than the actual tenure fact (Memory 114).

**Impact:** Memory entries may be less precise over time; contradiction links may reference the wrong prior memory. Conversations still function correctly.  
**Workaround:** Use Qwen3 or DeepSeek R1 if memory accuracy is important. Lexi is best suited for creative conversations where long-term memory precision is less critical.  
**Fix target:** No code fix planned — this is a characteristic of the model, not the pipeline.

### KI-008 — Web search trigger is model-dependent on local GGUF models
**Status:** Open / by design  
**Affected:** All local models (varies by model)  
**Description:** Web search requires the model to emit a `WEB_SEARCH_NEEDED:` marker in its response. API models (Claude, GPT-4) do this reliably. Small local models (7B) vary: Qwen3 triggers it consistently; DeepSeek R1 triggers it when the query clearly requires current information; Lexi rarely triggers it and instead gracefully tells the user to search manually.  
**Impact:** Web search may not fire automatically on some local models. The user can ask explicitly, but results depend on the model.  
**Workaround:** Use an API model or Qwen3 if web search reliability matters.  
**Fix target:** No reliable fix without a separate intent-classification model.

---

### KI-014 — Ensemble mode disabled for local models in the CPU binary
**Status:** By design  
**Affected:** Binary (AppImage / deb / rpm) users with local GGUF models  
**Description:** CPU-mode local model inference runs synchronously on the CPU and has no reliable interrupt mechanism. In ensemble mode, a local model that stalls or never produces an end-of-generation token blocks the entire phase indefinitely. To prevent this, local models are disabled in the ensemble model picker in production binary builds.  
**Workaround:** Use API models (Claude, GPT-4, Grok) for ensemble mode. If you need local models in ensemble, build from source — the developer build has no restriction. A CUDA-optimized binary (coming soon) will re-enable local models in ensemble with proper GPU acceleration.  
**Fix target:** CUDA binary release.

---

### KI-005 — Untested models may require prompt format tuning
**Status:** Open / by design  
**Description:** Zynkbot ships with verified optimizations for Qwen3, DeepSeek R1 Distill Llama 8B, and Llama 3.1 Lexi Uncensored V2. Other GGUF models should work but have not been tested. Models using non-standard prompt formats or tokenizer types may produce incomplete or malformed responses.  
**Workaround:** Check `local_models.rs` → `build_prompt_for_model` to add a detection path for your model family.

---

## Networking

### KI-009 — Unsyncing a device also removes the ZynkLink pairing
**Status:** Fixed in this release  
**Affected:** Users who have both ZynkSync and ZynkLink active between the same two devices  
**Description:** ZynkSync and ZynkLink now maintain independent trust relationships via the `sync_paired` column. Unsyncing only clears the ZynkSync pairing; the ZynkLink pairing remains active. Unlinking only clears the ZynkLink pairing; the ZynkSync pairing remains active. Each can be revoked independently without affecting the other.

---

### KI-010 — ZynkLink pairing appeared in the ZynkSync device list
**Status:** Fixed in this release  
**Affected:** Users who established a ZynkLink pairing without a ZynkSync pairing  
**Description:** Establishing a ZynkLink pairing would register the remote device in `zynk_devices` with `is_paired = 1`, causing it to appear in the ZynkSync panel as a paired sync device even though no sync pairing had been established. The `sync_paired` column now tracks sync pairings separately — ZynkLink-only devices no longer appear in the ZynkSync panel.

---

### KI-011 — Pre-existing memories are orphaned after first sync
**Status:** Open  
**Affected:** Users who have existing memories on a device before performing their first ZynkSync with a new partner device  
**Description:** When two devices sync for the first time, memories that already existed on the receiving device before the sync are not automatically merged or associated with the synced identity. They remain as orphaned records in the local database — accessible locally but not part of the synced memory set. New memories created after the first sync are handled correctly.  
**Workaround:** No workaround currently. Orphaned memories remain visible and usable in local conversation but will not propagate to other devices.  
**Fix target:** v1.0 — requires an identity merge step during the first sync handshake to adopt pre-existing memories into the synced namespace.

---

### KI-016 — Memory extraction produces "kitchen sink" summaries
**Status:** Open  
**Affected:** Memory quality overall  
**Description:** When the user says something focused and short (e.g., "I am using a VPN but I want to use a VPN with X, I don't trust it"), the extraction LLM sometimes pulls in unrelated context from earlier in the conversation and produces a single memory containing many unrelated facts (dictation habits, gaming usernames, device names, and the actual new fact all in one memory). This dilutes the memory's embedding across topics, weakens semantic search accuracy, and — as a downstream effect — prevents the relationship-detection pipeline from finding thematic links because low similarity scores filter the memory out of candidate pairs before the LLM classifier ever sees them.  
**Example:** Memory 346 (mentions dictation, iPhone 13, OnePlus 12R, Clash of Clans, VPN) had cosine similarity 0.22 with memory 347 (focused on ProtonVPN configuration) — well below the candidate-pair threshold. The two obviously belong linked but never got classified.  
**Fix target:** Tighten the extraction prompt to keep extracted content close to the current turn's actual new information; explicitly discourage re-stating already-stored facts.

---

### KI-017 — Memory extraction duplicates already-stored facts
**Status:** Open  
**Affected:** Memory quality; contributes to KI-016  
**Description:** The extraction step includes previously stored facts in the output of a new memory rather than treating them as already-known context. Example: memory 346's nearest neighbor is memory 341 at cosine similarity 0.81 — because 346 re-states most of 341 (dictation, iPhone 13, Clash of Clans) alongside the new VPN fact. Near-duplicate memories inflate the DB, weaken retrieval precision, and cause link-detection to waste time on redundant candidates.  
**Fix target:** Extraction prompt should treat retrieved memories as "already known, do not restate — only capture what's new in this turn."

---

### KI-015 — Android scoped storage blocks scan of files created by other apps
**Status:** Workaround in place (MANAGE_EXTERNAL_STORAGE permission); proper fix planned  
**Affected:** Android 11+ devices (API 30+) using ZynkLink file sharing  
**Description:** Files placed into `Downloads/ZynkbotShare/` by apps other than Zynkbot (Chrome downloads, screenshots, files copied via the system file manager, etc.) are invisible to Zynkbot's directory scan due to Android's scoped storage security model. Zynkbot can enumerate files it created itself, but Android's kernel filters foreign-owned files out of the `readdir` result before Zynkbot's code sees them. This means the phone reports "0 files indexed" to a peer that's browsing its share, even when the user can clearly see the file in the Android Files app.  
**Workaround:** The `MANAGE_EXTERNAL_STORAGE` permission is now declared in the manifest and requested at first launch on Android 11+. The user must toggle "Allow access to manage all files" in the settings screen that opens automatically. Once granted, Zynkbot has full raw-filesystem access and the scan works normally.  
**Fix target:** Migrate to a proper Storage Access Framework (SAF) integration for Play Store distribution. `MANAGE_EXTERNAL_STORAGE` is restricted by Google Play to specific allowed use cases (file managers, backup/sync apps) and requires explicit approval during Play Store review. See ROADMAP.md for the SAF migration plan.  
**Impact:** Any Android 11+ user who declines the "All files access" prompt will still be able to send files that Zynkbot itself downloaded, but files they add to ZynkbotShare via other apps will not be visible to peers until they grant the permission.

---

## Debug Logging

### KI-006 — Verbose debug output in development builds
**Status:** Fixed  
**Description:** Several `println!` statements in `lib.rs` and `zynksync.rs` dumped full LLM responses and raw HTTP payloads to the terminal. Gated behind `#[cfg(debug_assertions)]` — silent in release builds, visible in `cargo tauri dev`.

---

## Mobile UI

### KI-018 — ZChat emoji picker overflows the screen on narrow Android phones
**Status:** Open  
**Affected:** Android users tapping the 😊 button in ZChat on phones with narrow screens (~360–411px CSS width)  
**Description:** The emoji picker in `ZChatModal.jsx` renders an inline grid of emoji buttons above the input row. It has no width cap or horizontal scroll container, so on a narrow phone the grid runs off the right edge of the screen — the leftmost emojis are visible but the rest can't be reached because the panel isn't scrollable. The Tab S3 (wider screen) shows the full row and works normally.  
**Fix target:** Two reasonable directions. (a) Constrain the picker to the modal width with `max-width: 100%; overflow-x: auto; flex-wrap: wrap;` and enlarge the touch target — keeps a consistent Zynkbot picker on desktop and mobile. (b) Hide the picker button entirely on Android (`{!isAndroid && ...}` around the 😊 button, same pattern as the VoiceButton fix in v0.9.4 hotfix). Android keyboards already expose a full emoji set via the keyboard's emoji key — duplicating it in-app is redundant and the phone's picker is better. Preferred: (b) on mobile, keep the small in-app picker on desktop where OS emoji entry is clumsier.

---

## Installation

### KI-022 — Linux install fails to build: ALSA development headers not installed
**Status:** Fixed (installer)
**Affected:** Every fresh Linux install (all distributions) from the point desktop Vosk dictation landed
**Description:** `install.sh` never installed ALSA development headers. `cpal` (`Cargo.toml:120`, used for desktop Vosk dictation) depends on `alsa` -> `alsa-sys`, whose build script resolves the `alsa` pkg-config package. Without `libasound2-dev` present, `cargo build` fails during the dependency build and the install aborts.
**Why it went unnoticed:** Development machines already had `libasound2-dev` installed from earlier work, and `.github/workflows/release.yml` already listed it, so both local builds and CI passed while every clean install broke. The failure was only reachable on a machine that had never built audio code before.
**Workaround (for anyone on an affected build):** `sudo apt install libasound2-dev` and re-run `./install.sh`.
**Fix:** Added to all three distribution branches in `install.sh` — `libasound2-dev` (Debian/Ubuntu), `alsa-lib-devel` (Fedora), `alsa-lib` (Arch). Only the Debian case was reported; the Fedora and Arch branches had the same gap and were fixed at the same time.
**Credit:** Reported by a beta tester, who identified the missing package.

---

## Voice & Dictation

### KI-019 — No offline dictation on Windows; Vosk is compiled out rather than unavailable
**Status:** Open — must fix before v1.0  
**Affected:** All Windows users. Dictation on Windows requires an OpenAI API key and a network round-trip, so the offline-first guarantee does not hold on Windows.  
**Description:** Vosk works on Windows — alphacep ships a prebuilt `vosk-win64-0.3.45` SDK containing `libvosk.lib` and `libvosk.dll`. Windows support is partly wired already: `install.bat` downloads that SDK into `zynkbot_rust/src-tauri/lib/vosk/`, and `START_ZYNKBOT.bat` adds that directory to `PATH` when `libvosk.dll` is present. The feature is nevertheless unreachable on Windows because four separate gates compile it out:

1. `Cargo.toml` — `vosk = "0.3"` sits under `[target.'cfg(target_os = "linux")'.dependencies]`, so the crate is never built on Windows.
2. `build.rs` — every Vosk linker flag is inside `#[cfg(target_os = "linux")]`.
3. `lib.rs` — `mod vosk_desktop;` is declared under `#[cfg(target_os = "linux")]`.
4. `lib.rs` — `start_vosk_recording` / `stop_vosk_recording` return an error stub for `cfg(not(any(target_os = "android", target_os = "linux")))`.

The `build.rs` comment records the motive: *"gate all Vosk linker flags to Linux so the Windows build doesn't try to find a non-existent libvosk.lib."* That resolved a link error by disabling the feature rather than supplying the library, and the disablement was never revisited once `install.bat` began downloading the SDK.

**Two supporting defects found while investigating:**

- **`install.bat` extracts only 2 of the 5 required files.** The Windows Vosk build is MinGW-based and the zip also ships `libstdc++-6.dll`, `libwinpthread-1.dll` and `libgcc_s_seh-1.dll`. The extract step at `install.bat:701` copies only `libvosk.lib` and `libvosk.dll`, so even a fully successful download leaves `libvosk.dll` unable to load for want of its runtime dependencies.
- **The Vosk download has no retry and fails quietly.** A failure prints a single `[WARNING]` line in the middle of a long install log and installation continues, so a Windows user ends up with no offline dictation and no clear indication why.

**Workaround:** None on Windows. Dictation falls back to OpenAI Whisper (cloud), which requires an API key and network access.  
**Fix target:** v1.0. Widen the four gates to `cfg(any(target_os = "linux", target_os = "windows"))`, add a Windows branch in `build.rs` emitting `cargo:rustc-link-search=native=<manifest>/lib/vosk`, and extract all five SDK files in `install.bat`. No `find_model_dir()` change is needed for source builds — its fourth candidate, `<CARGO_MANIFEST_DIR>/gen/android/app/src/main/assets/vosk-model`, already resolves to the model bundled in the repo.  
**Open risk:** `libvosk.lib` is a MinGW-produced import library and Zynkbot's Windows build is MSVC. This normally links for a plain C API such as Vosk's, but it is unverified. If MSVC rejects the import library, the fallback is runtime `LoadLibrary` binding of `libvosk.dll` instead of link-time binding — a materially larger change.  
**Related:** Packaged (non-source) builds cannot locate the Vosk model at all, because `find_model_dir()` depends on `CARGO_MANIFEST_DIR`, which is baked in at compile time. This affects Linux `.deb`/AppImage builds as much as Windows and is tracked separately as part of packaging work.

---

### KI-020 — Enabling Vosk on Windows makes the Vosk SDK a hard build requirement, with no fallback
**Status:** Open — introduced by the KI-019 fix; decide before v1.0  
**Affected:** Windows users who build without running `install.bat` first, or whose Vosk SDK download failed  
**Description:** Un-gating Vosk for Windows (KI-019) adds `cargo:rustc-link-search=native=<manifest>/lib/vosk` in `build.rs` and makes `vosk = "0.3"` a Windows dependency. `lib/vosk/libvosk.lib` therefore becomes a **link-time requirement** on Windows. That file is not committed — the SDK is ~66 MB — so it only exists if `install.bat` downloaded it.

`START_ZYNKBOT.bat` compiles on first launch and does **not** download the SDK; it only prepends `lib\vosk` to `PATH` when `libvosk.dll` already exists. So a user who goes straight to the launcher, or whose earlier Vosk download failed, gets a linker error naming `libvosk.lib` with nothing to indicate that a missing optional SDK is the cause. Before KI-019 this could not happen, because the Windows build ignored Vosk entirely.

**Options:**
1. **Download the SDK from `START_ZYNKBOT.bat` too** when `libvosk.lib` is absent, mirroring how the launcher already auto-detects CUDA. Makes the build self-healing and keeps dictation on by default. *Preferred.*
2. Document `install.bat` as mandatory on Windows and leave the launcher alone. Cheapest, but the failure mode stays cryptic.
3. Put Windows Vosk behind an opt-in cargo feature, so a default Windows build never breaks. Safest for the build, but offline dictation is then off by default, which defeats the purpose of KI-019.

**Fix target:** Pick one before v1.0. Option 1 is the recommendation; the launcher already has the conditional `PATH` plumbing to hang it off.

---

## Build

### KI-021 — `import_persona_collection` references a module that does not exist, so `cargo build` always fails
**Status:** Open  
**Affected:** Everyone who runs `install.bat`, on every platform  
**Description:** `src/bin/import_persona_collection.rs:19` calls `app_lib::commands::persona_memory::import_persona_memory_collection(...)`, but there is no `persona_memory` module — `commands/mod.rs` declares 17 modules and that is not among them, and nothing else in the tree defines it. The build fails with:

```
error[E0433]: cannot find `persona_memory` in `commands`
error: could not compile `app` (bin "import_persona_collection") due to 1 previous error
```

**Why it goes unnoticed in normal use:** `START_ZYNKBOT.bat` runs `tauri dev`, which builds only `--bin app` and never touches the broken binary. `install.bat` runs a plain `cargo build`, which builds *all* targets and therefore always hits it. The main application and library compile fine — `app.exe` links successfully — so the failure is limited to this one auxiliary binary.

**User-visible effect:** `install.bat` prints `[WARNING] Build failed - see errors above` and then, a few lines later, `[OK] Installation Complete`. The app does work afterwards, but the installer contradicts itself and the failure looks fatal. A new tester would reasonably conclude the install is broken.

**Fix target:** Three options — add the missing `commands::persona_memory` module (if the persona-collection feature is still intended; note `migrations/0009_persona_memory_collections.sql` exists, suggesting it was started), delete the stale binary, or keep it out of default builds with `required-features` in `Cargo.toml`. Whichever is chosen, `install.bat` should not report both a failed build and a successful installation in the same run.

---

*Last updated: 2026-09-01*

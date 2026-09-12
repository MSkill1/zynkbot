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

### KI-035 — Android first launch hung on a black screen (fixed)
**Status:** Fixed on `voice` (2026-09-11)
**Affected:** Every fresh Android install; it hit both fresh installs on the OnePlus 12R that day (ANR traces 14:58 and 15:32). Looked intermittent because it is timing-dependent.
**Description:** On Android the Rust event loop is its own thread. Plugins were registered inside `setup()` via `app.handle().plugin()`, which initialises each plugin while holding Tauri's `PluginStore` mutex, and a mobile plugin's `initialize()` dispatches to the UI thread over JNI. The UI thread, loading the first page, takes that same mutex from `shouldOverrideUrlLoading`. Each waited on the other until the 5 s ANR: the user saw a black screen right after the first permission prompt and had to kill the app.
**Fix:** plugins are registered on the `Builder` before `setup`, so they are initialised before any webview exists.

---

### KI-036 — Assistant-role step dead-ended during onboarding (fixed)
**Status:** Fixed on `voice` (2026-09-11)
**Description:** The native role request is refused on OxygenOS just as on GrapheneOS ("Role is not requestable: android.app.role.ASSISTANT"). The fallback opened `VOICE_INPUT_SETTINGS` — an overview that only shows the current assistant — with a Toast for guidance that vanished under it, and advanced the permission queue immediately, so pressing Back landed in the installer mid-download with the role unset. Without the role, "Hey Zynk" cannot answer from the lock screen. The direct per-role intent (`MANAGE_DEFAULT_APP`) opens the list but needs `android.permission.MANAGE_ROLE_HOLDERS`; an app cannot use it (it works from `adb shell` only because the shell holds that permission).
**Fix:** a persistent dialog explains the taps (Digital assistant app → tap the current assistant → choose Zynkbot → Back until you are back in Zynkbot), opens the default-apps list (`MANAGE_DEFAULT_APPS_SETTINGS`), and the queue pauses until `onResume`, which logs whether the role is held. Verified: Zynkbot holds the role after a fresh install.

---

### KI-037 — Android first run offered the desktop's GGUF model download (fixed)
**Status:** Fixed on `voice` (2026-09-11)
**Description:** `SetupWizard` decided "mobile" with `window.innerWidth <= 768`, evaluated at module load — before the WebView applied `width=device-width` — so `innerWidth` read the raw physical width (1264 px OnePlus, 1080 px Pixel) and Android took the desktop path, offering ~5 GB local models it cannot run. Latent since the wizard was written; it only shows on a first run, which is why it went unseen.
**Fix:** keyed on the Android bridge global (`window.AndroidPaths`); width stays only as the narrow-desktop-window fallback.

---

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

### KI-038 — Device name asked twice when pairing (fixed)
**Status:** Fixed on `voice` (2026-09-11)
**Description:** After the name was saved, the pairing action was resumed through the closure from the previous render, where `hasCustomName` was still false, so the prompt reopened and the user had to name the device again.
**Fix:** the resumed call carries `nameConfirmed` and skips that check.

---

### KI-039 — Keys pushed from another device did not show until the app was restarted (fixed)
**Status:** Fixed on `voice` (2026-09-11)
**Description:** The receiving backend applied pushed keys live and emitted `api-keys-updated` / `backup-key-updated`, but the Memory Manager and the model dropdown loaded their state once on mount and did not listen, so the "set up your backup key" prompt stayed and the dropdown stayed empty until a restart — with all keys, `backup.key` and the R2 credentials already on disk.
**Fix:** both listen and refetch (the dropdown coalesces the per-key burst into one request). Verified: 14 keys received in 0.4 s, one refetch, no restart.

---

### KI-050 — Reinstalling the app on a phone leaves a stale second device in every peer's ZynkSync list
**Status:** Open — post-beta. Cause known; fix deferred because it touches the pairing path. Workaround is a delete.
**Affected:** Any device that is reinstalled (or has its app data wiped, including a release build installed over a debug build) and then pairs again. Seen 2026-09-12: the OnePlus paired as `12R` before the reinstall and as `Oneplus-453A` after, from the same address; the desktop, the Pixel and the Windows install each kept both entries until the old one was deleted by hand.
**Description:** A device's identity is a UUID written to a file in the app's private data (`user_identity.rs`, `get_or_create_device_id`). Uninstalling removes the file; the next launch mints a new UUID and, on Android, a new default name from its last four characters. `zynk_devices` is keyed on `device_id` (UNIQUE), and the pairing handler upserts `ON CONFLICT (device_id)`, so a new id is a new row; nothing compares the incoming `device_ip` against existing rows. The old row stays "paired", its cert stays pinned, and every peer keeps trying to sync with it.
**Workaround:** delete the old entry in ZynkSync (it is expelled from all peers); re-pair if needed.
**Fix candidates:** (1) in the pairing-verify handler, before the upsert, expel any paired row with the same `device_ip` and a different `device_id`, logging the swap — about fifteen lines using the existing `expel_device`; risk is a home router handing an old phone's address to a different device, which would then need to re-pair; (2) carry the device id through the encrypted backup so a restore reclaims it, which also fixes the personal wake verifier being keyed to a device id that a reinstall changes. Decide after the beta.

---

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

### KI-028 — Conversation history sync duplicates messages and silently skips some threads
**Status:** Open — deferred to the ZynkSync refactor; not a beta blocker (testers should treat all sync behaviour as untested until the refactor lands)
**Affected:** Any two devices syncing conversation history over ZynkSync (observed OnePlus 12R ↔ Pixel 10 Pro XL, 2026-09-07). Covers GitHub #4 (duplicate and stale history entries) and #12 (clearing history or memories does not propagate).
**Description:** Three separate defects in `zynksync.rs` combine to corrupt synced history:
1. *Skipped threads.* `get_modified_conversations` sends only sessions whose `last_active` is later than the last sync time, comparing the values as text. Rows written through the voice path used SQLite's `datetime('now')` format (`2026-09-04 17:51:12`) while the cursor is RFC 3339 (`2026-09-04T…`); a space sorts before `T`, so those rows always looked older than the cursor and were never sent. 40 of 156 sessions on the OnePlus never reached the Pixel. Build29 (migration 0010) normalises every stored timestamp, but the cursor has already passed those rows, so they still will not sync on their own.
2. *Duplicates.* `receive_conversations_from_peer` skips a message only if one with the same `session_id`, `created_at` and `role` already exists. The same format mismatch defeated that check, so every resend inserted a second copy. The Pixel holds 328 rows for a thread whose counter says 162; the OnePlus holds 50 for one that says 30. `message_count` is not repaired, so the History panel under-reports.
3. *Resend on every restart.* The per-peer last-sync time is held in memory only, so each app restart resends the entire history, which multiplied the copies.
Deletions are not propagated at all (no tombstones), which is #12.
**Workaround:** None. Build29 stops new duplicates from forming for messages written after it. Migration 0011 (memory branch, 2026-09-07) removes the existing exact duplicates, repairs `message_count`, and adds a unique index so a resend cannot create a second row; skipped threads and non-propagating deletions remain until the outbox refactor.
**Fix target:** ZynkSync refactor (outbox model): durable per-peer send cursor, duplicate check keyed on `entry_hash` rather than timestamp text, tombstones for deleted sessions and messages, and a one-time migration that removes exact duplicate rows and recomputes `message_count`.

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
**Status:** Resolved on `voice` (build34, 2026-09-07) by design change  
**Affected:** Android 11+ devices using ZynkLink file sharing or the Knowledge Base  
**Description:** Files placed into `Downloads/ZynkbotShare/` by apps other than Zynkbot are invisible to Zynkbot's directory scan under scoped storage. Until 2026-09-07 the app declared and requested `MANAGE_EXTERNAL_STORAGE` ("All files access") to see them; Google Play only grants that permission to file managers and similar, so it was removed.  
**Resolution:** files enter ZynkbotShare through the in-app **Add file** picker (`AndroidPaths.pickFile`, which copies the file into the folder) and the Knowledge Base through **Add files** in the KB Manager (`AndroidPaths.copyToKnowledgeBase`, which copies into the app-private KB folder). Files dropped into the folder by other apps remain invisible; that is now expected behaviour and is documented in INSTALLATION_TROUBLESHOOTING.md.  
**Impact:** none for users who add files from inside the app.

---

## Debug Logging

### KI-006 — Verbose debug output in development builds
**Status:** Fixed  
**Description:** Several `println!` statements in `lib.rs` and `zynksync.rs` dumped full LLM responses and raw HTTP payloads to the terminal. Gated behind `#[cfg(debug_assertions)]` — silent in release builds, visible in `cargo tauri dev`.

---

## Mobile UI

### KI-040 — Black screen when returning to the app after a while away (reopened 2026-09-12; second fix pending verification)
**Status:** First fix (2026-09-11) was not enough: on a fresh OnePlus install on 2026-09-12, coming back from the assistant-settings screen still showed black until the first tap. Second fix on `voice` (2026-09-12), to be verified on the next APK by repeating that exact flow.
**Why the first fix missed:** it invalidated the WebView from `onResume`, but Android recreates the window surface after `onResume`, so the invalidate could land before there was a surface to draw into; nothing scheduled a frame once the surface appeared, and the first touch was what finally forced one.
**Second fix:** `onResume` sets a flag; the first `onWindowFocusChanged(true)` after it, which fires only once the window is attached with its surface, consumes the flag and does a one-frame visibility toggle on the WebView, forcing the compositor to render without input. Once per return, so no blink on ordinary focus changes.
**Description:** Android drops the window surface while another activity is up for long (16 s in Settings during onboarding did it). On return the WebView did not draw into the new surface until a touch generated input, so the app sat black. Tauri's base activity only resumes plugins.
**Fix:** `onResume` calls the WebView's `onResume`/`resumeTimers` and posts `requestLayout`/`invalidate`. Resume-side only: pausing the WebView in `onPause` would stop the JS the hands-free path relies on while the app is in the background.

---

### KI-018 — ZChat emoji picker overflows the screen on narrow Android phones (fixed)
**Status:** Fixed on `voice` (2026-09-11) — the grid was `repeat(7, 1fr)`, and `1fr` cannot shrink below min-content, so at ~360 dp (the most common Android width; the OnePlus 12R under its display-size override) it ran past the modal; now `repeat(auto-fit, minmax(36px, 1fr))`, six columns at 360 dp. The System Controls header had the same squeeze (Voice/Report labels spilling below their buttons); fixed the same day.  
**Affected:** Android users tapping the 😊 button in ZChat on phones with narrow screens (~360–411px CSS width)  
**Description:** The emoji picker in `ZChatModal.jsx` renders an inline grid of emoji buttons above the input row. It has no width cap or horizontal scroll container, so on a narrow phone the grid runs off the right edge of the screen — the leftmost emojis are visible but the rest can't be reached because the panel isn't scrollable. The Tab S3 (wider screen) shows the full row and works normally.  
**Fix target:** Two reasonable directions. (a) Constrain the picker to the modal width with `max-width: 100%; overflow-x: auto; flex-wrap: wrap;` and enlarge the touch target — keeps a consistent Zynkbot picker on desktop and mobile. (b) Hide the picker button entirely on Android (`{!isAndroid && ...}` around the 😊 button, same pattern as the VoiceButton fix in v0.9.4 hotfix). Android keyboards already expose a full emoji set via the keyboard's emoji key — duplicating it in-app is redundant and the phone's picker is better. Preferred: (b) on mobile, keep the small in-app picker on desktop where OS emoji entry is clumsier.

---

## Desktop UI

### KI-041 — Two close buttons on desktop modals, one of them closing the sidebar underneath (fixed)
**Status:** Fixed on `voice` (2026-09-11)
**Description:** The sidebar's floating toggle was hidden with the HTML `hidden` attribute, which is only UA-stylesheet `display:none`; the button's inline `display:flex` (added when centring the icon for phones) overrode it, so the toggle stayed live under every open modal — a second ✕ that closed the sidebar. The API Keys modal's own ✕ was `position:fixed` to the viewport, which hugs the panel on a phone but floated far to the right of the centred 700 px panel on desktop.
**Fix:** the toggle is conditionally rendered, not `hidden`; the modal's ✕ is sticky inside the panel.

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

### KI-019 — No offline dictation on Windows; Vosk is compiled out rather than unavailable (fixed)
**Status:** Fixed on `voice`; verified on Windows 2026-09-11 — the four gates below are widened, `build.rs` emits the Windows link-search, MSVC accepted the MinGW import library (the predicted CRT mismatch surfaces only as `LNK4098`, a warning), the NSIS bundle carries the four DLLs and the Vosk model next to `app.exe`, and Vosk and Whisper dictation were both confirmed at runtime. The offline-first guarantee now holds on Windows. KI-020 remains.  
**Affected:** All Windows users. Dictation on Windows requires an OpenAI API key and a network round-trip, so the offline-first guarantee does not hold on Windows.  
**Description:** Vosk works on Windows — alphacep ships a prebuilt `vosk-win64-0.3.45` SDK containing `libvosk.lib` and `libvosk.dll`. Windows support is partly wired already: `install.bat` downloads that SDK into `zynkbot_rust/src-tauri/lib/vosk/`, and `START_ZYNKBOT.bat` adds that directory to `PATH` when `libvosk.dll` is present. The feature is nevertheless unreachable on Windows because four separate gates compile it out:

1. `Cargo.toml` — `vosk = "0.3"` sits under `[target.'cfg(target_os = "linux")'.dependencies]`, so the crate is never built on Windows.
2. `build.rs` — every Vosk linker flag is inside `#[cfg(target_os = "linux")]`.
3. `lib.rs` — `mod vosk_desktop;` is declared under `#[cfg(target_os = "linux")]`.
4. `lib.rs` — `start_vosk_recording` / `stop_vosk_recording` return an error stub for `cfg(not(any(target_os = "android", target_os = "linux")))`.

The `build.rs` comment records the motive: *"gate all Vosk linker flags to Linux so the Windows build doesn't try to find a non-existent libvosk.lib."* That resolved a link error by disabling the feature rather than supplying the library, and the disablement was never revisited once `install.bat` began downloading the SDK.

**Two supporting defects found while investigating:**

- ~~**`install.bat` extracts only 2 of the 5 required files.**~~ Fixed: the extract step now copies `libvosk.lib`, `libvosk.dll`, `libstdc++-6.dll`, `libwinpthread-1.dll` and `libgcc_s_seh-1.dll` (checked 2026-09-09). Still open for the *installer*: the NSIS/MSI package must carry the four DLLs next to `app.exe`; `tauri.windows.conf.json` now lists them as resources (2026-09-09, unverified until a Windows build runs).
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

### KI-023 — Wake-word command is captured but not dispatched until the app is foregrounded
**Status:** Fixed on `voice` (build19–29) — hands-free turns are answered natively by `ZynkAssistantSession` / `NativeVoiceAnswerer` and joined to the current thread; the app no longer needs to be foregrounded
**Affected:** Android, "Hey Zynk" while the app is backgrounded (observed on Pixel 10 Pro XL, 2026-09-01)
**Description:** Saying "Hey Zynk" while the app is not in the foreground works up to a point — the wake word triggers, dictation runs, and the spoken text is captured correctly. But the message is never sent and no answer is produced. Opening the app causes the queued message to send immediately and Zynkbot to respond, which shows the transcript survived and only the dispatch was deferred.
**Why this matters:** Hands-free use is the entire point of the wake word. Having to open the app to complete the request removes the feature's reason to exist.
**Fix target:** The transcript reaches the JS layer but the send path appears to depend on foreground state — likely a suspended timer, a paused WebView, or a send that is queued behind a UI effect that does not run while backgrounded. Dispatch needs to complete from the background service path rather than waiting for the WebView to resume.
**Impact:** Wake word is demo-only until this is fixed. Screen-off wake word (the separate work in progress) cannot be validated end-to-end while this is outstanding, because a successful screen-off detection would hit the same wall.

---

### KI-024 — Wake word from the Android home screen only shows a popup of the transcript, nothing is sent
**Status:** Fixed on `voice` — same fix as KI-023, verified on the OnePlus 12R home screen
**Affected:** Android, wake word triggered from the launcher/home screen (observed on OnePlus 12R, 2026-09-01)
**Description:** Triggering "Hey Zynk" from the home screen produces a popup containing the dictated text and nothing further. No message is sent, and no response is generated or spoken.
**Relationship to KI-023:** Both are failures of the same step — a captured transcript that never reaches the send path. Filed separately because the observable behaviour differs (silent queue versus a visible popup that leads nowhere), and it is not yet confirmed they share one cause. Fixing KI-023 should be verified against this case explicitly rather than assumed to cover it.
**Fix target:** Determine whether the popup is the full-screen-intent notification path or a separate toast, then route it into the same dispatch fix as KI-023.

---

## Build

### KI-043 — Windows: the app cannot be rebuilt while it is running
**Status:** Open — developer-facing only
**Affected:** Anyone developing on Windows
**Description:** `build.rs` copies `lib/vosk/libvosk.dll` on every build, and Windows locks a DLL that a running process has loaded, so `cargo build` / `cargo check` fails with `The process cannot access the file because it is being used by another process (os error 32)` while Zynkbot is open — with nothing pointing at the running app as the cause. Linux does not lock loaded libraries, so the same build succeeds there. Hit twice on 2026-09-11.
**Fix target:** skip the copy when the destination is up to date, or write the resources into the bundle directory rather than beside the running executable; at minimum, name the cause in the error.

---

### KI-021 — `import_persona_collection` references a module that does not exist, so `cargo build` always fails (fixed)
**Status:** Fixed — `cargo check --all-targets` is clean on `voice` as of 2026-09-11 (Windows); this entry had gone stale.  
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

## Chat & Responses

### KI-042 — Safety classifier failed on long input and fell back to keyword matching (fixed)
**Status:** Fixed on `voice` (2026-09-11)
**Description:** toxic-bert has 512 position embeddings. Input that tokenised past 512 failed the forward pass ("index-select invalid index 512 with dim size 512"), and the containment layer fell back to keyword-only matching — a degraded check that passed a long message in Guardian mode with `✅ Content allowed`.
**Fix:** token ids and attention mask are truncated to 512 (CLS kept at index 0), so long inputs are classified on their first 512 tokens instead of dropping to keywords.

---

### KI-044 — OpenAI "Pro" models offered in the picker but rejected with 404 "This is not a chat model" (fixed)
**Status:** Fixed on `voice` (2026-09-12)
**Affected:** All platforms; surfaced in ensemble mode, where one failed provider spoils the round
**Description:** The OpenAI model list in Settings → API Keys included `gpt-5.5-pro`, `gpt-5.4-pro`, `gpt-5.2-pro` and `o1-pro`. OpenAI serves those only through its Responses API; every Zynkbot request goes to `/v1/chat/completions`, which answers `404 This is not a chat model`. Selecting a Pro model therefore broke every OpenAI turn with an error that did not say what to change.
**Fix:** the Pro entries are gone from the picker (`APIKeyModal.jsx`), and `openai.rs` maps a Pro id that is still stored from an older build to its chat sibling (`gpt-5.2-pro` → `gpt-5.2`, logged) when the target is api.openai.com. xAI and Ollama-compatible endpoints share that module and are left alone.
**Note:** the second error seen in the same ensemble round — custom endpoint `model 'frozen-14b-pilot-v1:latest' not found` — was the desktop's own custom-model setting, restored from the Linux backup, naming a model the Windows Ollama does not have. Pick a model the desktop has (`llama3.2:3b`) until the Linux blobs are copied over. The same session also changed how phones get their model: the phone used to copy the desktop's model name once at connect time (and again on every key push) and send it with each request, so a change on the desktop did not reach the phone. Now the desktop proxy replaces the model in every request from a paired device with the desktop's current selection, the phone-side model query and the pushing of `CUSTOM_*` keys are removed, and the phone stores only a `desktop` placeholder. The remote device never picks a model.

---

### KI-045 — Selecting text in a modal also selects the page behind it (fixed)
**Status:** Fixed on `voice` (2026-09-12); verify on desktop after the next rebuild, and on Android with the next APK
**Affected:** All platforms; reported from the Ensemble modal on desktop
**Description:** Modal overlays cover the page visually but not for text selection. Dragging a selection past the panel's edge, or pressing Ctrl+A inside it, highlighted the whole chat underneath as well, and a copy took all of it.
**Fix:** `index.css` marks the page unselectable while any overlay is mounted (`#root:has(...)`) and restores selection inside the overlay itself. Class-based overlays (`modal-overlay`, `zfb-overlay`, `kb-manager-overlay`, `kb-selector-overlay`, `graph-modal-overlay`, `onboarding-modal-overlay`) are matched by class; inline-styled overlay roots carry `data-modal=""` (Ensemble, Cost guide, Memory manager, Memory report, Snap-ins, Z chat, ZynkSync, Voice, Setup wizard). Side drawers (history panel, settings sidebar) are deliberately not covered — the chat stays visible beside them, so selecting it is legitimate. New modals must use one of the overlay classes or add `data-modal=""` to their backdrop.

---

### KI-046 — Pull / Stop model failed with "Failed to start ollama: program not found" although Ollama was installed and running (fixed)
**Status:** Fixed on `voice` (2026-09-12)
**Affected:** Desktop (Windows seen; Linux and Mac exposed the same way)
**Description:** Pull Model and Stop Model shell out to the `ollama` CLI by bare name. An app instance launched before Ollama was installed, or from a launcher with a minimal environment, does not have the PATH entry the installer added, so the spawn failed with "program not found" while Ollama itself was serving fine on 11434.
**Fix:** the CLI is looked up on PATH first and then in the stock install locations (`%LOCALAPPDATA%\Programs\Ollama`, `C:\Program Files\Ollama`, `/usr/local/bin`, `/usr/bin`, Homebrew, the Mac app bundle). The error now names the path it tried and suggests a restart.

---

### KI-047 — Confirmation dialogs skipped on Windows: Einstein demo loaded and "Clear All" deleted memories without asking (fixed)
**Status:** Fixed on `voice` (2026-09-12); verified on the Windows desktop dev build, Linux and Android to be re-checked on the next builds
**Affected:** Windows desktop. Linux and Android were unaffected.
**Description:** Every confirmation in the app used the browser's synchronous `window.confirm()`. On Windows the WebView2 engine hands JavaScript dialogs to the host through an event it does not await, so a `confirm()` raised from a click handler returned `true` before any dialog appeared. Only a confirm reached after an `await` (a later tick) displayed. Seen 2026-09-12 on a fresh install: loading the Einstein demo skipped its "load this demo?" prompt, and Clear All skipped both of its warnings and deleted 59 memories immediately, stopping only at the conversation-history prompt that comes after the delete call. Linux (WebKitGTK) and Android render the dialog natively, which is why it never showed there. (A dev-build detail seen at the same time, the success notification appearing twice, is React StrictMode double-running effects and is not this bug.)
**Fix:** a shared `confirmDialog()` in `src/utils/confirmDialog.js` wraps the Tauri dialog plugin's native `confirm()`, which shows a real OS dialog from Rust and resolves with the user's actual answer on Windows, Linux, macOS and Android. All 25 `window.confirm` call sites across 10 files now `await confirmDialog(...)`; two handlers (clear conversation, exit onboarding) became async to do so. `dialog:default` in `tauri.conf.json` already grants the permission, so no Rust change was needed. Outside the Tauri shell (plain browser) it falls back to `window.confirm`, which works there.
**Follow-up:** `alert()` uses the same WebView2 path. It carries no return value, so nothing is bypassed, but a sync `alert()` in a click handler may go unshown on Windows. Migrating those to the plugin's `message()` is a separate, lower-priority change.

---

### KI-048 — "Fresh" install inherited old preferences: the uninstaller never removed the WebView profile (fixed)
**Status:** Fixed on `voice` (2026-09-12) for Windows; Linux uninstaller updated on the same reasoning, to be verified on the next Linux boot
**Affected:** Windows and Linux desktop. Android is unaffected (uninstalling the app removes its WebView data).
**Description:** The app's remembered preferences (`zynkbot_voice_input_source`, `zynkbot_preferred_model`, `zynkbot_hey_zynk_enabled`, `zynkbot_tts_enabled`, `zynkbot_keep_screen_awake`, `zynkbot_web_search_auto`, onboarding flags) live in the WebView's localStorage. Tauri keeps that profile in a folder named after the app identifier, `%LOCALAPPDATA%i.containai.zynkbot` on Windows (`~/.local/share/ai.containai.zynkbot` on Linux), not in the `zynkbot` data folder the uninstaller wiped. On the 2026-09-12 fresh-install test the profile from 2026-08-30 survived, so the new install started with dictation set to OpenAI (chosen before Windows had Vosk) and the preferred backend already "custom", silently invalidating the new-user test.
**Fix:** `uninstall.bat` "Delete ALL data" now removes the WebView profile as a third location and names it in the prompt; `uninstall.sh` does the same. Until an install is redone, the folder can be removed by hand with the app closed.

---

### KI-049 — A stated fact was "extracted" but no memory was created: strict marker parsing + prompt heading echoed by small models (fixed)
**Status:** Fixed on `voice` (2026-09-12); verify on desktop with a small local model and on the next APK
**Affected:** All platforms; most likely with small local models (3B class), which the phone relies on
**Description:** "I have a dog named Mike" with llama3.2:3b produced `FACT EXTRACTION:` on one line and `Albert has a dog named Mike.` on the next. The prompt's own section heading was "PART 1 — FACT EXTRACTION:", so the model echoed the heading as its label instead of the required `MEMORY_EXTRACT:` marker. The parser accepted only a line beginning with the exact marker, found nothing, and no log recorded the miss; the display stripper hides only the exact marker, so the model's heading leaked into the visible reply and looked like a successful extraction.
**Fix:** (1) prompt headings no longer resemble output labels ("HOW TO SAVE PERSONAL FACTS", "PART 1 — MEMORY_EXTRACT") and both prompt variants state that no other label or heading may be written; (2) the parser tolerates markdown wrappers, a space for the underscore, the echoed headings, a bare heading with the fact on the next line, and hides every consumed line from the display; (3) a warning is logged when a heading appears with no fact after it. Also: the sync-tombstone path now removes the Einstein demo persona when it empties the device, matching Clear All.

---

### KI-025 — LLM responses are not streamed; nothing appears until the full response arrives
**Status:** Open — Tier 1 v1.0 item
**Affected:** All platforms, all backends
**Description:** Asking for a long answer produces no visible output until the entire response has been generated. The response should begin rendering as soon as the first tokens are available.
**Fix target:** Use `stream: true` for API backends (Claude, OpenAI, Grok) and llama.cpp's streaming generation callback for local and Ollama backends. This is a transport and rendering change only — it does not touch the memory graph, contradiction detection, or wake-word logic.
**Impact:** Perceived responsiveness. On a long answer the app currently looks frozen, which is the single most visible weakness in normal use. Also compounds KI-026: with no streaming and no stop control, a long spoken answer cannot be interrupted or previewed.

---

### KI-026 — No way to stop speech or output once it starts; Clear should become Stop
**Status:** Fixed on `voice` (build21) — Stop halts native speech, OpenAI TTS and text output; tapping the Z overlay cancels a hands-free turn
**Affected:** All platforms; most acute on Android with TTS enabled
**Description:** Once Zynkbot begins speaking a response there is no control to stop it. The existing TTS-stop work is not reachable from the UI in normal use.
**Fix target:** Replace the Clear button with a Stop button that halts speech and text output together. Clearing is already covered by the new-conversation button next to the history control, so the Clear button is redundant and its position is the natural home for Stop.
**Impact:** A long spoken response has to be waited out. Reported directly by the maintainer during device testing, 2026-09-01.

---

### KI-027 — Hands-free "set a timer" is confirmed aloud but no timer is set
**Status:** Fixed on `voice` (build26) — `VoiceCommands.kt` parses timer/alarm/stopwatch before the model and fires the `AlarmClock` intent; confirmation is spoken only after the clock app accepted it
**Affected:** Android, hands-free ("Hey Zynk") path only
**Description:** Asked hands-free to set a timer, Zynkbot replies with a spoken confirmation including the correct end time, but no timer exists and nothing happens when the time arrives. The in-app dictation path recognises timer, alarm and stopwatch requests (`parseVoiceCommand` in `useVoiceSession.js`) and hands them to the clock app through `VoiceCommandBridge`; the hands-free path (`ZynkAssistantSession` / `WakeWordService` → `NativeVoiceAnswerer`) sends the transcript straight to the language model, which has no clock and invents the confirmation.
**Fix target:** Port the command parser to Kotlin and run it before the model on the hands-free path; fire the `AlarmClock` intent from the session, and speak a confirmation only after the clock app accepted it, otherwise say it could not be set.
**Impact:** A confidently wrong answer about a timer is worse than no answer. Reported by the maintainer during device testing, 2026-09-04.

---

### KI-029 — Whisper invents words in silence
**Status:** Open — known behaviour of the Whisper model; mitigated downstream  
**Affected:** Hands-free turns with the OpenAI Whisper dictation engine selected  
**Description:** When the wake word fires and nobody speaks, Whisper sometimes returns a short phrase from nowhere ("Thanks for watching!", "Bye.", "Wow.", observed 2026-09-09). The word gate, the strict-mode shape test and the model's NO_QUERY instruction catch these, so nothing is answered, but a phantom transcript can appear in the log.  
**Fix target:** discard a Whisper transcript when the recorder measured no speech above the noise floor, or when the result matches Whisper's known filler phrases.

---

### KI-030 — Sync and cloud backup do not carry the new memory fields
**Status:** Open — after the beta, with the sync refactor  
**Affected:** Multi-device users; anyone restoring a backup  
**Description:** Migration 0011 added `tags`, `sentiment`, `event_date` use, and the `memory_entities` table. ZynkSync's memory payload and the R2 backup export were written before them and do not include tags or entities, so a memory arriving on a second device or restored from backup loses them.  
**Fix target:** extend `SyncMemory` and the backup export/import to carry the new columns and the entities rows; part of the outbox refactor.

---

### KI-031 — Ensemble replies never carry the knowledge-base note
**Status:** Open  
**Affected:** Ensemble mode with the KB button on  
**Description:** The single-model path shows a header note when the KB search found no real match (see kb_prompt.rs). The ensemble path uses the same prompt wording but never sets `kb_note` on its reply, so an ungrounded ensemble answer shows no note.  
**Fix target:** thread the KB outcome through `run_ensemble` into the synthesised reply's metadata.

---

### KI-032 — Mic-button dictation is recorded as typed
**Status:** Open — cosmetic  
**Affected:** The `input_mode` column added in migration 0011  
**Description:** Messages are recorded as `hands_free` or `typed`; text dictated with the in-app mic button is indistinguishable from typed text because the page does not tell the backend which it was.  
**Fix target:** pass an `input_mode` from the page when the message came from VoiceButton.

---

### KI-033 — Tapping a tag in About me filtered nothing (fixed)
**Status:** Fixed on `memory` (2026-09-09)  
**Description:** The Memory Manager's list query did not include the new `tags` column, so the tag filter fell back to a text search and matched nothing. The query now returns tags.

### KI-034 — The GitHub Actions Android release job has never worked (fixed)

**Status:** Fixed 2026-09-09 — workflow rewritten (the Android job runs `tauri android build`, both desktop jobs get the Vosk library, Windows ships NSIS only, and the workflow can be run by hand without a tag); manual run 34419918942 on `voice` was green on all three jobs  
**Description:** `.github/workflows/release.yml` (job `release-android`) runs `./gradlew assembleArm64Release` directly in `gen/android` without first running `tauri android build`. Gradle's `settings.gradle` applies `tauri.settings.gradle` unconditionally, and that file, `app/tauri.build.gradle.kts`, `app/tauri.properties` and `assets/tauri.conf.json` are generated by the Tauri CLI and gitignored, so on a fresh checkout Gradle fails at configuration time. The job has also never actually run: at the last tagged release (`v0.9.5-beta1`) both desktop jobs failed and the Android job was skipped.

**Effect:** no Android artefact is attached to GitHub releases by CI. The Google Play AAB and the tester APKs are built locally (`zynkbot_rust/CLAUDE.md`). **Fix:** replace the `gradlew` step with `npm run tauri android build -- --aab --target aarch64` (which generates the missing files and runs Gradle), and repair the desktop jobs the Android job depends on.

---

*Last updated: 2026-09-09*

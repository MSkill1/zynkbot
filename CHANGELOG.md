# Changelog

This file documents notable changes to Zynkbot from the initial public release forward.

For the full commit history, see [GitHub](https://github.com/MSkill1/zynkbot/commits/main).

---

## [Unreleased] — road to 1.0 <!-- branch v1, started 2026-09-15 -->

### Desktop and phone
- API Keys: once a key is stored, the provider button reads "View plan" and opens the provider's plan/billing page (it said "Get Key" and opened the key page whether or not you had one).

### Android voice
- A hands-free question that needs a web search now gets one when "Auto-execute in voice sessions" is on; the assistant-role path had been asking "want me to search?" regardless of the setting (GitHub #26, KI-062).
- A hands-free listen ends when the room falls well below your own speaking level, not only when it falls to the pre-speech floor; a room that settled at a steady murmur after you finished kept the recording running to the 30 s cap. The level is logged every 2 s while listening.

### Memory
- The "Remembered on request" filter in the Memory Manager now shows the memories stored with "Remember:"; it had matched nothing since the feature shipped (KI-059).
- A memory can no longer be given a date in the future (a memory from 9/5 was filed under December, KI-066).
- About me says how many memories have no date and so are not on the timeline.

### Android voice (tester report)
- "Set a time for five minutes" sets a timer: dictation often drops the r in "timer" (GitHub #29).
- The share folder has an "Add file" button on Android. Files placed in `Download/ZynkbotShare` by other apps are invisible to Zynkbot under scoped storage; the picker copies the file in. The Kotlin side existed, the button did not.

### Chat
- You can message your own devices. Any device paired for sync now shows a Chat button in ZynkSync settings; a message to it travels over the verified sync connection, so a note typed on the PC arrives on the phone with no link pairing. Messages to another user's linked device work as before.

### Sync
- Internal: the shared peer-to-peer layer (identity, certificate, HTTPS server, pinned client, device registry, presence, peer verification) is its own module, `transport`. ZynkSync, ZynkLink and ZChat are now three services on it, each registering its own routes and owning its own tables. No behaviour change; the two-peer harness is the check.
- A device that had never generated a pairing code had no record of itself, so every sync it started failed while recording the sync time (foreign-key error). Found by the new two-peer sync test harness; the device now records itself when its server starts.
- On the first sync between two devices with no memories, conversation history was never sent (the code returned early as "nothing to sync"). History now moves regardless.
- Pairing accepts `host:port`; the sync port is defined once and a peer's port is always taken from its stored record.

### Android voice
- The wake-word verifier can be read from the app's data folder (a personal profile no longer needs a release), understands the new position-independent profile format, and a profile marked for "whoever owns this phone" enforces without knowing the user id.

## [0.9.6-beta2] — 2026-09-14 <!-- drafted by Claude 2026-09-14 from a tester's history report -->

### Conversation history
- "New" starts a new thread. It used to empty the screen and keep the same thread, so everything said afterwards — typed or hands-free — was appended to the conversation the user thought they had left, and History showed one thread holding several conversations with no previous one to go back to.
- A thread appears in History as soon as its first message is sent, under "Current thread", instead of only after the first reply had finished.
- Threads can be renamed (pencil next to the pin). A blank name goes back to the automatic one. The delete × in History is larger and easier to tap on a phone.
- Message counts were 2 short on every thread (a one-exchange thread read "0 messages"); counted correctly now, and existing counts are repaired on first start.
- Android: a hands-free exchange that finished while the app was in the background is shown in the chat when the app comes back to the front.

### Android stability
- The personal wake-word verifier no longer enforces for anyone (it still scores and logs, and clips are still collected): in use it rejected most of its owner's real "Hey Zynk"s. It returns once retrained.
- Fixed a repeating crash ("Unable to start service SyncForegroundService") when Android restarted the app's sync service after reclaiming memory: the restart happens with the app in the background, where a foreground service is not allowed, and the refusal was not handled. The service now declines such restarts instead of crashing (reported by a tester on Android 17, reproduced on a Pixel 10 Pro XL, 2026-09-14).

## [0.9.6-beta1] — 2026-09-13 <!-- draft by Claude 2026-09-09, extended 2026-09-13 after the four-device test run; review wording -->

### Android voice
- Every "Hey Zynk" now runs natively through the Android assistant role: chime, Z overlay, tap-Z-to-cancel, Stop, replies spoken with the built-in voice and joined to the current thread. The old in-app wake path is gone.
- Timer, alarm and stopwatch by voice, handled by the clock app without a model call; stopwatch falls back to opening the clock app.
- Fewer false triggers: silence gate, back-off after repeated fruitless firings, faster close when nobody speaks, a default-deny instruction for TV and fragments, and a personal verifier trained from the owner's own clips (enforcing on the developer's phones; off with clip collection on for testers).
- The personal wake-word verifier enforces for the user it was trained on (by user id, which survives reinstalls) and only logs for anyone else.
- When Android refuses the in-app assistant-role prompt (seen on a GrapheneOS Pixel, Android 17), Zynkbot opens the system's assistant picker instead.
- The Vosk/Whisper selector applies to hands-free too; Whisper falls back to Vosk.
- "Remember colon …" saves a fact word for word by voice; Vosk's misspellings of "colon" are accepted.
- Chime and spoken replies use the media volume.
- A hands-free listen may run up to 30 seconds (was 12), long enough to dictate a paragraph.
- Wake word is opt-in (off by default) with the battery and false-trigger facts stated on the toggle.
- The detector skips its models while the room is quiet (below −52 dBFS for 2 s): an overnight run had it working at full rate under the screen-off wake lock and drained a Pixel.
- Coming back from the assistant-settings screen no longer leaves a black screen until the first tap (three attempts; the last one verified on stock Android and GrapheneOS).
- First-run fixes on Android: a startup deadlock, the assistant picker, keys pushed from another device showing without a restart, the name asked twice, and the desktop's model-download prompt no longer offered on a phone.
- The screen-off path (used when the assistant service is not bound) now plays the sent tone *before* the round trip, listens up to 30 s, and keeps the detector paused until the reply has been spoken — it used to hear the phone's own answer and fire on it.
- The first spoken line after the text-to-speech engine connects is no longer swallowed (GrapheneOS engine).
- "Play a tone when a trigger hears nothing" toggle in Voice Settings, off by default: a false trigger costs one chime and then silence.
- When the wake word is being cautious after repeated empty firings and drops a plain statement, it now says "Were you asking me a question? I'm only answering questions for a few minutes" instead of nothing.
- Whisper dictation in a noisy room: the silence detector follows the quietest level of the last two seconds, so an air conditioner no longer keeps it recording to the 30 s cap.
- Phones without a trained voice profile keep the 60 newest wake clips (20 with one), and Voice Settings has a "Send my wake-word clips" button with a live count that lights up at 30 real clips and hands a zip to the share sheet — nothing is sent by the app itself. An answered question labels its clip as real, not only clock commands.
- The personal verifier's threshold is 0.55; the shipped profile enforces for its owner only.

### Desktop
- Every "are you sure?" prompt is a native dialog on all platforms. On Windows the browser's own prompts were being skipped as if answered "yes": the Einstein demo loaded and Clear All deleted 59 memories with no question asked.
- Ollama is detected on startup and when Settings opens (running / installed but not running / not installed), with the live model list; the Ollama command is found in its standard install locations even when the app was launched with a minimal PATH.
- Paired phones always get the model selected on the desktop through the Ollama proxy; the phone no longer chooses, and custom-endpoint settings are no longer pushed between devices.
- OpenAI's "-pro" models are no longer offered (they are not chat models and returned 404); selecting text in a dialog no longer selects the page behind it; one close button per dialog.
- History shows the current thread first; the empty knowledge base offers "Add files"; the safety classifier fails open on very long input instead of falling back to keyword matching.

### Windows
- Offline Vosk dictation works on Windows: the installer carries the Vosk library, its runtime DLLs and the model. (Previously compiled out; Whisper was the only option.)
- The installer now installs per machine, under `C:\Program Files\Zynkbot`. The per-user location it used before was the app's own data folder under a different letter case, so program files sat beside the database and models. The uninstaller never deleted user data, but an older installer will quietly add a second, per-user copy next to the fixed one — remove it from Settings → Apps if you installed an earlier beta.
- The uninstaller's "delete all data" also removes the WebView profile that had let a "fresh" install inherit old preferences.

### Build and release
- The release workflow builds and signs the Linux (.deb, .rpm, AppImage), Windows (NSIS) and Android (APK and AAB) packages on a tag, and can be run by hand without one; Linux and Windows packages carry the Vosk library and model.

### Reporting
- Problem reports no longer mask ordinary file paths (the base64 pattern matched path segments); keys, tokens and pairing codes are still masked.
- With the conversation box unticked, the report's log tail no longer quotes your message, memory titles or the reply (a report on 2026-09-09 still carried them).

### Memory
- Memories carry an event date, a category from a fixed list, tags, tone and named entities from the same call that decides whether to remember; older memories are annotated in the background once.
- "About me" report in the Memory Manager, read from the local database; every line opens its memory; tap a tag to filter.
- Memories from hands-free turns are marked and listed separately.
- After tapping a tag in About me, an "← About me" button on the filter banner returns to the report with the Tags section open.
- Memories stored with "Remember:" are marked when written; a "Remembered on request" checkbox in the Memory Manager shows only them, and About me opens with a "You asked me to remember" section (20 at a time). Memories stored before this build carry no mark.
- Explicit Remember is stored even when it contradicts an older memory; a contradiction no longer blocks it.
- One timestamp format everywhere; duplicate conversation rows removed and prevented.
- Small local models that echo the prompt's headings instead of the memory marker are tolerated: the fact is still stored, the heading no longer leaks into the reply or the spoken answer ("Part 2:" included).
- A request no longer dies silently when a recalled memory holds an emoji, an accented letter or an em dash at a particular position (a byte-based cut in a log line; six such places replaced).
- The model is given the current date and time, so it stops trying to search for the date.

### Sync
- Deleting a stale device entry can no longer knock a live device at the same address off the mesh: removal notices name their target and are ignored by anyone else.
- The Ollama proxy log names the requesting device.

### Knowledge base
- Fabrication guard: the model is told what the search found (real matches, weak matches, nothing) and the reply header says when an answer is not grounded; no memory is extracted from an ungrounded reply.
- Android: Add files through the system picker; "All files access" no longer requested.
- The log names the document, chunk and score behind every knowledge-base answer.

### App
- Report a problem: a local text report (version, build, device, masked log tail, optional conversation) from System Controls or under any reply.
- Pinned threads; the current thread is restored after a restart; renamed peers show their new name; a peer's changed address is learned from its own requests; the backup key travels with "Push to all devices".
- 16 KB page-size alignment (Vosk 0.3.75, ONNX Runtime 1.29); full-screen-intent and photo permissions removed; Google Play readiness plan in the roadmap.
- Continuous integration runs the unit tests, the Tauri command-contract guards and an Android compile check on every push; the Android build compiles without warnings.

### Known at release (see docs/KNOWN_ISSUES.md)
- A device that pairs *after* the keys were saved does not receive them; press "Push to all devices" once (KI-055).
- An Anthropic key created at organisation level needs a workspace; create the key inside a workspace (KI-056).
- A false wake-word trigger in steady noise on the Whisper engine can produce a junk memory; use Vosk, the default (KI-057).
- Reinstalling a phone leaves a stale copy of it in other devices' lists; delete it (KI-050).
- Conversation-history sync can duplicate or skip threads; rebuilt after the beta (KI-028).

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

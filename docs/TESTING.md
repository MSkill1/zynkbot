# Testing Zynkbot

*How the test suites are run, what they are for, what they deliberately do not cover, and the manual checks that only a device can do. Started 2026-09-17 on the road to 1.0.*

## The standard

A test earns its place only if it is one of three things:

1. **A daily path** — something a user does every day, exercised end to end (a chat request through a fake model; two sync peers exchanging memories). One test per behaviour, not one per branch.
2. **A regression** — a bug that actually happened. Named after its known-issue or GitHub number (`ki_065_new_starts_a_new_thread`); it must fail before the fix and pass after. Every fix from 0.9.6-beta2 onward ships with one.
3. **A contract** — the shape of what crosses a boundary: sync payloads, the Tauri command wire contract, the database after migrations.

Not in scope: imagined inputs nobody has produced, exhaustive branch coverage, UI layout, model judgement. The suite should stay a few thousand lines, not grow toward the size of the product.

## Running the suites

All commands from `zynkbot_rust/`.

| Suite | Command | Notes |
|---|---|---|
| Rust unit tests | `cd src-tauri && LD_LIBRARY_PATH=$PWD/lib/vosk cargo test --lib` | `LD_LIBRARY_PATH` is required on Linux: the test binary links `libvosk.so` and exits 127 before any test runs without it. Several tests load the bundled system models (`models/system/`, fetched by `install.sh`). |
| Command contract guards | `cd src-tauri && cargo test --test command_contract` | Every `invoke()` in the frontend resolves to a registered command; no underscore-prefixed command parameters (see `tests/command_contract.rs`). |
| Frontend unit tests | `CI=true npx react-scripts test --watchAll=false` | Jest; today `src/hooks/useVoiceSession.test.js`. |
| Android (Rust) compile check | `cd src-tauri && cargo ndk -t arm64-v8a check --lib` | Catches `#[cfg(target_os = "android")]` breakage without an APK build. Note: running this invalidates the cached OpenSSL build for the next `tauri android build`; put the NDK `bin` directory on `PATH` for that build (see CLAUDE.md). |
| Android (Kotlin) compile | `JAVA_HOME=~/android-studio/jbr npm run tauri android build -- --debug --apk --target aarch64` | The only thing that compiles the Kotlin. CI does this automatically when a `.kt`, manifest or Gradle file changes. |

Continuous integration (`.github/workflows/test.yml`) runs the first four on every push (documentation-only pushes are skipped) and the Kotlin compile when Kotlin changed. The release workflow is separate and runs only on a `v*` tag.

## What is covered today (2026-09-17)

- **Conversation history** (`conversation_history.rs`): open / log / count / rename, count repair at startup.
- **Memory extras** (`memory_extras.rs`): event-date validation, namespace resolution, tag cleaning.
- **Prompt pieces**: KB fabrication guard, question extraction, NLP helpers, conversation-engine prompt assembly.
- **Chat helpers** (`commands/chat.rs`): explicit-remember parsing and reply cleanup.
- **LLM clients**: response parsing for Anthropic, OpenAI, xAI; local embeddings.
- **Containment / safety classifier**: mode handling; fail-open on long input.
- **Migrations** (`db.rs`): the schema applies cleanly to an empty database and creates every table.
- **Frontend**: voice-command parsing, native-turn merging, speech cleanup, speak-or-not rules.
- **Contract**: the Tauri command wire contract.

## What is not covered, and the plan

In order of evidence (where the known issues came from):

1. **ZynkSync** — pairing, key push, memory/history sync, deletions, edits, device identity. No tests. Plan: a two-peer harness (two service instances with separate in-memory databases and identities, real router, loopback) exercising the behaviours listed in the sync section below; it is also the acceptance test for the sync rebuild.
2. **Chat pipeline end to end** — containment → recall → prompt → model → history → extraction. Plan: a fake model backend so `generate_reply` runs in-process.
3. **Memory pipeline** — dedupe, contradiction, dates/tags. Mostly reached through (2).
4. **Android Kotlin** — pure logic (command parsing, verifier scoring, speech cleanup) as JVM unit tests; everything else is manual (below).
5. Deliberately untested: local GGUF model loading (hardware), the safety classifier's judgement (model behaviour), UI layout, the R2 upload itself (network).

## Sync behaviours the harness must cover

Each is one test. Those marked *rebuild* are expected to fail on the pre-rebuild code and are kept `#[ignore]`d with the known-issue number until the rebuild makes them pass.

1. Pair with a code: both sides hold each other's device row and certificate; the joining device adopts the host's user id.
2. Introduce: a third device pairs with the host and learns the second.
3. Memory added on A appears on B with every field (date, tags, entities, provenance). *rebuild — KI-030*
4. Memory deleted on A is gone on B; a stale peer cannot recreate it.
5. Memory edited on A: B has the new text and the old one does not come back. *rebuild — KI-060*
6. Memories that existed on B before pairing are adopted, not orphaned. *rebuild — KI-011*
7. API keys pushed to all devices reach every peer, including one that paired after the keys were saved. *rebuild — KI-055*
8. A thread with N messages on A is on B with N messages, once; a second sync sends nothing. *rebuild — KI-028*
9. A thread deleted on A is gone on B. *rebuild — KI-028, #12*
10. Renaming a device shows on peers without re-pairing.
11. Expelling a stale device entry affects only that device id, not a live device at the same address (KI-053 regression).
12. Reinstall (new device id, same user id, restored backup): peers gain no ghost row. *rebuild — KI-050*
13. Contract: the memory and conversation payloads round-trip through serde with every field.
14. Contract: the mTLS client refuses an unpinned certificate.

## Manual device checklist

These cannot be automated. Run before a release; record results in the release notes.

**Conversation history (any platform, nine steps)**
1. Existing threads present; message counts match content.
2. Send a message; open History while the reply is streaming: the thread is under "Current thread" with the right count.
3. New → confirm → send → History: the thread just left is first under Today, unchanged; the new one under Current thread.
4. Resume a previous thread: its messages return; a message sent now lands in it.
5. Rename with the pencil; the name persists across close/reopen; blank restores the automatic title.
6. Pin / unpin moves the thread between groups.
7. Dictation via the mic button lands in the thread on screen.
8. A thread whose first reply failed is listed only while current.
9. Escape during rename cancels.

**Android voice**
- Assistant-role setup on stock Android and GrapheneOS; the app is drawn normally on return (KI-040, KI-051).
- "Hey Zynk" with the screen off: chime, spoken answer; timer / alarm / stopwatch commands; the timer must actually fire with the app hidden and with the phone asleep (KI-061).
- A hands-free exchange on a blank conversation appears on screen and in History.
- With a personal voice profile enforcing: the owner is accepted, another voice is not.
- Leave the phone alone for several hours: no crash dialogs (KI-064).

**Installers**
- Windows: upgrade in place over the previous version; the database is byte-identical before and after; the uninstaller leaves user data unless told otherwise (KI-054, KI-048).
- Linux: `.deb` upgrades in place.
- Android: the release APK installs over the previous release without a wipe (same signing key).

**Fresh-user pass (all four devices)** — back up the desktop's data folder; uninstall everywhere; install; onboard the desktop first; pair the others to it; run the checks above; restore the desktop and re-pair.

# Zynkbot as the phone's digital assistant

**Status:** Design note, agreed direction. Gated on the on-device checks in §6 before code.
**Date:** September 2026
**Origin:** Product decision (2026-09-03): "Hey Zynk" must behave like Siri or Google
Assistant — heard, processed, answered by voice — with no notification, nothing on the
lock screen, and without the user having opened the app first. Android has a formal
slot for this (`VoiceInteractionService`, API 21+). Zynkbot should be built for it.
**Target platform:** standard Android. GrapheneOS is an adaptation afterwards, never a
design driver. Primary test device: OnePlus 12R (CPH2611, OxygenOS/ColorOS 16, **Android 16**).
Checked 2026-09-03 over ADB: the standard assistant picker (`Settings$ManageAssistActivity`)
exists on this ROM and Google currently holds `android.app.role.ASSISTANT`. Whether a
sideloaded app is offered in that list is G1 and needs a build.

Everything in §2 was verified by reading the source on the `voice` branch. Everything
in §6 is *unverified* and must be tested on the OnePlus before Phase 2 begins.

---

## 1. The requirement, stated once

| | Today | Required |
|---|---|---|
| Wake word listening starts | only after the app is opened in the foreground | whenever the phone is on |
| Saying "Hey Zynk" with the phone locked | posts a notification which, if a permission is granted, wakes the screen and opens the app | chime, listen, speak the answer; screen stays dark |
| Where the answer appears | the full app window | speaker; text lands in chat history for later |
| After the reply | (bug) may keep the mic open | strictly one-shot; back to passive listening |
| "Set a timer for ten minutes" | handled in JS inside the app window | handled without opening Zynkbot's window |

The deal-breaker is row one. The rest is what "done properly" looks like.

---

## 2. What exists today (verified)

### 2.1 Three layers, and where each one lives

```
 ┌─ Listener ─────────────┐   ┌─ Capture + routing ───────┐   ┌─ Response surface ───────┐
 │ WakeWordService.kt      │   │ Vosk dictation            │   │ MainActivity WebView      │
 │ foreground service,     │──▶│ (Kotlin when screen off,  │──▶│ JS calls Rust via Tauri   │
 │ own mic, 3 ONNX models  │   │  JS when app is visible)  │   │ then OpenAI TTS from JS   │
 └────────────────────────┘   └───────────────────────────┘   └──────────────────────────┘
```

**The brain is already native.** `send_message_with_memory` (`src-tauri/src/commands/chat.rs`,
~1,500 lines) does memory retrieval, containment, knowledge-base RAG, the LLM call
(`src-tauri/src/llm/anthropic.rs`, `openai.rs`, Ollama), streaming (`stream-token`
events) and post-reply memory extraction. `App.jsx` (~line 710) is a thin caller:
`invoke('send_message_with_memory', {...})`, append tokens, then
`voice.speakResponse(text)`.

The brain's only coupling to the window is plumbing:
- It is reached through Tauri's JS↔Rust bridge, which exists only inside the WebView.
- It takes a `tauri::AppHandle`, used for `emit()` (stream tokens, `memory-processing-*`,
  `contradiction-detected`) and for managed state.
- The database path is resolved without Tauri (`db::get_app_data_dir()`), so the core
  can find its data with no window present.

### 2.2 Why notifications and the lock screen are involved at all

`WakeWordService.deliverTranscriptToApp()` needs `MainActivity` to reach RESUMED so the
WebView's JS runs (`MainActivity.kt` documents that `evaluateJavascript()` silently
drops on a paused WebView). Bringing an Activity to the front from a background service
is restricted on stock Android 10+. The code's two routes:

1. **Screen on, not GrapheneOS:** plain `startActivity()`. The comment says this
   "requires SYSTEM_ALERT_WINDOW". **The manifest does not declare that permission and
   nothing requests it**, so this route is expected to fail on stock Android too. This
   was previously misreported as a GrapheneOS-only limitation; it is not.
2. **Always:** a high-priority notification with a full-screen intent. Android 14 denies
   `USE_FULL_SCREEN_INTENT` to non-alarm/non-call apps by default; the app never asks
   for it. Screen-off wake works on the Pixel only because it was granted by hand over
   ADB; the same grant fails on ColorOS over ADB.

Neither route is a design choice. Both are workarounds for one fact: **the response
pipeline can only run inside a resumed app window.** Remove that fact and both
workarounds disappear.

### 2.3 Why the listener needs the app opened first

`WakeWordService.onStartCommand` calls `startForeground(..., FOREGROUND_SERVICE_TYPE_MICROPHONE)`.
On Android 14 that throws `SecurityException` unless the app is in a foreground-eligible
state (the code catches it and stops the service). So if the phone was locked before
Zynkbot was opened, nothing is listening.

### 2.4 Two UX defects found on the way

- **Conversation mode is mislabeled.** `VoiceModal.jsx` shows the toggle as
  *"Keep screen awake — Useful for hands-free use during a session."* It does keep the
  screen awake (`navigator.wakeLock`) **and** it arms `conversationLoopActiveRef`, which
  re-opens the microphone after every spoken reply with no wake word. Nothing in the
  label says so. Commit 91bb2c6 made the screen-off path ignore it; the in-app path
  still loops.
- **"Speak response" is a mode, not a consequence.** `zynkbot_tts_enabled` speaks every
  reply everywhere, including typed ones, until the user remembers to turn it off. And
  because it defaults **off**, a locked-screen "Hey Zynk" on a fresh install produced
  no spoken reply at all — the screen-off path checked the same flag.

Both fixed in Phase 0 (see §7): the loop is removed, `shouldSpeakReply()` implements
the rule in §5, and the toggle is relabelled "Speak replies in the app".

### 2.5 Inventory of Android-native code

| File | Lines | Role |
|---|---|---|
| `MainActivity.kt` | 879 | six `addJavascriptInterface` bridges: folder picker, paths, camera, `VoskBridge`, `WakeWordBridge`, `VoiceCommandBridge`; transcript hand-off in `onNewIntent`/`onResume` |
| `WakeWordService.kt` | 578 | ONNX wake word, screen-off Vosk dictation, chimes, notification delivery |
| `SyncForegroundService.kt` | 59 | keeps ZynkSync alive |

All three live in `src-tauri/gen/android/`, which Tauri regenerates. A
`VoiceInteractionService` cannot live there. This is why the plugin refactor (roadmap
item 2) is the vehicle for this work, not a separate cleanup.

---

## 3. What the assistant role gives us — and what it does not

**Assistant** here means the app the user picks at *Settings → Apps → Default apps →
Digital assistant app*. One app holds the role at a time.

| Gives | Does not give |
|---|---|
| A **system-owned overlay window** (`VoiceInteractionSession`) drawn over any app. Not an Activity, so the background-launch restrictions in §2.2 do not apply. This is the "pulsing light / live transcript" surface. | **Always-on hotword.** `AlwaysOnHotwordDetector` needs DSP support reserved in practice for the phone maker's assistant. `WakeWordService` stays, with its low-priority "listening" notification (Android requires one for a background microphone). |
| A **hardware trigger**: long-press power / home / corner swipe opens Zynkbot instead of Gemini. Flag `supportsLaunchVoiceAssistFromKeyguard` allows it from the lock screen. | **The brain moved for free.** The session is a native view. Either it hosts its own WebView wired to Rust, or the Rust core gets a native door (§4.3). We take the door. |
| **Legitimate app launching** from the session (`startAssistantActivity`) — "set a timer" opens the Clock app from the background. | **A guarantee about ColorOS.** OEM assistant handling varies; §6 G1/G2. |
| **Screen context** on invocation (`onHandleAssist`: the visible text and a screenshot). "Hey Zynk, remember this" on a web page becomes a feature. | **Gemini kept.** Choosing Zynkbot displaces the phone's assistant for the gesture. Must be opt-in and explained. |
| **A process the OS keeps alive** — the system binds the assistant at boot. Directly relevant to the deal-breaker (§6 G3 decides how far). | |

---

## 4. Target architecture

```
                        ┌──────────────────────────────────────────────────────┐
                        │ tauri-plugin-zynk-android  (new; Kotlin + Rust glue) │
                        │                                                      │
 system binds at boot ─▶│ ZynkAssistantService : VoiceInteractionService       │
                        │   • starts WakeWordService                           │
                        │   • showSession() on wake word / gesture             │
                        │                                                      │
 "Hey Zynk" ───────────▶│ WakeWordService (as today, relocated)                │
                        │                                                      │
                        │ ZynkAssistantSession : VoiceInteractionSession       │
                        │   • overlay: listening state, live transcript, reply │
                        │   • Vosk dictation (one shot)                        │
                        │   • route: local command grammar → Clock intents     │
                        │           otherwise → ZynkCore.send()                │
                        │   • speak reply (Android TextToSpeech), sentence-wise│
                        │   • startAssistantActivity(MainActivity) only when   │
                        │     the reply needs the full app                     │
                        │                                                      │
                        │ ZynkCore (Kotlin)  ⇄ JNI ⇄  app_lib (Rust cdylib)    │
                        └──────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
                                    core::chat::send(ctx, req, sink)   ◀── Tauri command
                                    (the existing brain, minus AppHandle)    wrapper (unchanged
                                                                             behaviour for the
                                                                             desktop + in-app path)
```

### 4.1 `ZynkAssistantService`
Declared in the plugin's manifest with `BIND_VOICE_INTERACTION` and a metadata XML naming
the session service and `supportsLaunchVoiceAssistFromKeyguard="true"`. On `onReady()` it
starts `WakeWordService`. It exposes `showSession(args)` to the wake-word detector so
detection opens the overlay instead of posting a notification.

### 4.2 `ZynkAssistantSession`
Owns one interaction from chime to spoken reply. Strictly one-shot (§5). Hosts the
overlay view. Runs the existing Kotlin Vosk dictation (`startKotlinVoskDictation` today
lives in `WakeWordService`; it moves here). Routes the transcript:

1. Local command grammar (today `parseVoiceCommand` in `useVoiceSession.js`: timer,
   alarm, stop, never-mind). Ported to Kotlin so no round-trip is needed; fires
   `AlarmClock.ACTION_SET_TIMER` etc. through `startAssistantActivity`.
2. Otherwise `ZynkCore.send(...)`, streaming tokens into the overlay and into a
   sentence buffer for TTS.

### 4.3 `ZynkCore` — the native door into the Rust brain
`app_lib` is already a `cdylib`. Add a small JNI surface (`jni` crate):
`Java_ai_containai_zynkbot_ZynkCore_sendMessage(...)` with a callback object for tokens
and completion. Behind it, refactor `send_message_with_memory` so the Tauri command is a
thin wrapper around `core::chat::send(ctx, request, sink)` where:

- `ctx` carries what the command currently pulls from managed state;
- `sink` is a trait with `token(&str)`, `event(name, json)`, `done(result)`. The Tauri
  wrapper implements it with `AppHandle::emit`; the JNI wrapper with a Java callback.

This is the same seam the test suite has wanted: the brain becomes callable from a
plain Rust test with a recording sink. No behaviour change on desktop.

**Open question (G4):** on Android, is Tauri's runtime alive when no Activity exists?
If yes, the JNI door can share its state. If no, `core::` must initialise itself from
`db::get_app_data_dir()` — which it can, since paths do not depend on Tauri.

### 4.4 Text to speech
Today: OpenAI `tts-1` fetched from JS, requiring an OpenAI key and the WebView. In the
session: Android's built-in `TextToSpeech` (offline, no key, works with the screen off)
as the default; OpenAI/other voices as an option later. Speak on sentence boundaries
from the token stream — this also fixes the 20–30 s "silence then everything at once"
delay, which is caused by JS waiting for the whole `invoke` to resolve before calling
`speakResponse`.

### 4.5 What `MainActivity` keeps
The in-app experience: typed chat, the mic button, camera, folder picker. The voice
bridges shrink to "start/stop passive listening" and "open the assistant session". The
transcript hand-off via `onNewIntent`, the transcript notification channel, the
full-screen-intent code and `USE_FULL_SCREEN_INTENT` are all deleted in Phase 4.

---

## 5. Interaction contract (product decisions, 2026-09-03)

1. **One shot.** One "Hey Zynk", one question, one reply, then passive wake-word
   listening. The app never re-opens the microphone on its own. The conversation loop
   is removed, not defaulted off. If a "keep screen awake" option survives, it does
   only that and says so.
2. **Answer in the channel you used, with one override.**
   - "Hey Zynk" with the app not in front → spoken; text also lands in chat history.
   - Dictating or typing into the open app → text only. The user is reading, or is in
     public.
   - Settings keep a **"Speak replies in the app"** toggle, **default off**, for
     hands-free use with the app open. It never affects the hands-free path, which
     always speaks.
   - Phone on silent/vibrate → never speak.
3. **The wake word works whenever the listener is running**, including with the app
   open. Rule 2 decides only how the reply is delivered, not whether "Hey Zynk" is heard.
4. **Interrupt** by saying "Hey Zynk" or "stop", or tapping. Existing behaviour, kept.

---

## 6. Verification gates on the OnePlus (before Phase 2)

Each is a throwaway spike, not product code. Record results here.

| Gate | Question | If no |
|---|---|---|
| **G1** | Can a sideloaded third-party app be selected as Digital assistant on ColorOS, and does the hardware gesture invoke it? | OEM restriction; the session overlay still works when opened by our own wake word — confirm that separately. |
| **G2** | Does the `VoiceInteractionSession` overlay show over the lock screen (screen on, locked)? With the screen off, does TTS from the session play? | Speaking with the screen dark is the primary case; overlay-over-keyguard is secondary. |
| **G3** | Can `WakeWordService` start its microphone foreground service from `ZynkAssistantService.onReady()` on Android 14 with no visible Activity? | Fallback A: does showing the session count as foreground-eligible? Fallback B: `BOOT_COMPLETED` receiver + `SYSTEM_ALERT_WINDOW` (must be declared **and** granted; today it is neither). |
| **G4** | With `MainActivity` destroyed, is Tauri's Rust runtime still alive in the process? | `core::` initialises independently (§4.3). |
| **G5** | Android `TextToSpeech` from a service, screen off, speaker audible, interruptible? | Evaluate alternatives before Phase 3. |
| **G6** | What does the user lose by displacing Gemini on this phone (gesture, "Hey Google")? Document for testers. | — |

---

## 7. Phasing

| Phase | Work | Depends on |
|---|---|---|
| **0** | Fix the two UX defects in §2.4 in JS (remove the loop; reply rule + "Speak replies in the app" toggle). Run G1–G6 spikes. | nothing |
| **1** | Extract the three Kotlin files into `tauri-plugin-zynk-android`, behaviour unchanged. Fix the OnePlus non-detection and the restart churn first so "unchanged" is checkable. | KI-023/024 understood |
| **2** | Rust: `core::chat::send` + sink trait; Tauri wrapper; JNI door; Rust test with recording sink. | G4 |
| **3** | `ZynkAssistantService` + `ZynkAssistantSession`; Kotlin command grammar; TTS. | Phase 1, 2; G1–G3, G5 |
| **4** | Delete the notification/full-screen-intent launch path and the `onNewIntent` transcript hand-off. | Phase 3 verified on the OnePlus |
| **5** | Local model on device as the offline brain behind the same door. | deferred; not a prerequisite |

---

## 8. Risks and open questions

- **OEM behaviour.** ColorOS may restrict third-party assistants or kill background
  services aggressively. G1/G3 are the early warning.
- **Battery.** An always-open microphone is the cost of a custom wake word without DSP
  access. Unchanged from today; worth measuring once it starts at boot.
- **Two dictation stacks** (Kotlin Vosk in the session, JS Vosk for the in-app mic
  button) remain. Acceptable for now; unify after Phase 3.
- **Desktop parity.** None of this changes Linux/Windows. The in-app path stays as is.
- **GrapheneOS.** Not a design input. Test at the end; if the overlay or gesture is
  blocked there, document it.

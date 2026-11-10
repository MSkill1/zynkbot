# Zynkbot on Android: how the build is put together

*Checked against the code on 2026-11-10 (`v1`, a7599eb).*

This is a description of the Android build as it exists in the source, written for a developer who has to change it. It is not a user guide. Paths under `zynkbot_rust/src-tauri/gen/android/app/src/main/java/ai/containai/zynkbot/` are written below as just the Kotlin file name. Rust paths are relative to `zynkbot_rust/src-tauri/src/`. App version at this commit: 0.9.6.

The short version: the app is a Tauri 2 WebView app (`ai.containai.zynkbot`) whose Rust core is compiled into `libapp_lib.so`. The Kotlin layer around it is large for a Tauri app because the hands-free voice pipeline (wake word, dictation, spoken reply) runs in native code without the WebView, and reaches the Rust core through JNI functions rather than through Tauri.

## 1. Layout: what lives where on the phone

All Rust-side state lives under the app-private files directory. `db::get_app_data_dir()` (`db.rs`) has an Android branch: it takes `$HOME` (falling back to `/data/data/ai.containai.zynkbot` when it is not set) and appends `files/zynkbot`. Kotlin's `context.filesDir` is `/data/data/ai.containai.zynkbot/files`, so the two sides agree on the tree:

| Path (under `/data/data/ai.containai.zynkbot/files/`) | Written by | Purpose |
|---|---|---|
| `zynkbot/zynkbot.db` | `db::get_db_path` | SQLite database; migrations in `src-tauri/migrations/` |
| `zynkbot/.env` | `commands/models.rs` (`set_api_key`, `set_preferred_backend`), `zynksync.rs` | API keys, `ZYNK_MODEL_BACKEND`, pushed sync settings. Loaded by `env_file::load` (through `load_env_file` in `lib.rs`, and through `android_jni::ensure_env_loaded` when the service starts without the app screen). Read-only from Kotlin via `EnvFile.kt` |
| `zynkbot/.zynk_user_id`, `.zynk_device_id`, `.zynk_device_name`, `.zynk_current_session`, `.zynk_user_info.txt` | `user_identity.rs` `IdentityManager` (Android branch uses `get_app_data_dir()`; desktop uses `dirs::config_dir()/Zynkbot`) | User and device identity; the thread the app last showed |
| `zynkbot/backup.key`, `zynkbot/backup_key_acknowledged` | `commands/backup.rs` `key_path`, `key_acknowledged_path` | R2 backup encryption key and the "I saved it" flag |
| `zynkbot/user_profile.json` | `db::get_user_profile_path` | user profile |
| `zynkbot/knowledge_base/<user_id>/` | Rust `kb_rag::get_kb_folder_path`; Kotlin `ZynkbotPathsBridge.copyToKnowledgeBase` writes to the same folder | Knowledge Base documents the Rust scanner indexes |
| `zynkbot/wake_triggers/` | `WakeWordService.saveTriggerClip`, `labelLastClip` | Wake-word trigger clips, features and labels (section 7) |
| `zynkbot/wake-word-models/` | `WakeVerifierTrainer`, `WakeWordService.startEnrollment`, `wake_profile.rs` | The taught voice profile `hey_zynk_verifier.json`, the trainer's state `verifier_training.json`, and the enrollment marker `enrolling` (section 7) |
| `zynkbot/logs/` | `app_log.rs`, `WakeLog.kt` | `zynkbot.log` (the Rust log, three generations) and `wake.log` (the hands-free log, rotated at 1,000,000 bytes with one `.1` generation) |
| `zynkbot/models/system`, `zynkbot/models/user` | created in `lib.rs run().setup` | `models/user` holds the local chat model, a `.gguf` file (`llm::local_models::resolve_default_model_path`). `models/system` holds the supporting models, downloaded by `commands/setup_wizard.rs` (for example `system/all-MiniLM-L6-v2`); they are not in the APK |
| `ZynkbotShare/` | `ZynkShareProvider.shareDir()` | The share folder (below). `seal.rs` writes seals to `ZynkbotShare/seals` on Android |
| `vosk-model/` | `MainActivity.extractVoskModelIfNeeded` (copied from APK assets) | Vosk speech model. Note: directly under `files/`, not under `zynkbot/` |
| `wake-word-models/` | `WakeWordBridge.isModelReady` (copied from APK assets) | The five wake-word files: `melspectrogram.onnx`, `embedding_model.onnx`, `hey_zynk.onnx`, `silero_vad.onnx`, `generic_negatives.json`. Also directly under `files/`. Not the same folder as `zynkbot/wake-word-models/` |

Six SharedPreferences files hold Kotlin-only state: `zynkbot_voice` (keys `input_source` and `fruitless_tone`, section 8), `zynkbot_setup` (key `asked_assistant_role`, section 4), `zynkbot_share` (keys `changed`, set by `ZynkShareProvider` when a file arrives outside the Rust index and consumed in `onResume`, and `old_share_migrated`), `zynkbot_power` (keys `starts` and `battery_exemption_asked`, section 4), `zynk_boot` (`WakeBoot`: `hey_zynk_enabled`, `threshold`, `start_on_boot`) and `zynk_timer` (`ZynkTimer`, section 6a).

The share folder is Zynkbot's own storage location since 2026-09-19: `filesDir/ZynkbotShare`, created by `ZynkShareProvider.shareDir()` and published to the system by `ZynkShareProvider` (a `DocumentsProvider`, authority `ai.containai.zynkbot.share`, registered in the manifest with the `DOCUMENTS_PROVIDER` intent filter and guarded by `MANAGE_DOCUMENTS`). It appears as "Zynkbot" in the Files app drawer and in any app's file picker; `ShareReceiverActivity` (`ACTION_SEND` / `ACTION_SEND_MULTIPLE`, `*/*`) puts Zynkbot in every app's Share menu and copies into the same folder. The page registers the folder as the device's single ZynkLink share (`ZynkLinkPanel.jsx` calls `AndroidPaths.getShareDir()`, removes any share record with another path and registers this one).

Why a provider: since Android 11 an app can read only the files it created itself in a public folder, so when the share lived in `Download/ZynkbotShare` anything another app put there was invisible to the Rust scan (KI-015). With the folder in app storage the provider is the one route in, every file is written by this process, and `std::fs` reads it unchanged. `MainActivity.migrateOldShareFolder()` moves the files we own out of the old Download folder once (preference `old_share_migrated`). The folder is not a media folder, so images in it are not in the gallery; after an image download the page offers a copy in `Pictures/Zynkbot` (`AndroidPaths.saveToGallery`). The Knowledge Base stays app-private for the same reason it always was.

## 2. The generated-but-hand-edited Android project

`tauri android init` generates `src-tauri/gen/android/`. In this repository the directory is committed and hand-edited, and re-running `tauri android init` would overwrite the hand-written files with Tauri's defaults. `zynkbot_rust/CLAUDE.md` states the rule. Its own file list is behind this table: it says 16 Kotlin files and names 14.

Hand-written and tracked in git (line counts from `wc -l` on 2026-11-10):

| File | Lines | Role |
|---|---|---|
| `MainActivity.kt` | 1543 | Activity, permission queue, model unpacking, battery prompt, all six JavaScript bridges |
| `WakeWordService.kt` | 1221 | Foreground service. Owns the microphone state and the one stream. ONNX wake-word detector, the gates (quiet, Silero VAD, playback), enrollment, clips and labels, watchdog, hand-off to the assistant session |
| `MicState.kt` | 128 | The microphone state machine. Pure Kotlin, no Android types |
| `MicStream.kt` | 107 | The one `AudioRecord` reader. Hands 80 ms chunks to a sink that can be swapped; attaches the echo canceller |
| `Dictation.kt` | 216 | One dictation as a `MicStream.Sink`: Vosk or OpenAI, with the hands-free timers |
| `OpenAiDictation.kt` | 225 | OpenAI dictation helpers: energy endpointer, WAV wrap, upload, silence-phrase guard. It no longer records |
| `OpenAiRealtime.kt` | 481 | Live transcription over a WebSocket to the OpenAI Realtime API |
| `ZynkAssistantSession.kt` | 447 | One assistant invocation: Z overlay, dictation, answer |
| `NativeVoiceAnswerer.kt` | 489 | JNI call into Rust, streaming text-to-speech, pending turns |
| `VoiceCommands.kt` | 277 | Timer, alarm, stopwatch parsing; runs them through `ZynkTimer` (clock-app intents only as the fallback) |
| `ZynkTimer.kt` | 320 | Zynkbot's own timers, alarms and stopwatch (2026-10-25): state in SharedPreferences, one `AlarmManager.setAlarmClock` per entry, countdown notifications, the ring |
| `TimerReceiver.kt` | 22 | Exact-alarm target: fire, Cancel from a notification. After a reboot it reschedules timers and calls `WakeBoot.tryStart` |
| `TimerRingService.kt` | 96 | Short foreground service that speaks the line and rings for 30 s |
| `WakeVerifier.kt` | 103 | Loads and scores the taught voice profile (logistic regression over pooled embedding features) |
| `WakeVerifierTrainer.kt` | 350 | Trains the voice profile on the phone from the labelled clips and the shipped generic negatives, and decides whether the profile enforces |
| `WakeBoot.kt` | 100 | Starts `WakeWordService` without the page (after a reboot, an update, or when the assistant service binds) and posts a notice if it could not |
| `WakeLog.kt` | 59 | Hands-free log: logcat plus `files/zynkbot/logs/wake.log` |
| `ZynkNotify.kt` | 43 | Silent notification for a card raised while the app is not in front; called from Rust |
| `PackageReplacedReceiver.kt` | 46 | After an app update: restart the listener, and post "Zynkbot was updated" if the assistant service does not bind |
| `ZynkAssistantService.kt` | 74 | `VoiceInteractionService` (assistant role) |
| `ZynkAssistantSessionService.kt` | 14 | Creates a `ZynkAssistantSession` per invocation |
| `ZynkRecognitionService.kt` | 29 | Stub `RecognitionService` required by the assistant role |
| `ZynkShareProvider.kt` | 227 | `DocumentsProvider`: Zynkbot as a storage location in the Files app |
| `ShareReceiverActivity.kt` | 73 | "Share to Zynkbot" from any app's Share menu |
| `SyncForegroundService.kt` | 84 | `dataSync` foreground service |
| `ZynkCore.kt` | 61 | JNI declarations for `nativeInit` and `nativeSendMessage` |
| `EnvFile.kt` | 31 | Read `files/zynkbot/.env` from Kotlin |
| Total Kotlin (27 files) | 6866 | |

Also hand-edited and tracked: `app/src/main/AndroidManifest.xml`, `app/build.gradle.kts` (dependencies, SDK levels, signing), `app/proguard-rules.pro`, `app/src/main/res/xml/voice_interaction_service.xml`, `res/xml/recognition_service.xml` (the `android.speech` meta-data for the stub recognition service, added after a Pixel parse error on 2026-10-27), `res/xml/file_paths.xml`, `app/src/main/res/raw/wake_chime.wav` and `wake_chime_close.wav`, and `app/src/main/assets/`: the Vosk model (68 MB on disk) and `wake-word-models/` (5.5 MB) with `melspectrogram.onnx`, `embedding_model.onnx`, `hey_zynk.onnx`, `silero_vad.onnx` (speech gate), `generic_negatives.json` (the trainer's shipped negative set) and `ATTRIBUTION.txt`. No voice profile ships in the APK (since 2026-11-03, comment in `WakeVerifier.kt`). `buildSrc/` (Tauri's Gradle `RustPlugin` and `BuildTask`) and `keystore.properties.template` are tracked.

Generated by the Tauri CLI at build time and gitignored (`gen/android/.gitignore`, `app/.gitignore`): `app/tauri.properties` (versionName/versionCode), `app/tauri.build.gradle.kts`, `tauri.settings.gradle`, `app/src/main/assets/tauri.conf.json`, `app/proguard-tauri.pro`, `app/src/main/java/ai/containai/zynkbot/generated/*.kt` (`WryActivity`, `TauriActivity`, `RustWebView`, `Rust.kt`, etc.), and `jniLibs/**/*.so`. `keystore.properties` is gitignored.

## 3. Manifest

Permissions declared in `AndroidManifest.xml` and what uses them:

| Permission | Used by |
|---|---|
| `INTERNET` | LLM API calls and the ZynkSync/ZynkLink HTTP server in Rust; `OpenAiDictation.transcribe`; the `OpenAiRealtime` WebSocket; the download fallbacks in `VoskBridge.downloadModel` and `WakeWordBridge.downloadModels` |
| `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_DATA_SYNC` | `SyncForegroundService` (`FOREGROUND_SERVICE_TYPE_DATA_SYNC`) |
| `FOREGROUND_SERVICE_MICROPHONE` | `WakeWordService` (`FOREGROUND_SERVICE_TYPE_MICROPHONE` on Android 14+) |
| `FOREGROUND_SERVICE_SHORT_SERVICE` | `TimerRingService` (`shortService` type: no special-use declaration, three-minute system cap, we use 30 s) |
| `POST_NOTIFICATIONS` | the service notifications (sync, Hey Zynk, timers and alarms), the "Hands-free needs setup" notice from `WakeWordService.assistantUnavailable`, the `WakeBoot` and `PackageReplacedReceiver` notices, and `ZynkNotify` cards; requested at runtime on 13+ by `requestNotificationPermissionIfNeeded` |
| `READ_EXTERNAL_STORAGE` (maxSdkVersion 32) | requested only by `FolderPickerBridge.pick` on SDK <= 32 |
| `WRITE_EXTERNAL_STORAGE` (maxSdkVersion 29) | requested in `onCreate` on SDK <= 28 before `ensureShareDir()` |
| `CAMERA` | `AndroidCameraBridge.takePicture` |
| `RECORD_AUDIO` | `MicStream`, the only class that creates an `AudioRecord`. `WakeWordService`, `VoskBridge` and the assistant session all read through it |
| `MODIFY_AUDIO_SETTINGS` | declared; no direct use found in the Kotlin sources. vosk-android's `SpeechService` is no longer used. Whether anything still needs it is open (see "Facts to verify") |
| `com.android.alarm.permission.SET_ALARM` | `VoiceCommands.execute` clock-app fallback (AlarmClock intents) |
| `SCHEDULE_EXACT_ALARM` | `ZynkTimer`: the timer/alarm deadline is an exact alarm. A user-granted special permission on Android 12+, denied by default on 14+; the Voice page offers the switch (`VoiceCommandBridge.openExactAlarmSettings`). Not `USE_EXACT_ALARM`, which Play restricts to clock apps |
| `RECEIVE_BOOT_COMPLETED` | `TimerReceiver`: puts pending timers and alarms back after a reboot and calls `WakeBoot.tryStart` to restart the listener |
| `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS` | `MainActivity.maybeAskBatteryExemption` (KI-087): one system prompt, from the second app start |
| `WAKE_LOCK` | `WakeWordService` partial wake locks; `ZynkAssistantSession.wakeScreen` |

`MANAGE_EXTERNAL_STORAGE`, `USE_FULL_SCREEN_INTENT` and `READ_MEDIA_IMAGES` were removed on 2026-09-07 for Google Play and must not come back (CLAUDE.md).

Services. There are three foreground services, none exported: `.SyncForegroundService` (`foregroundServiceType="dataSync"`), `.WakeWordService` (`microphone`) and `.TimerRingService` (`shortService`). The assistant role needs three exported services: `.ZynkAssistantService` (permission `BIND_VOICE_INTERACTION`, intent filter `android.service.voice.VoiceInteractionService`, meta-data `android.voice_interaction` pointing at `@xml/voice_interaction_service`), `.ZynkAssistantSessionService` (same permission), and `.ZynkRecognitionService` (permission `BIND_RECOGNITION_SERVICE`, intent filter `android.speech.RecognitionService`, meta-data `android.speech` pointing at `@xml/recognition_service`). The voice-interaction XML declares `sessionService`, `recognitionService`, `supportsAssist="true"` and `supportsLaunchVoiceAssistFromKeyguard="true"`. The recognition service is a stub that answers `ERROR_CLIENT`; it exists only because the framework's `VoiceInteractionServiceInfo` parser rejects an assistant without one (comment in `ZynkRecognitionService.kt`).

Receivers: `.TimerReceiver` (timer and alarm actions, `BOOT_COMPLETED`) and `.PackageReplacedReceiver` (`MY_PACKAGE_REPLACED`, KI-101).

Providers: `ZynkShareProvider` (section 1) and a `FileProvider` (`${applicationId}.fileprovider`, paths in `@xml/file_paths`) that serves the camera output file and the zip made by `AndroidPaths.shareWakeClips`. `MainActivity` is `launchMode="singleTask"`, exported, with `MAIN`/`LAUNCHER` and `LEANBACK_LAUNCHER`.

The `<queries>` block declares intents rather than package names so the app can resolve and bind on Android 11+: browser (`VIEW` http/https), `TTS_SERVICE` and `RecognitionService` (the Pixel fix: without the TTS entry the engine resolved as blocked and TTS init failed), the clock actions (`SET_TIMER`, `SET_ALARM`, `SHOW_TIMERS`, `SHOW_ALARMS`, `DISMISS_ALARM`, `SNOOZE_ALARM`), `DIAL` tel, `SENDTO` smsto/mailto, calendar `INSERT`, `VIEW` geo and google.navigation, `WEB_SEARCH`, `MEDIA_PLAY_FROM_SEARCH`, `IMAGE_CAPTURE`, and `SEND` `*/*`.

The manifest still carries a comment about an OpenCL backend with no element under it. It is a leftover from the dropped GPU work (section 11).

## 4. Startup sequence in MainActivity

`MainActivity.onCreate` runs, in order:

1. `enableEdgeToEdge()`, then `super.onCreate`.
2. `ZynkCore.nativeInit(applicationContext)`. This hands the Rust core the Java VM, the application context and the `ZynkNotify` class, and loads `.env` (section 9).
3. The content view gets a fixed background (`0xFF181A20`) and an insets listener that pads for status bar, gesture bar and cutout, because the WebView cannot pad for them itself (comment in code; targetSdk 36 makes edge-to-edge mandatory).
4. `setShowWhenLocked(true)` and `setTurnScreenOn(true)` (or the legacy window flags below API 27).
5. On SDK <= 28 without `WRITE_EXTERNAL_STORAGE`: request it (`REQ_WRITE_STORAGE`), and create the share folder in the result callback. Otherwise `ensureShareDir()` now, which also runs the one-time move out of `Download/ZynkbotShare`.
6. The permission queue (`permissionQueue`, an `ArrayDeque` of lambdas) is filled in this order: `requestRecordAudioIfNeeded`, `requestNotificationPermissionIfNeeded`, `requestAssistantRoleIfNeeded`, `requestLocalNetworkPermissionIfNeeded`. `runNextPermissionRequest()` pops one. Each step either shows a dialog and returns, or calls `runNextPermissionRequest()` itself when nothing is needed. One at a time because Android shows one permission dialog per Activity at once, and the code comment records that firing all together silently dropped the microphone request. Microphone is first because the wake word needs it.
7. `extractVoskModelIfNeeded()`: if `files/vosk-model` is missing or empty, copy `assets/vosk-model` recursively (`copyAssetDir`) on a background thread.
8. `startSyncService()`: `startForegroundService` for `SyncForegroundService`.

The assistant-role step (`requestAssistantRoleIfNeeded`) runs once per install (`zynkbot_setup/asked_assistant_role`): on Android 10+, if `ROLE_ASSISTANT` is available and not held, it launches `createRequestRoleIntent(ROLE_ASSISTANT)`. If the role is still not held afterwards it shows a dialog, "Make Zynkbot your assistant". "Open settings" sets `awaitingAssistantPick` and calls `openAssistantPicker()`, which tries `ACTION_MANAGE_DEFAULT_APPS_SETTINGS` and then `ACTION_VOICE_INPUT_SETTINGS`. The queue resumes in `onResume`. Holding the role is what lets `WakeWordService` open an assistant session (section 6).

The local-network step exists because Android 16 (SDK 36) makes LAN access a runtime permission (`android.permission.ACCESS_LOCAL_NETWORK`, spelled as a string literal because older SDK stubs lack the constant); without an explicit request, ZynkSync pairing on 192.168.x.x was blocked.

The wake-word models are not unpacked in `onCreate`. `WakeWordBridge.isModelReady()` copies any missing file from `assets/wake-word-models/` to `files/wake-word-models/` on demand (ONNX Runtime needs a real file path), and `downloadModels()` does the same with a GitHub-release download as a last resort.

Bridges are registered in `onWebViewCreate(webView)`, which the generated `WryActivity.setWebView` calls once the Tauri WebView exists.

`WakeWordService` is not started in `onCreate`. Two things start it: the page, through `WakeWordBridge.start` (section 6), and `WakeBoot.tryStart`, which needs no page. `WakeBoot.tryStart` is called from `ZynkAssistantService.onReady`, from `TimerReceiver` on `BOOT_COMPLETED`, and from `PackageReplacedReceiver` after an update. It does nothing unless the stored "Hey Zynk" toggle is on (`zynk_boot/hey_zynk_enabled`), `start_on_boot` is on (the default) and the service is not already running. It uses the stored threshold (default 0.72), checks after 8 s, and posts "Hey Zynk is off since the restart" if the service is not up.

`onResume` does these things: tells the page (`window.__zynkResumed`); clears the `ZynkNotify` card notification; calls `maybeAskBatteryExemption()`, which from the second app start asks once for the battery-optimisation exemption (`ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS`, preferences `zynkbot_power`); passes a pending share-folder change to the page (`ZynkShareProvider.takeChanged`, `window.__zynkShareChanged`); resumes the WebView and nudges `window.__nativeTurns()`; repaints a stale WebView surface (KI-040, `forceWebViewFrame`); resumes the permission queue after an assistant pick; calls `ZynkAssistantSession.current?.hideOverlay()` so the Z overlay does not sit over the app; and opens the assistant picker when the intent carries the extra `open_assistant_picker` (set by the "Hands-free needs setup" notification).

`onNewIntent` only stores the intent. The old path that delivered a wake transcript to the page (`wake_word_transcript`, `window.__handleScreenOffTranscript`) is gone. `onBackPressed` calls `moveTaskToBack(true)` (2026-10-01), so Back leaves the app running.

## 5. The bridges (JavascriptInterface objects)

Six objects are registered with `webView.addJavascriptInterface(...)` in `MainActivity.onWebViewCreate`. Results go back to the page by `evaluateJavascript` calling `window.__*` callbacks the page installs beforehand.

**`AndroidFolderPicker`** (`FolderPickerBridge`)
- `pick()`: request legacy storage permission if SDK <= 32, then `OpenDocumentTree`; resolves `window.__fpResolve(path)` / `__fpReject`.

**`AndroidPaths`** (`ZynkbotPathsBridge`)
- `setVoiceInputSource(src)`: store `'vosk'` or `'openai'` in SharedPreferences `zynkbot_voice/input_source`.
- `setFruitlessTone(on)`: store whether the closing tone plays when a wake led to nothing (`zynkbot_voice/fruitless_tone`).
- `keepScreenOn(on)`: hold the screen on while a large ZynkLink transfer runs.
- `getShareDir()`: absolute path of the share folder (`filesDir/ZynkbotShare`), created if missing.
- `openShareFolder()`: opens the Files app on the Zynkbot location. It tries, in order, `android.provider.action.BROWSE` on the provider root, `ACTION_VIEW` on the root, `ACTION_VIEW` on the root document, `VIEW_DOWNLOADS`, and then shows a Toast.
- `shareFileWritten(path)`: a peer download or an added file landed; refresh any Files-app view.
- `saveToGallery(path)`: copy an image from the share folder into `Pictures/Zynkbot` through MediaStore; returns `""` or an error.
- `saveImageBase64ToGallery(base64, mime)`: the same for image data the page holds.
- `moveFileTo(path)`: the model page's "Move to": opens the system save-as picker (`CreateDocument`) for a model file; `__zmvResolve` / `__zmvReject`.
- `pickFile()`: `OpenMultipleDocuments`, copies each picked file into ZynkbotShare on a thread; `__zfpResolve('<count>')` / `__zfpReject`.
- `pickImages()`: `PickMultipleVisualMedia(20)`, image only; `__pickFilesResolve([uri...])` / `__pickFilesReject`.
- `pickDocuments()`: `OpenMultipleDocuments("*/*")`; same callbacks.
- `readFileBase64(uri)`: read a `content://` URI through `ContentResolver`, base64, no wrap.
- `readImageJpegBase64(uri, maxSide)`: the picture decoded by Android, turned upright, scaled so the long side is at most `maxSide`, returned as JPEG in base64.
- `readFileText(uri)`: read a `content://` URI as UTF-8 text.
- `probeFile(uri)`: classify a picked file as image, pdf, doc, docx, text or binary.
- `copyToCache(uri)`: copy a picked file into the app cache so Rust can read it by path.
- `getMimeType(uri)`, `getFileName(uri)`: from the resolver; the name falls back to the last path segment.
- `copyToKnowledgeBase(uri, userId)`: copy into `files/zynkbot/knowledge_base/<userId>/`, never overwriting (`name (2).ext`); returns the new path or `""`.
- `getWakeClipStats()`, `shareWakeClips()`: counts of the saved wake clips, and a zip of them handed to the system share sheet through the FileProvider.
- `openUrl(url)`: `ACTION_VIEW` for http/https only.
- `getDeviceInfo()`: manufacturer, model, Android version, SDK.

**`AndroidCamera`** (`AndroidCameraBridge`)
- `takePicture()`: request `CAMERA` if needed, then `TakePicture` into a cache temp file through the FileProvider; `__camResolve(path)` / `__camReject`.

**`VoskBridge`** (the in-app mic button, both engines; section 8)
- `isModelReady()`: `files/vosk-model` exists.
- `getModelDir()`: its path.
- `startListening()`: request `RECORD_AUDIO` if missing, then start a `Dictation` fed from the bridge's own `MicStream("VoskBridge")`. The engine follows the stored selector: OpenAI when the choice is `'openai'` and a key is on file, else Vosk. Fires `__voskStarted`, `__voskLive(text)`, `__voskPartial(text)`, `__voskResult(text)`, `__voskError(msg)`.
- `stopListening()`: `dictation.stop()`; the final result arrives through `__voskResult`.
- `downloadModel()`: download `vosk-model-small-en-us-0.15.zip` from alphacephei.com into `files/vosk-model`; `__voskDownloadProgress(n)`, `__voskModelReady`, `__voskDownloadError`. No page code calls it at this commit.

**`WakeWordBridge`**
- `isModelReady()`: the five wake-word files are present, extracting from assets if not.
- `start(threshold)`: refuse without `RECORD_AUDIO` (`__wakeWordError`); wire `NativeVoiceAnswerer.onSpeakingChanged` to `__nativeSpeaking(bool)`, `onTurnCompleted` to `__nativeTurns()`, `WakeWordService.onStateChanged` to `__wakeWordState(name)` and `ZynkTimer.onChanged` to `__timerChanged()`; store the choice with `WakeBoot.setHeyZynkEnabled(true, threshold)`; `startForegroundService(WakeWordService)` with extras `threshold` and `modelDir`. A refused background start is caught and reported through `__wakeWordError`.
- `stop()`: `WakeBoot.setHeyZynkEnabled(false, ...)` and `stopService(WakeWordService)`.
- `state()`, `wakeState()`: the microphone state by name: `off`, `loading`, `listening`, `dictating`, `turn`, `paused`.
- `enrollmentStatus()`: JSON with `state` (`none`, `active`, `done`), `count`, `needed`, `enforcing`, `profile`. The Voice page card asks every 1.5 s while it is open, which is also how the service knows the card is open.
- `startEnrollment()`: start teaching again from nothing (section 7).
- `markAnswered()`: the page answered a wake question; calls `WakeWordService.reportOutcome(true)`.
- `startOnBoot()`, `setStartOnBoot(on)`: the "start after a restart" setting in `WakeBoot`.
- `stopSpeaking()`: `NativeVoiceAnswerer.stopSpeaking()` (the in-app Stop button).
- `isNativeSpeaking()`: `NativeVoiceAnswerer.speaking`.
- `drainNativeTurns()`: JSON array of finished hands-free exchanges, cleared on return.
- `log(message)`: breadcrumbs from the page into the log (tag `ZynkWeb`).
- `downloadModels()`: extract from assets, else fetch from the `wake-word-models` GitHub release; `__wakeWordDownloadProgress`, `__wakeWordModelReady`, `__wakeWordDownloadError`.

**`VoiceCommandBridge`** (in-app typed or dictated commands, `App.jsx handleSendMessage` via `parseVoiceCommand`)
- `setTimer(seconds)`, `setAlarm(hour, minute, label)`, `startStopwatch()`, `cancelTimer(seconds)`, `cancelAlarm(hour, minute)`, `checkTimer()`, `checkAlarm()`, `stopStopwatch()`, `resetStopwatch()`, `checkStopwatch()`: since 2026-10-25 each builds a `VoiceCommands.Cmd`, runs `VoiceCommands.execute` and returns the spoken-style line from `VoiceCommands.confirmation`, so typed and hands-free commands share one code path. `cancelTimerById(id)`, `cancelAlarmById(id)` serve the chips. `timerState()` returns the JSON the chips render (`timers[]`, `alarms[]`, stopwatch, `ringing`, `exact_allowed`); `ZynkTimer.onChanged` fires `window.__timerChanged()` on every change. `stopRinging()`, `canScheduleExact()`, `openExactAlarmSettings()`.

Web-side users (files under `zynkbot_rust/src/` that name each bridge):

- `AndroidFolderPicker`: `components/KnowledgeBasePanel.jsx`.
- `AndroidPaths`: `App.jsx`, `hooks/useVoiceInput.js`, `hooks/useVoiceSession.js`, and the components `ChatMessage.jsx`, `EnsembleModal.jsx`, `KnowledgeBaseManager.jsx`, `ReportProblemModal.jsx`, `SetupWizard.jsx`, `ZChatModal.jsx`, `ZynkFileBrowserModal.jsx`, `ZynkLinkPanel.jsx`, `ZynkSyncPanel.jsx`.
- `AndroidCamera`: `App.jsx`.
- `VoskBridge`: `components/VoiceButton.jsx`, `components/VoiceModal.jsx`, `hooks/useVoiceInput.js`.
- `WakeWordBridge` and `VoiceCommandBridge`: `App.jsx`, `components/VoiceModal.jsx`, `hooks/useVoiceSession.js`.

## 6. The hands-free pipeline end to end

The pipeline was rebuilt on 2026-10-20 around three rules: one state machine decides who has the microphone (`MicState`), one object reads the microphone (`MicStream`), and every change goes through one function (`WakeWordService.apply`). Dictation no longer opens its own recorder inside the service. It is a sink on the same stream.

**The state machine (`MicState.kt`).** States: `OFF`, `LOADING`, `LISTENING`, `DICTATING`, `TURN`, `PAUSED`. Events: `PageOn`, `PageOff`, `ModelsLoaded`, `ModelsFailed`, `Wake`, `DictationStarted(ownsTurn)`, `DictationEnded`, `TurnBegun`, `TurnEnded`, `ForeignStarted`, `ForeignEnded`, `StreamLost`, `Expired`. An event that makes no sense in the current state is refused. The class has no Android types, so it can be tested without a phone.

**The stream (`MicStream.kt`).** One `AudioRecord` with source `VOICE_RECOGNITION` (KI-069, 2026-11-08), 16 kHz mono PCM16, buffer `max(minBuf, CHUNK_SAMPLES * 2 * 8)`. A thread reads 1280 samples (80 ms) at a time and hands each chunk to the current `sink`. The sink can be swapped while the stream runs: the detector, a `Dictation`, or nothing. The stream attaches an `AcousticEchoCanceler` when the device has one and writes the result to `WakeLog`. `onLost` reports a dead recorder; `close()` waits up to 500 ms for the thread.

**Starting the listener.** `useVoiceSession.js` `armWakeWord()` is the only page code that calls `WakeWordBridge.start`. It calls `start(0.72)` when the "Hey Zynk" toggle is on and the models are ready: on toggle change, on `visibilitychange`, and 5 s after in-app dictation or web TTS ends (the page calls `WakeWordBridge.stop()` while those run). `WakeBoot.tryStart` starts the service without the page (section 4). There is no in-app wake handler; every wake is handled natively.

**`WakeWordService.onCreate` and `onStartCommand`.** `onCreate` starts `WakeLog`, creates the notification channels, registers the screen receiver, the recording watch and the playback watch, and starts a watchdog that runs once a minute. `onStartCommand` reads `threshold` (default 0.5) and `modelDir`, calls `startForeground` once per service life (type `MICROPHONE` on Android 14+, `DATA_SYNC` on 10 to 13; any exception stops the service), pre-loads the Vosk model into `sharedVoskModel` on a thread, and calls `apply(MicState.Event.PageOn)`. It returns `START_STICKY`.

**`apply(event)`.** Synchronized. It passes the event to `MicState.on(event)` and carries out the actions that come back: `LOAD_MODELS`, `UNLOAD_MODELS`, `OPEN_STREAM`, `CLOSE_STREAM`, `SINK_DETECTOR`, `SINK_DICTATION`, `SINK_NONE`, `ABORT_DICTATION`. Then it pushes the new state to the page through `onStateChanged`. The public entry points are thin wrappers: `dictate(d, ownsTurn)`, `beginTurn()`, `endTurn()`, `stateName()`.

**Models (`loadModels`).** Runs on a thread and is guarded by a generation counter so a stale load cannot finish after a stop (KI-097). It creates ONNX Runtime sessions for the mel model, the embedding model, the classifier and, if the file is present, Silero VAD. Each session uses one thread with spinning off (battery, 2026-10-26). It then applies `ModelsLoaded` and calls `trainVerifierInBackground()`.

**The detector (`processChunk`).** Each 80 ms chunk goes through these steps in order.

1. The chunk is kept in `recentChunks` (`TRIGGER_CLIP_CHUNKS = 40`, 3.2 s). If a cooldown is running, the chunk is only counted down.
2. Quiet skip: after more than `QUIET_SKIP_AFTER = 25` chunks in a row below `COMPUTE_GATE_DB = -52.0`, no model is run.
3. Speech gate, first stage: Silero VAD runs on the chunk (`VAD_FRAME = 512` samples with `VAD_CONTEXT = 64`). If no voice was seen for more than `VAD_SKIP_AFTER_FRAMES = 62` frames, the wake models are not run. After `VAD_MAX_FAILURES = 5` errors in a row the gate is switched off for the rest of the run.
4. Mel model (`[1,1280]` in, 5 frames of 32 bins out) into `melBuffer` (`MEL_WINDOW = 76` frames). Embedding model (`[1,76,32,1]` in, 96 floats out) into `embBuffer` (`EMB_WINDOW = 16`). Classifier (`[1,16,96]` in, one score out). The mel output is not normalised with openWakeWord's `mel/10 + 2`, because `hey_zynk.onnx` was trained without it (comment in code; do not add it without retraining).
5. Two chunks in a row above `threshold` make a candidate. The checks below then run on that chunk. A candidate that fails one is ignored with half a cooldown.
6. Quiet gate: the level of the 3.2 s ring must be at least -65 dBFS when the VAD is loaded, else `SILENCE_GATE_DB = -46.0`.
7. Playback gate: if this phone was playing audio now or within `PLAYBACK_TAIL_MS = 1000` (`playbackRecently()`, fed by an `AudioManager.AudioPlaybackCallback`), the candidate is ignored and its clip is saved labelled false. The exception is a ringing timer (`ZynkTimer.ringing`).
8. Speech gate, second stage: the highest VAD score over the last 2 s (`vadRecentMax()`) must reach `VAD_SPEECH = 0.5`. Otherwise the clip is saved labelled false.
9. Enrollment: while teaching is active, a wake counts only when the Voice page card is open. It is saved as a real clip, the chime plays, and nothing else happens (section 7).
10. Voice profile: `currentVerifier()` loads the taught profile and reloads it when the file changes. If no profile enforces on this phone, the candidate is ignored and nothing is saved. If the profile's score is at or below its threshold, the candidate is ignored and the clip is saved.
11. Accepted: `saveTriggerClip`, a full cooldown (`COOLDOWN_CHUNKS = 50`, about 4 s), `onWake()`.

So without an enforcing voice profile the wake word answers nothing. The profile is described in `WakeVerifierTrainer.kt` as "a noise and TV filter, not a voice lock", and it must not be described as security.

**Strict mode is off.** The code for a stricter mode after repeated misses is still there (`STRICT_HITS = 4`, `STRICT_SCORE = 0.90`, `MISSES_TO_BACK_OFF = 3` within `MISS_WINDOW_MS`, `BACK_OFF_MS` of 10 minutes), but `STRICT_MODE_ENABLED = false` since 2026-09-20, so `noteOutcome` never enters it. Misses are still counted and labelled.

**Hand-off (`onWake`).** It stops a ringing timer, releases the audio-loop wake lock and takes a partial wake lock `zynkbot:screen_off_wake` for 25 s. On a thread it applies `Wake`, plays `R.raw.wake_chime` through `MediaPlayer` with `USAGE_MEDIA`, and calls `ZynkAssistantService.instance?.triggerSession()` (`showSession(Bundle(), 0)`). On success the wake lock is extended to 90 s. If there is no assistant service or the session is refused, `assistantUnavailable` records a miss, plays the close tone, posts one "Hands-free needs setup" notification per service run (its tap opens `MainActivity` with the extra `open_assistant_picker`), and ends the turn. The older fallback that dictated and answered inside the service, and handed a transcript to the page, was removed on 2026-11-04.

**`ZynkAssistantSession`.** `ZynkAssistantSessionService.onNewSession` returns a new session. `onShow`: `sessionActive = true`, `current = this`, stop a ringing timer, `wakeScreen()` (a `SCREEN_BRIGHT_WAKE_LOCK | ACQUIRE_CAUSES_WAKEUP | ON_AFTER_RELEASE` lock, `zynkbot:assistant_session`, `SCREEN_LOCK_MS = 90_000`), `setKeepAwake(true)`, show the Z overlay in `LISTENING` mode, `startListening()`. The overlay (`ZView`, `LOGO_DP = 120`, `BOTTOM_MARGIN_DP = 140`) is a Z in `0xFF50FA7B` that draws in over 450 ms. A tap on it calls `cancelByUser()`: report a miss, stop listening and speech, play the close tone only if the "fruitless tone" setting is on, hide.

Dictation in the session: `startListening` picks OpenAI when the selector is `"openai"` and an `OPENAI_API_KEY` is on file, else Vosk, and `startDictation` builds a `Dictation` with `timers = true`. The audio comes from the service's stream: `WakeWordService.dictate(d, ownsTurn = true)` swaps the sink. If "Hey Zynk" is off (the session was opened by the assistant gesture), the session opens its own `MicStream`. A cancelled dictation hides the session. A failed OpenAI dictation falls back to Vosk once.

`answerAndFinish(transcript)`: word gate (blank, fewer than 2 or more than `MAX_QUERY_WORDS = 60` words: report a miss, close tone only when the fruitless-tone setting is on, hide). `VoiceCommands.parse` handles timer, alarm and stopwatch locally: close tone, `VoiceCommands.execute`, `reportOutcome(true)`, and `NativeVoiceAnswerer.say` speaks the confirmation or `FAILED_LINE`. Otherwise the Z switches to `THINKING`, the close tone plays as a "sent" signal, and `NativeVoiceAnswerer.answer(context, transcript)` runs. `hide()` follows either way. `onHide`: `sessionActive = false`, release the screen lock, stop listening, stop the animation, and `WakeWordService.instance?.endTurn()`, which returns the microphone to the detector.

**`NativeVoiceAnswerer.answer`.** Sets `speaking = true`, notifies `onSpeakingChanged` (so the app's Stop button lights), and opens a turn with `WakeWordService.beginTurn()` if none is open, ending it afterwards. `answerInner` gets the process-wide `TextToSpeech` (`Locale.US`; `pickBestVoice`; `warmUp` speaks a silent utterance first), creates a `SentenceSpeaker`, and calls `ZynkCore.nativeSendMessage(transcript, "", "", "", "sovereign", callback)`. Empty `userId`, `sessionId` and `backend` mean "resolve in Rust".

The Rust side (`android_jni.rs`, `Java_ai_containai_zynkbot_ZynkCore_nativeSendMessage`) resolves identity through `user_identity::get_identity()` and the session through `get_current_session_id()` (or creates `voice-<uuid>`), emits a `voice-session` event with the id, picks the backend with `resolve_voice_backend()`, loads recent turns (`load_recent_turns`, 40 messages), appends `HANDS_FREE_NOTE` to the model-facing message, and runs `commands::chat::generate_reply(..., hands_free = true)` on a process-wide tokio runtime. A `JniSink` forwards `onToken`, `onEvent`, `onDone(replyJson)` and `onError` to Kotlin.

`SentenceSpeaker.feed` buffers tokens and speaks each completed sentence (a terminator followed by whitespace, at least 12 characters in) with `engine.speak(text, QUEUE_FLUSH for the first and QUEUE_ADD after, null, id)`. The `MEMORY_EXTRACT` and `WEB_SEARCH_NEEDED` markers cut the stream, and a trailing `MEMORY: yes|no` line is dropped before speaking (`withoutMarker`). `cleanForSpeech` strips markdown. An `onEvent("web-search")` resets the speaker and speaks `SEARCHING_LINE` ("Let me check that."). In `onDone`, a reply that starts with `NO_QUERY` is discarded and reported as a miss. Otherwise a `Turn` (session id, question, answer, time, pre-search answer, sources) is added to `pendingTurns` and `onTurnCompleted` nudges the page (`window.__nativeTurns`), which drains `WakeWordBridge.drainNativeTurns()`. `finish` speaks any remainder and waits up to 60 s for the utterances to settle. If anything was spoken, `WakeWordService.labelClip(true)` marks the wake clip as real. A failure speaks `ERROR_LINE`.

The chimes are on `USAGE_MEDIA`. TTS is called with null params and the code sets no audio attributes on the engine, so which stream a reply plays on is not decided in the code (see "Facts to verify").

**Recording watch.** `registerRecordingWatch` registers an `AudioManager.AudioRecordingCallback`. `onRecordingsChanged` treats any active recording whose `clientAudioSessionId` differs from `stream.sessionId` as someone else and applies `ForeignStarted`; when the last one ends it applies `ForeignEnded`. The state machine decides what follows.

**Watchdog.** Once a minute `watchdogTick` reopens a closed stream while the state is `LISTENING`, lifts a `PAUSED` that has lasted longer than `PAUSE_MAX_MS = 120_000`, and closes a `TURN` longer than `TURN_MAX_MS = 180_000` or a `DICTATING` longer than `DICTATION_MAX_MS = 60_000`.

**Wake locks.** `zynkbot:audio_loop` (partial) is held from `ACTION_SCREEN_OFF` to `ACTION_SCREEN_ON`, and taken in `openStream` if the screen is already off, so the detector keeps running with the screen dark. `zynkbot:screen_off_wake` (partial, 25 s then 90 s) covers one wake. `zynkbot:assistant_session` (screen bright) lights the screen for the session, up to 90 s.

**Logs.** `WakeLog` writes every hands-free line to logcat and to `files/zynkbot/logs/wake.log`. "Report a problem" can attach the last 200 lines (section 9).

## 6a. Zynkbot's own timer, alarm and stopwatch (2026-10-25)

Why not the clock app: a timer handed to it with the phone asleep was confirmed aloud and never rang (KI-061, GitHub #29); nothing can ask it the time left; neither test phone's clock app cancels a running timer from outside, and `ACTION_DISMISS_TIMER` only dismisses one already ringing. So `ZynkTimer` keeps the state itself.

- **State:** SharedPreferences `zynk_timer`: `timers` and `alarms` as JSON arrays (`id`, `end`/`at` ms, `total` seconds), `sw_start`/`sw_elapsed` for the stopwatch, `next_id`. Any process (the app, `WakeWordService`, the assistant session) reads the same state. Several timers and alarms at a time; the same alarm time twice is one alarm.
- **Deadline:** `AlarmManager.setAlarmClock(AlarmClockInfo(at, showIntent), firePending)` per entry (request codes 1000+id for timers, 3000+id for alarms; cancel actions 2000+id / 4000+id). `setAlarmClock` survives Doze and the process being killed and shows the alarm icon in the status bar. It needs `SCHEDULE_EXACT_ALARM`; `canScheduleExact()` is checked on every set and the clock-app fallback is used when it is false.
- **Firing:** `TimerReceiver` (`ACTION_FIRE`, `kind`, `id`) removes the entry and starts `TimerRingService` with the spoken line. The service goes foreground as `shortService` (Android 14+; plain foreground below), posts the "Timer done / Stop" notification, speaks the line with `NativeVoiceAnswerer.say` (which pauses the detector while it speaks), then loops the device's default alarm tone (`USAGE_ALARM`) until Stop or `RING_SECONDS = 30`. `ZynkTimer.ringing` is true meanwhile.
- **Stopping the ring:** the notification's Stop action, `ZynkTimer.stopRinging()` from the page (the orange button), the assistant session's `onShow`, and `WakeWordService.onWake`. The detector's playback gate (`playbackRecently()`) ignores wakes while the phone plays audio; it makes an exception while `ZynkTimer.ringing`, so "Hey Zynk" over the alarm tone depends on the echo canceller `MicStream` attaches, which is logged, not assumed.
- **Notifications:** channel `zynk_timer_v2` at default importance, silent (the first version was `IMPORTANCE_LOW`, and Android 16 folded it into the app's collapsed silent bundle with the ZynkSync and Hey Zynk entries, so the Cancel button was never seen). One countdown notification per timer (`setChronometerCountDown`) and one per alarm, ids 74000+id and 75000+id, each with Cancel → `TimerReceiver`/`cancelById`.
- **Reboot:** `TimerReceiver` on `BOOT_COMPLETED` → `rescheduleAfterBoot`: entries still in the future are re-armed and re-notified, past ones dropped.
- **Page:** `useVoiceSession` keeps `timerState` from `VoiceCommandBridge.timerState()` (push via `window.__timerChanged`, 1 s tick while something runs); `App.jsx` renders one chip per timer/alarm plus the stopwatch, and the full-width Stop while ringing. On the desktop `src/utils/zynkTimer.js` is the same shape in the page itself (rings only while the app is open; alarms are phone-only).
- **Parsing:** both parsers (`VoiceCommands.kt`, `useVoiceSession.js`) normalise dictation first: `a.m.`/`p.m.` → am/pm, `7.30` → `7:30`, punctuation to spaces, `stop watch` → `stopwatch`; `1026` → 10:26; a clock time (colon or am/pm) after "timer" is an alarm, since dictation hears "alarm" as "timer". Control words come before the duration patterns so "stop timer" never sets one. `cancel the 10 minute timer` / `cancel the 7 30 alarm` target one of several; a bare cancel with several set names them and asks.

## 7. Trigger clips, labels and the taught voice profile

**Clips.** `WakeWordService.saveTriggerClip(score, pooled, label)` runs on every accepted wake, every enrollment wake, and every candidate rejected by the playback gate, the second VAD stage or the voice profile. It writes, off the audio thread, to `files/zynkbot/wake_triggers/`:

- `<yyyyMMdd-HHmmss>-<score to 3 decimals>.wav`: 16 kHz mono PCM16, 44-byte RIFF header (`wavHeader`), the last 40 chunks (3.2 s) ending with the chunk that fired.
- `<same stem>.scores.txt`: the classifier score for each of those chunks.
- `<same stem>.feat`: the 192 pooled features the voice profile is trained on (`WakeVerifier.pooledFeatures`: the mean and the max of each of the 96 embedding dimensions).
- `<same stem>.real` or `<same stem>.false`: the label, when one is known.

**Retention.** Clips labelled real are never pruned. Of the rest, the newest `TRIGGER_CLIPS_KEPT = 20` are kept when a profile enforces, else `TRIGGER_CLIPS_KEPT_UNVERIFIED = 60`. Pruning deletes the WAV and the scores file. For a clip labelled false the `.feat` and the label stay, so it remains a training negative. For an unlabelled clip the `.feat` goes too.

**Labels.** `reportOutcome(useful)` and `labelClip(real)` reach `labelLastClip`. The last report wins (KI-076). A clip is labelled real when a clock command ran (`ZynkAssistantSession.answerAndFinish`), when the page answered a wake question (`WakeWordBridge.markAnswered`), or when a hands-free reply was spoken (`NativeVoiceAnswerer`, `labelClip(true)`). An answered clip whose profile score was below `TRAIN_REAL_MIN_SCORE = 0.80` is not used for training (KI-103). `NO_QUERY`, an empty or out-of-range transcript, a tap on the Z and a missing assistant service report a miss. A useful outcome clears the miss count and calls `trainVerifierInBackground()`.

**Teaching (enrollment).** The Voice page has a "Teach Zynkbot your voice" card (`components/VoiceModal.jsx`). `WakeWordBridge.startEnrollment()` calls `WakeWordService.startEnrollment`, which deletes the clips, the profile and `verifier_training.json`, and writes the marker `files/zynkbot/wake-word-models/enrolling` with `active`. While the marker is active and the card is open, each wake that passes the gates is saved as a real clip and answered with the chime only. The card counts as open for `CARD_OPEN_MS = 5000` after each `enrollmentStatus()` call. At `ENROLL_CLIPS = 20` the marker becomes `done` and training starts.

**Training on the phone (`WakeVerifierTrainer.kt`).** It needs `MIN_REAL_CLIPS = 20` real clips and retrains after every `RETRAIN_EVERY_NEW_REAL = 5` new ones. Negatives are the phone's own `.false` clips plus `assets/wake-word-models/generic_negatives.json` (110 clips). The model is a standardised, class-balanced, L2 logistic regression (C = 0.05) fitted by gradient descent and checked with five folds. The threshold is the 5th percentile of the owner's scores, kept within 0.05 and 0.5. A new profile replaces the old one only if its held-out numbers are not worse. The state is kept in `verifier_training.json`.

**Enforcement.** The trainer writes a version 3 profile to `files/zynkbot/wake-word-models/hey_zynk_verifier.json`. It writes `owners: ["*"]` when the profile it replaces already enforced, or when the held-out numbers clear both bars (`AUTO_ENFORCE_KEEP = 0.9` and `AUTO_ENFORCE_STOP = 0.9`). Otherwise the owners list is empty: the profile is scored and logged, and the wake word stays off. `WakeVerifier.enforcesOn(context)` is true when `owners` holds `"*"` or this user's id. `WakeVerifier.score` reads version 3 profiles (192 "meanmax" features) and the older version 2 layout (1728 features).

**Sync and backup.** `wake_profile.rs` carries the profile to the user's other devices under the name `ZYNK_WAKE_VOICE_PROFILE` (in `sync_secrets`, through the outbox) and into the cloud backup. A received profile is enforced as it is.

**Lab tools.** `labs/wake_verifier/train.py` and the other scripts in that folder are desktop tools for experiments. The app does not use them. `AndroidPaths.shareWakeClips()` zips the clips for the share sheet; on a debug build they can also be read with `adb shell run-as ai.containai.zynkbot`.

## 8. Dictation engines

There are two engines, Vosk (on the phone) and OpenAI (over the network), one class that runs either (`Dictation`), and one selector.

**Selector.** The Voice settings choice is stored in `localStorage` as `zynkbot_voice_input_source` (`'vosk'` default or `'openai'`). `useVoiceSession.setVoiceInputSource` also calls `AndroidPaths.setVoiceInputSource(val)`, and the same value is pushed again just before arming the wake word, so a stale native copy is corrected. Kotlin stores it in SharedPreferences `zynkbot_voice`, key `input_source`, and `MainActivity.voiceInputSource(context)` reads it. `EnvFile.read(context, "OPENAI_API_KEY")` decides whether the OpenAI choice is usable.

**`Dictation` (`Dictation.kt`).** `Dictation(engine, voskModel, apiKey, timers, listener, stream)` is a `MicStream.Sink`. It never opens the microphone; a `MicStream` feeds it. `Engine` is `VOSK` or `WHISPER`. With `timers = true` (hands-free) it stops after `SILENCE_MS = 1500` of quiet, after `NO_SPEECH_MS = 6000` with no speech at all, and at `CAP_MS = 30_000`. With `timers = false` (the mic button) the user stops it. `onFinal(null)` means cancelled or failed, `""` means nothing was heard.

- Vosk: `org.vosk.Recognizer` on the shared model (`Dictation.voskModel(context)`, which loads `WakeWordService.sharedVoskModel` once). The model is the bundled `assets/vosk-model` (68 MB on disk), extracted to `files/vosk-model` at first launch.
- OpenAI (`Engine.WHISPER`): the chunks go to `OpenAiDictation.Endpointer` and, when streaming is on, to an `OpenAiRealtime` live session. The result is the live text passed through `OpenAiDictation.guardSilence`, or, without a live result, `OpenAiDictation.transcribe(key, toWav(pcm))`.

**`OpenAiDictation.kt`.** It no longer records. `Endpointer(timers).feed` decides when speech starts and ends: onset is `SPEECH_ONSET_CHUNKS = 2` chunks in a row above `max(noise floor * ONSET_FLOOR_MULT (2.5), MIN_SPEECH_RMS (120))`; `PRE_ROLL_MS = 500` before the onset is kept; no onset within `SPEECH_WAIT_MS = 8_000` gives `""`; `TRAILING_SILENCE_MS = 1_200` of quiet ends it, where quiet also means under `PEAK_QUIET_RATIO = 0.12` of the loudest speech; `MAX_TOTAL_MS = 30_000` caps a hands-free recording and `PAGE_CAP_MS = 180_000` caps the mic button. `transcribe` posts the WAV as multipart to `https://api.openai.com/v1/audio/transcriptions` with `model` = `MODEL = "gpt-4o-mini-transcribe"` (2026-10-28; it was `whisper-1`), `language=en`, `response_format=json`, `HTTP_TIMEOUT_MS = 20_000`. `guardSilence` drops the stock phrases the model returns for silence, for clips under 2.5 s. `streamingEnabled(context)` is true unless `.env` sets `DICTATION_STREAM` to `0`, `false`, `off` or `no`.

**`OpenAiRealtime.kt`.** A WebSocket (okhttp) to `wss://api.openai.com/v1/realtime?intent=transcription`. Audio is resampled to `TARGET_RATE = 24_000`. The session uses `OpenAiDictation.MODEL` and server-side voice detection with 700 ms of silence. `CONNECT_TIMEOUT_MS = 5_000`, `FINAL_WAIT_MS = 3_000`. If the live session gives nothing, `Dictation` falls back to the upload.

**Mic button in the app (`useVoiceInput.js`).** `isAndroid()` is `!!window.VoskBridge`. On Android both engines go through `VoskBridge.startListening()` / `stopListening()`; Kotlin reads the selector itself. `__voskLive` and `__voskPartial` carry text while the user speaks, `__voskResult` the final text. `VoskBridge` builds the `Dictation` with `timers = false` and feeds it from its own `MicStream("VoskBridge")`, not from the service's stream. The page calls `WakeWordBridge.stop()` while the mic button dictates, so the detector is off meanwhile; a code comment gives that as the reason the button keeps its own stream for now, and calls sharing the service's stream a later step. Before 2026-10-28 the page recorded OpenAI audio through the WebView's `getUserMedia`; that path is no longer used on Android.

**Hands-free.** `ZynkAssistantSession.startDictation` builds the same `Dictation` with `timers = true` and gets its audio from the service's stream (section 6).

**Rust.** The Rust recording commands refuse on Android by `#[cfg(target_os = "android")]` in `lib.rs`: `start_vosk_recording` and `stop_vosk_recording` return "Android uses the VoskBridge Kotlin interface.", and `start_openai_recording` and `stop_openai_recording` return a message that still names the old WebAudio path. `vosk_desktop.rs` (cpal plus the `vosk` crate) is compiled only for Linux and Windows (`Cargo.toml` target sections).

## 9. Rust on Android

`Cargo.toml` `[lib] name = "app_lib"`, `crate-type = ["cdylib", "rlib"]`, so the Rust core is `libapp_lib.so`. The generated `Rust.kt` loads it (`System.loadLibrary("app_lib")`) and `ZynkCore` loads it again (idempotent) so the JNI door works even if called first.

Two doors into one core:

1. The WebView. Tauri IPC (`Rust.kt ipc`) dispatches to the `#[tauri::command]` functions registered in `lib.rs run()`; chat goes through `commands::chat::generate_reply` with a `TauriSink`.
2. `ZynkCore`, two `#[no_mangle] extern "system"` functions in `android_jni.rs` (compiled only with `#[cfg(target_os = "android")]`, declared as `mod android_jni` in `lib.rs`).
   - `nativeInit(context)`: calls `ensure_env_loaded()` and keeps the Java VM, the context and the `ZynkNotify` class, so `android_jni::post_notice(title, text)` can post a notification later (used by `response_sink.rs`).
   - `nativeSendMessage(...)`: calls the same `generate_reply` with a `JniSink`. It keeps one `tokio` multi-thread runtime for the life of the process in a `OnceCell`, because `generate_reply` spawns background memory work that needs a runtime handle.

`ensure_env_loaded()` loads `get_app_data_dir()/.env` with `env_file::load`, so a hands-free turn has its keys when the service was started without the app screen (2026-11-03).

**The on-device model.** The phone runs a local GGUF model on the CPU through `llama-cpp-2` with the `dynamic-backends` feature (`Cargo.toml`, Android dependencies). `llm/local_models.rs` loads the CPU modules by name with `ggml_backend_load`, skips the armv9 variants (cap at armv8.6, 2026-10-24) and asks for no GPU layers on Android. `resolve_default_model_path()` uses `LOCAL_MODEL_PATH` or the first `.gguf`, sorted by name, in `models/user`. An OpenCL backend was built and measured on 2026-10-23 and 24 and dropped (`Cargo.toml` comment: a third of the CPU's speed on the Adreno 740, refused on the Pixel's PowerVR).

**Reply prompt and memory work with a local model.** The same `generate_reply` runs the memory pipeline described in `MEMORY_PROCESSING_PIPELINE.md`. The switch is `on_device` in `commands/chat.rs`, and it follows the backend name (it ends in `.gguf` or contains `local`), not the platform. With a local model selected:

- The reply prompt uses the short instruction block in `conversation_engine.rs`, rewritten on 2026-11-09 (KI-106). It is about 200 words; the test `on_device_instructions_stay_short` holds it under 215. It carries 8 history messages cut to `ON_DEVICE_MESSAGE_CHARS = 400` each (the window moves in blocks of four, `ON_DEVICE_HISTORY_BLOCK`, so the engine keeps the same prompt prefix) and 4 recalled memories cut to `ON_DEVICE_MEMORY_CHARS = 300`.
- No `MEMORY: yes|no` marker is asked for, so every message that passes the first gate (not trivia, not a rewrite request) goes on to extraction, which decides.
- Extraction uses the `EXTRACTION_SCHEMA_SLIM` grammar (`memory_prompts.rs`): fields `fact`, `title`, `event_date`, `passing`, of which `fact` and `title` are required.
- The fact is stored first. The links are asked for afterwards (`links_after_store` in `commands/chat.rs`): `ask_llm_for_links_only` with at most 4 candidates, then `filter_local_links` (`lib.rs`), which applies the confidence floor, drops some link types and turns `updates` into `contradicts` so the card asks the user. A contradiction is also stored first and asked about after.
- All of it runs in the `memory_jobs` worker, so a message's facts can appear in the Memory Manager some time later. The Memory Manager says so when the device model is selected (`MemoryManagerModal.jsx`).

With a cloud model selected on the phone, the pipeline is the cloud one.

**Backend choice for hands-free.** `resolve_voice_backend()` in `android_jni.rs` uses `ZYNK_MODEL_BACKEND` when it is usable: a cloud provider with its key, or a local choice when `resolve_default_model_path()` finds a model file. Otherwise it falls through to Anthropic, OpenAI, xAI, Mistral and the custom endpoint, and returns `None` when none is configured. (Its doc comment still says there is no on-device model; the code below it handles one.) `get_best_available_backend` in `commands/chat.rs` skips its scan of `models/user` for a `.gguf` file on Android.

**Where output goes.** `lib.rs` shadows `println!` and `eprintln!` crate-wide with macros that call into `app_log.rs`. Each line goes to the real stdout or stderr, to a ring buffer of `CAPACITY = 800` lines, and to the on-disk log `logs/zynkbot.log` under the app data directory (KI-082, three generations). `commands/report.rs` builds "Report a problem" from the on-disk log first and the ring buffer second, redacted, and can append up to 200 lines of `logs/wake.log`. On Android, stdout and stderr are also redirected to logcat by `tao`'s `ndk_glue.rs` (tao 0.35.3 in `Cargo.lock`) with tag `RustStdoutStderr`, so `adb logcat -s RustStdoutStderr` shows the Rust log. Kotlin tags include `WakeWordService`, `Dictation`, `OpenAiDictation`, `OpenAiRealtime`, `ZynkAssistantSession`, `ZynkAssistantService`, `NativeVoiceAnswerer`, `WakeVerifier`, `WakeVerifierTrainer`, `WakeBoot`, `PackageReplaced`, `VoiceCommands`, `ZynkTimer`, `TimerRingService`, `SyncForegroundService`, `MainActivityLifecycle`, `ZynkWeb` (page breadcrumbs), and `Tauri` (generated `Logger.kt`).

**Startup in `lib.rs run().setup`.** `app_log::announce()`; create `models/system` and `models/user` under the data dir; load `.env` from `get_app_data_dir()/.env` through `load_env_file`, which calls `env_file::load` (not the `dotenv` crate, KI-083), with dev-path fallbacks otherwise; register the dialog, shell and opener plugins; warm up BERT NER on a thread; on another thread `db::create_pool()` runs `sqlx::migrate!("./migrations")`, then `conversation_history::ensure_tables`, `seed_system_memories`, and 20 s later `index_system_documentation`; then `auto_start_http_server()` starts the HTTP server for ZynkSync and ZynkLink.

**Android-specific Cargo settings.**

- `[target.'cfg(target_os = "android")'.dependencies]`: `llama-cpp-2` with `dynamic-backends`, `llama-cpp-sys-2` (so `local_models.rs` can call `ggml_backend_load`), `openssl` with `vendored`, and `jni = "0.21"`.
- `cpal` and the plain `llama-cpp-2` are under `cfg(not(target_os = "android"))`. `vosk` is Linux and Windows only.
- `zynkbot_rust/.cargo/config.toml` sets `rustflags = ["-C", "target-feature=+fp16"]` for `aarch64-linux-android` (needed by candle's gemm). The NDK paths are not committed. The file still has `OPENCL_INCLUDE_DIR` and `OPENCL_LIBRARY` entries pointing at `src-tauri/android-opencl`, and that folder still exists. Both are leftovers from the dropped OpenCL backend.
- The dev profile sets `opt-level = 3` for twenty packages: `candle-core`, `candle-nn`, `candle-transformers`, `gemm` and its six variants, `half`, `matrixmultiply`, `ndarray`, `rayon`, `rayon-core`, `safetensors`, `tokenizers`, `sqlx-core`, `sqlx-sqlite` and `libsqlite3-sys`. A comment records that an unoptimised debug build made a hands-free answer take 14 s on the OnePlus (2026-09-03). `llama-cpp-sys-2` is also built at `opt-level = 3` with `debug = false` in the dev profile, because the NDK r30 clang crashes on its debug info (2026-11-04).

Files that contain `target_os = "android"` (nine): `android_jni.rs`, `commands/chat.rs`, `commands/knowledge_base.rs`, `db.rs`, `lib.rs`, `llm/local_models.rs`, `response_sink.rs`, `seal.rs`, `user_identity.rs`.

## 10. Building and signing

Builds are made locally, on the desktop or the laptop. The workflow files are described below as files only.

**Local (from `zynkbot_rust/CLAUDE.md`).** Android Studio's bundled JDK is required; the system Java 8 does not compile Gradle:

```
JAVA_HOME=~/android-studio/jbr npm run tauri android build -- --debug --apk --target aarch64   # test build
JAVA_HOME=~/android-studio/jbr npm run tauri android build -- --aab --target aarch64            # Play upload
```

A debug build is needed for `run-as` access to `files/` (per the `Cargo.toml` comment). CLAUDE.md adds three points for the debug build: the NDK's tool folder must be on `PATH`, the APK is about 1.4 GB, and the exit code and the APK's version stamp (`aapt2 dump badging`) should be checked before the output is used, because a failed build leaves the previous APK in place.

The Tauri CLI regenerates the gitignored files (`tauri.properties` with `versionName`/`versionCode` derived from `tauri.conf.json`'s `version`, `tauri.build.gradle.kts`, `tauri.settings.gradle`, `assets/tauri.conf.json`), then runs Gradle. Gradle's `rust` plugin (`buildSrc/.../RustPlugin.kt`) defines product flavours `universal`, `arm64`, `arm`, `x86`, `x86_64` on an `abi` dimension, and each `rustBuild<Arch><Profile>` task (`BuildTask.kt`) calls back into the CLI to produce the `.so` for that ABI, which is merged into `jniLibs`. Gradle 8.14.3 (wrapper), Android Gradle Plugin 8.11.0, Kotlin 1.9.25. `compileSdk = 36`, `targetSdk = 36`, `minSdk = 26`.

The Tauri-generated `assets/tauri.conf.json` says `minSdkVersion: 24` (Tauri's default). Gradle's `minSdk = 26` is what the APK is built with, so Android 8.0 is the floor.

Because `settings.gradle` applies the gitignored `tauri.settings.gradle` unconditionally, calling `gradlew` directly on a fresh checkout fails. The build has to go through `tauri android build` (KI-034).

**Signing.** `build.gradle.kts` reads `gen/android/keystore.properties` (`storeFile`, `storePassword`, `keyAlias`, `keyPassword`; template in `keystore.properties.template`) into the `release` signing config. `check(ksFile != null)` fails configuration if the file is missing, for debug builds too. A release build is signed by Gradle with the release key. The `debug` build type sets no signing config of its own, keeps JNI debug symbols (`jniLibs.keepDebugSymbols` for all four ABIs) and allows cleartext traffic (`usesCleartextTraffic` placeholder true). CLAUDE.md gives a recipe for signing a debug build by hand with the debug keystore:

```
~/Android/Sdk/build-tools/<version>/apksigner sign \
  --ks ~/.android/debug.keystore --ks-pass pass:android \
  --out /tmp/app-signed.apk <unsigned.apk>
```

Whether the debug APK that Gradle produces still needs that step is not verified against the code (see "Facts to verify").

**The workflow files.** Two files under `.github/workflows/` describe automated builds. This section says what the files contain. It does not say they are run.

- `release.yml` has triggers for tags `v*` and for a manual run, and two jobs, `release` (desktop) and `release-android`. `release-android` does not depend on the desktop job. Its steps: Ubuntu; Temurin Java 17; Android SDK packages `platform-tools`, `platforms;android-36`, `build-tools;36.0.0`; `sdkmanager "ndk;30.0.15729638"` with `NDK_HOME` and `ANDROID_NDK_HOME` set and the NDK's `llvm` tool folder on `PATH`; Node 20; Rust stable with the `aarch64-linux-android` target; `libssl-dev perl`; the keystore decoded from the secret `ANDROID_KEYSTORE_BASE64` into `$HOME/keystores/zynkbot-release.jks`, and `keystore.properties` written from `ANDROID_STORE_PASSWORD`, `ANDROID_KEY_ALIAS`, `ANDROID_KEY_PASSWORD`; `npm ci`; then `npm run tauri android build -- --apk --aab --target aarch64`. The outputs are `app-universal-release.apk` and `app-universal-release.aab`. On a tag the APK is attached to the GitHub release as `Zynkbot_<version>.apk`, with the version taken from `tauri.conf.json`. A comment in the file records that the job called `gradlew` directly until 2026-09-09 and could not work that way (KI-034). CLAUDE.md records the current form as verified on v0.9.6-beta5 (2026-10-27).
- `test.yml` has four jobs: `guards`, `unit-tests`, `android-check` and `kotlin-check`. `android-check` installs NDK 27.0.12077973 and runs `cargo ndk -t arm64-v8a -P 26 check --lib`. It checks arm64 only (the file says "No armv7 check") and builds no APK. `kotlin-check` builds an unsigned debug APK and discards it, only when a `.kt` file, `AndroidManifest.xml` or `build.gradle.kts` changed or on a manual run.

**16 KB page-size alignment.** Google Play requires 16 KB-aligned native libraries for apps targeting Android 15+ (this app targets 36). The dependencies are `vosk-android 0.3.75` and `onnxruntime-android 1.29.0` (both in `app/build.gradle.kts`), and `libapp_lib.so` is built with NDK 30. The repository sets no `-Wl,-z,max-page-size` flag, so the alignment comes from the NDK's default linker settings.

The set of native libraries is larger than it was before the on-device model. `jniLibs/arm64-v8a` from the local debug build of 2026-11-10 holds 13 files: `libapp_lib.so`, `libc++_shared.so`, `libggml.so`, `libggml-base.so`, seven `libggml-cpu-android_*` modules, `libllama.so` and `libllama-common.so` (the last is copied for debug builds only, per CLAUDE.md). `readelf -lW` on those 13 files shows every `LOAD` segment at `0x4000`. The four libraries that come from the AARs (`libvosk.so`, `libonnxruntime.so`, `libonnxruntime4j_jni.so`, `libjnidispatch.so`) are not in that folder and were not checked this time, and no `zipalign` check was run on a current APK. The last full check on record was build50 on 2026-09-09, before the larger set. To verify a built APK:

```
unzip -o app.apk 'lib/arm64-v8a/*' -d /tmp/apk
llvm-readelf -l /tmp/apk/lib/arm64-v8a/*.so | grep -E 'LOAD|Align'   # every LOAD must show Align 0x4000
~/Android/Sdk/build-tools/36.0.0/zipalign -c -P 16 -v 4 app.apk          # checks the APK's own .so alignment
```

**Versions.** `src-tauri/tauri.conf.json`, `Cargo.toml` and `package.json` say `0.9.6`. The generated `tauri.properties` says `versionName 0.9.6`, `versionCode 9006`.

## 11. Things that differ from desktop

- **Local GGUF models run on the CPU** (since 2026-10-23): `llama-cpp-2` with `dynamic-backends`, modules loaded by name, armv8.6 cap, no GPU (OpenCL was tried and dropped; section 9).
- **No Rust-side dictation.** `vosk_desktop.rs` and `cpal` are desktop only. On Android both the mic button and hands-free use the Kotlin `Dictation` class (section 8).
- **No opening folders, files or URLs from Rust.** `commands/knowledge_base.rs` returns "not supported on Android" for the open-folder, open-file and open-URL commands; the page hides the KB "Open folder" button when `window.AndroidFolderPicker` exists (`KnowledgeBasePanel.jsx`) and uses `AndroidPaths.openUrl` for links.
- **File access is by `content://` URI.** Picked files are read through the `AndroidPaths` readers (`ContentResolver`), not by path in Rust; the camera capture and `copyToCache` are the exceptions (a real path in the app cache). Files enter the KB and ZynkbotShare by copy (section 1).
- **Identity file location.** Android keeps `.zynk_*` files under `files/zynkbot`; desktop keeps them under `dirs::config_dir()/Zynkbot`.
- **Input behaviour.** Enter sends on desktop; on Android (`window.AndroidPaths` present) Enter inserts a newline (`App.jsx`).
- **Voice output.** In-app TTS on desktop is web audio; the hands-free reply on Android is Android `TextToSpeech`, and the app's Stop button is wired to both (`useVoiceSession.stopTts`).
- **Background services.** `SyncForegroundService` runs from launch so ZynkSync/ZynkLink keep working; there is no equivalent on desktop.
- **Assistant role and wake word.** Desktop has no wake word. On Android it depends on the `VoiceInteractionService` registration, the `microphone` foreground service, the assistant role, and a taught voice profile that enforces (section 7).
- **Restart behaviour.** `WakeBoot`, `TimerReceiver` and `PackageReplacedReceiver` bring the listener back after a reboot or an update, and the app asks once for the battery-optimisation exemption (section 4).
- **Notifications from Rust.** `android_jni::post_notice` posts a silent notification through `ZynkNotify` when a card is raised while the app is not in front.
- **Local network permission.** Android 16 needs `ACCESS_LOCAL_NETWORK` at runtime for ZynkSync pairing.
- **Bundled Vosk model.** The APK ships the model in assets and copies it out at first launch; desktop looks for it through `vosk_desktop::find_model_dir`.
- **Seals.** `seal.rs` writes seals into `ZynkbotShare/seals` on Android.

## Facts to verify

These are the points the code does not settle. Items from the earlier list that the code now answers have been removed.

1. `zynkbot_rust/CLAUDE.md` says "16 hand-written Kotlin files" and names 14. There are 27 (section 2). The file needs its list brought up to date.
2. Alignment: `llvm-readelf` on the four AAR libraries (`libvosk.so`, `libonnxruntime.so`, `libonnxruntime4j_jni.so`, `libjnidispatch.so`) and `zipalign -c -P 16 -v 4` on a current release APK. The 13 libraries built from this repository were checked (section 10).
3. `MODIFY_AUDIO_SETTINGS` has no direct caller in the Kotlin sources, and `SpeechService` is gone. Find out whether `AcousticEchoCanceler`, vosk-android or the WebView needs it. If nothing does, it can be removed (section 3).
4. `TextToSpeech.speak` is called with null params and no audio attributes are set on the engine. Which audio stream a spoken reply plays on is a device matter the code does not decide (section 6).
5. Who sets `HOME` in the app process. `db::get_app_data_dir()` reads it; nothing in this repository's Rust or Kotlin sets it, and no setter was found in the `tao`, `wry` or `tauri` sources. If it is unset the fallback gives the same path (section 1).
6. `VoskBridge.downloadModel` fetches `vosk-model-small-en-us-0.15.zip` from alphacephei.com. No page code calls it at this commit. Whether that zip is the same model as the bundled one is not verified (section 5).
7. Whether a debug APK from `tauri android build -- --debug --apk` comes out signed with the debug key or needs the hand `apksigner` step in CLAUDE.md. No build was run for this check (section 10).

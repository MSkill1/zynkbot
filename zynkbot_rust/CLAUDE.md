# Zynkbot Dev Notes

## Android — DO NOT re-run `tauri android init`

`src-tauri/gen/android/` is committed to git and contains hand-edited files (full table in
`docs/architecture_and_development/ANDROID_ARCHITECTURE.md`, section 2):

- `app/src/main/java/ai/containai/zynkbot/*.kt` — 13 hand-written Kotlin files: `MainActivity.kt`
  (activity, permission queue, model unpacking, JavaScript bridges), `WakeWordService.kt`,
  `ZynkAssistantService.kt`, `ZynkAssistantSessionService.kt`, `ZynkAssistantSession.kt`,
  `ZynkRecognitionService.kt`, `NativeVoiceAnswerer.kt`, `OpenAiDictation.kt`, `VoiceCommands.kt`,
  `WakeVerifier.kt`, `ZynkCore.kt`, `EnvFile.kt`, `SyncForegroundService.kt`
  (the `generated/` subfolder is Tauri's and gitignored)
- `app/src/main/AndroidManifest.xml` — permissions (legacy READ/WRITE_EXTERNAL_STORAGE with
  maxSdkVersion only; MANAGE_EXTERNAL_STORAGE, USE_FULL_SCREEN_INTENT and READ_MEDIA_IMAGES were
  removed 2026-09-07 for Google Play — do not re-add them, files come in through the system picker),
  services, the assistant-role declarations
- `app/build.gradle.kts` — dependencies (vosk-android, onnxruntime), SDK levels (minSdk 26,
  target/compile 36), release signing from `keystore.properties`
- `app/src/main/res/xml/voice_interaction_service.xml`, `res/raw/wake_chime*.wav`,
  `app/src/main/assets/vosk-model/` and `assets/wake-word-models/`

Running `tauri android init` again will overwrite these files with Tauri's defaults,
breaking the foreground service, the ZynkbotShare folder, and the permission setup.
If you need to re-init for any reason, diff first and reapply the changes manually.

## Building Android APKs

Must use Android Studio's bundled JDK (system Java 8 JRE won't compile Gradle):

```
JAVA_HOME=~/android-studio/jbr npm run tauri android build -- --debug --apk --target aarch64   # test build
JAVA_HOME=~/android-studio/jbr npm run tauri android build -- --aab --target aarch64            # Play upload
```

Sign a debug build with the debug keystore (release builds are signed by Gradle from `keystore.properties`):

```
~/Android/Sdk/build-tools/<version>/apksigner sign \
  --ks ~/.android/debug.keystore --ks-pass pass:android \
  --out /tmp/app-signed.apk <unsigned.apk>
```

## ZynkbotShare folder (Android)

Files shared via ZynkLink on Android live in `Downloads/ZynkbotShare/`
(`/storage/emulated/0/Download/ZynkbotShare/`). The app creates this at launch via
`Environment.getExternalStoragePublicDirectory(DIRECTORY_DOWNLOADS)` — no storage
permissions needed for files the app creates there. Files placed there by other apps
(e.g. system Files app) may not be readable via raw File API on Android 11+ due to
scoped storage. Resolved 2026-09-07: files are added through the in-app picker
(`pickFile` copies into ZynkbotShare, `copyToKnowledgeBase` copies into the KB folder),
so no storage permission is needed and none is requested.

## CI

The `release-android` job in `.github/workflows/release.yml` cannot work as written (KI-034): it runs
`gradlew` without `tauri android build`, so the gitignored Tauri-generated Gradle files are missing.
Release artefacts are built locally for now.

## Phase 2 TODO (filed, not built)

Proper arbitrary-folder sharing on Android via SAF (ACTION_OPEN_DOCUMENT_TREE +
takePersistableUriPermission + Kotlin bridge to convert content:// URIs to paths
readable by Rust std::fs). Tracked as task #6 / android-phase2.

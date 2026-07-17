# Zynkbot Dev Notes

## Android — DO NOT re-run `tauri android init`

`src-tauri/gen/android/` is committed to git and contains hand-edited files:

- `app/src/main/java/com/zynkbot/app/MainActivity.kt` — foreground service startup,
  ZynkbotPathsBridge (getShareDir / openShareFolder), FolderPickerBridge
- `app/src/main/java/com/zynkbot/app/SyncForegroundService.kt` — foreground service
  with API-version-conditional startForeground()
- `app/src/main/AndroidManifest.xml` — trimmed permissions (no MANAGE_EXTERNAL_STORAGE,
  no requestLegacyExternalStorage)

Running `tauri android init` again will overwrite these files with Tauri's defaults,
breaking the foreground service, the ZynkbotShare folder, and the permission setup.
If you need to re-init for any reason, diff first and reapply the changes manually.

## Building Android APKs

Must use Android Studio's bundled JDK (system Java 8 JRE won't compile Gradle):

```
JAVA_HOME=~/android-studio/jbr npm run tauri android build -- --apk
```

Sign with debug keystore:

```
~/Android/Sdk/build-tools/<version>/apksigner sign \
  --ks ~/.android/debug.keystore --ks-pass pass:android \
  --out /tmp/app-signed.apk <unsigned.apk>
```

## ZynkbotShare folder (Android)

Files shared via ZynkLink on Android live in `Downloads/Zynkbot/`
(`/storage/emulated/0/Download/Zynkbot/`). The app creates this at launch via
`Environment.getExternalStoragePublicDirectory(DIRECTORY_DOWNLOADS)` — no storage
permissions needed for files the app creates there. Files placed there by other apps
(e.g. system Files app) may not be readable via raw File API on Android 11+ due to
scoped storage; if this proves a problem, the fix is a one-time SAF grant on the
folder at first launch (Phase 2 work).

## Phase 2 TODO (filed, not built)

Proper arbitrary-folder sharing on Android via SAF (ACTION_OPEN_DOCUMENT_TREE +
takePersistableUriPermission + Kotlin bridge to convert content:// URIs to paths
readable by Rust std::fs). Tracked as task #6 / android-phase2.

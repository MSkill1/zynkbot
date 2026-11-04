# Android baseline checks

The Android side of Zynkbot is hand-written Kotlin with almost no automated tests, and it
is the part that will move when the code becomes a Tauri plugin. These checks record what
works on a phone today so the same run can say what still works afterwards. Started
2026-11-04, before the removal of the in-service wake-word fallback (KI-051) and the plugin
move. Reports from each run live in `~/Documents/Zynk_related/android_baseline/`.

## Tier one: a script, over ADB, no hands

```
labs/android_baseline/android_checks.sh <adb serial> <device name on the desktop> [report.md] [label]
```

Run from the desktop with the desktop app running (the sync round-trip check reads the
desktop's own peer table). It checks: the app and build installed; the assistant role
held; the wake-word and sync services in the foreground with the right service types;
the assistant service and session service bound by the system; the sync server listening;
the phone reached the desktop within the last two and a half minutes; the share provider
answers a query; the assistant session opens on the assist key and the wake service hands
over and returns to listening; Back sends the app to the background with everything still
running; the screen going off leaves the wake service and the server up; no crash during
the run; the Rust core's lines reach logcat. The timer ring service cannot be started by
the shell on a release build (it is not exported), so it is marked SKIP there.

Each check is PASS, FAIL or SKIP with the evidence; the exit code is 1 on any FAIL.

## Tier two: by hand, ten minutes a phone

With the screen locked:

1. "Hey Zynk, what time is it?" — chime, spoken answer, close tone; the Z overlay comes
   and goes. (The assistant session, the native answer, the on-device or cloud model.)
2. "Hey Zynk, set a timer for one minute." — spoken confirmation; it rings at one minute
   and speaks the line; "Hey Zynk" or Stop silences it. (ZynkTimer, TimerRingService.)
3. "Hey Zynk" and nothing — the close tone after the listening window, no reply.

With the app open:

4. Tap the mic button, say a sentence — the words appear in the box (dictation, Vosk or
   Whisper per the selector).
5. Send a message that holds a fact — the fact appears in Memory Manager, and on the
   desktop within a minute (extraction, sync).
6. Memory Manager → the memory map opens and names are tappable.
7. ZynkLink → a file from the desktop lists and downloads; the Files app shows the
   Zynkbot share.
8. Realtime voice (OpenAI) connects and answers, if a key is set.
9. Share a photo to Zynkbot from the gallery — it arrives in the thread.

After the removal of KI-051's fallback, also: with Zynkbot *not* the digital assistant,
"Hey Zynk" gives the close tone and a "Hands-free needs setup" notification that opens
Default apps when tapped.

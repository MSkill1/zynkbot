# Zynkbot 0.9.6-beta2 — release notes (draft for Matt's wording pass, 2026-09-14)

*Text for the GitHub Release page. Checksums are filled in at publish time.*

A fix release for 0.9.6-beta1, from the first day of tester feedback. Install it over beta1 on every device; nothing is wiped and no setup is repeated.

## What's fixed

- **"New" starts a new conversation.** In beta1 the New button only cleared the screen; the thread underneath stayed the same, so everything said afterwards — typed or by "Hey Zynk" — was added to the conversation you thought you had left. Conversation History then showed one thread holding several conversations, and there was no previous one to go back to. Now New starts a separate thread and the old one stays in History exactly as it was.
- **A conversation appears in History as soon as you send the first message**, under "Current thread", instead of only after the first reply had finished.
- **Rename a conversation** with the pencil next to the pin. A blank name puts the automatic title back.
- **Message counts were 2 short** on every thread (a one-exchange thread said "0 messages"). Correct now; existing counts are repaired the first time this version starts.
- **Android: a repeating crash** ("Unable to start service SyncForegroundService") when Android restarted the app's sync service after reclaiming memory. The app no longer crashes on its own in the background.
- **Android:** a "Hey Zynk" exchange that finished while the app was in the background now shows up in the chat when you come back to it.
- The delete × in History is larger and easier to tap on a phone.

Not changed in this release: syncing of conversation history between devices is still unreliable (threads can go missing or appear twice on another device); that is being rebuilt after the beta.

## Files in this release

| File | For | Notes |
|---|---|---|
| `Zynkbot_0.9.6-beta2_x64-setup.exe` | Windows 10/11, 64-bit | Unsigned: Windows shows "Windows protected your PC" — click **More info → Run anyway**. Asks for administrator permission. Installs over beta1; your data is kept. |
| `Zynkbot_0.9.6-beta2_amd64.deb` / `.rpm` / `.AppImage` | Linux | `sudo apt install ./Zynkbot_0.9.6-beta2_amd64.deb` (or the rpm, or run the AppImage). |
| `Zynkbot_0.9.6-beta2.apk` | Android 8+ (arm64) | Sideload: open the file on the phone and allow the install. Installs over beta1 from the same key without losing data. |

SHA-256 checksums: *(filled in at publish)*.

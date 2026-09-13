# Zynkbot 0.9.6-beta1 — release notes (draft for Matt's wording pass, 2026-09-13)

*Text for the GitHub Release page. Checksums are filled in at publish time.*

Zynkbot is a private assistant with a memory: it runs on your own desktop and phone, keeps what it learns about you on your devices, and syncs between them directly over your home network. No account, no server in the middle. This beta is the first cut of the version headed for Google Play.

## What's new since 0.9.4

- **The phone version is now the phone's assistant.** Zynkbot asks to become Android's digital assistant at setup. "Hey Zynk" then works with the screen off or locked: a chime, a Z on screen (tap to cancel), the answer spoken aloud, and timers, alarms and the stopwatch set straight in the clock app with no AI model involved. Nothing in that path goes to Google. Decline the role and Zynkbot still works as an app; you only lose hands-free.
- **Offline dictation everywhere.** The mic button uses Vosk on the device on Linux, Windows and Android; no key, no cloud. OpenAI Whisper remains an option.
- **"About me".** A report the app writes about you from your own memories, entirely on the device: what it knows, since when, the people and places that recur, what changed, what you asked it to remember. Every line opens the memory behind it.
- **Use your desktop's model from your phone.** Pair the phone with the desktop and it can answer through the model running in Ollama on the desktop — no API key anywhere. Whatever model you pick on the desktop is what the phone gets.
- **Report a problem.** A button under every reply and in Settings builds a text report (version, device, masked log) you copy into an email or a GitHub issue. Nothing is sent automatically.
- **Memories know when things happened**, carry tags and categories, and "Remember: …" stores something word for word.
- Dozens of fixes from a four-device test run: native confirmation prompts on every platform (Windows used to skip them), the Android black-screen-on-return bug, the wake word hearing the phone's own reply, silent first lines from text-to-speech, a crash on memories containing an emoji or an em dash, the Windows installer location, and more. Full list in `CHANGELOG.md`.

## Files in this release

| File | For | Notes |
|---|---|---|
| `Zynkbot_0.9.6-beta1_x64-setup.exe` | Windows 10/11, 64-bit | Unsigned: Windows shows "Windows protected your PC" — click **More info → Run anyway**. Asks for administrator permission; installs to Program Files. If you installed an earlier beta, remove it first from Settings → Apps. |
| `Zynkbot_0.9.6-beta1_amd64.deb` / `.rpm` / `.AppImage` | Linux | `sudo apt install ./Zynkbot_0.9.6-beta1_amd64.deb` (or the rpm, or run the AppImage). |
| `app-universal-release.apk` | Android 8+ (arm64) | Sideload: open the file on the phone and allow the install. Installs over an earlier beta from the same key without losing data. The Play Store listing follows once the store paperwork is done. |

SHA-256 checksums: *(filled in at publish)*.

## First run

1. **Desktop first.** Install, open, follow the setup. Add an API key under Settings → API Keys, or point Custom / Ollama at `http://localhost:11434/v1` if you run Ollama.
2. **Phone.** Install, open, say yes to becoming the assistant (Settings → Default apps → Digital assistant app → Zynkbot), then Settings → ZynkSync → enter the code shown on the desktop's ZynkSync panel.
3. On the desktop press **Push to all devices** in API Keys once — a phone that pairs after the keys were entered does not receive them by itself (known, KI-055).
4. Everything else — memories, knowledge base, the wake word — works the same on both.

**Anthropic keys:** create the key *inside a workspace* in the Anthropic Console. A key made at organisation level is refused by the API without an extra header this beta does not send (KI-056).

## Known behaviour, not bugs

- The wake word can fire on a loud TV or steady noise (a fan, an air conditioner). After three empty firings in five minutes it becomes cautious for ten minutes and answers only questions and requests; it will tell you so if it drops a statement. In a noisy room, turn it off. Tapping the Z to cancel a false firing is the right response.
- Give it a few seconds after a "Hey Zynk" before the next one, and after closing or locking the phone.
- One hands-free listen is capped at 30 seconds.
- Zynkbot answers more slowly than the same model asked directly: every turn carries a safety check, your relevant memories and documents, and a second call afterwards to decide what to remember.
- Sync of *conversation history* between devices is unreliable in this beta (threads missing or doubled); memories sync fine. Being rebuilt after the beta. Please don't report it.
- Reinstalling the app on a phone leaves a stale copy of that phone in the other devices' ZynkSync lists; delete it.
- Battery: the wake-word detector costs battery; charging is unaffected. Keep the phone plugged in during long tests.

## Privacy, briefly

Dictation is on-device unless you choose Whisper. Wake-word audio is checked on the phone and discarded; a two-second clip is kept on the phone when the word fires, only so a personal voice profile can be trained, and it leaves the phone only if you send it. API keys go to the provider you chose and nowhere else. Sync is device-to-device over your network with pinned certificates. The problem report never sends itself.

## Reporting

Use the ⚑ Report button (under any reply, or in Settings), copy the text, and paste it into a GitHub issue or an email. Tick the conversation box only if you're comfortable sharing that thread; keys and codes are masked either way. Say what you did, what you expected, what happened.

# Zynkbot 0.9.6-beta3 — release notes (draft for Matt's wording pass, 2026-09-22)

*Text for the GitHub Release page. Checksums are the ones GitHub reports for the attached files, verified against downloaded copies.*

The first release from the road-to-1.0 branch. Install it over beta2 on every device; nothing is wiped and no setup is repeated. Tested as an upgrade over beta2 and as a fresh install on Windows, Linux, and two Android phones.

## What's new

- **Chat between your own devices.** Any device paired for sync shows a Chat button in ZynkSync settings; a note typed on the PC arrives on the phone with no extra pairing.
- **Read aloud.** Every reply has a speaker button next to Copy and Regenerate; tap again to stop. Uses the same voice as spoken replies, so it needs an OpenAI key.
- **The app starts in Sovereign mode, not Guardian.** Guardian's blocking is a demonstration of a safety layer and gave new users a hard "no" on ordinary questions. Sovereign warns on the same triggers instead of refusing. Child and HIPAA modes are unchanged and still opt-in.
- **Android: Zynkbot is a location in the phone's Files app**, beside Downloads and Drive, and appears in every app's Share menu. Anything put there is shared with linked devices, and downloads from linked devices land there. Files Zynkbot owned in the old `Download/ZynkbotShare` folder are moved over once.
- **Android: after downloading a photo** from a linked device, Zynkbot offers to add a copy to the gallery.
- **"About me"** in the Memory Manager says how many memories have no date and so are not on the timeline.

## What's fixed

- **Hands-free questions that need the web are searched at once**, no "want me to search?" step (there was no way to say yes by voice). It says "Let me check that" as the search starts, fetches its sources all at once instead of one after another, and shows the same "View n search sources" list under the answer that a typed search gets.
- **A file added to a shared folder shows up on other devices at their next Refresh.** Before, the other device saw the folder as it was when the sharing computer last opened its own ZynkLink menu.
- **Files shared from a Windows computer keep their folders.** A file in a subfolder was listed on Linux and Android as one oddly named file, `sub\file.txt`, instead of a file inside a folder.
- **Windows: deleting a Knowledge Base file waits for your answer.** The confirmation used the browser's own prompt, which on Windows answered "yes" by itself, so the file was deleted first and Cancel could not stop it.
- **The ZynkLink Refresh button shows its own status** ("Refreshing…", then "✓ Refreshed") instead of a message at the bottom of the panel that vanished after two seconds.
- **"Shared With Me" appears within about 5 seconds** even when a linked device is switched off. It used to wait up to a minute for each device in turn.
- **A file sent to the Knowledge Base from a linked device shows "Indexing n / N"** after the transfer and is marked done only once it can be asked about. A large PDF indexes for minutes on a phone; before, the bar said finished while chat silently waited.
- **A desktop's conversation history now reaches the phones.** Every start used to re-send the whole history in one request, which the phone rejected as too large; the backlog was silently dropped. History now moves 300 messages at a time.
- **Memory: the "Remembered on request" filter** shows the memories stored with "Remember:"; it had matched nothing since the feature shipped. **A memory can no longer be dated in the future.**
- **Resolving a memory contradiction closes the dialog the moment you confirm**; the resolution runs behind it. It used to hold the dialog through a 10-second timeout for every unreachable device.
- **When the model answers and then searches on its own, its first answer stays** as its own message, and the searched answer appears beneath it. It used to be replaced.
- **API Keys: once a key is stored, the provider button reads "View plan"** and opens the plan page; it said "Get Key" whether or not you had one.
- **Android voice:** a dictation that broke mid-recording no longer keeps the microphone (the wake word stayed deaf until a restart); "set a time for five minutes" sets a timer; the wake word's strict mode is off (after three empty wakes it demanded a confidence a real "Hey Zynk" rarely reaches, and the phone went deaf for ten minutes); a personal wake-word profile can be dropped into the app's data folder without a new release.
- **Sync:** a device that had never generated a pairing code failed every sync it started; the first sync between two devices with no memories never sent conversation history; pairing accepts `host:port`.

## Known issues

- **Conversation history between two devices can duplicate messages or skip a thread.** Being rebuilt; not fixed by a quick patch.
- **Reinstalling the app, or wiping a phone's data, leaves a stale entry for the old identity in other devices' ZynkSync lists**, and the stale entry can be handed on to newly paired devices. Delete the old entry; it clears everywhere. In one case the stale entry made a phone add itself as a peer and log a database error once a minute; deleting the entry stops it. A freshly installed desktop may also not receive the other devices by introduction: pair it to each one directly.
- **A device that pairs after API keys were entered doesn't receive them automatically.** Workaround: "Push to all devices" in Settings → API Keys.
- **An Anthropic API key created without a workspace fails to authenticate.** Create the key inside a workspace in the Anthropic Console.
- **A false "Hey Zynk" trigger can occasionally be logged as a memory.** Delete it from the Memory Manager.

## Files in this release

| File | For | Notes |
|---|---|---|
| `Zynkbot_0.9.6-beta3_x64-setup.exe` | Windows 10/11, 64-bit | Unsigned: Windows shows "Windows protected your PC" — click **More info → Run anyway**. Asks for administrator permission. Installs over beta2; your data is kept. |
| `Zynkbot_0.9.6-beta3_amd64.deb` / `.rpm` / `.AppImage` | Linux | `sudo apt install ./Zynkbot_0.9.6-beta3_amd64.deb` (or the rpm, or run the AppImage). |
| `Zynkbot_0.9.6-beta3.apk` | Android 8+ (arm64) | Sideload: open the file on the phone and allow the install. Installs over beta2 from the same key without losing data. |

SHA-256 checksums (`SHA256SUMS.txt` is attached to the release):

```
6e6d138874cbaabf08602c763db701ff616011040f8e6bec504f8c31906d1b25  Zynkbot-0.9.6-beta3-1.x86_64.rpm
f3bb10b44023634d96e6bcb1ef881077f8760a2b640fa44d88c1ee5d7a9c0a34  Zynkbot_0.9.6-beta3.apk
fd9b09999efe2393818ea4f9e2f889a26c3f1d79daa94dfac4123fd9851d269d  Zynkbot_0.9.6-beta3_amd64.AppImage
ae02e7398d1d2852f958eea7dac5693075e02109cebc017806f40e8c52d61ccb  Zynkbot_0.9.6-beta3_amd64.deb
56dda9c8474c38206aa6aea45bbc5f2f885f5ae8d5080b4f2883990864f1e23d  Zynkbot_0.9.6-beta3_x64-setup.exe
```

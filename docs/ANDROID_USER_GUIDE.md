<!-- Written by Claude; every statement checked against the code on branch local-model, last full pass 2026-10-27 (beta 5). Matt reviews all documentation before 1.0. -->

# Zynkbot on Android: user guide (beta)

This guide is for anyone installing Zynkbot on an Android phone, including beta testers. It describes what the app does today, as built. Where a term needs explaining, the explanation is on the same line.

## 1. Installing

Beta testers are invited through Google Play. If you received an invitation link, open it on the phone, tap "Join", then install Zynkbot from the Play Store like any other app. Updates arrive through Play.

Without an invitation, you install it from a file:

1. On the phone, open the GitHub Releases page: https://github.com/MSkill1/zynkbot/releases/latest
2. Download the `.apk` file. An APK is an Android install package. The build is for arm64 phones, which is every mainstream Android phone from the last several years.
3. Tap the downloaded file. Android asks you to allow installs from this source (your browser or file manager); the exact wording depends on the Android version. Allow it. This is asked once per app that installs things, not every time.
4. Tap Install, then Open.

Minimum Android version: 8.0. Newer is better; some features below only exist on newer versions and say so.

## 2. First launch: the permission prompts

Zynkbot asks for permissions one at a time, in this order. Each prompt waits for you to answer before the next one appears.

1. Microphone. Needed for the mic button in the chat and for hands-free "Hey Zynk". Without it the app still works as a text chat.
2. Notifications (Android 13 and newer only). Zynkbot shows two quiet, permanent notifications while it works in the background: "Listening for Hey Zynk" and "Memory sync active". Android requires a notification for any app that keeps a microphone or network task alive in the background. If you decline, those background tasks may be stopped by the system.
3. Digital assistant role (Android 10 and newer). Android asks whether Zynkbot should be your phone's assistant app, the role Google Assistant normally holds. Some phones refuse to show that question (the Pixel test phone, running GrapheneOS on Android 17, does); then Zynkbot opens the phone's "Default digital assistant app" screen for you and you pick Zynkbot there. Saying yes lets Zynkbot listen and answer on top of any app and on the lock screen, and lets the wake-word listener come back after a reboot without you opening the app. This prompt appears once per install. If you decline, hands-free still works but in a reduced form (see section 3).
4. Local network (Android 16 and newer only). Lets Zynkbot talk to your other Zynkbot devices on the same Wi-Fi for syncing, file sharing and ZChat (section 8). If Zynkbot is only on this one phone, declining is fine.
5. Battery (from the second start, once). Android asks whether Zynkbot may ignore battery optimisation. Saying yes keeps the background sync and the "Hey Zynk" listener from being frozen by the phone after a few hours off screen, which some phone makers do (seen on a OnePlus). Zynkbot asks only once.

To grant the assistant role later: open the phone's Settings, then Apps, then Default apps, then Digital assistant app, and choose Zynkbot. On a Pixel the screen is titled "Default digital assistant app" and lists every installed assistant, including one installed from a file. Menu names vary by phone maker.

Zynkbot does not ask for "all files access" or for access to your photos. Files reach the app through the system file picker, which needs no permission.

The Back gesture puts Zynkbot in the background instead of closing it, so "Hey Zynk" and sync keep running. System Controls (the ⚙️ button) has a "Help and documentation" row that opens these guides on the web.


## 3. Talking to it hands-free

Two things happen when you talk to Zynkbot: the phone listens for the words "Hey Zynk", and then turns what you say next into text. Both can run entirely on the phone.

- The wake-word listener always runs on the phone. Its models ship inside the app; nothing is downloaded and no audio leaves the device.
- Turning your speech into text (dictation) uses an offline engine called Vosk by default. It runs on the phone with no internet connection, and the audio never leaves the device.
- If you prefer, you can switch dictation to OpenAI Whisper, an online service that returns better punctuation. Then, and only then, each recording is sent to OpenAI. OpenAI's published API data policy (read 2026-09-09) says API data is not used to train its models and lists the transcription service as keeping no audio for abuse monitoring; its general rule for other services is up to 30 days. Section 6 explains how to choose.

Hands-free is off until you turn it on. In the app, open System Controls (the ⚙️ button), then the Voice page, and switch on the wake word under "Hey Zynk".

What happens, step by step:

1. Say "Hey Zynk". The phone plays a short chime. If Zynkbot is your assistant app, a green Z appears near the bottom of the screen, over whatever is there, including the lock screen. The screen lights up if it was off.
2. Speak your question after the chime.
3. When you stop talking, Zynkbot waits about a second and a half of silence, then plays a second, shorter tone. That tone means your words were sent.
4. The Z changes from a slow pulse (listening) to a running light (thinking).
5. The reply is spoken with the phone's built-in text-to-speech voice, sentence by sentence as it arrives. Zynkbot picks the best-quality English voice installed on the phone. The question and the reply are also added to the current conversation thread in the app. If the answer needed a web search, the reply in the thread shows a "View n search sources" line you can open.

Limits on one listen:

- If nothing is heard for 6 seconds after the chime, Zynkbot closes with the short tone and does nothing else.
- One listen is capped at 30 seconds. This is a fixed limit set in the app, not a limit of either dictation engine; anything said after 30 seconds is cut off.
- A tap on the Z cancels, whether it is listening or speaking. You hear the short tone, and the phone goes back to waiting for "Hey Zynk".
- The Stop button in the app also cuts off a spoken reply.
- Zynkbot answers one question per wake. It does not keep listening after a reply. Say "Hey Zynk" again for the next one.

Without the assistant role, "Hey Zynk" still works, but there is no Z on screen, nothing to tap to cancel, and dictation always uses the offline engine (section 6).

Other apps' recordings: if another app is using the microphone (a call, a voice memo, another assistant), Zynkbot pauses its listening and resumes when that app is done. If the end of the other recording is missed, Zynkbot checks again after two minutes and takes the microphone back when it is free.


After a restart. If Hey Zynk was on, it comes back on its own about a minute after the phone boots, without opening the app (the Voice page switch "Start listening when the phone starts", on by default). If the phone refuses, a notification "Hey Zynk is off since the restart" offers a tap that starts it. Measured working on a Pixel 10 Pro XL and a OnePlus 12R, 2026-10-25.

## 4. What it does without asking a model

Some requests are handled on the phone and never sent to a model. Numbers can be spoken as words ("ten minutes") or digits.

Timers, alarms and the stopwatch are Zynkbot's own (since 2026-10-25). Nothing is handed to the phone's clock app: Zynkbot sets an exact alarm with Android, so a timer rings with the screen off, with the app closed, and after the phone has been asleep for an hour. When it goes off it says what it is ("Your ten-minute timer is done", "It's 7 30 AM. Your alarm.") and rings for 30 seconds unless you stop it. Each timer and alarm shows as a small chip above the message box with a Cancel button, and as a notification in the shade with a Cancel button. You can have several at once.

One-time permission. Android 12 and later let an app ring at an exact time only with your say-so, and Android 14 and later leave it off by default. The Voice page shows a row "Let Zynkbot run its own timers and alarms" with an Allow button until the switch is on; Allow opens Android's "Alarms & reminders" page for Zynkbot. Until it is on, timers and alarms go to the phone's clock app as they did before, and the spoken confirmation says so.

Timers. "Set a timer for ten minutes", "timer for ninety seconds", "five minute timer", "two and a half minutes". "Set a time for ten minutes" works too, because dictation often drops the r. "Stop timer" or "cancel timer" cancels it; with several running Zynkbot names them and you say "cancel the ten minute timer". "Check timer" says how much is left on each.

Alarms. "Set an alarm for 7 30 AM", "alarm at 6", "wake me up at 7 15", "set an alarm for eight fifteen". With AM or PM, or a 24-hour time such as 18 30, the time is taken as given. Without either, a time from 1 to 12 means the next one to come: "eight fifteen" said in the morning before 8:15 is 8:15 AM, said after it is 8:15 PM. Dictation's "four" for "for" ("set an alarm four eight fifteen") is understood, and if the app asks "AM or PM?", "set it for 8:15 PM" answers it. "Cancel alarm", "cancel the 7 30 alarm", "check alarms". If dictation hears "timer" for "alarm" ("set a timer for 7 30 AM"), a clock time after "timer" is still taken as an alarm.

Stopwatch. "Start stopwatch", "stop stopwatch" (keeps the time; "start stopwatch" again continues), "reset stopwatch", "check stopwatch" or "check stopwatch time".

When it rings. Say "Hey Zynk" (any wake stops the ring; whether the phone hears you over its own alarm tone depends on its echo cancelling), tap the big orange Stop button that appears above the message box, or tap Stop on the notification. Left alone it stops after 30 seconds.

Punctuation from the cloud dictation ("Stop, stopwatch.", "7.30 a.m.") is handled.

With the permission off, and the clock app unable to take the command either, Zynkbot says "I couldn't reach the clock app to do that."

Using the Remember function ("Remember:"). Start a message with the word remember and a colon, then the fact, and Zynkbot saves exactly those words as a memory, with no rewording. This works spoken (hands-free or the mic button) and typed in the chat as "Remember: ...". Because the offline dictation engine often mishears the word "colon", these spellings are all accepted at the start of a message:

- remember colon
- remember colin
- remember collin
- remember cohen
- remember colan
- remember call and
- remember call in
- remember call on

Only a plain space or punctuation may follow the word. "Remember Cohen's birthday is in May" is treated as an ordinary sentence, and so is "remember colonial history". After saving, the model replies with one short confirmation of what was stored.

## 5. When it stays silent on purpose

Hands-free listening hears everything in the room, so Zynkbot has several checks that end a listen without a reply. In each case you hear the short close tone and nothing else.

- Fragments. A transcript of one word, or more than 60 words, is dropped. One word is noise; sixty words is a TV programme.
- TV, radio, other people. Every hands-free transcript is sent to the model with a note that it may not be addressed to an assistant. The model is told to reply with nothing at all unless the text is plainly a question or a request. Statements, narration, news and dialogue get no answer.
- Quiet audio and no speech. A detection is ignored when the audio is quieter than -65 dBFS, or when a small speech detector (Silero VAD, running on the phone) heard no speech in the last two seconds. Loud TV sound still counts as speech; that is what the verifier below is for.
- The phone's own sound. While the phone plays audio (music, a video, Zynkbot's own chime or spoken reply) and for a moment after, detections are ignored, and the echo canceller is on while listening so the phone's own sound is not taken for your voice. Zynkbot does not wake over music playing on the phone.

Strict mode, which earlier betas switched on after repeated false triggers, is gone since beta 3; the checks above and the verifier below replace it.

The personal verifier. This is a small second check that runs after the wake word fires and asks: does this sound like the owner saying "Hey Zynk"? You train it on the phone: on the Voice page, "Teach Zynkbot your voice", tap Start and say "Hey Zynk" twenty times, waiting for the count between. The phone trains the verifier itself from those clips against the false triggers it has collected, and the teaching is cumulative: "Teach again" adds to what it has rather than starting over. It starts enforcing on its own once its numbers are good enough (on held-out clips it keeps at least nine in ten of your own wakes and stops at least nine in ten of the false ones); once it enforces, a later retrain never switches it off. It enforces only for the person it was trained on, recognised by the Zynkbot user id that all of that person's paired devices share; for anyone else it only logs its opinion.

The clips. The phone saves a short audio clip (about 3 seconds) when the wake word fires, under the app's private storage in a folder called `wake_triggers`, labelled by whether the wake led to an answer or command. These clips are the verifier's training material. Once the verifier is enforcing, only the newest 20 trigger clips are kept. Nothing is uploaded; the clips stay on the phone and are deleted with the app.

Hands-free works without the verifier. The verifier only reduces misfires: it cannot make Zynkbot wake when the wake-word model itself did not fire.

A wake that the verifier rejects, or that is flagged by the checks above, ends with the short close tone and nothing spoken; Zynkbot does not announce the rejection.

## 6. Choosing the dictation engine

Dictation means turning your speech into text. Open System Controls, then the Voice page, and look under "Dictation Engine". The choice applies to both the mic button in the chat and hands-free "Hey Zynk". With the offline engine, the words appear in the message box as you speak; the box has Copy and Clear buttons while there is text in it.

Offline, Vosk (the default). Runs on the phone. No internet needed and no audio leaves the device. It produces lowercase text with no punctuation, and numbers come out as words ("ten minutes"). Models understand requests fine without punctuation.

OpenAI Whisper. Sends the recording to OpenAI and gets text back with punctuation and capitals. It needs an OpenAI key entered under API keys in the app (the entry is named OPENAI_API_KEY). If the key is missing, hands-free quietly uses Vosk instead. Hands-free waits up to 8 seconds for you to start speaking, records until you pause for just over a second, and never records more than 30 seconds. If the upload fails, that one listen falls back to Vosk. What OpenAI does with the audio is described in section 3.

## 7. Files and folders

Where Zynkbot keeps its data. Memories, conversations, settings and the knowledge base live in the app's private storage. Other apps cannot see it, and uninstalling Zynkbot deletes it.

Knowledge Base. A knowledge base is a set of your own documents that Zynkbot can search when answering: text, Markdown, PDF, HTML, and Word documents (.docx and the older .doc). In the Knowledge Base manager, tap "Add files". The system file picker opens; you can select several files at once. Zynkbot copies each one into its private knowledge base folder and never changes the original. A file with the same name as an existing one is saved as "name (2)".

Indexing, and why it is needed. Adding a file only puts it in the folder. Before Zynkbot can talk about what is in it, it has to read the file and build its own searchable index of the contents, in its private database on the phone. That step is called indexing, and it is a button in the Knowledge Base manager. Until a file is indexed, Zynkbot cannot see what it says. Indexing runs on the phone and sends nothing anywhere; it can take a while for a long PDF. Then, when you ask a question with the KB button on, Zynkbot looks up the most relevant passages from the index and reads them before answering.

The Zynkbot model. On a fresh install, after the small required models, setup offers the Zynkbot model: Qwen3-4B-Instruct-2507 in the Q4_0 file, 2.4 GB, downloaded from Hugging Face. It runs on the phone with no cloud and no key. Download it on Wi-Fi and keep the app open; if the download is interrupted it continues where it stopped the next time, and the file is checked against its checksum before it is used. "Later" leaves a Download button on the Current Model page. The model needs about 3 GB of free memory while it runs; a phone from 2023 or later with 8 GB of RAM answers a first question in a thread in 10 to 20 seconds and later ones in under 10.

Web search. When a question needs something current, Zynkbot offers a search in a card with a query you can edit, and searches only when you tap it; hands-free it searches without asking. With a Claude, OpenAI, Grok or Mistral key the search runs through that provider's own search, on your key, and the answer lists the pages it used. With a local model it uses a Brave Search key or a search server address from Keys and servers, or DuckDuckGo without either. DuckDuckGo blocks a burst of searches from one network for a few minutes, and Zynkbot tells you when that happens rather than saying there were no results. To have Zynkbot read a particular page, put its address in your message; the page is fetched on the phone and answered from, no search involved.

The Zynkbot location. Zynkbot is a storage location on the phone: it appears as "Zynkbot" in the Files app's drawer and in any app's file picker, and Zynkbot is in every app's Share menu. A file saved or shared into it belongs to Zynkbot and shows up in the ZynkLink file list. There is no permission to grant and no folder in Downloads any more (earlier betas used Downloads/ZynkbotShare; files Zynkbot owned there were moved over once).

Pictures put into the Zynkbot location do not appear in the gallery unless you accept the "add to gallery" offer after a download, which copies them into Pictures/Zynkbot.

## 8. Your other Zynkbot devices: sync, file sharing and ZChat

If you run Zynkbot on more than one device (another phone, a laptop), the devices can find each other on the same Wi-Fi and work together. This is included in the beta. Nothing in it uses the internet or any server; the devices talk to each other directly over an encrypted connection, and only devices you have paired are accepted.

Pairing. On one device, open the sidebar (the ⚙️ System Controls button) and find ZynkSync; it shows that device's pairing code in the form address:code, for example 192.168.0.100:123456. On the other device, in the same panel, enter the address and the six-digit code and confirm. Pairing is done once per pair of devices.

ZynkSync. Once paired, memories, conversations and settings such as keys are copied between the devices in the background, so a fact you told the laptop is known to the phone. The sync was rebuilt in beta 4 around an outbox: each change is queued on the device that made it and delivered once to every paired device, in order, with nothing sent twice. A thread deleted on one device closes on the others. The thread you wrote to last is the current one on every device: a device you pick up lands in it, and one left open follows it once it has been quiet for 45 seconds, never while you are typing. If a pairing code is wrong, the app says so. Please report anything that does not arrive.

ZynkLink. Files can be sent from one device to the other. On the phone, files you receive land in the Zynkbot location (section 7). A file received into the knowledge base is indexed on arrival; the Knowledge Base page shows "Indexing n / N" while that runs. New files on a linked device appear at the next Refresh, which shows what it found.

ZChat. A private text chat between two of your devices, or between two people who have linked their devices. For your own synced devices, open the ZynkSync page and tap the Chat button next to the device; for a device linked by code, the ZynkLink menu has the Chat entry. Messages are stored on the two devices and nowhere else. A message is sent the moment you tap send; if the other device is off or away, it is kept and delivered the next time the two devices sync. The chat window can be left open while the other device types.

## 9. Memory

How memories are made (since beta 5). Your message is not stored as a memory. After each reply, Zynkbot takes the personal facts out of what you said, up to three per message, and stores each as its own memory in the third person with a title and, when you said one, the day it happened. What you actually said is kept with the facts: the Memory Manager groups the facts from one message under "What I said", with the day above; tap the quote to read the whole message, and it is editable from any of its facts. "Remember:" still stores your words exactly. Translate, rewrite, reprint and proofread requests store nothing. The work happens in the background, in order, and finishes after a restart if it was interrupted; with the on-device model a new memory can take a few minutes to appear.

Memory Manager. The Memory Manager (a button in the app; on a phone it sits in the bottom row of buttons) lists every memory. You can edit, delete or add memories there. A "Remembered on request" checkbox above the list shows only the memories you stored with "Remember:".

Passing facts expire. A parking level, an errand, a trip, how this week is going: a fact like that leaves recall two weeks after its day. It stays in the Memory Manager tagged EXPIRED, with a Keep button that makes it permanent. A memory replaced by a newer one is tagged REPLACED, with "Undo replacement".

Memory cards. When a new fact contradicts a stored one, or looks like an update to it, Zynkbot asks on a card: which is right, or "Is this an update?". The card waits until you answer: it shows on the next open or return to the app, and if the app is off screen the phone posts a notification "Zynkbot has a question about a memory". Tap it to answer.

About me. In the Memory Manager, tap "About me". It builds a report from the memories on this phone, in these sections:

- You asked me to remember: everything stored with "Remember:", newest first, 20 at a time with a Show more button (only memories stored from this version on; older ones carry no mark)
- Timeline, by when it happened
- Categories
- Tags
- People, places and things
- Belief changes
- Kitchen (shown only if there are any)
- Heard hands-free (shown only if there are any)
- Tone
- Most recent memories

Each memory title in the report is a link. Tap it to open that memory to edit or delete it; closing it returns you to the report. Tapping a tag filters the memory list by that tag.

Heard hands-free. Memories that came from a "Hey Zynk" exchange are marked separately and listed in this section, so you can check what the phone picked up when you were not looking at it. Only the last 30 are shown.

## 10. Reporting a problem

Two places: the "⚑ Report" button in the sidebar, and a small "⚑ Report" link under any reply from Zynkbot. The second one starts the report with the first 300 characters of that reply.

What a report contains. Everything is built on the phone from the app's own state:

- Zynkbot version and build number
- Platform and, on Android, the phone maker, model and Android version (nothing that identifies you)
- The model backend in use (for example which provider)
- Your description of what happened
- **The current conversation, only if you tick the box.** Untick it and the report contains no conversation and no memories: the log lines that would quote your message, a memory title or the model's reply are blanked out too. You can still report the problem.
- The last 300 lines of Zynkbot's own log, read from the log file it keeps on the phone (`logs/zynkbot.log` in the app's private storage, which reaches back days), and, if you tick "Include the Hey Zynk log", the hands-free listener's own log too

What is masked. Even when you include the conversation, it is masked before it is shown to you. Before anything is shown, the log and the conversation text are scanned and the following are replaced with "[redacted]": provider API keys (such as keys beginning with sk-), any KEY=value, SECRET, TOKEN, PASSWORD or PASSPHRASE pairs, bearer tokens, long hexadecimal strings (including the backup key), long base64 strings, and ZynkSync pairing codes.

Nothing is sent automatically. The report is a block of text on your screen. Read it, tap "Copy report", then paste it into a new GitHub issue. The "Open GitHub issues" button opens https://github.com/MSkill1/zynkbot/issues/new in your browser.

## 11. Known limits in this beta

- ZynkSync was rebuilt in beta 4. If a memory or a thread does not arrive on a paired device, report it; it is not a known problem any more.
- The wake word can fire on a loud TV or radio nearby. Loudness cannot tell your voice from the TV at room distance; that is what the verifier and the back-off in section 5 are for. Expect some misfires with a TV on.
- The chime, the close tone and spoken replies all use the phone's media volume, the same slider as music and videos. If replies are too quiet, turn up media volume.
- Hands-free replies in the app are always spoken. The "Speak replies in the app" switch only affects replies to typed or mic-button messages, and that uses OpenAI's voice, so it needs the OpenAI key. A 🔊 Read aloud button under each reply reads that one reply the same way.

## 12. Privacy notes

- Stays on the phone: memories, conversation history, the knowledge base, settings, API keys, wake-word audio clips, and all speech recognition when the offline engine is selected.
- Deleted for good: a conversation or memory you delete is overwritten in the database file at that moment, not left in free space until the file is compacted.
- Leaves the phone: your messages and relevant memories go to whichever model provider you configured under API keys, and only when you ask a question. Hands-free questions are sent the same way.
- Leaves the phone if you choose Whisper: the audio of each dictation goes to OpenAI.
- Leaves the phone if you use "Push keys": your API keys and the memory backup key are sent to your own paired Zynkbot devices on the local network, and nowhere else.
- Never sent by the app on its own: problem reports, logs, or the wake-word clips.

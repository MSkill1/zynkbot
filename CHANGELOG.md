# Changelog

This file documents notable changes to Zynkbot from the initial public release forward.

For the full commit history, see [GitHub](https://github.com/MSkill1/zynkbot/commits/main).

---

## Unreleased (on `local-model` after beta 5)

- **Web search answers stream.** A cloud model's own search (Claude, OpenAI, Grok, Mistral) now sends its answer word by word as it is written instead of all at once after the searches; the sources still attach at the end. Measured first word: 3.5 to 11 s depending on the provider.
- **The waiting bubble says what it is waiting for** ("Looking through your memories…", "Reading the page…", "Searching the web…", "Waiting for Claude…") and the log records the time to the first word of every reply.
- **Pictures stay with the thread while it is open.** The newest three pictures go back to the model with the history, so it can compare two photos sent in different messages; after a picture message the model writes one line about what the picture showed, kept under the message in the thread.
- **Cloud models recall more.** Memory search returns 30 instead of 15 and the prompt holds 30 instead of 20 (local and on-device limits unchanged).
- **Settings file could be cut to one line when the app was killed while saving (KI-100), fixed:** the `.env` is now written atomically with a `.env.bak` beside it, a failed read is never treated as an empty file, and at startup the app puts back any key it holds in its own sync records that the environment lacks.
- **Dictation uses `gpt-4o-mini-transcribe`** (was `whisper-1`) on phone, Linux and Windows, and logs how long each transcription took.
- **Live dictation on the Linux desktop (OpenAI engine):** words appear in the composer as you speak, over OpenAI's Realtime transcription connection; the upload-after-stop path remains as the fallback (`DICTATION_STREAM=0`, no key, or a connection failure). Phone and Windows paths still upload after you stop.
- **The model format is read from the model file** (its declared architecture) when the file name says nothing, so a renamed or fine-tuned model gets the right chat markers.
- Conversation History: the section headers (Current thread, Pinned, Today, Yesterday, This Week) have their own colours.
- **Pinned conversations sync** between devices (migration 0025: `pinned_at`; the newest change to a pin wins everywhere).
- **One shared conversation across devices.** The thread any device wrote to last is the current one on all of them: a device lands in it when it is brought back to the front, when the app starts, or when a sync brings news and the device has been idle 45 s; it never moves while you are typing, waiting for a reply or have a file attached, and "New" is respected until another device writes after it.
- **"Last synced" in the sync panel**, per device: how long since a round finished cleanly, in amber when the device is online but nothing has synced for five minutes.

- **Live dictation on Android (OpenAI engine):** words appear while you talk, in stretches after each pause, instead of after the stop button.
- **Conversation modes:** a drop-down beside the thread title (Tool, Debate, Premise, Editing) puts one short instruction block in front of the prompt for that conversation; a help line under the title says what the mode does. Checked case by case against Claude on 2026-10-29 (`mode_probe`): Editing and Debate change the answers clearly; a Companion mode was tried and folded into Normal, whose voice now also says direct, no cheering on, no therapy language.
- **One shared conversation, follow-up:** a device touched just before another device wrote now catches up once it has been quiet for 45 s (before, it never did until the app was reopened).
- Phones: a small line under a picture in the chat says it can be held to save it to the gallery.

- **ZynkLink file browser:** files in your own share folder can be removed or moved (desktop: save dialog; Android: the system picker); a Sort row (Name, Size, Newest) above the list, remembered; the download cancel button is labelled "Cancel"; "Browse shared files on this device" looks like the other buttons.
- **Keys page:** each provider group folds; closed on open, the header still says whether the key is set.

## [0.9.6-beta5] — 2026-10-27 <!-- branch local-model; entries for 10-26/27 drafted by Claude, for Matt's review -->

### Timer, alarm and stopwatch (Android; branch `timer`, 2026-10-25)
- Zynkbot runs its own timers and alarms instead of handing them to the phone's clock app. They ring with the screen off or the app closed (the fix for a timer that was confirmed but never rang, #29), say what they are, and ring for 30 seconds unless stopped by "Hey Zynk", the Stop button in the app, or Stop on the notification. Several at once, each with a chip above the message box and a notification, both with Cancel. New commands: "check timer", "stop timer" / "cancel timer" / "cancel the ten minute timer", "cancel alarm" / "cancel the 7 30 alarm", "check alarms", "stop stopwatch", "reset stopwatch", "check stopwatch". Needs Android's "Alarms & reminders" switch for Zynkbot, offered on the Voice page.
- The desktop app has the same timer and stopwatch inside the app (rings while the app is open); alarms are phone-only.
- Hey Zynk comes back on its own after a restart when it was on (Voice page: "Start listening when the phone starts"); if the phone refuses, a notification offers a tap.
- The hidden MEMORY marker at the end of a cloud reply is never shown or spoken, even when the model glues it to the last sentence.
- Coming back to the app from another app no longer brings the keyboard up.
- System Controls has a "Help and documentation" row.

### Memory (pipeline rewrite, 2026-10-24, branch `memory-types`)
- Your message is no longer stored as a memory. Zynkbot now takes the personal facts out of it, up to three per message, each as its own memory in the third person with a title and, when you said one, the date it happened. What you actually said is kept with each fact and shown in the Memory Manager and to the model.
- How it decides: a cloud model ends each reply with a hidden yes or no on whether the message held anything to keep; on yes a separate background call lists the facts. The on-device model is not asked for the yes/no (a small model answers that and nothing else) and goes straight to the fact call, which answers "none" when there is nothing. "Remember:" still stores your words exactly as typed.
- Facts from one message are linked to each other, and when they are recalled together the model sees the message once with its facts underneath, instead of the same quote repeated.
- Quick follow-up questions no longer lose the previous message's memories: the background work is a queue that runs in order when the model is idle, waits when you are talking, and survives a restart.
- Guards against invention by small models: a fact that shares no word with your message, a restatement of a fact already taken from it, or a near-duplicate of one is dropped; at most three memories per message.
- On the phone with the device model: the prompt it reads is cut down (8 recent messages of 400 characters, 4 memories of 300, 85 words of instructions), so the first answer in a thread takes 10–20 seconds and later ones under 10; the Memory Manager says that memories from the device model take minutes to appear.
- Junk memories that the old pipeline had stored from raw messages and hands-free turns are gone from the developer's database; nothing in the app deletes yours. The Memory Manager is the place to review them.

### Memory, continued (2026-10-26 and 27, branch `memory-types`)
- A memory card (a contradiction, or "is this an update?") is kept until it is answered: it shows on the next open or return to the app, and the phone posts a notification "Zynkbot has a question about a memory" when the app is off screen.
- Passing facts expire: a parking level, an errand, a trip, this week's mood leaves recall two weeks after its day. It stays in the Memory Manager tagged EXPIRED with a Keep button. A memory line in the prompt carries the day the thing happened when that differs from the day it was said.
- The Memory Manager groups the facts from one message under "What I said", with the day above; tap the quote to read the whole message; the message is editable from any of its facts.
- Translate, rewrite, reprint and proofread requests store no memory.
- Names come out cleaner: the fact prefix "User", stray function words ("of", "is") and glued word fragments no longer appear in a memory's names.

### The Zynkbot model at first run
- A fresh install offers the Zynkbot model (Qwen3-4B-Instruct-2507, Q4_0, 2.4 GB) as a download at the end of setup, on the phone as well as the desktop, with a progress bar and "Later". The download resumes where it stopped if the app was closed, and the file's checksum is verified before it is used. "Later" leaves a Download button on the Current Model page.

### Zynkbot's own guides
- A question about Zynkbot itself (pairing, Ensemble, permissions, what a button does) is answered from the built-in user guides, which are indexed on the device. Dictation's "zinc sync" and "zinc link" count too. If the guides do not cover it, the answer says so instead of guessing.

### Web search
- Where a search goes now depends on the model you are using. With a Claude, OpenAI, Grok or Mistral key, the search runs through that provider's own web search tool on your key, and the reply is that model's answer with the pages it used under "View sources"; the provider bills it per search. With a local model, results come from a Brave Search key or a search server address set in Keys and servers, or from DuckDuckGo with neither. The search card, and the hands-free search, work as before.
- DuckDuckGo's "prove you are human" page is reported as a temporary block instead of "no results"; searches are spaced a few seconds apart, and a known block is waited out.
- A searched answer is saved into the thread like any other reply.

### Reading a page you point at
- A web address in your message makes Zynkbot fetch that page on your device and answer from its text, with the page under "View sources". No search involved; works with every model, local ones included. A page that cannot be read is reported as such.

### Documents
- Word documents, both .docx and the older .doc, can be added to the Knowledge Base and attached to a message; the words are read, not the formatting.

### Timer and alarm fixes
- "Set an alarm four eight fifteen" (dictation's version of "for") sets 8:15; a time with no am or pm means the next 8:15 to come; "set it for 8:15" after "am or pm?" works; "alarm 4 15 pm" is 4:15 PM.
- An alarm the app set is no longer disowned by the model a moment later ("I can't set alarms"): the model now knows the app's own clock did it.

### Privacy
- Deleted conversations and memories are overwritten in the database file at the moment of deletion, not left in free space until the file is compacted.
- A hidden "MEMORY: no" line no longer reaches the screen under a web-search line.

### Chat prompt
- Zynkbot may now ask first: if one or two missing details would change the answer, it asks for them briefly instead of guessing; otherwise it answers and says what it assumed. In testing before 1.0.

### Cloud models
- When Anthropic or OpenAI reports a problem inside the reply stream (overloaded, rate limit), Zynkbot now shows that message instead of an empty bubble. Until now the error was dropped and the reply came back blank; on 2026-10-21 that happened four times in a quarter of an hour. Anthropic overload is retried once after three seconds before the error is shown.
- An empty reply from any model is shown as an error ("returned an empty reply. Try again"), never as a blank message.

### Containment
- The word lists behind the pattern check match whole words only. "something" no longer counts as "meth" and "begun" no longer counts as "gun"; a message about posting a plush toy was flagged as a harmful instruction request on 2026-10-21.

### Dictation
- Live text: with Vosk (offline) dictation, the words appear in the message box as they are recognised, at the spot where the cursor was when you tapped the mic; the finished transcript takes their place. Android and desktop. Whisper (OpenAI) still shows its text only when you stop.
- A Copy button appears in the message box whenever it has text, for using Zynkbot as the dictation tool and pasting the text elsewhere. A Clear button beside it empties the box.
- Desktop (Linux and Windows): Vosk closes an utterance at a pause of about a second, and the desktop code read back only the last one when you stopped, so the words before a pause could be lost. Every finished utterance is now kept and joined.

### Web search
- What the model wrote around its search request is kept, whichever side of the request it wrote it on. Until now everything after the request line was dropped, and GPT had started putting the request first and the answer after it, so the answer the user watched appear vanished when the search ran.

### System Controls
- One section per page: the sidebar is a list of buttons (Getting Started, Containment, Current Model, Voice, ZynkSync, ZynkLink, Knowledge Base, Snap-ins, ZynkCluster) and each opens a single page that covers the sidebar. Getting Started stays at the top until onboarding is done, then drops to the bottom; a checkbox on its page pins it back.
- The contradiction window's explanation box has a dictation button.

### "Hey Zynk" on Android
- Turning the toggle off and on no longer kills the listener: the models are loaded safely against a stop or a second start arriving mid-load (KI-097), and a resume after a pause reloads them if a stop took them away.
- The listener no longer dies after a false wake (KI-096): two paths could restart it in the same instant, the second stream was taken for another app's recording, and both were stopped with nothing to restart them. Restarts are now one at a time, and a once-a-minute check restarts a listener that is off for no reason.
- Microphone control rebuilt (step 1 of the plan in the 2026-10-20 audit). The listener service owns one microphone stream and one state: off, loading, listening, dictating, turn (chime, thinking, speaking), paused for another app. The wake detector, the hands-free dictation, the assistant session's dictation and the spoken reply take that stream in turn instead of each opening their own recorder, so the fourteen separate start, stop and re-arm paths that collided (KI-096, KI-097) no longer exist; every one of them is now an event into a single state machine, with a unit test (the first Kotlin test in the project). The once-a-minute check stays as the net. The mic button in the app keeps its own stream until step 2.
- The wake-word verifier was learning the television. Every wake the model dismissed as "not a request" was being filed as a real wake (the "handled" result was read as "spoken"), so a phone sitting by a TV trained its voice profile on the TV: 26 of 74 wakes on the Pixel on 2026-10-20. The label is now decided in one place: a spoken answer or a clock command is real, everything else is false. Because the clips already on the phone cannot be sorted afterwards, the first start of this build clears them and the profile built from them, once, and collection starts over. The phone trains itself again after 20 real wakes.
- The verifier can now only get better or stay the same. Its training set is cumulative: when old clips are pruned to save space, a labelled clip's features and label stay (about 2 KB each), so a retrain never trades last week's negatives for today's. And a retrained profile replaces the one in use only if it is at least as good on clips it did not train on (five-fold held out, within one point on both rates and no drop in their sum); otherwise the old profile stays and the log says why.
- A hands-free transcript the containment guard flagged is no longer read aloud as a warning: a "not a request" verdict carries no warning, and the guard's banner is never spoken.
- A wake followed by silence closes after 6 seconds instead of holding the microphone for 30.
- A detection while the phone itself is playing sound (a notification tone, an app's "boing", a video), or in the second after, is ignored. An echo canceller is requested on the microphone stream; the log says whether the phone enabled it. Trade-off: "Hey Zynk" does not wake the phone while the phone is playing music.
- A detection with no human voice in the previous 2 seconds is ignored (Silero VAD, a 2.3 MB pre-trained speech-or-not model, MIT licence). With no voice for 2 seconds the wake-word models are not run at all, which also saves battery.
- A quiet voice from across the room is no longer thrown away: the old "too quiet" rule (−46 dBFS) rejected two real wakes at 0.99 confidence on 2026-10-19. With the speech gate deciding whether a voice was present, the level only has to rule out true silence (−65 dBFS).
- The personal verifier is trained on the phone from your own wake clips, the way Siri keeps learning your voice: every answered wake is a positive example, every fruitless one a negative, retrained as they accumulate. Nothing leaves the phone. It logs its opinion ("verifier would reject") until its held-out numbers clear the bar (at least nine in ten of your own wakes kept and nine in ten of the false ones stopped), then enforces on its own, and a later retrain never switches enforcement off. It is a noise and TV filter, not a voice lock.

## [0.9.6-beta4] — 2026-10-17 <!-- branch sync-rebuild, merged into v1 2026-10-16; text reviewed by Matt 2026-10-16 -->

### ZynkSync, rebuilt (memory sync between your own devices)
- Every change (a memory added, edited or deleted; a message; a thread renamed or deleted; an API key) goes into a queue on the device that made it, and each paired device takes the queue from where it left off. There is one way for a change to travel; the old push-and-compare paths are gone.
- A change reaches the other devices within seconds. The 60-second cycle stays as the safety net.
- A device joining the network receives everything in slices, and what is written during the catch-up still arrives.
- A device that was away catches up when it returns, even one away longer than the queue keeps (7 days): it gets a full re-send, and nothing doubles.
- Deletions win: a memory deleted on one device stays deleted everywhere, a restore from a cloud backup leaves deleted memories deleted, and a thread deleted on one device closes on the others where it was open.
- Copies of one memory that existed on two devices before the rebuild merge into one instead of doubling.
- API keys and the backup key travel the same queue. The newest save wins even when one device's clock is wrong (KI-081).
- A joining device and the host's other devices see each other at once, no restart needed (KI-075). A removed device that re-joins gets a full catch-up (KI-095).
- A message meant for one device is refused by any other (KI-053). A wrong pairing code says so: "Invalid or expired pairing code".
- Restore is one button and restores memories from the cloud backup, nothing else. A reinstall is a new device and pairs by code. (An earlier design of 2026-10-02 carried the device identity in the backup so a reinstall was the same device; it was withdrawn on 2026-10-08, KI-092, before this beta. This line was wrong until 2026-10-27.)
- ZynkSync panel: Refresh shows it was pressed, and the per-device ✕ is a labelled Remove button.

### Android
- Back backgrounds the app instead of killing the sync server (KI-077). Peers reconnect after a restart.
- The app asks once for the battery-optimisation exemption, from its second start (KI-087).

### Voice
- Hey Zynk, Android: the voice check that runs after the wake word fires (the personal verifier) now enforces on its own. A profile the phone trains from the owner's wake clips is switched on when, tested on clips it did not train on, it keeps at least 90% of the owner's wakes and stops at least 90% of the noise, TV and other-voice clips; a retrain never switches enforcement off. The built-in profile is a new one in the shift-tolerant format (version 3) for the developer's phone, built from 45 wake clips against 110 noise/TV clips and 27 of another person saying the wake word: held out it kept 41 of 45 and stopped 124 of 137. It is a filter, not a lock.
- Hey Zynk, Android: teaching it your voice. With the wake word on, a card on the Voice page has a Start button: say "Hey Zynk", wait for the count to move, say it again, twenty times. Each one only chimes. Until the twenty are in, "Hey Zynk" answers nothing; Pause stops it counting as well, so a television cannot add to the twenty while you are away. The count is kept across app restarts. Start and "Teach again" first show what teaching does and that it keeps learning afterwards; "Teach again" also says it clears the clips already on the phone. "Send my wake-word clips" lights up once a taught set is on the phone, for a tester who is asked to send theirs. At twenty the phone trains its personal voice profile from those clips and switches it on if it tests well enough (see above). "Teach again" starts over, clearing the phone's earlier wake clips and its own profile. Until now the phone only learned from wakes that ended in a spoken answer, and a wake you tapped away was filed as "not me", so there was no way to teach it on purpose.
- Hey Zynk, Android: the speech gate (the check that someone is talking before the wake model runs) recovers from an error instead of switching itself off for the rest of the run. The error came from the detector being reset from another thread mid-chunk; the reset now happens on the audio thread. On 2026-10-21 the gate was off for twenty minutes without anything in the log but one line.
- Timers understand spoken fractions ("two and a half minutes", "half an hour"). A misheard word between "timer" and the number no longer loses the timer (KI-093).

### Logs and settings
- Every printed line also goes to logs/zynkbot.log on disk, rotating at 5 MB. Problem reports include it (KI-082).
- Windows: a saved local model path no longer hides the cloud models after a restart (KI-083).
- History panel: a crash in the production build when sync updated the list (fixed).

## [0.9.6-beta3] — 2026-09-22 <!-- branch v1, started 2026-09-15; Link and Windows lines drafted by Claude 2026-09-21, review wording -->

### Desktop and phone
- A phone with only a cloud API key answers typed messages from the first one. The fresh-install placeholder "local" was kept as the model choice on a phone whose key was typed in by hand (a pushed key refreshed the choice, a typed one did not), so every typed message failed with "Local models not supported on Android" while "Hey Zynk" answered normally. Typed chat now falls back to a provider with a key the way hands-free always has, and a phone no longer stores "local" as a choice (tester, Pixel 7a, 2026-09-22).
- API Keys: once a key is stored, the provider button reads "View plan" and opens the provider's plan/billing page (it said "Get Key" and opened the key page whether or not you had one).

### Android voice
- A hands-free answer that came from a web search shows the same "View n search sources" drop-down under it that a typed search gets; the links were only described in the spoken answer before (2026-09-22).
- A hands-free question that needs a web search always gets one now, no confirmation step. There was never a way to say "yes" to "want me to search?" by voice, so a spoken "yes" was heard as a new, unrelated question (GitHub #26). The "Auto-execute in voice sessions" toggle this used to depend on is removed; it no longer had anything to control.
- Hands-free web search: the search fetches its three source pages at once instead of one after another, and says "Let me check that" as soon as it starts. The whole round trip used to be silent and could run past a minute (tester report, 2026-09-20).
- A hands-free listen ends when the room falls well below your own speaking level, not only when it falls to the pre-speech floor; a room that settled at a steady murmur after you finished kept the recording running to the 30 s cap. The level is logged every 2 s while listening.

### Memory
- The "Remembered on request" filter in the Memory Manager now shows the memories stored with "Remember:"; it had matched nothing since the feature shipped (KI-059).
- A memory can no longer be given a date in the future (a memory from 9/5 was filed under December, KI-066).
- About me says how many memories have no date and so are not on the timeline.

### Android voice (tester report)
- "Set a time for five minutes" sets a timer: dictation often drops the r in "timer" (GitHub #29).
- The share folder has an "Add file" button on Android. Files placed in `Download/ZynkbotShare` by other apps are invisible to Zynkbot under scoped storage; the picker copies the file in. The Kotlin side existed, the button did not.

### Everything
- The app now starts in Sovereign mode, not Guardian. Guardian's blocking is a demo of a safety layer, built on a classifier not tuned for real conversation; a new user's first hard block from it was a bad first impression for a feature that exists to be shown, not to gate real use. Sovereign warns on the same triggers instead of refusing. Child and HIPAA modes are unchanged and still opt-in.

### Chat
- You can message your own devices. Any device paired for sync now shows a Chat button in ZynkSync settings; a message to it travels over the verified sync connection, so a note typed on the PC arrives on the phone with no link pairing. Messages to another user's linked device work as before.

### Chat
- Every reply has a 🔊 Read aloud button next to Copy and Regenerate; tap again to stop. Uses the same voice as spoken replies, so it needs an OpenAI key.
- When the model answers and then runs a web search on its own, its first answer now stays on screen and in the history as its own message, and the searched answer appears beneath it. It used to be replaced.
- Resolving a memory contradiction closes the dialog the moment you confirm; the resolution runs behind it. It used to hold the dialog for the whole resolution, including a 10-second timeout for every paired device that was unreachable. The blocking pop-ups it showed are a short notice at the bottom of the window now.

### Android voice
- In-app dictation lets go of the microphone on every failure, not only on a clean stop. A dictation that broke mid-recording kept the microphone, and the wake-word listener, which pauses whenever anything else records, stayed deaf until the app was restarted (Pixel, 2026-09-19, twenty "Hey Zynk" attempts ignored). The listener also resumes on its own after two minutes paused, whatever is still recording.

### Link
- A file added to a shared folder shows up on other devices at their next Refresh. The list served to other devices was rebuilt only when the sharing computer opened its own ZynkLink menu, so two images moved into the laptop's folder stayed invisible from the desktop through any number of Refreshes (2026-09-22).
- The Refresh button shows its own status ("Refreshing…", then "✓ Refreshed") instead of a message at the bottom of the panel that vanished after two seconds.
- Files shared from a Windows computer are listed with forward slashes. A file in a subfolder was recorded as `sub\file.txt`, which Linux and Android read as one oddly named file rather than a file in a folder (found in the laptop's share manifest, 2026-09-21).
- Android: Zynkbot is now a location in the phone's Files app, beside Downloads and Drive, and appears in every app's Share menu. Anything put there, by Share, by moving it in Files, or with ➕ Add file, is shared with linked devices, and downloads from linked devices land there. Until now the share was a folder under Download, and Android hid every file another app put in it from Zynkbot (KI-015). Files Zynkbot owned in the old folder are moved over once.
- Android: after downloading an image from a linked device, Zynkbot offers to add a copy to the photo gallery (Pictures/Zynkbot), since the new location is not a media folder.
- Link: a file sent to the Knowledge Base from a linked device now shows "Indexing n / N pieces" after the transfer, and is marked done only once it can be asked about. On a phone a large PDF indexes for minutes after the transfer bar fills; before, the bar said finished while chat silently waited on the indexing.
- Link: the "Shared With Me" list asks every linked device at once and waits at most 5 seconds for each, instead of one device at a time with a 60-second wait; a linked device that is switched off no longer holds the list back for a minute.
- Android voice: strict mode is off. After three wakes that did not end in an answered question it demanded a level of confidence a real "Hey Zynk" rarely reaches and the phone went deaf for ten minutes. A wake that ends in an answer, spoken or on screen, now counts toward the "Send my wake-word clips" total.

### Windows
- Knowledge Base: deleting a file now waits for your answer. The confirmation used the browser's own prompt, which on Windows returns "yes" before the dialog appears (KI-047), so the file was deleted first and Cancel could not stop it.

### Sync
- A desktop's conversation history now reaches the phones. The "sent up to here" marker for history lived in memory, so every app start re-sent the whole history in one request, the phone rejected it as too large, and the backlog was silently dropped. The marker is now stored per peer and moved only once the peer has accepted the push; a push carries at most 300 messages, so a large history arrives over a few sync cycles (KI-068).
- Internal: the shared peer-to-peer layer (identity, certificate, HTTPS server, pinned client, device registry, presence, peer verification) is its own module, `transport`. ZynkSync, ZynkLink and ZChat are now three services on it, each registering its own routes and owning its own tables. No behaviour change; the two-peer harness is the check.
- A device that had never generated a pairing code had no record of itself, so every sync it started failed while recording the sync time (foreign-key error). Found by the new two-peer sync test harness; the device now records itself when its server starts.
- On the first sync between two devices with no memories, conversation history was never sent (the code returned early as "nothing to sync"). History now moves regardless.
- Pairing accepts `host:port`; the sync port is defined once and a peer's port is always taken from its stored record.

### Android voice
- The wake-word verifier can be read from the app's data folder (a personal profile no longer needs a release), understands the new position-independent profile format, and a profile marked for "whoever owns this phone" enforces without knowing the user id.

## [0.9.6-beta2] — 2026-09-14 <!-- drafted by Claude 2026-09-14 from a tester's history report -->

### Conversation history
- "New" starts a new thread. It used to empty the screen and keep the same thread, so everything said afterwards — typed or hands-free — was appended to the conversation the user thought they had left, and History showed one thread holding several conversations with no previous one to go back to.
- A thread appears in History as soon as its first message is sent, under "Current thread", instead of only after the first reply had finished.
- Threads can be renamed (pencil next to the pin). A blank name goes back to the automatic one. The delete × in History is larger and easier to tap on a phone.
- Message counts were 2 short on every thread (a one-exchange thread read "0 messages"); counted correctly now, and existing counts are repaired on first start.
- Android: a hands-free exchange that finished while the app was in the background is shown in the chat when the app comes back to the front.

### Android stability
- The personal wake-word verifier no longer enforces for anyone (it still scores and logs, and clips are still collected): in use it rejected most of its owner's real "Hey Zynk"s. It returns once retrained.
- Fixed a repeating crash ("Unable to start service SyncForegroundService") when Android restarted the app's sync service after reclaiming memory: the restart happens with the app in the background, where a foreground service is not allowed, and the refusal was not handled. The service now declines such restarts instead of crashing (reported by a tester on Android 17, reproduced on a Pixel 10 Pro XL, 2026-09-14).

## [0.9.6-beta1] — 2026-09-13 <!-- draft by Claude 2026-09-09, extended 2026-09-13 after the four-device test run; review wording -->

### Android voice
- Every "Hey Zynk" now runs natively through the Android assistant role: chime, Z overlay, tap-Z-to-cancel, Stop, replies spoken with the built-in voice and joined to the current thread. The old in-app wake path is gone.
- Timer, alarm and stopwatch by voice, handled by the clock app without a model call; stopwatch falls back to opening the clock app.
- Fewer false triggers: silence gate, back-off after repeated fruitless firings, faster close when nobody speaks, a default-deny instruction for TV and fragments, and a personal verifier trained from the owner's own clips (enforcing on the developer's phones; off with clip collection on for testers).
- The personal wake-word verifier enforces for the user it was trained on (by user id, which survives reinstalls) and only logs for anyone else.
- When Android refuses the in-app assistant-role prompt (seen on a GrapheneOS Pixel, Android 17), Zynkbot opens the system's assistant picker instead.
- The Vosk/Whisper selector applies to hands-free too; Whisper falls back to Vosk.
- "Remember colon …" saves a fact word for word by voice; Vosk's misspellings of "colon" are accepted.
- Chime and spoken replies use the media volume.
- A hands-free listen may run up to 30 seconds (was 12), long enough to dictate a paragraph.
- Wake word is opt-in (off by default) with the battery and false-trigger facts stated on the toggle.
- The detector skips its models while the room is quiet (below −52 dBFS for 2 s): an overnight run had it working at full rate under the screen-off wake lock and drained a Pixel.
- Coming back from the assistant-settings screen no longer leaves a black screen until the first tap (three attempts; the last one verified on stock Android and GrapheneOS).
- First-run fixes on Android: a startup deadlock, the assistant picker, keys pushed from another device showing without a restart, the name asked twice, and the desktop's model-download prompt no longer offered on a phone.
- The screen-off path (used when the assistant service is not bound) now plays the sent tone *before* the round trip, listens up to 30 s, and keeps the detector paused until the reply has been spoken — it used to hear the phone's own answer and fire on it.
- The first spoken line after the text-to-speech engine connects is no longer swallowed (GrapheneOS engine).
- "Play a tone when a trigger hears nothing" toggle in Voice Settings, off by default: a false trigger costs one chime and then silence.
- When the wake word is being cautious after repeated empty firings and drops a plain statement, it now says "Were you asking me a question? I'm only answering questions for a few minutes" instead of nothing.
- Whisper dictation in a noisy room: the silence detector follows the quietest level of the last two seconds, so an air conditioner no longer keeps it recording to the 30 s cap.
- Phones without a trained voice profile keep the 60 newest wake clips (20 with one), and Voice Settings has a "Send my wake-word clips" button with a live count that lights up at 30 real clips and hands a zip to the share sheet — nothing is sent by the app itself. An answered question labels its clip as real, not only clock commands.
- The personal verifier's threshold is 0.55; the shipped profile enforces for its owner only.

### Desktop
- Every "are you sure?" prompt is a native dialog on all platforms. On Windows the browser's own prompts were being skipped as if answered "yes": the Einstein demo loaded and Clear All deleted 59 memories with no question asked.
- Ollama is detected on startup and when Settings opens (running / installed but not running / not installed), with the live model list; the Ollama command is found in its standard install locations even when the app was launched with a minimal PATH.
- Paired phones always get the model selected on the desktop through the Ollama proxy; the phone no longer chooses, and custom-endpoint settings are no longer pushed between devices.
- OpenAI's "-pro" models are no longer offered (they are not chat models and returned 404); selecting text in a dialog no longer selects the page behind it; one close button per dialog.
- History shows the current thread first; the empty knowledge base offers "Add files"; the safety classifier fails open on very long input instead of falling back to keyword matching.

### Windows
- Offline Vosk dictation works on Windows: the installer carries the Vosk library, its runtime DLLs and the model. (Previously compiled out; Whisper was the only option.)
- The installer now installs per machine, under `C:\Program Files\Zynkbot`. The per-user location it used before was the app's own data folder under a different letter case, so program files sat beside the database and models. The uninstaller never deleted user data, but an older installer will quietly add a second, per-user copy next to the fixed one — remove it from Settings → Apps if you installed an earlier beta.
- The uninstaller's "delete all data" also removes the WebView profile that had let a "fresh" install inherit old preferences.

### Build and release
- The release workflow builds and signs the Linux (.deb, .rpm, AppImage), Windows (NSIS) and Android (APK and AAB) packages on a tag, and can be run by hand without one; Linux and Windows packages carry the Vosk library and model.

### Reporting
- Problem reports no longer mask ordinary file paths (the base64 pattern matched path segments); keys, tokens and pairing codes are still masked.
- With the conversation box unticked, the report's log tail no longer quotes your message, memory titles or the reply (a report on 2026-09-09 still carried them).

### Memory
- Memories carry an event date, a category from a fixed list, tags, tone and named entities from the same call that decides whether to remember; older memories are annotated in the background once.
- "About me" report in the Memory Manager, read from the local database; every line opens its memory; tap a tag to filter.
- Memories from hands-free turns are marked and listed separately.
- After tapping a tag in About me, an "← About me" button on the filter banner returns to the report with the Tags section open.
- Memories stored with "Remember:" are marked when written; a "Remembered on request" checkbox in the Memory Manager shows only them, and About me opens with a "You asked me to remember" section (20 at a time). Memories stored before this build carry no mark.
- Explicit Remember is stored even when it contradicts an older memory; a contradiction no longer blocks it.
- One timestamp format everywhere; duplicate conversation rows removed and prevented.
- Small local models that echo the prompt's headings instead of the memory marker are tolerated: the fact is still stored, the heading no longer leaks into the reply or the spoken answer ("Part 2:" included).
- A request no longer dies silently when a recalled memory holds an emoji, an accented letter or an em dash at a particular position (a byte-based cut in a log line; six such places replaced).
- The model is given the current date and time, so it stops trying to search for the date.

### Sync
- Deleting a stale device entry can no longer knock a live device at the same address off the mesh: removal notices name their target and are ignored by anyone else.
- The Ollama proxy log names the requesting device.

### Knowledge base
- Fabrication guard: the model is told what the search found (real matches, weak matches, nothing) and the reply header says when an answer is not grounded; no memory is extracted from an ungrounded reply.
- Android: Add files through the system picker; "All files access" no longer requested.
- The log names the document, chunk and score behind every knowledge-base answer.

### App
- Report a problem: a local text report (version, build, device, masked log tail, optional conversation) from System Controls or under any reply.
- Pinned threads; the current thread is restored after a restart; renamed peers show their new name; a peer's changed address is learned from its own requests; the backup key travels with "Push to all devices".
- 16 KB page-size alignment (Vosk 0.3.75, ONNX Runtime 1.29); full-screen-intent and photo permissions removed; Google Play readiness plan in the roadmap.
- Continuous integration runs the unit tests, the Tauri command-contract guards and an Android compile check on every push; the Android build compiles without warnings.

### Known at release (see docs/KNOWN_ISSUES.md)
- A device that pairs *after* the keys were saved does not receive them; press "Push to all devices" once (KI-055).
- An Anthropic key created at organisation level needs a workspace; create the key inside a workspace (KI-056).
- A false wake-word trigger in steady noise on the Whisper engine can produce a junk memory; use Vosk, the default (KI-057).
- Reinstalling a phone leaves a stale copy of it in other devices' lists; delete it (KI-050).
- Conversation-history sync can duplicate or skip threads; rebuilt after the beta (KI-028).

---

## [0.9.5-beta1] — 2026-08-23 — Offline Voice Dictation + Stability Fixes

### Highlights
- Offline voice dictation via Vosk ships on Linux (cpal mic capture) and Android (Kotlin bridge + bundled model)
- OpenAI Whisper (cloud) available as an alternative voice engine, selectable in Settings
- Android cold-start black-screen eliminated
- ZynkSync restart crash (port reuse) fixed

### Features
- **Vosk offline dictation (Linux)** — `vosk_desktop.rs` + `cpal` mic capture; transcription runs on-device with no internet required
- **Vosk on Android** — Kotlin bridge in `MainActivity.kt` with bundled `vosk-android` AAR; same command surface as desktop
- **Voice engine selector** — Settings dropdown chooses between Vosk (offline) and OpenAI Whisper (cloud)
- **Cloud backup (R2/S3)** — AES-256-GCM encrypted backup includes memories, conversation history, and embeddings; passphrase-derived key; tombstone-safe restore propagates to sync peers

### Bug Fixes
- **ZynkSync restart crash** — `SO_REUSEADDR` on `TcpListener::bind` (port 57963); rapid stop/restart no longer fails with "address already in use"
- **Android black-screen cold-start** — CSS `html, body, #root { background: #181a20 }` prevents flash before React mounts; React `needsSetup === null` guard shows a loading screen; Android `windowBackground` theme attribute set to match app background

### Internal
- Vosk dependency moved to `[target.'cfg(target_os = "linux")'.dependencies]` — Windows CI no longer tries to link `libvosk.lib`
- `libasound2-dev` added to Linux CI apt-get step (required by `cpal`)
- `build.rs` rpath/link-search block gated to `#[cfg(target_os = "linux")]` — MSVC no longer rejects `-Wl,-rpath` flags
- Release workflow: `prerelease` field set automatically from tag name (`beta`/`alpha`/`rc` suffix → prerelease)

---

## [0.9.3] — 2026-08-01 — Ensemble, Model Picker, and Sync Fixes

### Highlights
- Ensemble Mode overhauled with parallel execution and improved synthesis
- Mistral added as a 4th API provider
- Per-provider model picker — choose which model each provider uses
- Several ZynkSync correctness fixes that were silently breaking contradiction detection and deletion propagation

### Features
- Mistral API support (alongside Anthropic, OpenAI, and xAI)
- Per-provider model picker in Settings — select the specific model for each API provider
- New conversation button
- Ensemble Mode Phase 1 now runs all models in parallel (previously sequential)
- Ensemble Mode Phase 2 synthesis improved: better consensus detection, tighter memory injection, API fallback if local model fails
- GPU conflict guard in Ensemble modal — warns when Custom/Ollama and a local GGUF are both selected (shared CUDA device)
- Live-verified model lists for all four providers
- ZynkLink file visibility and own-share UX polish

### Bug Fixes
- **Contradiction resolution deletion not propagating** — resolving a contradiction by keeping the new memory deleted the old one locally but never notified paired devices; deletion now propagates immediately via ZynkSync
- **Real-time deletion missing tombstone timestamp** — delete-by-hash requests sent to peers were missing `deleted_at`, so the recreation guard on the receiving side could never fire; timestamp now included in all real-time deletion payloads
- **Ensemble local models grayed out on desktop** — production build check incorrectly disabled local GGUF models on all production builds, not just Android; desktop release builds now correctly allow local model selection
- **Memory decision API calls rejected by newer models** — `temperature` parameter sent to all providers was deprecated in newer models (claude-sonnet-5, gpt-5.5, grok-4.5), causing silent 400 errors and fallback failures; removed from all four provider helpers

### Internal
- Android CI: fixed NDK toolchain not on PATH during OpenSSL cross-compilation
- Android CI: fixed `sdkmanager` not found (setup-android action added before NDK install step)

---

## [0.9.2] — 2026-07-25 — Android Phase 1

### Highlights
- Android beta — full-featured app (API models + local ML stack) for Android phones and tablets
- Cross-device sync now works between Android and desktop over encrypted LAN
- Production release signing wired in; APK available on the GitHub release page

### Android
- App identifier set to `ai.containai.zynkbot` (matches Play Console draft)
- Full local ML stack runs on Android via Candle (embeddings, NER, safety classifier)
- ZynkSync, ZynkLink file sharing, and ZChat all functional on Android
- `MANAGE_EXTERNAL_STORAGE` requested at launch on Android 11+ for ZynkbotShare folder visibility
- `WRITE_EXTERNAL_STORAGE` requested on Android ≤ 9 (API 28)
- Foreground service starts correctly on Android 8 (API 26) with version-conditional notification channel
- `ZynkbotShare` folder created at `Downloads/ZynkbotShare/` on first launch

### Bug Fixes
- Fixed ZynkLink pairing showing "Remote Device XXXXXXXX" — acceptor now sends its device name during handshake
- Fixed tablet startup crash caused by permission request flow in `onCreate`
- Fixed Open in Files app crashing on Android 8 (external storage URI format invalid below API 29)
- Fixed `getShareDir()` and `openShareFolder()` crashing the app when called before storage permission granted

### UI
- Per-device expel button (×) in ZynkSync Synced Devices list — removes a device and notifies peers
- ZynkSync pairing code display no longer wraps mid-code on narrow screens

### Internal
- Release APK signed with production RSA-4096 keystore; signing config loaded from `keystore.properties` (gitignored)
- Repo-wide `.gitignore` patterns for `*.jks`, `*.keystore`, `keystore.properties`

---

## [0.9.0] — 2026-07-13 — First Public Release

First public release of Zynkbot as an open source project.

### Highlights
- Local-first AI assistant with persistent semantic memory
- Pure Rust/Tauri desktop app — no Python runtime required
- Supports local GGUF models, OpenAI, Anthropic, and xAI APIs
- Cross-device memory sync (ZynkSync) and peer-to-peer file sharing (ZynkLink)
- Device-to-device messaging (ZChat) with no cloud relay
- Conversation history with search and session resume
- Knowledge Base with RAG — index your own documents for semantic search
- Containment modes: Guardian, Child, HIPAA, Sovereign, Witness
- Multi-model Ensemble Mode with consensus detection
- Runs entirely on your machine — no telemetry, no phone-home

### Features
- Web search result links open in the system browser
- Image attachment support: JPG, PNG, GIF, WebP, BMP — routed to the vision API of the active cloud model
- ZynkSync pause/resume broadcasts to all paired devices instantly
- First-run setup wizard automatically downloads all required AI models
- Contradiction modal resolution propagates memory deletions to sync peers
- Session ID visible alongside User ID and Device ID in identity panel

### Security
- LAN traffic encrypted with TLS 1.3 (self-signed certificates, automatically trusted on pairing)
- Sync endpoints reject requests from unsynced devices
- Unsync propagates automatically to the peer device
- Pairing code rate limiting: 5 attempts per 5-minute window per IP

### Bug Fixes
- Fixed contradiction modal keep new/keep old buttons being swapped
- Fixed `original_text` not preserved when memory stored via contradiction resolution (KI-012)
- Fixed `original_text` not included in ZynkSync payloads (KI-013)
- Fixed contradiction modal crash on first memory conflict detection
- Fixed Anthropic streaming token counter always showing 0
- Fixed child mode system prompt not injected into OpenAI API calls
- Fixed sync never transferring memories when auto-sync was disabled
- Restored `Remember:` command for forcing memory storage
- Fixed `remove_api_key` not finding the `.env` file
- System memories no longer appear in user hybrid search results (KI-003)

### Internal
- `lib.rs` broken into 9 domain command modules for maintainability
- Verbose debug output gated behind `#[cfg(debug_assertions)]`
- CPU-only mode forced for embeddings and safety classifier

See [docs/FEATURES.md](docs/FEATURES.md) for the full feature list.

---

## Contributing to this changelog

If you're submitting a pull request, add a line to `[Unreleased]` describing what changed.
When a version ships, unreleased entries move under a new version heading with a date.

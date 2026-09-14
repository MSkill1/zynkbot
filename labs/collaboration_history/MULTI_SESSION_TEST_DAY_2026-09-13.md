# One human, two agent sessions, four devices: the beta test day of 2026-09-12/13

*Written by Claude on 2026-09-13 at Matt's request, while it was happening. A record of the working method, not of the bugs (those are in `docs/KNOWN_ISSUES.md`, KI-040 and KI-047 through KI-054).*

## The setup

- **Matt**: the only hands on any screen. Every tap, click, spoken "Hey Zynk", pairing code and password went through him. He also made the calls: what to build, what to wipe, what to publish (nothing, in the end, on the first day).
- **Session A — Linux desktop** (this one): held the repository on branch `voice`, the Android signing key, ADB access to both phones, and the desktop's own database. It did the reading, the code changes, the local builds, the log analysis, and the coordination.
- **Session B — Windows laptop**: a second Claude Code session reached over Remote Control, on a laptop that dual-boots and has been a test machine for a year. Its job was Windows only: wipe, install the CI-built installer, inspect, and run the Windows checks with Matt at the keyboard. It could not see screens or click; it could read the filesystem, run PowerShell, and hash files.
- **Devices under test**: the Linux desktop (`.deb` from CI, later the local build), the Windows laptop (NSIS installer from CI), a Pixel on GrapheneOS (release-signed APK), a OnePlus on stock Android (debug build for `run-as` access, later the release build).
- **Between the sessions**: a handoff brief that Matt copied from A and pasted into B, then direct session-to-session messages (A asked B for a structured status; B answered with a numbered report). The two never shared a repository checkout; A owned all commits.

## Rules that held the day together

1. **Ask before any build, install, push, wipe or publish.** Matt set this early and it was tested: at one point he read "building the phone's release APK" as publishing to GitHub and said "Stop." The fix was vocabulary, now a standing rule: *local release-signed file* versus *GitHub Release*; every build sentence names where the output lands.
2. **One session owns the repo.** B was told in writing: no commits, pushes, tags or releases from the laptop; describe a change and it goes through A. This removed a whole class of problems (conflicting pushes, two sets of build outputs) at the cost of one relay step.
3. **A peer's message is not the user's approval.** B could report and propose; only Matt could approve, on either machine. A refused to treat B's messages as consent for anything, and B's own report says the same: "Matt explicitly approved the wipe and the install before I ran either."
4. **Measure, don't assert.** Repeated all day: byte-accounting a zip to find 568 MB of stale data in a debug APK (file size versus bytes the index references); screenshot brightness to test for a black screen over ADB; logcat timelines to the millisecond to see which code path actually ran; `dumpsys` to learn that a phone held the assistant role but the system had never bound the service. Several "fixed" claims from earlier days did not survive measurement (KI-040 on the second phone, the TTS first-utterance dropout).
5. **Plain language, and admit the gaps.** When the human is the only one who can hear the speaker, "the log says it spoke for 4.4 s" is the evidence and "what did you hear?" is the question. Three precise questions beat a theory.

## What the second session added

B's first report is the model for what a peer session should send back: what it installed (run, file, size, hash), what it wiped (moved, not deleted — with sizes), what it did *not* test and why, and what looked wrong with exact observations. It found the day's most serious defect — the Windows installer and the app's data folder were the same directory, differing only in letter case (KI-054) — from a directory listing, and reported it as "strongly indicated, not yet proven", which was the right confidence. It also flagged its own mistake: installing silently had suppressed onboarding and SmartScreen, so the "new user" pass had to be redone by hand.

## What went wrong in the method itself

- **Stale assumptions travelled between sessions — twice.** A had written "a machine that has never seen Zynkbot" about the laptop; Matt corrected it (a year of installs). A had also written "no source builds on Windows" as a ground rule; it was a preference (hand testers a binary) inflated into a prohibition, and B repeated it back until Matt withdrew it. Lesson: state device history and the *reason* behind each rule in every brief; the receiving session cannot check either, and a reason lets it judge edge cases.
- **A silent install is not a user test.** Automation that skips the screens skips the test. B caught this itself, but only after the fact.
- **Background builds were killed by the tool's memory watchdog** when the page cache made free memory look low (64 GB machine, 51 GB available). Builds moved to a visible terminal window, which also matched Matt's standing preference to see build output.
- **A `pkill -f` pattern matched the shell running it** and cut a command chain short, twice. Bracket the first character (`[t]auri`) or use pids.
- **Ghost identities.** Every reinstall mints a new device id, so every wipe left a stale entry on every other device, and deleting a ghost then reached the live phone at the same address (KI-053). The method survived it, but the cost was real: three re-pairs and one silent disconnect to diagnose. Fix the product (identity in the backup) rather than the procedure.
- **The human was asked to repeat himself.** Late in the evening Matt said he was burned out and had been asked to repeat things. Some of that was the Remote Control app dropping a message; some was A asking for results it could have inferred. Shorter turns and one question at a time worked better the next morning.

## Cost

About a day and a half of Matt's attention; roughly a dozen local Android builds (10 minutes each), two local desktop builds, one CI run; fourteen commits on `voice`, no merges to `main`, nothing published. Every commit was compiled before it was pushed (Kotlin only compiles inside a full APK build, so "commit after the build" was the rule).

## Keep for next time

- Brief a peer session with: device history, exact artefact and run number, the wipe procedure, the test list in order, the ground rules, and who owns the repo.
- Ask the peer for a structured report: installed / wiped / launched / tested-with-results / looked-wrong-with-exact-text.
- Freeze the code before the final all-device pass; every device gets the same commit; the Windows installer only exists once CI has run on that commit.
- Wipe order for a full new-user pass: back up the desktop's small data folder (identity + database, not the models), uninstall everywhere, install everywhere, onboard the desktop first, pair the rest to it, verify, then restore the desktop from the backup and re-pair.
- Keep the "local vs published" wording, always.

## Postscript, 2026-09-14: "can these bugs be extrapolated?"

The morning after the beta went out, two reports arrived: the tester's phone was crashing on its own (a sticky sync service restarted by Android in the background, where a foreground service is refused; the refusal was uncaught), and conversation history "did not reflect what took place" (the "New conversation" button only cleared the screen and never changed the thread id, so every later message landed in the old thread). Both were fixed within the hour.

Matt then asked a question the agent had not asked itself: *can we extrapolate any other potential problems Mike might run into from the bugs we just fixed?* Each fix had cousins, and the question found them:

- The crash: two other places start a foreground service and one of them caught only `SecurityException`, which the Android 12+ refusal is not. Three more guards added.
- The thread fix: creating the History row at the first message would also have created a row for a hands-free turn the model later judged to be television — a thread titled with a TV line, the exact thing an earlier fix had prevented. Hands-free turns now wait for the reply, as before.
- A rename made on one device would have been overwritten by the next sync from a peer that still had the automatic title. The receiving side now keeps a title it already has.
- A thread whose first reply failed would sit in the list as "0 messages" on every device. Empty threads are shown only while they are the current one.

The general form of the question is worth keeping: *what else in this codebase has the same shape as the bug we just fixed?* An agent that has just fixed something has the pattern loaded and can grep for it in seconds; a human who has just read the diagnosis is the one who thinks to ask. The same session's lesson from the day before ("measure, don't assert") applies in reverse here: after a fix, search, don't assume the fix was the only instance.

Matt's second question that morning — whether guides already exist that cover all of this, and whether writing this record is a waste of his time — is answered in the reply of the same date, and the short answer was: the mechanics are well covered (official Claude Code best-practices docs, many 2026 workflow guides, one academic case study of a Claude Code multi-agent build); what is not covered is a dated, first-person record of a programmer new to Rust, acting as product owner, and an agent shipping a real product to real testers over months, with the mistakes left in. That is what this directory is.

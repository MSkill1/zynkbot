# Can the wake-word verifier be a lock? — an experiment, not a feature (yet)

*Filed 2026-09-13 after a discussion between Matt and Claude. Nothing here is built; this is the test that decides whether "only respond to my voice" can ever be described as security.*

## The disagreement, stated fairly

Matt's position: a speaker verifier that only accepts the owner is a high barrier. To defeat it an attacker needs a recording of the owner saying "Hey Zynk" and a way to play it back; that is comparable in practice to a PIN, and banks already verify customers by voice.

Claude's position: the barrier is unmeasured, and the two things it is being compared to are not the same kind of thing.

- The bank systems are trained on minutes of speech across many calls, use dedicated speaker-recognition models with liveness checks, and sit alongside other factors. Ours is a straight-line classifier over 1,728 numbers from a wake-word model, trained on 33 clips of one person against 60 clips of noise. It has never been shown a *different person* saying the phrase.
- A PIN has a known false-accept rate (1 in 10,000 per guess, with lockout). The verifier's false-accept rate against other voices is unknown. On 2026-09-12 two triggers that were not Matt's voice scored 0.485 and 0.500 against a 0.50 threshold — coin-flip territory — while Matt's own scored 0.79–0.98.
- Replay is cheaper than it sounds: a two-second clip of the phrase is enough, and anyone in the same house hears it daily. Voice cloning from a short sample is a consumer tool now, so "record him" is not even required.

Neither position is settled by argument. The bank comparison is the strongest point for Matt; the untested-against-impostors point is the strongest for Claude. Hence the experiment.

## What "secure enough" would have to mean

A lock needs a bound on false accepts. Proposed bar, to be argued before the test, not after: **no impostor in the test set passes in 20 attempts**, and **a replayed recording of the owner is rejected** (which requires some liveness signal the current model does not have). If only the first holds, the honest label is "responds to your voice", never "lock".

## The test

Equipment: the Pixel or OnePlus with the owner's verifier enforcing (threshold as shipped, 0.55), logcat capture of `WakeWordService` lines (`verifier=` score on every trigger).

1. **Owner baseline** — Matt, 20 attempts at arm's length, 20 across the room. Record every score. Expect ≥ 0.79 as on 09-12.
2. **Impostors, natural voice** — at least five other people (aim for two who sound like Matt: family), 20 attempts each, same distances. Record every score. Any accept is a finding.
3. **Impostors, imitating** — same people, told to imitate Matt after hearing him. 20 attempts each.
4. **Replay** — Matt's own clip (any `.real` file from `wake_triggers`) played from another phone's speaker at arm's length, 20 attempts. Expect it to pass; the question is by how much.
5. **Clone** — a synthetic "Hey Zynk" made from 30 s of Matt's speech with a consumer cloning tool, 20 attempts.

Report: a table of scores per person per condition; the false-accept count at 0.55 and at 0.65; and the owner's false-reject count at each.

## What the result decides

- Zero accepts in (2) and (3): "only responds to my voice" can go in Settings as a *convenience gate* with a plain-language note that a recording defeats it. Still not called a lock.
- Zero accepts in (2)–(5): reopen the question properly — that would mean a liveness signal exists somewhere we have not noticed, which is unlikely with this model.
- Any accept in (2): the feature stays what it is today, a false-trigger filter.

## Cost

An afternoon with five volunteers and the two phones. No code beyond reading scores out of logcat. Worth doing before the personal-wake-phrase paid feature ships, because the marketing for that will be tempted to say "secure".

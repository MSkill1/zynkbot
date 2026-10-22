# Other-voice "Hey Zynk" samples

The on-phone verifier trains against noise/TV negatives only (`generic_negatives.json`
has no human voice), so a self-taught profile learns "not noise" well and "not another
person" barely. A small one-time set of other people saying the wake word fixes that.

**Collecting:** https://containai.ai/voice/ (source: the `containai-site` repository,
`voice/index.html`). The person opens it on their phone, taps Start, says "Hey Zynk"
twenty times with a counter, taps Send and shares one WAV: 16 kHz mono, twenty 3 s
clips each followed by 1 s of silence, so the clips split at fixed offsets
(clip k = samples [k·64000, k·64000 + 48000)). The page uploads nothing.

**Using:** run each clip through the wake model on the desktop and keep only the ones
that fire (a cough or a word counted by the page's level detector is dropped), then
compute the 192 pooled features (`verifier_sim.py` pipeline) and add them to the shipped
negatives under an "other voices" heading. The phone's trainer must leave out any voice
set that resembles its own owner's clips, so a contributor's own phone is not trained
against them. That trainer change is not written yet (2026-10-22).

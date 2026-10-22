# Dry run of a containai.ai/voice recording: split the fixed-layout WAV into its clips,
# stream each through the phone's wake chain (mel -> embedding -> hey_zynk), keep the ones
# the detector fires on, and write their 192 pooled features at the fire point.
#   python split_voice_sample.py <models dir> <recording.wav> <out.json> [verifier.json]
# Layout from the page: 16 kHz mono, each clip 3.0 s followed by 1.0 s of silence.
import sys, json, wave, numpy as np, onnxruntime as ort
R, wav_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
verifier = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
so = ort.SessionOptions(); so.intra_op_num_threads = 2
mel = ort.InferenceSession(f"{R}/melspectrogram.onnx", so); emb = ort.InferenceSession(f"{R}/embedding_model.onnx", so); kws = ort.InferenceSession(f"{R}/hey_zynk.onnx", so)
run = lambda s, x: s.run(None, {s.get_inputs()[0].name: x})[0]
with wave.open(wav_path) as w:
    assert w.getframerate() == 16000 and w.getnchannels() == 1, "expected 16 kHz mono"
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
CLIP, GAP = 48000, 16000
n_clips = len(pcm) // (CLIP + GAP)
def vscore(f):
    if not verifier: return None
    m, s, c = (np.array(verifier[k], dtype=np.float32) for k in ("mean", "scale", "coef")); s[s == 0] = 1
    return float(1 / (1 + np.exp(-(verifier["intercept"] + ((f - m) / s) @ c))))
rows = []
for k in range(n_clips):
    # Each clip is scored with the second of silence that follows it. The chain needs
    # about 2.5 s of lead-in before its first score and the phrase sits near the clip's
    # end, so on the bare 3 s not one of the owner's own clips fired (2026-10-22); with
    # the gap, as in the continuous stream on the phone, they do.
    clip = pcm[k * (CLIP + GAP): (k + 1) * (CLIP + GAP)]
    melbuf, embbuf, best, best_feat, scores = [], [], -1.0, None, []
    for i in range(0, len(clip) - 1279, 1280):
        chunk = (clip[i:i + 1280].astype(np.float32) / 32768.0)[None, :]
        melbuf.extend(list(run(mel, chunk).reshape(-1, 32)[:5])); melbuf = melbuf[-76:]
        if len(melbuf) < 76: continue
        embbuf.append(run(emb, np.array(melbuf, dtype=np.float32).reshape(1, 76, 32, 1)).reshape(-1)); embbuf = embbuf[-16:]
        if len(embbuf) < 16: continue
        E = np.array(embbuf, dtype=np.float32)
        p = float(run(kws, E[None, :, :]).reshape(-1)[0]); scores.append(p)
        if p > best: best, best_feat = p, np.concatenate([E.mean(0), E.max(0)])
    level = 20 * np.log10(max(np.sqrt(np.mean((clip[:CLIP] / 32768.0) ** 2)), 1e-9))
    fires = best >= 0.72 and sum(1 for a, b in zip(scores, scores[1:]) if a >= 0.72 and b >= 0.72) > 0
    rows.append({"clip": k, "max_score": round(best, 3), "fires": bool(fires), "level_dbfs": round(float(level), 1),
                 "verifier": None if best_feat is None else vscore(best_feat), "features": None if best_feat is None else [round(float(v), 5) for v in best_feat]})
kept = [r for r in rows if r["fires"]]
print(f"{wav_path}: {n_clips} clips, {len(kept)} fire the wake model (two chunks >= 0.72)")
for r in rows:
    v = "" if r["verifier"] is None else f"  verifier {r['verifier']:.3f}"
    print(f"  clip {r['clip']:2d}: max {r['max_score']:.3f} {'fire' if r['fires'] else '----'}  {r['level_dbfs']:6.1f} dBFS{v}")
json.dump({"source": wav_path, "rows": rows}, open(out_path, "w"))

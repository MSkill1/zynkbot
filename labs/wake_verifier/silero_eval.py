import sys, wave, glob, os, numpy as np, onnxruntime as ort
model, root = sys.argv[1], sys.argv[2]
s = ort.InferenceSession(model)
def probs(path):
    with wave.open(path) as w: pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
    state = np.zeros((2,1,128), np.float32); ctx = np.zeros(64, np.float32); out=[]
    for i in range(0, len(pcm)-511, 512):
        x = np.concatenate([ctx, pcm[i:i+512]])[None,:].astype(np.float32)
        p, state = s.run(None, {"input": x, "state": state, "sr": np.array(16000, np.int64)})
        out.append(float(p[0,0])); ctx = pcm[i+448:i+512]
    return np.array(out)
rows = []
for wav in sorted(glob.glob(f"{root}/*/*.wav")):
    stem = wav[:-4]; real = os.path.exists(stem+".real"); false = os.path.exists(stem+".false")
    if real == false: continue  # unlabelled or double-labelled
    p = probs(wav)
    rows.append((wav.split('/')[-2], "real" if real else "false", p[-62:].max(), p[-78:].max(), (p[-62:]>0.5).sum()))
import collections
for label in ("real","false"):
    r = [x for x in rows if x[1]==label]
    m2 = np.array([x[2] for x in r]); m25 = np.array([x[3] for x in r]); n = np.array([x[4] for x in r])
    print(f"{label}: n={len(r)}  max-VAD last 2.0 s: <0.3: {(m2<0.3).sum()}  0.3-0.5: {((m2>=0.3)&(m2<0.5)).sum()}  >=0.5: {(m2>=0.5).sum()} | last 2.5 s >=0.5: {(m25>=0.5).sum()} | speech frames in last 2 s: median {np.median(n):.0f}")
    for d in sorted(set(x[0] for x in r)):
        rr=[x for x in r if x[0]==d]; mm=np.array([x[2] for x in rr]); print(f"   {d}: n={len(rr)} >=0.5: {(mm>=0.5).sum()}  min={mm.min():.2f}")

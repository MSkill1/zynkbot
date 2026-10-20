# Extract the phone's 192 pooled features (mean+max over the 16-embedding window at the fire
# point) for every clip, write the generic negative set, and simulate the on-device trainer.
import sys, glob, os, wave, json, numpy as np, onnxruntime as ort
R, root, out = sys.argv[1], sys.argv[2], sys.argv[3]
so = ort.SessionOptions(); so.intra_op_num_threads = 2
mel = ort.InferenceSession(f"{R}/melspectrogram.onnx", so); emb = ort.InferenceSession(f"{R}/embedding_model.onnx", so)
def run(sess, x): return sess.run(None, {sess.get_inputs()[0].name: x})[0]
def feats(path):
    with wave.open(path) as w: pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    melbuf, embbuf = [], []
    for i in range(0, len(pcm) - 1279, 1280):
        chunk = (pcm[i:i+1280].astype(np.float32) / 32768.0)[None, :]
        m = run(mel, chunk).reshape(-1, 32); melbuf.extend(list(m[:5])); melbuf = melbuf[-76:]
        if len(melbuf) < 76: continue
        e = run(emb, np.array(melbuf, dtype=np.float32).reshape(1, 76, 32, 1)).reshape(-1); embbuf.append(e); embbuf = embbuf[-16:]
    if len(embbuf) < 16: return None
    E = np.array(embbuf); return np.concatenate([E.mean(0), E.max(0)])
rows = []
for wav in sorted(glob.glob(f"{root}/*/*.wav")):
    d = wav.split('/')[-2]; b = os.path.basename(wav); stem = wav[:-4]
    real = os.path.exists(stem+".real"); false = os.path.exists(stem+".false")
    # train.py's positive globs (Matt's 2026-09-08/09 enrolment sessions)
    glob_pos = (d=="phase1" and (b.startswith("20260908-134") or b.startswith("20260908-1349"))) or (d=="phase2" and b.startswith("20260909-11"))
    if d == "mike_20260915": label = "mike_real" if (real and not false) else ("mike_false" if false and not real else "mike_both")
    elif glob_pos: label = "real"
    elif real and not false: label = "real"
    elif false and not real: label = "false"
    elif real and false: label = "both"
    else: label = "false" if d in ("clips","clips23","clips24","clips34","clips_px_0908","phase1","phase2") else "unlabelled"
    f = feats(wav)
    if f is None: continue
    rows.append((d, b, label, f))
X = np.array([r[3] for r in rows]); lab = np.array([r[2] for r in rows])
np.savez(f"{out}/features.npz", X=X, lab=lab, name=np.array([r[1] for r in rows]), d=np.array([r[0] for r in rows]))
import collections; print("labels:", dict(collections.Counter(lab)))
neg = X[lab=="false"]
json.dump({"version": 1, "dims": 192, "note": "mean+max of the 16x96 embedding window at the fire point, from false-trigger clips (noise, TV, electronic sounds) collected on the developer's phones 2026-09-04..09-22; no personal voice",
           "rows": [[round(float(v), 4) for v in r] for r in neg]}, open(f"{out}/generic_negatives.json", "w"))
print("generic negatives:", len(neg), "-> generic_negatives.json", os.path.getsize(f"{out}/generic_negatives.json")//1024, "KB")

# --- simulate the Kotlin trainer: standardise, balanced logistic regression, L2, full-batch GD
def train(Xp, Xn, iters=1500, lr=0.05, C=0.05):
    Xa = np.vstack([Xp, Xn]); y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    mu = Xa.mean(0); sd = Xa.std(0); sd[sd==0] = 1; Z = (Xa-mu)/sd
    n = len(y); wpos = n/(2*len(Xp)); wneg = n/(2*len(Xn)); sw = np.where(y==1, wpos, wneg)
    lam = 1.0/(C*n); w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1/(1+np.exp(-(Z@w+b))); g = (p-y)*sw
        w -= lr*(Z.T@g/n + lam*w); b -= lr*g.mean()
    return mu, sd, w, b
def score(mu, sd, w, b, X): return 1/(1+np.exp(-(((X-mu)/sd)@w+b)))
pos = X[lab=="real"]; mike = X[lab=="mike_real"]
print(f"positives (Matt) {len(pos)}, negatives {len(neg)}, Mike's real {len(mike)}")
rng = np.random.default_rng(0); idx = rng.permutation(len(pos)); folds = np.array_split(idx, 5)
nidx = rng.permutation(len(neg)); nfolds = np.array_split(nidx, 5)
held_p, held_n, held_m = [], [], []
for k in range(5):
    tp = np.setdiff1d(idx, folds[k]); tn = np.setdiff1d(nidx, nfolds[k])
    mu, sd, w, b = train(pos[tp], neg[tn])
    held_p += list(score(mu,sd,w,b,pos[folds[k]])); held_n += list(score(mu,sd,w,b,neg[nfolds[k]])); held_m += list(score(mu,sd,w,b,mike))
held_p, held_n, held_m = map(np.array, (held_p, held_n, held_m))
for thr in (0.1, 0.2, 0.3, 0.5):
    print(f"thr {thr}: Matt real kept {(held_p>=thr).sum()}/{len(held_p)}  noise stopped {(held_n<thr).sum()}/{len(held_n)}  Mike (other voice) let through {(held_m>=thr).mean()*100:.0f}%")
mu, sd, w, b = train(pos, neg); ps = score(mu,sd,w,b,pos)
print("auto threshold (5th percentile of own real scores, clamped 0.05..0.5):", round(float(np.clip(np.percentile(ps,5),0.05,0.5)),3))

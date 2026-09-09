# Personal wake-word verifier trainer (desktop). Usage:
#   python3 train.py <dir with melspectrogram.onnx + embedding_model.onnx> <clips root>
# Clips root holds the wake-trigger .wav files pulled from the phone (adb run-as, files/zynkbot/wake_triggers);
# the positive/negative globs below encode which sessions were real "hey zynk" and which were false triggers,
# so edit them for a new user. Writes hey_zynk_verifier.json; add the "owners" device-id list by hand.
# Requires numpy, onnxruntime, scikit-learn. Voice clips are never committed to the repository.

import sys, glob, os, wave, json, numpy as np, onnxruntime as ort
R = sys.argv[1]; S = sys.argv[2]
so = ort.SessionOptions(); so.intra_op_num_threads = 1
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
    E = np.array(embbuf)                       # 16 x 96, the classifier's own input at the fire point
    return np.concatenate([E.flatten(), E.mean(0), E.max(0)])
def stamp(p):
    b = os.path.basename(p); b = b.split("-", 1)[1] if b.startswith(("pixel-", "oneplus-")) else b
    return b[:15]  # YYYYMMDD-HHMMSS
pos = sorted(glob.glob(f"{S}/verifier/phase1/20260908-134*.wav") + glob.glob(f"{S}/verifier/phase1/20260908-1349*.wav") + glob.glob(f"{S}/verifier/phase2/20260909-11*.wav"))
neg = sorted(glob.glob(f"{S}/clips23/*.wav") + glob.glob(f"{S}/clips24/*.wav") + glob.glob(f"{S}/clips34/*.wav"))
neg += [p for p in glob.glob(f"{S}/clips_px_0908/*.wav") if stamp(p) >= "20260908-103700"]
neg += [p for p in glob.glob(f"{S}/verifier/phase1/*.wav") if "20260908-11" in os.path.basename(p)]
neg += [p for p in glob.glob(f"{S}/verifier/phase2/*.wav") if "20260908-19" in os.path.basename(p)]
neg = sorted(set(neg))
X, y, names = [], [], []
for p in pos:
    f = feats(p);  X.append(f); y.append(1); names.append(os.path.basename(p))
for p in neg:
    f = feats(p)
    if f is None: continue
    X.append(f); y.append(0); names.append(os.path.basename(p))
X = np.array(X); y = np.array(y)
print(f"positives {int(y.sum())}, negatives {int((y==0).sum())}, feature dim {X.shape[1]}")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
for thr in (0.3, 0.5, 0.7, 0.9):
    tp = int(((p >= thr) & (y == 1)).sum()); fn = int(((p < thr) & (y == 1)).sum())
    fp = int(((p >= thr) & (y == 0)).sum()); tn = int(((p < thr) & (y == 0)).sum())
    print(f"thr {thr:.1f}: real kept {tp}/{tp+fn}, false stopped {tn}/{tn+fp}")
print("lowest cross-validated score among the real clips:", round(float(p[y==1].min()), 3))
print("highest cross-validated score among the false clips:", round(float(p[y==0].max()), 3))
worst = np.argsort(-p[y==0])[:5]
print("false clips the verifier likes most:", [(names[np.where(y==0)[0][i]], round(float(p[y==0][i]), 2)) for i in worst])
clf.fit(X, y)
np.save(f"{S}/verifier/phase2_X.npy", X); np.save(f"{S}/verifier/phase2_y.npy", y)
import pickle; pickle.dump(clf, open(f"{S}/verifier/phase2_clf.pkl", "wb"))

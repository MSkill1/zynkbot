# Build a v3 (mean+max pooled, 192-number) personal verifier profile on the desktop, the way
# the phone's WakeVerifierTrainer would, from features.npz written by verifier_sim.py.
#   python build_profile_v3.py features.npz <owner user id> out.json
# Positives: the owner's "real" clips. Negatives: "false" (noise/TV) plus "mike_real", a second
# person saying the wake word, so the profile learns "not this voice" and not only "not noise".
# Held-out numbers use the phone's exact rule (five folds by index, per-fold 5th-percentile
# threshold) and are written into the file so the on-phone trainer has a bar to beat.
import sys, json, numpy as np
npz, owner, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = np.load(npz); X, lab = d["X"], d["lab"]
def train(Xp, Xn, iters=1500, lr=0.05, C=0.05):
    Xa = np.vstack([Xp, Xn]); y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    mu = Xa.mean(0); sd = Xa.std(0); sd[sd==0] = 1; Z = (Xa-mu)/sd
    n = len(y); wpos = n/(2*len(Xp)); wneg = n/(2*len(Xn)); sw = np.where(y==1, wpos, wneg)
    lam = 1.0/(C*n); w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1/(1+np.exp(-(Z@w+b))); g = (p-y)*sw
        w -= lr*(Z.T@g/n + lam*w); b -= lr*g.mean()
    return mu, sd, w, b
def score(m, X): mu, sd, w, b = m; return 1/(1+np.exp(-(((X-mu)/sd)@w+b)))
def auto_thr(m, Xp): s = np.sort(score(m, Xp)); return float(np.clip(s[int((len(s)-1)*0.05)], 0.05, 0.5))
pos = X[lab=="real"]; noise = X[lab=="false"]; other = X[lab=="mike_real"]
neg = np.vstack([noise, other]); nneg = np.r_[np.zeros(len(noise)), np.ones(len(other))]  # 1 = other voice
# held out, phone rule
kept = 0; stopped = 0; stopped_noise = 0; stopped_other = 0
for k in range(5):
    tp = np.array([i % 5 != k for i in range(len(pos))]); tn = np.array([i % 5 != k for i in range(len(neg))])
    m = train(pos[tp], neg[tn]); thr = auto_thr(m, pos[tp])
    kept += int((score(m, pos[~tp]) >= thr).sum())
    sn = score(m, neg[~tn]) < thr; stopped += int(sn.sum())
    stopped_noise += int(sn[nneg[~tn]==0].sum()); stopped_other += int(sn[nneg[~tn]==1].sum())
keep = kept/len(pos); stop = stopped/len(neg)
print(f"held out: owner kept {kept}/{len(pos)} ({keep*100:.0f}%), negatives stopped {stopped}/{len(neg)} ({stop*100:.0f}%): noise/TV {stopped_noise}/{len(noise)}, other voice {stopped_other}/{len(other)}")
m = train(pos, neg); thr = auto_thr(m, pos); mu, sd, w, b = m
ps = score(m, pos); ns = score(m, neg)
print(f"final: threshold {thr:.3f}; training set keeps {(ps>=thr).sum()}/{len(pos)}, stops {(ns<thr).sum()}/{len(neg)}")
r5 = lambda a: [round(float(v), 5) for v in a]
json.dump({
    "version": 3,
    "trained": f"2026-10-22 on the desktop (build_profile_v3.py) from {len(pos)} real clips (2026-09-08/09/22) vs {len(noise)} noise/TV + {len(other)} other-voice negatives",
    "threshold": round(thr, 4), "mean": r5(mu), "scale": r5(sd), "coef": r5(w), "intercept": round(float(b), 6),
    "owners": [owner],
    "heldout_keep": round(keep, 4), "heldout_stop": round(stop, 4),
    "note": "v3 mean+max pooled features (192). Enforced only for the user id in owners, elsewhere log-only. A noise/TV/other-voice filter, not a voice lock: held out it kept %d of %d of the owner's wakes and stopped %d of %d negatives." % (kept, len(pos), stopped, len(neg)),
}, open(out, "w"))
print("wrote", out)

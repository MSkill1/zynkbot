"""
Train the hey_zynk.onnx wake word classifier.

Pipeline:
  1. Load all positive WAV samples from positive_samples/wav/
  2. Download negative samples (openWakeWord's background clips) if needed
  3. Extract openWakeWord embeddings for all clips
  4. Train a small MLP classifier
  5. Export to ONNX as hey_zynk.onnx

Input ONNX spec (matches existing openWakeWord classifiers):
  name="x.1"  shape=[1, 16, 96]  dtype=float32
Output ONNX spec:
  name="output"  shape=[1, 1]  dtype=float32  (probability)

Run: python train_hey_zynk.py
"""

import os
import sys
import random
import numpy as np
import soundfile as sf
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import onnx
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
OWW_MODELS = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / \
             "site-packages" / "openwakeword" / "resources" / "models"

MEL_MODEL_PATH   = str(OWW_MODELS / "melspectrogram.onnx")
EMB_MODEL_PATH   = str(OWW_MODELS / "embedding_model.onnx")
POSITIVE_WAV_DIR = Path("positive_samples/wav")
NEG_CLIPS_DIR    = Path("negative_samples")
OUTPUT_ONNX      = Path("hey_zynk.onnx")

# ── pipeline constants (verified against models) ───────────────────────────
SAMPLE_RATE         = 16000
CHUNK_SAMPLES       = 1280          # 80ms per chunk
MEL_FRAMES_PER_CHUNK = 5            # mel frames per 1280-sample chunk
MEL_BINS            = 32
MEL_WINDOW          = 76            # embedding model input window
EMB_SIZE            = 96
EMB_WINDOW          = 16            # classifier input window
STEP_SIZE           = 8             # slide mel window every 8 frames


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a WAV file and resample/mono-ify to target SR."""
    data, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if file_sr != sr:
        # Simple nearest-neighbour resample (good enough for training)
        ratio = sr / file_sr
        new_len = int(len(data) * ratio)
        indices = (np.arange(new_len) / ratio).astype(int)
        data = data[np.clip(indices, 0, len(data) - 1)]
    return data


def extract_embeddings(audio: np.ndarray, mel_sess: ort.InferenceSession,
                       emb_sess: ort.InferenceSession,
                       pad_to: float = 4.0) -> np.ndarray:
    """
    Run the full mel → embedding pipeline on a clip.
    Returns array of shape [N, EMB_WINDOW, EMB_SIZE] — one row per detection window.
    Returns empty array if clip is too short even after padding.

    Minimum audio needed for one classifier window:
      76 (mel_window) + 15*8 (15 more steps) = 196 mel frames
      196 / 5 frames_per_chunk = ~40 chunks = ~3.2 seconds
    pad_to=4.0 gives ~22 embeddings → 7 classifier windows per clip.
    """
    audio = audio.astype(np.float32)

    min_samples = int(pad_to * SAMPLE_RATE)
    if len(audio) < min_samples:
        pad_total = min_samples - len(audio)
        pad_pre  = pad_total // 2
        pad_post = pad_total - pad_pre
        audio = np.concatenate([
            np.zeros(pad_pre, dtype=np.float32),
            audio,
            np.zeros(pad_post, dtype=np.float32),
        ])

    # Collect mel frames chunk by chunk
    mel_frames = []
    n_chunks = len(audio) // CHUNK_SAMPLES
    for i in range(n_chunks):
        chunk = audio[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
        mel_out = mel_sess.run(None, {"input": chunk.reshape(1, -1)})[0]
        # mel_out shape: [1, 1, MEL_FRAMES_PER_CHUNK, MEL_BINS]
        for f in range(mel_out.shape[2]):
            mel_frames.append(mel_out[0, 0, f, :])   # [MEL_BINS]

    if len(mel_frames) < MEL_WINDOW:
        return np.empty((0, EMB_WINDOW, EMB_SIZE), dtype=np.float32)

    # Slide window over mel frames to produce embeddings
    embeddings = []
    i = 0
    while i + MEL_WINDOW <= len(mel_frames):
        window = np.stack(mel_frames[i:i + MEL_WINDOW])  # [76, 32]
        inp = window.reshape(1, MEL_WINDOW, MEL_BINS, 1).astype(np.float32)
        emb = emb_sess.run(None, {"input_1": inp})[0]    # [1,1,1,96]
        embeddings.append(emb.reshape(EMB_SIZE))
        i += STEP_SIZE

    if len(embeddings) < EMB_WINDOW:
        return np.empty((0, EMB_WINDOW, EMB_SIZE), dtype=np.float32)

    # Slide window over embeddings to produce [N, 16, 96] training samples
    samples = []
    for j in range(len(embeddings) - EMB_WINDOW + 1):
        samples.append(np.stack(embeddings[j:j + EMB_WINDOW]))   # [16, 96]

    return np.stack(samples).astype(np.float32)   # [N, 16, 96]


def download_negative_samples():
    """Download a small set of negative clips from openWakeWord's GitHub releases."""
    import urllib.request
    import zipfile

    NEG_CLIPS_DIR.mkdir(exist_ok=True)
    if any(NEG_CLIPS_DIR.glob("*.wav")):
        print("Negative samples already present.")
        return

    # FSD50K background sounds subset used by openWakeWord (small download)
    url = ("https://github.com/dscripka/openWakeWord/releases/download/"
           "v0.1.1/wakeword_model_training_data.zip")
    zip_path = NEG_CLIPS_DIR / "neg_data.zip"
    print(f"Downloading negative samples from openWakeWord releases...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith(".wav"):
                    zf.extract(name, NEG_CLIPS_DIR)
        zip_path.unlink()
        print(f"Downloaded negative samples.")
    except Exception as e:
        print(f"Could not download negative samples: {e}")
        print("Generating synthetic negative samples instead (white noise + silence).")
        _generate_synthetic_negatives()


def _generate_synthetic_negatives(n: int = 200):
    """Fallback: generate silence + noise clips as negatives."""
    NEG_CLIPS_DIR.mkdir(exist_ok=True)
    for i in range(n // 2):
        # silence
        sf.write(str(NEG_CLIPS_DIR / f"silence_{i:04d}.wav"),
                 np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
        # white noise
        sf.write(str(NEG_CLIPS_DIR / f"noise_{i:04d}.wav"),
                 np.random.randn(SAMPLE_RATE * 2).astype(np.float32) * 0.05, SAMPLE_RATE)


class WakeWordMLP(nn.Module):
    """Small MLP matching the input/output spec of openWakeWord classifiers."""
    def __init__(self, window: int = EMB_WINDOW, emb: int = EMB_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * emb, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):        # x: [batch, 16, 96]
        return self.net(x)       # [batch, 1]


def main():
    print("=== Hey Zynk Wake Word Trainer ===\n")

    if not any(POSITIVE_WAV_DIR.glob("*.wav")):
        print(f"No WAV files in {POSITIVE_WAV_DIR}.")
        print("Run generate_samples.py first.")
        sys.exit(1)

    download_negative_samples()

    print("Loading ONNX feature extractors...")
    mel_sess = ort.InferenceSession(MEL_MODEL_PATH, providers=["CPUExecutionProvider"])
    emb_sess = ort.InferenceSession(EMB_MODEL_PATH, providers=["CPUExecutionProvider"])

    # ── extract positive features ───────────────────────────────────────────
    print("\nExtracting positive features...")
    pos_samples = []
    pos_files = list(POSITIVE_WAV_DIR.glob("*.wav"))
    for i, wav in enumerate(pos_files):
        audio = load_audio(str(wav))
        feats = extract_embeddings(audio, mel_sess, emb_sess)
        if feats.shape[0] > 0:
            pos_samples.append(feats)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pos_files)}")

    if not pos_samples:
        print("ERROR: no usable positive samples. Check WAV files are long enough.")
        sys.exit(1)

    X_pos = np.concatenate(pos_samples, axis=0)
    print(f"  Positive windows: {len(X_pos)}")

    # ── extract negative features ───────────────────────────────────────────
    print("\nExtracting negative features...")
    neg_samples = []
    neg_wavs = list(NEG_CLIPS_DIR.rglob("*.wav"))
    random.shuffle(neg_wavs)
    target_neg = max(len(X_pos) * 5, 500)   # ~5:1 negatives to positives

    for wav in neg_wavs:
        if sum(s.shape[0] for s in neg_samples) >= target_neg:
            break
        try:
            audio = load_audio(str(wav))
            feats = extract_embeddings(audio, mel_sess, emb_sess)
            if feats.shape[0] > 0:
                neg_samples.append(feats)
        except Exception:
            continue

    if not neg_samples:
        print("No negative samples found — generating synthetic ones.")
        _generate_synthetic_negatives()
        for wav in NEG_CLIPS_DIR.glob("*.wav"):
            audio = load_audio(str(wav))
            feats = extract_embeddings(audio, mel_sess, emb_sess)
            if feats.shape[0] > 0:
                neg_samples.append(feats)

    X_neg = np.concatenate(neg_samples, axis=0)
    # Subsample negatives to keep training balanced
    if len(X_neg) > target_neg:
        idx = np.random.choice(len(X_neg), int(target_neg), replace=False)
        X_neg = X_neg[idx]
    print(f"  Negative windows: {len(X_neg)}")

    # ── assemble dataset ────────────────────────────────────────────────────
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))], axis=0)

    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    split = int(0.85 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    print(f"\nDataset: {len(X_train)} train / {len(X_val)} val")
    print(f"Positive ratio: {y_train.mean():.2%}")

    # ── train ───────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on {device}...")

    model = WakeWordMLP().to(device)
    pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(60):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(Xv)
                val_loss = criterion(val_pred, yv).item()
                val_acc  = ((val_pred > 0.5) == (yv > 0.5)).float().mean().item()
            print(f"  epoch {epoch+1:3d} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # ── export to ONNX ──────────────────────────────────────────────────────
    print(f"\nExporting to {OUTPUT_ONNX} ...")
    model.eval().cpu()
    dummy = torch.zeros(1, EMB_WINDOW, EMB_SIZE, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(OUTPUT_ONNX),
        input_names=["x.1"],
        output_names=["output"],
        dynamic_axes={"x.1": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
    )

    # Verify
    check = ort.InferenceSession(str(OUTPUT_ONNX), providers=["CPUExecutionProvider"])
    test_out = check.run(None, {"x.1": np.zeros((1, EMB_WINDOW, EMB_SIZE), dtype=np.float32)})[0]
    print(f"Verified: output shape={test_out.shape}, value={test_out[0, 0]:.4f}")

    print(f"\n✓ hey_zynk.onnx saved.")
    print("Upload to: https://github.com/MSkill1/zynkbot/releases/tag/wake-word-models")


if __name__ == "__main__":
    main()

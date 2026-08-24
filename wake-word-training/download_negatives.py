"""
Download real-speech negative samples from LibriSpeech test-clean.
Converts up to MAX_CLIPS FLAC files to 16kHz mono WAV in negative_samples/librispeech/.

Run: python download_negatives.py
Size: ~346 MB download, extracts ~500 clips (~150 MB WAVs), then cleans up FLAC.
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

OUT_DIR = Path("negative_samples/librispeech")
SAMPLE_RATE = 16000
MAX_CLIPS = 500
URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
TAR_PATH = Path("negative_samples/test-clean.tar.gz")


def convert_flac(flac_path: Path, wav_path: Path):
    data, sr = sf.read(str(flac_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sr
        new_len = int(len(data) * ratio)
        indices = (np.arange(new_len) / ratio).astype(int)
        data = data[np.clip(indices, 0, len(data) - 1)]
    sf.write(str(wav_path), data, SAMPLE_RATE)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(OUT_DIR.glob("*.wav"))
    if len(existing) >= MAX_CLIPS:
        print(f"Already have {len(existing)} clips in {OUT_DIR}. Nothing to do.")
        return

    if not TAR_PATH.exists():
        print(f"Downloading LibriSpeech test-clean (~346 MB)...")
        import subprocess
        # Use -s (silent) + stderr=DEVNULL to avoid pipe-buffer deadlock with progress output
        subprocess.run(
            ["curl", "-L", "--retry", "3", "-s", "-o", str(TAR_PATH), URL],
            check=True, stderr=subprocess.DEVNULL,
        )
        print("Download complete.")
    else:
        print(f"Tarball already present at {TAR_PATH}.")

    print(f"Extracting and converting up to {MAX_CLIPS} clips...")
    converted = len(existing)
    already_done = {f.stem for f in existing}

    with tarfile.open(TAR_PATH, "r:gz") as tar:
        for member in tar:
            if converted >= MAX_CLIPS:
                break
            if not member.name.endswith(".flac"):
                continue
            stem = Path(member.name).stem
            if stem in already_done:
                converted += 1
                continue
            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                raw = f.read()
                tmp = OUT_DIR / (stem + ".flac")
                tmp.write_bytes(raw)
                wav_path = OUT_DIR / (stem + ".wav")
                convert_flac(tmp, wav_path)
                tmp.unlink()
                converted += 1
                if converted % 50 == 0:
                    print(f"  {converted}/{MAX_CLIPS} clips converted")
            except Exception as e:
                print(f"  skip {member.name}: {e}")

    TAR_PATH.unlink(missing_ok=True)
    print(f"\nDone. {converted} WAV clips in {OUT_DIR}/")
    print("Re-run train_hey_zynk.py to retrain with real speech negatives.")


if __name__ == "__main__":
    main()

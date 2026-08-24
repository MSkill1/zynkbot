"""
Generate "Hey Zynk" positive training samples using edge-tts.
Produces 16kHz mono WAV files in positive_samples/wav/.
Run: python generate_samples.py
"""

import asyncio
import os
import subprocess
import sys

import edge_tts

PHRASE = "Hey Zynk"

# 24 English voices covering different accents, genders, ages
VOICES = [
    "en-US-GuyNeural",       "en-US-JennyNeural",
    "en-US-AriaNeural",      "en-US-DavisNeural",
    "en-US-TonyNeural",      "en-US-NancyNeural",
    "en-US-AmberNeural",     "en-US-AnaNeural",
    "en-US-ChristopherNeural","en-US-EricNeural",
    "en-GB-RyanNeural",      "en-GB-SoniaNeural",
    "en-GB-MaisieNeural",    "en-GB-ThomasNeural",
    "en-AU-NatashaNeural",   "en-AU-WilliamNeural",
    "en-CA-LiamNeural",      "en-CA-ClaraNeural",
    "en-IN-NeerjaNeural",    "en-IN-PrabhatNeural",
    "en-IE-ConnorNeural",    "en-IE-EmilyNeural",
    "en-NZ-MitchellNeural",  "en-ZA-LeahNeural",
]

# Speech rate variants per voice: normal, slightly slow, slightly fast
RATES = ["+0%", "-10%", "+10%"]

OUT_MP3 = "positive_samples/mp3"
OUT_WAV = "positive_samples/wav"


async def generate_one(voice: str, rate: str, index: int) -> str:
    name = f"{index:04d}_{voice}_{rate.replace('%','').replace('+','p').replace('-','m')}"
    mp3_path = os.path.join(OUT_MP3, name + ".mp3")
    wav_path = os.path.join(OUT_WAV, name + ".wav")

    if os.path.exists(wav_path):
        return wav_path

    try:
        tts = edge_tts.Communicate(text=PHRASE, voice=voice, rate=rate)
        await tts.save(mp3_path)
        # Convert to 16kHz mono WAV using ffmpeg
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, check=True
        )
        os.remove(mp3_path)
        return wav_path
    except Exception as e:
        print(f"  SKIP {voice} {rate}: {e}")
        return ""


async def main():
    os.makedirs(OUT_MP3, exist_ok=True)
    os.makedirs(OUT_WAV, exist_ok=True)

    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg")
        sys.exit(1)

    tasks = []
    idx = 0
    for voice in VOICES:
        for rate in RATES:
            tasks.append(generate_one(voice, rate, idx))
            idx += 1

    print(f"Generating {len(tasks)} samples for \"{PHRASE}\"...")
    results = await asyncio.gather(*tasks)
    generated = [r for r in results if r]
    print(f"Done: {len(generated)}/{len(tasks)} samples in {OUT_WAV}/")


if __name__ == "__main__":
    asyncio.run(main())

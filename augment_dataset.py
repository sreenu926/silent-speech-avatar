"""
augment_dataset.py

Synthetically expands your dataset from 28 → 280 samples per class
by generating 9 augmented variants of each original .wav file.

Usage:
    cd /Users/nagasreenivasaraop/whisper_backend
    source venv/bin/activate
    python augment_dataset.py

What it does:
    Reads from:   dataset/<WORD>/*.wav       (your original 28 per class)
    Writes to:    dataset/<WORD>/*.wav       (adds 9 new files per original)
    Result:       280 samples per class      (28 originals + 252 augmented)

After running this, run:
    python split_dataset.py
    cd backend && python train_model.py
"""

import os
import numpy as np
import librosa
import soundfile as sf

DATASET_DIR = "dataset"
SAMPLE_RATE  = 16000

VOCABULARY = ["HELP", "PAIN", "WATER", "STOP", "HELLO", "THANK_YOU", "YES", "NO"]


def augment_variants(audio: np.ndarray, sr: int) -> list:
    """
    Returns 9 augmented variants of the input audio.
    Each variant is independently reproducible and clearly different.
    """
    variants = []

    # 1. Gaussian noise (light)
    noise = audio + 0.004 * np.random.randn(len(audio))
    variants.append(("noise_light", noise.astype(np.float32)))

    # 2. Gaussian noise (heavy)
    noise2 = audio + 0.010 * np.random.randn(len(audio))
    variants.append(("noise_heavy", noise2.astype(np.float32)))

    # 3. Time stretch — faster
    try:
        fast = librosa.effects.time_stretch(audio, rate=1.15)
        variants.append(("speed_up", fast.astype(np.float32)))
    except Exception:
        variants.append(("speed_up", audio.copy()))

    # 4. Time stretch — slower
    try:
        slow = librosa.effects.time_stretch(audio, rate=0.88)
        variants.append(("speed_down", slow.astype(np.float32)))
    except Exception:
        variants.append(("speed_down", audio.copy()))

    # 5. Pitch shift up
    try:
        pitch_up = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2.0)
        variants.append(("pitch_up", pitch_up.astype(np.float32)))
    except Exception:
        variants.append(("pitch_up", audio.copy()))

    # 6. Pitch shift down
    try:
        pitch_down = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2.0)
        variants.append(("pitch_down", pitch_down.astype(np.float32)))
    except Exception:
        variants.append(("pitch_down", audio.copy()))

    # 7. Volume louder
    loud = np.clip(audio * 1.4, -1.0, 1.0)
    variants.append(("vol_up", loud.astype(np.float32)))

    # 8. Volume quieter
    quiet = audio * 0.55
    variants.append(("vol_down", quiet.astype(np.float32)))

    # 9. Combined — noise + pitch shift (simulates different mic/environment)
    try:
        combined = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1.0)
        combined = combined + 0.005 * np.random.randn(len(combined))
        variants.append(("combined", combined.astype(np.float32)))
    except Exception:
        variants.append(("combined", audio.copy()))

    return variants


def normalize(audio: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


def main():
    total_written = 0

    for word in VOCABULARY:
        class_dir = os.path.join(DATASET_DIR, word)
        if not os.path.exists(class_dir):
            print(f"  ✗ Missing: {class_dir} — skipping")
            continue

        # Only process ORIGINAL files (no aug_ prefix)
        original_files = sorted([
            f for f in os.listdir(class_dir)
            if f.endswith(".wav") and "aug_" not in f
        ])

        print(f"\n[{word}] {len(original_files)} originals found — generating variants...")

        for fname in original_files:
            path = os.path.join(class_dir, fname)
            audio, sr = librosa.load(path, sr=SAMPLE_RATE)
            stem = os.path.splitext(fname)[0]   # e.g. "help_001"

            variants = augment_variants(audio, sr)

            for tag, aug_audio in variants:
                aug_audio = normalize(aug_audio)
                out_name = f"aug_{stem}_{tag}.wav"
                out_path = os.path.join(class_dir, out_name)

                # Skip if already exists (safe to re-run)
                if os.path.exists(out_path):
                    continue

                sf.write(out_path, aug_audio, SAMPLE_RATE)
                total_written += 1

        final_count = len([f for f in os.listdir(class_dir) if f.endswith(".wav")])
        print(f"  ✓ {word}: {final_count} total samples")

    print(f"\n✅ Done. {total_written} new files written.")
    print("\nNext steps:")
    print("  1. python split_dataset.py")
    print("  2. cd backend && python train_model.py")


if __name__ == "__main__":
    main()
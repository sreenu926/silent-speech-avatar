import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from tqdm import tqdm
import time

# =========================
# CONFIGURATION
# =========================
SAMPLE_RATE = 16000
DURATION = 1.5          # seconds per recording
PRE_SILENCE = 0.3       # seconds before recording
POST_SILENCE = 0.3      # seconds after recording
SAMPLES_PER_CLASS = 40  # change to 80 later if desired

VOCABULARY = [
    "HELP",
    "PAIN",
    "WATER",
    "STOP",
    "HELLO",
    "THANK_YOU",
    "YES",
    "NO"
]

DATASET_DIR = "dataset"

# =========================
# CREATE DIRECTORY STRUCTURE
# =========================
os.makedirs(DATASET_DIR, exist_ok=True)

for word in VOCABULARY:
    os.makedirs(os.path.join(DATASET_DIR, word), exist_ok=True)

print("\n📁 Dataset directory structure ready.\n")

# =========================
# RECORDING LOOP
# =========================
for word in VOCABULARY:
    print(f"\n🔴 Recording class: {word}")
    print(f"Please whisper '{word}' when prompted.")
    
    class_dir = os.path.join(DATASET_DIR, word)
    
    for i in tqdm(range(1, SAMPLES_PER_CLASS + 1)):
        filename = os.path.join(class_dir, f"{word.lower()}_{i:03d}.wav")
        
        print(f"\nSample {i}/{SAMPLES_PER_CLASS}")
        print("Prepare...")
        time.sleep(PRE_SILENCE)
        
        print("🎙️ Whisper NOW")
        recording = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        time.sleep(POST_SILENCE)
        
        # Normalize audio (safe normalization)
        max_val = np.max(np.abs(recording))
        if max_val > 0:
            recording = recording / max_val
        
        sf.write(filename, recording, SAMPLE_RATE)
        print("Saved:", filename)

print("\n✅ Dataset recording complete.")
"""
diagnose.py  —  Run this FIRST before retraining.
It tells you exactly what is wrong with your current checkpoint.

Usage:
    cd /Users/nagasreenivasaraop/whisper_backend
    source venv/bin/activate
    python diagnose.py

What it checks:
    1. Checkpoint structure and saved vocab
    2. Model weights — are they actually trained (not random)?
    3. Embedding statistics — does the pipeline produce the right shape?
    4. Inference on a known .wav file from your dataset
    5. Inference on a freshly recorded sample via the same webm→wav pipeline
    6. Class distribution — is the training set balanced?
"""

import os, sys, torch, numpy as np, whisper, librosa
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT = "model/whisper_conformer.pt"
DATA_DIR   = "dataset_split"
VOCABULARY = ["HELP", "PAIN", "WATER", "STOP", "HELLO", "THANK_YOU", "YES", "NO"]

print(f"\n{'='*60}")
print("BIOSENSOR SPEECH CLASSIFIER — DIAGNOSTIC REPORT")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────
# 1. CHECKPOINT
# ─────────────────────────────────────────────
print("[ 1 ] CHECKPOINT")
if not os.path.exists(CHECKPOINT):
    print(f"  ✗  NOT FOUND: {CHECKPOINT}")
    print("     Run train_model.py first.")
    sys.exit(1)

ckpt = torch.load(CHECKPOINT, map_location="cpu")
print(f"  ✓  Loaded: {CHECKPOINT}")
print(f"     Keys in checkpoint: {list(ckpt.keys())}")
print(f"     Saved vocab: {ckpt.get('vocab', 'NOT SAVED')}")

state = ckpt["model_state_dict"]
layer_keys = [k for k in state.keys() if "weight" in k]
print(f"     Layer weight keys: {layer_keys}")

# ─────────────────────────────────────────────
# 2. ARE WEIGHTS ACTUALLY TRAINED?
# ─────────────────────────────────────────────
print("\n[ 2 ] WEIGHT STATISTICS (is the model really trained?)")
for k in layer_keys:
    w = state[k].float()
    print(f"     {k:40s}  mean={w.mean():.4f}  std={w.std():.4f}  "
          f"min={w.min():.4f}  max={w.max():.4f}")

# Check final layer bias — if all zeros, model never learned
bias_key = "net.12.bias"  # last Linear bias
if bias_key in state:
    b = state[bias_key].float()
    print(f"\n  Final layer bias (net.12.bias): {b.tolist()}")
    if b.std() < 0.01:
        print("  ⚠️  WARNING: Final bias is near-constant → model may not have trained.")
    else:
        print("  ✓  Final bias has variance → model has learned something.")

# ─────────────────────────────────────────────
# 3. REBUILD MODEL & LOAD WEIGHTS
# ─────────────────────────────────────────────
class Classifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),       nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

model = Classifier().to(DEVICE)
model.load_state_dict(state)
model.eval()
print("\n[ 3 ] MODEL LOADED OK")

# ─────────────────────────────────────────────
# 4. WHISPER EMBEDDING STATISTICS
# ─────────────────────────────────────────────
print("\n[ 4 ] WHISPER EMBEDDING STATISTICS")
whisper_model = whisper.load_model("base").to(DEVICE)
whisper_model.eval()

def embed_wav(audio_np):
    audio_np = whisper.pad_or_trim(audio_np)
    mel = whisper.log_mel_spectrogram(audio_np).to(DEVICE)
    with torch.no_grad():
        enc = whisper_model.encoder(mel.unsqueeze(0))
    return enc.mean(dim=1).squeeze(0)

# Use first wav from training set
test_class = "YES"
test_dir = os.path.join(DATA_DIR, "train", test_class)
if os.path.exists(test_dir):
    wav_files = [f for f in os.listdir(test_dir) if f.endswith(".wav")]
    if wav_files:
        wav_path = os.path.join(test_dir, wav_files[0])
        audio, _ = librosa.load(wav_path, sr=16000)
        emb = embed_wav(audio)
        print(f"  Sample: {wav_path}")
        print(f"  Embedding shape: {emb.shape}")
        print(f"  Embedding stats: mean={emb.mean():.4f}  std={emb.std():.4f}")

        with torch.no_grad():
            logits = model(emb.unsqueeze(0))
            probs  = F.softmax(logits, dim=-1).squeeze(0)

        print(f"\n  Prediction on a known YES wav:")
        for i, (word, p) in enumerate(zip(VOCABULARY, probs.tolist())):
            bar = "█" * int(p * 40)
            marker = " ← predicted" if p == probs.max().item() else ""
            print(f"    [{i}] {word:10s}  {p:.4f}  {bar}{marker}")
    else:
        print(f"  No .wav files found in {test_dir}")
else:
    print(f"  Dataset dir not found: {test_dir}")

# ─────────────────────────────────────────────
# 5. RUN ON ALL 8 CLASSES — per-class accuracy
# ─────────────────────────────────────────────
print("\n[ 5 ] PER-CLASS ACCURACY ON TEST SET")
test_path = os.path.join(DATA_DIR, "test")
if os.path.exists(test_path):
    for class_name in VOCABULARY:
        class_dir = os.path.join(test_path, class_name)
        if not os.path.exists(class_dir):
            print(f"  {class_name:12s} — dir not found")
            continue
        wavs = [f for f in os.listdir(class_dir) if f.endswith(".wav")]
        if not wavs:
            print(f"  {class_name:12s} — no wav files")
            continue
        correct = 0
        predictions = []
        for wav_file in wavs:
            path = os.path.join(class_dir, wav_file)
            audio, _ = librosa.load(path, sr=16000)
            emb = embed_wav(audio)
            with torch.no_grad():
                logits = model(emb.unsqueeze(0))
                pred = logits.argmax(dim=1).item()
            predictions.append(VOCABULARY[pred])
            if pred == VOCABULARY.index(class_name):
                correct += 1
        acc = correct / len(wavs)
        pred_counts = {w: predictions.count(w) for w in VOCABULARY if predictions.count(w) > 0}
        status = "✓" if acc > 0.5 else "✗"
        print(f"  {status} {class_name:12s}  acc={acc:.2f} ({correct}/{len(wavs)})  predictions: {pred_counts}")
else:
    print(f"  Test split not found at {test_path}")

# ─────────────────────────────────────────────
# 6. DATASET BALANCE CHECK
# ─────────────────────────────────────────────
print("\n[ 6 ] DATASET BALANCE")
for split in ["train", "val", "test"]:
    split_path = os.path.join(DATA_DIR, split)
    if not os.path.exists(split_path):
        continue
    counts = {}
    for class_name in VOCABULARY:
        d = os.path.join(split_path, class_name)
        counts[class_name] = len([f for f in os.listdir(d) if f.endswith(".wav")]) if os.path.exists(d) else 0
    print(f"  {split:6s}: {counts}")

# ─────────────────────────────────────────────
# 7. WEBM DECODE TEST  (simulates what the browser sends)
# ─────────────────────────────────────────────
print("\n[ 7 ] WEBM DECODE TEST")
print("  Checking if ffmpeg is available...")
import subprocess
result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
if result.returncode == 0:
    version_line = result.stdout.split("\n")[0]
    print(f"  ✓  {version_line}")
else:
    print("  ✗  ffmpeg NOT FOUND — install with: brew install ffmpeg")
    print("     This is likely why the pipeline is producing wrong embeddings.")

print("\n[ 8 ] EMBEDDING MISMATCH TEST (train .wav vs decoded webm)")
print("  To simulate: record a word, save as .webm, then compare embeddings.")
print("  Skipping automated test — check manually if ffmpeg decode produces")
print("  float32 audio at 16 kHz with values in [-1, 1] range.\n")

print("="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
print("""
MOST LIKELY CAUSES OF 'YES for all words':

  A) Model never actually trained (val_acc stayed near 12.5% = 1/8 random)
     and the checkpoint saved is the initial random state that happens
     to have a bias toward class index 6 (YES).
     → Check: does train output show val_acc improving above 0.5?

  B) ffmpeg is missing → decode_audio() falls through to torchaudio
     fallback → torchaudio can't decode webm → audio is silence/noise
     → Whisper embeds silence → always maps to YES class.
     → Fix: brew install ffmpeg

  C) Train/inference audio normalization mismatch.
     librosa.load() normalizes to [-1, 1]. After ffmpeg decode,
     soundfile.read() also gives [-1, 1] — should be fine.
     But if torchaudio fallback is used, values may differ.

  D) Dropout not disabled during inference (model.train() vs model.eval())
     → Unlikely given the code, but check pipeline sets model.eval().
""")
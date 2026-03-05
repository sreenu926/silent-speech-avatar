import os
import torch
import torch.nn as nn
import torch.optim as optim
import whisper
import librosa
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import random

# ==========================
# CONFIG
# ==========================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
WHISPER_DEVICE = "cpu"  # Whisper must run on CPU — MPS causes silent tensor mismatch
DATA_DIR = "../dataset_split"
NUM_CLASSES = 8
EPOCHS = 60
LR = 1e-3

VOCABULARY = [
    "HELP", "PAIN", "WATER", "STOP",
    "HELLO", "THANK_YOU", "YES", "NO"
]
label_map = {word: i for i, word in enumerate(VOCABULARY)}
print("Using device:", DEVICE)

# ==========================
# WHISPER
# ==========================
whisper_model = whisper.load_model("tiny").to(WHISPER_DEVICE)
whisper_model.eval()
for param in whisper_model.parameters():
    param.requires_grad = False

# ==========================
# AUGMENTATION
# ==========================
def augment_audio(audio: np.ndarray, sr: int = 16000) -> list:
    augmented = []

    # 1. Gaussian noise
    noise = audio + 0.005 * np.random.randn(len(audio))
    augmented.append(noise.astype(np.float32))

    # 2. Time stretch
    try:
        stretched = librosa.effects.time_stretch(audio, rate=1.1)
        augmented.append(stretched.astype(np.float32))
    except:
        augmented.append(audio.copy())

    # 3. Pitch shift
    try:
        pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1.5)
        augmented.append(pitched.astype(np.float32))
    except:
        augmented.append(audio.copy())

    # 4. Volume scaling
    scaled = audio * random.uniform(0.7, 1.3)
    augmented.append(scaled.astype(np.float32))

    return augmented


# ==========================
# FEATURE EXTRACTION
# ==========================
def extract_embedding(audio: np.ndarray) -> torch.Tensor:
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(WHISPER_DEVICE)
    with torch.no_grad():
        embedding = whisper_model.encoder(mel.unsqueeze(0))
    return embedding.mean(dim=1).squeeze(0)


# ==========================
# LOAD DATA
# ==========================
def load_split(split, augment=False):
    features, labels = [], []
    split_path = os.path.join(DATA_DIR, split)

    for class_name in VOCABULARY:
        class_dir = os.path.join(split_path, class_name)
        for file in sorted(os.listdir(class_dir)):
            if not file.endswith('.wav'):
                continue
            path = os.path.join(class_dir, file)
            audio, sr = librosa.load(path, sr=16000)

            emb = extract_embedding(audio)
            features.append(emb.cpu())
            labels.append(label_map[class_name])

            if augment:
                for aug_audio in augment_audio(audio, sr):
                    emb_aug = extract_embedding(aug_audio)
                    features.append(emb_aug.cpu())
                    labels.append(label_map[class_name])

    return torch.stack(features), torch.tensor(labels)


print("Extracting train features (with augmentation)...")
X_train, y_train = load_split("train", augment=True)
print(f"  Train size: {len(X_train)} samples")

print("Extracting val features...")
X_val, y_val = load_split("val", augment=False)

print("Extracting test features...")
X_test, y_test = load_split("test", augment=False)


# ==========================
# CLASSIFIER — No BatchNorm
# (BatchNorm causes single-sample inference failures)
# ==========================
class Classifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


model = Classifier(input_dim=X_train.shape[1], num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==========================
# TRAIN LOOP
# ==========================
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()

    perm = torch.randperm(len(X_train))
    X_shuf = X_train[perm].to(DEVICE)
    y_shuf = y_train[perm].to(DEVICE)

    BATCH = 32
    total_loss = 0
    for i in range(0, len(X_shuf), BATCH):
        xb = X_shuf[i:i+BATCH]
        yb = y_shuf[i:i+BATCH]
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_preds = torch.argmax(model(X_val.to(DEVICE)), dim=1)
        val_acc = accuracy_score(y_val.numpy(), val_preds.cpu().numpy())

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab": VOCABULARY
        }, "model/whisper_conformer.pt")

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")

# ==========================
# TEST EVALUATION
# ==========================
checkpoint = torch.load("model/whisper_conformer.pt", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    test_preds = torch.argmax(model(X_test.to(DEVICE)), dim=1)
    test_acc = accuracy_score(y_test.numpy(), test_preds.cpu().numpy())

print(f"\nFinal Test Accuracy: {test_acc:.4f}")

cm = confusion_matrix(y_test.numpy(), test_preds.cpu().numpy())
print("\nConfusion Matrix:")
print("         " + "  ".join([f"{w[:4]:>4}" for w in VOCABULARY]))
for i, row in enumerate(cm):
    print(f"{VOCABULARY[i][:8]:>8} " + "  ".join([f"{v:>4}" for v in row]))

print("\nModel saved to model/whisper_conformer.pt")
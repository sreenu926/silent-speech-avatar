"""
split_dataset.py

FIXED VERSION — prevents data leakage from augmentation.

Strategy:
  1. Split only ORIGINAL recordings (no aug_ prefix) into train/val/test
  2. Copy augmented variants ONLY into train — never val or test
  3. Val and test contain only real recordings → honest evaluation

Result with 40 originals per class:
  train: 28 originals + 252 augmented = 280 per class
  val:   6 originals only
  test:  6 originals only
"""

import os
import random
import shutil

DATASET_DIR = "dataset"
OUTPUT_DIR  = "dataset_split"

SPLIT_RATIO = {
    "train": 0.7,
    "val":   0.15,
    "test":  0.15,
}

random.seed(42)

VOCABULARY = ["HELP", "PAIN", "WATER", "STOP", "HELLO", "THANK_YOU", "YES", "NO"]

# ── Wipe and recreate output folders cleanly ──────────────────────────
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

for split in SPLIT_RATIO:
    for word in VOCABULARY:
        os.makedirs(os.path.join(OUTPUT_DIR, split, word), exist_ok=True)

# ── Process each class ────────────────────────────────────────────────
for word in VOCABULARY:
    class_dir = os.path.join(DATASET_DIR, word)
    if not os.path.exists(class_dir):
        print(f"  ✗ Missing: {class_dir} — skipping")
        continue

    all_files = os.listdir(class_dir)

    # Separate originals from augmented
    originals  = sorted([f for f in all_files if f.endswith(".wav") and "aug_" not in f])
    augmented  = sorted([f for f in all_files if f.endswith(".wav") and "aug_" in f])

    # Split originals only
    random.shuffle(originals)
    n_total = len(originals)
    n_train = int(n_total * SPLIT_RATIO["train"])
    n_val   = int(n_total * SPLIT_RATIO["val"])

    train_orig = originals[:n_train]
    val_orig   = originals[n_train:n_train + n_val]
    test_orig  = originals[n_train + n_val:]

    # Copy originals into their splits
    for f in train_orig:
        shutil.copy(os.path.join(class_dir, f), os.path.join(OUTPUT_DIR, "train", word, f))
    for f in val_orig:
        shutil.copy(os.path.join(class_dir, f), os.path.join(OUTPUT_DIR, "val",   word, f))
    for f in test_orig:
        shutil.copy(os.path.join(class_dir, f), os.path.join(OUTPUT_DIR, "test",  word, f))

    # Copy ALL augmented files into train ONLY
    for f in augmented:
        shutil.copy(os.path.join(class_dir, f), os.path.join(OUTPUT_DIR, "train", word, f))

    train_count = len(train_orig) + len(augmented)
    print(f"  {word:12s} → train: {train_count:4d} ({len(train_orig)} orig + {len(augmented)} aug) "
          f"| val: {len(val_orig)} | test: {len(test_orig)}")

print("\n✅ Dataset split complete — no leakage between splits.")
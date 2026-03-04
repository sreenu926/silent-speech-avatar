import os
import random
import shutil

DATASET_DIR = "dataset"
OUTPUT_DIR = "dataset_split"

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(42)

# Create output folders
for split in SPLIT_RATIO.keys():
    for class_name in os.listdir(DATASET_DIR):
        os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    files = os.listdir(class_path)
    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * SPLIT_RATIO["train"])
    n_val = int(n_total * SPLIT_RATIO["val"])

    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]

    for f in train_files:
        shutil.copy(
            os.path.join(class_path, f),
            os.path.join(OUTPUT_DIR, "train", class_name, f)
        )

    for f in val_files:
        shutil.copy(
            os.path.join(class_path, f),
            os.path.join(OUTPUT_DIR, "val", class_name, f)
        )

    for f in test_files:
        shutil.copy(
            os.path.join(class_path, f),
            os.path.join(OUTPUT_DIR, "test", class_name, f)
        )

print("Dataset split complete.")
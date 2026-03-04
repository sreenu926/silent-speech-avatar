"""
model/load_model.py

Loads the trained MLP Classifier checkpoint saved by train_model.py.

Checkpoint format (saved by train_model.py):
    {
        "model_state_dict": OrderedDict(...),
        "vocab": ["HELP", "PAIN", "WATER", "STOP", "HELLO", "THANK_YOU", "YES", "NO"]
    }

The Classifier architecture must match exactly what was used during training.
"""

import os
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Default checkpoint path — override with CHECKPOINT_PATH env var
DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "whisper_conformer.pt"
)

VOCABULARY = ["HELP", "PAIN", "WATER", "STOP", "HELLO", "THANK_YOU", "YES", "NO"]
NUM_CLASSES = len(VOCABULARY)


# ─────────────────────────────────────────────
# Classifier definition
# Must match train_model.py EXACTLY (same layer sizes, same dropout values).
# ─────────────────────────────────────────────
class Classifier(nn.Module):
    def __init__(self, input_dim: int = 512, num_classes: int = NUM_CLASSES):
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
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(checkpoint_path: str | None = None) -> Classifier:
    """
    Load the trained Classifier from a checkpoint.

    If no checkpoint exists yet (e.g. you haven't trained yet), returns a
    randomly-initialised model so the server still starts — the frontend
    will still work, predictions will just be random until you train.
    """
    path = checkpoint_path or os.environ.get("CHECKPOINT_PATH", DEFAULT_CHECKPOINT)

    model = Classifier(input_dim=512, num_classes=NUM_CLASSES).to(DEVICE)

    if not os.path.exists(path):
        logger.warning(
            f"Checkpoint not found at '{path}'. "
            "Server will start with random weights. "
            "Run train_model.py first, then restart the server."
        )
        model.eval()
        return model

    try:
        ckpt = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        vocab = ckpt.get("vocab", VOCABULARY)
        logger.info(
            f"Checkpoint loaded from '{path}' | vocab: {vocab}"
        )
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        logger.warning("Falling back to random weights.")

    model.eval()
    return model
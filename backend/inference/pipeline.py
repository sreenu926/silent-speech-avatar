"""
inference/pipeline.py

Receives raw audio bytes from the browser (audio/webm;codecs=opus),
decodes to 16 kHz float32 PCM using ffmpeg piped directly to stdout as
raw PCM — no intermediate file, no soundfile dependency.

Returns JSON matching what the frontend expects.
"""

import time
import logging
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import whisper

logger = logging.getLogger(__name__)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
WHISPER_DEVICE = "cpu"  # Whisper must run on CPU — MPS causes silent tensor mismatch
VOCABULARY = ["HELP", "PAIN", "WATER", "STOP", "HELLO", "THANK_YOU", "YES", "NO"]


# ─────────────────────────────────────────────
# Audio decode — ffmpeg webm/opus → float32 PCM
# Pipes raw bytes IN, gets raw f32le PCM OUT.
# No temp files, no soundfile, no torchaudio needed.
# ─────────────────────────────────────────────
def decode_audio(raw_bytes: bytes) -> np.ndarray:
    """
    Decode any browser audio format → 16 kHz mono float32 numpy array.
    Uses ffmpeg stdin→stdout pipe: most reliable cross-format approach.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",          # read from stdin
        "-ar", "16000",          # resample to 16 kHz
        "-ac", "1",              # mono
        "-f", "f32le",           # raw float32 little-endian PCM
        "pipe:1",                # write to stdout
    ]
    result = subprocess.run(
        cmd,
        input=raw_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg decode failed: {err}")

    # Convert raw bytes → float32 numpy
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()

    if len(audio) == 0:
        raise RuntimeError("ffmpeg produced empty audio — check input bytes.")

    # Log audio stats so we can verify it is not silence
    logger.info(
        f"Decoded audio: {len(audio)} samples ({len(audio)/16000:.2f}s) | "
        f"mean={audio.mean():.4f}  std={audio.std():.4f}  "
        f"max_abs={np.abs(audio).max():.4f}"
    )

    if np.abs(audio).max() < 0.001:
        logger.warning(
            "Audio is near-silence (max < 0.001). "
            "Check microphone. Prediction will be unreliable."
        )

    return audio


# ─────────────────────────────────────────────
# Inference Pipeline
# ─────────────────────────────────────────────
class InferencePipeline:
    def __init__(self, model: torch.nn.Module):
        self.model  = model
        self.device = DEVICE

        logger.info("Loading Whisper tiny encoder on CPU...")
        self.whisper_model = whisper.load_model("tiny").to(WHISPER_DEVICE)
        self.whisper_model.eval()
        for p in self.whisper_model.parameters():
            p.requires_grad = False
        logger.info(f"Whisper ready on {WHISPER_DEVICE}. Classifier on {self.device}.")

    @torch.no_grad()
    def _extract_embedding(self, audio: np.ndarray) -> torch.Tensor:
        """float32 numpy → Whisper tiny encoder embedding (384-dim) on CPU."""
        audio = whisper.pad_or_trim(audio)
        mel   = whisper.log_mel_spectrogram(audio).to(WHISPER_DEVICE)
        enc   = self.whisper_model.encoder(mel.unsqueeze(0))
        return enc.mean(dim=1).squeeze(0).to(self.device)

    def run(self, raw_bytes: bytes) -> dict:
        t_received = time.perf_counter() * 1000

        # 1. Decode browser audio → float32 PCM
        audio = decode_audio(raw_bytes)

        # 2. Whisper embedding
        t_inf_start = time.perf_counter() * 1000
        embedding   = self._extract_embedding(audio)

        logger.info(
            f"Embedding: mean={embedding.mean():.4f}  std={embedding.std():.4f}"
        )

        # 3. Classify
        logits  = self.model(embedding.unsqueeze(0))
        probs   = F.softmax(logits, dim=-1).squeeze(0)
        pred_idx    = probs.argmax().item()
        confidence  = probs[pred_idx].item()
        t_inf_end   = time.perf_counter() * 1000

        class_label = VOCABULARY[pred_idx]
        t_response  = time.perf_counter() * 1000

        # Log top-3 for server-side debugging
        top3 = sorted(enumerate(probs.tolist()), key=lambda x: -x[1])[:3]
        top3_str = "  ".join([f"{VOCABULARY[i]}={p:.3f}" for i, p in top3])
        logger.info(f"Result: {class_label} ({confidence:.3f}) | Top3: {top3_str}")

        return {
            "class_label":   class_label,
            "confidence":    round(confidence, 6),
            "probabilities": [round(float(p), 6) for p in probs.tolist()],
            "latency_ms":    round(t_response - t_received, 2),
            "timestamps": {
                "received":        round(t_received,  2),
                "inference_start": round(t_inf_start, 2),
                "inference_end":   round(t_inf_end,   2),
                "response_sent":   round(t_response,  2),
            },
        }
"""
model/conformer.py
Conformer classifier head on top of Whisper Encoder features.
Input:  [B, T, D] — Whisper encoder output
Output: [B, N]   — Class logits over N vocabulary words
"""

import torch
import torch.nn as nn


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size=31, padding=15, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_conv = self.conv_norm(x).transpose(1, 2)
        x = x + self.conv(x_conv).transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.norm_out(x)


class WhisperConformerClassifier(nn.Module):
    """
    Whisper Encoder (frozen) + Conformer classifier head.
    whisper_encoder: pre-loaded openai/whisper encoder module
    num_classes:     vocabulary size (e.g., 20 emergency/daily words)
    conformer_dim:   projection dimension for Conformer
    num_blocks:      number of Conformer blocks
    """

    def __init__(
        self,
        whisper_encoder: nn.Module,
        num_classes: int,
        conformer_dim: int = 256,
        num_blocks: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.whisper_encoder = whisper_encoder
        # Freeze Whisper encoder weights
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False

        whisper_dim = self.whisper_encoder.config.d_model  # typically 512 or 768
        self.proj = nn.Linear(whisper_dim, conformer_dim)
        self.conformer_blocks = nn.Sequential(
            *[ConformerBlock(conformer_dim, num_heads, dropout=dropout) for _ in range(num_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(conformer_dim, num_classes)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        input_features: [B, 80, 3000] mel spectrogram from Whisper processor
        returns logits:  [B, num_classes]
        """
        with torch.no_grad():
            enc_out = self.whisper_encoder(input_features).last_hidden_state  # [B, T, D]
        x = self.proj(enc_out)             # [B, T, conformer_dim]
        x = self.conformer_blocks(x)       # [B, T, conformer_dim]
        x = self.pool(x.transpose(1, 2))   # [B, conformer_dim, 1]
        x = x.squeeze(-1)                  # [B, conformer_dim]
        return self.classifier(x)          # [B, num_classes]
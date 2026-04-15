from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class Q0MonotonicLoopEosController(nn.Module):
    """
    Lightweight q0-only controller operating on the codebook-specific g0 representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        progress_num_buckets: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.progress_head = nn.Linear(hidden_dim, progress_num_buckets)
        self.loop_head = nn.Linear(hidden_dim, 1)
        self.eos_head = nn.Linear(hidden_dim, 1)

    def forward(self, q0_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        if q0_repr.ndim != 3:
            raise ValueError(f"Expected q0_repr shape [B,T,D], got {tuple(q0_repr.shape)}")
        hidden = self.shared(q0_repr)
        return {
            "q0_progress_logits": self.progress_head(hidden),
            "q0_loop_logits": self.loop_head(hidden),
            "q0_eos_logits": self.eos_head(hidden),
        }

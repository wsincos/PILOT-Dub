from __future__ import annotations

import torch
import torch.nn as nn


class PlannerStateG0Adapter(nn.Module):
    """Small gated adapter that injects planner state into q0-specific g0.

    The adapter operates after the q0 head's representation projection and
    before the q0 classifier, so it changes the dominant content codebook
    without perturbing q1-q3 residual codebooks.
    """

    def __init__(
        self,
        repr_dim: int,
        plan_dim: int,
        num_viseme_classes: int = 11,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        gate_init: float = -2.5,
    ):
        super().__init__()
        self.repr_dim = int(repr_dim)
        self.plan_dim = int(plan_dim)
        self.num_viseme_classes = int(num_viseme_classes)
        input_dim = self.repr_dim + self.plan_dim + 3 + self.num_viseme_classes
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.repr_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.logit_gate = nn.Parameter(torch.tensor(float(gate_init)))

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_gate)

    def forward(
        self,
        q0_repr: torch.Tensor,
        plan_state: torch.Tensor,
        occurrence: torch.Tensor | None = None,
        remaining: torch.Tensor | None = None,
        stop: torch.Tensor | None = None,
        viseme_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return adapted q0 representation and delta.

        Args:
            q0_repr: [B, T, Dg]
            plan_state: [B, T, Dp]
            occurrence / remaining / stop: [B, T] or [B, T, 1]
            viseme_logits: [B, T, C]
        """
        if occurrence is None:
            occurrence = q0_repr.new_zeros(q0_repr.shape[:2] + (1,))
        if remaining is None:
            remaining = q0_repr.new_zeros(q0_repr.shape[:2] + (1,))
        if stop is None:
            stop = q0_repr.new_zeros(q0_repr.shape[:2] + (1,))
        if occurrence.ndim == 2:
            occurrence = occurrence.unsqueeze(-1)
        if remaining.ndim == 2:
            remaining = remaining.unsqueeze(-1)
        if stop.ndim == 2:
            stop = stop.unsqueeze(-1)
        if viseme_logits is None:
            viseme_logits = q0_repr.new_zeros(q0_repr.shape[:2] + (self.num_viseme_classes,))
        if viseme_logits.shape[-1] != self.num_viseme_classes:
            if viseme_logits.shape[-1] > self.num_viseme_classes:
                viseme_logits = viseme_logits[..., : self.num_viseme_classes]
            else:
                pad = q0_repr.new_zeros(
                    viseme_logits.shape[:-1] + (self.num_viseme_classes - viseme_logits.shape[-1],)
                )
                viseme_logits = torch.cat([viseme_logits, pad], dim=-1)

        features = torch.cat(
            [
                q0_repr,
                plan_state.to(dtype=q0_repr.dtype),
                occurrence.to(dtype=q0_repr.dtype),
                remaining.to(dtype=q0_repr.dtype),
                stop.to(dtype=q0_repr.dtype),
                viseme_logits.to(dtype=q0_repr.dtype),
            ],
            dim=-1,
        )
        delta = self.gate.to(dtype=q0_repr.dtype) * self.net(features)
        return q0_repr + delta, delta

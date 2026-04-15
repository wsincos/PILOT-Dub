from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class DelayedHierarchicalResidualConditioner(nn.Module):
    """
    Build explicit same-real-time earlier-codebook context under delayed codebook pattern.

    Current delayed position p for codebook q corresponds to real time:
        t = p - 1 - q

    For any earlier codebook q' < q with the same real time t, its delayed position is:
        p' = p - (q - q')

    This module collects those earlier-codebook tokens, embeds them with the
    codebook-specific audio embedding, aggregates them, and injects the residual
    context into the corresponding codebook-specific hidden states.
    """

    def __init__(
        self,
        dim_model: int,
        n_codebooks: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        context_scale_init_per_codebook: Optional[List[float]] = None,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.n_codebooks = n_codebooks

        if context_scale_init_per_codebook is None:
            context_scale_init_per_codebook = [0.0] + [0.03] * (n_codebooks - 1)
        if len(context_scale_init_per_codebook) != n_codebooks:
            raise ValueError(
                f"Expected {n_codebooks} context scales, got {len(context_scale_init_per_codebook)}"
            )

        self.context_scales = nn.Parameter(
            torch.tensor(context_scale_init_per_codebook, dtype=torch.float32)
        )
        self.context_projs = nn.ModuleList(
            [
                nn.Identity()
                if q == 0
                else nn.Sequential(
                    nn.LayerNorm(dim_model),
                    nn.Linear(dim_model, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, dim_model),
                )
                for q in range(n_codebooks)
            ]
        )

    def _build_context_from_shifted_tokens(
        self,
        shifted_token_ids: torch.Tensor,
        audio_embedding: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            shifted_token_ids: [B, Q, T_shifted]

        Returns:
            context: [B, Q, T_shifted, D]
            valid:   [B, Q, T_shifted]
        """
        if shifted_token_ids.ndim != 3:
            raise ValueError(
                f"Expected shifted_token_ids shape [B,Q,T], got {tuple(shifted_token_ids.shape)}"
            )
        batch_size, n_codebooks, time_len = shifted_token_ids.shape
        if n_codebooks != self.n_codebooks:
            raise ValueError(
                f"Expected {self.n_codebooks} codebooks, got {n_codebooks}"
            )

        device = shifted_token_ids.device
        dtype = next(audio_embedding.parameters()).dtype
        context = torch.zeros(
            batch_size,
            self.n_codebooks,
            time_len,
            self.dim_model,
            device=device,
            dtype=dtype,
        )
        counts = torch.zeros(
            batch_size,
            self.n_codebooks,
            time_len,
            1,
            device=device,
            dtype=dtype,
        )

        for q in range(1, self.n_codebooks):
            for q_prev in range(q):
                shift = q - q_prev
                if time_len <= shift:
                    continue
                prev_ids = shifted_token_ids[:, q_prev, : time_len - shift]
                prev_emb = audio_embedding(prev_ids, codebook_idx=q_prev)
                context[:, q, shift:, :] += prev_emb
                counts[:, q, shift:, :] += 1.0

        valid = counts.squeeze(-1) > 0
        context = context / counts.clamp_min(1.0)
        return context, valid

    def forward(
        self,
        base_hidden: torch.Tensor,
        shifted_token_ids: torch.Tensor,
        audio_embedding: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            base_hidden: [B, Q, T, D]
            shifted_token_ids: [B, Q, T]
        """
        if base_hidden.ndim != 4:
            raise ValueError(f"Expected base_hidden shape [B,Q,T,D], got {tuple(base_hidden.shape)}")
        if shifted_token_ids.shape[:3] != base_hidden.shape[:3]:
            raise ValueError(
                f"Shape mismatch: hidden {tuple(base_hidden.shape[:3])} vs shifted tokens {tuple(shifted_token_ids.shape)}"
            )

        context, valid = self._build_context_from_shifted_tokens(
            shifted_token_ids=shifted_token_ids,
            audio_embedding=audio_embedding,
        )

        outputs = base_hidden.clone()
        context_norm_means: List[torch.Tensor] = []
        for q in range(1, self.n_codebooks):
            delta = self.context_projs[q](context[:, q, :, :])
            delta = self.context_scales[q] * delta
            outputs[:, q, :, :] = torch.where(
                valid[:, q, :].unsqueeze(-1),
                outputs[:, q, :, :] + delta,
                outputs[:, q, :, :],
            )
            if bool(valid[:, q, :].any().item()):
                context_norm_means.append(delta[valid[:, q, :]].norm(dim=-1).mean())
            else:
                context_norm_means.append(delta.new_tensor(0.0))

        metrics = {
            "hier_context_valid_mask": valid.any(dim=1),  # [B, T]
            "hier_context_norm_means": torch.stack(
                [outputs.new_tensor(0.0)] + context_norm_means,
                dim=0,
            ).detach(),
            "hier_context_scales": self.context_scales.detach(),
        }
        return outputs, metrics

    def forward_step(
        self,
        base_hidden_step: torch.Tensor,
        curr_generated: List[List[int]],
        step: int,
        audio_embedding: nn.Module,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            base_hidden_step: [B, Q, 1, D], usually B=1
            curr_generated: delayed-step history, one list per codebook
            step: current delayed step index
        """
        if base_hidden_step.ndim != 4 or base_hidden_step.shape[2] != 1:
            raise ValueError(
                f"Expected base_hidden_step shape [B,Q,1,D], got {tuple(base_hidden_step.shape)}"
            )

        batch_size = base_hidden_step.shape[0]
        shifted_tokens = torch.full(
            (batch_size, self.n_codebooks, 1),
            fill_value=0,
            device=device,
            dtype=torch.long,
        )
        valid = torch.zeros(
            (batch_size, self.n_codebooks, 1),
            device=device,
            dtype=torch.bool,
        )

        # Build same-real-time earlier-codebook context for the current delayed position.
        for q in range(1, self.n_codebooks):
            for q_prev in range(q):
                shift = q - q_prev
                prev_step = step - shift
                if prev_step < 0:
                    continue
                if prev_step >= len(curr_generated[q_prev]):
                    continue
                shifted_tokens[0, q_prev, 0] = int(curr_generated[q_prev][prev_step])
                valid[0, q_prev, 0] = True

        # Materialize context using the same logic as training, but only at T=1.
        outputs = base_hidden_step.clone()
        context_norm_means: List[torch.Tensor] = [outputs.new_tensor(0.0)]
        for q in range(1, self.n_codebooks):
            ctx_sum = outputs.new_zeros((batch_size, self.dim_model))
            count = outputs.new_zeros((batch_size, 1))
            for q_prev in range(q):
                if not bool(valid[0, q_prev, 0].item()):
                    continue
                emb = audio_embedding(shifted_tokens[:, q_prev, 0], codebook_idx=q_prev)  # [B, D]
                ctx_sum += emb
                count += 1.0
            if bool((count > 0).any().item()):
                ctx = ctx_sum / count.clamp_min(1.0)
                delta = self.context_projs[q](ctx)
                delta = self.context_scales[q] * delta
                outputs[:, q, 0, :] = outputs[:, q, 0, :] + delta
                context_norm_means.append(delta.norm(dim=-1).mean().detach())
            else:
                context_norm_means.append(outputs.new_tensor(0.0))

        metrics = {
            "hier_context_valid_mask": valid.any(dim=1),  # [B, 1]
            "hier_context_norm_means": torch.stack(context_norm_means, dim=0),
            "hier_context_scales": self.context_scales.detach(),
        }
        return outputs, metrics

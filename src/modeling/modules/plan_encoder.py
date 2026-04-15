from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grounded_planner import GroundedPlannerBlock


class LegacyAudiovisualPlanEncoder(nn.Module):
    """
    Checkpoint-compatible planner used by the v14.5/v5 artifacts.

    The newer AudiovisualPlanEncoder was later upgraded to a grounded dual-memory
    block, which changes parameter names and breaks strict loading for the old
    v14.5 checkpoints. This legacy module intentionally preserves the old
    parameter layout:

      input_norm, video_proj, text_proj, ref_proj, context_norm, encoder, output_norm
    """

    def __init__(
        self,
        dim_model: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_text_condition: bool = True,
        use_ref_condition: bool = True,
    ):
        super().__init__()
        self.use_text_condition = use_text_condition
        self.use_ref_condition = use_ref_condition
        self.input_norm = nn.LayerNorm(dim_model)
        self.video_proj = nn.Linear(dim_model, dim_model)
        self.text_proj = nn.Linear(dim_model, dim_model) if use_text_condition else None
        self.ref_proj = nn.Linear(dim_model, dim_model) if use_ref_condition else None
        self.context_norm = nn.LayerNorm(dim_model)
        layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(dim_model)

    @staticmethod
    def _masked_mean(x: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
        if lengths is None:
            return x.mean(dim=1)
        max_len = x.shape[1]
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.to(device=x.device).unsqueeze(1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=x.dtype)
        return (x * mask.unsqueeze(-1).to(dtype=x.dtype)).sum(dim=1) / denom

    def forward(
        self,
        video_embeddings: torch.Tensor,
        video_lens: torch.Tensor,
        text_memory: Optional[torch.Tensor] = None,
        text_lens: Optional[torch.Tensor] = None,
        ref_memory: Optional[torch.Tensor] = None,
        ref_lens: Optional[torch.Tensor] = None,
        text_summary: Optional[torch.Tensor] = None,
        ref_summary: Optional[torch.Tensor] = None,
        text_token_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.input_norm(video_embeddings)
        x = self.video_proj(x)

        context = torch.zeros_like(x)
        if self.text_proj is not None:
            if text_summary is None and text_memory is not None:
                text_summary = self._masked_mean(text_memory, text_lens)
            if text_summary is not None:
                context = context + self.text_proj(text_summary).unsqueeze(1)
        if self.ref_proj is not None:
            if ref_summary is None and ref_memory is not None and ref_memory.shape[1] > 0:
                ref_summary = self._masked_mean(ref_memory, ref_lens)
            if ref_summary is not None:
                context = context + self.ref_proj(ref_summary).unsqueeze(1)

        x = self.context_norm(x + context)
        video_max_len = x.shape[1]
        video_pad_mask = torch.arange(video_max_len, device=x.device).unsqueeze(0) >= video_lens.unsqueeze(1)
        x = self.encoder(x, src_key_padding_mask=video_pad_mask)
        x = self.output_norm(x)
        return {"plan_hidden": x}


class AudiovisualPlanEncoder(nn.Module):
    """
    Backward-compatible planner module name, now implemented as a grounded
    dual-memory monotonic planner.
    """

    def __init__(
        self,
        dim_model: int,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_text_condition: bool = True,
        use_ref_condition: bool = True,
        use_global_text_summary: bool = True,
        use_global_ref_summary: bool = True,
    ):
        super().__init__()
        self.use_text_condition = use_text_condition
        self.use_ref_condition = use_ref_condition
        self.use_global_text_summary = use_global_text_summary
        self.use_global_ref_summary = use_global_ref_summary

        self.input_norm = nn.LayerNorm(dim_model)
        self.video_proj = nn.Linear(dim_model, dim_model)
        self.video_act = nn.GELU()
        self.video_dropout = nn.Dropout(dropout)

        self.text_proj = nn.Linear(dim_model, dim_model) if use_text_condition else None
        self.ref_proj = nn.Linear(dim_model, dim_model) if use_ref_condition else None
        self.text_global_proj = nn.Linear(dim_model, dim_model) if use_global_text_summary else None
        self.ref_global_proj = nn.Linear(dim_model, dim_model) if use_global_ref_summary else None
        self.context_norm = nn.LayerNorm(dim_model)

        self.layers = nn.ModuleList(
            [
                GroundedPlannerBlock(
                    dim_model=dim_model,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(dim_model)

    def forward(
        self,
        video_embeddings: torch.Tensor,
        video_lens: torch.Tensor,
        text_memory: torch.Tensor,
        text_lens: torch.Tensor,
        ref_memory: Optional[torch.Tensor] = None,
        ref_lens: Optional[torch.Tensor] = None,
        text_summary: Optional[torch.Tensor] = None,
        ref_summary: Optional[torch.Tensor] = None,
        text_token_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.input_norm(video_embeddings)
        x = self.video_proj(x)
        x = self.video_act(x)
        x = self.video_dropout(x)

        if self.text_proj is not None:
            text_memory = self.text_proj(text_memory)
        if self.ref_proj is not None and ref_memory is not None and ref_memory.shape[1] > 0:
            ref_memory = self.ref_proj(ref_memory)

        if self.text_global_proj is not None and text_summary is not None:
            x = x + self.text_global_proj(text_summary).unsqueeze(1)
        if self.ref_global_proj is not None and ref_summary is not None:
            x = x + self.ref_global_proj(ref_summary).unsqueeze(1)
        x = self.context_norm(x)

        video_max_len = x.shape[1]
        video_pad_mask = torch.arange(video_max_len, device=x.device).unsqueeze(0) >= video_lens.unsqueeze(1)
        text_pad_mask = torch.arange(text_memory.shape[1], device=x.device).unsqueeze(0) >= text_lens.unsqueeze(1)
        ref_pad_mask = None
        if ref_memory is not None and ref_lens is not None and ref_memory.shape[1] > 0:
            ref_pad_mask = torch.arange(ref_memory.shape[1], device=x.device).unsqueeze(0) >= ref_lens.unsqueeze(1)

        last_text_weights = None
        gate_traces = []
        for layer in self.layers:
            x, text_weights, _, gates = layer(
                x=x,
                self_pad_mask=video_pad_mask,
                text_memory=text_memory,
                text_pad_mask=text_pad_mask,
                ref_memory=ref_memory,
                ref_pad_mask=ref_pad_mask,
            )
            last_text_weights = text_weights
            gate_traces.append(gates.detach())

        x = self.output_norm(x)
        outputs: Dict[str, torch.Tensor] = {
            "plan_hidden": x,
        }
        if last_text_weights is not None:
            text_positions = torch.arange(last_text_weights.shape[-1], device=x.device, dtype=x.dtype)
            text_expected_positions = (last_text_weights * text_positions.view(1, 1, -1)).sum(dim=-1)
            text_entropy = -(last_text_weights.clamp_min(1e-8) * last_text_weights.clamp_min(1e-8).log()).sum(dim=-1)
            mono_delta = F.relu(text_expected_positions[:, :-1] - text_expected_positions[:, 1:])
            valid_pair_mask = (~video_pad_mask[:, :-1]) & (~video_pad_mask[:, 1:])
            mono_loss = mono_delta[valid_pair_mask].mean() if bool(valid_pair_mask.any().item()) else x.new_tensor(0.0)
            outputs.update(
                {
                    "plan_text_attention": last_text_weights,
                    "plan_text_expected_positions": text_expected_positions.detach(),
                    "plan_text_entropy": text_entropy.detach(),
                    "plan_monotonic_loss": mono_loss,
                    "plan_gate_means": torch.stack(gate_traces, dim=0).mean(dim=(1, 2)).detach(),
                }
            )
        return outputs

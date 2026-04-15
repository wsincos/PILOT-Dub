from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundedPlannerBlock(nn.Module):
    def __init__(
        self,
        dim_model: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(dim_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.text_attn_norm = nn.LayerNorm(dim_model)
        self.text_attn = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ref_attn_norm = nn.LayerNorm(dim_model)
        self.ref_attn = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.gate_norm = nn.LayerNorm(dim_model * 3)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim_model * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        self.ffn_norm = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        self_pad_mask: Optional[torch.Tensor],
        text_memory: torch.Tensor,
        text_pad_mask: Optional[torch.Tensor],
        ref_memory: Optional[torch.Tensor],
        ref_pad_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        x_self = self.self_attn_norm(x)
        self_ctx, _ = self.self_attn(
            x_self,
            x_self,
            x_self,
            key_padding_mask=self_pad_mask,
            need_weights=False,
        )
        x = x + self_ctx

        q_text = self.text_attn_norm(x)
        text_ctx, text_weights = self.text_attn(
            q_text,
            text_memory,
            text_memory,
            key_padding_mask=text_pad_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        if ref_memory is not None and ref_memory.shape[1] > 0:
            q_ref = self.ref_attn_norm(x)
            ref_ctx, ref_weights = self.ref_attn(
                q_ref,
                ref_memory,
                ref_memory,
                key_padding_mask=ref_pad_mask,
                need_weights=True,
                average_attn_weights=True,
            )
        else:
            ref_ctx = torch.zeros_like(text_ctx)
            ref_weights = None

        gate_inp = torch.cat([x, text_ctx, ref_ctx], dim=-1)
        gates = torch.sigmoid(self.gate_mlp(self.gate_norm(gate_inp)))
        text_gate = gates[..., :1]
        ref_gate = gates[..., 1:]
        x = x + text_gate * text_ctx + ref_gate * ref_ctx

        x = x + self.ffn(self.ffn_norm(x))
        return x, text_weights, ref_weights, gates


class GroundedDualMemoryMonotonicPlanner(nn.Module):
    """
    Planner that grounds each video step to token-level text memory and
    local reference memory, rather than only global summaries.
    """

    def __init__(
        self,
        dim_model: int,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_global_text_summary: bool = True,
        use_global_ref_summary: bool = True,
    ):
        super().__init__()
        self.use_global_text_summary = use_global_text_summary
        self.use_global_ref_summary = use_global_ref_summary

        self.input_norm = nn.LayerNorm(dim_model)
        self.video_proj = nn.Linear(dim_model, dim_model)
        self.video_act = nn.GELU()
        self.video_dropout = nn.Dropout(dropout)

        self.text_proj = nn.Linear(dim_model, dim_model)
        self.ref_proj = nn.Linear(dim_model, dim_model)
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
    ) -> Dict[str, torch.Tensor]:
        x = self.input_norm(video_embeddings)
        x = self.video_proj(x)
        x = self.video_act(x)
        x = self.video_dropout(x)

        text_memory = self.text_proj(text_memory)
        if ref_memory is not None and ref_memory.shape[1] > 0:
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
            if bool(valid_pair_mask.any().item()):
                mono_loss = mono_delta[valid_pair_mask].mean()
            else:
                mono_loss = x.new_tensor(0.0)
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

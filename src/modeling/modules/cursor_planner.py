from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CursorPlannerBlock(nn.Module):
    def __init__(
        self,
        dim_model: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        cursor_window_radius: float,
    ):
        super().__init__()
        self.cursor_window_radius = float(cursor_window_radius)

        self.self_attn_norm = nn.LayerNorm(dim_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cursor_query = nn.Linear(dim_model, dim_model)
        self.cursor_key = nn.Linear(dim_model, dim_model)
        self.cursor_value = nn.Linear(dim_model, dim_model)

        self.text_out_proj = nn.Linear(dim_model, dim_model)
        self.gate_norm = nn.LayerNorm(dim_model * 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim_model * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_self = self.self_attn_norm(x)
        self_ctx, _ = self.self_attn(
            x_self,
            x_self,
            x_self,
            key_padding_mask=self_pad_mask,
            need_weights=False,
        )
        x = x + self_ctx

        q = self.cursor_query(x)
        k = self.cursor_key(text_memory)
        v = self.cursor_value(text_memory)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (x.shape[-1] ** 0.5)  # [B, Tv, Ttxt]

        if text_pad_mask is not None:
            logits = logits.masked_fill(text_pad_mask.unsqueeze(1), float("-inf"))

        raw_weights = F.softmax(logits, dim=-1)
        positions = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
        expected_pos = (raw_weights * positions.view(1, 1, -1)).sum(dim=-1)  # [B, Tv]

        # Localize reads around the predicted cursor instead of free-form full-text reading.
        dist2 = (positions.view(1, 1, -1) - expected_pos.unsqueeze(-1)) ** 2
        local_bias = -(dist2 / (2.0 * max(self.cursor_window_radius, 1e-3) ** 2))
        local_logits = logits + local_bias
        if text_pad_mask is not None:
            local_logits = local_logits.masked_fill(text_pad_mask.unsqueeze(1), float("-inf"))
        local_weights = F.softmax(local_logits, dim=-1)

        text_ctx = torch.matmul(local_weights, v)
        text_ctx = self.text_out_proj(text_ctx)

        gate = torch.sigmoid(self.gate_mlp(self.gate_norm(torch.cat([x, text_ctx], dim=-1))))
        x = x + gate * text_ctx
        x = x + self.ffn(self.ffn_norm(x))
        return x, logits, expected_pos, local_weights


class ExplicitCursorStatePlanner(nn.Module):
    """
    Planner with explicit cursor state, local text-window reading, and
    remaining/stop prediction.
    """

    def __init__(
        self,
        dim_model: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        cursor_window_radius: float = 4.0,
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

        self.text_proj = nn.Linear(dim_model, dim_model)
        self.text_global_proj = nn.Linear(dim_model, dim_model) if use_global_text_summary else None
        self.ref_global_proj = nn.Linear(dim_model, dim_model) if use_global_ref_summary else None
        self.context_norm = nn.LayerNorm(dim_model)

        self.layers = nn.ModuleList(
            [
                CursorPlannerBlock(
                    dim_model=dim_model,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    cursor_window_radius=cursor_window_radius,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(dim_model)
        self.remaining_head = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.stop_head = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

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

        text_memory = self.text_proj(text_memory)
        if self.text_global_proj is not None and text_summary is not None:
            x = x + self.text_global_proj(text_summary).unsqueeze(1)
        if self.ref_global_proj is not None and ref_summary is not None:
            x = x + self.ref_global_proj(ref_summary).unsqueeze(1)
        x = self.context_norm(x)

        video_max_len = x.shape[1]
        video_pad_mask = torch.arange(video_max_len, device=x.device).unsqueeze(0) >= video_lens.unsqueeze(1)
        text_pad_mask = torch.arange(text_memory.shape[1], device=x.device).unsqueeze(0) >= text_lens.unsqueeze(1)

        last_cursor_logits = None
        last_expected = None
        last_local_weights = None
        for layer in self.layers:
            x, cursor_logits, expected_pos, local_weights = layer(
                x=x,
                self_pad_mask=video_pad_mask,
                text_memory=text_memory,
                text_pad_mask=text_pad_mask,
            )
            last_cursor_logits = cursor_logits
            last_expected = expected_pos
            last_local_weights = local_weights

        x = self.output_norm(x)

        mono_delta = F.relu(last_expected[:, :-1] - last_expected[:, 1:]) if last_expected is not None else None
        valid_pair_mask = (~video_pad_mask[:, :-1]) & (~video_pad_mask[:, 1:]) if last_expected is not None else None
        mono_loss = (
            mono_delta[valid_pair_mask].mean()
            if mono_delta is not None and bool(valid_pair_mask.any().item())
            else x.new_tensor(0.0)
        )
        local_entropy = (
            -(last_local_weights.clamp_min(1e-8) * last_local_weights.clamp_min(1e-8).log()).sum(dim=-1)
            if last_local_weights is not None
            else x.new_zeros((x.shape[0], x.shape[1]))
        )

        remaining_pred = self.remaining_head(x).squeeze(-1)
        stop_logits = self.stop_head(x).squeeze(-1)

        return {
            "plan_hidden": x,
            "plan_cursor_logits": last_cursor_logits,
            "plan_cursor_expected_positions": last_expected.detach() if last_expected is not None else None,
            "plan_cursor_entropy": local_entropy.detach(),
            "plan_monotonic_loss": mono_loss,
            "plan_remaining_pred": remaining_pred,
            "plan_stop_logits": stop_logits,
        }

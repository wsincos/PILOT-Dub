from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .viseme_utils import (
    NUM_VISEME_CLASSES,
    VIS_SIL,
    build_align_label_to_viseme_table,
    build_text_token_to_viseme_table,
)


class PhonemeVisemePlannerBlock(nn.Module):
    def __init__(
        self,
        dim_model: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        cursor_window_radius: float,
        num_viseme_classes: int,
        compatibility_scale: float,
    ):
        super().__init__()
        self.cursor_window_radius = float(cursor_window_radius)
        self.compatibility_scale = float(compatibility_scale)

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
        self.viseme_state_proj = nn.Linear(dim_model, num_viseme_classes)

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
        text_token_viseme_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        logits = torch.matmul(q, k.transpose(-2, -1)) / (x.shape[-1] ** 0.5)

        if text_pad_mask is not None:
            logits = logits.masked_fill(text_pad_mask.unsqueeze(1), float("-inf"))

        viseme_logits = self.viseme_state_proj(x)
        compat_bias = x.new_zeros(logits.shape)
        if text_token_viseme_ids is not None:
            viseme_log_probs = F.log_softmax(viseme_logits, dim=-1)
            gather_ids = text_token_viseme_ids.clamp_min(0).clamp_max(viseme_log_probs.shape[-1] - 1)
            compat_bias = viseme_log_probs.gather(
                dim=-1,
                index=gather_ids.unsqueeze(1).expand(-1, viseme_log_probs.shape[1], -1),
            )
            if text_pad_mask is not None:
                compat_bias = compat_bias.masked_fill(text_pad_mask.unsqueeze(1), 0.0)
            logits = logits + self.compatibility_scale * compat_bias

        raw_weights = F.softmax(logits, dim=-1)
        positions = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
        expected_pos = (raw_weights * positions.view(1, 1, -1)).sum(dim=-1)

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
        return x, logits, expected_pos, local_weights, viseme_logits, compat_bias


class PhonemeVisemeDualStatePlanner(nn.Module):
    def __init__(
        self,
        dim_model: int,
        hidden_dim: int,
        text_vocab_path: str,
        align_vocab_path: str,
        text_vocab_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        cursor_window_radius: float = 4.0,
        num_viseme_classes: int = NUM_VISEME_CLASSES,
        compatibility_scale: float = 0.5,
        use_text_condition: bool = True,
        use_ref_condition: bool = True,
        use_global_text_summary: bool = True,
        use_global_ref_summary: bool = True,
    ):
        super().__init__()
        self.num_viseme_classes = int(num_viseme_classes)
        self.use_text_condition = use_text_condition
        self.use_ref_condition = use_ref_condition
        self.use_global_text_summary = use_global_text_summary
        self.use_global_ref_summary = use_global_ref_summary

        self.text_token_to_viseme = build_text_token_to_viseme_table(
            text_vocab_path, table_size=text_vocab_size
        )
        self.align_label_to_viseme = build_align_label_to_viseme_table(align_vocab_path)

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
                PhonemeVisemePlannerBlock(
                    dim_model=dim_model,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    cursor_window_radius=cursor_window_radius,
                    num_viseme_classes=num_viseme_classes,
                    compatibility_scale=compatibility_scale,
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
        self.activity_head = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.viseme_head = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_viseme_classes),
        )

    def _map_text_ids_to_viseme(self, text_token_ids: torch.Tensor) -> torch.Tensor:
        table = self.text_token_to_viseme.to(device=text_token_ids.device)
        safe_ids = text_token_ids.clamp_min(0).clamp_max(table.shape[0] - 1)
        return table[safe_ids]

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
        text_token_viseme_ids = None
        if text_token_ids is not None:
            text_token_viseme_ids = self._map_text_ids_to_viseme(text_token_ids.to(device=x.device, dtype=torch.long))

        last_cursor_logits = None
        last_expected = None
        last_local_weights = None
        last_viseme_logits = None
        compat_bias_mean = None
        for layer in self.layers:
            x, cursor_logits, expected_pos, local_weights, viseme_logits, compat_bias = layer(
                x=x,
                self_pad_mask=video_pad_mask,
                text_memory=text_memory,
                text_pad_mask=text_pad_mask,
                text_token_viseme_ids=text_token_viseme_ids,
            )
            last_cursor_logits = cursor_logits
            last_expected = expected_pos
            last_local_weights = local_weights
            last_viseme_logits = viseme_logits
            compat_bias_mean = compat_bias.abs().mean().detach()

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
        activity_logits = self.activity_head(x).squeeze(-1)
        viseme_logits = self.viseme_head(x)

        return {
            "plan_hidden": x,
            "plan_cursor_logits": last_cursor_logits,
            "plan_cursor_expected_positions": last_expected.detach() if last_expected is not None else None,
            "plan_cursor_entropy": local_entropy.detach(),
            "plan_monotonic_loss": mono_loss,
            "plan_remaining_pred": remaining_pred,
            "plan_stop_logits": stop_logits,
            "plan_activity_logits": activity_logits,
            "plan_viseme_logits": viseme_logits,
            "plan_viseme_compatibility_mean": compat_bias_mean,
        }

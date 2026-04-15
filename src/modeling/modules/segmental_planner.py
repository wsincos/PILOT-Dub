from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualStatePlannerBlock(nn.Module):
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
        self.ffn_norm = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, self_pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x_self = self.self_attn_norm(x)
        self_ctx, _ = self.self_attn(
            x_self,
            x_self,
            x_self,
            key_padding_mask=self_pad_mask,
            need_weights=False,
        )
        x = x + self_ctx
        x = x + self.ffn(self.ffn_norm(x))
        return x


class VisualStateSegmentalPlanner(nn.Module):
    """
    First stage of the v15.0 reboot:
    - learn visual speech states from video
    - predict MFA-aligned phoneme position over a planner-specific phoneme stream
    - output segmental boundary / duration related states
    """

    def __init__(
        self,
        dim_model: int,
        hidden_dim: int,
        planner_phone_vocab_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_viseme_classes: int = 11,
        use_text_condition: bool = True,
        use_ref_condition: bool = True,
    ):
        super().__init__()
        self.use_text_condition = use_text_condition
        self.use_ref_condition = use_ref_condition
        self.input_norm = nn.LayerNorm(dim_model)
        self.video_proj = nn.Linear(dim_model, dim_model)
        self.video_act = nn.GELU()
        self.video_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                VisualStatePlannerBlock(
                    dim_model=dim_model,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(dim_model)
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
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.planner_phone_head = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, planner_phone_vocab_size),
        )
        self.occurrence_head = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
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
        x = self.video_act(x)
        x = self.video_dropout(x)

        video_max_len = x.shape[1]
        video_pad_mask = torch.arange(video_max_len, device=x.device).unsqueeze(0) >= video_lens.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, self_pad_mask=video_pad_mask)
        x = self.output_norm(x)

        activity_logits = self.activity_head(x).squeeze(-1)
        viseme_logits = self.viseme_head(x)
        boundary_logits = self.boundary_head(x)
        planner_phone_logits = self.planner_phone_head(x)
        occurrence_pred = self.occurrence_head(x).squeeze(-1)
        remaining_pred = self.remaining_head(x).squeeze(-1)
        stop_logits = self.stop_head(x).squeeze(-1)

        valid_pair_mask = (~video_pad_mask[:, :-1]) & (~video_pad_mask[:, 1:]) if x.shape[1] > 1 else None
        mono_delta = F.relu(torch.sigmoid(stop_logits[:, :-1]) - torch.sigmoid(stop_logits[:, 1:])) if x.shape[1] > 1 else None
        mono_loss = (
            mono_delta[valid_pair_mask].mean()
            if mono_delta is not None and bool(valid_pair_mask.any().item())
            else x.new_tensor(0.0)
        )

        return {
            "plan_hidden": x,
            "plan_activity_logits": activity_logits,
            "plan_viseme_logits": viseme_logits,
            "plan_boundary_logits": boundary_logits,
            "plan_phone_logits": planner_phone_logits,
            "plan_occurrence_pred": occurrence_pred,
            "plan_remaining_pred": remaining_pred,
            "plan_stop_logits": stop_logits,
            "plan_monotonic_loss": mono_loss,
        }

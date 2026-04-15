import math
import torch
import torch.nn as nn

class FusionLayer(nn.Linear):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)


class LocalCrossAttentionFusion(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        window_left: int,
        window_right: int,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if dim_model % num_heads != 0:
            raise ValueError(f"dim_model ({dim_model}) must be divisible by num_heads ({num_heads})")

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.window_left = window_left
        self.window_right = window_right

        self.query_norm = nn.LayerNorm(dim_model) if use_layer_norm else nn.Identity()
        self.context_norm = nn.LayerNorm(dim_model) if use_layer_norm else nn.Identity()
        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_positions: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        if query.numel() == 0 or context.numel() == 0:
            return query

        batch_size, target_len, _ = query.shape
        _, _, dim_model = context.shape
        if dim_model != self.dim_model:
            raise ValueError(f"context dim {dim_model} != fusion dim {self.dim_model}")

        offsets = torch.arange(
            -self.window_left,
            self.window_right + 1,
            device=query.device,
            dtype=query_positions.dtype,
        )
        window_indices = query_positions.unsqueeze(-1) + offsets.view(1, 1, -1)
        valid = (window_indices >= 0) & (window_indices < context_lens.view(batch_size, 1, 1))
        max_index = max(context.shape[1] - 1, 0)
        window_indices = window_indices.clamp(min=0, max=max_index)

        batch_indices = torch.arange(batch_size, device=query.device).view(batch_size, 1, 1)
        batch_indices = batch_indices.expand_as(window_indices)
        window_context = context[batch_indices, window_indices]

        q = self.q_proj(self.query_norm(query)).view(batch_size, target_len, self.num_heads, self.head_dim)
        normed_context = self.context_norm(window_context)
        k = self.k_proj(normed_context).view(batch_size, target_len, -1, self.num_heads, self.head_dim)
        v = self.v_proj(normed_context).view(batch_size, target_len, -1, self.num_heads, self.head_dim)

        attn_scores = torch.einsum("bthd,btwhd->bthw", q, k) / math.sqrt(self.head_dim)
        neg_fill = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~valid.unsqueeze(2), neg_fill)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights * valid.unsqueeze(2)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        fused = torch.einsum("bthw,btwhd->bthd", attn_weights, v).reshape(batch_size, target_len, self.dim_model)
        fused = self.out_proj(fused)
        fused = self.dropout(fused)
        return query + fused


class GatedLocalCrossAttentionFusion(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        window_left: int,
        window_right: int,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.fusion = LocalCrossAttentionFusion(
            dim_model=dim_model,
            num_heads=num_heads,
            window_left=window_left,
            window_right=window_right,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_positions: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.fusion(
            query=query,
            context=context,
            query_positions=query_positions,
            context_lens=context_lens,
        )
        delta = fused - query
        valid_query = (query_positions >= 0).unsqueeze(-1)
        return torch.where(valid_query, query + self.gate * delta, query)

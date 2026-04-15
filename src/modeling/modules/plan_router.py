from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class DualPathCodebookPlanRouter(nn.Module):
    def __init__(
        self,
        dim_model: int,
        n_codebooks: int,
        summary_dim: Optional[int] = None,
        gate_hidden_dim: int = 512,
        window_left: int = 1,
        window_right: int = 4,
        dropout: float = 0.1,
        route_scale_init: float = 0.05,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.n_codebooks = n_codebooks
        self.summary_dim = int(summary_dim or dim_model)
        self.window_left = int(window_left)
        self.window_right = int(window_right)
        self.window_size = self.window_left + self.window_right + 1

        # Coarse path attention
        self.coarse_q_proj = nn.Linear(dim_model, self.summary_dim)
        self.coarse_k_proj = nn.Linear(dim_model, self.summary_dim)
        self.coarse_v_proj = nn.Linear(dim_model, self.summary_dim)

        # Boundary path attention
        self.boundary_q_proj = nn.Linear(dim_model, self.summary_dim)
        self.boundary_k_proj = nn.Linear(dim_model, self.summary_dim)
        self.boundary_v_proj = nn.Linear(dim_model, self.summary_dim)

        self.coarse_out_proj = nn.Linear(self.summary_dim, dim_model)
        self.boundary_out_proj = nn.Linear(self.summary_dim, dim_model)

        gate_input_dim = dim_model * 3
        self.gate_norm = nn.LayerNorm(gate_input_dim)
        self.gate_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gate_input_dim, gate_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(gate_hidden_dim, 1),
                )
                for _ in range(n_codebooks)
            ]
        )
        self.coarse_route_projs = nn.ModuleList(
            [nn.Linear(dim_model, dim_model) for _ in range(n_codebooks)]
        )
        self.boundary_route_projs = nn.ModuleList(
            [nn.Linear(dim_model, dim_model) for _ in range(n_codebooks)]
        )
        self.route_scale = nn.Parameter(torch.full((n_codebooks,), float(route_scale_init)))

    def _gather_plan_window(
        self,
        plan_hidden: torch.Tensor,
        query_positions: torch.Tensor,
        context_lens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, dim_model = plan_hidden.shape
        _, target_len = query_positions.shape
        offsets = torch.arange(
            -self.window_left,
            self.window_right + 1,
            device=plan_hidden.device,
            dtype=torch.long,
        ).view(1, 1, self.window_size)
        base_positions = query_positions.to(device=plan_hidden.device, dtype=torch.long).unsqueeze(-1)
        positions = base_positions + offsets

        if context_lens is None:
            max_valid = torch.full(
                (batch_size, 1, 1),
                fill_value=plan_hidden.shape[1],
                device=plan_hidden.device,
                dtype=torch.long,
            )
        else:
            max_valid = context_lens.to(device=plan_hidden.device, dtype=torch.long).view(batch_size, 1, 1)

        valid = (
            (base_positions >= 0)
            & (positions >= 0)
            & (positions < max_valid)
        )
        clipped_positions = positions.clamp(min=0, max=max(plan_hidden.shape[1] - 1, 0))

        expanded = plan_hidden.unsqueeze(1).expand(-1, target_len, -1, -1)
        gathered = torch.gather(
            expanded,
            2,
            clipped_positions.unsqueeze(-1).expand(-1, -1, -1, dim_model),
        )
        return gathered, valid

    @staticmethod
    def _masked_attention_pool(
        query_proj: torch.Tensor,
        key_proj: torch.Tensor,
        value_proj: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = (query_proj.unsqueeze(2) * key_proj).sum(dim=-1) / math.sqrt(query_proj.shape[-1])
        scores = scores.masked_fill(~valid_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * valid_mask.to(dtype=attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (attn.unsqueeze(-1) * value_proj).sum(dim=2)

    def forward(
        self,
        query_hidden: torch.Tensor,
        plan_hidden: torch.Tensor,
        query_positions: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        window_hidden, valid_mask = self._gather_plan_window(
            plan_hidden=plan_hidden,
            query_positions=query_positions,
            context_lens=context_lens,
        )
        query_valid_mask = valid_mask.any(dim=-1)

        coarse_query = self.coarse_q_proj(query_hidden)
        coarse_key = self.coarse_k_proj(window_hidden)
        coarse_value = self.coarse_v_proj(window_hidden)
        coarse_summary = self._masked_attention_pool(
            coarse_query,
            coarse_key,
            coarse_value,
            valid_mask,
        )
        coarse_summary = self.coarse_out_proj(coarse_summary)
        coarse_summary = torch.where(
            query_valid_mask.unsqueeze(-1),
            coarse_summary,
            torch.zeros_like(coarse_summary),
        )

        boundary_query = self.boundary_q_proj(query_hidden)
        boundary_key = self.boundary_k_proj(window_hidden)
        boundary_value = self.boundary_v_proj(window_hidden)
        boundary_summary = self._masked_attention_pool(
            boundary_query,
            boundary_key,
            boundary_value,
            valid_mask,
        )
        boundary_summary = self.boundary_out_proj(boundary_summary)
        boundary_summary = torch.where(
            query_valid_mask.unsqueeze(-1),
            boundary_summary,
            torch.zeros_like(boundary_summary),
        )

        gate_input = torch.cat([query_hidden, coarse_summary, boundary_summary], dim=-1)
        gate_input = self.gate_norm(gate_input)

        routed_hidden = []
        routing_lambdas = []
        for q in range(self.n_codebooks):
            lambda_q = torch.sigmoid(self.gate_mlps[q](gate_input))
            coarse_q = self.coarse_route_projs[q](coarse_summary)
            boundary_q = self.boundary_route_projs[q](boundary_summary)
            route_q = lambda_q * coarse_q + (1.0 - lambda_q) * boundary_q
            out_q = query_hidden + self.route_scale[q] * route_q
            out_q = torch.where(query_valid_mask.unsqueeze(-1), out_q, query_hidden)
            lambda_q = torch.where(query_valid_mask.unsqueeze(-1), lambda_q, torch.full_like(lambda_q, 0.5))
            routed_hidden.append(out_q)
            routing_lambdas.append(lambda_q.squeeze(-1))

        routed_hidden = torch.stack(routed_hidden, dim=1)  # [B, Q, T, D]
        routing_lambdas = torch.stack(routing_lambdas, dim=-1)  # [B, T, Q]

        metrics = {
            "routing_lambdas": routing_lambdas,
            "routing_valid_mask": query_valid_mask,
            "route_scale_values": self.route_scale.detach(),
        }
        return routed_hidden, metrics


class RealTimeAlignedAsymmetricCodebookPlanRouter(nn.Module):
    def __init__(
        self,
        dim_model: int,
        n_codebooks: int,
        summary_dim: Optional[int] = None,
        gate_hidden_dim: int = 512,
        window_left: int = 1,
        window_right: int = 4,
        dropout: float = 0.1,
        route_scale_init_per_codebook: Optional[list[float]] = None,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.n_codebooks = n_codebooks
        self.summary_dim = int(summary_dim or dim_model)
        self.window_left = int(window_left)
        self.window_right = int(window_right)
        self.window_size = self.window_left + self.window_right + 1

        self.coarse_q_proj = nn.Linear(dim_model, self.summary_dim)
        self.coarse_k_proj = nn.Linear(dim_model, self.summary_dim)
        self.coarse_v_proj = nn.Linear(dim_model, self.summary_dim)

        self.boundary_q_proj = nn.Linear(dim_model, self.summary_dim)
        self.boundary_k_proj = nn.Linear(dim_model, self.summary_dim)
        self.boundary_v_proj = nn.Linear(dim_model, self.summary_dim)

        self.coarse_out_proj = nn.Linear(self.summary_dim, dim_model)
        self.boundary_out_proj = nn.Linear(self.summary_dim, dim_model)

        gate_input_dim = dim_model * 3
        self.q0_gate_norm = nn.LayerNorm(gate_input_dim)
        self.q0_gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )

        self.coarse_route_proj_q0 = nn.Linear(dim_model, dim_model)
        self.boundary_route_projs = nn.ModuleList(
            [nn.Linear(dim_model, dim_model) for _ in range(n_codebooks)]
        )

        if route_scale_init_per_codebook is None:
            route_scale_init_per_codebook = [0.05, 0.03, 0.02, 0.02]
        if len(route_scale_init_per_codebook) != n_codebooks:
            raise ValueError(
                f"Expected {n_codebooks} route scales, got {len(route_scale_init_per_codebook)}"
            )
        self.route_scale = nn.Parameter(
            torch.tensor(route_scale_init_per_codebook, dtype=torch.float32)
        )

    def _gather_plan_window_single(
        self,
        plan_hidden: torch.Tensor,
        query_positions: torch.Tensor,
        context_lens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, dim_model = plan_hidden.shape
        _, target_len = query_positions.shape
        offsets = torch.arange(
            -self.window_left,
            self.window_right + 1,
            device=plan_hidden.device,
            dtype=torch.long,
        ).view(1, 1, self.window_size)
        base_positions = query_positions.to(device=plan_hidden.device, dtype=torch.long).unsqueeze(-1)
        positions = base_positions + offsets

        if context_lens is None:
            max_valid = torch.full(
                (batch_size, 1, 1),
                fill_value=plan_hidden.shape[1],
                device=plan_hidden.device,
                dtype=torch.long,
            )
        else:
            max_valid = context_lens.to(device=plan_hidden.device, dtype=torch.long).view(batch_size, 1, 1)

        valid = (base_positions >= 0) & (positions >= 0) & (positions < max_valid)
        clipped_positions = positions.clamp(min=0, max=max(plan_hidden.shape[1] - 1, 0))

        expanded = plan_hidden.unsqueeze(1).expand(-1, target_len, -1, -1)
        gathered = torch.gather(
            expanded,
            2,
            clipped_positions.unsqueeze(-1).expand(-1, -1, -1, dim_model),
        )
        return gathered, valid

    @staticmethod
    def _masked_attention_pool(
        query_proj: torch.Tensor,
        key_proj: torch.Tensor,
        value_proj: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = (query_proj.unsqueeze(2) * key_proj).sum(dim=-1) / math.sqrt(query_proj.shape[-1])
        scores = scores.masked_fill(~valid_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * valid_mask.to(dtype=attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (attn.unsqueeze(-1) * value_proj).sum(dim=2)

    def _summarize_window(
        self,
        query_hidden: torch.Tensor,
        plan_hidden: torch.Tensor,
        query_positions: torch.Tensor,
        context_lens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window_hidden, valid_mask = self._gather_plan_window_single(
            plan_hidden=plan_hidden,
            query_positions=query_positions,
            context_lens=context_lens,
        )
        query_valid_mask = valid_mask.any(dim=-1)

        coarse_query = self.coarse_q_proj(query_hidden)
        coarse_key = self.coarse_k_proj(window_hidden)
        coarse_value = self.coarse_v_proj(window_hidden)
        coarse_summary = self._masked_attention_pool(
            coarse_query,
            coarse_key,
            coarse_value,
            valid_mask,
        )
        coarse_summary = self.coarse_out_proj(coarse_summary)
        coarse_summary = torch.where(
            query_valid_mask.unsqueeze(-1),
            coarse_summary,
            torch.zeros_like(coarse_summary),
        )

        boundary_query = self.boundary_q_proj(query_hidden)
        boundary_key = self.boundary_k_proj(window_hidden)
        boundary_value = self.boundary_v_proj(window_hidden)
        boundary_summary = self._masked_attention_pool(
            boundary_query,
            boundary_key,
            boundary_value,
            valid_mask,
        )
        boundary_summary = self.boundary_out_proj(boundary_summary)
        boundary_summary = torch.where(
            query_valid_mask.unsqueeze(-1),
            boundary_summary,
            torch.zeros_like(boundary_summary),
        )
        return coarse_summary, boundary_summary, query_valid_mask

    def forward(
        self,
        query_hidden: torch.Tensor,
        plan_hidden: torch.Tensor,
        query_positions: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if query_positions.ndim != 3 or query_positions.shape[-1] != self.n_codebooks:
            raise ValueError(
                f"Expected query_positions to have shape [B, T, Q={self.n_codebooks}], got {tuple(query_positions.shape)}"
            )

        routed_hidden = []
        routing_lambdas = []
        valid_masks = []

        for q in range(self.n_codebooks):
            pos_q = query_positions[:, :, q]
            coarse_summary_q, boundary_summary_q, valid_mask_q = self._summarize_window(
                query_hidden=query_hidden,
                plan_hidden=plan_hidden,
                query_positions=pos_q,
                context_lens=context_lens,
            )

            if q == 0:
                gate_input = torch.cat([query_hidden, coarse_summary_q, boundary_summary_q], dim=-1)
                gate_input = self.q0_gate_norm(gate_input)
                lambda_q = torch.sigmoid(self.q0_gate_mlp(gate_input))
                route_q = (
                    lambda_q * self.coarse_route_proj_q0(coarse_summary_q)
                    + (1.0 - lambda_q) * self.boundary_route_projs[q](boundary_summary_q)
                )
            else:
                lambda_q = torch.zeros(
                    (*query_hidden.shape[:2], 1),
                    device=query_hidden.device,
                    dtype=query_hidden.dtype,
                )
                route_q = self.boundary_route_projs[q](boundary_summary_q)

            out_q = query_hidden + self.route_scale[q] * route_q
            out_q = torch.where(valid_mask_q.unsqueeze(-1), out_q, query_hidden)
            lambda_q = torch.where(
                valid_mask_q.unsqueeze(-1),
                lambda_q,
                torch.zeros_like(lambda_q),
            )

            routed_hidden.append(out_q)
            routing_lambdas.append(lambda_q.squeeze(-1))
            valid_masks.append(valid_mask_q)

        routed_hidden = torch.stack(routed_hidden, dim=1)  # [B, Q, T, D]
        routing_lambdas = torch.stack(routing_lambdas, dim=-1)  # [B, T, Q]
        routing_valid_mask = torch.stack(valid_masks, dim=-1).any(dim=-1)  # [B, T]

        metrics = {
            "routing_lambdas": routing_lambdas,
            "routing_valid_mask": routing_valid_mask,
            "route_scale_values": self.route_scale.detach(),
        }
        return routed_hidden, metrics

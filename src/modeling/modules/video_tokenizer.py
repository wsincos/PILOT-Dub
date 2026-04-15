
"""
Video Tokenizer 流程说明 (EMA-VQ):
--------------------------------------------------
1. 输入预处理 (Preprocessing):
   - 通过 Projector (Linear + GELU + LN) 将`lip feature`映射到码本隐藏空间 (z)。

2. 向量量化 (Vector Quantization):
   - 匹配：计算 z 与码本 (Embedding Weight) 中所有向量的 L2 距离，寻找最近邻索引 (token_ids)。
   - 替换：从码本中提取对应的向量，得到量化后的特征 (quantized/embeddings)。
   - 梯度直通 (STE): 前向传播使用 quantized,反向传播将梯度直接传给 z,解决离散操作不可导问题。

3. 参数更新机制 (Update Mechanism) [重点]:
   - Projector (可学习参数): 通过 Loss 产生的梯度进行更新 (梯度下降)。
   - Embedding (码本权重): 
     * 不接收梯度，不通过优化器更新。
     * 采用 EMA (指数移动平均) 逻辑：每轮计算选中该 code 的 z 的质心(平均中心),
       通过 .data.copy_() 直接覆盖更新权重，使其平滑地追随数据分布。

4. 输出项说明:
   - embeddings: 替换后的连续向量，供后续 Decoder 或 Transformer 使用。
   - token_ids: 离散的索引序列，用于压缩存储或作为生成式任务的标签。
   - loss: Commitment Loss,约束 Projector 产生的 z 不要偏离码本中心太远。
--------------------------------------------------
"""

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F



class _DepthwisePointwiseBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, padding: int):
        super().__init__()
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            dilation=dilation,
            padding=padding,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        residual = x
        y = x.transpose(1, 2)  # [B, C, T]
        y = self.dw(y)
        y = self.pw(y)
        y = y.transpose(1, 2)
        y = self.norm(y)
        y = self.act(y)
        return residual + y


class EMAVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        use_l2_norm: bool = True,
        random_restart: bool = True,
        restart_threshold: float = 1.0,
        restart_warmup_steps: int = 500,
        restart_interval: int = 100,
    ):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.use_l2_norm = use_l2_norm
        self.random_restart = random_restart
        self.restart_threshold = restart_threshold
        self.restart_warmup_steps = restart_warmup_steps
        self.restart_interval = restart_interval
        self.register_buffer("restart_step", torch.zeros((), dtype=torch.long))

        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)
        self.embedding.weight.requires_grad = False

        self.register_buffer("ema_cluster_size", torch.zeros(n_embed))
        self.register_buffer("ema_w", torch.zeros(n_embed, embed_dim))

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: [B, T, D]
        flat_z = z.reshape(-1, self.embed_dim)
        embed = self.embedding.weight
        if self.use_l2_norm:
            flat_z = F.normalize(flat_z, dim=-1)
            embed = F.normalize(embed, dim=-1)
        dist = torch.cdist(flat_z, embed, p=2)
        encoding_indices = torch.argmin(dist, dim=1)
        encodings = F.one_hot(encoding_indices, self.n_embed).type(flat_z.dtype)
        quantized = self.embedding(encoding_indices).view(z.shape)

        if mask is not None:
            flat_mask = mask.reshape(-1)
            if flat_mask.dtype != torch.bool:
                flat_mask = flat_mask.bool()
            valid_encodings = encodings[flat_mask]
            valid_z = flat_z[flat_mask]
        else:
            flat_mask = None
            valid_encodings = encodings
            valid_z = flat_z

        if self.training:
            if valid_encodings.numel() == 0:
                ema_cluster_size = self.ema_cluster_size
                ema_w = self.ema_w
            else:
                ema_cluster_size = self.ema_cluster_size * self.decay + (1.0 - self.decay) * valid_encodings.sum(0)
                dw = valid_encodings.t() @ valid_z
                ema_w = self.ema_w * self.decay + (1.0 - self.decay) * dw
            self.ema_cluster_size.copy_(ema_cluster_size)

            self.ema_w.copy_(ema_w)

            n = ema_cluster_size.sum()
            cluster_size = (ema_cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            self.embedding.weight.data.copy_(ema_w / cluster_size.unsqueeze(1))

            if self.random_restart and valid_z.numel() > 0:
                self.restart_step.add_(1)
                if (
                    self.restart_step.item() >= self.restart_warmup_steps
                    and (self.restart_step.item() - self.restart_warmup_steps) % self.restart_interval == 0
                ):
                    dead_codes = ema_cluster_size < self.restart_threshold
                    num_dead = int(dead_codes.sum().item())
                    if num_dead > 0:
                        rand_idx = torch.randint(0, valid_z.shape[0], (num_dead,), device=valid_z.device)
                        self.embedding.weight.data[dead_codes] = valid_z[rand_idx]

        if valid_z.numel() == 0:
            commitment_loss = z.new_zeros(())
        else:
            if self.use_l2_norm:
                quant_flat = quantized.reshape(-1, self.embed_dim)
                if flat_mask is not None:
                    quant_flat = quant_flat[flat_mask]
                quant_norm = F.normalize(quant_flat, dim=-1)
                commitment_loss = F.mse_loss(quant_norm.detach(), valid_z)
            else:
                if flat_mask is None:
                    commitment_loss = F.mse_loss(quantized.detach(), z)
                else:
                    valid_quantized = quantized.reshape(-1, self.embed_dim)[flat_mask]
                    commitment_loss = F.mse_loss(valid_quantized.detach(), valid_z)
        vq_loss = self.beta * commitment_loss

        quantized = z + (quantized - z).detach()
        return quantized, encoding_indices.view(z.shape[:-1]), vq_loss


class VideoTokenizerEMA(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        embed_dim: int = 2048,
        n_embed: int = 512,
        use_vq: bool = True,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        use_l2_norm: bool = True,
        random_restart: bool = True,
        restart_threshold: float = 1.0,
        restart_warmup_steps: int = 500,
        restart_interval: int = 100,
        temporal_layers: int = 2,
        temporal_kernel_size: int = 5,
        temporal_dilation: int = 1,
    ):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.temporal_layers = temporal_layers
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_dilation = temporal_dilation
        self.temporal_stack = self._build_temporal_stack(embed_dim)
        self.use_vq = use_vq
        self.vq = EMAVectorQuantizer(
            n_embed=n_embed,
            embed_dim=embed_dim,
            beta=beta,
            decay=decay,
            eps=eps,
            use_l2_norm=use_l2_norm,
            random_restart=random_restart,
            restart_threshold=restart_threshold,
            restart_warmup_steps=restart_warmup_steps,
            restart_interval=restart_interval,
        )

    def _build_temporal_stack(self, channels: int) -> nn.Module:
        if self.temporal_layers <= 0:
            return nn.Identity()
        layers = []
        for i in range(self.temporal_layers):
            dilation = self.temporal_dilation * (2 ** i)
            pad = (self.temporal_kernel_size - 1) // 2 * dilation
            layers.append(
                _DepthwisePointwiseBlock(
                    channels,
                    kernel_size=self.temporal_kernel_size,
                    dilation=dilation,
                    padding=pad,
                )
            )
        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z = self.project(x)
        z = self.temporal_stack(z)
        if not self.use_vq:
            return {
                "embeddings": z,
                "pre_vq_embeddings": z,
                "token_ids": None,
                "loss": z.new_zeros(()),
            }
        mask = None
        if lengths is not None:
            max_len = z.shape[1]
            mask = torch.arange(max_len, device=z.device)[None, :] < lengths[:, None]
        quantized, token_ids, vq_loss = self.vq(z, mask=mask)
        return {
            "embeddings": quantized,
            "pre_vq_embeddings": z,
            "token_ids": token_ids,
            "loss": vq_loss,
        }

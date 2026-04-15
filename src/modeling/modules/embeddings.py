import math
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F

class TokenEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model

        self.dropout = torch.nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.dim_model)

    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> torch.Tensor:
        return self.word_embeddings.weight[index : index + 1]

    def forward(self, x: torch.Tensor):
        X = self.word_embeddings(x)
        X = self.dropout(X)

        return X


class AudioEmbedding(nn.Module):
    def __init__(
        self,
        n_codebooks: int,
        vocab_size: int,
        dim_model: int,
        dropout: float = 0.0,
        n_special: int = 0,
    ):
        """
        AudioEmbedding handles multiple TokenEmbeddings for different codebooks.
        It serves as both a high-level module (via forward) and a container (via list-like access).

        Args:
            n_codebooks (int): Number of codebooks.
            vocab_size (int): Vocabulary size for each codebook (excluding special tokens).
            dim_model (int): Embedding dimension.
            dropout (float): Dropout probability.
            n_special (int): Number of special tokens to reserve in the vocabulary.
        """
        super().__init__()
        self.n_codebooks = n_codebooks
        # Calculate total vocab size including special tokens
        self.n_audio_tokens = [vocab_size + n_special] * n_codebooks 

        # Create a ModuleList of TokenEmbeddings for each codebook
        self.audio_embeddings = nn.ModuleList([
            TokenEmbedding(
                vocab_size=self.n_audio_tokens[k],
                dim_model=dim_model,
                dropout=dropout
            ) for k in range(n_codebooks)
        ])

    @property
    def audio_vocabs(self):
        """Returns the vocabulary sizes for all codebooks."""
        return self.n_audio_tokens
    
    
    def __len__(self):
        """Allows len(audio_embedding)"""
        return self.n_codebooks
    
    def __getitem__(self, idx):
        """Allows audio_embedding[k] access"""
        return self.audio_embeddings[idx]

    def __iter__(self):
        """Allows iteration: for emb in audio_embedding"""
        return iter(self.audio_embeddings)

    def forward(self, x: torch.Tensor, codebook_idx: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor. 
               Shape [B, K, T] for multi-codebook input.
               Shape [B, T] for single-codebook input (requires codebook_idx).
            codebook_idx: If set, processes x as a single codebook input using the specific embedding.
        Returns:
            [B, K, T, D] if input was [B, K, T]
            [B, T, D] if input was [B, T] with codebook_idx
        """
        # Case 1: Single Codebook (Specific access)
        if codebook_idx is not None:
            if not (0 <= codebook_idx < self.n_codebooks):
                raise ValueError(f"Invalid codebook index {codebook_idx}. Must be in range [0, {self.n_codebooks - 1}].")
            return self.audio_embeddings[codebook_idx](x)

        # Case 2: Multi Codebook (Standard forward pass)
        # Expected x shape: [B, K, T]
        if x.ndim == 3:
            B, K, T = x.shape
            assert K == self.n_codebooks, f"Input has {K} codebooks, model expects {self.n_codebooks}"

            outputs = []
            for k in range(self.n_codebooks):
                # Extract sequence for k-th codebook: [B, K, T] -> [B, T]
                # Pass through k-th embedding layer
                emb = self.audio_embeddings[k](x[:, k, :]) 
                outputs.append(emb)
            
            # Stack back to [B, K, T, D]
            return torch.stack(outputs, dim=1) 
        
        raise ValueError(f"Invalid input shape {x.shape}. Expected [B, K, T] or provide codebook_idx for [B, T].")


class MaskEmbedding(nn.Module):
    def __init__(self, max_n_spans: int, d_model: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(max_n_spans, d_model), requires_grad=True)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # indices shape: [B, ...]
        # output shape: [B, ..., D]
        return F.embedding(indices, self.embedding)
    
    def __getitem__(self, indices):
        return self.embedding[indices]


class LearnedToken(nn.Module):
    def __init__(self, dim_model: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, dim_model), requires_grad=True)

    def forward(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = self.embedding.device
        return self.embedding.to(device).expand(batch_size, -1).unsqueeze(1)


class LearnedTypeEmbedding(nn.Module):
    def __init__(self, dim_model: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, 1, dim_model), requires_grad=True)

    def forward(
        self,
        batch_size: int,
        seq_len: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = self.embedding.device
        return self.embedding.to(device).expand(batch_size, seq_len, -1)


class LengthBucketEmbedding(nn.Module):
    def __init__(self, dim_model: int, num_buckets: int = 32, bucket_size: int = 10):
        super().__init__()
        self.num_buckets = num_buckets
        self.bucket_size = bucket_size
        self.embedding = nn.Embedding(num_buckets, dim_model)

    def forward(self, lengths: torch.Tensor) -> torch.Tensor:
        bucket_ids = torch.clamp(lengths // self.bucket_size, max=self.num_buckets - 1)
        return self.embedding(bucket_ids).unsqueeze(1)


class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.x_scale = math.sqrt(dim_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 4000))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.dim_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(
                0, x.size(1), dtype=torch.float32
            ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if positions is None:
            self.extend_pe(x)
            output = x.unsqueeze(-1) if x.ndim == 2 else x
            output = output * self.x_scale + self.alpha * self.pe[:, : x.size(1)]
            return self.dropout(output)

        positions = positions.to(device=x.device, dtype=torch.long)
        max_pos = int(positions.max().item()) if positions.numel() > 0 else 0
        self.extend_pe(torch.tensor(0.0, device=x.device).expand(1, max_pos + 1))
        pos_embeddings = self.pe[0, positions]
        output = x * self.x_scale + self.alpha * pos_embeddings
        return self.dropout(output)


class LearnedPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        max_len: int = 5000,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.max_len = max_len

        self.position_embeddings = nn.Embedding(self.max_len, self.dim_model)
        self.dropout = torch.nn.Dropout(p=dropout)

    @property
    def weight(self) -> torch.Tensor:
        return self.position_embeddings.weight

    def embedding(self, index: int) -> torch.Tensor:
        return self.position_embeddings.weight[index : index + 1]

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if positions is None:
            positions = torch.arange(
                0, x.size(1), dtype=torch.long, device=x.device
            ).unsqueeze(0)
        else:
            positions = positions.to(device=x.device, dtype=torch.long)
        pos_embeddings = self.position_embeddings(positions)
        output = x + pos_embeddings
        return self.dropout(output)


class RoPEPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        base: int = 10000,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.base = base
        self.dropout = torch.nn.Dropout(p=dropout)

        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, dim_model, 2).float() / dim_model)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.size(-1) % 2 != 0:
            raise ValueError(f"RoPE requires even dim, got {x.size(-1)}")

        seq_len = x.size(1)
        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype).unsqueeze(0)
        else:
            positions = positions.to(device=x.device, dtype=self.inv_freq.dtype)
            if positions.ndim == 1:
                positions = positions.unsqueeze(0)

        sinusoid_inp = positions.unsqueeze(-1) * self.inv_freq
        sin = sinusoid_inp.sin()
        cos = sinusoid_inp.cos()

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)

        return self.dropout(x_rot)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple

class VoiceCraftHeads(nn.Module):
    def __init__(
        self,
        n_codebooks: int,
        d_model: int,
        audio_vocab_size: int,
        n_special: int = 0
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        target_vocab_size = audio_vocab_size + n_special
        
        # create a separate head for each codebook
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, audio_vocab_size // 2),
                nn.GELU(),
                nn.Linear(audio_vocab_size // 2, target_vocab_size)
            ) for _ in range(n_codebooks)
        ])

    def forward(self, x: torch.Tensor, codebook_idx: Optional[int] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Decoder output [B, T, D]
            codebook_idx: If set, returns logits for a specific codebook.
        Returns:
            If codebook_idx is None: [B, K, T, V] (Stacked logits for all codebooks)
            If codebook_idx is set:  [B, T, V] (Logits for specific codebook)
        """
        # Case 1: Specific codebook (e.g., during inference/generation loop)
        if codebook_idx is not None:
            if not (0 <= codebook_idx < self.n_codebooks):
                raise ValueError(f"Invalid codebook index {codebook_idx}")
            return self.heads[codebook_idx](x)

        # Case 2: All codebooks (e.g., during training)
        # x is [B, T, D], we apply each head to x
        outputs = []
        for head in self.heads:
            outputs.append(head(x)) # [B, T, V]
        
        # Stack them: [B, K, T, V]
        return torch.stack(outputs, dim=1)


    def __len__(self):
        return len(self.heads)

    def __getitem__(self, idx):
        return self.heads[idx]

    def __iter__(self):
        return iter(self.heads)


class TwoStepCodebookHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        repr_dim: int,
        target_vocab_size: int,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.input_proj = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.repr_proj = nn.Linear(hidden_dim, repr_dim)
        self.classifier = nn.Linear(repr_dim, target_vocab_size)

    def forward_with_repr(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.norm(x)
        x = self.act(self.input_proj(x))
        g = self.repr_proj(x)
        logits = self.classifier(g)
        return logits, g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_repr(x)
        return logits


class TwoStepVoiceCraftHeads(nn.Module):
    def __init__(
        self,
        n_codebooks: int,
        d_model: int,
        audio_vocab_size: int,
        hidden_dim: int,
        repr_dim: int,
        n_special: int = 0,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        target_vocab_size = audio_vocab_size + n_special
        self.heads = nn.ModuleList(
            [
                TwoStepCodebookHead(
                    d_model=d_model,
                    hidden_dim=hidden_dim,
                    repr_dim=repr_dim,
                    target_vocab_size=target_vocab_size,
                )
                for _ in range(n_codebooks)
            ]
        )

    def forward_with_reprs(
        self,
        x: Optional[torch.Tensor] = None,
        codebook_inputs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = []
        reprs = []
        if codebook_inputs is not None:
            if codebook_inputs.ndim != 4:
                raise ValueError(f"Expected codebook_inputs to have shape [B, K, T, D], got {codebook_inputs.shape}")
            if codebook_inputs.shape[1] != self.n_codebooks:
                raise ValueError(
                    f"Expected {self.n_codebooks} codebook inputs, got {codebook_inputs.shape[1]}"
                )
        elif x is None:
            raise ValueError("Either x or codebook_inputs must be provided.")

        for idx, head in enumerate(self.heads):
            curr_input = codebook_inputs[:, idx, :, :] if codebook_inputs is not None else x
            curr_logits, curr_repr = head.forward_with_repr(curr_input)
            logits.append(curr_logits)
            reprs.append(curr_repr)
        return torch.stack(logits, dim=1), torch.stack(reprs, dim=1)

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        codebook_idx: Optional[int] = None,
        codebook_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if codebook_idx is not None:
            if not (0 <= codebook_idx < self.n_codebooks):
                raise ValueError(f"Invalid codebook index {codebook_idx}")
            if codebook_inputs is not None:
                if codebook_inputs.ndim != 4:
                    raise ValueError(f"Expected codebook_inputs to have shape [B, K, T, D], got {codebook_inputs.shape}")
                return self.heads[codebook_idx](codebook_inputs[:, codebook_idx, :, :])
            if x is None:
                raise ValueError("x must be provided when codebook_inputs is None.")
            return self.heads[codebook_idx](x)
        logits, _ = self.forward_with_reprs(x=x, codebook_inputs=codebook_inputs)
        return logits

    def __len__(self):
        return len(self.heads)

    def __getitem__(self, idx):
        return self.heads[idx]

    def __iter__(self):
        return iter(self.heads)


class NormalizedMLPProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.normalize(x, dim=-1)


class CTCHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

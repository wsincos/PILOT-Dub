# import torch
# import torch.nn as nn
# from torchmetrics.classification import MulticlassAccuracy
# from typing import List, Union

# class CodebookTopKAccuracy(nn.ModuleList):
#     def __init__(self, n_codebooks: int, token_size: Union[int, List[int]], top_k: int = 10):
#         """
#         Args:
#             n_codebooks: codebook number.
#             token_size: number of tokens per codebook. If int, all codebooks have the same number of tokens.
#             top_k: calculate accuracy over top k predictions.
#         """
#         super().__init__(
#             [MulticlassAccuracy(
#                 num_classes=token_size if isinstance(token_size, int) else token_size[k],
#                 top_k=top_k,
#                 average="micro",
#                 multidim_average="global",
#                 ignore_index=None,
#             ) for k in range(n_codebooks)]
#         )
#     def forward(self, logits: List[torch.Tensor], targets: List[torch.Tensor]):
#         """
#         Batch calculate accuracy for all codebooks
#         """
#         return [self[k](logits[k], targets[k]) for k in range(len(self))]

import torch
import torch.nn as nn
from typing import List, Union

class SimpleTopKAccuracy(nn.Module):
    """
    A lightweight Top-K Accuracy Calculation Module.
    Replaces torchmetrics to avoid OOM caused by creating One-Hot tensors with large word lists.
    """
    def __init__(self, top_k: int = 1, ignore_index: int = -100):
        super().__init__()
        self.top_k = top_k
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: [Batch, Length, Vocab] or [N, Vocab]
            targets: [Batch, Length] or [N]
        """
        with torch.no_grad():
            # 展平以便统一处理
            if logits.dim() > 2:
                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.reshape(-1)

            # 创建有效 mask (忽略 padding)
            mask = targets != self.ignore_index
            valid_count = mask.sum()

            if valid_count == 0:
                return torch.tensor(0.0, device=logits.device)

            if self.top_k == 1:
                preds = logits.argmax(dim=-1)
                correct = (preds == targets) & mask
                return correct.sum() / valid_count
            else:
                _, preds = logits.topk(self.top_k, dim=-1) # [N, k]
                targets_expanded = targets.unsqueeze(-1)
                correct_matrix = preds.eq(targets_expanded) # [N, k]
                correct_row = correct_matrix.any(dim=-1) & mask
                return correct_row.sum() / valid_count

class CodebookTopKAccuracy(nn.ModuleList):
    def __init__(self, n_codebooks: int, token_size: Union[int, List[int]] = None, top_k: int = 10):
        """
        Args:
            n_codebooks: codebook number.
            top_k: calculate accuracy over top k predictions.
        """
        super().__init__(
            [SimpleTopKAccuracy(top_k=top_k, ignore_index=-100) for _ in range(n_codebooks)]
        )

    def forward(self, logits: List[torch.Tensor], targets: List[torch.Tensor]):
        """
        Batch calculate accuracy for all codebooks
        """
        return [self[k](logits[k], targets[k]) for k in range(len(self))]
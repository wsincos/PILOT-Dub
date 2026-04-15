import torch
import torch.nn as nn
import typing as tp

class VoiceCraftLoss(nn.Module):
    """
    Loss for VoiceCraft
    Calculating the CrossEntropy Loss of K codebooks and summing them up.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, model_output: tp.Dict[str, torch.Tensor]) -> tp.Tuple[torch.Tensor, dict]:
        """
        Args:
            model_output:
                - logits: [K, Total_N, Vocab]
                - targets: [K, Total_N]
        """
        logits = model_output['logits']
        targets = model_output['targets']
        
        total_loss = 0.0
        metrics = {}
        
        n_codebooks = logits.shape[0]
        
        for k in range(n_codebooks):
            loss_k = self.loss_fn(logits[k], targets[k])
            total_loss += loss_k
            metrics[f"ce_loss_{k}"] = loss_k.item()
            
        metrics["total_loss"] = total_loss.item()
        
        return total_loss, metrics
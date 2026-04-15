import torch
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """
    Args:
        logits: [B, Vocab]
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        filter_value: value to replace filtered logits with
        min_tokens_to_keep: minimum number of tokens to keep regardless of filtering
    Returns:
        filtered logits: [B, Vocab]
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk(logits, top_k) returns a tuple: (values, indices)
        # values: [B, top_k], indices: [B, top_k]
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None] # [B, 1]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    """
    Args:
        logits: [B, Vocab]
        top_k: int
        top_p: float
        temperature: float
    Returns:
        token: [B, 1]
    """
    # Temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    
    # Sample
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    return token
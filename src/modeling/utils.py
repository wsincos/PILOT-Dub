import torch

def load_ckpt_from_origin(ckpt_fn):
    ckpt = torch.load(ckpt_fn, map_location='cpu')

    # Support both legacy checkpoints with "model" and Lightning checkpoints with "state_dict".
    if "model" in ckpt:
        state_dict = ckpt.pop("model")
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        return ckpt

    # If all keys are prefixed with "model.", strip it for direct model loading.
    if state_dict and all(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        
        # 1. Audio Embedding
        # Old: audio_embedding.0. -> New: audio_embedding.audio_embeddings.0.
        if k.startswith("audio_embedding."):
            parts = k.split(".")
            # Check if the second part is a digit (0, 1, 2, 3...)
            if len(parts) > 1 and parts[1].isdigit():
                # Insert 'audio_embeddings'
                new_k = k.replace(f"audio_embedding.{parts[1]}", f"audio_embedding.audio_embeddings.{parts[1]}")
        
        # 2. Predict Layer (Heads)
        # Old: predict_layer.0. -> New: predict_layer.heads.0.
        elif k.startswith("predict_layer."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                new_k = k.replace(f"predict_layer.{parts[1]}", f"predict_layer.heads.{parts[1]}")
            elif len(parts) > 3 and parts[1] == "heads" and parts[2].isdigit():
                if parts[3] == "0":
                    new_k = k.replace(
                        f"predict_layer.heads.{parts[2]}.0",
                        f"predict_layer.heads.{parts[2]}.input_proj",
                    )

        # 3. Mask Embedding
        # Old: mask_embedding -> New: mask_embedding.embedding
        elif k == "mask_embedding":
            new_k = "mask_embedding.embedding"

        # 4. Special Tokens (if needed)
        elif k == "eog":
            new_k = "processor.eog"
        elif k == "eos":
            new_k = "processor.eos"
        new_state_dict[new_k] = v

    ckpt["state_dict"] = new_state_dict
    return ckpt

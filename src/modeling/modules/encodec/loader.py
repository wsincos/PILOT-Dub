"""Checkpoint loading utilities for EnCodec model."""

import logging
import typing as tp
from pathlib import Path
import torch
import omegaconf

from . import builders
from .model import CompressionModel

logger = logging.getLogger(__name__)

def model_from_checkpoint(
    checkpoint_path: tp.Union[Path, str],
    device: tp.Union[torch.device, str] = 'cpu'
):
    """
    Load EnCodec model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pth or .th)
        device: Device to load model on
        
    Returns:
        Loaded EncodecModel ready for inference
    """
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading compression model from: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location='cpu')
    
    if 'xp.cfg' not in state:
        raise ValueError(f"Invalid checkpoint format - missing 'xp.cfg': {checkpoint_path}")
    
    cfg = state['xp.cfg']
    cfg.device = device
    
    # Build model using config
    compression_model = builders.get_compression_model(cfg).to(device)
    
    # Load weights from best_state
    if 'best_state' not in state or not state['best_state']:
        raise ValueError(f"Invalid checkpoint - missing 'best_state': {checkpoint_path}")
    
    compression_model.load_state_dict(state['best_state']['model'])
    compression_model.eval()
    
    logger.info("Compression model loaded successfully!")
    return compression_model

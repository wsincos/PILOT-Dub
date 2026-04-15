"""
Lightweight standalone EnCodec module.
Extracted from audiocraft to provide minimal dependencies.
"""

from .model import CompressionModel, EncodecModel
from .loader import model_from_checkpoint

__all__ = ['CompressionModel', 'EncodecModel', 'model_from_checkpoint']

"""Quantization module."""
from .base import BaseQuantizer, DummyQuantizer, QuantizedResult
from .vq import ResidualVectorQuantizer

__all__ = [
    'BaseQuantizer', 
    'DummyQuantizer', 
    'QuantizedResult', 
    'ResidualVectorQuantizer'
]

# src/models/base.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import typing as tp

class GenerativeAudioModel(ABC, nn.Module):
    """
    Base class for all generative audio models (Base API)。
    """
    
    @abstractmethod
    def forward(self, batch: tp.Dict) -> tp.Dict:
        """Training forward pass."""
        ...

    @abstractmethod
    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Inference generation."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...
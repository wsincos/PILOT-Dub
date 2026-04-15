import torch.nn as nn


class AdaptingLayer(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
"""
This module combines our layernorm, mlp, and attention submodules into a transformer block module.


"""

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # save residual
        res = x
        return x

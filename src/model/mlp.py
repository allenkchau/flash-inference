"""
Given an input x:(batch_size, seq_len, hidden_size), our MLP does hidden -> expanded -> hidden

This block is responsible for most of the FLOPS in a transformer.

For the activation I decided to use GELU.
"""


import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # get dims
        input_dim = config.hidden_size
        expanded_dim = config.mlp_hidden_size

        # linear layer, activation, linear layer
        self.fc1 = nn.Linear(in_features=input_dim, out_features=expanded_dim, dtype=config.dtype, device=config.device)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(in_features=expanded_dim, out_features=input_dim, dtype=config.dtype, device=config.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.fc1(x)
        z2 = self.gelu(z1)
        y = self.fc2(z2)
        return y



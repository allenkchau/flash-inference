"""
We have an input tensor of shape: (batch_size, seq_len, hidden_dim).
LayerNorm normalizes the last dimension only.
"""

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        # these parameters are learned and applied after normalization so we can "undo" the normalization if necessary
        # there is a unique scale and shift parameter for every feature dimension
        # scale param
        self.gamma= nn.Parameter(torch.ones(config.hidden_size, dtype=config.dtype, device=config.device))
        # shift param
        self.beta = nn.Parameter(torch.zeros(config.hidden_size, dtype=config.dtype, device=config.device))

        # param for numerical stability (prevent division by 0)
        self.eps = 1e-5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # calculate mean and variance
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt((var + self.eps))

        # apply scale and shift
        y = self.gamma * x_hat + self.beta
        return y

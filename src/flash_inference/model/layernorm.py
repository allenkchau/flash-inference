import torch
import torch.nn as nn

"""
The entire point of LayerNorm is to stabilize the values flowing through the transformer so training can happen.
This prevents gradient explosion or vanishing basically.

The layer takes the activations, centers and scales it (mean=0 and va=1) across the feature dimension.
Then it applies learnable scale and shift params.

This is important:
LayerNorm makes each token's vector representation "well-conditioned" before passing to the next op in the transformer.
This is why modern LLMs utilize the idea of pre-layernorm.

Transformers care about relationships within a sequence, so normalizing across features aligns with the problem.
Also later in decoding, different examples could have different seq_len so batch norm just breaks.
"""

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()

        # unpack what we need from config
         = config.

        # layernorm formula
        # small constant added in denominator for numerical stability
        self.eps = 1e-5


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.shape

        # compute mean and variance statistics
        mean = 
        var = 

        return res

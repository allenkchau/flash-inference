from mpmath.math2 import EPS
import torch
import torch.nn as nn

"""
The entire point of LayerNorm is to stabilize the values flowing through the transformer so training can happen.
This prevents gradient explosion or vanishing basically.

The layer takes the activations, centers and scales it (mean=0 and var=1) across the feature dimension.
Then it applies learnable scale and shift params.

This is important:
LayerNorm makes each token's vector representation "well-conditioned" before passing to the next op in the transformer.
This is why modern LLMs utilize the idea of pre-layernorm.

Transformers care about relationships within a sequence, so normalizing across features aligns with the problem.
Also later in decoding, different examples could have different seq_len so batch norm just breaks.
"""

from flash_inference.configs.model_config import ModelConfig

class LayerNorm(nn.Module):
    def __init__(self, config: ModelConfig, eps: float = 1e-5):
        super().__init__()

        # layernorm formula
        # small constant added in denominator for numerical stability
        self.eps = eps

        # learned shift param
        self.beta = nn.Parameter(torch.zeros(config.model_dim, device=config.device, dtype=config.dtype))
        # learned scale param
        self.gamma = nn.Parameter(torch.ones(config.model_dim, device=config.device, dtype=config.dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.shape

        # compute mean and variance statistics over the model_dim dimension
        mean = torch.mean(x, dim=-1, keepdim=True)        # shape: batch_size, seq_len, 1; without keepdim the last dimension is removed and we have shape: batch_size, seq_len
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)        # in Layernorm we use population var and not sample var so we set unbiased to be False

        # we have self.eps to var and then sqrt because that way the denominator doesnt't shrink to 0; sqrt(self.eps) > self.eps so self.eps on the outside can still be very small
        res = (x - mean) / torch.sqrt(var + self.eps)

        # apply our learned params
        res = res * self.gamma + self.beta

        return res

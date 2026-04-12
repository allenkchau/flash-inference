import torch
import torch.nn as nn
from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.attention import MHAttention
from flash_inference.model.layernorm import LayerNorm
from flash_inference.model.mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # initialize modules
        self.mlp = MLP(config)
        self.attn = MHAttention(config)
        self.ln1 = LayerNorm(config)
        self.ln2 = LayerNorm(config)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply layernorm
        residual = x
        x = self.ln1(x)

        # apply attention
        x = self.attn(x)

        # add residual
        x = x + residual

        residual = x
        x = self.ln2(x)

        # apply MLP
        x = self.mlp(x)
        x = x + residual
        
        return x




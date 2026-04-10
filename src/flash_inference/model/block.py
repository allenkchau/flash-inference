import torch
import torch.nn as nn
from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.attention import MHAttention

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        MHAttention()

    def forward(x: torch.Tensor) -> torch.Tensor:
        # apply attention

        # apply MLP

        

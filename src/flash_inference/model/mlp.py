import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.activations import build_activation


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # our linear layers in the FFN
        # modern transformers don't usually have a bias term
        self.W1 = nn.Linear(in_features=config.model_dim, out_features=config.mlp_hidden_size, bias=config.mlp_bias, device=config.device, dtype=config.dtype)
        self.W2 = nn.Linear(in_features=config.mlp_hidden_size, out_features=config.model_dim, bias=config.mlp_bias, device=config.device, dtype=config.dtype)

        # activation
        self.activation = build_activation(config.mlp_activation)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.activation(self.W1(x)))

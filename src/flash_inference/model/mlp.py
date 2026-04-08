import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig

class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # our linear layers in the FFN
        # modern transformers don't usually have a bias term
        self.W1 = nn.Linear(config.mlp_hidden_size, bias=False, device=config.device, dtype=config.dtype)
        self.W2 = nn.Linear(config.mlp_hidden_size, bias=False, device=config.device, dtype=config.dtype)

        # activation
        self.gelu = nn.GELU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # run the first linear layer
        y = self.W1 @ x

        y = self.gelu(y)

        # run the second linear layer
        y = self.W2 @ y

        return y

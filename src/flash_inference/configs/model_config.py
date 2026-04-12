"""
Class for the transformer config
"""

import torch
from dataclasses import dataclass

from flash_inference.model.activations import Activation

# since a config just hold data and not behavior we can use dataclass
# we don't want our config instance to mutate after we make it
@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    # this is used in our embedding matrix
    vocab_size: int

    # number of attention heads
    num_heads: int

    # width of the token representation flowing through the transformer
    model_dim: int

    # maximum sequence length the model is built to handle
    max_seq_len: int

    # bias term for mlp layer
    bias: bool

    # activation for mlp layer
    mlp_activation: Activation

    weight_tying: bool

    device: torch.device
    dtype: torch.dtype


    # post init is where we enforce invariants and validate inputs; properties are better for computing derived values
    def __post_init__(self):
        
        # validate integer fields
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        # validate dtype and device
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if not isinstance(self.device, torch.device):
            raise TypeError("device must be a torch.device")

        # we want the model to split the total embedding dimension equally among multiple parallel attention heads
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                "model_dim must be divisible by num_heads"
            )

    # properties look like data to the user but are internally computed
    # in general, we should use properties when the value can be derived from other values
    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads

    @property
    def mlp_hidden_size(self) -> int:
        return self.model_dim * 4


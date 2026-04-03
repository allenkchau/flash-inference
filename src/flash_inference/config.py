"""
Class for the transformer config
"""

import torch
from dataclasses import dataclass

# since a config just hold data and not behavior we can use dataclass
# we don't want our config instance to mutate after we make it
@dataclass(frozen=True)
class Config:
    num_layers: int
    # this is used in our embedding matrix
    vocab_size: int

    # number of attention heads
    num_heads: int

    # width of the token representation flowing through the transformer
    hidden_size: int

    # during decoding, we will have up to max_seq_len tokens in the output including the prompt
    max_seq_len: int

    device: torch.device
    dtype: torch.dtype


    # 
    def __post_init__(self):
        # we want the model to split the total embedding dimension equally among multiple parallel attention heads
        assert self.hidden_size % self.num_heads == 0, "Token embedding dim should divide evenly with number of attention heads"

    # properties look like data to the user but are internally computed
    # in general, we should use properties when the value can be derived fro other values
    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads


from dataclasses import dataclass
import torch

@dataclass(frozen=True)     # config is immutable after creation
class Config:
    # model size and structure
    # number of tokens the model can represent
    vocab_size: int               
    #   
    hidden_size: int
    # number of seq transformer blocks
    num_layers: int
    # number of parallel attention heads/layer
    num_heads: int

    # for sequences and inference later
    max_seq_len: int
    max_batch_size: int

    # how model runs
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, "Hidden size dimension doesn't divide into num heads dimension evenly!"

    # define properties that are computed on the fly
    @property                   
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def mlp_hidden_size(self) -> int:
        return self.hidden_size * 4


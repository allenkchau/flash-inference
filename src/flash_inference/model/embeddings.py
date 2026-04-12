import torch
import torch.nn as nn

"""
This is where we go from tokens to vectors with semantic and position info.
These vectors are ready to be fed to the transformer model.
"""

from flash_inference.configs.model_config import ModelConfig

class Embeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # unpack what we need from the config
        self.max_seq_len = config.max_seq_len

        # token embeddings for semantic meaning
        self.tok_embedding_table = nn.Embedding(num_embeddings=config.vocab_size, 
                                            embedding_dim=config.model_dim, 
                                            device=config.device, 
                                            dtype=config.dtype)

        # positional embeddings for spatial context
        self.pos_embedding_table = nn.Embedding(num_embeddings=config.max_seq_len, 
                                            embedding_dim=config.model_dim, 
                                            device=config.device, 
                                            dtype=config.dtype)

    # the first part of the transformer is the embedding layer
    # we take in the token ids here and return embedding vectors

    # the input_ids have shape (batch_size, seq_len)
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # get dimensions from input ids
        _, seq_len = input_ids.shape

        # make sure seq_len doesn't exceed max_seq_len
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
            )

        # we also should validate the dtype of input_ids since embeddings expect an int
        if input_ids.dtype != torch.long:
            raise ValueError(
                f"input_ids has dtype {input_ids.dtype} instead of torch.long"
            )

        tok_embeddings = self.tok_embedding_table(input_ids)     # shape: (batch_size, seq_len, model_dim)

        positions = torch.arange(seq_len, device=input_ids.device)       # shape: (seq_len)
        pos_embeddings = self.pos_embedding_table(positions)        # shape: (seq_len, model_dim)

        # combine token embeddings with learned positional embeddings; this works thanks to broadcasting
        embeddings = tok_embeddings + pos_embeddings.unsqueeze(0)       # shape: (batch_size, seq_len, model_dim)

        return embeddings

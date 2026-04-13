"""
Ultimately a transformer is just many transformer block stacked together.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.block import TransformerBlock
from flash_inference.model.embeddings import Embeddings
from flash_inference.model.layernorm import LayerNorm


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # store config
        self.config = config

        self.weight_tying = config.weight_tying

        # embeddings
        self.embeddings = Embeddings(config)

        # stack of transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

        # final layer norm
        # after the many residual additions in the blocks, before we turn hidden states -> logits, we want to make sure activations are well-scaled
        self.ln = LayerNorm(config)

        # final linear layer: converts model_dim -> vocab size
        # check if we use weight tying
        # we don't need a bias term because usually there is not such term in the output head
        self.output = nn.Linear(
            in_features=config.model_dim,
            out_features=config.vocab_size,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )
        if self.weight_tying:
            self.output.weight = self.embeddings.tok_embedding_table.weight


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # transform input token IDs to embeddings
        x = self.embeddings(input_ids)

        # run through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln(x)

        # just some notes here
        # at this point x has shape: batch_size, seq_len, model_dim
        # it's helpful to think of each slice x[b, t, :] as the model's contextualized represenation of token t after seeing all tokens up to t

        logits = self.output(x)

        # now shape is: batch_size, seq_len, vocab_size
        # logits[b, t, :] represents how likely each vocab token is to be the next token after position t

        return logits

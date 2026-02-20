"""
This module combines our layernorm, mlp, and attention submodules into a transformer block module.

Input: (B, T, H)
Output: (B, T, H)

I was initially going to remove return_kv because I thought every time we read from the cache we would need to update it like in standard decode.
So return_kv would always be true when we pass in a valid kv_cache right?

However, I realized that there are legitimate cases where we would want to read from the cache and not update it.
This would include cases like:
- beam search (we run attention using the same cached prefix but we don't want to append to the cache until we decide which branch survives)
- speculative decoding (a smaller model proposes several potential tokens which have to be verified by a bigger model; if rejected, rollback occurs)
"""

from src.model.attention import MHAttention
from src.model.layernorm import LayerNorm
from src.model.mlp import MLP
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1: LayerNorm = LayerNorm(config)
        self.attn: MHAttention = MHAttention(config)
        self.ln2: LayerNorm = LayerNorm(config)
        self.mlp: MLP = MLP(config)

    def forward(self, x: torch.Tensor, kv_cache=None, layer_idx=None, return_kv=False):
        # save residual
        res = x

        # normalize
        h = self.ln1(x)

        # attention
        if kv_cache is None:
            # baseline path (no cache)
            h = self.attn(h)
            K_out = V_out = None
        else:
            # cached path (prefill or decode determined inside attention)
            if return_kv:
                h, K_out, V_out = self.attn(h, kv_cache=kv_cache, layer_idx=layer_idx, return_kv=True)
            else:
                h = self.attn(h, kv_cache=kv_cache, layer_idx=layer_idx, return_kv=False)
                K_out = V_out = None

        # add residual
        x = res + h

        # repeat for MLP
        res = x
        h = self.ln2(x)
        h = self.mlp(h)
        x = res + h

        if return_kv:
            return (x, K_out, V_out)
        return x



class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # get config params
        self.hidden_size = config.hidden_size
        self.max_seq_len = config.max_seq_len
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size

        # let's also store the config itself so KVCache can access it if necessary
        self.config = config

        # transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(self.num_layers)])

        # token embedding (B, T) -> (B, T, H)
        self.tok_embed = nn.Embedding(self.vocab_size, self.hidden_size, dtype=config.dtype, device=config.device)

        # learned absolute positional embeddings
        self.pos_embed = nn.Embedding(self.max_seq_len, self.hidden_size, dtype=config.dtype, device=config.device)

        # final layer norm
        self.lnf = LayerNorm(config)

        # output head to produce logits (I'm not implementing weight tying for now)
        self.lm_head = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size, dtype=config.dtype, device=config.device)


    def forward(self, input_ids: torch.Tensor,  kv_cache=None, return_kv=False, position_ids=None):

        if return_kv:
            kv_updates = []

        B, T = input_ids.shape
        assert T <= self.max_seq_len, "Input length is greater than max_seq_len"
        assert input_ids.dtype == torch.long

        device = input_ids.device

        # token embeddings
        x = self.tok_embed(input_ids)

        # positional embeddings
        if position_ids is None:
            # baseline or prefill
            position_ids = torch.arange(T, device=device, dtype=torch.long)  # (T,)
            pos = self.pos_embed(position_ids).unsqueeze(0)                 # (1, T, H) -> broadcasts over B
        else:
            # ensure correct shape (B, T)
            assert position_ids.shape == input_ids.shape, "position_ids must have shape (B, T)"
            assert position_ids.dtype == torch.long, "position_ids must be torch.long"
            assert position_ids.device == device, "position_ids must be on same device as input_ids"
            pos = self.pos_embed(position_ids)                              # (B, T, H)

        x = x + pos                                 # (B, T, H)

        # pass through transformer blocks
        for i, block in enumerate(self.blocks):
            if return_kv:
                x, K, V = block(x, kv_cache=kv_cache, layer_idx=i, return_kv=return_kv)
                kv_updates.append((K, V))
            else:
                x = block(x, kv_cache=kv_cache, layer_idx=i, return_kv=return_kv)

        # apply last layernorm and output head
        x = self.lnf(x)
        logits = self.lm_head(x)

        if return_kv:
            return (logits, kv_updates)
        return logits

"""
We have an input tensor of shape: (batch_size, seq_len, hidden_dim).
Our output tensor should be the same shape.

We have 4 linear projections in total.
"""

import math
import torch
import torch.nn as nn

class MHAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # get dims
        self.hidden_dim = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.max_seq_len = config.max_seq_len

        # key, query, value matrices
        self.Wq = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, dtype=config.dtype, device=config.device)
        self.Wk = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, dtype=config.dtype, device=config.device)
        self.Wv = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, dtype=config.dtype, device=config.device)
        self.Wo = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, dtype=config.dtype, device=config.device)

        # causal mask with shape (1, 1, max_seq_len, max_seq_len)
        mask = torch.triu(
            torch.full((self.max_seq_len, self.max_seq_len), float("-inf"), dtype=torch.float32, device=config.device),
            diagonal=1,
        )
        mask = mask.unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, kv_cache=None, layer_idx=None, return_kv=False):
        B, T, _ = x.shape

        # helper for baseline and prefill
        def full_attention(x_full: torch.Tensor):
            Q = self.Wq(x_full)
            K = self.Wk(x_full)
            V = self.Wv(x_full)

            new_shape = (B, T, self.num_heads, self.head_dim)
            Q = Q.reshape(new_shape).transpose(1, 2)  # (B, Nh, T, Hd)
            K = K.reshape(new_shape).transpose(1, 2)  # (B, Nh, T, Hd)
            V = V.reshape(new_shape).transpose(1, 2)  # (B, Nh, T, Hd)

            scores = Q @ K.transpose(-2, -1)          # (B, Nh, T, T)
            scores = scores / math.sqrt(self.head_dim)

            assert T <= self.max_seq_len
            scores = scores + self.causal_mask[:, :, :T, :T]  # (B, Nh, T, T)

            probs = torch.softmax(scores, dim=-1)
            context = probs @ V                               # (B, Nh, T, Hd)

            merged = context.transpose(1, 2).reshape(B, T, self.hidden_dim)  # (B, T, H)
            y = self.Wo(merged)
            return y, K, V

        # baseline
        if kv_cache is None:
            y, _, _ = full_attention(x)
            return y

        
        # prefill
        if kv_cache.cur_len == 0:
            y, K, V = full_attention(x)
            if return_kv:
                return y, K, V
            return y

        # decode
        assert T == 1
        assert layer_idx is not None, "layer_idx required for decode with kv_cache"
        assert kv_cache.cur_len < self.max_seq_len, "KV cache is full"

        # project only the new token
        Q_new = self.Wq(x)  # (B, 1, H)
        K_new = self.Wk(x)  # (B, 1, H)
        V_new = self.Wv(x)  # (B, 1, H)

        new_shape = (B, 1, self.num_heads, self.head_dim)
        Q_new = Q_new.reshape(new_shape).transpose(1, 2)  # (B, Nh, 1, Hd)
        K_new = K_new.reshape(new_shape).transpose(1, 2)  # (B, Nh, 1, Hd)
        V_new = V_new.reshape(new_shape).transpose(1, 2)  # (B, Nh, 1, Hd)

        # read cached K/V up to cur_len 
        K_cached, V_cached = kv_cache.get_kv(layer_idx)    # (B, Nh, cur_len, Hd)

        # form totals for attention (cached + new)
        K_total = torch.cat([K_cached, K_new], dim=2)      # (B, Nh, cur_len+1, Hd)
        V_total = torch.cat([V_cached, V_new], dim=2)      # (B, Nh, cur_len+1, Hd)

        # query is last position, so causal mask not needed
        scores = Q_new @ K_total.transpose(-2, -1)         # (B, Nh, 1, cur_len+1)
        scores = scores / math.sqrt(self.head_dim)

        probs = torch.softmax(scores, dim=-1)
        context = probs @ V_total                          # (B, Nh, 1, Hd)

        merged = context.transpose(1, 2).reshape(B, 1, self.hidden_dim)  # (B, 1, H)
        y = self.Wo(merged)

        if return_kv:
            # return new K/V only because runtime will take care of append and bump cur_len once per token
            return y, K_new, V_new
        return y

       

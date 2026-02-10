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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # apply our weight matrices
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # reshape into heads
        new_shape = (*Q.shape[:-1], self.num_heads, self.head_dim)
        Q = torch.reshape(Q, new_shape)
        K = torch.reshape(K, new_shape)
        V = torch.reshape(V, new_shape)

        # transpose for attention math (right now we have B, T, H, D) but we don't want to mix up the attention heads
        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)
        attn_scores = Q @ K.transpose(-2, -1)

        # divide by sqrt of head dim (scale for stability)
        attn_scores /= math.sqrt(self.head_dim)

        # apply causal mask
        # we have to slice the mask up to our input x because our mask in the register buffer is up to max_seq_len so it may not be aligned with x
        assert T <= self.max_seq_len
        attn_scores = attn_scores + self.causal_mask[:, :, :T, :T]

        # apply softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # weighted sum of values
        context = attn_probs @ V

        # merge attention heads
        context = context.transpose(-3, -2)                              # switch back num heads and seq_len
        merged = torch.flatten(context, start_dim=2, end_dim=3)        # combine the attention heads together

        # output projection
        y = self.Wo(merged)
        return y

import math
import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig

class MHAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.model_dim = config.model_dim

        # K, V, and Q matrices 
        self.Wq = nn.Linear(in_features=config.model_dim, out_features=config.model_dim, device=config.device, dtype=config.dtype, bias=config.bias)
        self.Wk = nn.Linear(in_features=config.model_dim, out_features=config.model_dim, device=config.device, dtype=config.dtype, bias=config.bias)
        self.Wv = nn.Linear(in_features=config.model_dim, out_features=config.model_dim, device=config.device, dtype=config.dtype, bias=config.bias)

        # output matrix that takes concatenated result from all attention heads and captures the info to send to FFN
        self.Wo = nn.Linear(in_features=config.model_dim, out_features=config.model_dim, device=config.device, dtype=config.dtype, bias=config.bias)

        # causal mask (lower triangular)
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, device=config.device, dtype=torch.bool))
        self.register_buffer("attn_mask", mask)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.shape

        # get the K, V, Q projections; all still have shape: batch_size, seq_len, model_dim
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # we know model_dim = num_heads * head_dim; right now each token vector is a concatenation of all attn heads
        # we want to split the heads b/c this expands the model's ability to focus on different positions in the text
        # reshape the matrices
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # after reshaping we need to permute because we want to compute attn within each head so each head should behave like an independent batch
        # think about it like we go from 1 batch of size B to B * num_heads independent mini-batches; shape: batch_size, num_heads, seq_len, head_dim
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # calculate attention scores
        # shape is now batch_size, num_heads, seq_len, seq_len; rows are query positions and cols are key positions
        attn_scores = Q @ K.transpose(-2, -1)

        # scale attn scores
        scaled_attn_scores = attn_scores / (math.sqrt(self.head_dim))

        # apply our attn mask
        # our mask is max_seq_len by max_seq_len but we only want the seq_len by seq_len prefix
        mask = self.attn_mask[:seq_len, :seq_len]

        # set everything that is False (not valid position) in the mask to negative inf
        masked_attn_scores = scaled_attn_scores.masked_fill(mask == False, float("-inf"))

        # take softmax
        # for a fixed query token i, we want a probability distribution over all keys j so we normalize over all columns dim=-1
        # we divide by the square root of the head dim to prevent the dot products from growing too large in high dimensional spaces
        softmax_res = torch.softmax(masked_attn_scores, dim=-1)

        # shape: batch_size, num_heads, seq_len, head_dim
        weighted_sum = softmax_res @ V

        # transpose so seq_len becomes the 1st dim and then recombine the attn heads
        # shape: batch_size, seq_len, num_heads, head_dim
        weighted_sum = weighted_sum.permute(0, 2, 1, 3).contiguous()
        # shape: batch_size, seq_len, model_dim
        weighted_sum = weighted_sum.view(batch_size, seq_len, self.model_dim)

        # this info is just regarding the two steps above
        # permute changes how PyTorch interprets the strides so after the operation the tensor is usually non-contiguous
        # .contiguous() creates a new tensor in memory with the same values but laid out contiguously
        # reshape automatically calls contiguous if needed but view requires contiguous memory and is pretty fast

        # the two steps above are the same as this code; I went with the first approach just because it is more explicit
        #weighted_sum = weighted_sum.reshape(batch_size, seq_len, self.model_dim)

        # apply Wo
        res = self.Wo(weighted_sum)

        return res
        


# I plan to implement other version of attention below

# # all queries share a single K and V vector for each layer
# class MQAttention:
#     def __init__(self):
#         pass

# # queries are grouped and each group share a K and V vector for each layer
# class GQAttention:
#     def __init__(self):
#         pass

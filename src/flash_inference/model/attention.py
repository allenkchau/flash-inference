import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig

class MHAttention:
    def __init__(self, config: ModelConfig):

        # K, V, and Q matrices 
        self.Wq = nn.Linear()
        self.Wk = 
        self.Wv = 

        # attention mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.shape
        self.Wq()
        self.Wk()
        self.Wv()
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

"""
Ultimately a transformer is just many transformer block stacked together.
"""
import torch
import torch.nn as nn
from flash_inference.configs.model_config import ModelConfig


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig)

from enum import Enum
import torch.nn as nn

class Activation(Enum):
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"

def build_activation(activation: Activation) -> nn.Module:
    if activation == Activation.GELU:
        return nn.GELU()
    if activation == Activation.RELU:
        return nn.ReLU()
    if activation == Activation.SILU:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {activation}")

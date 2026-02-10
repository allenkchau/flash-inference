"""
Test functionality of model componenets

Usage:
python -m pytest tests/test_model.py -v

For print statements
python -m pytest -s tests/test_model.py -v
"""

from src.model.config import Config
from src.model.layernorm import LayerNorm
from src.model.mlp import MLP
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config(
        vocab_size=1024,
        hidden_size=1024,
        num_layers=12,
        num_heads=8,
        max_seq_len=10,
        max_batch_size=1,
        dtype=torch.float32,
        device=device
    )

def test_layernorm():
    module = LayerNorm(config=config)
    module.eval()

    # batch of 1, seq_len=4
    x = torch.ones(1, 4, config.hidden_size, dtype=config.dtype, device=config.device)
    output = module(x)

    # add dimensions to beta and broadcast across batch and seq
    print(module.beta)
    print(module.beta.view(1, 1, -1))
    expected = module.beta.view(1, 1, -1).expand_as(output)
    print(expected)

    assert x.shape == output.shape, "Shape mismatch between input and output for LayerNorm"
    assert x.device == output.device, "Device mismatch between input and output for LayerNorm"
    torch.testing.assert_close(output, expected, msg="Accuracy error in LayerNorm")


def test_mlp():
    module = MLP(config=config)
    module.eval()

    # batch of 1, seq_len=4
    x = torch.ones(1, 4, config.hidden_size, dtype=config.dtype, device=config.device)
    output = module(x)

    assert x.shape == output.shape, "Shape mismatch between input and output for LayerNorm"
    assert x.device == output.device, "Device mismatch between input and output for LayerNorm"


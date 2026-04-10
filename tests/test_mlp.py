import pytest
import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.activations import Activation
from flash_inference.model.mlp import MLP


def _make_config(activation: Activation, bias: bool, model_dim: int = 16) -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        vocab_size=128,
        num_heads=4,
        model_dim=model_dim,
        max_seq_len=32,
        bias=bias,
        mlp_activation=activation,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _reference_activation(activation: Activation) -> nn.Module:
    if activation == Activation.GELU:
        return nn.GELU()
    if activation == Activation.RELU:
        return nn.ReLU()
    if activation == Activation.SILU:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {activation}")


@pytest.mark.parametrize("activation", [Activation.GELU, Activation.RELU, Activation.SILU])
@pytest.mark.parametrize("bias", [False, True])
def test_mlp_forward_matches_reference(activation: Activation, bias: bool):
    torch.manual_seed(0)
    config = _make_config(activation=activation, bias=bias)

    custom_mlp = MLP(config)
    ref_mlp = nn.Sequential(
        nn.Linear(config.model_dim, config.mlp_hidden_size, bias=bias),
        _reference_activation(activation),
        nn.Linear(config.mlp_hidden_size, config.model_dim, bias=bias),
    )

    with torch.no_grad():
        ref_mlp[0].weight.copy_(custom_mlp.W1.weight)
        if bias:
            ref_mlp[0].bias.copy_(custom_mlp.W1.bias)
        ref_mlp[2].weight.copy_(custom_mlp.W2.weight)
        if bias:
            ref_mlp[2].bias.copy_(custom_mlp.W2.bias)

    x = torch.randn(3, 5, config.model_dim, dtype=config.dtype, device=config.device)
    y_custom = custom_mlp(x)
    y_ref = ref_mlp(x)

    torch.testing.assert_close(y_custom, y_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("activation", [Activation.GELU, Activation.RELU, Activation.SILU])
@pytest.mark.parametrize("bias", [False, True])
def test_mlp_backward_matches_reference(activation: Activation, bias: bool):
    torch.manual_seed(1)
    config = _make_config(activation=activation, bias=bias)

    custom_mlp = MLP(config)
    ref_mlp = nn.Sequential(
        nn.Linear(config.model_dim, config.mlp_hidden_size, bias=bias),
        _reference_activation(activation),
        nn.Linear(config.mlp_hidden_size, config.model_dim, bias=bias),
    )

    with torch.no_grad():
        ref_mlp[0].weight.copy_(custom_mlp.W1.weight)
        if bias:
            ref_mlp[0].bias.copy_(custom_mlp.W1.bias)
        ref_mlp[2].weight.copy_(custom_mlp.W2.weight)
        if bias:
            ref_mlp[2].bias.copy_(custom_mlp.W2.bias)

    x_custom = torch.randn(2, 4, config.model_dim, dtype=config.dtype, device=config.device, requires_grad=True)
    x_ref = x_custom.detach().clone().requires_grad_(True)
    grad_out = torch.randn_like(x_custom)

    y_custom = custom_mlp(x_custom)
    y_ref = ref_mlp(x_ref)

    y_custom.backward(grad_out)
    y_ref.backward(grad_out)

    torch.testing.assert_close(y_custom, y_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(x_custom.grad, x_ref.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(custom_mlp.W1.weight.grad, ref_mlp[0].weight.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(custom_mlp.W2.weight.grad, ref_mlp[2].weight.grad, rtol=1e-5, atol=1e-6)

    if bias:
        torch.testing.assert_close(custom_mlp.W1.bias.grad, ref_mlp[0].bias.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(custom_mlp.W2.bias.grad, ref_mlp[2].bias.grad, rtol=1e-5, atol=1e-6)

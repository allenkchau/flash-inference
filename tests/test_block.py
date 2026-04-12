import pytest
import torch

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.activations import Activation
from flash_inference.model.attention import MHAttention
from flash_inference.model.block import TransformerBlock
from flash_inference.model.layernorm import LayerNorm
from flash_inference.model.mlp import MLP


def _make_config(
    bias: bool,
    activation: Activation = Activation.GELU,
    model_dim: int = 16,
    num_heads: int = 4,
    max_seq_len: int = 32,
    weight_tying: bool = False,
) -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        vocab_size=128,
        num_heads=num_heads,
        model_dim=model_dim,
        max_seq_len=max_seq_len,
        bias=bias,
        mlp_activation=activation,
        weight_tying=weight_tying,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _manual_pre_ln_block_forward(block: TransformerBlock, x: torch.Tensor) -> torch.Tensor:
    h = x + block.attn(block.ln1(x))
    return h + block.mlp(block.ln2(h))


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", [Activation.GELU, Activation.RELU, Activation.SILU])
def test_block_initializes_expected_submodules(bias: bool, activation: Activation):
    config = _make_config(bias=bias, activation=activation)
    block = TransformerBlock(config)

    assert isinstance(block.ln1, LayerNorm)
    assert isinstance(block.attn, MHAttention)
    assert isinstance(block.ln2, LayerNorm)
    assert isinstance(block.mlp, MLP)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", [Activation.GELU, Activation.RELU, Activation.SILU])
def test_block_forward_preserves_shape_and_dtype(bias: bool, activation: Activation):
    torch.manual_seed(0)
    config = _make_config(bias=bias, activation=activation, model_dim=32, num_heads=8, max_seq_len=64)
    block = TransformerBlock(config)

    x = torch.randn(3, 11, config.model_dim, dtype=config.dtype, device=config.device)
    y = block(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", [Activation.GELU, Activation.RELU, Activation.SILU])
def test_block_forward_matches_manual_pre_layernorm_formula(bias: bool, activation: Activation):
    torch.manual_seed(1)
    config = _make_config(bias=bias, activation=activation, model_dim=16, num_heads=4, max_seq_len=64)
    block = TransformerBlock(config)

    x = torch.randn(2, 7, config.model_dim, dtype=config.dtype, device=config.device)

    y_block = block(x)
    y_manual = _manual_pre_ln_block_forward(block, x)

    torch.testing.assert_close(y_block, y_manual, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", [Activation.GELU, Activation.RELU, Activation.SILU])
def test_block_backward_matches_manual_pre_layernorm_formula(bias: bool, activation: Activation):
    torch.manual_seed(2)
    config = _make_config(bias=bias, activation=activation, model_dim=16, num_heads=4, max_seq_len=64)
    block = TransformerBlock(config)

    x_block = torch.randn(2, 6, config.model_dim, dtype=config.dtype, device=config.device, requires_grad=True)
    x_manual = x_block.detach().clone().requires_grad_(True)
    grad_out = torch.randn_like(x_block)

    y_block = block(x_block)
    y_block.backward(grad_out)
    x_block_grad = x_block.grad.detach().clone()
    param_grads_block = {
        name: p.grad.detach().clone()
        for name, p in block.named_parameters()
    }

    block.zero_grad(set_to_none=True)

    y_manual = _manual_pre_ln_block_forward(block, x_manual)
    y_manual.backward(grad_out)
    x_manual_grad = x_manual.grad.detach().clone()
    param_grads_manual = {
        name: p.grad.detach().clone()
        for name, p in block.named_parameters()
    }

    torch.testing.assert_close(y_block, y_manual, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(x_block_grad, x_manual_grad, rtol=1e-5, atol=1e-6)

    assert param_grads_block.keys() == param_grads_manual.keys()
    for name in param_grads_block:
        torch.testing.assert_close(param_grads_block[name], param_grads_manual[name], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", [Activation.GELU, Activation.RELU, Activation.SILU])
def test_block_is_causal_no_future_token_leakage(bias: bool, activation: Activation):
    torch.manual_seed(3)
    config = _make_config(bias=bias, activation=activation, model_dim=16, num_heads=4, max_seq_len=64)
    block = TransformerBlock(config)

    x = torch.randn(2, 8, config.model_dim, dtype=config.dtype, device=config.device)
    x_modified = x.clone()
    x_modified[:, -1, :] = x_modified[:, -1, :] + 100.0

    y = block(x)
    y_modified = block(x_modified)

    torch.testing.assert_close(y[:, :-1, :], y_modified[:, :-1, :], rtol=1e-5, atol=1e-6)

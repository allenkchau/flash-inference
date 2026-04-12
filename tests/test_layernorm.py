import torch

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.activations import Activation
from flash_inference.model.layernorm import LayerNorm


def _make_config(model_dim: int = 16, weight_tying: bool = False) -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        vocab_size=128,
        num_heads=4,
        model_dim=model_dim,
        max_seq_len=32,
        bias=False,
        mlp_activation=Activation.GELU,
        weight_tying=weight_tying,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_layernorm_forward_matches_torch_layernorm():
    torch.manual_seed(0)
    config = _make_config(model_dim=16)

    custom_ln = LayerNorm(config, eps=1e-5)
    torch_ln = torch.nn.LayerNorm(config.model_dim, eps=1e-5, elementwise_affine=True)

    with torch.no_grad():
        torch_ln.weight.copy_(custom_ln.gamma)
        torch_ln.bias.copy_(custom_ln.beta)

    x = torch.randn(4, 7, config.model_dim, dtype=config.dtype, device=config.device)

    y_custom = custom_ln(x)
    y_torch = torch_ln(x)

    torch.testing.assert_close(y_custom, y_torch, rtol=1e-5, atol=1e-6)


def test_layernorm_backward_matches_torch_layernorm():
    torch.manual_seed(1)
    config = _make_config(model_dim=32)

    custom_ln = LayerNorm(config, eps=1e-5)
    torch_ln = torch.nn.LayerNorm(config.model_dim, eps=1e-5, elementwise_affine=True)

    with torch.no_grad():
        torch_ln.weight.copy_(custom_ln.gamma)
        torch_ln.bias.copy_(custom_ln.beta)

    x_custom = torch.randn(
        3, 5, config.model_dim, dtype=config.dtype, device=config.device, requires_grad=True
    )
    x_torch = x_custom.detach().clone().requires_grad_(True)

    grad_out = torch.randn_like(x_custom)

    y_custom = custom_ln(x_custom)
    y_torch = torch_ln(x_torch)

    y_custom.backward(grad_out)
    y_torch.backward(grad_out)

    torch.testing.assert_close(y_custom, y_torch, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(x_custom.grad, x_torch.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(custom_ln.gamma.grad, torch_ln.weight.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(custom_ln.beta.grad, torch_ln.bias.grad, rtol=1e-5, atol=1e-6)

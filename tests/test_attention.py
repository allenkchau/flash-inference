import pytest
import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.activations import Activation
from flash_inference.model.attention import MHAttention


def _make_config(
    bias: bool,
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
        mlp_activation=Activation.GELU,
        weight_tying=weight_tying,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _copy_to_torch_mha(custom_attn: MHAttention, torch_attn: nn.MultiheadAttention, bias: bool) -> None:
    with torch.no_grad():
        torch_attn.in_proj_weight.copy_(
            torch.cat(
                [custom_attn.Wq.weight, custom_attn.Wk.weight, custom_attn.Wv.weight],
                dim=0,
            )
        )
        if bias:
            torch_attn.in_proj_bias.copy_(
                torch.cat([custom_attn.Wq.bias, custom_attn.Wk.bias, custom_attn.Wv.bias], dim=0)
            )
        torch_attn.out_proj.weight.copy_(custom_attn.Wo.weight)
        if bias:
            torch_attn.out_proj.bias.copy_(custom_attn.Wo.bias)


@pytest.mark.parametrize("bias", [False, True])
def test_attention_forward_matches_torch_multihead_attention(bias: bool):
    torch.manual_seed(0)
    config = _make_config(bias=bias, model_dim=16, num_heads=4, max_seq_len=64)

    custom_attn = MHAttention(config)
    torch_attn = nn.MultiheadAttention(
        embed_dim=config.model_dim,
        num_heads=config.num_heads,
        dropout=0.0,
        bias=bias,
        batch_first=True,
    )
    _copy_to_torch_mha(custom_attn, torch_attn, bias=bias)

    x = torch.randn(3, 7, config.model_dim, dtype=config.dtype, device=config.device)
    causal_block_mask = torch.triu(
        torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=config.device),
        diagonal=1,
    )

    y_custom = custom_attn(x)
    y_torch, _ = torch_attn(x, x, x, attn_mask=causal_block_mask, need_weights=False)

    torch.testing.assert_close(y_custom, y_torch, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bias", [False, True])
def test_attention_backward_matches_torch_multihead_attention(bias: bool):
    torch.manual_seed(1)
    config = _make_config(bias=bias, model_dim=32, num_heads=8, max_seq_len=64)

    custom_attn = MHAttention(config)
    torch_attn = nn.MultiheadAttention(
        embed_dim=config.model_dim,
        num_heads=config.num_heads,
        dropout=0.0,
        bias=bias,
        batch_first=True,
    )
    _copy_to_torch_mha(custom_attn, torch_attn, bias=bias)

    x_custom = torch.randn(2, 6, config.model_dim, dtype=config.dtype, device=config.device, requires_grad=True)
    x_torch = x_custom.detach().clone().requires_grad_(True)
    grad_out = torch.randn_like(x_custom)

    causal_block_mask = torch.triu(
        torch.ones(x_custom.size(1), x_custom.size(1), dtype=torch.bool, device=config.device),
        diagonal=1,
    )

    y_custom = custom_attn(x_custom)
    y_torch, _ = torch_attn(x_torch, x_torch, x_torch, attn_mask=causal_block_mask, need_weights=False)

    y_custom.backward(grad_out)
    y_torch.backward(grad_out)

    torch.testing.assert_close(y_custom, y_torch, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(x_custom.grad, x_torch.grad, rtol=1e-5, atol=1e-6)

    q_grad_ref, k_grad_ref, v_grad_ref = torch_attn.in_proj_weight.grad.chunk(3, dim=0)
    torch.testing.assert_close(custom_attn.Wq.weight.grad, q_grad_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(custom_attn.Wk.weight.grad, k_grad_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(custom_attn.Wv.weight.grad, v_grad_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(custom_attn.Wo.weight.grad, torch_attn.out_proj.weight.grad, rtol=1e-5, atol=1e-6)

    if bias:
        q_bias_grad_ref, k_bias_grad_ref, v_bias_grad_ref = torch_attn.in_proj_bias.grad.chunk(3, dim=0)
        torch.testing.assert_close(custom_attn.Wq.bias.grad, q_bias_grad_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(custom_attn.Wk.bias.grad, k_bias_grad_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(custom_attn.Wv.bias.grad, v_bias_grad_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(custom_attn.Wo.bias.grad, torch_attn.out_proj.bias.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bias", [False, True])
def test_attention_is_causal_no_future_leakage(bias: bool):
    torch.manual_seed(2)
    config = _make_config(bias=bias, model_dim=16, num_heads=4, max_seq_len=32)
    attention = MHAttention(config)

    x = torch.randn(2, 6, config.model_dim, dtype=config.dtype, device=config.device)
    x_modified = x.clone()
    x_modified[:, -1, :] = x_modified[:, -1, :] + 100.0

    y = attention(x)
    y_modified = attention(x_modified)

    torch.testing.assert_close(y[:, :-1, :], y_modified[:, :-1, :], rtol=1e-5, atol=1e-6)

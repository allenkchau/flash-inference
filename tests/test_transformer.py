import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.activations import Activation
from flash_inference.model.attention import MHAttention
from flash_inference.model.block import TransformerBlock
from flash_inference.model.embeddings import Embeddings
from flash_inference.model.transformer import Transformer


def _make_config(
    bias: bool,
    activation: Activation = Activation.GELU,
    model_dim: int = 16,
    num_heads: int = 4,
    max_seq_len: int = 32,
    vocab_size: int = 128,
    num_layers: int = 2,
    weight_tying: bool = False,
) -> ModelConfig:
    return ModelConfig(
        num_layers=num_layers,
        vocab_size=vocab_size,
        num_heads=num_heads,
        model_dim=model_dim,
        max_seq_len=max_seq_len,
        bias=bias,
        mlp_activation=activation,
        weight_tying=weight_tying,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _manual_transformer_forward(model: Transformer, input_ids: torch.Tensor, weight_tying: bool) -> torch.Tensor:
    x = model.embeddings(input_ids)
    for block in model.blocks:
        x = block(x)
    x = model.ln(x)
    if weight_tying:
        x = F.linear(x, model.embeddings.tok_embedding_table.weight, model.output.bias)
    else:
        x = model.output(x)
    return x


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("weight_tying", [False, True])
def test_transformer_output_shape(bias: bool, weight_tying: bool):
    torch.manual_seed(0)
    config = _make_config(
        bias=bias,
        model_dim=32,
        num_heads=8,
        max_seq_len=64,
        vocab_size=257,
        num_layers=3,
        weight_tying=weight_tying,
    )
    model = Transformer(config)

    input_ids = torch.randint(0, config.vocab_size, (4, 11), device=config.device, dtype=torch.long)
    logits = model(input_ids)

    assert logits.shape == (4, 11, config.vocab_size)
    assert logits.dtype == config.dtype
    assert logits.device == config.device


@pytest.mark.parametrize("bias", [False, True])
def test_embeddings_output_shape(bias: bool):
    torch.manual_seed(1)
    config = _make_config(bias=bias, model_dim=24, num_heads=6, max_seq_len=64)
    embeddings = Embeddings(config)

    input_ids = torch.randint(0, config.vocab_size, (3, 9), device=config.device, dtype=torch.long)
    out = embeddings(input_ids)

    assert out.shape == (3, 9, config.model_dim)
    assert out.dtype == config.dtype
    assert out.device == config.device


@pytest.mark.parametrize("bias", [False, True])
def test_attention_output_shape(bias: bool):
    torch.manual_seed(2)
    config = _make_config(bias=bias, model_dim=16, num_heads=4, max_seq_len=64)
    attention = MHAttention(config)

    x = torch.randn(2, 7, config.model_dim, dtype=config.dtype, device=config.device)
    out = attention(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", [Activation.GELU, Activation.RELU, Activation.SILU])
def test_block_output_shape(bias: bool, activation: Activation):
    torch.manual_seed(3)
    config = _make_config(bias=bias, activation=activation, model_dim=32, num_heads=8, max_seq_len=64)
    block = TransformerBlock(config)

    x = torch.randn(3, 10, config.model_dim, dtype=config.dtype, device=config.device)
    out = block(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


@pytest.mark.parametrize("bias", [False, True])
def test_embeddings_raises_when_seq_len_exceeds_max_seq_len(bias: bool):
    config = _make_config(bias=bias, model_dim=16, num_heads=4, max_seq_len=8)
    embeddings = Embeddings(config)

    too_long = torch.randint(0, config.vocab_size, (2, 9), device=config.device, dtype=torch.long)

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        embeddings(too_long)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", [Activation.GELU, Activation.RELU, Activation.SILU])
@pytest.mark.parametrize("weight_tying", [False, True])
def test_transformer_forward_matches_manual_pipeline(bias: bool, activation: Activation, weight_tying: bool):
    torch.manual_seed(4)
    config = _make_config(
        bias=bias,
        activation=activation,
        model_dim=16,
        num_heads=4,
        max_seq_len=32,
        vocab_size=101,
        num_layers=2,
        weight_tying=weight_tying,
    )
    model = Transformer(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 8), device=config.device, dtype=torch.long)

    logits_model = model(input_ids)
    logits_manual = _manual_transformer_forward(model, input_ids, weight_tying=weight_tying)

    torch.testing.assert_close(logits_model, logits_manual, rtol=1e-5, atol=1e-6)


def test_transformer_weight_tying_shares_output_and_token_weights():
    config = _make_config(
        bias=False,
        model_dim=16,
        num_heads=4,
        max_seq_len=32,
        vocab_size=101,
        num_layers=2,
        weight_tying=True,
    )
    model = Transformer(config)

    assert isinstance(model.output, nn.Linear)
    assert model.output.weight.data_ptr() == model.embeddings.tok_embedding_table.weight.data_ptr()


def test_transformer_without_weight_tying_has_independent_output_layer():
    config = _make_config(
        bias=False,
        model_dim=16,
        num_heads=4,
        max_seq_len=32,
        vocab_size=101,
        num_layers=2,
        weight_tying=False,
    )
    model = Transformer(config)

    assert isinstance(model.output, nn.Linear)
    assert model.output.weight.data_ptr() != model.embeddings.tok_embedding_table.weight.data_ptr()

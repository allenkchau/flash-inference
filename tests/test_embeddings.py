import pytest
import torch

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.activations import Activation
from flash_inference.model.embeddings import Embeddings


def _make_config(
    model_dim: int = 16,
    max_seq_len: int = 32,
    vocab_size: int = 128,
    weight_tying: bool = False,
) -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        vocab_size=vocab_size,
        num_heads=4,
        model_dim=model_dim,
        max_seq_len=max_seq_len,
        bias=False,
        mlp_activation=Activation.GELU,
        weight_tying=weight_tying,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_embeddings_output_shape_dtype_and_device():
    torch.manual_seed(0)
    config = _make_config(model_dim=24, max_seq_len=64, vocab_size=257)
    embeddings = Embeddings(config)

    input_ids = torch.randint(0, config.vocab_size, (3, 11), device=config.device, dtype=torch.long)
    out = embeddings(input_ids)

    assert out.shape == (3, 11, config.model_dim)
    assert out.dtype == config.dtype
    assert out.device == config.device


def test_embeddings_matches_manual_token_plus_positional_sum():
    torch.manual_seed(1)
    config = _make_config(model_dim=32, max_seq_len=64, vocab_size=101)
    embeddings = Embeddings(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 9), device=config.device, dtype=torch.long)
    out = embeddings(input_ids)

    token = embeddings.tok_embedding_table(input_ids)
    positions = torch.arange(input_ids.size(1), device=config.device)
    positional = embeddings.pos_embedding_table(positions).unsqueeze(0)
    expected = token + positional

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_embeddings_raises_when_seq_len_exceeds_max_seq_len():
    config = _make_config(model_dim=16, max_seq_len=8, vocab_size=128)
    embeddings = Embeddings(config)

    too_long = torch.randint(0, config.vocab_size, (2, 9), device=config.device, dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        embeddings(too_long)


def test_embeddings_raises_when_input_ids_not_long_dtype():
    config = _make_config(model_dim=16, max_seq_len=16, vocab_size=128)
    embeddings = Embeddings(config)

    bad_dtype_ids = torch.randint(0, config.vocab_size, (2, 5), device=config.device, dtype=torch.int32)
    with pytest.raises(ValueError, match="instead of torch.long"):
        embeddings(bad_dtype_ids)


def test_embeddings_backward_populates_embedding_grads():
    torch.manual_seed(2)
    config = _make_config(model_dim=16, max_seq_len=32, vocab_size=64)
    embeddings = Embeddings(config)

    input_ids = torch.randint(0, config.vocab_size, (4, 7), device=config.device, dtype=torch.long)
    out = embeddings(input_ids)
    loss = out.pow(2).mean()
    loss.backward()

    assert embeddings.tok_embedding_table.weight.grad is not None
    assert embeddings.pos_embedding_table.weight.grad is not None
    assert torch.isfinite(embeddings.tok_embedding_table.weight.grad).all()
    assert torch.isfinite(embeddings.pos_embedding_table.weight.grad).all()

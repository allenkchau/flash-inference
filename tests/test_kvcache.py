"""
Test functionality of KVCache class.

Run:
    python -m pytest tests/test_kvcache.py -v
"""

import pytest
import torch

from src.model.config import Config
from src.runtime.kv_cache import KVCache


def make_config(device):
    return Config(
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=8,
        max_batch_size=1,
        dtype=torch.float32,
        device=device,
    )


def test_kvcache_set_and_get():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = make_config(device)

    batch_size = 1
    cache = KVCache(config=config, batch_size=batch_size)

    B = batch_size
    Nh = config.num_heads
    # sequence length of 5 just to test
    T = 5
    Hd = config.head_dim

    k_full = torch.randn(B, Nh, T, Hd, device=device, dtype=config.dtype)
    v_full = torch.randn(B, Nh, T, Hd, device=device, dtype=config.dtype)

    # Bulk write (prefill)
    cache.set(layer_idx=0, k_full=k_full, v_full=v_full, seq_len=T)

    # Validate cur_len
    assert cache.cur_len == T

    # Validate retrieval
    k_cached, v_cached = cache.get_kv(layer_idx=0)

    assert k_cached.shape == (B, Nh, T, Hd)
    assert v_cached.shape == (B, Nh, T, Hd)

    torch.testing.assert_close(k_cached, k_full)
    torch.testing.assert_close(v_cached, v_full)


def test_kvcache_append():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = make_config(device)

    batch_size = 1
    cache = KVCache(config=config, batch_size=batch_size)

    B = batch_size
    Nh = config.num_heads
    T = 3
    Hd = config.head_dim

    # Initial set
    k_full = torch.randn(B, Nh, T, Hd, device=device, dtype=config.dtype)
    v_full = torch.randn(B, Nh, T, Hd, device=device, dtype=config.dtype)
    cache.set(layer_idx=0, k_full=k_full, v_full=v_full, seq_len=T)

    # New token
    k_new = torch.randn(B, Nh, 1, Hd, device=device, dtype=config.dtype)
    v_new = torch.randn(B, Nh, 1, Hd, device=device, dtype=config.dtype)

    cache.append(layer_idx=0, k_new=k_new, v_new=v_new, pos=T)
    cache.cur_len += 1

    k_cached, v_cached = cache.get_kv(layer_idx=0)

    assert k_cached.shape == (B, Nh, T + 1, Hd)
    assert v_cached.shape == (B, Nh, T + 1, Hd)

    # Check last position equals appended value
    torch.testing.assert_close(k_cached[:, :, T, :], k_new[:, :, 0, :])
    torch.testing.assert_close(v_cached[:, :, T, :], v_new[:, :, 0, :])


def test_kvcache_invalid_layer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = make_config(device)

    cache = KVCache(config=config, batch_size=1)

    with pytest.raises(AssertionError):
        cache.get_kv(layer_idx=999)


def test_kvcache_out_of_bounds_append():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = make_config(device)

    cache = KVCache(config=config, batch_size=1)

    B = 1
    Nh = config.num_heads
    Hd = config.head_dim

    k_new = torch.randn(B, Nh, 1, Hd, device=device, dtype=config.dtype)
    v_new = torch.randn(B, Nh, 1, Hd, device=device, dtype=config.dtype)

    # Position exceeds max_seq_len
    with pytest.raises(AssertionError):
        cache.append(layer_idx=0, k_new=k_new, v_new=v_new, pos=config.max_seq_len)

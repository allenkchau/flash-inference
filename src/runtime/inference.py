"""
Low-level inference primitives for autoregressive decoding.

This module contains the execution logic for transformer inference with
optional KV caching. It provides functions such as prefill and
decode_step that operate one level above the model's forward pass
but below high-level generation strategies.

- model/:  pure transformer math
- runtime/inference.py:  efficient execution + KV cache handling
- runtime/generate.py:  decoding strategy (greedy, sampling, etc.)
"""

import torch


def prefill(model, prompt_ids, kv_cache):
    logits, kv_updates = model(prompt_ids, kv_cache=kv_cache, return_kv=True)
    B, T = prompt_ids.shape

    # go through all the updates and update our cache appropriately
    for i, update in enumerate(kv_updates):
        K, V = update
        kv_cache.set(layer_idx=i, k_full=K, v_full=V, seq_len=T)
    kv_cache.cur_len = T
    return logits

def decode_step(model, next_token_id, kv_cache):
    B, _ = next_token_id.shape
    shape = (B, 1)
    position_ids = torch.full(shape, kv_cache.cur_len, dtype=torch.long, device=next_token_id.device)
    assert next_token_id.dtype == torch.long
    assert next_token_id.device == next(model.parameters()).device

    logits, kv_updates = model(next_token_id, kv_cache=kv_cache, return_kv=True, position_ids=position_ids)

    assert len(kv_updates) == model.num_layers

    # go through all the updates and update our cache appropriately
    for i, update in enumerate(kv_updates):
        K_new, V_new = update
        kv_cache.append(layer_idx=i, k_new=K_new, v_new=V_new, pos=kv_cache.cur_len)

    kv_cache.cur_len += 1
    return logits[:, -1, :]

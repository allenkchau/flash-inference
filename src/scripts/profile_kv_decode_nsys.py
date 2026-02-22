#!/usr/bin/env python3
"""
Profile a short KV-cached decode run with Nsight Systems (nsys).

Target: AWS EC2 A10G (headless). Produces a small, readable trace.

Usage:

  # 1) (Recommended) sanity run
  python - m src.scripts.profile_kv_decode_nsys

  # 2) Profile with Nsight Systems (trace only the NVTX capture region)
  mkdir -p profiles
  nsys profile \
    -o profiles/kv_decode_nsys \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    python -m src.scripts.profile_kv_decode_nsys

Then download/open:
  profiles/kv_decode_nsys.nsys-rep
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict

import torch

from src.model.config import Config
from src.model.transformer import Transformer
from src.runtime.kv_cache import KVCache
from src.runtime.inference import prefill, decode_step


def make_random_prompt(model: torch.nn.Module, batch_size: int, seq_len: int, seed: int = 1234) -> torch.Tensor:
    device = next(model.parameters()).device
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randint(
        low=0,
        high=model.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
        generator=g,
    )


def main() -> None:
    # ---------
    # Settings
    # ---------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "This profiling script is intended for CUDA GPUs (A10G)."

    # Keep model moderate so traces are manageable. You can scale up later.
    cfg = Config(
        vocab_size=1024,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        max_seq_len=2048,
        max_batch_size=1,
        dtype=torch.float32,
        device=device,
    )

    batch_size = 1
    prompt_len = 512
    decode_steps = 50         # number of decode_step calls to profile
    warmup_steps = 10         # warmup decode steps (not profiled)

    # ---------------
    # Build the model
    # ---------------
    model = Transformer(config=cfg).to(device)
    # In case other parts of your code expect model.config
    model.config = cfg

    model.eval()
    torch.set_grad_enabled(False)

    # Optional: keep GPU clocks stable-ish by preventing power-gating hiccups
    # (No-op on many systems; safe to leave.)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 can speed fp32 GEMMs on Ampere

    # ---------------
    # Prepare prompt
    # ---------------
    prompt = make_random_prompt(model, batch_size=batch_size, seq_len=prompt_len, seed=1234)

    # ---------------
    # Warmup (NOT profiled)
    # ---------------
    kv_cache = KVCache(config=model.config, batch_size=batch_size)

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("WARMUP")

    _ = prefill(model, prompt_ids=prompt, kv_cache=kv_cache)
    # First token from prefill logits
    next_id = torch.argmax(_[:, -1, :], dim=-1).to(torch.long).unsqueeze(1)
    # Append like your generation loop would (not strictly required for cache correctness)
    prompt_plus = torch.cat([prompt, next_id], dim=1)

    # Warmup decode steps
    for _i in range(warmup_steps):
        last_logits = decode_step(model, next_id, kv_cache=kv_cache)
        next_id = torch.argmax(last_logits, dim=-1).to(torch.long).unsqueeze(1)
        prompt_plus = torch.cat([prompt_plus, next_id], dim=1)

    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    # Reset cache so profiled region starts from a clean prefill state
    kv_cache.reset()

    # ---------------
    # Profile region
    # ---------------
    # We use NVTX capture start/stop so nsys can record only this region.
    torch.cuda.nvtx.range_push("CAPTURE")
    torch.cuda.profiler.start()

    # Prefill (profiled)
    torch.cuda.nvtx.range_push("PREFILL")
    logits = prefill(model, prompt_ids=prompt, kv_cache=kv_cache)
    torch.cuda.nvtx.range_pop()

    # First generated token from prefill logits (profiled)
    torch.cuda.nvtx.range_push("FIRST_TOKEN_FROM_PREFILL")
    next_id = torch.argmax(logits[:, -1, :], dim=-1).to(torch.long).unsqueeze(1)
    torch.cuda.nvtx.range_pop()

    # Decode steps (profiled)
    torch.cuda.nvtx.range_push("DECODE_LOOP")
    for i in range(decode_steps):
        torch.cuda.nvtx.range_push(f"DECODE_STEP_{i:03d}")
        last_logits = decode_step(model, next_id, kv_cache=kv_cache)
        next_id = torch.argmax(last_logits, dim=-1).to(torch.long).unsqueeze(1)
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()

    torch.cuda.profiler.stop()
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    print("Done. Wrote NVTX ranges for PREFILL and DECODE steps.")


if __name__ == "__main__":
    main()

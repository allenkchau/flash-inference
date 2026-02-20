#!/usr/bin/env python3
"""
Benchmark sweep: baseline vs KV cache across prompt lengths.

Writes:
  - results/sweep_summary.csv
  - results/sweep_series.jsonl

Assumes you have:
  - src.runtime.generate_baseline.generate_baseline
  - src.runtime.generate_kv.generate_kv
  - src.model.config.Config
  - src.model.transformer.Transformer
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pandas as pd

from src.model.config import Config
from src.model.transformer import Transformer
from src.runtime.generate_baseline import generate_baseline
from src.runtime.generate_kv import generate_kv


RESULTS_DIR = Path("results")
SUMMARY_CSV = RESULTS_DIR / "sweep_summary.csv"
SERIES_JSONL = RESULTS_DIR / "sweep_series.jsonl"


def make_random_prompt(model: torch.nn.Module, batch_size: int, seq_len: int, seed: int) -> torch.Tensor:
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


def summarize_run(
    method: str,
    prompt_len: int,
    max_new_tokens: int,
    prefill_ms: float,
    decode_ms_total: float,
    decode_times_ms: List[float],
    extra: Dict,
) -> Dict:
    decode_steps = len(decode_times_ms)
    decode_ms_per_tok = (decode_ms_total / decode_steps) if decode_steps > 0 else 0.0

    row = {
        "method": method,
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "prefill_ms": float(prefill_ms),
        "decode_ms_total": float(decode_ms_total),
        "decode_steps": int(decode_steps),
        "decode_ms_per_tok": float(decode_ms_per_tok),
        **extra,
    }
    return row


def warmup(model, prompt: torch.Tensor, warmup_tokens: int) -> None:
    model.eval()
    with torch.no_grad():
        _ = generate_baseline(model=model, input_ids=prompt, max_new_tokens=warmup_tokens, use_tqdm=False)
        _ = generate_kv(model=model, input_ids=prompt, max_new_tokens=warmup_tokens, use_tqdm=False)

    if torch.cuda.is_available() and prompt.device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------
    # Benchmark settings
    # -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep this reasonably small for sweeps; scale up later.
    cfg = Config(
        vocab_size=1024,
        hidden_size=256,     # <- suggest smaller for sweep speed
        num_layers=4,
        num_heads=8,
        max_seq_len=2048,
        max_batch_size=1,
        dtype=torch.float32,
        device=device,
    )

    batch_size = 1
    max_new_tokens = 256

    # Prompt length sweep
    prompt_lens = [8, 32, 64, 128, 512, 1024]

    # Repeats per prompt length (for median later)
    repeats = 5

    # Warmup
    warmup_runs = 2
    warmup_tokens = min(16, max_new_tokens)

    # Deterministic prompts per prompt_len
    base_seed = 1234

    # -----------------
    # Build model
    # -----------------
    model = Transformer(config=cfg).to(device)
    model.config = cfg  # in case you rely on model.config elsewhere

    # -----------------
    # Warmup (using a mid-size prompt)
    # -----------------
    warm_prompt_len = min(64, max(prompt_lens))
    warm_prompt = make_random_prompt(model, batch_size, warm_prompt_len, seed=base_seed + warm_prompt_len)
    for _ in range(warmup_runs):
        warmup(model, warm_prompt, warmup_tokens)

    rows: List[Dict] = []

    # Open JSONL once and append lines
    with SERIES_JSONL.open("w") as f_jsonl:
        for prompt_len in prompt_lens:
            # Ensure room for generation
            if prompt_len + max_new_tokens > cfg.max_seq_len:
                print(f"Skipping prompt_len={prompt_len} (would exceed max_seq_len with generation).")
                continue

            # Use the same prompt for baseline and kv within each repeat
            for r in range(repeats):
                seed = base_seed + prompt_len * 100 + r
                prompt = make_random_prompt(model, batch_size, prompt_len, seed=seed)

                # ---- Baseline ----
                out_ids_b, pre_b, dec_tot_b, dec_list_b = generate_baseline(
                    model=model,
                    input_ids=prompt,
                    max_new_tokens=max_new_tokens,
                    use_tqdm=False,
                )

                # ---- KV ----
                out_ids_k, pre_k, dec_tot_k, dec_list_k = generate_kv(
                    model=model,
                    input_ids=prompt,
                    max_new_tokens=max_new_tokens,
                    use_tqdm=False,
                )

                # Correctness check (greedy should match)
                if not torch.equal(out_ids_b, out_ids_k):
                    raise RuntimeError(
                        f"Mismatch baseline vs kv for prompt_len={prompt_len}, repeat={r}"
                    )

                extra = {
                    "batch_size": batch_size,
                    "hidden_size": cfg.hidden_size,
                    "num_layers": cfg.num_layers,
                    "num_heads": cfg.num_heads,
                    "dtype": str(cfg.dtype),
                    "device": str(device),
                    "repeat": r,
                }

                rows.append(summarize_run(
                    method="baseline",
                    prompt_len=prompt_len,
                    max_new_tokens=max_new_tokens,
                    prefill_ms=pre_b,
                    decode_ms_total=dec_tot_b,
                    decode_times_ms=dec_list_b,
                    extra=extra,
                ))

                rows.append(summarize_run(
                    method="kv",
                    prompt_len=prompt_len,
                    max_new_tokens=max_new_tokens,
                    prefill_ms=pre_k,
                    decode_ms_total=dec_tot_k,
                    decode_times_ms=dec_list_k,
                    extra=extra,
                ))

                # Write per-step series (for Plot A)
                # We store just the decode_times_ms; prompt_len lets plotter align x-axis.
                f_jsonl.write(json.dumps({
                    "method": "baseline",
                    "prompt_len": prompt_len,
                    "max_new_tokens": max_new_tokens,
                    "repeat": r,
                    "decode_times_ms": dec_list_b,
                }) + "\n")

                f_jsonl.write(json.dumps({
                    "method": "kv",
                    "prompt_len": prompt_len,
                    "max_new_tokens": max_new_tokens,
                    "repeat": r,
                    "decode_times_ms": dec_list_k,
                }) + "\n")

                # Keep GPU queue clean between repeats
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize()

            print(f"Finished prompt_len={prompt_len}")

    # Write summary CSV
    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)

    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {SERIES_JSONL}")


if __name__ == "__main__":
    main()

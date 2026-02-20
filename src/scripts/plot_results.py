#!/usr/bin/env python3
"""
Plot benchmark results for baseline vs KV cache.

Expected inputs:
  - results/sweep_summary.csv
      Columns (minimum):
        method,prompt_len,max_new_tokens,prefill_ms,decode_ms_total,decode_steps
      Optional:
        decode_ms_per_tok (will be computed if missing)

  - results/sweep_series.jsonl
      Each line is a JSON object with (minimum):
        {
          "method": "baseline" | "kv",
          "prompt_len": int,
          "max_new_tokens": int,
          "decode_times_ms": [float, float, ...]
        }

Outputs:
  - figures/prefill_vs_prompt.png
  - figures/avg_decode_vs_prompt.png
  - figures/decode_vs_seq_len_prompt_<T>.png  (one per prompt_len found)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
SUMMARY_CSV = RESULTS_DIR / "sweep_summary.csv"
SERIES_JSONL = RESULTS_DIR / "sweep_series.jsonl"


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_summary_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"method", "prompt_len", "max_new_tokens", "prefill_ms", "decode_ms_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    # decode_steps is useful; compute if missing and possible
    if "decode_steps" not in df.columns:
        # If you logged decode_steps elsewhere, add it. Otherwise we can't infer it reliably.
        # We'll allow missing, but avg decode plot may be less precise.
        df["decode_steps"] = pd.NA

    # Compute decode_ms_per_tok if missing and decode_steps present
    if "decode_ms_per_tok" not in df.columns:
        if df["decode_steps"].notna().all():
            df["decode_ms_per_tok"] = df["decode_ms_total"] / df["decode_steps"]
        else:
            # fallback: if not available, leave NaN
            df["decode_ms_per_tok"] = pd.NA

    return df


def plot_prefill_vs_prompt(df: pd.DataFrame) -> Path:
    # median across repeats (if you have repeats)
    agg = (
        df.groupby(["method", "prompt_len"], as_index=False)["prefill_ms"]
        .median()
        .sort_values(["prompt_len", "method"])
    )

    fig = plt.figure()
    ax = plt.gca()

    for method, sub in agg.groupby("method"):
        ax.plot(sub["prompt_len"], sub["prefill_ms"], marker="o", label=method)

    ax.set_title("Prefill latency vs prompt length")
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Prefill latency (ms)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    out = FIGURES_DIR / "prefill_vs_prompt.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_avg_decode_vs_prompt(df: pd.DataFrame) -> Path:
    # If decode_ms_per_tok is NaN because decode_steps missing, try to compute from decode_ms_total
    # but we still need decode_steps for that. So we require decode_ms_per_tok to be present or computable.
    if df["decode_ms_per_tok"].isna().all():
        raise ValueError(
            "decode_ms_per_tok is missing/NaN for all rows. "
            "Add decode_steps to sweep_summary.csv or directly write decode_ms_per_tok."
        )

    agg = (
        df.groupby(["method", "prompt_len"], as_index=False)["decode_ms_per_tok"]
        .median()
        .sort_values(["prompt_len", "method"])
    )

    fig = plt.figure()
    ax = plt.gca()

    for method, sub in agg.groupby("method"):
        ax.plot(sub["prompt_len"], sub["decode_ms_per_tok"], marker="o", label=method)

    ax.set_title("Average decode latency per token vs prompt length")
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Avg decode latency (ms/token)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    out = FIGURES_DIR / "avg_decode_vs_prompt.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def load_series_jsonl(path: Path) -> List[Dict]:
    runs: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            runs.append(json.loads(line))
    return runs


def build_seq_len_x(prompt_len: int, decode_times_ms: List[float]) -> List[int]:
    """
    Align x-axis for per-step decode timing.
    In your generators, the first generated token comes from prefill logits and is appended.
    Then decode_times_ms corresponds to subsequent decode steps.

    After prefill + first token append, current length is prompt_len + 1.
    For j-th decode step (0-indexed), sequence length during that step is (prompt_len + 1 + j).
    """
    start_len = prompt_len + 1
    return [start_len + j for j in range(len(decode_times_ms))]


def plot_decode_vs_seq_len(runs: List[Dict]) -> List[Path]:
    """
    Creates one plot per prompt_len found.
    Each plot overlays baseline vs kv for that prompt_len (if both exist).
    If multiple runs exist (repeats), it plots the median per-step (aligned by step index).
    """
    # group by (prompt_len, method)
    grouped: Dict[Tuple[int, str], List[List[float]]] = {}

    for r in runs:
        method = r.get("method")
        prompt_len = int(r.get("prompt_len"))
        decode_times = r.get("decode_times_ms")
        if method is None or decode_times is None:
            continue
        grouped.setdefault((prompt_len, method), []).append(list(decode_times))

    prompt_lens = sorted({pl for (pl, _m) in grouped.keys()})
    outputs: List[Path] = []

    for pl in prompt_lens:
        fig = plt.figure()
        ax = plt.gca()

        for method in sorted({m for (_pl, m) in grouped.keys() if _pl == pl}):
            series_list = grouped[(pl, method)]
            # Align by step index; take median across repeats at each step
            min_len = min(len(s) for s in series_list)
            if min_len == 0:
                continue

            trimmed = [s[:min_len] for s in series_list]
            med = pd.DataFrame(trimmed).median(axis=0).tolist()

            x = build_seq_len_x(pl, med)
            ax.plot(x, med, label=method)

        ax.set_title(f"Per-step decode latency vs sequence length (prompt_len={pl})")
        ax.set_xlabel("Sequence length during decode step (tokens)")
        ax.set_ylabel("Decode step latency (ms)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend()

        out = FIGURES_DIR / f"decode_vs_seq_len_prompt_{pl}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        outputs.append(out)

    return outputs


def main() -> None:
    ensure_dirs()

    if SUMMARY_CSV.exists():
        df = load_summary_csv(SUMMARY_CSV)
        out1 = plot_prefill_vs_prompt(df)
        print(f"Wrote {out1}")

        try:
            out2 = plot_avg_decode_vs_prompt(df)
            print(f"Wrote {out2}")
        except ValueError as e:
            print(f"Skipping avg decode plot: {e}")
    else:
        print(f"Missing {SUMMARY_CSV} — skipping summary plots.")

    if SERIES_JSONL.exists():
        runs = load_series_jsonl(SERIES_JSONL)
        outs = plot_decode_vs_seq_len(runs)
        for o in outs:
            print(f"Wrote {o}")
    else:
        print(f"Missing {SERIES_JSONL} — skipping per-step series plots.")


if __name__ == "__main__":
    main()

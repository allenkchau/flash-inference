"""
PyTorch Profiler -> TensorBoard trace writer.

This script writes TensorBoard profiler logs to:
  results/tb_trace/

Then view with:
  tensorboard --logdir results/tb_trace
and open http://localhost:6006  (Profile tab)

"""

from pathlib import Path
import torch
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)

from src.model.config import Config
from src.model.transformer import Transformer
from src.runtime.generate_kv import generate_kv


def main():
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config(
        vocab_size=1024,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        max_seq_len=1024,
        max_batch_size=1,
        dtype=torch.float32,
        device=device,
    )

    model = Transformer(cfg).to(device)
    model.config = cfg

    B, T = 1, 256
    prompt = torch.randint(0, cfg.vocab_size, (B, T), device=device, dtype=torch.long)

    # keep each profiled iteration small
    tokens_per_iter = 16

    logdir = Path("results/tb_trace")
    logdir.mkdir(parents=True, exist_ok=True)

   
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # warmuo
    model.eval()
    with torch.no_grad():
        _ = generate_kv(model, prompt, max_new_tokens=8, use_tqdm=False)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # ---- profiler schedule ----
    # Total iterations = wait + warmup + active = 1 + 1 + 2 = 4
    sched = torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1)

    with profile(
        activities=activities,
        schedule=sched,
        on_trace_ready=tensorboard_trace_handler(str(logdir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,   # set True only if you need call stacks (much slower)
        with_flops=True,
    ) as prof:
        total_iters = 1 + 1 + 2
        for i in range(total_iters):
            with record_function(f"generate_kv_iter_{i}"):
                with torch.no_grad():
                    _ = generate_kv(model, prompt, max_new_tokens=tokens_per_iter, use_tqdm=False)

            prof.step()

    files = list(logdir.rglob("*"))
    print(f"Wrote TensorBoard logs to: {logdir.resolve()}")
    print(f"Found {len(files)} files under {logdir}:")
    for p in files[:25]:
        print("  ", p)

    # print a small summary table too
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=25))


if __name__ == "__main__":
    main()

import time
import statistics
import logging
from typing import Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # profiling knobs
    profiling: bool = False,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating per-step timing of each training phase.

    When profiling=True, wraps every phase of every step with wall-clock timers
    (GPU-synchronized where available) and emits structured metrics via the
    standard tunalab logging convention:
        logger.info("...", extra={"metrics": {"profile/fwd_s": ..., ...}})

    Example log output per step (rendered by whatever logging handler is active):
        [profile] step   3  data=0.023s  fwd=0.412s  bwd=0.831s  opt=0.134s  total=1.400s

    LLM Compiler Hint: When composing this feature with others, time EVERY phase
    that appears in the combined loop — not just fwd/bwd/opt. For example:
    - data fetch: time spent blocking on next(iter) before the forward pass
    - forward pass: model(batch), GPU-synchronized
    - backward pass: loss.backward(), GPU-synchronized
    - optimizer step + zero_grad
    - gradient clipping (if present) — report as clip_time
    - validation (if present) — report separately as profile/val_s
    - LR scheduler step (if present)
    - gradient accumulation: report per micro-step AND per optimizer step
    - any other blocking work between steps

    Use torch.cuda.synchronize() before stopping each timer when CUDA is available
    so async GPU ops are counted in the correct phase.
    Skip step 0 from summary statistics (Triton/compile overhead dominates).
    Report mean and median for each phase so outliers are visible.
    Always emit profile lines with both print(..., flush=True) AND logger.info so
    output is visible in SLURM logs regardless of whether a log handler is
    configured. logger.info alone is silently dropped when no handler is attached.
    """
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    data_times, fwd_times, bwd_times, opt_times = [], [], [], []
    step = 0

    train_iter = iter(train_loader)
    while True:
        # --- data fetch ---
        t0 = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            break
        sync()
        t1 = time.perf_counter()

        # --- forward ---
        loss = model(batch)
        sync()
        t2 = time.perf_counter()

        # --- backward ---
        loss.backward()
        sync()
        t3 = time.perf_counter()

        # --- optimizer ---
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync()
        t4 = time.perf_counter()

        data, fwd, bwd, opt = t1 - t0, t2 - t1, t3 - t2, t4 - t3
        total = data + fwd + bwd + opt
        tag = " [COMPILE]" if step == 0 else ""

        if profiling:
            msg = (
                f"[profile] step {step:3d}{tag}"
                f"  data={data:.3f}s  fwd={fwd:.3f}s  bwd={bwd:.3f}s"
                f"  opt={opt:.3f}s  total={total:.3f}s"
            )
            print(msg, flush=True)
            logger.info(
                msg,
                extra={"metrics": {
                    "profile/data_s": data,
                    "profile/fwd_s":  fwd,
                    "profile/bwd_s":  bwd,
                    "profile/opt_s":  opt,
                    "profile/total_s": total,
                }},
            )

        if step > 0:
            data_times.append(data); fwd_times.append(fwd)
            bwd_times.append(bwd);   opt_times.append(opt)

        step += 1

    if profiling and data_times:
        totals = [d+f+b+o for d,f,b,o in zip(data_times, fwd_times, bwd_times, opt_times)]
        summary = (
            "[profile] summary (excl. step 0)  "
            f"data={statistics.mean(data_times):.3f}/{statistics.median(data_times):.3f}s  "
            f"fwd={statistics.mean(fwd_times):.3f}/{statistics.median(fwd_times):.3f}s  "
            f"bwd={statistics.mean(bwd_times):.3f}/{statistics.median(bwd_times):.3f}s  "
            f"opt={statistics.mean(opt_times):.3f}/{statistics.median(opt_times):.3f}s  "
            f"total={statistics.mean(totals):.3f}/{statistics.median(totals):.3f}s  (mean/median)"
        )
        print(summary, flush=True)
        logger.info(summary)

    return {"model": model}

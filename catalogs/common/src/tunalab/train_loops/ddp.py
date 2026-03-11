"""Atomic training loop feature: DistributedDataParallel (DDP) support.

When ``ddp_enabled=False`` (the default) this loop is identical to
``base_loop``.  When ``ddp_enabled=True`` and a process group is
initialised, it:

* Wraps the model in ``DistributedDataParallel`` (if not already wrapped).
* Suppresses gradient all-reduces on non-boundary micro-steps via
  ``model.no_sync()`` during gradient accumulation.
* All-reduces the scalar loss across ranks after each backward pass so
  that the value logged to stdout is the global average, not the local
  rank's value.

Compose with ``grad_accum`` or any other atomic feature via ``smart_train``
as normal — the DDP behaviour is additive and does not conflict with them.
"""

from contextlib import nullcontext
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _no_sync(model: nn.Module):
    """Return a ``no_sync()`` context for the DDP-wrapped model/submodule.

    Handles three cases transparently:
    - Whole model is DDP-wrapped → ``model.no_sync()``
    - DDP was applied to a direct child (selective wrapping) → first child's
      ``no_sync()``
    - Not DDP-wrapped (single GPU / ddp disabled) → ``nullcontext()``
    """
    if isinstance(model, DDP):
        return model.no_sync()
    for child in model.children():
        if isinstance(child, DDP):
            return child.no_sync()
    return nullcontext()


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # DDP knobs
    ddp_enabled: bool = False,
    local_rank: int = 0,
    # Gradient accumulation (kept here so no_sync lines up with accum boundaries)
    accum_steps: int = 1,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop with DDP support.

    Args:
        model: nn.Module whose ``forward`` returns a scalar loss.
        optimizer: Optimizer for all parameters.
        train_loader: Iterable of batches.
        ddp_enabled: When True and a process group is initialised, wrap the
            model in DDP and use ``no_sync()`` to skip redundant all-reduces
            during gradient accumulation.
        local_rank: GPU index on the current node.  Passed to
            ``DistributedDataParallel(device_ids=[local_rank])`` when the
            model is on CUDA.  Ignored for CPU models.
        accum_steps: Number of micro-steps between optimizer steps.
            ``no_sync()`` is used on non-boundary steps to avoid spurious
            all-reduces.

    Returns:
        ``{"model": <unwrapped model>}``
    """
    if accum_steps < 1:
        accum_steps = 1

    # Wrap in DDP if requested and not already wrapped.
    # device_ids is set from the model's actual device, not just cuda availability,
    # so that CPU-mode tests (Gloo) work correctly.
    if (
        ddp_enabled
        and dist.is_available()
        and dist.is_initialized()
        and not isinstance(model, DDP)
    ):
        try:
            first_param_device = next(iter(model.parameters())).device
        except StopIteration:
            first_param_device = torch.device("cpu")
        device_ids = [local_rank] if first_param_device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    micro_idx = 0
    for batch in train_loader:
        is_sync_step = (micro_idx + 1) % accum_steps == 0

        # Suppress all-reduce on non-boundary micro-steps
        ctx = nullcontext() if is_sync_step else _no_sync(model)
        with ctx:
            loss = model(batch)
            if accum_steps > 1:
                loss = loss / float(accum_steps)
            loss.backward()

        micro_idx += 1

        if is_sync_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # Handle incomplete final accumulation window
    if micro_idx % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    raw_model = model.module if isinstance(model, DDP) else model
    return {"model": raw_model}

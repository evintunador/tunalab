from typing import Optional, Dict, Any
import os

import torch
import torch.nn as nn
import torch.distributed as dist

import tunalab.checkpointer as checkpointer
from tunalab.distributed import is_main_process, cpu_barrier


@torch.no_grad()
def _eval_loss(model: nn.Module, loader) -> float:
    """Helper to compute validation loss, averaged across all DDP ranks."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        loss = model(batch)
        total += float(loss.detach().cpu().item())
        count += 1
    if was_training:
        model.train()

    # Average across ranks so every process selects "best" on the same number.
    # Reduce summed loss and count, then divide once: (Σ total) / (Σ count).
    # Tensor must be on CUDA — NCCL cannot reduce CPU tensors.
    if dist.is_available() and dist.is_initialized():
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cuda")
        t = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t[0].item() / max(t[1].item(), 1.0)

    return total / max(count, 1)


def _save_best(raw_model, optimizer, val_loss, step, output_dir, kwargs):
    """Save checkpoint on rank 0 then cpu_barrier so no rank races ahead."""
    if is_main_process():
        checkpointer.save_checkpoint(
            filepath=os.path.join(output_dir, "checkpoints", "best_model.pt"),
            metadata={"val_loss": val_loss, "step": step, "config": kwargs.get("config", {})},
            model=raw_model,
            optimizer=optimizer,
        )
    # All ranks wait for rank 0 to finish writing before proceeding.
    # Uses the secondary Gloo group when available so NCCL is never blocked.
    cpu_barrier()


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # feature knobs
    save_best_model: bool = False,
    output_dir: Optional[str] = None,
    val_loader=None,
    val_interval: int = 10,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic feature for saving the best model based on validation loss.

    In distributed training (DDP), only rank 0 writes the checkpoint.  All
    ranks synchronise via ``cpu_barrier()`` after the write so that no rank
    races ahead to a collective that would deadlock while rank 0 is busy with
    NFS I/O.
    """
    model.train()
    best_val_loss = float('inf')
    result = {"model": model}

    optimizer.zero_grad(set_to_none=True)

    step_count = 0
    for batch in train_loader:
        loss = model(batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if save_best_model:
            if val_loader is None or output_dir is None:
                raise ValueError(
                    "val_loader and output_dir must be provided when save_best_model is True."
                )

            if step_count % val_interval == 0:
                val_loss = _eval_loss(model, val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    result['best_val_loss'] = best_val_loss
                    raw_model = model.module if hasattr(model, 'module') else model
                    _save_best(raw_model, optimizer, best_val_loss, step_count,
                               output_dir, kwargs)

        step_count += 1

    # Final validation check
    if (
        save_best_model
        and val_loader is not None
        and output_dir is not None
        and step_count % val_interval != 0
    ):
        val_loss = _eval_loss(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            result['best_val_loss'] = best_val_loss
            raw_model = model.module if hasattr(model, 'module') else model
            _save_best(raw_model, optimizer, best_val_loss, step_count,
                       output_dir, kwargs)

    return result

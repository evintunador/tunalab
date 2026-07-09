from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.distributed as dist


@torch.no_grad()
def _eval_loss(model: nn.Module, loader) -> float:
    """Compute validation loss, averaged across all DDP ranks.

    In single-process mode the all_reduce is a no-op so this is identical
    to the previous behaviour.
    """
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        loss = model(batch)
        total += float(loss.detach().cpu().item())
        count += 1
    if was_training:
        model.train()

    # Average across ranks so all processes see the same validation number.
    # Reduce summed loss and count, then divide once: (Σ total) / (Σ count).
    # Reducing per-rank means would give mean/val_steps (deflation bug).
    # Tensor must be on CUDA — NCCL cannot reduce CPU tensors.
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([total, float(count)], dtype=torch.float64, device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        global_loss = t[0].item() / max(t[1].item(), 1.0)
        return global_loss

    return total / max(count, 1)


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # validation knobs
    val_loader=None,
    val_interval: int = 10,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating validation during training.

    In distributed training, the validation loss is averaged across all ranks
    via ``dist.all_reduce`` so that every process records the same metric.
    """
    model.train()

    val_loss_history: List[float] = []

    step_count = 0
    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if val_loader is not None and step_count > 0 and step_count % val_interval == 0:
            val_loss = _eval_loss(model, val_loader)
            val_loss_history.append(val_loss)
        step_count += 1

    # Final validation at end of training
    if val_loader is not None and (step_count == 0 or step_count % val_interval != 0):
        final_val_loss = _eval_loss(model, val_loader)
        val_loss_history.append(final_val_loss)

    result = {"model": model}
    if val_loader is not None:
        result["val_loss_history"] = val_loss_history

    return result

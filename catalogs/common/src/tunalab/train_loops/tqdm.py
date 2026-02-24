from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm as tqdm_auto


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # tqdm knobs
    use_tqdm: bool = False,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """A training loop with an optional tqdm progress bar."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    pbar = None
    if use_tqdm:
        is_map_style = not isinstance(train_loader.dataset, IterableDataset)
        total = len(train_loader) if is_map_style else None
        pbar = tqdm_auto(desc="Training", leave=False, total=total)

    try:
        for batch in train_loader:
            loss = model(batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
    finally:
        if pbar is not None:
            pbar.close()

    return {"model": model}
from typing import Dict, Any
import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    **kwargs,
) -> Dict[str, Any]:
    """
    A base 'zero feature' training loop to build your new atomic feature loop off of.
    
    This is the simplest possible training loop that all atomic features should be
    backwards compatible with when using default arguments.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for batch in train_loader:
        loss = model(batch)  # Model is responsible for unpacking batch and calculating loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {"model": model}


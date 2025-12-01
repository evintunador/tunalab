from typing import Dict, Any, Protocol, runtime_checkable
import torch
import torch.nn as nn


@runtime_checkable
class TrainingLoop(Protocol):
    """
    Protocol for training loop functions.
    
    All training loops must implement a `run_training` function with this signature.
    This ensures composability and standardization across all atomic features and
    compiled training loops.
    """
    
    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute a training loop.
        
        Args:
            model: PyTorch model to train. The model's forward method must accept
                   a batch from the train_loader and return a single loss tensor.
            optimizer: PyTorch optimizer
            train_loader: Training data loader (map-style or iterable-style)
            **kwargs: Additional feature-specific arguments
            
        Returns:
            Dictionary containing at minimum {'model': nn.Module}, with additional
            keys depending on the features used (e.g., 'val_loss_history', 'metrics')
        """
        ...
"""Helper functions for testing optimizers."""

from typing import Dict, Any, Type
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def run_learning_test(
    optimizer_class: Type[torch.optim.Optimizer],
    optimizer_kwargs: Dict[str, Any],
    device: str = 'cpu',
    num_epochs: int = 3,
    improvement_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Run a simple learning task to verify optimizer functionality.
    
    Creates a simple classification task and trains a model, verifying
    that the loss decreases by at least the improvement threshold.
    
    Args:
        optimizer_class: Optimizer class to test
        optimizer_kwargs: Kwargs to pass to optimizer constructor
        device: Device to run on
        num_epochs: Number of training epochs
        improvement_threshold: Required loss reduction ratio (e.g., 0.8 = 20% improvement)
    
    Returns:
        Dictionary with metrics:
            - pre_loss: Initial loss
            - post_loss: Final loss
            - loss_reduction: Relative improvement (pre-post)/pre
            - total_steps: Number of optimization steps taken
    
    Raises:
        AssertionError: If loss doesn't improve enough
    """
    torch.manual_seed(42)
    
    # Create a simple but non-trivial learning task
    X = torch.randn(2048, 32).to(device)
    true_weights = torch.randn(32).to(device)
    threshold = 0.0
    y = ((X * true_weights).sum(dim=1) > threshold).long().to(device)
    
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    
    # Simple 3-layer network
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    
    def measure_loss(model, dataloader):
        """Measure average loss over dataset."""
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_samples = 0
            for batch_X, batch_y in dataloader:
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                total_loss += loss.item() * batch_X.size(0)
                total_samples += batch_X.size(0)
        return total_loss / total_samples
    
    # Measure initial loss
    pre_loss = measure_loss(model, dl)
    
    # Create optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    
    # Training loop
    model.train()
    total_steps = 0
    
    for epoch in range(num_epochs):
        for batch_X, batch_y in dl:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_steps += 1
    
    # Measure final loss
    post_loss = measure_loss(model, dl)
    
    # Check for learning
    if not (post_loss < pre_loss * improvement_threshold):
        raise AssertionError(
            f"Optimizer {optimizer_class.__name__} failed to learn: "
            f"pre_loss={pre_loss:.4f}, post_loss={post_loss:.4f}, "
            f"improvement={((pre_loss - post_loss) / pre_loss * 100):.1f}% "
            f"(required: {((1 - improvement_threshold) * 100):.1f}%)"
        )
    
    return {
        "pre_loss": pre_loss,
        "post_loss": post_loss,
        "loss_reduction": (pre_loss - post_loss) / pre_loss,
        "total_steps": total_steps,
        "optimizer_class": optimizer_class.__name__,
    }


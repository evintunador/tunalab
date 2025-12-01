"""Helper functions for testing training loops."""

from typing import Dict, Any, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleTestTrainingModel(nn.Module):
    """
    A simple wrapper that combines a backbone and loss function.
    
    This is used for testing training loops that expect a model
    with a forward method that returns a loss.
    """
    def __init__(self, backbone: nn.Module, loss_fn: Callable):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def forward(self, batch):
        xb, yb = batch
        logits = self.backbone(xb)
        return self.loss_fn(logits, yb)


def run_training_smoke_test(
    run_training_fn: Callable,
    device: str = 'cpu',
    improvement_threshold: float = 0.9,
    **loop_kwargs,
) -> Dict[str, Any]:
    """
    Smoke test for training loops - verify training runs and improves loss.
    
    Args:
        run_training_fn: Training loop function with signature
            run_training(model, optimizer, train_loader, **kwargs)
        device: Device to run on
        improvement_threshold: Required loss reduction ratio
        **loop_kwargs: Additional kwargs to pass to training loop
    
    Returns:
        Dictionary with metrics including pre_loss, post_loss
    
    Raises:
        AssertionError: If training fails or loss doesn't improve
    """
    torch.manual_seed(0)
    X = torch.randn(2048, 32).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    backbone = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))
    model = SimpleTestTrainingModel(backbone, nn.CrossEntropyLoss()).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # Measure pre-training loss
    with torch.no_grad():
        pre_loss = model((X, y)).item()

    # Run training
    result = run_training_fn(
        model=model,
        optimizer=optim,
        train_loader=dl,
        **loop_kwargs,
    )

    # Measure post-training loss
    with torch.no_grad():
        post_loss = model((X, y)).item()

    # Verify result format
    if not isinstance(result, dict):
        raise AssertionError("run_training(...) must return dict")
    if 'model' not in result:
        raise AssertionError("Result dictionary must contain 'model' key")
    if not isinstance(result['model'], nn.Module):
        raise AssertionError("'model' key must be an nn.Module instance")

    # Verify learning happened
    if not (post_loss < pre_loss * improvement_threshold):
        raise AssertionError(
            f"Training did not sufficiently improve loss: "
            f"pre={pre_loss:.4f}, post={post_loss:.4f}"
        )

    return {
        "pre_loss": pre_loss,
        "post_loss": post_loss,
        "loss_reduction": (pre_loss - post_loss) / pre_loss,
        "result_keys": list(result.keys()),
    }


def run_base_loop_compliance_test(
    run_training_fn: Callable,
    device: str = 'cpu',
) -> None:
    """
    Test that a training loop with default args behaves like base_loop.
    
    This ensures atomic training loop features maintain backward compatibility
    by comparing their behavior (with defaults) against the base loop.
    
    Args:
        run_training_fn: Training loop function to test
        device: Device to run on
    
    Raises:
        AssertionError: If behavior differs from base_loop
    """
    # Import base_loop
    from tunalab.train_loops.base_loop import run_training as base_run_training
    
    # Create deterministic test setup
    torch.manual_seed(42)
    X = torch.randn(128, 16).to(device)
    y = torch.randint(0, 3, (128,)).to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=32, shuffle=False)  # No shuffle for determinism
    
    # Test with two identical models
    torch.manual_seed(42)
    backbone1 = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))
    model1 = SimpleTestTrainingModel(backbone1, nn.CrossEntropyLoss()).to(device)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
    
    torch.manual_seed(42)  # Reset for identical init
    backbone2 = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))
    model2 = SimpleTestTrainingModel(backbone2, nn.CrossEntropyLoss()).to(device)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    
    # Run base_loop
    base_result = base_run_training(
        model=model1,
        optimizer=optimizer1,
        train_loader=dl,
    )
    
    # Run feature loop with default arguments
    feature_result = run_training_fn(
        model=model2,
        optimizer=optimizer2,
        train_loader=dl,
    )
    
    # Verify both return dicts with 'model' key
    if not isinstance(base_result, dict) or 'model' not in base_result:
        raise AssertionError("base_loop must return dict with 'model' key")
    if not isinstance(feature_result, dict) or 'model' not in feature_result:
        raise AssertionError("Feature loop must return dict with 'model' key")
    
    # Compare model parameters to ensure identical training
    base_params = list(base_result['model'].parameters())
    feature_params = list(feature_result['model'].parameters())
    
    if len(base_params) != len(feature_params):
        raise AssertionError("Model parameter count mismatch with base_loop")
    
    for i, (base_p, feature_p) in enumerate(zip(base_params, feature_params)):
        if not torch.allclose(base_p.data, feature_p.data, atol=1e-6, rtol=1e-5):
            max_diff = torch.max(torch.abs(base_p.data - feature_p.data)).item()
            raise AssertionError(
                f"Parameter {i} differs from base_loop behavior. "
                f"This indicates default arguments don't produce base_loop-equivalent behavior. "
                f"Max diff: {max_diff:.2e}"
            )
    
    # Verify result dict only contains 'model' key (with some exceptions)
    expected_keys = {'model'}
    extra_keys = set(feature_result.keys()) - expected_keys
    
    # Allow certain keys that are informational
    allowed_extra = {'used_amp'}  # mixed_precision always returns this
    unexpected_keys = extra_keys - allowed_extra
    
    if unexpected_keys:
        raise AssertionError(
            f"With default arguments, should only return {expected_keys} keys "
            f"(plus {allowed_extra} if applicable), "
            f"but also returned: {unexpected_keys}. "
            f"This suggests default arguments don't disable the feature."
        )


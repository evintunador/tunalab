"""
Tests for the tqdm atomic feature.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, IterableDataset

from tunalab.train_loops.tqdm import run_training
from tunalab.testing import SimpleTestTrainingModel, get_available_devices


class _SimpleIterableDataset(IterableDataset):
    """Minimal IterableDataset yielding fixed (X, y) batches for testing."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, batch_size: int, n_batches: int):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_batches = n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.X[: self.batch_size], self.y[: self.batch_size]


@pytest.mark.parametrize(
    "run_training_fn,device",
    [(run_training, d) for d in get_available_devices() + ["cpu"]],
)
def test_tqdm_with_iterable_dataset(run_training_fn, device):
    """tqdm=True must not crash when used with an IterableDataset (no __len__).

    Previously, 'if pbar:' would raise TypeError because tqdm's __bool__ is
    undefined when total=None (indeterminate bar).  The fix changes all pbar
    truthiness checks to 'if pbar is not None:'.
    """
    torch.manual_seed(0)
    backbone = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
    model = SimpleTestTrainingModel(backbone, nn.CrossEntropyLoss()).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    X = torch.randn(64, 16).to(device)
    y = torch.randint(0, 2, (64,)).to(device)

    dataset = _SimpleIterableDataset(X, y, batch_size=16, n_batches=4)
    loader = DataLoader(dataset, batch_size=None)

    # Must not raise TypeError from tqdm.__bool__
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        use_tqdm=True,
    )
    assert "model" in result


@pytest.mark.parametrize(
    "run_training_fn,device",
    [(run_training, d) for d in get_available_devices() + ["cpu"]],
)
def test_tqdm_with_map_style_dataset(run_training_fn, device):
    """tqdm=True must work normally for map-style datasets (has __len__)."""
    torch.manual_seed(0)
    backbone = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
    model = SimpleTestTrainingModel(backbone, nn.CrossEntropyLoss()).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    X = torch.randn(64, 16).to(device)
    y = torch.randint(0, 2, (64,)).to(device)
    loader = DataLoader(TensorDataset(X, y), batch_size=16)

    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        use_tqdm=True,
    )
    assert "model" in result


# Export for smart_train compiler discovery
__specific_tests__ = [
    test_tqdm_with_iterable_dataset,
    test_tqdm_with_map_style_dataset,
]

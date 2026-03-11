"""Tests for the checkpoint_best_model atomic training loop feature."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tunalab.train_loops.checkpoint_best_model import run_training
from tunalab.testing import (
    SimpleTestTrainingModel,
    get_available_devices,
    run_base_loop_compliance_test,
)


def _make_loader(device, n=64, feat=8, batch=16):
    X = torch.randn(n, feat).to(device)
    y = (X.sum(1) > 0).long().to(device)
    return DataLoader(TensorDataset(X, y), batch_size=batch)


def _make_model(device, feat=8):
    backbone = nn.Sequential(nn.Linear(feat, 16), nn.ReLU(), nn.Linear(16, 2))
    return SimpleTestTrainingModel(backbone, nn.CrossEntropyLoss()).to(device)


# ---------------------------------------------------------------------------
# Base-loop compliance (feature disabled by default)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", get_available_devices())
def test_base_loop_compliance(device):
    run_base_loop_compliance_test(run_training, device=device)


# ---------------------------------------------------------------------------
# Feature behaviour (save_best_model=True)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", get_available_devices())
def test_saves_when_loss_improves(device, tmp_path):
    """Checkpoint must be written when validation loss improves."""
    torch.manual_seed(0)
    train_dl = _make_loader(device)
    val_dl = _make_loader(device, n=32)
    model = _make_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    mock_save = MagicMock()
    with patch("tunalab.checkpointer.save_checkpoint", mock_save):
        result = run_training(
            model=model,
            optimizer=optimizer,
            train_loader=train_dl,
            save_best_model=True,
            output_dir=str(tmp_path),
            val_loader=val_dl,
            val_interval=1,
        )

    assert mock_save.called, "save_checkpoint must be called at least once"
    assert "best_val_loss" in result
    assert isinstance(result["best_val_loss"], float)


@pytest.mark.parametrize("device", get_available_devices())
def test_raises_without_val_loader(device):
    """save_best_model=True without val_loader/output_dir must raise."""
    torch.manual_seed(0)
    train_dl = _make_loader(device)
    model = _make_model(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="val_loader"):
        run_training(
            model=model, optimizer=optimizer, train_loader=train_dl,
            save_best_model=True, output_dir="/tmp",
            val_loader=None,
        )


# ---------------------------------------------------------------------------
# DDP-safe checkpoint: only rank 0 writes; cpu_barrier called after each save
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", get_available_devices())
def test_checkpoint_only_on_rank0(device, tmp_path):
    """When is_main_process() returns False, save_checkpoint must NOT be called."""
    torch.manual_seed(0)
    train_dl = _make_loader(device)
    val_dl = _make_loader(device, n=32)
    model = _make_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    mock_save = MagicMock()
    with (
        patch("tunalab.checkpointer.save_checkpoint", mock_save),
        patch("tunalab.train_loops.checkpoint_best_model.is_main_process",
              return_value=False),
        patch("tunalab.train_loops.checkpoint_best_model.cpu_barrier"),
    ):
        run_training(
            model=model, optimizer=optimizer, train_loader=train_dl,
            save_best_model=True, output_dir=str(tmp_path),
            val_loader=val_dl, val_interval=1,
        )

    mock_save.assert_not_called()


@pytest.mark.parametrize("device", get_available_devices())
def test_cpu_barrier_called_after_save(device, tmp_path):
    """cpu_barrier() must be called after every checkpoint write (rank 0 path)."""
    torch.manual_seed(0)
    train_dl = _make_loader(device)
    val_dl = _make_loader(device, n=32)
    model = _make_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    mock_save = MagicMock()
    mock_barrier = MagicMock()
    with (
        patch("tunalab.checkpointer.save_checkpoint", mock_save),
        patch("tunalab.train_loops.checkpoint_best_model.is_main_process",
              return_value=True),
        patch("tunalab.train_loops.checkpoint_best_model.cpu_barrier", mock_barrier),
    ):
        run_training(
            model=model, optimizer=optimizer, train_loader=train_dl,
            save_best_model=True, output_dir=str(tmp_path),
            val_loader=val_dl, val_interval=1,
        )

    # Every successful save must be followed by a barrier
    assert mock_save.call_count == mock_barrier.call_count, (
        f"save called {mock_save.call_count}x but barrier called {mock_barrier.call_count}x"
    )
    assert mock_barrier.call_count >= 1, "cpu_barrier must be called at least once"


__specific_tests__ = [
    test_base_loop_compliance,
    test_saves_when_loss_improves,
    test_raises_without_val_loader,
    test_checkpoint_only_on_rank0,
    test_cpu_barrier_called_after_save,
]

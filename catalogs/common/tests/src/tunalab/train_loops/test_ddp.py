"""Tests for the DDP atomic training loop feature."""

import os
import socket

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset

from tunalab.train_loops.ddp import run_training
from tunalab.testing import (
    SimpleTestTrainingModel,
    get_available_devices,
    run_training_smoke_test,
    run_base_loop_compliance_test,
)


# ---------------------------------------------------------------------------
# Standard compliance / smoke tests (single-process, feature disabled)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", get_available_devices())
def test_base_loop_compliance(device):
    """With default args, DDP loop must be identical to base_loop."""
    run_base_loop_compliance_test(run_training, device=device)


@pytest.mark.parametrize("device", get_available_devices())
def test_smoke_ddp_disabled(device):
    """DDP disabled: model must learn (same contract as base_loop)."""
    run_training_smoke_test(run_training, device=device)


@pytest.mark.parametrize("device", get_available_devices())
def test_smoke_ddp_disabled_accum(device):
    """DDP disabled with gradient accumulation: model must still learn."""
    run_training_smoke_test(run_training, device=device, accum_steps=4)


# ---------------------------------------------------------------------------
# ddp_enabled=True on a single process (no dist group) — must not crash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", get_available_devices())
def test_ddp_enabled_no_group_single_process(device):
    """When ddp_enabled=True but no process group exists, degrades gracefully."""
    torch.manual_seed(0)
    X = torch.randn(64, 16).to(device)
    y = (X.sum(1) > 0).long().to(device)
    loader = DataLoader(TensorDataset(X, y), batch_size=16)

    model = SimpleTestTrainingModel(
        nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 2)),
        nn.CrossEntropyLoss(),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    result = run_training(model=model, optimizer=optimizer, train_loader=loader,
                          ddp_enabled=True, local_rank=0)

    assert isinstance(result, dict)
    assert "model" in result
    assert not isinstance(result["model"], torch.nn.parallel.DistributedDataParallel)


# ---------------------------------------------------------------------------
# Multi-process DDP correctness tests (Gloo backend, CPU)
# Module-level worker functions so they can be pickled by mp.spawn.
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _ddp_worker(rank: int, world_size: int, port: int, output_dir: str,
                accum_steps: int) -> None:
    """Spawn worker: init Gloo group, run DDP training, save param flat to file."""
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        torch.manual_seed(42)
        backbone = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        model = SimpleTestTrainingModel(backbone, nn.CrossEntropyLoss())

        torch.manual_seed(100 + rank)  # each rank sees different data
        X = torch.randn(64, 8)
        y = torch.randint(0, 2, (64,))
        loader = DataLoader(TensorDataset(X, y), batch_size=8)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        result = run_training(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            ddp_enabled=True,
            local_rank=rank,
            accum_steps=accum_steps,
        )

        params_flat = torch.cat([
            p.data.cpu().flatten() for p in result["model"].parameters()
        ])
        torch.save(params_flat, os.path.join(output_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("accum_steps", [1, 2])
def test_ddp_weights_identical_across_ranks(accum_steps, tmp_path):
    """DDP training must produce byte-identical model weights on all ranks."""
    import torch.multiprocessing as mp

    world_size = 2
    port = _find_free_port()
    mp.spawn(
        _ddp_worker,
        args=(world_size, port, str(tmp_path), accum_steps),
        nprocs=world_size,
        join=True,
    )

    params = [torch.load(str(tmp_path / f"rank{r}.pt")) for r in range(world_size)]
    for r in range(1, world_size):
        assert torch.allclose(params[0], params[r], atol=1e-5), (
            f"accum_steps={accum_steps}: weights differ between rank 0 and rank {r}. "
            f"Max diff: {(params[0] - params[r]).abs().max().item():.2e}"
        )

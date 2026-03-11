import os
import signal
from unittest import mock

import pytest
import torch

from tunalab.distributed import (
    DistributedManager,
    is_available,
    is_initialized,
    get_rank,
    get_local_rank,
    get_world_size,
    is_main_process,
    barrier,
    cpu_barrier,
    setup_signal_handlers,
    _resolve_slurm_master_addr,
    _resolve_slurm_master_port,
)


def test_standalone_functions_default_state():
    """Test that standalone functions return correct defaults before manager is used."""
    assert is_available() == torch.distributed.is_available()
    assert not is_initialized()
    assert get_rank() == 0
    assert get_local_rank() == 0
    assert get_world_size() == 1
    assert is_main_process()
    barrier()   # Should be a no-op
    cpu_barrier()  # Should also be a no-op


def test_context_manager_basic():
    """Test basic context manager functionality in non-distributed environment."""
    with DistributedManager() as manager:
        assert manager is not None
        assert isinstance(manager.device, torch.device)
        assert manager.rank == 0
        assert manager.local_rank == 0
        assert manager.world_size == 1
        assert not manager.is_distributed
        assert manager.is_main_process

        assert is_available() == torch.distributed.is_available()
        assert is_initialized() == manager.is_initialized()
        assert get_rank() == manager.rank
        assert get_local_rank() == manager.local_rank
        assert get_world_size() == manager.world_size
        assert is_main_process() == manager.is_main_process

    assert not is_initialized()
    assert get_rank() == 0
    assert get_world_size() == 1


def test_device_selection_logic():
    """Test device selection priority: CUDA > MPS > CPU."""
    with DistributedManager() as manager:
        if torch.cuda.is_available():
            assert manager.device.type == "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            assert manager.device.type == "mps"
        else:
            assert manager.device.type == "cpu"


@mock.patch('torch.cuda.is_available', return_value=False)
@mock.patch('torch.backends.mps.is_available', return_value=False)
def test_cpu_fallback(mock_mps, mock_cuda):
    """Test that device falls back to CPU when no accelerators available."""
    with DistributedManager() as manager:
        assert manager.device.type == "cpu"


def test_all_gather_object_single_process():
    with DistributedManager() as manager:
        test_obj = {"key": "value", "number": 42}
        result = manager.all_gather_object(test_obj)
        assert result == [test_obj]


def test_all_reduce_single_process():
    with DistributedManager() as manager:
        tensor = torch.tensor([1.0, 2.0, 3.0])
        original = tensor.clone()
        result = manager.all_reduce(tensor)
        assert torch.allclose(result, original)


def test_broadcast_single_process():
    with DistributedManager() as manager:
        tensor = torch.tensor([5.0, 10.0, 15.0])
        original = tensor.clone()
        result = manager.broadcast(tensor, src=0)
        assert torch.allclose(result, original)


def test_set_seed_deterministic():
    with DistributedManager() as manager:
        manager.set_seed(42)
        rand1 = torch.rand(3)
        manager.set_seed(42)
        rand2 = torch.rand(3)
        assert torch.allclose(rand1, rand2)


def test_set_seed_rank_aware():
    """Different ranks should produce different random numbers with the same base seed."""
    with DistributedManager() as manager1:
        manager1.set_seed(42)
        rand1 = torch.rand(3)

    with DistributedManager() as manager2:
        from tunalab import distributed as dist_module
        dist_module._DIST_STATE["rank"] = 1
        try:
            manager2.set_seed(42)
            rand2 = torch.rand(3)
        finally:
            dist_module._DIST_STATE["rank"] = 0

    assert not torch.allclose(rand1, rand2)


# ---------------------------------------------------------------------------
# Distributed environment detection tests
# ---------------------------------------------------------------------------

@mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}, clear=True)
@mock.patch('torch.distributed.is_available', return_value=True)
@mock.patch('torch.distributed.init_process_group')
@mock.patch('torch.distributed.new_group')
@mock.patch('torch.distributed.destroy_process_group')
@mock.patch('torch.cuda.is_available', return_value=True)
@mock.patch('torch.cuda.set_device')
def test_torchrun_environment_detection(
    mock_set_device, mock_cuda_available,
    mock_destroy, mock_new_group, mock_init, mock_available,
):
    """Test detection of torchrun environment variables."""
    mock_new_group.return_value = mock.MagicMock()
    with DistributedManager() as manager:
        mock_init.assert_called_once()
        mock_new_group.assert_called_once_with(backend='gloo')
        assert manager.is_distributed
        assert manager.rank == 0
        assert manager.local_rank == 0
        assert manager.world_size == 2

        assert is_initialized()
        assert get_rank() == 0
        assert get_local_rank() == 0
        assert get_world_size() == 2
        assert is_main_process()

    # cleanup destroys CPU group + default group => 2 calls
    assert mock_destroy.call_count == 2

    assert not is_initialized()
    assert get_rank() == 0
    assert get_world_size() == 1


@mock.patch.dict(os.environ, {
    "SLURM_PROCID": "1",
    "SLURM_LOCALID": "1",
    "SLURM_NTASKS": "4",
    "SLURM_SRUN_COMM_HOST": "node001",
    "SLURM_SRUN_COMM_PORT": "12345",
}, clear=True)
@mock.patch('torch.distributed.is_available', return_value=True)
@mock.patch('torch.distributed.init_process_group')
@mock.patch('torch.distributed.new_group')
@mock.patch('torch.distributed.destroy_process_group')
@mock.patch('torch.cuda.is_available', return_value=True)
@mock.patch('torch.cuda.set_device')
def test_slurm_environment_detection(
    mock_set_device, mock_cuda_available,
    mock_destroy, mock_new_group, mock_init, mock_available,
):
    """Test detection of SLURM environment variables (srun path with COMM_HOST)."""
    mock_new_group.return_value = mock.MagicMock()
    with DistributedManager() as manager:
        mock_init.assert_called_once()
        assert manager.is_distributed
        assert manager.rank == 1
        assert manager.local_rank == 1
        assert manager.world_size == 4

        assert is_initialized()
        assert get_rank() == 1
        assert get_local_rank() == 1
        assert get_world_size() == 4
        assert not is_main_process()

        assert os.environ["MASTER_ADDR"] == "node001"
        assert os.environ["MASTER_PORT"] == "29500"  # fallback: no SLURM_JOB_ID in this test env

    assert mock_destroy.call_count == 2


@mock.patch.dict(os.environ, {
    "SLURM_PROCID": "0",
    "SLURM_LOCALID": "0",
    "SLURM_NTASKS": "2",
    "SLURM_JOB_ID": "99999",
    "SLURM_JOB_NODELIST": "gpu[01-02]",
}, clear=True)
@mock.patch('torch.distributed.is_available', return_value=True)
@mock.patch('torch.distributed.init_process_group')
@mock.patch('torch.distributed.new_group')
@mock.patch('torch.distributed.destroy_process_group')
@mock.patch('torch.cuda.is_available', return_value=True)
@mock.patch('torch.cuda.set_device')
@mock.patch('tunalab.distributed.subprocess.check_output', return_value="gpu01\ngpu02\n")
def test_slurm_fallback_master_addr_port(
    mock_subproc,
    mock_set_device, mock_cuda_available,
    mock_destroy, mock_new_group, mock_init, mock_available,
):
    """Test SLURM detection falls back to scontrol + job-ID-seeded port."""
    mock_new_group.return_value = mock.MagicMock()
    with DistributedManager() as manager:
        mock_init.assert_called_once()
        assert manager.is_distributed
        # MASTER_ADDR resolved from scontrol output
        assert os.environ["MASTER_ADDR"] == "gpu01"
        # MASTER_PORT must be deterministic for job_id=99999
        expected_port = str(__import__('random').Random(99999).randint(20000, 60000))
        assert os.environ["MASTER_PORT"] == expected_port


# ---------------------------------------------------------------------------
# cpu_barrier tests
# ---------------------------------------------------------------------------

def test_cpu_barrier_no_op_single_process():
    """cpu_barrier() must be a no-op when not distributed."""
    # Should not raise
    cpu_barrier()


def test_cpu_barrier_via_manager_static():
    """DistributedManager.cpu_barrier() is callable as a static method."""
    DistributedManager.cpu_barrier()  # should not raise


# ---------------------------------------------------------------------------
# SLURM address/port resolver unit tests
# ---------------------------------------------------------------------------

@mock.patch.dict(os.environ, {"SLURM_SRUN_COMM_HOST": "headnode"}, clear=False)
def test_resolve_slurm_master_addr_prefers_comm_host():
    assert _resolve_slurm_master_addr() == "headnode"


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch('tunalab.distributed.subprocess.check_output', return_value="node42\nnode43\n")
def test_resolve_slurm_master_addr_scontrol(mock_sub):
    os.environ["SLURM_JOB_NODELIST"] = "node[42-43]"
    assert _resolve_slurm_master_addr() == "node42"


@mock.patch.dict(os.environ, {}, clear=True)
def test_resolve_slurm_master_addr_fallback():
    # No SLURM_SRUN_COMM_HOST, no nodelist -> 127.0.0.1
    assert _resolve_slurm_master_addr() == "127.0.0.1"



@mock.patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}, clear=True)
def test_resolve_slurm_master_port_deterministic():
    """Port is stable for a given SLURM_JOB_ID."""
    port1 = _resolve_slurm_master_port()
    port2 = _resolve_slurm_master_port()
    assert port1 == port2
    assert 20000 <= int(port1) <= 60000


# ---------------------------------------------------------------------------
# setup_signal_handlers tests
# ---------------------------------------------------------------------------

def test_setup_signal_handlers_registers():
    """setup_signal_handlers() must not raise and must register handlers."""
    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_sigint = signal.getsignal(signal.SIGINT)
    try:
        setup_signal_handlers()
        # After setup, handlers must not be the default (SIG_DFL) or Python's default (KeyboardInterrupt)
        assert signal.getsignal(signal.SIGTERM) not in (signal.SIG_DFL, None)
        assert signal.getsignal(signal.SIGINT) not in (signal.SIG_DFL, None)
    finally:
        # Restore original handlers to avoid side effects in other tests
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)

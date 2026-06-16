import os
import signal
import subprocess
import random
from datetime import timedelta
from typing import TypeVar, List, Any, Optional
import logging

import torch
import torch.distributed as dist
import numpy as np

logger = logging.getLogger(__name__)


_DIST_STATE = {
    "is_initialized": False,
    "rank": 0,
    "local_rank": 0,
    "world_size": 1,
}

# Secondary Gloo process group for CPU-side collectives.
# When the primary backend is NCCL, long-running GPU-side collectives hold the
# NCCL watchdog thread.  Checkpoint saves on rank 0 (which block on NFS I/O)
# must therefore synchronise via this Gloo group instead.
_CPU_GROUP: Optional[Any] = None


# ---------------------------------------------------------------------------
# Standalone distributed accessors
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Checks if torch.distributed is available in this build of PyTorch."""
    return dist.is_available()


def is_initialized() -> bool:
    """Returns True if the distributed process group has been initialized."""
    return _DIST_STATE["is_initialized"]


def get_rank() -> int:
    """Gets the rank of the current process, defaulting to 0 if not in a distributed context."""
    return _DIST_STATE["rank"]


def get_local_rank() -> int:
    """Gets the local rank of the current process, defaulting to 0 if not in a distributed context."""
    return _DIST_STATE["local_rank"]


def get_world_size() -> int:
    """Gets the total number of processes, defaulting to 1 if not in a distributed context."""
    return _DIST_STATE["world_size"]


def is_main_process() -> bool:
    """Returns True if the current process is the main one (rank 0)."""
    return get_rank() == 0


def is_main() -> bool:
    # alias for clarity if users prefer shorter name
    return is_main_process()


def barrier() -> None:
    """Synchronizes all processes via the default group. No-op if not initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cpu_barrier() -> None:
    """CPU-native (Gloo) barrier across all ranks.

    Safe to call while GPU kernels may be in flight — it never touches the
    NCCL group and therefore cannot trigger the NCCL watchdog.  Use this in
    checkpoint code where rank 0 blocks on NFS I/O while other ranks wait.

    Falls back to the default barrier when no secondary Gloo group exists
    (e.g. when the primary backend is already Gloo).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    if _CPU_GROUP is not None:
        dist.barrier(group=_CPU_GROUP)
    else:
        dist.barrier()


def setup_signal_handlers() -> None:
    """Register SIGTERM/SIGINT handlers for graceful multi-node shutdown.

    When SLURM (or the user via Ctrl-C) sends SIGTERM/SIGINT, we clean up
    all process groups before exiting so that other ranks are not left
    blocked inside a collective operation.

    Call this once early in your training script, after logging is set up.
    """
    def _handler(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        rank = get_rank()
        logger.info(f"[Rank {rank}] Received {sig_name} — shutting down gracefully.")
        if is_initialized():
            try:
                _destroy_cpu_group()
                dist.destroy_process_group()
                logger.info(f"[Rank {rank}] Process groups destroyed.")
            except Exception as exc:
                logger.error(f"[Rank {rank}] Error during DDP cleanup: {exc}")
        import sys
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    logger.debug("Signal handlers registered for SIGTERM and SIGINT.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_slurm_master_addr() -> str:
    """Return the hostname for MASTER_ADDR from SLURM environment variables.

    Priority:
      1. ``SLURM_SRUN_COMM_HOST`` (set automatically by ``srun``)
      2. First node from ``scontrol show hostnames $SLURM_JOB_NODELIST``
      3. ``"127.0.0.1"`` as a last resort
    """
    if "SLURM_SRUN_COMM_HOST" in os.environ:
        return os.environ["SLURM_SRUN_COMM_HOST"]
    nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
    if nodelist:
        try:
            out = subprocess.check_output(
                ["scontrol", "show", "hostnames", nodelist],
                text=True, timeout=10,
            )
            first = out.splitlines()[0].strip()
            if first:
                return first
        except Exception as exc:
            logger.warning(f"Could not resolve SLURM node list via scontrol: {exc}")
    return "127.0.0.1"


def _resolve_slurm_master_port() -> str:
    """Return a port string for MASTER_PORT from SLURM environment variables.

    We deliberately do NOT use ``SLURM_SRUN_COMM_PORT``.  That port is the one
    srun itself has already bound for its own communication channel; handing it
    to PyTorch's ``init_process_group`` causes rank 0 to fail to bind (address
    already in use) and hang silently until timeout.

    Instead we derive a fresh port that is:
      1. Stable across all ranks of the same job (seeded by ``SLURM_JOB_ID``)
      2. Unlikely to collide with other concurrent jobs (the range is large)
    """
    job_id = int(os.environ.get("SLURM_JOB_ID", 0))
    if job_id:
        return str(random.Random(job_id).randint(20000, 60000))
    return "29500"


def _destroy_cpu_group() -> None:
    """Destroy the secondary Gloo group if it exists."""
    global _CPU_GROUP
    if _CPU_GROUP is not None:
        try:
            dist.destroy_process_group(_CPU_GROUP)
        except Exception as exc:
            logger.debug(f"Could not destroy CPU group: {exc}")
        _CPU_GROUP = None


# ---------------------------------------------------------------------------
# DistributedManager
# ---------------------------------------------------------------------------

T = TypeVar('T')


class DistributedManager:
    """
    A context manager to handle distributed training environments.

    This class automatically detects and initializes the process group for
    distributed training if run in a torchrun or SLURM environment. It
    manages the device placement and provides convenience methods for
    distributed operations. It also updates a module-level state that can
    be accessed via standalone functions (e.g., `get_rank()`, `is_main_process()`).

    When the NCCL backend is used (GPU training), a secondary Gloo process
    group is created automatically for use in ``cpu_barrier()``, which lets
    checkpoint code safely synchronise across ranks without risking NCCL
    watchdog timeouts.
    """

    def __init__(self):
        self.device: torch.device = torch.device("cpu")

    # Properties read the single shared state to avoid duplication/drift
    @property
    def is_distributed(self) -> bool:
        return is_initialized()

    @property
    def rank(self) -> int:
        return get_rank()

    @property
    def local_rank(self) -> int:
        return get_local_rank()

    @property
    def world_size(self) -> int:
        return get_world_size()

    @property
    def is_main_process(self) -> bool:
        return is_main_process()

    # Static methods emulate torch.distributed API for convenience when using `as dist`
    @staticmethod
    def is_available() -> bool:
        return is_available()

    @staticmethod
    def is_initialized() -> bool:
        return is_initialized()

    @staticmethod
    def get_rank() -> int:
        return get_rank()

    @staticmethod
    def get_local_rank() -> int:
        return get_local_rank()

    @staticmethod
    def get_world_size() -> int:
        return get_world_size()

    @staticmethod
    def is_main() -> bool:
        return is_main_process()

    @staticmethod
    def barrier() -> None:
        barrier()

    @staticmethod
    def cpu_barrier() -> None:
        cpu_barrier()

    def __enter__(self):
        """Initializes the distributed environment."""
        if is_available() and self._is_dist_env():
            self._init_distributed()
        else:
            logger.info("Running in single-process mode")

        self._set_device()
        logger.info(
            f"Process initialized — Rank: {self.rank}/{self.world_size}, "
            f"Device: {self.device}"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleans up the distributed environment."""
        self.cleanup()
        global _CPU_GROUP
        _CPU_GROUP = None
        _DIST_STATE["is_initialized"] = False
        _DIST_STATE["rank"] = 0
        _DIST_STATE["local_rank"] = 0
        _DIST_STATE["world_size"] = 1

    def _is_dist_env(self) -> bool:
        """Checks if the script is running in a *multi-rank* distributed environment.

        Gates on the world size, not merely the presence of the env var: ``srun``
        always sets ``SLURM_NTASKS`` (even ``=1`` for a single-task launch), and a
        world size of 1 is single-process regardless of launcher. Treating
        ``SLURM_NTASKS=1`` as distributed would needlessly initialize a process
        group (and, under bare ``srun``, hang on rendezvous). Only coordinate when
        there is more than one rank.
        """
        world_size = int(
            os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS") or 1
        )
        return world_size > 1

    def _init_distributed(self) -> None:
        """Sets up the process group and updates shared state."""
        global _CPU_GROUP

        if "SLURM_PROCID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
            local_rank = int(os.environ["SLURM_LOCALID"])
            world_size = int(os.environ["SLURM_NTASKS"])

            # Set rendezvous coordinates if not already provided (e.g. by torchrun)
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = _resolve_slurm_master_addr()
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = _resolve_slurm_master_port()

            logger.info(
                f"Detected SLURM environment — Rank {rank}/{world_size}, "
                f"MASTER_ADDR={os.environ['MASTER_ADDR']}, "
                f"MASTER_PORT={os.environ['MASTER_PORT']}"
            )
        elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            logger.info(f"Detected torchrun environment — Rank {rank}/{world_size}")
        else:
            return  # Not a distributed environment

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        device_id = torch.device("cuda", local_rank) if backend == "nccl" else None
        timeout_s = int(os.environ.get("TORCH_DIST_TIMEOUT_SECONDS", 1800))

        logger.info(f"Initializing process group: backend={backend}")
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            device_id=device_id,
            timeout=timedelta(seconds=timeout_s),
        )

        # Create a secondary Gloo group so that cpu_barrier() never touches NCCL.
        if backend == "nccl":
            _CPU_GROUP = dist.new_group(backend="gloo")
            logger.debug("Secondary Gloo process group created for cpu_barrier().")

        _DIST_STATE["is_initialized"] = True
        _DIST_STATE["rank"] = rank
        _DIST_STATE["local_rank"] = local_rank
        _DIST_STATE["world_size"] = world_size
        logger.info("Process group initialized successfully.")

    def _set_device(self) -> None:
        """Sets the device for the current process."""
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
            logger.debug(f"Set CUDA device to {self.device}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.debug("Set device to MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            logger.debug("Set device to CPU")

    def cleanup(self) -> None:
        """Destroys the process groups."""
        if self.is_distributed:
            _destroy_cpu_group()
            dist.destroy_process_group()

    def all_gather_object(self, obj: T) -> List[T]:
        """Gathers a pickleable object from all processes and returns a list."""
        if not self.is_distributed:
            return [obj]
        output_list: List[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(output_list, obj)
        return output_list

    def broadcast_object(self, obj: T, src: int = 0) -> T:
        """Broadcasts a pickleable object from a source rank to all other processes."""
        if not self.is_distributed:
            return obj
        obj_list = [obj] if self.rank == src else [None]
        dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]

    def all_reduce(self, tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        """Reduces the tensor data across all processes."""
        if self.is_distributed:
            dist.all_reduce(tensor, op=op)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcasts a tensor from a source rank to all other processes."""
        if self.is_distributed:
            dist.broadcast(tensor, src=src)
        return tensor

    def set_seed(self, seed: int) -> None:
        """Sets a deterministic, rank-aware seed for reproducibility."""
        final_seed = seed + self.rank
        random.seed(final_seed)
        np.random.seed(final_seed)
        torch.manual_seed(final_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(final_seed)
        logger.info(f"Set seed to {final_seed} (base: {seed}, rank: {self.rank})")

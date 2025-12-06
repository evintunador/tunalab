"""TunaLab: Composable ML experiment infrastructure."""
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Core primitives
from tunalab.configuration import Config, compose_config, load_config
from tunalab.checkpointer import save_checkpoint, load_checkpoint
from tunalab.reproducibility import ReproducibilityManager
from tunalab.device import get_default_device, get_available_devices, to_device, to_dtype
from tunalab.distributed import barrier, is_main_process, DistributedManager
from tunalab.smart_train import smart_train
from tunalab.evaluation import EvaluationRunner, register_handler
from tunalab import tracking

# Protocols
from tunalab.protocols import DaemonHook, StorageBackend, LLMClient, TrainingLoop

__all__ = [
    "Config", "compose_config", "load_config",
    "save_checkpoint", "load_checkpoint",
    "ReproducibilityManager",
    "get_default_device", "get_available_devices", "to_device", "to_dtype",
    "barrier", "is_main_process", "DistributedManager",
    "smart_train",
    "DaemonHook", "StorageBackend", "LLMClient", "TrainingLoop",
    "EvaluationRunner", "register_handler", 
    "tracking",
]

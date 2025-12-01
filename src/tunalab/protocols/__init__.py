"""Protocol definitions for tunalab interfaces."""
from tunalab.protocols.daemon_hook import DaemonHook
from tunalab.protocols.storage_backend import StorageBackend
from tunalab.protocols.llm_client import LLMClient, strip_code_fences
from tunalab.protocols.base_loop import TrainingLoop

__all__ = [
    "DaemonHook", 
    "StorageBackend", 
    "LLMClient", 
    "strip_code_fences", 
    "TrainingLoop"
]


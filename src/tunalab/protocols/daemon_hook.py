"""Protocol for daemon hooks that monitor experiment runs."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class DaemonHook(Protocol):
    """Interface for external processes to monitor experiment runs."""
    
    def on_run_start(self) -> None:
        """Called when the experiment run starts."""
        ...
    
    def on_run_end(self) -> None:
        """Called when the experiment run ends (successfully or not)."""
        ...


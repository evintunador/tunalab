"""Protocol for artifact storage backends."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Interface for artifact storage backends."""
    
    def upload(self, source_dir: str) -> None:
        """Uploads artifacts from a source to a remote destination directory."""
        ...
    
    def download(self, destination_dir: str) -> None:
        """Downloads artifacts from a remote source to a local destination directory."""
        ...


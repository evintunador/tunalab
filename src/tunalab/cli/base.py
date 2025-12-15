from typing import Protocol


class Command(Protocol):
    """Protocol that all CLI commands must implement."""
    
    name: str
    """The command name as it appears in CLI (e.g., 'multi-run')"""
    
    description: str
    """Short description shown in help text"""
    
    @staticmethod
    def main() -> None:
        """Entry point for the command. Called when command is invoked."""
        ...


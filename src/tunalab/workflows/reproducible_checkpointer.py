"""Workflow combining checkpointer with reproducibility metadata."""
from typing import Any
from tunalab.checkpointer import save_checkpoint
from tunalab.reproducibility import ReproducibilityManager


def save_reproducible_checkpoint(
    filepath: str,
    rm: ReproducibilityManager,
    *,
    metadata: dict = None,
    **stateful_objects: Any,
) -> str:
    """
    Save checkpoint with full reproducibility metadata.

    Args:
        filepath: Path where checkpoint should be saved
        rm: ReproducibilityManager instance for the current run
        metadata: Optional user-provided additional metadata to save
        **stateful_objects: Objects with state_dict() method (model, optimizer, etc.)

    Returns:
        Path to the saved checkpoint file

    Example:
        with ReproducibilityManager(output_dir="./outputs", is_main_process=True) as rm:
            # Train...
            save_reproducible_checkpoint(
                "./outputs/checkpoint.pt",
                rm=rm,
                model=model,
                optimizer=optimizer,
                metadata={"extra_key": "value"}
            )
    """
    rm_metadata = {
        "git_info": rm.get_git_info(),
        "rng_state": rm.get_rng_states(),
        "software_environment": rm.software_environment,
        "runtime_environment": rm.runtime_environment,
    }
    # Merge user metadata (if provided), with RM metadata taking precedence on conflict
    user_metadata = metadata.copy() if metadata is not None else {}
    user_metadata.update(rm_metadata)
    return save_checkpoint(filepath, metadata=user_metadata, **stateful_objects)


"""Loss functions for NLP tasks."""

from tunalab.nn_modules.losses.fused_cross_entropy import (
    TorchLinearCELoss,
    FusedLinearCELoss,
    LIGER_AVAILABLE,
)

__all__ = [
    "TorchLinearCELoss",
    "FusedLinearCELoss",
    "LIGER_AVAILABLE",
]


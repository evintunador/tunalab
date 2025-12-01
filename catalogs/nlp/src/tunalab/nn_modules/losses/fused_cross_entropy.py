from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import triton
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    # Dummy class to avoid NameError in type hints or if instantiated (will raise in init)
    class LigerFusedLinearCrossEntropyLoss:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): pass


class TorchLinearCELoss(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param D: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param weight: optional weight tensor (V, D) for weight tying
    """

    def __init__(self, D: int, V: int, dtype: torch.dtype, ignore_index: int = -100, weight: torch.Tensor = None):
        super().__init__()
        self.weight = weight
        if self.weight is None:
            self.lin = torch.nn.Linear(
                in_features=D, out_features=V, bias=False, dtype=dtype
            )
        else:
            if self.weight.shape != (V, D):
                raise ValueError(f"Expected weight shape ({V}, {D}), got {self.weight.shape}")
            self.lin = None
            
        self.ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, x, y):
        if self.lin is not None:
            logits = self.lin(x)
        else:
            logits = F.linear(x, self.weight)
        return self.ce_loss(logits, y)


class FusedLinearCELoss(torch.nn.Module):
    def __init__(self, D: int, V: int, dtype: torch.dtype, ignore_index: int = -100, weight: torch.Tensor = None):
        super().__init__()
        if not LIGER_AVAILABLE:
            raise ImportError("liger_kernel is not installed")

        self.weight = weight
        if self.weight is None:
            self.lin = torch.nn.Linear(
                in_features=D, out_features=V, bias=False, dtype=dtype
            )
        else:
            if self.weight.shape != (V, D):
                raise ValueError(f"Expected weight shape ({V}, {D}), got {self.weight.shape}")
            self.lin = None

        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, x, y):
        w = self.lin.weight if self.lin is not None else self.weight
        return self.ce_loss(w, x, y)

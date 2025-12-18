import torch.nn.functional as F
from torch import nn, Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        return F.rms_norm(x, (x.size(-1),))
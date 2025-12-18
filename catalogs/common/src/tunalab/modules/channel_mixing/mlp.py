import torch
import torch.nn as nn

from tunalab.modules.activations.relu2 import ReLU2
from .fp8_linear import FP8Linear


def next_multiple(x, n):
    """Round x up to the next multiple of n."""
    return int(((int(x) + n - 1) // n) * n)


torch.set_float32_matmul_precision('medium')
torch._dynamo.config.recompile_limit = 100


##################################################
############# PRIMARY PYTORCH MODULE #############
##################################################

class MLP(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int | None = None, 
        out_dim: int | None = None, 
        activation: str = "silu", 
        dropout: float = 0.0, 
        fp8: bool = False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        hidden_dim = hidden_dim if hidden_dim is not None else in_dim * 4
        self.hidden_dim = next_multiple(x=hidden_dim, n=128)

        self.act_str = activation.lower()
        act_registry = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "relu2": ReLU2(),
            "gelu": nn.GELU(),
        }
        self.act_fn = act_registry[self.act_str]

        self.Wup = FP8Linear(in_features=self.in_dim, out_features=self.hidden_dim, fp8=fp8) 
        self.Wdown = FP8Linear(in_features=self.hidden_dim, out_features=self.out_dim, fp8=fp8)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.Wdown(self.act_fn(self.Wup(x))))


########################################################
# PRECOMPILED IMPLEMENTATION FOR TESTING torch.compile #
########################################################

@torch.compile(mode='default')
def fwd(inp, w_up, w_down, act_fn, dropout):
    return dropout(w_down(act_fn(w_up(inp))))

class PreCompiledMLP(MLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fwd(x, self.Wup, self.Wdown, self.act_fn, self.dropout)
    

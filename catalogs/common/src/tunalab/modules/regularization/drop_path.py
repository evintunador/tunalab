import torch
import torch.nn as nn


##################################################
############# PRIMARY PYTORCH MODULE #############
##################################################

def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    
    This implementation is consistent with the following logic:
    - It drops the entire sample with probability `drop_prob`.
    - It scales the output by `1 / (1 - drop_prob)` to maintain the expected value.
    - It broadcasts the mask across all dimensions except the first (batch) dimension.
    
    Args:
        x: Input tensor of shape (B, ...).
        drop_prob: Probability of dropping the path.
        training: Whether the module is in training mode.
        
    Returns:
        The input tensor, randomly zeroed out and scaled.
    """
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    # Handle broadcasting: (B, 1, 1, ...) for input (B, C, H, W) or (B, N, D)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
        
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


########################################################
# PRECOMPILED IMPLEMENTATION FOR TESTING torch.compile #
########################################################

@torch.compile(mode='default')
def compiled_drop_path_fwd(x, drop_prob, training):
    return drop_path(x, drop_prob, training)


class PreCompiledDropPath(DropPath):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return compiled_drop_path_fwd(x, self.drop_prob, self.training)

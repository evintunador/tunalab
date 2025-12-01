import torch
import torch.nn as nn


##################################################
############# PRIMARY PYTORCH MODULE #############
##################################################


class EfficientDropPath(nn.Module):
    """
    Efficient Drop Path (Stochastic Depth) that skips computation for dropped samples.
    Wraps a module and only executes it on kept batch elements.
    
    Constraints:
    - Only works along the batch dimension (dim 0).
    - Drops a constant number of samples per batch (floor(B * drop_prob)).
    - Input shape must be (B, ...). Packed sequences (1, Tokens, D) will likely result in no drops
      unless drop_prob >= 1.0, effectively disabling drop path for that mode (which is intended behavior
      if we only drop at batch level).
    """
    def __init__(self, module: nn.Module, drop_prob: float = 0.):
        super().__init__()
        self.module = module
        self.drop_prob = drop_prob

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.:
            return self.module(x)

        B = x.shape[0]
        # Determine number of samples to keep
        # We ensure deterministic count for a given B
        n_drop = int(B * self.drop_prob)
        
        # If we can't drop at least one sample, we process everything.
        if n_drop == 0:
            return self.module(x)
            
        n_keep = B - n_drop
        
        # Select indices to keep
        perm = torch.randperm(B, device=x.device)
        keep_indices = perm[:n_keep]
        
        # Check for MPS half precision issues
        is_mps_low_precision = (x.device.type == 'mps' and x.dtype in (torch.float16, torch.bfloat16))

        # Gather inputs
        if is_mps_low_precision:
            # Workaround: Cast to float32 for index_select (gather)
            x_kept = x.float()[keep_indices].to(x.dtype)
        else:
            x_kept = x[keep_indices]
        
        # Compute only on kept samples
        out_kept = self.module(x_kept)
        
        # Prepare output
        # We assume output shape has same non-batch dims as out_kept
        out_shape = (B,) + out_kept.shape[1:]
        
        if is_mps_low_precision:
            # Workaround: Perform scatter in float32
            output_fp32 = torch.zeros(out_shape, device=x.device, dtype=torch.float32)
            output_fp32[keep_indices] = out_kept.float()
            output = output_fp32.to(out_kept.dtype)
        else:
            output = out_kept.new_zeros(out_shape)
            # Scatter results back
            output[keep_indices] = out_kept
        
        # Scale to maintain expected value
        # The effective keep rate is n_keep / B
        scale = B / n_keep
        output.mul_(scale)
        
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}'


########################################################
# PRECOMPILED IMPLEMENTATION FOR TESTING torch.compile #
########################################################


class PreCompiledEfficientDropPath(EfficientDropPath):
    def __init__(self, module: nn.Module, drop_prob: float = 0.):
        super().__init__(module, drop_prob)
        # Compile the implementation
        self._forward_impl = torch.compile(self._forward_impl, mode='default')




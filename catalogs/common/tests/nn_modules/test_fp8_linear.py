"""Tests for FP8Linear module."""

import pytest
import torch
import torch.nn as nn
from tunalab.testing import get_tolerances_for_dtype
from tunalab.nn_modules.channel_mixing.fp8_linear import FP8Linear, is_hopper_available


class TestFP8Linear:
    """Test suite for FP8Linear module."""
    
    @pytest.mark.parametrize("in_features,out_features", [
        (128, 128),
        (512, 2048),
        (1024, 512),
    ])
    def test_forward_backward(self, in_features, out_features, device, dtype):
        """Test forward and backward passes."""
        # FP8 only works on Hopper GPUs
        fp8_enabled = is_hopper_available() and device == 'cuda'
        
        module = FP8Linear(
            in_features=in_features,
            out_features=out_features,
            fp8=fp8_enabled
        ).to(device, dtype)
        
        x = torch.randn(2048, in_features, device=device, dtype=dtype, requires_grad=True)
        
        out = module(x)
        assert out.shape == (2048, out_features)
        assert out.dtype == dtype
        
        out.sum().backward()
        assert x.grad is not None
        assert module.weight.grad is not None
    
    @pytest.mark.skipif(not is_hopper_available(), reason="Requires Hopper GPU")
    def test_fp8_vs_standard(self):
        """Test that FP8 and standard modes both work."""
        device = 'cuda'
        dtype = torch.bfloat16
        
        # Standard mode
        module_std = FP8Linear(128, 128, fp8=False).to(device, dtype)
        x = torch.randn(32, 128, device=device, dtype=dtype)
        out_std = module_std(x)
        
        # FP8 mode  
        module_fp8 = FP8Linear(128, 128, fp8=True).to(device, dtype)
        module_fp8.load_state_dict(module_std.state_dict())
        module_fp8.train()  # FP8 only applies in training mode
        out_fp8 = module_fp8(x)
        
        # They should be similar but not exact due to quantization
        assert out_fp8.shape == out_std.shape
        # Loose tolerance for FP8 quantization
        assert torch.allclose(out_fp8, out_std, atol=0.1, rtol=0.1)


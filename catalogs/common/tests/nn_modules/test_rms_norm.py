"""Tests for RMSNorm module."""

import pytest
import torch
import torch.nn.functional as F
from tunalab.testing import compare_modules, get_tolerances_for_dtype
from tunalab.nn_modules.norms.rms_norm import RMSNorm


class TestRMSNorm:
    """Test suite for RMSNorm module."""
    
    @pytest.mark.parametrize("dim", [128, 768, 2048])
    def test_forward_backward(self, dim, device, dtype):
        """Test that forward and backward passes run without errors."""
        module = RMSNorm(dim=dim).to(device, dtype)
        x = torch.randn(8, 32, dim, device=device, dtype=dtype, requires_grad=True)
        
        # Forward pass
        out = module(x)
        
        # Check output properties
        assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
        assert out.dtype == x.dtype, f"Expected dtype {x.dtype}, got {out.dtype}"
        assert out.device == x.device, f"Expected device {x.device}, got {out.device}"
        
        # Backward pass
        out.sum().backward()
        assert x.grad is not None, "Input gradient should exist"

    @pytest.mark.parametrize("dim", [128, 768, 2048])
    def test_matches_torch_rms_norm(self, dim, device, dtype):
        """Compare against PyTorch's F.rms_norm reference implementation."""
        module = RMSNorm(dim=dim).to(device, dtype)
        x = torch.randn(8, 32, dim, device=device, dtype=dtype, requires_grad=True)
        
        # Get module output
        out = module(x)
        
        # Get reference output
        ref = F.rms_norm(x, (x.size(-1),))
        
        # Compare
        tols = get_tolerances_for_dtype(dtype)
        assert torch.allclose(out, ref, **tols), (
            f"Output mismatch: max_abs_diff={(out - ref).abs().max():.6e}"
        )
        
        # Compare gradients
        out.sum().backward()
        x_grad_test = x.grad.clone()
        x.grad = None
        
        ref.sum().backward()
        x_grad_ref = x.grad
        
        assert torch.allclose(x_grad_test, x_grad_ref, **tols), (
            f"Gradient mismatch: max_abs_diff={(x_grad_test - x_grad_ref).abs().max():.6e}"
        )

    @pytest.mark.parametrize("batch_size,seq_len,dim", [
        (1, 1, 128),
        (4, 16, 256),
        (16, 128, 512),
    ])
    def test_various_shapes(self, batch_size, seq_len, dim, device, dtype):
        """Test module works with various input shapes."""
        module = RMSNorm(dim=dim).to(device, dtype)
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
        
        out = module(x)
        assert out.shape == x.shape


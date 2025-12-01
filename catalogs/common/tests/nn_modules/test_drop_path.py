"""Tests for DropPath module."""

import pytest
import torch
from tunalab.testing import compare_modules, get_tolerances_for_dtype
from tunalab.nn_modules.regularization.drop_path import DropPath, PreCompiledDropPath


class TestDropPath:
    """Test suite for DropPath module."""
    
    @pytest.mark.parametrize("drop_prob", [0.0, 0.1, 0.5])
    def test_forward(self, drop_prob, device, dtype):
        """Test forward pass."""
        module = DropPath(drop_prob=drop_prob).to(device, dtype)
        x = torch.randn(8, 128, 768, device=device, dtype=dtype)
        
        # Eval mode - should pass through unchanged
        module.eval()
        out = module(x)
        assert torch.allclose(out, x)
        
        # Train mode
        module.train()
        out = module(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        
        # With drop_prob=0, should still pass through
        if drop_prob == 0.0:
            assert torch.allclose(out, x)
    
    def test_statistical_properties(self, device):
        """Test that drop path preserves expected value."""
        drop_prob = 0.5
        module = DropPath(drop_prob=drop_prob).to(device)
        module.train()
        
        # Run many times to verify statistics
        x = torch.ones(100, 10, device=device)
        results = []
        for _ in range(1000):
            out = module(x)
            results.append(out.mean().item())
        
        mean_result = sum(results) / len(results)
        # Should be close to 1.0 due to rescaling
        assert abs(mean_result - 1.0) < 0.1
    
    @pytest.mark.parametrize("shape", [
        (4, 128, 768),
        (8, 256),
    ])
    def test_various_shapes(self, shape, device, dtype):
        """Test with various input shapes."""
        module = DropPath(drop_prob=0.1).to(device, dtype)
        x = torch.randn(*shape, device=device, dtype=dtype)
        
        module.eval()
        out = module(x)
        assert out.shape == x.shape
    
    def test_precompiled_matches_eager_eval(self, device, dtype):
        """Test PreCompiledDropPath matches DropPath in eval mode."""
        if device == "cpu":
            pytest.skip("torch.compile not optimized for CPU")
        
        drop_prob = 0.1
        ref = DropPath(drop_prob=drop_prob).to(device, dtype).eval()
        test = PreCompiledDropPath(drop_prob=drop_prob).to(device, dtype).eval()
        
        x = torch.randn(8, 128, 768, device=device, dtype=dtype, requires_grad=True)
        
        # In eval mode, output should be identical
        result = compare_modules(ref, test, (x,), get_tolerances_for_dtype(dtype))
        assert result.passed, f"PreCompiledDropPath eval mismatch: {result.failures}"


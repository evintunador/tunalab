"""Tests for EfficientDropPath module."""

import pytest
import torch
import torch.nn as nn
from tunalab.nn_modules.regularization.efficient_drop_path import EfficientDropPath, PreCompiledEfficientDropPath


class TestEfficientDropPath:
    """Test suite for EfficientDropPath module."""
    
    @pytest.mark.parametrize("drop_prob", [0.0, 0.1, 0.5])
    def test_forward_eval_mode(self, drop_prob, device, dtype):
        """Test that eval mode passes through module unchanged."""
        inner_module = nn.Identity()
        module = EfficientDropPath(inner_module, drop_prob=drop_prob).to(device, dtype).eval()
        
        x = torch.randn(8, 128, 768, device=device, dtype=dtype)
        out = module(x)
        
        # In eval mode, should be identical to direct module call
        assert torch.allclose(out, inner_module(x))
    
    @pytest.mark.parametrize("drop_prob", [0.0, 0.3, 0.5])
    def test_training_mode_shape(self, drop_prob, device, dtype):
        """Test that output shape is correct in training mode."""
        inner_module = nn.Identity()
        module = EfficientDropPath(inner_module, drop_prob=drop_prob).to(device, dtype).train()
        
        x = torch.randn(10, 128, device=device, dtype=dtype)
        out = module(x)
        
        assert out.shape == x.shape
        assert out.dtype == x.dtype
    
    def test_statistical_properties(self, device):
        """Test that scaling preserves expected value."""
        inner_module = nn.Identity()
        module = EfficientDropPath(inner_module, drop_prob=0.5).to(device).train()
        
        x = torch.ones(100, 10, device=device)
        results = []
        for _ in range(100):
            out = module(x)
            results.append(out.mean().item())
        
        mean_result = sum(results) / len(results)
        # Should be close to 1.0 due to rescaling
        assert abs(mean_result - 1.0) < 0.2
    
    def test_with_real_module(self, device, dtype):
        """Test with a real inner module, not just Identity."""
        inner = nn.Linear(64, 64).to(device, dtype)
        module = EfficientDropPath(inner, drop_prob=0.2).to(device, dtype)
        
        x = torch.randn(8, 64, device=device, dtype=dtype)
        
        module.eval()
        out_eval = module(x)
        assert out_eval.shape == (8, 64)
        
        module.train()
        out_train = module(x)
        assert out_train.shape == (8, 64)


"""Tests for ReLU2 activation."""

import pytest
import torch
from tunalab.nn_modules.activations.relu2 import ReLU2


class TestReLU2:
    """Test suite for ReLU2 activation."""
    
    def test_forward(self, device, dtype):
        """Test forward pass."""
        module = ReLU2().to(device, dtype)
        x = torch.randn(4, 16, device=device, dtype=dtype, requires_grad=True)
        
        out = module(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        
        # Verify computation: relu(x).clamp(max=255).square()
        expected = torch.relu(x).clamp(max=255.0).square()
        assert torch.allclose(out, expected)
        
        # Test backward
        out.sum().backward()
        assert x.grad is not None
    
    def test_values(self):
        """Test specific input values."""
        module = ReLU2()
        
        # Negative values should be zeroed then squared = 0
        x = torch.tensor([-1.0, -2.0])
        out = module(x)
        assert torch.all(out == 0.0)
        
        # Small positive values
        x = torch.tensor([1.0, 2.0])
        out = module(x)
        assert torch.allclose(out, torch.tensor([1.0, 4.0]))
        
        # Values over 255 should be clamped
        x = torch.tensor([300.0])
        out = module(x)
        assert torch.allclose(out, torch.tensor([255.0**2]))


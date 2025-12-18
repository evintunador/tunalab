import torch

from tunalab.modules.activations.relu2 import ReLU2


class TestReLU2:
    def test_forward(self, device, dtype):
        """Test forward pass."""
        module = ReLU2().to(device, dtype)
        x = torch.randn(4, 16, device=device, dtype=dtype, requires_grad=True)
        
        out = module(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        
        expected = torch.relu(x).clamp(max=255.0).square()
        assert torch.allclose(out, expected)
        
        out.sum().backward()
        assert x.grad is not None
    
    def test_values(self):
        module = ReLU2()
        
        x = torch.tensor([-1.0, -2.0])
        out = module(x)
        assert torch.all(out == 0.0)
        
        x = torch.tensor([1.0, 2.0])
        out = module(x)
        assert torch.allclose(out, torch.tensor([1.0, 4.0]))
        
        x = torch.tensor([300.0])
        out = module(x)
        assert torch.allclose(out, torch.tensor([255.0**2]))


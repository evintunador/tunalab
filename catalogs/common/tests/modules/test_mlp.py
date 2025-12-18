import pytest
import torch

from tunalab.testing import compare_modules, get_tolerances_for_dtype
from tunalab.modules.channel_mixing.mlp import MLP, PreCompiledMLP


class TestMLP:
    @pytest.mark.parametrize("dim", [128, 512])
    @pytest.mark.parametrize("activation", ["relu", "silu", "gelu"])
    def test_forward_backward(self, dim, activation, device, dtype):
        module = MLP(in_dim=dim, activation=activation).to(device, dtype)
        x = torch.randn(8, 32, dim, device=device, dtype=dtype, requires_grad=True)
        
        out = module(x)
        assert out.shape[:-1] == x.shape[:-1]
        assert out.dtype == x.dtype
        
        out.sum().backward()
        assert x.grad is not None
        for p in module.parameters():
            assert p.grad is not None
    
    @pytest.mark.parametrize("dim", [128, 512])
    def test_precompiled_matches_eager(self, dim, device, dtype):
        if device == "cpu":
            pytest.skip("torch.compile not optimized for CPU")
        
        ref = MLP(in_dim=dim).to(device, dtype)
        test = PreCompiledMLP(in_dim=dim).to(device, dtype)
        test.load_state_dict(ref.state_dict())
        
        x = torch.randn(8, 32, dim, device=device, dtype=dtype, requires_grad=True)
        
        tols = get_tolerances_for_dtype(dtype)
        if dtype == torch.float16 and device == "mps":
            # torch.compile on MPS with FP16 can have larger numerical differences
            tols = {'atol': 0.1, 'rtol': 1.0}
        
        result = compare_modules(ref, test, (x,), tols, copy_state_dict=False)
        assert result.passed, f"PreCompiledMLP mismatch: {result.failures}"
    
    @pytest.mark.parametrize("in_dim,hidden_dim,out_dim", [
        (256, None, None),
        (256, 1024, None),
        (256, 1024, 128),
    ])
    def test_various_dimensions(self, in_dim, hidden_dim, out_dim, device, dtype):
        module = MLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim
        ).to(device, dtype)
        
        x = torch.randn(4, 16, in_dim, device=device, dtype=dtype)
        out = module(x)
        
        expected_out_dim = out_dim if out_dim is not None else in_dim
        assert out.shape[-1] == expected_out_dim


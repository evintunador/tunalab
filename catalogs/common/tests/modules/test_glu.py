import pytest
import torch

from tunalab.testing import compare_modules, get_tolerances_for_dtype
from tunalab.modules.channel_mixing.glu import GLU, PreCompiledGLU


class TestGLU:
    @pytest.mark.parametrize("dim", [128, 512])
    @pytest.mark.parametrize("activation", ["relu", "silu", "relu2"])
    def test_forward_backward(self, dim, activation, device, dtype):
        module = GLU(in_dim=dim, activation=activation).to(device, dtype)
        x = torch.randn(8, 32, dim, device=device, dtype=dtype, requires_grad=True)
        
        out = module(x)
        assert out.shape[:-1] == x.shape[:-1], f"Batch/seq dims should match"
        assert out.dtype == x.dtype
        
        out.sum().backward()
        assert x.grad is not None
        for p in module.parameters():
            assert p.grad is not None

    @pytest.mark.parametrize("dim", [128, 512])
    @pytest.mark.parametrize("activation", ["relu", "silu"])
    def test_precompiled_matches_eager(self, dim, activation, device, dtype):
        if device == "cpu":
            pytest.skip("torch.compile not optimized for CPU")
        
        ref_module = GLU(in_dim=dim, activation=activation).to(device, dtype)
        test_module = PreCompiledGLU(in_dim=dim, activation=activation).to(device, dtype)
        
        test_module.load_state_dict(ref_module.state_dict())
        
        x = torch.randn(8, 32, dim, device=device, dtype=dtype, requires_grad=True)
        
        tols = get_tolerances_for_dtype(dtype)
        if dtype == torch.float16 and device == "mps":
            # torch.compile on MPS with FP16 can have larger numerical differences
            tols = {'atol': 0.1, 'rtol': 1.0}
        
        result = compare_modules(ref_module, test_module, (x,), tols, copy_state_dict=False)
        
        assert result.passed, f"PreCompiledGLU mismatch: {result.failures}"

    @pytest.mark.parametrize("in_dim,hidden_dim,out_dim", [
        (256, None, None),  # default hidden_dim
        (256, 512, None),   # custom hidden_dim
        (256, 512, 128),    # custom out_dim
    ])
    def test_various_dimensions(self, in_dim, hidden_dim, out_dim, device, dtype):
        module = GLU(
            in_dim=in_dim, 
            hidden_dim=hidden_dim, 
            out_dim=out_dim
        ).to(device, dtype)
        
        x = torch.randn(4, 16, in_dim, device=device, dtype=dtype)
        out = module(x)
        
        expected_out_dim = out_dim if out_dim is not None else in_dim
        assert out.shape[-1] == expected_out_dim

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.5])
    def test_dropout(self, dropout, device, dtype):
        dim = 128
        module = GLU(in_dim=dim, dropout=dropout).to(device, dtype)
        x = torch.randn(8, 32, dim, device=device, dtype=dtype)
        
        module.train()
        out = module(x)
        assert out.shape[:-1] == x.shape[:-1]


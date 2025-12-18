import pytest
import torch

from tunalab.modules.sequence_mixing.causal_self_attention import CausalSelfAttention


class TestCausalSelfAttention:
    @pytest.mark.parametrize("n_embd,n_head", [
        (256, 4),
        (512, 8),
        (768, 12),
    ])
    def test_forward_backward(self, n_embd, n_head, device, dtype):
        module = CausalSelfAttention(n_embd=n_embd, n_head=n_head).to(device, dtype)
        
        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, n_embd, device=device, dtype=dtype, requires_grad=True)
        
        out = module(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        
        out.sum().backward()
        assert x.grad is not None
        for p in module.parameters():
            assert p.grad is not None
    
    def test_causal_mask(self, device):
        n_embd, n_head = 128, 4
        module = CausalSelfAttention(n_embd=n_embd, n_head=n_head).to(device).eval()
        
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, n_embd, device=device)
        
        with torch.no_grad():
            out = module(x)
        
        assert out.shape == (batch_size, seq_len, n_embd)


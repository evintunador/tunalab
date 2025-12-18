import pytest
import torch

from tunalab.modules.sequence_mixing.flex_self_attention import FlexSelfAttention


class TestFlexSelfAttention:
    @pytest.mark.parametrize("n_embd,n_head", [
        (256, 4),
        (512, 8),
    ])
    def test_forward_backward(self, n_embd, n_head, device, dtype):
        if device != 'cuda':
            pytest.skip("FlexAttention only supports CUDA")
        
        module = FlexSelfAttention(n_embd=n_embd, n_head=n_head).to(device, dtype)
        
        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, n_embd, device=device, dtype=dtype, requires_grad=True)
        
        out = module(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        
        out.sum().backward()
        assert x.grad is not None


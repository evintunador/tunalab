import pytest
import torch
import torch.nn as nn
from tunalab.modules.losses.fused_cross_entropy import (
    TorchLinearCELoss,
    FusedLinearCELoss,
    LIGER_AVAILABLE,
)
from tunalab.testing import get_tolerances_for_dtype, compare_modules
from pathlib import Path
import shutil


TEST_ARTIFACTS_DIR = Path(__file__).parent.parent / "test_artifacts" / "modules"


@pytest.fixture(scope="session", autouse=True)
def clear_test_error_maps():
    if TEST_ARTIFACTS_DIR.exists():
        shutil.rmtree(TEST_ARTIFACTS_DIR)
    TEST_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class TestTorchLinearCELoss:
    @pytest.mark.parametrize("D,V", [
        (128, 1000),
        (512, 5000),
    ])
    def test_forward_backward(self, D, V, device, dtype):
        module = TorchLinearCELoss(D=D, V=V, dtype=dtype).to(device)
        
        batch_size, seq_len = 8, 32
        x = torch.randn(batch_size * seq_len, D, device=device, dtype=dtype, requires_grad=True)
        y = torch.randint(0, V, (batch_size * seq_len,), device=device)
        
        loss = module(x, y)
        
        assert loss.shape == ()  # Scalar loss
        assert loss.dtype == dtype
        assert loss.requires_grad
        
        loss.backward()
        assert x.grad is not None
        assert module.lin.weight.grad is not None
    
    @pytest.mark.parametrize("D,V", [(128, 1000)])
    def test_ignore_index(self, D, V, device, dtype):
        ignore_idx = -100
        module = TorchLinearCELoss(D=D, V=V, dtype=dtype, ignore_index=ignore_idx).to(device)
        
        batch_size, seq_len = 4, 16
        total_tokens = batch_size * seq_len
        x = torch.randn(total_tokens, D, device=device, dtype=dtype, requires_grad=True)
        
        # Create labels with some ignore_index values
        y = torch.randint(0, V, (total_tokens,), device=device)
        y[:seq_len] = ignore_idx  # First sequence should be ignored
        
        loss = module(x, y)
        
        # Loss should still be computed (averaged over non-ignored tokens)
        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    @pytest.mark.parametrize("D,V", [(128, 1000)])
    def test_weight_tying(self, D, V, device, dtype):
        shared_weight = torch.randn(V, D, device=device, dtype=dtype, requires_grad=True)
        
        module = TorchLinearCELoss(D=D, V=V, dtype=dtype, weight=shared_weight).to(device)
        
        batch_size, seq_len = 4, 16
        total_tokens = batch_size * seq_len
        x = torch.randn(total_tokens, D, device=device, dtype=dtype, requires_grad=True)
        y = torch.randint(0, V, (total_tokens,), device=device)
        
        loss = module(x, y)
        loss.backward()
        
        assert shared_weight.grad is not None
        assert module.lin is None  # No internal linear layer when weight tying


@pytest.mark.skipif(not LIGER_AVAILABLE, reason="liger_kernel not installed")
class TestFusedLinearCELoss:
    @pytest.mark.parametrize("D,V", [
        (128, 1000),
        (512, 5000),
    ])
    def test_forward_backward(self, D, V, device, dtype):
        if device != 'cuda':
            pytest.skip("FusedLinearCELoss requires CUDA")
        
        module = FusedLinearCELoss(D=D, V=V, dtype=dtype).to(device)
        
        batch_size, seq_len = 8, 32
        total_tokens = batch_size * seq_len
        x = torch.randn(total_tokens, D, device=device, dtype=dtype, requires_grad=True)
        y = torch.randint(0, V, (total_tokens,), device=device)
        
        loss = module(x, y)
        
        assert loss.shape == ()  # Scalar loss
        assert loss.requires_grad
        
        loss.backward()
        assert x.grad is not None
        assert module.lin.weight.grad is not None
    
    @pytest.mark.parametrize("D,V", [
        (128, 1000),
        (256, 2000),
    ])
    def test_matches_torch_implementation(self, D, V, device, dtype):
        if device != 'cuda':
            pytest.skip("FusedLinearCELoss requires CUDA")
        
        ref_module = TorchLinearCELoss(D=D, V=V, dtype=dtype).to(device)
        test_module = FusedLinearCELoss(D=D, V=V, dtype=dtype).to(device)
        
        test_module.lin.weight.data.copy_(ref_module.lin.weight.data)
        
        batch_size, seq_len = 8, 32
        total_tokens = batch_size * seq_len
        x = torch.randn(total_tokens, D, device=device, dtype=dtype, requires_grad=True)
        y = torch.randint(0, V, (total_tokens,), device=device)
        
        ref_loss = ref_module(x.clone().detach().requires_grad_(True), y)
        test_loss = test_module(x.clone().detach().requires_grad_(True), y)
        
        tols = get_tolerances_for_dtype(dtype)
        if dtype in [torch.float16, torch.bfloat16]:
            tols = {'atol': 1e-2, 'rtol': 1e-1}
        
        assert torch.allclose(ref_loss, test_loss, **tols), \
            f"Loss mismatch: ref={ref_loss.item():.6f}, test={test_loss.item():.6f}"
    
    @pytest.mark.parametrize("D,V", [(128, 1000)])
    def test_weight_tying(self, D, V, device, dtype):
        if device != 'cuda':
            pytest.skip("FusedLinearCELoss requires CUDA")
        
        shared_weight = torch.randn(V, D, device=device, dtype=dtype, requires_grad=True)
        
        module = FusedLinearCELoss(D=D, V=V, dtype=dtype, weight=shared_weight).to(device)
        
        batch_size, seq_len = 4, 16
        total_tokens = batch_size * seq_len
        x = torch.randn(total_tokens, D, device=device, dtype=dtype, requires_grad=True)
        y = torch.randint(0, V, (total_tokens,), device=device)
        
        loss = module(x, y)
        loss.backward()
        
        assert shared_weight.grad is not None
        assert module.lin is None  # No internal linear layer when weight tying


@pytest.mark.skipif(not LIGER_AVAILABLE, reason="liger_kernel not installed")
class TestFusedVsTorch:
    @pytest.mark.parametrize("D,V", [(128, 1000)])
    @pytest.mark.parametrize("use_weight_tying", [False, True])
    def test_gradient_equivalence(self, D, V, use_weight_tying, device, dtype):
        if device != 'cuda':
            pytest.skip("FusedLinearCELoss requires CUDA")
        
        torch.manual_seed(42)
        
        if use_weight_tying:
            shared_weight = torch.randn(V, D, device=device, dtype=dtype, requires_grad=True)
            ref_module = TorchLinearCELoss(D=D, V=V, dtype=dtype, weight=shared_weight).to(device)
            
            shared_weight_fused = shared_weight.clone().detach().requires_grad_(True)
            test_module = FusedLinearCELoss(D=D, V=V, dtype=dtype, weight=shared_weight_fused).to(device)
        else:
            ref_module = TorchLinearCELoss(D=D, V=V, dtype=dtype).to(device)
            test_module = FusedLinearCELoss(D=D, V=V, dtype=dtype).to(device)
            test_module.lin.weight.data.copy_(ref_module.lin.weight.data)
        
        batch_size, seq_len = 8, 32
        total_tokens = batch_size * seq_len
        x_ref = torch.randn(total_tokens, D, device=device, dtype=dtype, requires_grad=True)
        x_test = x_ref.clone().detach().requires_grad_(True)
        y = torch.randint(0, V, (total_tokens,), device=device)
        
        ref_loss = ref_module(x_ref, y)
        ref_loss.backward()
        
        test_loss = test_module(x_test, y)
        test_loss.backward()
        
        tols = get_tolerances_for_dtype(dtype)
        if dtype in [torch.float16, torch.bfloat16]:
            tols = {'atol': 1e-2, 'rtol': 1e-1}
        
        assert torch.allclose(ref_loss, test_loss, **tols), \
            f"Loss mismatch: ref={ref_loss.item():.6f}, test={test_loss.item():.6f}"
        
        assert torch.allclose(x_ref.grad, x_test.grad, **tols), \
            f"Input gradient mismatch: max_diff={(x_ref.grad - x_test.grad).abs().max():.6e}"
        
        if use_weight_tying:
            assert torch.allclose(shared_weight.grad, shared_weight_fused.grad, **tols), \
                f"Weight gradient mismatch: max_diff={(shared_weight.grad - shared_weight_fused.grad).abs().max():.6e}"
        else:
            assert torch.allclose(ref_module.lin.weight.grad, test_module.lin.weight.grad, **tols), \
                f"Weight gradient mismatch: max_diff={(ref_module.lin.weight.grad - test_module.lin.weight.grad).abs().max():.6e}"


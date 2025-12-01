import pytest
import torch.optim
from tunalab.testing import run_learning_test


class TestAdamW:
    @pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    def test_learning(self, device):
        """Test that AdamW can learn a simple task."""
        result = run_learning_test(
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={
                'lr': 1e-3,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 1e-2
            },
            device=device,
        )
        
        assert result['loss_reduction'] > 0.2  # At least 20% improvement


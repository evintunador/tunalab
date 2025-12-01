import pytest
import torch
from tunalab.testing import run_learning_test
from tunalab.optimizers.muon import MuonTestAdapter


class TestMuon:
    @pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    def test_learning(self, device):
        """Test that Muon can learn a simple task."""
        result = run_learning_test(
            optimizer_class=MuonTestAdapter,
            optimizer_kwargs={
                'muon_lr': 0.02,
                'adamw_lr': 3e-4,
                'adamw_betas': (0.9, 0.95),
                'momentum': 0.95,
                'nesterov': True,
                'ns_steps': 5,
                'weight_decay': 0.01,
            },
            device=device,
        )
        
        assert result['loss_reduction'] > 0.2  # At least 20% improvement


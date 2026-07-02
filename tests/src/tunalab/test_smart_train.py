"""
Comprehensive tests for smart_train.py functionality.
Tests the visual validations as proper unit tests.
"""

from typing import Dict, Any
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
from unittest.mock import MagicMock
from importlib import reload

from tunalab.testing import SimpleTestTrainingModel, get_available_devices
from tunalab.smart_train import smart_train, select_features_from_kwargs

AVAILABLE_DEVICES = get_available_devices()


# Helper function to create test data
def _create_test_data(device: str):
    """Create test data for smart_train tests."""
    torch.manual_seed(42)
    X = torch.randn(16, 4).to(device)
    y = torch.randint(0, 2, (16,)).to(device)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=8)
    backbone = nn.Linear(4, 2)
    loss_fn = nn.CrossEntropyLoss()
    model = SimpleTestTrainingModel(backbone, loss_fn).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, optimizer, dataloader


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_smart_train_no_features(device: str):
    """Test smart_train with no additional features."""
    
    model, optimizer, dataloader = _create_test_data(device)
    
    result = smart_train(model, optimizer, dataloader)
    
    assert isinstance(result, dict)
    assert 'model' in result
    assert isinstance(result['model'], nn.Module)


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_smart_train_single_feature(device: str):
    """Test smart_train with single feature (direct execution)."""
    
    model, optimizer, dataloader = _create_test_data(device)
    
    result = smart_train(
        model, optimizer, dataloader,
        accum_steps=2
    )
    
    assert isinstance(result, dict)
    assert 'model' in result
    assert isinstance(result['model'], nn.Module)


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_smart_train_multi_feature(device: str):
    """Test smart_train with multiple features (uses cached or real compilation)."""
    
    model, optimizer, dataloader = _create_test_data(device)
    
    # This test uses the real compilation chain (or cached results)
    # We're just verifying that smart_train works with multiple features
    result = smart_train(
        model, optimizer, dataloader,
        accum_steps=2, track_loss=True
    )
    
    # Assert that training completed and returned a valid result
    assert isinstance(result, dict)
    assert 'model' in result
    assert isinstance(result['model'], nn.Module)


def test_smart_train_unknown_kwargs():
    """Test that smart_train rejects unknown kwargs."""
    
    # This test is device-agnostic, so we can just use CPU
    model, optimizer, dataloader = _create_test_data('cpu')
    
    with pytest.raises(ValueError) as exc_info:
        smart_train(
            model, optimizer, dataloader,
            unknown_parameter=123
        )
    
    error_msg = str(exc_info.value)
    assert "Unknown kwargs" in error_msg


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_smart_train_none_filtering(device: str):
    """Test that smart_train filters out None values."""

    model, optimizer, dataloader = _create_test_data(device)

    # Should work the same as no additional features
    result = smart_train(
        model, optimizer, dataloader,
        accum_steps=None, val_loader=None
    )

    assert isinstance(result, dict)
    assert 'model' in result


# ---------------------------------------------------------------------------
# Unit tests for select_features_from_kwargs
# ---------------------------------------------------------------------------

_SENTINEL = object()  # stand-in for val_loader (just needs to be non-None)


def test_select_features_checkpoint_best_model_selected():
    """Full training kwargs must select checkpoint_best_model, not just device+multi_epoch.

    This is the regression test for the broken _select_most_specific_from_group
    logic that dropped checkpoint_best_model whenever every kwarg in the
    transitive overlap group was shared by multiple features.
    """
    kwargs = {
        'enable_logging': True,
        'save_best_model': True,
        'val_loader': _SENTINEL,
        'val_interval': 50,
        'output_dir': '/tmp',
        'device': 'cuda',
        'use_tqdm': True,
        'num_epochs': 1,
        'accum_steps': 1,
    }
    features = select_features_from_kwargs(kwargs)
    assert 'checkpoint_best_model' in features, (
        f"checkpoint_best_model missing from {features}"
    )


def test_select_features_validation_subsumed_by_checkpoint():
    """validation must not appear alongside checkpoint_best_model (it is subsumed)."""
    kwargs = {
        'save_best_model': True,
        'val_loader': _SENTINEL,
        'val_interval': 10,
        'output_dir': '/tmp',
    }
    features = select_features_from_kwargs(kwargs)
    assert 'checkpoint_best_model' in features
    assert 'validation' not in features, (
        f"validation should be subsumed by checkpoint_best_model, got {features}"
    )


def test_select_features_validation_only_without_save_best_model():
    """Without save_best_model, validation (not checkpoint_best_model) should be selected."""
    kwargs = {
        'val_loader': _SENTINEL,
        'val_interval': 10,
        'device': 'cuda',
    }
    features = select_features_from_kwargs(kwargs)
    assert 'validation' in features
    assert 'checkpoint_best_model' not in features


def test_select_features_superset_subsumes_subsets():
    """A more-specific feature (superset kwargs) must subsume all less-specific ones.

    Mirrors the real bucket_state_checkpoint > checkpoint_best_model > validation
    hierarchy using a mock discovery so the test is self-contained in tunalab.
    """
    from unittest.mock import patch

    fake_features = {
        'feat_small': {'k1', 'k2'},           # validation-like
        'feat_medium': {'k1', 'k2', 'k3'},    # checkpoint_best_model-like
        'feat_large': {'k1', 'k2', 'k3', 'k4'},  # bucket_state_checkpoint-like
        'feat_orthogonal': {'k5'},             # unrelated — should still appear
    }
    fake_kwarg_to_features = {}
    for feat, ks in fake_features.items():
        for k in ks:
            fake_kwarg_to_features.setdefault(k, set()).add(feat)

    user_kwargs = {'k1': 1, 'k2': 2, 'k3': 3, 'k4': 4, 'k5': 5}

    with patch('tunalab.smart_train.discover_atomic_feature_mappings',
               return_value=(fake_features, fake_kwarg_to_features)):
        features = select_features_from_kwargs(user_kwargs)

    assert 'feat_large' in features
    assert 'feat_orthogonal' in features
    assert 'feat_medium' not in features, "feat_medium should be subsumed by feat_large"
    assert 'feat_small' not in features, "feat_small should be subsumed by feat_large"


def test_select_features_idempotent():
    """Calling select_features_from_kwargs twice must return identical results.

    Regression test for the 133542f mutation bug where _find_overlapping_feature_groups
    mutated the feature_to_kwargs sets in-place, causing different results on
    repeated calls.
    """
    kwargs = {
        'save_best_model': True,
        'val_loader': _SENTINEL,
        'val_interval': 10,
        'output_dir': '/tmp',
        'device': 'cuda',
    }
    result1 = select_features_from_kwargs(kwargs)
    result2 = select_features_from_kwargs(kwargs)
    assert result1 == result2, (
        f"select_features_from_kwargs is not idempotent: {result1} vs {result2}"
    )


def test_dropped_kwarg_warns(caplog):
    """A kwarg consumed by no selected feature must produce a loud warning.

    Guards the silent-footgun class: a user passes a kwarg that no *selected*
    feature declares, so the behaviour it was meant to enable is silently
    dropped.  Here ``accum_steps`` selects ``grad_accum`` but ``patience``
    (an early_stopping kwarg, and early_stopping also needs ``val_loader``) is
    orphaned — select_features_from_kwargs must warn that it is inactive.
    """
    import logging
    kwargs = {
        'accum_steps': 2,     # → selects grad_accum
        'patience': 5,        # early_stopping-only; early_stopping not fully satisfied
    }
    with caplog.at_level(logging.WARNING, logger='tunalab.smart_train'):
        selected = select_features_from_kwargs(kwargs)
    assert 'grad_accum' in selected
    assert 'early_stopping' not in selected
    assert any('patience' in r.message and 'NOT consumed' in r.message
               for r in caplog.records), "expected a warning about the dropped patience kwarg"


def test_no_spurious_dropped_kwarg_warning(caplog):
    """A clean kwarg set (every kwarg consumed) must NOT warn."""
    import logging
    kwargs = {
        'save_best_model': True,
        'val_loader': _SENTINEL,       # singular → checkpoint_best_model
        'val_interval': 10,
        'output_dir': '/tmp',
    }
    with caplog.at_level(logging.WARNING, logger='tunalab.smart_train'):
        selected = select_features_from_kwargs(kwargs)
    assert 'checkpoint_best_model' in selected
    assert not any('NOT consumed' in r.message for r in caplog.records), \
        "clean kwarg set should not produce a dropped-kwarg warning"


def test_load_cached_loop(tmp_path, monkeypatch):
    """_load_cached_loop returns a matching cached loop and never compiles.

    Regression for the down-node failure: a worker without LLM credentials
    (llm_client=None) must still run an already-cached multi-feature loop instead
    of silently falling back to the do-nothing mock loop.
    """
    import importlib
    st = importlib.import_module("tunalab.smart_train")

    features = ["device", "logging", "multi_epoch", "tqdm"]
    name = st._make_descriptive_name(features)

    # Point the artifact root at a temp dir and plant a cached loop with the
    # correct __atomic_features__ marker.
    compiled_dir = tmp_path / "train_loops" / "llm_compiled"
    compiled_dir.mkdir(parents=True)
    loop_path = compiled_dir / f"{name}.py"
    loop_path.write_text(
        "__atomic_features__ = %r\n"
        "def run_training(model, optimizer, train_loader, **kwargs):\n"
        "    return {'model': model, 'ran_cached': True}\n" % features
    )
    monkeypatch.setattr(st, "get_artifact_root", lambda: tmp_path)

    # Exact-match feature set → returns the cached path.
    found = st._load_cached_loop(features)
    assert found == str(loop_path)

    # Order-independent.
    assert st._load_cached_loop(list(reversed(features))) == str(loop_path)

    # Feature mismatch / single feature / missing → None (caller falls back).
    assert st._load_cached_loop(features + ["grad_accum"]) is None
    assert st._load_cached_loop(["tqdm"]) is None
    assert st._load_cached_loop(["device", "logging", "nonexistent"]) is None


def test_smart_train_uses_cache_without_llm_client(tmp_path, monkeypatch):
    """With llm_client=None but a matching cached loop present, smart_train runs
    the cached loop (not the mock no-op)."""
    import importlib
    st = importlib.import_module("tunalab.smart_train")

    # The features select_features_from_kwargs picks for these kwargs
    # (accum_steps→grad_accum, num_epochs→multi_epoch = a 2-feature selection
    # that goes through the LLM-compile path, not the single-feature shortcut).
    kwargs = {"accum_steps": 2, "num_epochs": 1}
    features = select_features_from_kwargs(kwargs)
    assert len(features) > 1, "test needs a multi-feature selection"

    name = st._make_descriptive_name(features)
    compiled_dir = tmp_path / "train_loops" / "llm_compiled"
    compiled_dir.mkdir(parents=True)
    (compiled_dir / f"{name}.py").write_text(
        "__atomic_features__ = %r\n"
        "def run_training(model, optimizer, train_loader, **kwargs):\n"
        "    return {'model': model, 'ran_cached': True}\n" % sorted(features)
    )
    monkeypatch.setattr(st, "get_artifact_root", lambda: tmp_path)

    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randn(8, 4)), batch_size=4)

    # llm_client=None → must use cached loop, NOT the mock stub.
    result = st.smart_train(model, opt, loader, llm_client=None, **kwargs)
    assert result.get("ran_cached") is True, (
        "smart_train fell back to mock instead of using the cached compiled loop"
    )
import pytest
from pathlib import Path

from tunalab.analyze_experiments.configs import (
    find_configs,
    load_config,
    save_config,
    extract_metrics_section,
)


def test_find_configs_discovers_files(tmp_path):
    (tmp_path / "config1.experiments.yaml").touch()
    (tmp_path / "config2.experiments.yaml").touch()
    (tmp_path / "other.yaml").touch()  # Should not be found
    
    configs = find_configs(tmp_path)
    
    assert len(configs) == 2
    assert all(str(c).endswith(".experiments.yaml") for c in configs)


def test_find_configs_searches_recursively(tmp_path):
    (tmp_path / "root.experiments.yaml").touch()
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.experiments.yaml").touch()
    
    configs = find_configs(tmp_path)
    
    assert len(configs) == 2


@pytest.fixture
def sample_config_file(tmp_path):
    config_path = tmp_path / "test.experiments.yaml"
    config_content = """name: Test Analysis
defaults:
  x_axis_key: step
  x_axis_scale: 1.0
  smoothing: 0.0
experiments:
- experiments/**/*.jsonl
metrics:
- name: Loss
  keys: [loss, train_loss]
- name: Accuracy
  keys: [accuracy, acc]
"""
    config_path.write_text(config_content)
    return config_path


def test_load_config_returns_expected_structure(sample_config_file):
    config = load_config(sample_config_file)
    
    assert "name" in config
    assert "defaults" in config
    assert "experiments" in config
    assert "metrics" in config
    assert config["name"] == "Test Analysis"


def test_load_config_fills_missing_defaults(tmp_path):
    minimal_config = tmp_path / "minimal.experiments.yaml"
    minimal_config.write_text("name: Minimal\nexperiments: []\nmetrics: []")
    
    config = load_config(minimal_config)
    
    assert config["defaults"]["x_axis_scale"] == 1.0
    assert config["defaults"]["x_axis_key"] is None
    assert config["defaults"]["smoothing"] == 0.0


@pytest.mark.parametrize(
    "x_axis_key,x_axis_scale,smoothing",
    [
        (None, 1.0, 0.0),
        ("step", 1.0, 0.0),
        ("epoch", 0.001, 0.5),
    ],
)
def test_save_and_load_config_roundtrip(tmp_path, x_axis_key, x_axis_scale, smoothing):
    config_path = tmp_path / "test.experiments.yaml"
    
    original_config = {
        "name": "Test",
        "defaults": {
            "x_axis_key": x_axis_key,
            "x_axis_scale": x_axis_scale,
            "smoothing": smoothing,
        },
        "experiments": ["exp/**/*.jsonl"],
        "metrics": [{"name": "Loss", "keys": ["loss"]}],
    }
    
    save_config(config_path, original_config)
    loaded_config = load_config(config_path)
    
    assert loaded_config["name"] == original_config["name"]
    assert loaded_config["defaults"]["x_axis_key"] == x_axis_key
    assert loaded_config["defaults"]["x_axis_scale"] == x_axis_scale
    assert loaded_config["experiments"] == original_config["experiments"]


def test_save_config_preserves_metrics_comments(tmp_path):
    config_path = tmp_path / "test.experiments.yaml"
    
    # Create initial config with comment
    config_content = """name: Test
defaults:
  x_axis_key: null
  x_axis_scale: 1.0
  smoothing: 0.0
experiments:
- exp/**/*.jsonl
metrics:
# This is a comment
- name: Loss
  keys: [loss]
"""
    config_path.write_text(config_content)
    
    config = load_config(config_path)
    metrics_yaml = extract_metrics_section(config_path)
    
    config["name"] = "Modified Test"
    save_config(config_path, config, metrics_yaml)
    
    saved_content = config_path.read_text()
    assert "# This is a comment" in saved_content
    assert "Modified Test" in saved_content


def test_extract_metrics_section_returns_yaml_text(sample_config_file):
    metrics_yaml = extract_metrics_section(sample_config_file)
    
    assert isinstance(metrics_yaml, str)
    assert "name: Loss" in metrics_yaml or "name:" in metrics_yaml
    assert len(metrics_yaml) > 0


def test_extract_metrics_section_handles_missing_file(tmp_path):
    nonexistent = tmp_path / "nonexistent.experiments.yaml"
    result = extract_metrics_section(nonexistent)
    assert result == ""


def test_extract_metrics_section_handles_no_metrics(tmp_path):
    config_path = tmp_path / "no_metrics.experiments.yaml"
    config_path.write_text("name: Test\nexperiments: []")
    
    result = extract_metrics_section(config_path)
    assert result == ""


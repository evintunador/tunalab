import pytest
import json
from pathlib import Path

from tunalab.analyze_experiments.data import (
    Run,
    normalize_metrics,
    flatten_config,
)


@pytest.fixture
def sample_jsonl_file(tmp_path):
    jsonl_path = tmp_path / "run.jsonl"
    entries = [
        {"step": 0, "loss": 2.5, "lr": 0.001, "device": "cuda"},
        {"step": 1, "loss": 2.3, "lr": 0.001, "device": "cuda"},
        {"step": 2, "loss": 2.1, "lr": 0.001, "device": "cuda"},
    ]
    with open(jsonl_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    return jsonl_path


def test_run_from_path_loads_data(sample_jsonl_file):
    run = Run.from_path(sample_jsonl_file)
    
    assert len(run.df) == 3
    assert "step" in run.df.columns
    assert "loss" in run.df.columns
    assert run.df["loss"].tolist() == [2.5, 2.3, 2.1]


def test_run_from_path_extracts_static_values(sample_jsonl_file):
    run = Run.from_path(sample_jsonl_file)
    
    assert "lr" in run.static
    assert "device" in run.static
    assert run.static["lr"] == 0.001
    assert run.static["device"] == "cuda"
    
    assert "lr" not in run.df.columns
    assert "device" not in run.df.columns


def test_run_from_path_handles_empty_file(tmp_path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.touch()
    
    run = Run.from_path(empty_file)
    
    assert len(run.df) == 0
    assert run.static == {}


def test_run_from_path_skips_malformed_lines(tmp_path):
    jsonl_path = tmp_path / "malformed.jsonl"
    with open(jsonl_path, 'w') as f:
        f.write('{"step": 0, "loss": 1.0}\n')
        f.write('this is not json\n')
        f.write('{"step": 1, "loss": 0.9}\n')
    
    run = Run.from_path(jsonl_path)
    
    assert len(run.df) == 2
    assert run.df["loss"].tolist() == [1.0, 0.9]


@pytest.mark.parametrize(
    "metrics_config,expected_columns",
    [
        (
            [{"name": "Loss", "keys": ["loss"]}],
            ["Loss"]
        ),
        (
            [{"name": "Accuracy", "keys": ["acc", "accuracy"]}],
            ["Accuracy"]
        ),
        (
            [
                {"name": "Loss", "keys": ["loss"]},
                {"name": "Accuracy", "keys": ["accuracy"]}
            ],
            ["Loss", "Accuracy"]
        ),
    ],
)
def test_normalize_metrics_renames_columns(tmp_path, metrics_config, expected_columns):
    jsonl_path = tmp_path / "run.jsonl"
    entries = [
        {"step": 0, "loss": 1.0, "accuracy": 0.5},
        {"step": 1, "loss": 0.9, "accuracy": 0.6},
    ]
    with open(jsonl_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    run = Run.from_path(jsonl_path)
    normalized = normalize_metrics([run], metrics_config)
    
    df = normalized[str(run.id)]
    for col in expected_columns:
        assert col in df.columns


def test_normalize_metrics_uses_priority_keys(tmp_path):
    jsonl_path = tmp_path / "run.jsonl"
    entries = [
        {"step": 0, "train_loss": 1.0},  # Has train_loss but not loss
        {"step": 1, "train_loss": 0.9},
    ]
    with open(jsonl_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    run = Run.from_path(jsonl_path)
    metrics_config = [{"name": "Loss", "keys": ["loss", "train_loss"]}]
    normalized = normalize_metrics([run], metrics_config)
    
    df = normalized[str(run.id)]
    assert "Loss" in df.columns
    assert "train_loss" not in df.columns


def test_normalize_metrics_handles_multiple_runs(tmp_path):
    run1_path = tmp_path / "run1.jsonl"
    run2_path = tmp_path / "run2.jsonl"
    
    with open(run1_path, 'w') as f:
        f.write(json.dumps({"step": 0, "loss": 1.0}) + '\n')
        f.write(json.dumps({"step": 1, "loss": 0.9}) + '\n')
    
    with open(run2_path, 'w') as f:
        f.write(json.dumps({"step": 0, "loss": 2.0}) + '\n')
        f.write(json.dumps({"step": 1, "loss": 1.8}) + '\n')
    
    run1 = Run.from_path(run1_path)
    run2 = Run.from_path(run2_path)
    
    metrics_config = [{"name": "Loss", "keys": ["loss"]}]
    normalized = normalize_metrics([run1, run2], metrics_config)
    
    assert len(normalized) == 2
    assert "Loss" in normalized[str(run1.id)].columns
    assert "Loss" in normalized[str(run2.id)].columns


@pytest.mark.parametrize(
    "config,expected",
    [
        (
            {"a": 1, "b": 2},
            {"a": 1, "b": 2}
        ),
        (
            {"a": {"b": 1}},
            {"a.b": 1}
        ),
        (
            {"a": {"b": {"c": 1}}},
            {"a.b.c": 1}
        ),
        (
            {"a": {"b": 1, "c": 2}, "d": 3},
            {"a.b": 1, "a.c": 2, "d": 3}
        ),
    ],
)
def test_flatten_config(config, expected):
    result = flatten_config(config)
    assert result == expected


def test_flatten_config_custom_separator():
    config = {"a": {"b": 1}}
    result = flatten_config(config, sep="/")
    assert result == {"a/b": 1}


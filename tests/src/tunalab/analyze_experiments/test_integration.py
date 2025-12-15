import json
from pathlib import Path

from tunalab.analyze_experiments import (
    Run,
    normalize_metrics,
    load_config,
    save_config,
)


def test_full_workflow_load_normalize_display(tmp_path):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    
    run1_path = exp_dir / "run1.jsonl"
    run2_path = exp_dir / "run2.jsonl"
    
    with open(run1_path, 'w') as f:
        for i in range(5):
            f.write(json.dumps({
                "step": i,
                "loss": 2.0 - i * 0.2,
                "accuracy": 0.5 + i * 0.1,
                "lr": 0.001,
            }) + '\n')
    
    with open(run2_path, 'w') as f:
        for i in range(5):
            f.write(json.dumps({
                "step": i,
                "loss": 2.5 - i * 0.3,
                "accuracy": 0.4 + i * 0.15,
                "lr": 0.01,
            }) + '\n')
    
    config_path = tmp_path / "analysis.experiments.yaml"
    config = {
        "name": "Test Analysis",
        "defaults": {"x_axis_key": "step", "x_axis_scale": 1.0, "smoothing": 0.0},
        "experiments": [str(exp_dir / "*.jsonl")],
        "metrics": [
            {"name": "Loss", "keys": ["loss"]},
            {"name": "Accuracy", "keys": ["accuracy"]},
        ],
    }
    save_config(config_path, config)
    
    loaded_config = load_config(config_path)
    run1 = Run.from_path(run1_path)
    run2 = Run.from_path(run2_path)
    
    assert run1.static["lr"] == 0.001
    assert run2.static["lr"] == 0.01
    
    normalized = normalize_metrics([run1, run2], loaded_config["metrics"])
    
    assert len(normalized) == 2
    for run_id, df in normalized.items():
        assert "Loss" in df.columns
        assert "Accuracy" in df.columns
        assert len(df) == 5


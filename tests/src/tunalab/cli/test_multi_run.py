import pytest
import json
import tempfile
from pathlib import Path

from tunalab.cli.multi_run import (
    validate_config,
    expand_parameters,
    _expand_grid,
    _expand_paired,
    detect_old_parameter_format,
    build_command,
    RunEntry,
    create_sweep_directory,
    write_manifest_skeleton,
    update_manifest_entry,
    load_failed_runs,
)


@pytest.mark.parametrize(
    "config,is_valid",
    [
        ({"command": {"script": "main.py"}, "parameters": {"grid": {"lr": [0.1]}}}, True),
        ({"command": {"script": "main.py"}}, False),
        ({"parameters": {"grid": {"lr": [0.1]}}}, False),
        ({}, False),
        ({"raw_commands": ["python main.py"]}, True),
    ],
)
def test_validate_config(config, is_valid):
    missing = validate_config(config)
    assert (len(missing) == 0) == is_valid


@pytest.mark.parametrize(
    "grid_params,expected_count",
    [
        ({"lr": [0.001, 0.01], "wd": [0.0, 0.1]}, 4),
        ({"lr": [0.001, 0.01]}, 2),
        ({"lr": [0.001]}, 1),
        ({}, 1),
    ],
)
def test_expand_grid(grid_params, expected_count):
    result = _expand_grid(grid_params)
    assert len(result) == expected_count


def test_expand_grid_list_constant():
    grid_params = {
        "lr": [0.001, 0.01],
        "means": [[0.456, 0.428, 0.493]]
    }
    result = _expand_grid(grid_params)
    
    assert len(result) == 2
    for combo in result:
        assert combo["means"] == [0.456, 0.428, 0.493]


def test_expand_paired():
    paired_groups = [
        {
            "name": "architecture",
            "combinations": [
                {"num_layers": 4, "hidden_dim": 128},
                {"num_layers": 6, "hidden_dim": 256}
            ]
        }
    ]
    result = _expand_paired(paired_groups)
    
    assert len(result) == 2
    assert {"num_layers": 4, "hidden_dim": 128} in result


def test_expand_parameters_grid_and_paired():
    config = {
        "parameters": {
            "grid": {"lr": [0.001, 0.01]},
            "paired": [{
                "name": "arch",
                "combinations": [
                    {"layers": 4, "dim": 128},
                    {"layers": 6, "dim": 256}
                ]
            }]
        }
    }
    result = expand_parameters(config)
    
    assert len(result) == 4


def test_expand_parameters_conflict_detection():
    config = {
        "parameters": {
            "grid": {"lr": [0.001, 0.01]},
            "paired": [{
                "name": "arch",
                "combinations": [
                    {"lr": 0.001, "layers": 4},
                ]
            }]
        }
    }
    
    with pytest.raises(ValueError, match="both grid and paired"):
        expand_parameters(config)


def test_expand_parameters_raw_commands():
    config = {"raw_commands": ["python train.py --lr 0.001"]}
    result = expand_parameters(config)
    
    assert result == [{}]


def test_detect_old_parameter_format():
    assert detect_old_parameter_format({"lr": [0.001], "device": "cuda"}) is True
    assert detect_old_parameter_format({"grid": {"lr": [0.001]}}) is False
    assert detect_old_parameter_format({"device": "cuda"}) is False


@pytest.mark.parametrize(
    "cmd_type,expected_in_cmd",
    [
        ("python", "python train.py"),
        ("torchrun", ["torchrun", "--nproc_per_node=2", "train.py"]),
    ],
)
def test_build_command(cmd_type, expected_in_cmd):
    command_config = {
        "type": cmd_type,
        "script": "train.py",
        "nproc_per_node": 2
    }
    params = {"lr": 0.001}
    
    cmd = build_command(command_config, params)
    
    if isinstance(expected_in_cmd, list):
        for part in expected_in_cmd:
            assert part in cmd
    else:
        assert expected_in_cmd in cmd
    assert "--lr 0.001" in cmd


def test_build_command_with_list():
    command_config = {"type": "python", "script": "train.py"}
    params = {"means": [0.456, 0.428, 0.493]}
    
    cmd = build_command(command_config, params)
    assert "[0.456, 0.428, 0.493]" in cmd


def test_create_sweep_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        original_cwd = Path.cwd()
        try:
            os.chdir(tmpdir)
            
            config = {"name": "test_sweep"}
            sweep_dir = create_sweep_directory(config)
            
            assert sweep_dir.exists()
            assert (sweep_dir / "logs").exists()
            assert (sweep_dir / "sweep_config.yaml").exists()
            assert "test_sweep" in str(sweep_dir)
        finally:
            os.chdir(original_cwd)


def test_manifest_workflow():
    with tempfile.TemporaryDirectory() as tmpdir:
        sweep_dir = Path(tmpdir)
        
        runs = [
            RunEntry(
                run_id="run_1",
                run_index=1,
                parameters={"lr": 0.001},
                command="python train.py --lr 0.001",
                status="pending"
            ),
            RunEntry(
                run_id="run_2",
                run_index=2,
                parameters={"lr": 0.01},
                command="python train.py --lr 0.01",
                status="pending"
            )
        ]
        
        write_manifest_skeleton(sweep_dir, runs)
        
        manifest_path = sweep_dir / "runs_manifest.jsonl"
        assert manifest_path.exists()
        
        update_manifest_entry(
            sweep_dir,
            "run_1",
            {"status": "success", "exit_code": 0}
        )
        
        with open(manifest_path) as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        entry1 = json.loads(lines[0])
        assert entry1["status"] == "success"
        assert entry1["exit_code"] == 0


def test_load_failed_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        sweep_dir = Path(tmpdir)
        
        runs = [
            RunEntry(
                run_id="run_1",
                run_index=1,
                parameters={"lr": 0.001},
                command="python train.py --lr 0.001",
                status="success",
                exit_code=0
            ),
            RunEntry(
                run_id="run_2",
                run_index=2,
                parameters={"lr": 0.01},
                command="python train.py --lr 0.01",
                status="failed",
                exit_code=1
            )
        ]
        
        write_manifest_skeleton(sweep_dir, runs)
        
        failed = load_failed_runs(sweep_dir)
        
        assert len(failed) == 1
        assert failed[0]["run_id"] == "run_2"
        assert failed[0]["status"] == "pending"
        assert failed[0]["exit_code"] is None

import sys
import os
import subprocess
from pathlib import Path
import yaml

from tunalab.analyze_experiments import find_configs


def get_notebook_path():
    import tunalab.analyze_experiments as ae_module
    return Path(ae_module.__file__).parent / "notebook.py"


def list_analysis_configs():
    configs = find_configs(".")
    if not configs:
        print("No analysis configuration files found.")
        return
    
    print("Analysis configurations:")
    for cfg in configs:
        print(f"  - {cfg}")


def run_analysis(name_or_path=None):
    notebook_path = get_notebook_path()
    
    if not notebook_path.exists():
        print(f"Error: Analysis notebook not found at {notebook_path}")
        sys.exit(1)
    
    cmd = [sys.executable, "-m", "marimo", "run", str(notebook_path)]
    
    if name_or_path:
        config_path = resolve_config_path(name_or_path)
        if config_path:
            cmd.extend(["--", "--config", str(config_path)])
        else:
            print(f"Warning: Config '{name_or_path}' not found. Opening without pre-selection.")
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print("Error: 'marimo' is not installed or not in PATH.")


def edit_analysis(name_or_path=None):
    notebook_path = get_notebook_path()
    
    if not notebook_path.exists():
        print(f"Error: Analysis notebook not found at {notebook_path}")
        sys.exit(1)
    
    cmd = [sys.executable, "-m", "marimo", "edit", str(notebook_path)]
    
    if name_or_path:
        config_path = resolve_config_path(name_or_path)
        if config_path:
            cmd.extend(["--", "--config", str(config_path)])
        else:
            print(f"Warning: Config '{name_or_path}' not found. Opening without pre-selection.")
    
    print(f"Editing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print("Error: 'marimo' is not installed or not in PATH.")


def resolve_config_path(name_or_path: str) -> Path | None:
    name_or_path = name_or_path.strip()
    
    path = Path(name_or_path)
    if path.exists() and path.is_file():
        return path.resolve()
    
    if not name_or_path.endswith(".experiments.yaml"):
        name_or_path = name_or_path + ".experiments.yaml"
    
    configs = find_configs(".")
    
    for cfg in configs:
        if cfg.name == name_or_path or str(cfg) == name_or_path:
            return cfg.resolve()
    
    name_without_ext = name_or_path.replace(".experiments.yaml", "")
    for cfg in configs:
        cfg_name_without_ext = cfg.name.replace(".experiments.yaml", "")
        if cfg_name_without_ext == name_without_ext:
            return cfg.resolve()
    
    return None


def create_analysis_config(name="analysis"):
    if not name.endswith(".experiments.yaml"):
        if name.endswith(".yaml") or name.endswith(".yml"):
            name = name.rsplit(".", 1)[0] + ".experiments.yaml"
        else:
            name = name + ".experiments.yaml"
    
    config = {
        "name": name.replace(".experiments.yaml", "").replace("_", " ").title(),
        "defaults": {
            "x_axis_key": None,
            "x_axis_scale": 1.0,
            "smoothing": 0.0,
        },
        "experiments": [
            "experiments/**/*.jsonl",
        ],
        "metrics": [
            {
                "name": "Loss",
                "keys": ["loss", "train_loss"],
            },
        ],
    }
    
    path = Path(".") / name
    if path.exists():
        print(f"Error: Config file already exists: {path}")
        sys.exit(1)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created analysis configuration: {path}")


def _main_impl():
    """Handle analyze-experiments subcommands."""
    if len(sys.argv) < 2:
        print("Usage: tunalab analyze-experiments [list|run|edit|new] [args...]")
        print("  list                - List all analysis configuration files")
        print("  run [name_or_path]  - Run the analysis notebook (read-only)")
        print("  edit [name_or_path] - Edit the analysis notebook")
        print("  new [name]          - Create a new analysis configuration file")
        return
    
    subcommand = sys.argv[1]
    sys.argv.pop(1)
    
    if subcommand == "list":
        list_analysis_configs()
    elif subcommand == "run":
        name_or_path = sys.argv[1] if len(sys.argv) > 1 else None
        run_analysis(name_or_path)
    elif subcommand == "edit":
        name_or_path = sys.argv[1] if len(sys.argv) > 1 else None
        edit_analysis(name_or_path)
    elif subcommand == "new":
        name = sys.argv[1] if len(sys.argv) > 1 else "analysis"
        create_analysis_config(name)
    else:
        print(f"Unknown analyze-experiments subcommand: {subcommand}")
        print("Usage: tunalab analyze-experiments [list|run|edit|new] [args...]")
        sys.exit(1)


class Command:
    """Implementation of Command protocol to be recognized by cli registry"""
    
    name = "analyze-experiments"
    description = "Visualize and compare experiment results"
    
    @staticmethod
    def main():
        _main_impl()


if __name__ == "__main__":
    _main_impl()

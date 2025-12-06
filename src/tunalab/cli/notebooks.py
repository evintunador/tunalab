import sys
import os
import subprocess
from pathlib import Path
import yaml
import tunalab.notebooks
from tunalab.analysis import find_dashboards


def get_notebooks_dir():
    return os.path.dirname(tunalab.notebooks.__file__)


def list_notebooks():
    notebooks_dir = get_notebooks_dir()
    notebooks = [
        f for f in os.listdir(notebooks_dir) 
        if f.endswith(".py") and not f.startswith("_") and f != "__init__.py"
    ]
    print(f"Notebooks directory: {notebooks_dir}")
    print("Available notebooks:")
    for nb in sorted(notebooks):
        print(f"  - {nb[:-3]}")


def run_marimo(command, notebook_name, extra_args=None):
    notebooks_dir = get_notebooks_dir()
    if not notebook_name.endswith(".py"):
        notebook_name += ".py"
        
    path = os.path.join(notebooks_dir, notebook_name)
    
    if not os.path.exists(path):
        print(f"Error: Notebook '{notebook_name}' not found.")
        list_notebooks()
        sys.exit(1)
        
    cmd = [sys.executable, "-m", "marimo", command, path]
    if extra_args:
        cmd.extend(extra_args)
        
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print("Error: 'marimo' is not installed or not in PATH.")


def dashboard_list():
    """List all dashboard configuration files."""
    dashboards = find_dashboards(".")
    if not dashboards:
        print("No dashboard configuration files found.")
        return
    
    print("Dashboard configurations:")
    for db in dashboards:
        print(f"  - {db}")


def dashboard_run(name_or_path=None):
    """Run the dashboard notebook."""
    notebooks_dir = get_notebooks_dir()
    notebook_path = os.path.join(notebooks_dir, "analyze.py")
    
    if not os.path.exists(notebook_path):
        print(f"Error: Dashboard notebook not found at {notebook_path}")
        sys.exit(1)
    
    cmd = [sys.executable, "-m", "marimo", "run", notebook_path]
    
    if name_or_path:
        dashboard_path = resolve_dashboard_path(name_or_path)
        if dashboard_path:
            cmd.extend(["--", "--dashboard", str(dashboard_path)])
        else:
            print(f"Warning: Dashboard '{name_or_path}' not found. Opening without pre-selection.")
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print("Error: 'marimo' is not installed or not in PATH.")


def resolve_dashboard_path(name_or_path: str) -> Path | None:
    """
    Resolve a dashboard name or path to a full Path.
    
    Args:
        name_or_path: Dashboard name (e.g., "out-shakespeare-char") or path
        
    Returns:
        Path to the dashboard file, or None if not found
    """
    name_or_path = name_or_path.strip()
    
    path = Path(name_or_path)
    if path.exists() and path.is_file():
        return path.resolve()
    
    if not name_or_path.endswith(".dashboard.yaml"):
        name_or_path = name_or_path + ".dashboard.yaml"
    
    dashboards = find_dashboards(".")
    
    for db in dashboards:
        if db.name == name_or_path or str(db) == name_or_path:
            return db.resolve()
    
    name_without_ext = name_or_path.replace(".dashboard.yaml", "")
    for db in dashboards:
        db_name_without_ext = db.name.replace(".dashboard.yaml", "")
        if db_name_without_ext == name_without_ext:
            return db.resolve()
    
    return None


def dashboard_new(name="dashboard"):
    """Create a new boilerplate dashboard configuration file."""
    if not name.endswith(".dashboard.yaml"):
        if name.endswith(".yaml") or name.endswith(".yml"):
            name = name.rsplit(".", 1)[0] + ".dashboard.yaml"
        else:
            name = name + ".dashboard.yaml"
    
    config = {
        "name": name.replace(".dashboard.yaml", "").replace("_", " ").title(),
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
        print(f"Error: Dashboard file already exists: {path}")
        sys.exit(1)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created dashboard configuration: {path}")


def dashboard_main():
    """Handle dashboard subcommands."""
    if len(sys.argv) < 2:
        print("Usage: tunalab dashboard [list|run|new] [args...]")
        print("  list              - List all dashboard configuration files")
        print("  run [name_or_path] - Run the dashboard notebook")
        print("  new [name]         - Create a new dashboard configuration file")
        return
    
    subcommand = sys.argv[1]
    sys.argv.pop(1)  # Remove subcommand from argv
    
    if subcommand == "list":
        dashboard_list()
    elif subcommand == "run":
        name_or_path = sys.argv[1] if len(sys.argv) > 1 else None
        dashboard_run(name_or_path)
    elif subcommand == "new":
        name = sys.argv[1] if len(sys.argv) > 1 else "dashboard"
        dashboard_new(name)
    else:
        print(f"Unknown dashboard subcommand: {subcommand}")
        print("Usage: tunalab dashboard [list|run|new] [args...]")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: tunalab notebook [list|edit] <notebook_name>")
        print("       tunalab notebook <notebook_name>  (runs by default)")
        list_notebooks()
        return

    first_arg = sys.argv[1]
    
    if first_arg == "list":
        list_notebooks()
        return
        
    if first_arg == "edit":
        if len(sys.argv) < 3:
            print("Usage: tunalab notebook edit <notebook_name>")
            sys.exit(1)
        notebook_name = sys.argv[2]
        extra_args = sys.argv[3:]
        run_marimo("edit", notebook_name, extra_args)
    else:
        # Default behavior: treat first arg as notebook name and run it
        # Check if the user explicitly typed "run" just in case they are used to the old way
        if first_arg == "run" and len(sys.argv) >= 3:
            notebook_name = sys.argv[2]
            extra_args = sys.argv[3:]
        else:
            notebook_name = first_arg
            extra_args = sys.argv[2:]
            
        run_marimo("run", notebook_name, extra_args)

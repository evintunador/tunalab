import sys
import subprocess
from pathlib import Path


def get_notebook_paths():
    """Get paths to benchmark notebooks."""
    import tunalab.benchmarking.notebooks as notebooks_module
    notebooks_dir = Path(notebooks_module.__file__).parent
    return {
        'modules': notebooks_dir / 'modules.py',
        'optimizers': notebooks_dir / 'optimizers.py',
    }


def list_benchmarks():
    """List available benchmark notebooks."""
    print("Available benchmarks:")
    for name in ['modules', 'optimizers']:
        print(f"  - {name}")


def run_benchmark(benchmark_name):
    """Run a benchmark notebook in read-only mode."""
    notebooks = get_notebook_paths()
    
    if benchmark_name not in notebooks:
        print(f"Error: Unknown benchmark '{benchmark_name}'")
        print("\nAvailable benchmarks:")
        list_benchmarks()
        sys.exit(1)
    
    notebook_path = notebooks[benchmark_name]
    
    if not notebook_path.exists():
        print(f"Error: Benchmark notebook not found at {notebook_path}")
        sys.exit(1)
    
    cmd = [sys.executable, "-m", "marimo", "run", str(notebook_path)]
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print("Error: 'marimo' is not installed or not in PATH.")
        print("Install it with: pip install marimo")


def edit_benchmark(benchmark_name):
    """Edit a benchmark notebook."""
    notebooks = get_notebook_paths()
    
    if benchmark_name not in notebooks:
        print(f"Error: Unknown benchmark '{benchmark_name}'")
        print("\nAvailable benchmarks:")
        list_benchmarks()
        sys.exit(1)
    
    notebook_path = notebooks[benchmark_name]
    
    if not notebook_path.exists():
        print(f"Error: Benchmark notebook not found at {notebook_path}")
        sys.exit(1)
    
    cmd = [sys.executable, "-m", "marimo", "edit", str(notebook_path)]
    
    print(f"Editing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print("Error: 'marimo' is not installed or not in PATH.")
        print("Install it with: pip install marimo")


def _main_impl():
    """Handle benchmark subcommands."""
    if len(sys.argv) < 2:
        print("Usage: tunalab benchmark [list|run|edit] [benchmark_name]")
        print("  list               - List available benchmarks")
        print("  run <name>         - Run a benchmark notebook (read-only)")
        print("  edit <name>        - Edit a benchmark notebook")
        print("\nAvailable benchmarks:")
        list_benchmarks()
        return
    
    subcommand = sys.argv[1]
    sys.argv.pop(1)
    
    if subcommand == "list":
        list_benchmarks()
    elif subcommand == "run":
        if len(sys.argv) < 2:
            print("Error: Missing benchmark name")
            print("Usage: tunalab benchmark run <benchmark_name>")
            print("\nAvailable benchmarks:")
            list_benchmarks()
            sys.exit(1)
        benchmark_name = sys.argv[1]
        run_benchmark(benchmark_name)
    elif subcommand == "edit":
        if len(sys.argv) < 2:
            print("Error: Missing benchmark name")
            print("Usage: tunalab benchmark edit <benchmark_name>")
            print("\nAvailable benchmarks:")
            list_benchmarks()
            sys.exit(1)
        benchmark_name = sys.argv[1]
        edit_benchmark(benchmark_name)
    else:
        print(f"Unknown benchmark subcommand: {subcommand}")
        print("Usage: tunalab benchmark [list|run|edit] [benchmark_name]")
        sys.exit(1)


class Command:
    """Implementation of Command protocol to be recognized by cli registry"""
    
    name = "benchmark"
    description = "Run and visualize module/optimizer benchmarks"
    
    @staticmethod
    def main():
        _main_impl()


if __name__ == "__main__":
    _main_impl()


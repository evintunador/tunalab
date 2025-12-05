import sys
import os
import subprocess
import tunalab.notebooks


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
        print(f"  - {nb}")


def run_marimo(command, notebook_name, extra_args=None):
    notebooks_dir = get_notebooks_dir()
    if not notebook_name.endswith(".py"):
        notebook_name += ".py"
        
    path = os.path.join(notebooks_dir, notebook_name)
    
    if not os.path.exists(path):
        print(f"Error: Notebook '{notebook_name}' not found.")
        list_notebooks()
        sys.exit(1)
        
    cmd = ["marimo", command, path]
    if extra_args:
        cmd.extend(extra_args)
        
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print("Error: 'marimo' is not installed or not in PATH.")


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

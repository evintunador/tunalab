import sys
from tunalab.cli import multi_run, notebooks

def main():
    if len(sys.argv) < 2:
        print("Usage: tunalab <command> [args...]")
        print("Commands:\n  run-multi")
        sys.exit(1)

    command = sys.argv[1]
    # Remove the subcommand so the underlying module sees the rest of the args
    sys.argv.pop(1)

    if command == "run-multi":
        multi_run.main()
    elif command == "notebook":
        notebooks.main()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


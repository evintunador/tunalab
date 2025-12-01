#!/bin/bash
# Run bulk catalog tests on this experiment's namespace contributions
# Must be run from within this experiment's virtual environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

# Check if we're in the right venv
if [[ "$VIRTUAL_ENV" != *"$EXPERIMENT_DIR"* ]]; then
    echo "Warning: You may not be in this experiment's venv."
    echo "Expected venv in: $EXPERIMENT_DIR/.venv"
    echo "Current venv: $VIRTUAL_ENV"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Running bulk catalog tests for $(basename $EXPERIMENT_DIR)..."
echo "Bulk runners will test items from: tunalab.* namespace"
echo ""

# Run the bulk test runners
pytest "$REPO_ROOT/tests/bulk_runners/" -v "$@"
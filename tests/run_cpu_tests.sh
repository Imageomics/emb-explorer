#!/bin/bash
#SBATCH --account=PAS2136
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=emb-tests-cpu
#SBATCH --output=tests/cpu_test_results_%j.log

# ------------------------------------------------------------------
# CPU test runner for emb-explorer (OSC Pitzer)
#
# Usage:
#   sbatch tests/run_cpu_tests.sh                    # all non-GPU tests
#   sbatch tests/run_cpu_tests.sh tests/test_filters.py  # specific file
# ------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

source /fs/scratch/PAS2136/netzissou/venv/emb_explorer_pitzer/bin/activate

echo "=== CPU Test Run ==="
echo "Node:    $(hostname)"
echo "Python:  $(python --version)"
echo "Project: $PROJECT_DIR"
echo "===================="

if [[ -n "${1:-}" ]]; then
    echo "Running: pytest $* -m 'not gpu' -v"
    pytest "$@" -m "not gpu" -v
else
    echo "Running all CPU tests..."
    pytest tests/ -m "not gpu" -v
fi

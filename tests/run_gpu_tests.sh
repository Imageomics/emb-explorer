#!/bin/bash
# NOTE: Edit --account to match your SLURM allocation before submitting.
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=emb-tests-gpu
#SBATCH --output=tests/gpu_test_results_%j.log

# ------------------------------------------------------------------
# GPU test runner for emb-explorer
#
# Before first use:
#   1. Set --account above to your SLURM allocation (e.g. PAS2136)
#   2. Export VENV_DIR to point to your venv base directory
#
# Usage:
#   VENV_DIR=/path/to/venvs sbatch tests/run_gpu_tests.sh          # full suite on GPU node
#   VENV_DIR=/path/to/venvs sbatch tests/run_gpu_tests.sh --gpu    # GPU-marked tests only
# ------------------------------------------------------------------

set -euo pipefail

# Resolve project root — SLURM copies the script, so use $SLURM_SUBMIT_DIR
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# Activate venv — VENV_DIR must be set by the user
if [[ -z "${VENV_DIR:-}" ]]; then
    echo "ERROR: VENV_DIR is not set. Export it to your venv base directory." >&2
    echo "  e.g.: VENV_DIR=/fs/scratch/PAS2136/\$USER/venv sbatch tests/run_gpu_tests.sh" >&2
    exit 1
fi
source "$VENV_DIR/emb_explorer_pitzer/bin/activate"

# cuML/CuPy need nvidia libs on LD_LIBRARY_PATH
NVIDIA_LIBS="$(python -c 'import nvidia.cublas.lib, nvidia.cusolver.lib, nvidia.cusparse.lib; \
    print(nvidia.cublas.lib.__path__[0]); print(nvidia.cusolver.lib.__path__[0]); print(nvidia.cusparse.lib.__path__[0])' 2>/dev/null | tr '\n' ':')" || true
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== GPU Test Run ==="
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:  $(python --version)"
echo "Project: $PROJECT_DIR"
echo "===================="

if [[ "${1:-}" == "--gpu" ]]; then
    echo "Running GPU-marked tests only..."
    pytest tests/ -m gpu -v
else
    echo "Running full test suite on GPU node..."
    pytest tests/ -v
fi

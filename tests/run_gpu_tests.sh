#!/bin/bash
#SBATCH --account=PAS2136
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=emb-tests-gpu
#SBATCH --output=tests/gpu_test_results_%j.log

# ------------------------------------------------------------------
# GPU test runner for emb-explorer (OSC Pitzer)
#
# Usage:
#   sbatch tests/run_gpu_tests.sh            # GPU tests only
#   sbatch tests/run_gpu_tests.sh --all      # full suite on GPU node
# ------------------------------------------------------------------

set -euo pipefail

# Resolve project root — SLURM copies the script, so use $SLURM_SUBMIT_DIR
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# Activate venv ($VENV_DIR should point to the base venv directory)
VENV_DIR="${VENV_DIR:-/fs/scratch/PAS2136/netzissou/venv}"
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

if [[ "${1:-}" == "--all" ]]; then
    echo "Running FULL test suite on GPU node..."
    pytest tests/ -v
else
    echo "Running GPU-marked tests..."
    pytest tests/ -m gpu -v
fi

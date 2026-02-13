#!/bin/bash
#SBATCH --account=PAS2136
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=emb-tests-gpu
#SBATCH --output=tests/gpu_test_results_%j.log

# ------------------------------------------------------------------
# GPU test runner for emb-explorer (OSC Pitzer)
#
# Usage:
#   sbatch tests/run_gpu_tests.sh              # full suite on GPU node
#   sbatch tests/run_gpu_tests.sh --gpu        # GPU-marked tests only
# ------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

source /fs/scratch/PAS2136/netzissou/venv/emb_explorer_pitzer/bin/activate

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

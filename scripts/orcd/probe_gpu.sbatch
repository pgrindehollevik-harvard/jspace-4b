#!/usr/bin/env bash
# A short, low-cost allocation that proves the account can actually receive a GPU.
# Override the GPU type at submission time if desired, e.g.:
#   sbatch -G h200:1 scripts/orcd/probe_gpu.sbatch
#SBATCH --job-name=jspace-gpu-probe
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/orcd/%x-%j.out
#SBATCH --error=logs/orcd/%x-%j.err

set -euo pipefail
ROOT="${JSPACE_ROOT:-$HOME/orcd/scratch/jspace-4b}"
cd "$ROOT"
module load miniforge/24.3.0-0 || module load miniforge

echo "=== Slurm allocation ==="
env | sort | grep '^SLURM_' || true
echo "=== NVIDIA ==="
nvidia-smi
echo "=== PyTorch CUDA ==="
.venv/bin/python - <<'PY'
import json
import torch

assert torch.cuda.is_available(), "Slurm allocated no CUDA-visible GPU"
print(json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "device_count": torch.cuda.device_count(),
    "device": torch.cuda.get_device_name(0),
    "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
}, indent=2))
PY

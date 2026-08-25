#!/usr/bin/env bash
# CUDA + model + Jacobian-lens compatibility check. This produces no result artifact.
# Stage qwen3-4b first with scripts/orcd/stage_assets.sh qwen3-4b.
#SBATCH --job-name=jspace-cuda-smoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/orcd/%x-%j.out
#SBATCH --error=logs/orcd/%x-%j.err

set -euo pipefail
ROOT="${JSPACE_ROOT:-$HOME/orcd/scratch/jspace-4b}"
cd "$ROOT"
module load miniforge/24.3.0-0 || module load miniforge
export HF_HOME="${HF_HOME:-$ROOT/.hf}"
export HF_HUB_OFFLINE=1
export JSPACE_DEVICE=cuda
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi
.venv/bin/python -u -m jspace.orcd_smoke

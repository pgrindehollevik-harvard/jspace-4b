#!/usr/bin/env bash
# Engineering-only profile smoke test. Stage the selected profile first.
# Submit: sbatch --export=ALL,JSPACE_PROFILE=qwen3-8b scripts/orcd/scale_smoke.sbatch
#SBATCH --job-name=jspace-scale-smoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=logs/orcd/%x-%j.out
#SBATCH --error=logs/orcd/%x-%j.err

set -euo pipefail
: "${JSPACE_PROFILE:?submit with JSPACE_PROFILE=qwen3-8b or qwen3-14b}"
ROOT="${JSPACE_ROOT:-$HOME/orcd/scratch/jspace-4b}"
cd "$ROOT"
module load miniforge/24.3.0-0 || module load miniforge
export HF_HOME="${HF_HOME:-$ROOT/.hf}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export JSPACE_DEVICE=cuda
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi
.venv/bin/python -u -m jspace.scale_smoke --model "$JSPACE_PROFILE"

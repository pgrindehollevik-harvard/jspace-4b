#!/usr/bin/env bash
# Run from an ORCD login node, never from an allocated GPU job.
# Downloads public artifacts to scratch so compute jobs can work offline.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 qwen3-1.7b|qwen3-4b|qwen3-8b|qwen3-14b|qwen3-32b" >&2
  exit 2
fi

ROOT="${JSPACE_ROOT:-$HOME/orcd/scratch/jspace-4b}"
cd "$ROOT"
module load miniforge/24.3.0-0 || module load miniforge
export HF_HOME="${HF_HOME:-$ROOT/.hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1

.venv/bin/python -m jspace.stage_assets --model "$1"

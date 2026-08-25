#!/usr/bin/env bash
# Materialize the preregistered public-data subset on the login node.
set -euo pipefail

ROOT="${JSPACE_ROOT:-$HOME/orcd/scratch/jspace-4b}"
cd "$ROOT"
module load miniforge/24.3.0-0 || module load miniforge
export HF_HOME="${HF_HOME:-$ROOT/.hf}"
.venv/bin/python -u -m jspace.scale_data

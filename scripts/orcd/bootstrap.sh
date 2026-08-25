#!/usr/bin/env bash
# Set up a reproducible, user-owned checkout on an ORCD login node.
# This script never asks for, records, or exports credentials.
set -euo pipefail

REPO_URL="https://github.com/pgrindehollevik-harvard/jspace-4b.git"
JLENS_URL="https://github.com/anthropics/jacobian-lens.git"
JLENS_REV="581d398613e5602a5af361e1c34d3a92ea82ba8e"
ROOT="${JSPACE_ROOT:-$HOME/orcd/scratch/jspace-4b}"
REF="${JSPACE_REF:-verify-orcd-gpu-jobs}"

module load miniforge/24.3.0-0 || module load miniforge

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --user --upgrade uv
  export PATH="$(python -m site --user-base)/bin:$PATH"
fi

mkdir -p "$(dirname "$ROOT")"
if [[ -d "$ROOT/.git" ]]; then
  git -C "$ROOT" fetch origin "$REF"
  git -C "$ROOT" checkout "$REF"
  git -C "$ROOT" pull --ff-only origin "$REF"
else
  git clone --branch "$REF" "$REPO_URL" "$ROOT"
fi

if [[ -d "$ROOT/vendor/jacobian-lens/.git" ]]; then
  git -C "$ROOT/vendor/jacobian-lens" fetch origin "$JLENS_REV"
else
  git clone "$JLENS_URL" "$ROOT/vendor/jacobian-lens"
fi
git -C "$ROOT/vendor/jacobian-lens" checkout --detach "$JLENS_REV"

cd "$ROOT"
uv sync --frozen
uv pip install --python .venv/bin/python -e vendor/jacobian-lens
mkdir -p logs/orcd .hf

cat <<EOF
Bootstrap complete.
  root: $ROOT
  code ref: $(git rev-parse --short HEAD)
  jlens ref: $(git -C vendor/jacobian-lens rev-parse --short HEAD)

Next, stage public artifacts from this login node, for example:
  scripts/orcd/stage_assets.sh qwen3-4b
EOF

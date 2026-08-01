#!/bin/zsh
# Overnight penultimate-lens fit with restart-on-crash (jlens checkpoints per prompt).
cd /Users/peterflo/conductor/workspaces/hw0-v1/helsinki
log() { echo "$(date '+%F %T') $1" >> logs/supervisor.log }
export PYTORCH_ENABLE_MPS_FALLBACK=1
restarts=0
until .venv/bin/python -u -m hw0.fit_lens >> logs/fit_lens.log 2>&1; do
  code=$?
  restarts=$((restarts+1))
  log "lens fit crashed (restart $restarts, exit=$code) — resuming from jlens checkpoint"
  if [ $restarts -ge 60 ]; then log "lens fit: too many restarts, giving up"; exit 1; fi
  sleep 15
done
log "LENS_FIT_COMPLETE"

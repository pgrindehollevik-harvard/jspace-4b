#!/bin/zsh
# Overnight chain: penultimate-lens fit -> ladder re-run with the new lens ->
# grid if the gate passes. Everything restart-on-crash; all checkpointed.
cd "$(dirname "$0")/.."
log() { echo "$(date '+%F %T') $1" >> logs/supervisor.log }
export PYTORCH_ENABLE_MPS_FALLBACK=1

restarts=0
until .venv/bin/python -u -m jspace.fit_lens >> logs/fit_lens.log 2>&1; do
  code=$?
  restarts=$((restarts+1))
  log "lens fit crashed (restart $restarts, exit=$code) — resuming from jlens checkpoint"
  if [ $restarts -ge 60 ]; then log "lens fit: too many restarts, giving up"; exit 1; fi
  sleep 15
done
log "LENS_FIT_COMPLETE"

export JSPACE_LENS_PATH=results/lens_fit/qwen3-4b_pen_n32.pt
restarts=0
until .venv/bin/python -u -m jspace.calibrate >> logs/calibrate_pen.log 2>&1; do
  code=$?
  restarts=$((restarts+1))
  log "pen ladder crashed (restart $restarts, exit=$code) — resuming from checkpoint"
  if [ $restarts -ge 40 ]; then log "pen ladder: too many restarts, giving up"; exit 1; fi
  sleep 15
done
gate=$(python3 -c "import json; print(json.load(open('results/calibration_pen.json'))['stop_gate'])")
log "pen ladder done, stop_gate=$gate"
git add results/calibration_pen.json results/calibration_state_pen.json results/lens_fit/*.pt 2>/dev/null
git commit -q -m "Record penultimate-lens ladder verdict

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" 2>/dev/null && log "pen ladder committed"

if [ "$gate" != "PASS" ]; then
  log "pen stop-gate FAIL: negative replication is the result; grid does not run"
  exit 0
fi

export HF_HUB_OFFLINE=1
stalled=0
last_lines=-1
until .venv/bin/python -u -m jspace.run_grid --calibration results/calibration_pen.json >> logs/grid.log 2>&1; do
  code=$?
  lines=$(wc -l < results/grid.jsonl 2>/dev/null || echo 0)
  if [ "$lines" -gt "$last_lines" ]; then stalled=0; else stalled=$((stalled+1)); fi
  last_lines=$lines
  log "grid crashed (exit=$code, progress=$lines lines, consecutive-no-progress=$stalled)"
  if [ $stalled -ge 8 ]; then log "grid: no progress across 8 restarts, giving up"; exit 1; fi
  sleep 30
done
log "GRID_COMPLETE"

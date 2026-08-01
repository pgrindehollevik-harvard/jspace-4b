#!/bin/zsh
# Unattended chain: calibration (restart-on-crash, checkpointed stages resume) ->
# freeze results in git -> main grid with progress-aware auto-restart.
# Silent SIGKILLs during sustained MPS load are an observed failure mode on this
# machine; the policy is: restart forever WHILE progress is being made, give up
# only after 8 consecutive restarts with no new work recorded.
cd /Users/peterflo/conductor/workspaces/hw0-v1/helsinki
log() { echo "$(date '+%F %T') $1" >> logs/supervisor.log }

restarts=0
until .venv/bin/python -u -m hw0.calibrate >> logs/calibrate.log 2>&1; do
  code=$?
  restarts=$((restarts+1))
  log "calibration crashed (restart $restarts, exit=$code) — resuming from checkpoint"
  if [ $restarts -ge 40 ]; then log "calibration: too many restarts, giving up"; exit 1; fi
  sleep 15
done
if [ ! -f results/calibration.json ]; then
  log "calibration exited 0 without writing results — grid NOT started"
  exit 1
fi
gate=$(python3 -c "import json; print(json.load(open('results/calibration.json'))['stop_gate'])")
log "calibration done, stop_gate=$gate"
git add results/calibration.json results/calibration_state.json 2>/dev/null
git commit -q -m "Record calibration results and freeze band choice

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" && log "calibration committed"

if [ "$gate" != "PASS" ]; then
  log "stop-gate FAIL: per pre-registration the main grid must not run"
  exit 0
fi

export HF_HUB_OFFLINE=1
stalled=0
last_lines=-1
until .venv/bin/python -u -m hw0.run_grid >> logs/grid.log 2>&1; do
  code=$?
  lines=$(wc -l < results/grid.jsonl 2>/dev/null || echo 0)
  if [ "$lines" -gt "$last_lines" ]; then stalled=0; else stalled=$((stalled+1)); fi
  last_lines=$lines
  log "grid crashed (exit=$code, progress=$lines lines, consecutive-no-progress=$stalled)"
  if [ $stalled -ge 8 ]; then log "grid: no progress across 8 restarts, giving up"; exit 1; fi
  sleep 30
done
log "GRID_COMPLETE"

#!/bin/zsh
# Unattended chain: calibration (restart-on-crash, checkpointed stages resume) ->
# freeze results in git -> main grid with auto-restart. Detached via nohup+caffeinate.
cd /Users/peterflo/conductor/workspaces/hw0-v1/helsinki
log() { echo "$(date '+%F %T') $1" >> logs/supervisor.log }

restarts=0
until .venv/bin/python -u -m hw0.calibrate >> logs/calibrate.log 2>&1; do
  restarts=$((restarts+1))
  log "calibration crashed (restart $restarts) — resuming from checkpoint"
  if [ $restarts -ge 10 ]; then log "calibration: too many restarts, giving up"; exit 1; fi
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
restarts=0
until .venv/bin/python -u -m hw0.run_grid >> logs/grid.log 2>&1; do
  restarts=$((restarts+1))
  log "grid crashed (restart $restarts)"
  if [ $restarts -ge 20 ]; then log "grid: too many restarts, giving up"; exit 1; fi
  sleep 30
done
log "GRID_COMPLETE"

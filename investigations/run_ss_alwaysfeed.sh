#!/usr/bin/env bash
# "Just try a fix" — does training the model MORE on its own predictions stop the
# C-113 snowball? Crank scheduled-sampling always-feed (ss_epsilon_max 0.25 -> 1.0) on
# violet (active balancer, seed 42) and retrain. GPU-enforced (no silent CPU fallback);
# the config is trap-restored; diagnose_io_gain on the new artifact tells us if the
# free-running rollout now stays in-range (snowball stopped) or still explodes.
set -u
REPO="/home/simon/Documents/scripts/views_platform/views-models"
HN="/home/simon/Documents/scripts/views_platform/views-hydranet"
CONFIG="$REPO/models/violet_visitor/configs/config_hyperparameters.py"
ART="$REPO/models/violet_visitor/artifacts"
ENV="views-hydranet-env"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$REPO/logs/ss_alwaysfeed_${TS}.log"
mkdir -p "$REPO/logs"
echo "================ always-feed retrain — START $(date) ================"
echo "[log] $LOG"

PRE=$(conda run -n "$ENV" python -c "import torch;print('CUDAFLAG='+str(torch.cuda.is_available()))" 2>&1)
echo "[preflight] $(echo "$PRE" | grep -o 'CUDAFLAG=[A-Za-z]*')"
echo "$PRE" | grep -q "CUDAFLAG=True" || { echo "[ABORT] CUDA down — reload nvidia_uvm."; exit 3; }

BAK="$CONFIG.ssbak_${TS}"; cp "$CONFIG" "$BAK"
restore() { cp "$BAK" "$CONFIG"; echo "[restore] $(date) config restored: $(grep ss_epsilon_max "$CONFIG" | tr -d ' ')"; }
trap restore EXIT

sed -i -E "s/('ss_epsilon_max'[[:space:]]*:[[:space:]]*)[0-9.]+/\11.0/" "$CONFIG"
echo "[cfg] $(grep -E "ss_epsilon_max|freeze_multitask_balancer" "$CONFIG" | tr '\n' ' ')"
before="$(ls -t "$ART"/*.pt 2>/dev/null | head -1)"

cd "$REPO/models/violet_visitor" || { echo "[ABORT] no model dir"; exit 4; }
bash run.sh --train --run_type calibration > "$LOG" 2>&1 &
bg=$!
ok=no
for i in $(seq 1 42); do
  sleep 10
  kill -0 "$bg" 2>/dev/null || { echo "[verify] train exited early (see log)"; break; }
  if grep -qa "RUNNING ON CPU" "$LOG"; then echo "[ABORT-CPU] silent CPU fallback"; pkill -f "violet_visitor/main.py"; kill "$bg" 2>/dev/null; break; fi
  pid="$(pgrep -f 'violet_visitor/main.py --train' | head -1)"
  if [ -n "$pid" ] && nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qw "$pid"; then
    echo "[verify] ✓ on GPU (pid $pid) after ${i}0s"; ok=yes; break; fi
done
[ "$ok" = yes ] || echo "[WARN] not confirmed on GPU within ~7 min — check $LOG"

wait "$bg"; rc=$?
after="$(ls -t "$ART"/*.pt 2>/dev/null | head -1)"
echo "==== train END $(date) rc=$rc artifact=$(basename "$after") ===="
if [ "$rc" -eq 0 ] && [ -n "$after" ] && [ "$after" != "$before" ]; then
  echo ""; echo "================ diagnose_io_gain — DID THE SNOWBALL STOP? ================"
  conda run -n "$ENV" python "$HN/scripts/diagnose_io_gain.py" "$after" --label "violet_alwaysfeed"
else
  echo "[!] train failed / no new artifact"; grep -aE "RuntimeError|CUDA error|Error:" "$LOG" | tail -3
fi
echo "================ DONE $(date) ================"

#!/usr/bin/env bash
# freeze_h-removal characterization gate (register M-RT5, C-113/C-125).
# Eval pink_pirate (HEALTHY reference) on its EXISTING artifact with the
# freeze_h-removed inference path (views-hydranet branch chore/retire-freeze-h,
# editable-installed). Confirms dropping freeze_h="hl" → always-"none" does NOT
# regress CRPS/MCR vs pink's known reference (~lr_sb 0.13 / lr_ns 0.03 / lr_os 0.05).
# GPU-ENFORCED: CUDA pre-flight + on-GPU PID verification (no silent CPU fallback, C-115).
# No training, no config edits (the stray 'freeze_h' config key is now an ignored extra).

set -u
REPO_DIR="/home/simon/Documents/scripts/views_platform/views-models"
MODEL="$REPO_DIR/models/pink_pirate"
ART="calibration_model_20260603_145536.pt"
ENV="views-hydranet-env"
LOGDIR="$REPO_DIR/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/freezeh_pink_eval_${TS}.log"
mkdir -p "$LOGDIR"

echo "================ freeze_h-removal pink eval — START $(date) ================"
echo "[log] $LOG"

# ---- preflight: CUDA must be up (no silent CPU fallback) ----
PRE=$(conda run -n "$ENV" python -c "import torch;print('CUDAFLAG='+str(torch.cuda.is_available()))" 2>&1)
echo "[preflight] $(echo "$PRE" | grep -o 'CUDAFLAG=[A-Za-z]*')"
echo "$PRE" | grep -q "CUDAFLAG=True" || { echo "[ABORT] CUDA down — reload nvidia_uvm."; exit 3; }

# ---- launch eval (no train; saved artifact) ----
cd "$MODEL" || { echo "[ABORT] no model dir"; exit 4; }
echo "[cmd] bash run.sh -r calibration -e -sa -a $ART"
bash run.sh -r calibration -e -sa -a "$ART" > "$LOG" 2>&1 &
BG=$!

# ---- verify on GPU within ~6 min; abort on CPU fallback ----
ok=no
for i in $(seq 1 36); do
  sleep 10
  kill -0 "$BG" 2>/dev/null || { echo "[verify] eval process exited early (see log)"; break; }
  if grep -qa "RUNNING ON CPU" "$LOG"; then
    echo "[ABORT-CPU] silent CPU fallback detected"; pkill -f "pink_pirate/main.py"; kill "$BG" 2>/dev/null; break
  fi
  pid="$(pgrep -f 'pink_pirate/main.py' | head -1)"
  if [ -n "$pid" ] && nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qw "$pid"; then
    echo "[verify] ✓ on GPU (pid $pid) after ${i}0s"; ok=yes; break
  fi
done
if [ "$ok" != yes ]; then
  echo "[WARN] never confirmed on GPU within 6 min — letting it run, but check $LOG for CPU fallback."
fi

# ---- wait for completion (exit 137 = post-metric OOM, C-116 — metrics dumped first) ----
wait "$BG"; rc=$?
echo "==== eval END $(date) rc=$rc (137 = expected post-metric OOM) ===="

echo ""
echo "================ CRPS / MCR (from log) ================"
grep -aiE "crps|mcr|lr_sb|lr_ns|lr_os" "$LOG" | grep -aviE "warning|debug" | tail -40
echo "================ DONE — full log: $LOG ================"

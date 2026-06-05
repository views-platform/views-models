#!/usr/bin/env bash
# Balancer × seed sweep for C-124/C-113 (multi-seed confirmation).
# Pre-analysis: views-hydranet/reports/preanalysis_balancer_sweep.md
#
# 3x2 factorial on the violet base config: seed {42,4,99} x freeze_multitask_balancer {F,T}.
# Seed 42 is ALREADY DONE (bisect: active=...233938, frozen=...051634) — reused, not retrained.
# This driver trains the 4 NEW cells: seeds {4,99} x {active,frozen}, ~80 min each (~5.3h).
# GPU-ENFORCED (no silent CPU fallback, C-115); config trap-restored.

HN_DIR="/home/simon/Documents/scripts/views_platform/views-hydranet"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_DIR/models/violet_visitor/configs/config_hyperparameters.py"
ART="$REPO_DIR/models/violet_visitor/artifacts"
LOGDIR="$REPO_DIR/logs"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS="$LOGDIR/balancer_sweep_RESULTS.txt"
MASTER="$LOGDIR/balancer_sweep_master_${TS}.log"
ENV="views-hydranet-env"

mkdir -p "$LOGDIR"
exec > >(tee -a "$MASTER") 2>&1
echo "================ balancer x seed sweep — START $(date) ================"

BAK="$CONFIG.sweepbak_${TS}"
cp "$CONFIG" "$BAK"
restore() { cp "$BAK" "$CONFIG"; echo "[restore] $(date) config restored"; \
  grep -E "'(np_seed|torch_seed|freeze_multitask_balancer)'" "$CONFIG" | sed 's/^/  /'; }
trap restore EXIT

# ---- preflight: CUDA must be up (no silent CPU fallback) ----
PRE=$(conda run -n "$ENV" python -c "import torch;print('CUDAFLAG='+str(torch.cuda.is_available()))" 2>&1)
echo "[preflight] $(echo "$PRE" | grep -o 'CUDAFLAG=[A-Za-z]*')"
echo "$PRE" | grep -q "CUDAFLAG=True" || { echo "[ABORT] CUDA down — reload nvidia_uvm / clear wedge."; exit 3; }

cat > "$RESULTS" <<EOF
# Balancer x seed sweep — run $TS
# Pre-reg: views-hydranet/reports/preanalysis_balancer_sweep.md
# readout: free-running attractor (Part B). in-range <log13 healthy; >log20 pathological.
# seed 42 (reused from bisect): active ...233938 -> ~log16 PATHOLOGICAL ; frozen ...051634 -> ~log4-5 healthy
EOF

set_cell() {  # $1=seed  $2=frozen(true|false)
  sed -i -E "s/('np_seed': *)[0-9]+/\1$1/; s/('torch_seed': *)[0-9]+/\1$1/" "$CONFIG"
  sed -i "/'freeze_multitask_balancer'/d" "$CONFIG"
  if [ "$2" = "true" ]; then
    python3 - "$CONFIG" <<'PY'
import re,sys
p=sys.argv[1]; s=open(p).read()
new,n=re.subn(r"(\n(\s*)'freeze_h':[^\n]*\n)", r"\1\2'freeze_multitask_balancer': True,\n", s, count=1)
assert n==1; open(p,'w').write(new)
PY
  fi
}

run_cell() {  # $1=seed $2=frozen
  local seed="$1" frozen="$2" label="seed${1}_$( [ "$2" = true ] && echo frozen || echo active )"
  local tl="$LOGDIR/balancer_sweep_${label}_${TS}.log"
  local dl="$LOGDIR/balancer_sweep_${label}_diag_${TS}.log"
  echo ""; echo "==== CELL $label (seed=$seed frozen=$frozen) TRAIN START $(date) ===="
  set_cell "$seed" "$frozen"
  echo "[cfg] $(grep -E "'(np_seed|freeze_multitask_balancer)'" "$CONFIG" | tr '\n' ' ')"
  local before; before="$(ls -t "$ART"/*.pt 2>/dev/null | head -1)"
  bash "$REPO_DIR/models/violet_visitor/run.sh" --train --run_type calibration > "$tl" 2>&1 &
  local bg=$!
  local ok=no
  for i in $(seq 1 36); do
    sleep 10
    kill -0 "$bg" 2>/dev/null || { echo "[verify] train exited early"; break; }
    if grep -qa "RUNNING ON CPU" "$tl"; then echo "[ABORT-CELL] CPU fallback"; pkill -f "violet_visitor/main.py --train"; kill "$bg" 2>/dev/null; break; fi
    local pid; pid="$(pgrep -f 'violet_visitor/main.py --train' | head -1)"
    if [ -n "$pid" ] && nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qw "$pid"; then
      echo "[verify] ✓ on GPU (pid $pid) after ${i}0s"; ok=yes; break; fi
  done
  if [ "$ok" != yes ]; then echo "[ABORT-CELL] never confirmed on GPU"; pkill -f "violet_visitor/main.py --train"; kill "$bg" 2>/dev/null;
    echo "=== CELL: $label | NOT ON GPU — skipped ===" >> "$RESULTS"; return; fi
  wait "$bg"; local rc=$?
  local after; after="$(ls -t "$ART"/*.pt 2>/dev/null | head -1)"
  echo "==== CELL $label TRAIN END $(date) rc=$rc artifact=$(basename "$after") ===="
  { echo ""; echo "=== CELL: $label (seed=$seed frozen=$frozen) | train_exit=$rc | $(date) ==="; } >> "$RESULTS"
  if [ "$rc" -ne 0 ] || [ "$after" = "$before" ] || [ -z "$after" ]; then
    { echo "  (!) train failed / no new artifact"; grep -aE "RuntimeError|CUDA error|Error:" "$tl" | grep -avE "Lesson|month/s" | tail -3 | sed 's/^/  ERR> /'; } >> "$RESULTS"
  else
    conda run -n "$ENV" python "$HN_DIR/scripts/diagnose_io_gain.py" "$after" --label "$label" > "$dl" 2>&1
    { echo "  artifact: $(basename "$after")"; grep -E "Part B|seed U|U\[0|PATHOLOGICAL|healthy \(in-range\)" "$dl" | sed 's/^/  /'; } >> "$RESULTS"
  fi
}

run_cell 4 false
run_cell 4 true
run_cell 99 false
run_cell 99 true

echo ""; echo "================ SWEEP DONE $(date) ================"
cat "$RESULTS"
# trap restores config

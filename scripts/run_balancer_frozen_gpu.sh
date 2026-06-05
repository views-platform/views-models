#!/usr/bin/env bash
# Re-run the TEST (frozen balancer) arm of the C-111 bisect — GPU-ENFORCED.
# Pre-analysis: views-hydranet/reports/preanalysis_balancer_bisect.md
#
# Background: the prior frozen rerun SILENTLY fell back to CPU after the control
# arm's "CUDA unspecified launch failure" wedged the driver/GPU. This driver makes
# that impossible:
#   (1) HARD pre-flight gate — refuses to launch unless torch.cuda.is_available().
#   (2) Post-launch verification — aborts within minutes if the train process is
#       NOT on the GPU (or if the "RUNNING ON CPU" warning appears).
# Run only AFTER a reboot has cleared the wedged CUDA context.

HN_DIR="/home/simon/Documents/scripts/views_platform/views-hydranet"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_DIR/models/violet_visitor/configs/config_hyperparameters.py"
ART="$REPO_DIR/models/violet_visitor/artifacts"
LOGDIR="$REPO_DIR/logs"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS="$LOGDIR/balancer_bisect_RESULTS.txt"
TRAINLOG="$LOGDIR/balancer_frozen_gpu_train_${TS}.log"
DIAGLOG="$LOGDIR/balancer_frozen_gpu_diag_${TS}.log"
MASTER="$LOGDIR/balancer_frozen_gpu_master_${TS}.log"
ENVNAME="views-hydranet-env"

mkdir -p "$LOGDIR"
exec > >(tee -a "$MASTER") 2>&1
echo "=== TEST arm rerun (frozen balancer) — GPU-ENFORCED — START: $(date) ==="

# ---- (1) HARD CUDA PRE-FLIGHT GATE (no silent CPU fallback) -------------------
echo "[preflight] checking torch.cuda.is_available() ..."
# robust: grep a sentinel token in combined output (tail -1 was fragile under conda run)
PRE=$(conda run -n "$ENVNAME" python -c "import torch; print('CUDAFLAG=' + str(torch.cuda.is_available()))" 2>&1)
echo "[preflight] $(echo "$PRE" | grep -o 'CUDAFLAG=[A-Za-z]*' || echo 'CUDAFLAG=<unparsed>')"
if ! echo "$PRE" | grep -q "CUDAFLAG=True"; then
    echo "[ABORT] CUDA not available — refusing to train on CPU. (reload nvidia_uvm / clear the wedge, then re-run)"
    echo "[ABORT] raw preflight output:"; echo "$PRE" | tail -5
    exit 3
fi
echo "[preflight] GPU OK:"; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null

# ---- config: insert frozen flag; restore on ANY exit -------------------------
BAK="$CONFIG.gpurerun_bak_${TS}"
cp "$CONFIG" "$BAK"
restore_cfg() { cp "$BAK" "$CONFIG"; echo "[restore] $(date) config restored"; \
    grep -q freeze_multitask_balancer "$CONFIG" && echo "  [WARN] flag present" || echo "  OK flag absent"; }
trap restore_cfg EXIT
sed -i "/'freeze_multitask_balancer'/d" "$CONFIG"
python3 - "$CONFIG" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
new, n = re.subn(r"(\n(\s*)'freeze_h':[^\n]*\n)",
                 r"\1\2'freeze_multitask_balancer': True,\n", s, count=1)
assert n == 1, f"expected 1 insertion, got {n}"
open(p, "w").write(new)
PY
echo "[config] $(grep "'freeze_multitask_balancer'" "$CONFIG")"

before="$(ls -t "$ART"/*.pt 2>/dev/null | head -1)"
echo "[train] launching (bg) -> $TRAINLOG"
bash "$REPO_DIR/models/violet_visitor/run.sh" --train --run_type calibration > "$TRAINLOG" 2>&1 &
TRAIN_BG=$!

# ---- (2) POST-LAUNCH GPU VERIFICATION (abort fast if not on GPU) -------------
verified=no
for i in $(seq 1 36); do          # up to ~6 min of polling
    sleep 10
    if ! kill -0 "$TRAIN_BG" 2>/dev/null; then echo "[verify] train process exited early"; break; fi
    if grep -qa "RUNNING ON CPU" "$TRAINLOG"; then
        echo "[ABORT] training reported RUNNING ON CPU — killing."
        pkill -f "violet_visitor/main.py --train"; kill "$TRAIN_BG" 2>/dev/null; exit 4
    fi
    PID=$(pgrep -f "violet_visitor/main.py --train" | head -1)
    if [ -n "$PID" ] && nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qw "$PID"; then
        MEMUSE=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null | head -1)
        echo "[verify] ✓ CONFIRMED on GPU: pid $PID, gpu mem $MEMUSE (after ${i}0s)"
        verified=yes; break
    fi
done
if [ "$verified" != yes ]; then
    echo "[ABORT] never confirmed train pid on the GPU within ~6 min — killing to avoid a CPU grind."
    pkill -f "violet_visitor/main.py --train"; kill "$TRAIN_BG" 2>/dev/null; exit 5
fi

echo "[train] confirmed on GPU — waiting for completion ..."
wait "$TRAIN_BG"; rc=$?
after="$(ls -t "$ART"/*.pt 2>/dev/null | head -1)"
echo "=== TRAIN END: $(date) | exit=$rc | newest: $(basename "$after") ==="

{
    echo ""
    echo "=== ARM: test (frozen=true) GPU-RERUN | train_exit=$rc | $(date) ==="
} >> "$RESULTS"
if [ "$rc" -ne 0 ] || [ "$after" = "$before" ] || [ -z "$after" ]; then
    { echo "  (!) training failed (rc=$rc, no new artifact). tail:"; \
      grep -aE "RuntimeError|CUDA error|Error:|assert" "$TRAINLOG" | grep -avE "Lesson|month/s" | tail -5 | sed 's/^/  ERR> /'; } | tee -a "$RESULTS"
else
    echo "[diag] $(basename "$after") -> $DIAGLOG"
    conda run -n "$ENVNAME" python "$HN_DIR/scripts/diagnose_io_gain.py" "$after" --label "violet_frozen" > "$DIAGLOG" 2>&1
    { echo "  frozen artifact: $(basename "$after")"; \
      grep -E "Part B|seed U|U\[0|PATHOLOGICAL|healthy \(in-range\)" "$DIAGLOG" | sed 's/^/  /'; } | tee -a "$RESULTS"
fi
echo "" | tee -a "$RESULTS"
echo "=== GPU-RERUN DONE: $(date) ==="
# trap restores config on exit

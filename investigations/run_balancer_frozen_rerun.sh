#!/usr/bin/env bash
# Re-run ONLY the TEST (frozen balancer) arm of the C-111 bisect.
# The first attempt crashed ~4 min in with a transient "CUDA error: unspecified
# launch failure" (the control trained fine on the same GPU immediately before).
# Pre-analysis: views-hydranet/reports/preanalysis_balancer_bisect.md
#
# Fixes the prior driver bug (return-inside-pipe ran a bogus diagnostic on the
# wrong artifact): here the diagnostic runs ONLY on a genuinely new artifact.

HN_DIR="/home/simon/Documents/scripts/views_platform/views-hydranet"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_DIR/models/violet_visitor/configs/config_hyperparameters.py"
ART="$REPO_DIR/models/violet_visitor/artifacts"
LOGDIR="$REPO_DIR/logs"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS="$LOGDIR/balancer_bisect_RESULTS.txt"
TRAINLOG="$LOGDIR/balancer_frozen_rerun_train_${TS}.log"
DIAGLOG="$LOGDIR/balancer_frozen_rerun_diag_${TS}.log"
MASTER="$LOGDIR/balancer_frozen_rerun_master_${TS}.log"

mkdir -p "$LOGDIR"
exec > >(tee -a "$MASTER") 2>&1
echo "=== TEST arm rerun (frozen balancer) START: $(date) ==="

BAK="$CONFIG.frozenrerun_bak_${TS}"
cp "$CONFIG" "$BAK"
restore_cfg() { cp "$BAK" "$CONFIG"; echo "[restore] $(date) — config restored"; \
    grep -q freeze_multitask_balancer "$CONFIG" && echo "  [WARN] flag present" || echo "  OK flag absent"; }
trap restore_cfg EXIT

# insert freeze_multitask_balancer: True after freeze_h
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
echo "[before] newest artifact: $(basename "$before")"
echo "[train] -> $TRAINLOG"
bash "$REPO_DIR/models/violet_visitor/run.sh" --train --run_type calibration > "$TRAINLOG" 2>&1
rc=$?
after="$(ls -t "$ART"/*.pt 2>/dev/null | head -1)"
echo "=== TRAIN END: $(date) | exit=$rc | newest now: $(basename "$after") ==="

{
    echo ""
    echo "=== ARM: test (frozen=true) RERUN | train_exit=$rc | $(date) ==="
} >> "$RESULTS"

if [ "$rc" -ne 0 ] || [ "$after" = "$before" ] || [ -z "$after" ]; then
    {
        echo "  (!) training FAILED again (rc=$rc, no new artifact). Error tail:"
        grep -aE "RuntimeError|CUDA error|Error:|assert|nan|NaN|inf" "$TRAINLOG" \
            | grep -avE "Training \| Lesson|month/s" | tail -5 | sed 's/^/  ERR> /'
    } | tee -a "$RESULTS"
else
    echo "[diag] frozen artifact: $(basename "$after") -> $DIAGLOG"
    conda run -n views-hydranet-env python "$HN_DIR/scripts/diagnose_io_gain.py" \
        "$after" --label "violet_frozen" > "$DIAGLOG" 2>&1
    drc=$?
    {
        echo "  frozen artifact: $(basename "$after") (diag_exit=$drc)"
        grep -E "Part B|seed U|U\[0|PATHOLOGICAL|healthy \(in-range\)" "$DIAGLOG" | sed 's/^/  /'
        [ "$drc" -ne 0 ] && echo "  (diag error — see $DIAGLOG)"
    } | tee -a "$RESULTS"
fi
echo "" | tee -a "$RESULTS"
echo "=== RERUN DONE: $(date) ==="
# trap restores config
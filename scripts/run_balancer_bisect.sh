#!/usr/bin/env bash
# C-111 balancer-freeze bisect for C-113 (violet_visitor).
# Pre-analysis: views-hydranet/reports/preanalysis_balancer_bisect.md
#
# Retrains violet TWICE from scratch (one GPU at a time):
#   CONTROL: balancer ACTIVE (C-111 default)  -> expect explosion (reproduce)
#   TEST:    balancer FROZEN (pre-C-111)       -> expect in-range if C-111 is the cause
# Readout = retrain-free scripts/diagnose_io_gain.py on each fresh artifact
#   (free-running attractor: in-range ~log 10 healthy vs ~log 40 pathological).
# Config-only insertion of freeze_multitask_balancer; trap-restores config.
# NO `set -e` (a crash must not skip restore). The June-3 artifact is preserved
# (new trains are timestamped); diagnostic runs on the EXACT new artifact.

HN_DIR="/home/simon/Documents/scripts/views_platform/views-hydranet"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_DIR/models/violet_visitor/configs/config_hyperparameters.py"
ART="$REPO_DIR/models/violet_visitor/artifacts"
LOGDIR="$REPO_DIR/logs"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS="$LOGDIR/balancer_bisect_RESULTS.txt"
MASTER="$LOGDIR/balancer_bisect_master_${TS}.log"

mkdir -p "$LOGDIR"
exec > >(tee -a "$MASTER") 2>&1

echo "================================================================"
echo " C-111 balancer-freeze bisect — violet_visitor"
echo " CONTROL (active, expect explode) -> TEST (frozen, expect in-range)"
echo " started: $(date)"
echo "================================================================"

BAK="$CONFIG.bisect_bak_${TS}"
cp "$CONFIG" "$BAK"
restore_cfg() { cp "$BAK" "$CONFIG"; echo "[restore] $(date) — config restored"; \
                grep -q "freeze_multitask_balancer" "$CONFIG" \
                && echo "  [WARN] flag still present" || echo "  OK: flag absent"; }
trap restore_cfg EXIT

set_frozen() {  # $1 = true|false
    # remove any existing flag line first, then insert if true
    sed -i "/'freeze_multitask_balancer'/d" "$CONFIG"
    if [ "$1" = "true" ]; then
        python3 - "$CONFIG" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
new, n = re.subn(r"(\n(\s*)'freeze_h':[^\n]*\n)",
                 r"\1\2'freeze_multitask_balancer': True,\n", s, count=1)
assert n == 1, f"expected 1 insertion, got {n}"
open(p, "w").write(new)
PY
    fi
}

echo "# C-111 balancer bisect — run $TS" > "$RESULTS"
echo "# readout: free-running attractor (Part B). in-range <log13 healthy; >log20 pathological." >> "$RESULTS"
echo "# baseline (June-3, active): violet attractor ~log 40 -> expm1 ~1e17" >> "$RESULTS"
echo "" >> "$RESULTS"

run_arm() {  # $1 = arm name  $2 = frozen(true|false)
    local arm="$1" frozen="$2"
    local trainlog="$LOGDIR/balancer_bisect_${arm}_train_${TS}.log"
    local diaglog="$LOGDIR/balancer_bisect_${arm}_diag_${TS}.log"
    echo ""
    echo "----------------------------------------------------------------"
    echo "=== ARM '$arm' (frozen=$frozen) TRAIN START: $(date) ==="
    set_frozen "$frozen"
    echo "[config] $(grep -E "'freeze_multitask_balancer'" "$CONFIG" || echo "(flag absent => active, default)")"

    local before; before="$(ls -t "$ART"/*.pt 2>/dev/null | head -1)"
    bash "$REPO_DIR/models/violet_visitor/run.sh" --train --run_type calibration > "$trainlog" 2>&1
    local rc=$?
    local after; after="$(ls -t "$ART"/*.pt 2>/dev/null | head -1)"
    echo "=== ARM '$arm' TRAIN END: $(date) | exit=$rc ==="
    echo "  newest artifact: $after"

    {
        echo "=== ARM: $arm (frozen=$frozen) | train_exit=$rc | $(date) ==="
        if [ "$after" = "$before" ] || [ -z "$after" ]; then
            echo "  (!) no new artifact produced — training failed; see $trainlog"
            grep -iE "Traceback|Error:|Exception|CUDA" "$trainlog" | tail -3 | sed 's/^/  ERR> /'
            echo ""
            return
        fi
        echo "  artifact: $(basename "$after")"
        echo "  --- diagnostic (Part B attractor) ---"
    } | tee -a "$RESULTS"

    conda run -n views-hydranet-env python "$HN_DIR/scripts/diagnose_io_gain.py" \
        "$after" --label "violet_${arm}" > "$diaglog" 2>&1
    # capture Part B rollout lines + verdicts
    grep -E "Part B|seed U|U\[0|PATHOLOGICAL|healthy \(in-range\)" "$diaglog" | sed 's/^/  /' | tee -a "$RESULTS"
    echo "" | tee -a "$RESULTS"
}

run_arm control false
run_arm test true

echo ""
echo "================================================================"
echo " BISECT DONE: $(date)"
echo "================================================================"
cat "$RESULTS"
# trap restores config on exit

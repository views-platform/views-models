#!/usr/bin/env bash
# In-domain feedback-clamp experiment for C-113.
# Pre-analysis: views-hydranet/reports/preanalysis_feedback_clamp.md
#
# Re-evaluates violet_visitor (test) and pink_pirate (control) with the
# autoregressive feedback clamped to the per-target log1p data max
# [sb=10.7828, ns=7.6014, os=12.0921]. Config-only insertion of
# feedback_clamp_log1p; clamp logic lives in views-hydranet (editable install).
# One model on the GPU at a time. NO `set -e` (a crash must not skip restore).
#
# Launch: nohup bash scripts/run_feedback_clamp.sh &

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$REPO_DIR/logs"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS="$LOGDIR/feedback_clamp_RESULTS.txt"
MASTER="$LOGDIR/feedback_clamp_master_${TS}.log"
CLAMP="[10.7828, 7.6014, 12.0921]"

mkdir -p "$LOGDIR"
exec > >(tee -a "$MASTER") 2>&1

echo "================================================================"
echo " feedback-clamp experiment — violet (test) + pink (control)"
echo " clamp (log1p, per target sb/ns/os): $CLAMP"
echo " started: $(date)"
echo "================================================================"

declare -A CFG
CFG[violet_visitor]="$REPO_DIR/models/violet_visitor/configs/config_hyperparameters.py"
CFG[pink_pirate]="$REPO_DIR/models/pink_pirate/configs/config_hyperparameters.py"

# --- safety: back up both configs; restore ALL on any exit ---------------------
declare -A BAK
for m in violet_visitor pink_pirate; do
    BAK[$m]="${CFG[$m]}.clamp_bak_${TS}"
    cp "${CFG[$m]}" "${BAK[$m]}"
done
restore_all() {
    for m in violet_visitor pink_pirate; do
        cp "${BAK[$m]}" "${CFG[$m]}"
    done
    echo ""
    echo "[restore] $(date) — both configs restored"
    grep -H "'feedback_clamp_log1p'\|'freeze_h'" "${CFG[violet_visitor]}" "${CFG[pink_pirate]}" \
        | grep "freeze_h" || true
    echo "  (feedback_clamp_log1p should be ABSENT below — restored to baseline)"
    grep -l "feedback_clamp_log1p" "${CFG[violet_visitor]}" "${CFG[pink_pirate]}" \
        && echo "  [WARN] clamp key still present!" || echo "  OK: clamp key absent in both"
}
trap restore_all EXIT

insert_clamp() {  # $1 = config path
    python3 - "$1" "$CLAMP" <<'PY'
import re, sys
p, clamp = sys.argv[1], sys.argv[2]
s = open(p).read()
if "feedback_clamp_log1p" in s:
    sys.exit(0)  # idempotent
new, n = re.subn(r"(\n(\s*)'freeze_h':[^\n]*\n)",
                 r"\1\2'feedback_clamp_log1p': " + clamp + ",\n",
                 s, count=1)
assert n == 1, f"expected 1 insertion, made {n}"
open(p, "w").write(new)
PY
}

extract() {  # $1=model $2=armlog $3=rc
    {
        echo "=== MODEL: $1 | exit=$3 | $(date) ==="
        if grep -qE "step-wise/lr_sb_best/CRPS " "$2"; then
            grep -E "(step-wise|time-series-wise)/lr_(sb|ns|os)_best/(CRPS|MCR_sample) " "$2" \
                | sed -E 's/^wandb:[[:space:]]*/  /'
        else
            echo "  (!) no wandb CRPS summary — run likely crashed (or OOM teardown); see $2"
            grep -iE "Traceback|Error:|Exception" "$2" | tail -3 | sed 's/^/  ERR> /'
        fi
        echo ""
    } | tee -a "$RESULTS"
}

echo "# feedback-clamp results — run $TS — clamp $CLAMP" > "$RESULTS"
echo "# baseline (no clamp): violet lr_sb CRPS=2.13e17, lr_ns=2.78e9; pink lr_sb=0.13" >> "$RESULTS"
echo "# pre-registered: violet returns in-range (<10); pink unchanged (<10%)" >> "$RESULTS"
echo "" >> "$RESULTS"

for m in violet_visitor pink_pirate; do
    ARMLOG="$LOGDIR/feedback_clamp_${m}_${TS}.log"
    echo ""
    echo "----------------------------------------------------------------"
    echo "=== $m START: $(date) ==="
    insert_clamp "${CFG[$m]}"
    echo "[config] $(grep "'feedback_clamp_log1p'" "${CFG[$m]}")"
    echo "[log]    $ARMLOG"
    bash "$REPO_DIR/models/$m/run.sh" \
        --evaluate --run_type calibration --saved --report > "$ARMLOG" 2>&1
    rc=$?
    echo "=== $m END: $(date) | exit=$rc ==="
    extract "$m" "$ARMLOG" "$rc"
done

echo ""
echo "================================================================"
echo " DONE: $(date)"
echo "================================================================"
cat "$RESULTS"
# trap restores both configs on exit

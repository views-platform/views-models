#!/usr/bin/env bash
# freeze_h channel-isolation ablation for C-113 (violet_visitor).
# Pre-analysis plan: views-hydranet/reports/preanalysis_freezeh_ablation.md
#
# Re-evaluates the SAME trained artifact under four freeze_h settings to localize
# which feedback channel carries the autoregressive divergence. Config-only change
# (freeze_h); no code modification. One model on the GPU at a time (sequential).
#
# Launch: nohup bash scripts/run_freezeh_ablation.sh & (or harness background)
#
# NOTE: deliberately NO `set -e` — one arm crashing must not abort the remaining
# arms nor skip the config restore (which runs via a trap on EXIT).

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_DIR/models/violet_visitor/configs/config_hyperparameters.py"
LOGDIR="$REPO_DIR/logs"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS="$LOGDIR/ablation_freezeh_RESULTS.txt"
MASTER="$LOGDIR/ablation_freezeh_master_${TS}.log"

mkdir -p "$LOGDIR"
exec > >(tee -a "$MASTER") 2>&1

echo "================================================================"
echo " freeze_h channel-isolation ablation — violet_visitor"
echo " started: $(date)"
echo " config:  $CONFIG"
echo " results: $RESULTS"
echo "================================================================"

# --- safety: back up config, restore on ANY exit (success/crash/kill) ----------
BAK="$CONFIG.ablation_bak_${TS}"
cp "$CONFIG" "$BAK"
restore_config() {
    cp "$BAK" "$CONFIG"
    echo ""
    echo "[restore] $(date) — config restored from $BAK"
    grep "'freeze_h'" "$CONFIG"
}
trap restore_config EXIT

# --- sanity gate: must start from freeze_h="hl" --------------------------------
START_FREEZE=$(grep -oE "'freeze_h':[[:space:]]*\"[a-z]+\"" "$CONFIG")
echo "[sanity] starting config: $START_FREEZE"
if ! echo "$START_FREEZE" | grep -q '"hl"'; then
    echo "[ABORT] expected freeze_h=\"hl\" at start; found: $START_FREEZE"
    exit 2
fi

set_freeze() {  # $1 = arm value
    sed -i -E "s/('freeze_h':[[:space:]]*)\"[a-z]+\"/\\1\"$1\"/" "$CONFIG"
}

extract_crps() {  # $1 = arm  $2 = arm log  $3 = exit code
    {
        echo "=== ARM: $1 | exit=$3 | $(date) ==="
        if grep -qE "step-wise/lr_sb_best/CRPS " "$2"; then
            grep -E "(step-wise|time-series-wise)/lr_(sb|ns|os)_best/CRPS " "$2" \
                | sed -E 's/^wandb:[[:space:]]*/  /'
        else
            echo "  (!) no wandb CRPS summary found — run likely crashed; see $2"
            grep -iE "Traceback|Error:|Exception" "$2" | tail -3 | sed 's/^/  ERR> /'
        fi
        echo ""
    } | tee -a "$RESULTS"
}

echo "# freeze_h ablation results — violet_visitor — run $TS" > "$RESULTS"
echo "# pre-analysis: views-hydranet/reports/preanalysis_freezeh_ablation.md" >> "$RESULTS"
echo "# primary endpoint: step-wise/lr_sb_best/CRPS  (BOUNDED <1, EXPLODED >1e3)" >> "$RESULTS"
echo "" >> "$RESULTS"

# --- arm order: hl (reproduce baseline) -> all (KEY) -> hs -> none --------------
for arm in hl all hs none; do
    ARMLOG="$LOGDIR/ablation_freezeh_${arm}_${TS}.log"
    echo ""
    echo "----------------------------------------------------------------"
    echo "=== ARM '$arm' START: $(date) ==="
    set_freeze "$arm"
    echo "[config] $(grep "'freeze_h'" "$CONFIG")"
    echo "[log]    $ARMLOG"

    bash "$REPO_DIR/models/violet_visitor/run.sh" \
        --evaluate --run_type calibration --saved --report > "$ARMLOG" 2>&1
    rc=$?

    echo "=== ARM '$arm' END: $(date) | exit=$rc ==="
    extract_crps "$arm" "$ARMLOG" "$rc"
done

echo ""
echo "================================================================"
echo " ALL ARMS DONE: $(date)"
echo " consolidated results -> $RESULTS"
echo "================================================================"
cat "$RESULTS"
# trap restores freeze_h="hl" on exit

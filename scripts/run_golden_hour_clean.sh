#!/usr/bin/env bash
# Clean golden_hour run: retrain each constituent FROM SCRATCH (no --saved),
# then evaluate, then evaluate the ensemble. This is the post-C-111 run —
# the MultiTaskLoss balancer is now active during training (views-hydranet
# editable install on `development`, fix lines present in training_engine.make()).
#
# Run with: nohup bash scripts/run_golden_hour_clean.sh &
# (self-logs to logs/golden_hour_clean.log via tee)

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO_DIR/logs/golden_hour_clean.log"

mkdir -p "$REPO_DIR/logs"
exec > >(tee -a "$LOG") 2>&1

echo "=== golden_hour CLEAN pipeline (retrain + evaluate + ensemble) — post-C-111 ==="
echo "Started: $(date)"
echo ""

# Retrain from scratch then evaluate each constituent (ONE model on GPU at a time)
for model in pink_pirate violet_visitor blue_stranger; do
    echo "=== Training $model (clean, no --saved) ==="
    echo "Start: $(date)"
    bash "$REPO_DIR/models/$model/run.sh" --train --run_type calibration
    echo "$model training completed: $(date)"
    echo ""

    echo "=== Evaluating $model ==="
    echo "Start: $(date)"
    bash "$REPO_DIR/models/$model/run.sh" --evaluate --run_type calibration --saved --report
    echo "$model evaluation completed: $(date)"
    echo ""
done

echo "=== All three constituents retrained and evaluated ==="
echo ""

# Evaluate the ensemble (--saved to use the freshly-generated PF output)
echo "=== Evaluating golden_hour ensemble ==="
echo "Start: $(date)"
bash "$REPO_DIR/ensembles/golden_hour/run.sh" --evaluate --run_type calibration --saved --report
echo "golden_hour ensemble completed: $(date)"

echo ""
echo "=== ALL DONE ==="
echo "Finished: $(date)"

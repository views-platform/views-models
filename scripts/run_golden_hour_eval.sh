#!/usr/bin/env bash
# Full golden_hour pipeline: train + evaluate each constituent, then ensemble.
# Run with: nohup bash scripts/run_golden_hour_eval.sh > logs/golden_hour_eval.log 2>&1 &

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_DIR/logs"

echo "=== golden_hour full pipeline (train + evaluate + ensemble) ==="
echo "Started: $(date)"
echo ""

# Evaluate each constituent model (ONE model on GPU at a time)
# Training artifacts already exist — re-evaluating with --saved
for model in pink_pirate violet_visitor blue_stranger; do
    echo "=== Evaluating $model ==="
    echo "Start: $(date)"
    bash "$REPO_DIR/models/$model/run.sh" --evaluate --run_type calibration --saved --report
    echo "$model evaluation completed: $(date)"
    echo ""
done

echo "=== All three constituents trained and evaluated ==="
echo ""

# Evaluate the ensemble (--saved to use existing PF output from constituents)
echo "=== Evaluating golden_hour ensemble ==="
echo "Start: $(date)"
bash "$REPO_DIR/ensembles/golden_hour/run.sh" --evaluate --run_type calibration --saved --report
echo "golden_hour ensemble completed: $(date)"

echo ""
echo "=== ALL DONE ==="
echo "Finished: $(date)"

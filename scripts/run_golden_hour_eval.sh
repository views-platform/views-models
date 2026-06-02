#!/usr/bin/env bash
# Step 2-4: Evaluate all three golden_hour constituents, then the ensemble.
# Run with: nohup bash scripts/run_golden_hour_eval.sh &
# (the script tees its own output to logs/golden_hour_eval.log)

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO_DIR/logs/golden_hour_eval.log"

mkdir -p "$REPO_DIR/logs"
exec > >(tee -a "$LOG") 2>&1

echo "=== golden_hour evaluation pipeline ==="
echo "Started: $(date)"
echo ""

# Step 2: Evaluate each constituent model (one at a time — ONE model on GPU)
for model in pink_pirate violet_visitor blue_stranger; do
    echo "=== Evaluating $model ==="
    echo "Start: $(date)"
    bash "$REPO_DIR/models/$model/run.sh" --evaluate --run_type calibration --saved
    echo "$model completed: $(date)"
    echo ""
done

echo "=== All three constituents evaluated ==="
echo ""

# Step 4: Evaluate the ensemble (with --saved to use existing PF output)
echo "=== Evaluating golden_hour ensemble ==="
echo "Start: $(date)"
bash "$REPO_DIR/ensembles/golden_hour/run.sh" --evaluate --run_type calibration --saved
echo "golden_hour ensemble completed: $(date)"

echo ""
echo "=== ALL DONE ==="
echo "Finished: $(date)"

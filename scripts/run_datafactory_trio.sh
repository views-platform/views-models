#!/bin/bash
# Sequential calibration runner for datafactory parity trio + ensemble
# Waits for bright_starship if running, then runs bold_comet, blazing_meteor, stellar_horizon

set -e

REPO_DIR="/home/simon/Documents/scripts/views_platform/views-models"
LOG_FILE="$REPO_DIR/scripts/datafactory_trio_run.log"
CONDA_ENV="views-hydranet-env"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Datafactory Trio Calibration Runner ==="

# Step 1: Wait for any existing bright_starship calibration to finish
EXISTING_PID=$(pgrep -f "bright_starship/main.py.*calibration" 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    log "Waiting for existing bright_starship calibration (PID $EXISTING_PID) to finish..."
    while kill -0 $EXISTING_PID 2>/dev/null; do
        sleep 60
    done
    log "bright_starship calibration finished."
else
    log "No existing bright_starship calibration running. Running it now..."
    cd "$REPO_DIR/models/bright_starship"
    conda run -n $CONDA_ENV python main.py -r calibration -t -e 2>&1 | tee -a "$LOG_FILE"
    log "bright_starship calibration complete."
fi

# Step 2: bold_comet calibration
log "Starting bold_comet calibration..."
cd "$REPO_DIR/models/bold_comet"
conda run -n $CONDA_ENV python main.py -r calibration -t -e 2>&1 | tee -a "$LOG_FILE"
log "bold_comet calibration complete."

# Step 3: blazing_meteor calibration
log "Starting blazing_meteor calibration..."
cd "$REPO_DIR/models/blazing_meteor"
conda run -n $CONDA_ENV python main.py -r calibration -t -e 2>&1 | tee -a "$LOG_FILE"
log "blazing_meteor calibration complete."

# Step 4: stellar_horizon ensemble (NO -t flag — use saved models)
log "Starting stellar_horizon ensemble (using saved models)..."
cd "$REPO_DIR/ensembles/stellar_horizon"
conda run -n $CONDA_ENV python main.py -r calibration -e --saved 2>&1 | tee -a "$LOG_FILE"
log "stellar_horizon ensemble complete."

log "=== All datafactory trio calibrations finished ==="

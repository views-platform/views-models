#!/bin/bash
#
# Integration test runner for views-models
#
# Trains and evaluates each model on calibration and validation partitions.
# Logs results per model — never aborts on failure.
#
# Usage:
#   bash run_integration_tests.sh                          # all models, calibration + validation
#   bash run_integration_tests.sh --models "counting_stars purple_alien"  # subset
#   bash run_integration_tests.sh --partitions "calibration"              # one partition only
#   bash run_integration_tests.sh --timeout 3600           # 60-minute timeout per run
#

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────

TIMEOUT=1800  # 30 minutes per model+partition run
PARTITIONS="calibration validation"
FILTER_MODELS=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs/integration_test_$TIMESTAMP"

# ── Colors ────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

# ── Parse arguments ───────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            FILTER_MODELS="$2"
            shift 2
            ;;
        --partitions)
            PARTITIONS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash run_integration_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --models \"model1 model2\"    Run only these models (space-separated)"
            echo "  --partitions \"cal val\"      Partitions to test (default: calibration validation)"
            echo "  --timeout SECONDS           Timeout per run (default: 1800)"
            echo "  --help                      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Discover models ──────────────────────────────────────────────────

if [ -n "$FILTER_MODELS" ]; then
    MODELS=()
    for m in $FILTER_MODELS; do
        if [ -f "$MODELS_DIR/$m/run.sh" ]; then
            MODELS+=("$m")
        else
            echo -e "${YELLOW}WARNING: Model '$m' not found or has no run.sh — skipping${NC}"
        fi
    done
else
    MODELS=()
    for dir in "$MODELS_DIR"/*/; do
        model_name=$(basename "$dir")
        if [ -f "$dir/run.sh" ]; then
            MODELS+=("$model_name")
        fi
    done
    IFS=$'\n' MODELS=($(sort <<<"${MODELS[*]}")); unset IFS
fi

TOTAL_MODELS=${#MODELS[@]}
if [ "$TOTAL_MODELS" -eq 0 ]; then
    echo "No models found to test."
    exit 1
fi

# ── Create log directories ───────────────────────────────────────────

for partition in $PARTITIONS; do
    mkdir -p "$LOG_DIR/$partition"
done

# ── Header ───────────────────────────────────────────────────────────

echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  views-models integration test${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo "  Models:     $TOTAL_MODELS"
echo "  Partitions: $PARTITIONS"
echo "  Timeout:    ${TIMEOUT}s per run"
echo "  Logs:       $LOG_DIR"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""

# ── Run models ───────────────────────────────────────────────────────

declare -A RESULTS
PASS_COUNT=0
FAIL_COUNT=0
TIMEOUT_COUNT=0
RUN_INDEX=0
TOTAL_RUNS=$(( TOTAL_MODELS * $(echo $PARTITIONS | wc -w) ))

for model in "${MODELS[@]}"; do
    for partition in $PARTITIONS; do
        RUN_INDEX=$((RUN_INDEX + 1))
        model_log="$LOG_DIR/$partition/${model}.log"
        result_key="${model}__${partition}"

        echo -ne "[${RUN_INDEX}/${TOTAL_RUNS}] ${BOLD}${model}${NC} (${partition})... "

        # Record start time
        start_time=$(date +%s)

        # Run the model via its own run.sh, capturing all output
        # The run.sh handles conda env activation internally
        timeout "$TIMEOUT" bash -c "
            cd '$MODELS_DIR/$model' && \
            bash run.sh -r '$partition' -t -e
        " > "$model_log" 2>&1
        exit_code=$?

        # Record duration
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        duration_str="${duration}s"

        if [ "$exit_code" -eq 0 ]; then
            echo -e "${GREEN}PASS${NC} (${duration_str})"
            RESULTS[$result_key]="PASS"
            PASS_COUNT=$((PASS_COUNT + 1))
        elif [ "$exit_code" -eq 124 ]; then
            echo -e "${RED}TIMEOUT${NC} (>${TIMEOUT}s)"
            RESULTS[$result_key]="TIMEOUT"
            TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
            echo "=== TIMEOUT after ${TIMEOUT}s ===" >> "$model_log"
        else
            echo -e "${RED}FAIL${NC} (exit ${exit_code}, ${duration_str})"
            RESULTS[$result_key]="FAIL(${exit_code})"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
done

# ── Summary ──────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Summary${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Print table header
printf "%-30s" "Model"
for partition in $PARTITIONS; do
    printf "%-15s" "$partition"
done
echo ""
printf "%-30s" "-----"
for partition in $PARTITIONS; do
    printf "%-15s" "----------"
done
echo ""

# Print results per model
for model in "${MODELS[@]}"; do
    printf "%-30s" "$model"
    for partition in $PARTITIONS; do
        result_key="${model}__${partition}"
        result="${RESULTS[$result_key]:-SKIPPED}"
        if [ "$result" = "PASS" ]; then
            printf "${GREEN}%-15s${NC}" "$result"
        else
            printf "${RED}%-15s${NC}" "$result"
        fi
    done
    echo ""
done

echo ""
echo -e "  ${GREEN}Passed:${NC}   $PASS_COUNT"
echo -e "  ${RED}Failed:${NC}   $FAIL_COUNT"
if [ "$TIMEOUT_COUNT" -gt 0 ]; then
    echo -e "  ${RED}Timeout:${NC}  $TIMEOUT_COUNT"
fi
echo "  Total:    $TOTAL_RUNS"
echo ""

# ── Write summary log ────────────────────────────────────────────────

{
    echo "Integration Test Summary — $TIMESTAMP"
    echo "Models: $TOTAL_MODELS | Partitions: $PARTITIONS | Timeout: ${TIMEOUT}s"
    echo ""
    printf "%-30s" "Model"
    for partition in $PARTITIONS; do
        printf "%-15s" "$partition"
    done
    echo ""
    printf "%-30s" "-----"
    for partition in $PARTITIONS; do
        printf "%-15s" "----------"
    done
    echo ""
    for model in "${MODELS[@]}"; do
        printf "%-30s" "$model"
        for partition in $PARTITIONS; do
            result_key="${model}__${partition}"
            result="${RESULTS[$result_key]:-SKIPPED}"
            printf "%-15s" "$result"
        done
        echo ""
    done
    echo ""
    echo "Passed: $PASS_COUNT | Failed: $FAIL_COUNT | Timeout: $TIMEOUT_COUNT | Total: $TOTAL_RUNS"
} > "$LOG_DIR/summary.log"

echo "Full summary: $LOG_DIR/summary.log"
echo "Per-model logs: $LOG_DIR/{partition}/{model}.log"

# Exit with non-zero if any failures
if [ "$FAIL_COUNT" -gt 0 ] || [ "$TIMEOUT_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0

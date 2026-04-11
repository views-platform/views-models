#!/bin/bash
#
# Integration test runner for views-models
#
# Trains and evaluates each model on calibration and validation partitions
# using a single conda environment. Logs results per model — never aborts on failure.
#
# Full documentation: docs/run_integration_tests.md
#
# Quick examples:
#   bash run_integration_tests.sh                                         # all models
#   bash run_integration_tests.sh --models "counting_stars bad_blood"     # subset
#   bash run_integration_tests.sh --partitions "calibration"              # one partition
#   bash run_integration_tests.sh --level cm                              # only CM models
#   bash run_integration_tests.sh --level pgm                             # only PGM models
#   bash run_integration_tests.sh --library baseline                        # one library
#   bash run_integration_tests.sh --exclude "purple_alien novel_heuristics"  # skip models
#   bash run_integration_tests.sh --env my_conda_env                     # different env
#   bash run_integration_tests.sh --timeout 3600                         # 60-min timeout
#

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────

CONDA_ENV="views_pipeline"
TIMEOUT=1800
PARTITIONS="calibration validation"
FILTER_MODELS=""
FILTER_LEVEL=""
FILTER_LIBRARY=""
EXCLUDE_MODELS="purple_alien"
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
        --models)     FILTER_MODELS="$2"; shift 2 ;;
        --level)      FILTER_LEVEL="$2"; shift 2 ;;
        --library)    FILTER_LIBRARY="$2"; shift 2 ;;
        --exclude)    EXCLUDE_MODELS="$2"; shift 2 ;;
        --partitions) PARTITIONS="$2"; shift 2 ;;
        --timeout)    TIMEOUT="$2"; shift 2 ;;
        --env)        CONDA_ENV="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash run_integration_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env NAME                  Conda env to activate (default: views_pipeline)"
            echo "  --models \"m1 m2\"            Run only these models"
            echo "  --level cm|pgm              Run only models at this level of analysis"
            echo "  --library NAME              Run only models using this library (baseline|stepshifter|r2darts2|hydranet)"
            echo "  --exclude \"m1 m2\"           Skip these models (default: purple_alien)"
            echo "  --partitions \"cal val\"      Partitions to test (default: calibration validation)"
            echo "  --timeout SECONDS           Timeout per run (default: 1800)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Build exclusion set ──────────────────────────────────────────────

declare -A EXCLUDED
for m in $EXCLUDE_MODELS; do
    EXCLUDED[$m]=1
done

# ── Discover models ──────────────────────────────────────────────────

MODELS=()
if [ -n "$FILTER_MODELS" ]; then
    for m in $FILTER_MODELS; do
        if [[ -n "${EXCLUDED[$m]:-}" ]]; then
            echo -e "${YELLOW}Excluding: $m${NC}"
        elif [ -f "$MODELS_DIR/$m/main.py" ]; then
            MODELS+=("$m")
        else
            echo -e "${YELLOW}WARNING: '$m' not found — skipping${NC}"
        fi
    done
else
    for dir in "$MODELS_DIR"/*/; do
        model_name=$(basename "$dir")
        if [[ -n "${EXCLUDED[$model_name]:-}" ]]; then
            continue
        fi
        if [ -f "$dir/main.py" ]; then
            MODELS+=("$model_name")
        fi
    done
    IFS=$'\n' MODELS=($(sort <<<"${MODELS[*]}")); unset IFS
fi

# ── Filter by level (cm/pgm) if requested ────────────────────────────
#
# Classification failures (broken config_meta.py) used to be silently swallowed
# by `2>/dev/null`, hiding broken models from --level testing. Now we capture
# stderr, surface the error, and fail fast before running any models.

if [ -n "$FILTER_LEVEL" ]; then
    FILTERED=()
    CLASSIFICATION_ERRORS=()
    for model in "${MODELS[@]}"; do
        cls_stderr_file=$(mktemp)
        level=$(python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location('m', '$MODELS_DIR/$model/configs/config_meta.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.get_meta_config().get('level', ''))
" 2>"$cls_stderr_file")
        cls_exit=$?
        cls_stderr=$(cat "$cls_stderr_file")
        rm -f "$cls_stderr_file"

        if [ "$cls_exit" -ne 0 ]; then
            echo -e "${RED}ERROR${NC} classifying ${BOLD}${model}${NC}: config_meta.py failed to load" >&2
            last_err_line=$(echo "$cls_stderr" | grep -v '^$' | tail -1)
            [ -n "$last_err_line" ] && echo "  $last_err_line" >&2
            CLASSIFICATION_ERRORS+=("$model")
            continue
        fi

        if [ "$level" = "$FILTER_LEVEL" ]; then
            FILTERED+=("$model")
        fi
    done

    if [ "${#CLASSIFICATION_ERRORS[@]}" -gt 0 ]; then
        echo "" >&2
        echo -e "${RED}${BOLD}Aborting:${NC} ${#CLASSIFICATION_ERRORS[@]} model(s) could not be classified by --level filter:" >&2
        for m in "${CLASSIFICATION_ERRORS[@]}"; do
            echo "  - $m" >&2
        done
        echo "Fix the broken config_meta.py file(s) and re-run." >&2
        exit 2
    fi

    MODELS=("${FILTERED[@]}")
fi

# ── Filter by library (baseline/stepshifter/r2darts2/hydranet) ──────

if [ -n "$FILTER_LIBRARY" ]; then
    FILTERED=()
    for model in "${MODELS[@]}"; do
        req_file="$MODELS_DIR/$model/requirements.txt"
        if [ -f "$req_file" ] && grep -q "views-${FILTER_LIBRARY}" "$req_file"; then
            FILTERED+=("$model")
        fi
    done
    MODELS=("${FILTERED[@]}")
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
echo "  Conda env:  $CONDA_ENV"
echo "  Models:     $TOTAL_MODELS"
[ -n "$FILTER_LEVEL" ] && echo "  Level:      $FILTER_LEVEL"
[ -n "$FILTER_LIBRARY" ] && echo "  Library:    $FILTER_LIBRARY"
echo "  Excluded:   $EXCLUDE_MODELS"
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

        start_time=$(date +%s)

        timeout "$TIMEOUT" bash -c "
            eval \"\$(conda shell.bash hook)\"
            conda activate '$CONDA_ENV'
            cd '$MODELS_DIR/$model'
            python main.py -r '$partition' -t -e
        " > "$model_log" 2>&1
        exit_code=$?

        end_time=$(date +%s)
        duration=$((end_time - start_time))

        if [ "$exit_code" -eq 0 ]; then
            echo -e "${GREEN}PASS${NC} (${duration}s)"
            RESULTS[$result_key]="PASS"
            PASS_COUNT=$((PASS_COUNT + 1))
        elif [ "$exit_code" -eq 124 ]; then
            echo -e "${RED}TIMEOUT${NC} (>${TIMEOUT}s)"
            RESULTS[$result_key]="TIMEOUT"
            TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
            echo "=== TIMEOUT after ${TIMEOUT}s ===" >> "$model_log"
        else
            echo -e "${RED}FAIL${NC} (exit ${exit_code}, ${duration}s)"
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

printf "%-30s" "Model"
for partition in $PARTITIONS; do printf "%-15s" "$partition"; done
echo ""
printf "%-30s" "-----"
for partition in $PARTITIONS; do printf "%-15s" "----------"; done
echo ""

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
[ "$TIMEOUT_COUNT" -gt 0 ] && echo -e "  ${RED}Timeout:${NC}  $TIMEOUT_COUNT"
echo "  Total:    $TOTAL_RUNS"
echo ""

# ── Write summary log ────────────────────────────────────────────────

{
    echo "Integration Test Summary — $TIMESTAMP"
    echo "Env: $CONDA_ENV | Models: $TOTAL_MODELS | Excluded: $EXCLUDE_MODELS"
    echo "Partitions: $PARTITIONS | Timeout: ${TIMEOUT}s"
    echo ""
    printf "%-30s" "Model"
    for partition in $PARTITIONS; do printf "%-15s" "$partition"; done
    echo ""
    printf "%-30s" "-----"
    for partition in $PARTITIONS; do printf "%-15s" "----------"; done
    echo ""
    for model in "${MODELS[@]}"; do
        printf "%-30s" "$model"
        for partition in $PARTITIONS; do
            result="${RESULTS[${model}__${partition}]:-SKIPPED}"
            printf "%-15s" "$result"
        done
        echo ""
    done
    echo ""
    echo "Passed: $PASS_COUNT | Failed: $FAIL_COUNT | Timeout: $TIMEOUT_COUNT | Total: $TOTAL_RUNS"
} > "$LOG_DIR/summary.log"

echo "Full summary: $LOG_DIR/summary.log"
echo "Per-model logs: $LOG_DIR/{partition}/{model}.log"

if [ "$FAIL_COUNT" -gt 0 ] || [ "$TIMEOUT_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0

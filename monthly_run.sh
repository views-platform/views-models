#!/usr/bin/env bash

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

unset ZSH
unset ZSH_VERSION
unset ZSH_NAME
export SHELL=/bin/bash

run_folder () {
    local folder="$1"
    local abs_path="$BASE_DIR/$folder"

    echo "====================================="
    echo "Running: $folder"
    echo "====================================="

    (
        cd "$abs_path"
        if [[ "$folder" == ensembles/* ]]; then
            bash run.sh -m
        else
            bash  run.sh
        fi
    )

    echo "Finished: $folder"
    echo ""
}

# ── preflight: is the write path still open? ──────────────────────────────────────────
# Both Appwrite keys expire around 2026-11-30, and on pipeline-core's write path that
# expiry is SILENT: the log reads "Forecasts uploaded successfully" while nothing is
# uploaded (C-99). Everything below is hours of compute whose only product is an upload,
# so the question is asked first, once, before any of it. (#302)
#
# INTERPRETER. platform_env_load parses the coordinate registry with tomllib (3.11+), and
# this script is normally started from base, which may be older. Two candidates, in a
# fixed order, and the one chosen is printed: a preflight that quietly runs under a
# different interpreter than you assume is worse than no preflight (C-113).
preflight_python () {
    if python -c 'import tomllib' >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    local ensemble_python="$BASE_DIR/envs/views_ensemble/bin/python"
    if [ -x "$ensemble_python" ] && "$ensemble_python" -c 'import tomllib' >/dev/null 2>&1; then
        echo "$ensemble_python"
        return 0
    fi
    return 1
}

echo "====================================="
echo "Preflight: Appwrite write path"
echo "====================================="

if ! chosen_python="$(preflight_python)"; then
    echo "PREFLIGHT FAILED: nothing here can read the coordinate registry (needs Python 3.11+)." >&2
    echo "  tried: 'python' ($(python -V 2>&1))" >&2
    echo "         $BASE_DIR/envs/views_ensemble/bin/python" >&2
    echo "  Fix:   conda activate an environment with Python 3.11+, then re-run." >&2
    exit 1
fi
echo "interpreter: $chosen_python"

# The subshell confines the PATH change; platform_env.sh:43 states it uses whatever
# 'python' the caller has arranged, so this is the sanctioned way to choose one. Failures
# are absorbed here on purpose — the verdict below is what decides, and a run that
# produced no verdict line at all must not pass.
preflight_out="$(mktemp)"
(
    cd "$BASE_DIR"
    PATH="$(dirname "$chosen_python"):$PATH"
    # shellcheck source=tools/credentials/platform_env.sh
    . "$BASE_DIR/tools/credentials/platform_env.sh"
    platform_env_load
    python -m tools.liveness.appwrite_store
) > "$preflight_out" 2>&1 || true
cat "$preflight_out"
preflight_verdict="$(grep -m1 '^verdict: ' "$preflight_out" | cut -d' ' -f2- || true)"
rm -f "$preflight_out"

# An ALLOW-list, not the exit code, and not a deny-list. STORE_IDLE is exit 1 but means
# "nothing has landed lately" — which is the very thing this run exists to change, so it
# must not abort. Everything else stops, including a verdict this script has never heard
# of and the empty string: a gate that cannot tell "passed" from "did not run" is not a
# gate (C-113).
case "$preflight_verdict" in
    STORE_ACTIVE|STORE_IDLE)
        echo "Preflight passed (verdict: $preflight_verdict) — the key is accepted and the store answers."
        echo ""
        ;;
    *)
        echo "" >&2
        echo "PREFLIGHT FAILED (verdict: ${preflight_verdict:-<none produced>})." >&2
        echo "  The Appwrite write path did not answer as a working one. The facts above say why." >&2
        echo "  Nothing was run. Uploads from this run would have been silently discarded." >&2
        echo "  Both keys expire around 2026-11-30 — if that is the cause, rotate before re-running." >&2
        exit 1
        ;;
esac

run_folder "ensembles/pink_ponyclub"
run_folder "ensembles/skinny_love"
run_folder "ensembles/rude_boy"
run_folder "ensembles/first_love"
run_folder "postprocessors/un_fao"

echo "All monthly runs completed."
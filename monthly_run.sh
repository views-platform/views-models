#!/usr/bin/env bash

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

unset ZSH
unset ZSH_VERSION
unset ZSH_NAME
export SHELL=/bin/bash

# ── provenance: which package versions produced this forecast? ────────────────────────
# Your code is in git. The ~200 packages that ran alongside it are not: they live in
# `envs/`, which is gitignored and exists only on whichever laptop ran the month. So a
# delivered FAO forecast has been half-reproducible — the config side of the same gap is
# already registered as C-110, this is the dependency side (C-117).
#
# One `pip freeze` per environment, written into a TRACKED directory. `logs/` would not
# do: it is gitignored, so a snapshot there would be exactly as ephemeral as the thing it
# describes. Commit these with the run.
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
SNAPSHOT_DIR="$BASE_DIR/reports/env_snapshots"
CAPTURED_ENVS=""

capture_env_snapshot () {
    local folder="$1"
    local run_sh="$BASE_DIR/$folder/run.sh"
    local env_name env_python out

    # The environment is named in the launcher, not here — read it rather than restate
    # it, so this cannot drift from what actually ran.
    env_name="$(sed -n 's|^env_path="$project_path/envs/\(.*\)"$|\1|p' "$run_sh" | head -1)"
    if [ -z "$env_name" ]; then
        echo "   NOTE  no env_path in $folder/run.sh — no snapshot taken" >&2
        return 0
    fi

    # Four ensembles share envs/views_ensemble; capture each environment once per run.
    case " $CAPTURED_ENVS " in
        *" $env_name "*) return 0 ;;
    esac

    env_python="$BASE_DIR/envs/$env_name/bin/python"
    out="$SNAPSHOT_DIR/${RUN_ID}__${env_name}.txt"

    if [ ! -x "$env_python" ]; then
        echo "   NOTE  envs/$env_name has no python — no snapshot taken" >&2
        return 0
    fi

    mkdir -p "$SNAPSHOT_DIR"
    {
        echo "# environment snapshot — reports/env_snapshots"
        echo "# run_id:      $RUN_ID"
        echo "# environment: envs/$env_name"
        echo "# first used by: $folder"
        echo "# commit:      $(git -C "$BASE_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "# python:      $("$env_python" -V 2>&1)"
        echo "#"
        echo "# The commit above plus the packages below are jointly what produced this"
        echo "# month's forecasts. Neither half is sufficient on its own."
    } > "$out"

    # NON-FATAL on purpose, and this is a different case from the preflight below. A
    # failed preflight means the run itself is doomed, so it aborts. A failed snapshot
    # means the run is fine and only the record is missing — killing a forecast run to
    # protect a log file would be the worse trade. It is loud so it cannot pass unnoticed.
    if "$env_python" -m pip freeze >> "$out" 2>/dev/null; then
        echo "   snapshot: reports/env_snapshots/$(basename "$out") ($(grep -c '==' "$out") packages)"
        CAPTURED_ENVS="$CAPTURED_ENVS $env_name"
    else
        echo "   WARNING  pip freeze failed for envs/$env_name — this run has NO dependency record" >&2
        rm -f "$out"
    fi
}

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

    # AFTER the run, not before: run.sh may install into the environment, and the
    # snapshot must describe what was actually used, not what was there beforehand.
    capture_env_snapshot "$folder"

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
#!/usr/bin/env bash
# pod_run_fao_delivery.sh [--rehearsal <lessons>] [--preflight]
#
# ┌──────────────────────────────────────────────────────────────────────────────────────┐
# │  VERSION 0.1.0 — PROVISIONAL.  First script for the FAO leg; no CI runs it.           │
# │  It PUBLISHES to a store a partner consumes. Read --preflight output before trusting  │
# │  it with GPU hours, and read the REHEARSAL note below before trusting it with FAO.     │
# └──────────────────────────────────────────────────────────────────────────────────────┘
#
# Track B of views-models#499: the eight HydraNets forecast on global land, `rusty_bucket`
# pools them, the pooled forecast is published to the shelf, and the `un_fao` postprocessor
# curates `land` -> `land_gaul` and hands over to `unfao_bucket`.
#
# WHY THIS EXISTS. On 2026-09-29 every one of these steps was typed by hand on a rented pod.
# It worked, the pod was destroyed, and nothing recorded what had been done — so the next
# delivery started from the original broken state. That is the same defect a /falsify audit
# found three instances of in `pod_run_model.sh` (#516, #517, #523), and this is the fourth:
# the procedure existed only as commands in a terminal. `fimbulthul` has been unreachable
# since 2026-09-25, so a pod is not a stopgap here — it is the only hardware.
#
# THE ORDER MATTERS AND IS NOT OBVIOUS:
#   1. the 8 models      -r forecasting -t -f    (each ~2h at 300 lessons)
#   2. rusty_bucket      -r forecasting -f -sa   pooled from the SAVED member forecasts
#   3. publish           -p                       wire shards to production_forecasts
#   4. un_fao            postprocessors/un_fao/run.sh
# `-sa/--saved` at step 2 is required: without it the ensemble refetches instead of pooling
# what step 1 just produced.
#
# REHEARSAL. `--rehearsal <lessons>` patches only the pod's clone (never a tracked config),
# and every artefact it produces is MARKED unfit to deliver. But marking is all this script
# can do: nothing downstream REFUSES a marked rehearsal, because the refusal belongs in
# views-pipeline-core and is an ADR-013 contract change (views-models#523). A rehearsal
# therefore reaches the FAO shelf exactly like a real run. That is deliberate — it is the
# only way to test the chain — and it is why step 4 prints what it published by name.
#
# Progress:  cat /workspace/deliver/_fao/STAGE   ·   tail -f /workspace/deliver/_fao/run.log

set -uo pipefail

USAGE='usage: pod_run_fao_delivery.sh [--rehearsal <lessons>] [--preflight]'
REHEARSAL_LESSONS=""
PREFLIGHT_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --rehearsal)
            [ $# -ge 2 ] || { echo "!!! --rehearsal needs a lesson count, e.g. --rehearsal 40" >&2
                              echo "!!! $USAGE" >&2; exit 1; }
            REHEARSAL_LESSONS="$2"
            case "$REHEARSAL_LESSONS" in
                ''|*[!0-9]*) echo "!!! --rehearsal needs a positive integer, got: $REHEARSAL_LESSONS" >&2
                             exit 1 ;;
            esac
            [ "$REHEARSAL_LESSONS" -ge 1 ] || { echo "!!! --rehearsal 0 has nothing to rehearse" >&2; exit 1; }
            shift 2 ;;
        --preflight) PREFLIGHT_ONLY=1; shift ;;
        --*) echo "!!! unknown option: $1" >&2; echo "!!! $USAGE" >&2; exit 1 ;;
        *)   echo "!!! unexpected argument: $1" >&2; echo "!!! $USAGE" >&2; exit 1 ;;
    esac
done

# PODRUN_ROOT exists so this script can be exercised off a pod — the tests run its parser and
# its preflight, and a hardcoded /workspace makes both untestable. It relocates the workspace
# and nothing else: every substantive check below (credentials, conda, region, GPU, disk) is
# unaffected by it, so an override cannot turn a refusal into a pass.
ROOT="${PODRUN_ROOT:-/workspace}"
REPO=$ROOT/views-models
VENV=$ROOT/venv
OUT=$ROOT/deliver/_fao
LOG=$OUT/run.log
ENSEMBLE=rusty_bucket
MODELS="violet_visitor bold_comet blazing_meteor blue_stranger heavy_freighter pink_pirate bright_starship purple_alien"

# Checked, because an unchecked mkdir here fails off a pod and the script then runs on into
# a broken tee and a lock check that reports "another delivery is in progress" — a misleading
# message for a machine that simply has no /workspace. Found by the tests for this file.
mkdir -p "$OUT" 2>/dev/null || {
    echo "!!! cannot create $OUT" >&2
    echo "!!! This script runs ON A POD, where $ROOT exists. To exercise it elsewhere, set" >&2
    echo "!!!   PODRUN_ROOT=/some/writable/dir" >&2
    exit 1
}
rm -f "$OUT/STATUS" "$OUT/REHEARSAL" "$OUT/PUBLISHED"
exec > >(tee -a "$LOG") 2>&1

stage() { echo "$1" > "$OUT/STAGE"; echo "=== [$(date +%H:%M:%S)] $1 ==="; }
die()   {
    echo "FAILED:$(cat "$OUT/STAGE" 2>/dev/null)" > "$OUT/STATUS"
    printf '%s\n' "$1" > "$OUT/FAILURE"
    echo "!!! $1"
    exit 1
}

if ! mkdir "$OUT/.lock" 2>/dev/null; then
    # Distinguished deliberately: "exists" and "could not be created" need different actions,
    # and reporting the first for the second sends the operator looking for a run that never
    # started.
    if [ -d "$OUT/.lock" ]; then
        echo "!!! $OUT/.lock exists — another FAO delivery is in progress on this pod."
        echo "!!! If you are certain it is dead: rmdir $OUT/.lock"
    else
        echo "!!! could not create $OUT/.lock — $OUT is not writable."
    fi
    exit 1
fi
trap 'rmdir "$OUT/.lock" 2>/dev/null' EXIT

echo "### FAO delivery — started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ -n "$REHEARSAL_LESSONS" ] && echo "### REHEARSAL at $REHEARSAL_LESSONS lessons — output will be marked unfit to deliver"

# ── 0. preflight ──────────────────────────────────────────────────────────────────────
# Everything knowable before spending money, and the whole reason --preflight exists as a
# mode you can run on a fresh pod for seconds. At 300 lessons step 1 alone is ~16 GPU hours;
# discovering a missing credential or a missing conda after that is the failure this block is
# written to make impossible. Each check names what to do, not just what is wrong.
stage preflight
FAIL=0
note() { echo "  MISSING: $*"; FAIL=1; }

[ -d "$REPO/.git" ] || note "$REPO is not a checkout — clone views-models first (see docs/runpod_run_guide.md Phase 2)"
[ -x "$VENV/bin/python" ] || note "$VENV does not exist — run pod_run_model.sh once to build it, or follow Phase 2"

# The datafactory credential, for the fetch.
[ -s /root/.netrc ] || note "/root/.netrc — the datafactory fetch needs it (guide Step 2.3)"
if [ -s /root/.netrc ] && [ "$(stat -c %a /root/.netrc 2>/dev/null)" != "600" ]; then
    chmod 600 /root/.netrc
    echo "  fixed: /root/.netrc was not 600"
fi

# The publish credentials. Three of the nine are real secrets; the rest are identifiers.
# Checked HERE and not at first publish, because views-pipeline-core's own
# PredictionStoreConfig claims to read these "once at startup and fail loud" and does not —
# it is called from _build_datastore, i.e. after training (views-pipeline-core#557). Until
# that moves, this is the only check that happens before the money is spent.
for v in APPWRITE_ENDPOINT APPWRITE_DATASTORE_PROJECT_ID APPWRITE_DATASTORE_API_KEY; do
    eval "val=\${$v:-}"
    [ -n "$val" ] || note "\$$v — a publish secret; see reports/fao_delivery_runbook.md"
done
case "${APPWRITE_DATASTORE_API_KEY:-}" in
    *[![:print:]]*) note "\$APPWRITE_DATASTORE_API_KEY contains a non-printable character — re-paste it" ;;
esac

# conda, for the postprocessor leg ONLY. tools/launcher/postprocessor.sh uses
# `conda shell.bash hook` / `conda create --prefix` / `conda activate`, while this pod builds
# a uv venv — so the two legs need different interpreters and a pod can satisfy one and not
# the other. Without conda the delivery dies at step 4, after every GPU hour is spent.
command -v conda >/dev/null 2>&1 || note "conda — the un_fao postprocessor launcher requires it (tools/launcher/postprocessor.sh:72). A uv venv is not enough."

# The coordinate registry, which the FAO queryset derives its region from.
if [ -x "$VENV/bin/python" ] && [ -d "$REPO" ]; then
    REGION=$(cd "$REPO" && "$VENV/bin/python" -c \
        'import sys; sys.path.insert(0,"."); from postprocessors.un_fao.configs.config_queryset import REGION; print(REGION)' 2>/dev/null)
    if [ "$REGION" = "land_gaul" ]; then
        echo "  un_fao REGION resolves to land_gaul"
    else
        note "un_fao REGION resolved to '${REGION:-<error>}', expected land_gaul — the FAO wire is disarmed (C-110)"
    fi
fi

nvidia-smi -L >/dev/null 2>&1 || note "no GPU visible"
AVAIL_GB=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
# DISK_FLOOR_GB, and the honest state of what it is based on.
#
# pod_run_model.sh refuses below 40GB for ONE model ("one model needs ~20GB"), and this script
# delegates to it eight times — so 40 is a hard lower bound that will be re-checked at every
# model whatever is written here. A floor BELOW the floor of the thing it delegates to is a
# preflight that says "ready" and then refuses at model 3, which is the opposite of this
# script's purpose. The first version of this check said 60 for all eight, which was exactly
# that mistake: lower than the per-model transient for a run eight times the size.
#
# The retained component is an ESTIMATE and cannot be better than that yet: measured on this
# machine, one model's calibration output is ~2.5GB per predictions directory, and NO
# FORECASTING RUN HAS EVER COMPLETED ON THIS ROSTER, so the retained size of a forecast is
# unmeasured. A forecast has one origin against calibration's 13, so it should be smaller —
# "should be" is doing real work in that sentence.
#
# 40 transient + 8 x ~5GB retained, rounded up for the pooled ensemble, which is also
# unmeasured. Revise this number from the first completed run rather than reasoning about it
# again; that is the whole of the trigger.
DISK_FLOOR_GB=80
[ "${AVAIL_GB:-0}" -ge "$DISK_FLOOR_GB" ] || note "only ${AVAIL_GB:-0}GB free on $ROOT; eight forecasts plus the pool need >=${DISK_FLOOR_GB}GB (40GB is the per-model transient that pod_run_model.sh enforces on its own, eight times over, plus retained output)"

if [ "$FAIL" = "1" ]; then
    echo
    die "preflight failed — nothing above costs GPU time to fix. Fix them and re-run --preflight."
fi
echo "preflight OK — GPU visible, ${AVAIL_GB}GB free, credentials present, conda present, wire armed"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
    echo OK > "$OUT/STATUS"
    stage preflight_only_done
    echo "### --preflight only: nothing was run, nothing was published."
    echo "### Re-run without --preflight to start the delivery."
    exit 0
fi

# ── 1. the eight, forecasting ─────────────────────────────────────────────────────────
REH_ARGS=""
[ -n "$REHEARSAL_LESSONS" ] && REH_ARGS="--rehearsal $REHEARSAL_LESSONS"
for M in $MODELS; do
    stage "forecast:$M"
    # Delegated rather than reimplemented: pod_run_model.sh already carries the config
    # patch-and-verify, the appwrite check and the C-151 override, and a second copy of
    # those is a second place for them to rot.
    bash "$REPO/tools/podrun/pod_run_model.sh" $REH_ARGS --forecast "$M" \
        || die "forecast failed for $M — see /workspace/deliver/$M/FAILURE"
    [ "$(cat "$ROOT/deliver/$M/STATUS" 2>/dev/null)" = "OK" ] \
        || die "$M did not report OK — refusing to pool a partial roster"
done

# ── 2-3. pool and publish ─────────────────────────────────────────────────────────────
stage pool_and_publish
cd "$REPO/ensembles/$ENSEMBLE" || die "cannot enter $ENSEMBLE"
export WANDB_MODE=offline WANDB_SILENT=true
# -sa/--saved pools the member forecasts just written. Without it the ensemble refetches and
# the eight runs above are wasted. -p publishes the wire shards (ADR-013).
"$VENV/bin/python" main.py -r forecasting -f -sa -p || die "$ENSEMBLE pool-and-publish exited non-zero"

# ── 4. the FAO postprocessor ──────────────────────────────────────────────────────────
stage un_fao_postprocessor
cd "$REPO" || die "cannot enter repo"
# A refusal naming DeliveryNotFindableError is views-postprocessing 1.4.0 WORKING: that build
# verifies a delivery by what it refuses, and the message says which case it hit. Do not read
# it as this script failing.
bash postprocessors/un_fao/run.sh || die "the un_fao postprocessor exited non-zero — read the message before assuming the worst; DeliveryNotFindableError means 1.4.0's findability guard fired and it names every object it checked"

# ── 5. what actually landed ───────────────────────────────────────────────────────────
stage report
# By name, from the live surface, not inferred from an exit code. `python -m tools.liveness`
# is the six-surface dashboard; it answers "is it there?" rather than "did we send it?".
"$VENV/bin/python" -m tools.liveness > "$OUT/PUBLISHED" 2>&1 \
    || echo "(liveness reported non-zero — read $OUT/PUBLISHED)" >> "$OUT/PUBLISHED"
tail -40 "$OUT/PUBLISHED"

stage manifest
{
    echo "delivery:     un_fao via $ENSEMBLE"
    echo "finished:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "models:       $MODELS"
    echo "git:          $(git -C "$REPO" rev-parse --short HEAD)"
    if [ -n "$REHEARSAL_LESSONS" ]; then
        echo "mode:         REHEARSAL — NOT FIT TO DELIVER"
        echo "lessons:      $REHEARSAL_LESSONS (pod clones patched; tracked configs untouched)"
    else
        echo "mode:         production"
        echo "lessons:      as committed at the git sha above"
    fi
} > "$OUT/MANIFEST"
cat "$OUT/MANIFEST"

if [ -n "$REHEARSAL_LESSONS" ]; then
    {
        echo "THIS DELIVERY IS A REHEARSAL. THE FORECASTS ON THE FAO SHELF ARE UNDERTRAINED."
        echo
        echo "lessons:  $REHEARSAL_LESSONS"
        echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo
        echo "It ran the whole chain to prove the hops connect. Nothing downstream refuses a"
        echo "marked rehearsal (views-models#523), so these forecasts ARE on the shelf and the"
        echo "FAO API can serve them. They are structurally indistinguishable from real ones."
        echo "Supersede or remove them before anyone reads them as a forecast."
    } > "$OUT/REHEARSAL"
    echo
    echo "############################################################################"
    echo "### REHEARSAL COMPLETE — undertrained forecasts are NOW ON THE FAO SHELF."
    echo "### They are indistinguishable from real ones. Supersede them."
    echo "### $OUT/REHEARSAL"
    echo "############################################################################"
fi

echo OK > "$OUT/STATUS"
stage done
echo "### FAO delivery — COMPLETE"

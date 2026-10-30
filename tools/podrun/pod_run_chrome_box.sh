#!/usr/bin/env bash
# pod_run_chrome_box.sh [--preflight]
#
# PHASE 2 of the CRAF'd chain (epic #595, story #640): the eight `*_sparrow` pgm models and
# `chrome_box`, the ensemble that pools them and reconciles against `chrome_crate` draw by
# draw. Sibling of `pod_run_crafd_chain.sh`, which is phase 1.
#
# ⛔ RUN PHASE 1 FIRST, ON THIS FILESYSTEM. `chrome_box` declares
# `reconcile_with: "chrome_crate"` and reads that forecast from local disk through
# `EnsemblePathManager(cm_model)._get_generated_predictions_data_file_paths()`. Without it the
# run pools for minutes and then dies at reconcile — AFTER three hours of paid member
# training. The preflight below refuses that case in seconds instead.
#
# ## Provenance: this is not new code
#
# Both halves of this script already ran on a pod on 2026-10-28, from a provisional driver
# written to `/workspace/phase2_pgm.sh` and retained at
# `tools/podrun/evidence/2026-10-28_chrome_box/phase2_pgm.sh`, whose own header said it would
# be "ported into tools/podrun/ properly, with tests". This is that port.
#
#   the eight members     8/8 trained: 22, 21, 22, 22, 24, 25, 24, 24 min = 184 min (3.07 h)
#                         (`phase2.out` HEARTBEAT lines)
#   pool + reconcile      3 targets reconciled, mode=aligned-draws (2330712 grid rows,
#                         1024 draws), "Done. Runtime: 4.427 minutes" (`probe.out`)
#   measured peak         >= 68.2 GiB cgroup anon, 30s sampling through pool AND reconcile
#
# ⚠️ THEY HAVE NEVER SUCCEEDED IN ONE RUN, and that is what this script is for.
# `phase2_pgm.sh` trained all eight members and then died three times at reconcile on
# views-pipeline-core#562 — `load_cm_frame` looked for a `.parquet` file where a
# PredictionFrame ensemble writes a directory. `probe.out` then re-ran only the pool and
# reconcile, over those already-trained members, with a `sitecustomize.py` monkeypatching
# `load_cm_frame`. #562 shipped in 3.3.5 (published 2026-10-29), so the patch is DELETED
# rather than ported — and `verify_install` below asserts the installed version, because a
# venv that predates the release would replay the exact three-hour failure.
#
# ## Why it never calls run.sh
#
# Every step runs `$VENV/bin/python main.py` directly, and the ensemble runs `-f -sa`. With
# `-t` the manager calls `_train_model_artifact`, which shells out to `models/<m>/run.sh` and
# re-enters the conda lane, where C-151 and the per-model prefixes live (views-models#643,
# #644). `-sa` with the member artifacts already on disk makes the manager load them instead.
#
# ## It publishes nothing
#
# No publish credential, deliberately. This chain produces a reconciled forecast on disk and
# uploads nothing: the nine Appwrite publish variables are not merely unnecessary here —
# placing them on hardware we do not control is a cost with no benefit (RunPod guide rule 5,
# ADR-024). Publishing to the shelf happens afterwards, from the laptop. The seven-step path
# from here to the CRAF'd API is written in `pod_run_crafd_chain.sh`'s header.
#
# ## The deliverable
#
#   /workspace/deliver/chrome_box/parquet/   the pooled, reconciled pgm forecast
#   /workspace/deliver/chrome_box/MANIFEST   versions, timings, git sha, observed peak
#   /workspace/deliver/chrome_box/STATUS     OK or FAILED:<stage>
#   /workspace/deliver/chrome_box/FAILURE    the reason, on failure only
#   /workspace/deliver/chrome_box/memory.log the sampled cgroup series
#   /workspace/deliver/chrome_box/run.log    the whole transcript
#
# Progress is readable from outside at any time:  cat /workspace/deliver/chrome_box/STAGE
#
# ⛔ The member forecasts are NOT copied into deliver/, unlike phase 1 where they are 176 MB.
# Here they are 8 x 2,330,712 rows x 128 draws x 3 targets x 4 B = 26.7 GiB of duplication.
# They stay in `models/<m>/data/generated/`, which is where a re-run reads them from, and the
# pool is the product. If the pod itself is lost the members are lost with it; retrieving
# 26.7 GiB over the network to avoid retraining is its own worse problem.

set -uo pipefail

USAGE='usage: pod_run_chrome_box.sh [--preflight]'
# --preflight runs every check that costs nothing and then STOPS. It reports EVERY problem at
# once rather than one per attempt, which on rented hardware is the difference between one
# round trip and six.
PREFLIGHT_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --preflight) PREFLIGHT_ONLY=1; shift ;;
        --*) echo "!!! unknown option: $1" >&2; echo "!!! $USAGE" >&2; exit 1 ;;
        *)   echo "!!! this script takes no positional arguments; got: $*" >&2
             echo "!!! the chain is fixed — see MODELS/ENSEMBLE below" >&2
             echo "!!! $USAGE" >&2; exit 1 ;;
    esac
done

# Honours PODRUN_ROOT for the same reason the other scripts do, so all of them AGREE on where
# deliver/<name>/STATUS lives. On a pod it is /workspace and nothing changes.
ROOT="${PODRUN_ROOT:-/workspace}"
REPO=$ROOT/views-models
VENV=$ROOT/venv-crafd

ENSEMBLE=chrome_box
# The reconcile partner, named here because the preflight has to check for its output and
# because `chrome_box`'s own config is the authority: `reconcile_with: "chrome_crate"`.
PARTNER=chrome_crate
# The eight pgm members, in chrome_box's config_modelset.py order. Listed literally rather
# than parsed out of the config: this script must be readable on a pod with no python yet, and
# a chain that silently follows a reordered config is worse than one that disagrees loudly —
# tests/test_chrome_box_runner.py compares this list against the config and fails on drift.
MODELS="ashen_sparrow brisk_sparrow candid_sparrow dusky_sparrow eager_sparrow faint_sparrow gentle_sparrow humble_sparrow"

OUT=$ROOT/deliver/$ENSEMBLE
LOG=$OUT/run.log

# The pipeline-core floor, stated ONCE and asserted after the venv is found. This script
# INSTALLS NOTHING — it reuses the venv phase 1 built — so the floor here is not a constraint
# on a resolve, it is a claim about what is already on disk, and the only honest way to hold it
# is to read the installed version. Below 3.3.5, `load_cm_frame` cannot read a PredictionFrame
# cm forecast and the reconcile dies after the members are paid for
# (views-platform/views-pipeline-core#562, published 2026-10-29).
PC_FLOOR=3.3.5

# MEASURED 2026-10-28: >= 68.2 GiB cgroup anon through pool and reconcile, on a run that
# completed, against a 53.86 GiB frame-arithmetic floor. The authoritative figure, with its
# provenance and its lower-bound caveat, is in
# `ensembles/chrome_box/configs/config_hyperparameters.py`, pinned by
# `tests/test_chrome_box_memory_is_the_measured_peak.py`.
#
# 80 rather than 68: headroom is the property, not fit. The runpod postmortem's 31 GB machine
# did not OOM — it collapsed 25x, 32 steps/s to 0.11, from kernel reclaim, and ran for eight
# hours. 80 admits the 87 GiB class that was used and refuses the 57 GiB class, which at the
# measured peak is at 120%.
#
# ⛔ This is NOT phase 1's floor. `pod_run_crafd_chain.sh` sets 16, correctly, because
# chrome_crate is country-month: 96 MB of pool, ~2.0 MB peak RSS per member, 7 minutes. Using
# one phase's figure for the other is the mistake that produced views-models#648 and, in the
# other direction, told an operator to rent 87 GiB for a seven-minute job.
MEM_FLOOR_GB=80

# ⚠ DERIVED, not measured — the one floor in this script that is. From raw float32:
#
#   the pool            2,330,712 rows x 1024 draws x 3 targets x 4 B   = 26.7 GiB
#   the eight members   8 x 2,330,712 x 128 x 3 x 4                     = 26.7 GiB
#   the deliver/ copy of the pool                                       = 26.7 GiB
#   ----------------------------------------------------------------------------
#                                                                      ~80 GiB raw
#
# Parquet compresses, so 60 is the floor rather than 80. TRIGGER TO REPLACE IT: the first
# successful run writes `pool:` into the MANIFEST below — which `phase2_pgm.sh` never reached,
# because it died before the manifest stage. Use that figure and delete this arithmetic.
DISK_FLOOR_GB=60

mkdir -p "$OUT" || { echo "!!! cannot create $OUT" >&2; exit 1; }

# A run begins by clearing the previous run's verdict. Without this, STATUS from an earlier
# failed --preflight survives into the next invocation and a poller reads FAILED while the run
# is in fact progressing — observed on a pod 2026-10-28. STAGE is the live field; STATUS is a
# verdict and must not outlive its run (`feedback_status_fields_lie`).
rm -f "$OUT/STATUS" "$OUT/FAILURE"

stage() { echo "$1" > "$OUT/STAGE"; echo "=== [$(date +%H:%M:%S)] $1 ==="; }
die()   {
    echo "FAILED:$(cat "$OUT/STAGE" 2>/dev/null)" > "$OUT/STATUS"
    # Written directly, not through the tee subshell: on a hard kill (preemption, OOM) the
    # last buffered log lines can be lost, and that is exactly when the reason matters.
    printf '%s\n' "$1" > "$OUT/FAILURE"
    echo "!!! $1"
    exit 1
}

# ── preflight ─────────────────────────────────────────────────────────────────────────

stage preflight
FAIL=0
note() { echo "  MISSING: $*"; FAIL=1; }

[ -d "$REPO/.git" ] || note "$REPO is not a checkout — clone views-models first (docs/runpod_run_guide.md Phase 2)"
for M in $MODELS; do
    [ -f "$REPO/models/$M/main.py" ] || note "no such model: $REPO/models/$M/main.py"
done
[ -f "$REPO/ensembles/$ENSEMBLE/main.py" ] || note "no such ensemble: $REPO/ensembles/$ENSEMBLE/main.py"

# THE PARTNER'S FORECAST. The check phase 2 exists to have.
#
# A proxy for what the loader does, and named so the next reader can find the truth:
# `load_cm_frame` (pipeline-core `managers/ensemble/cm_forecast_loader.py:41`) calls
# `EnsemblePathManager(PARTNER)._get_generated_predictions_data_file_paths(run_type=...)` and
# raises "Cannot reconcile: no local forecast found for country model 'chrome_crate'" when it
# comes back empty. That is the exact string `phase2.out` printed three times on 2026-10-28,
# after the members were already trained.
PARTNER_GENERATED=$REPO/ensembles/$PARTNER/data/generated
if [ ! -d "$PARTNER_GENERATED" ]; then
    note "$PARTNER has produced no forecast on this filesystem ($PARTNER_GENERATED is absent). Run pod_run_crafd_chain.sh first — $ENSEMBLE reconciles against it and cannot proceed without the country totals."
elif [ -z "$(find "$PARTNER_GENERATED" -mindepth 1 -name '*forecasting*' -print -quit 2>/dev/null)" ]; then
    note "$PARTNER_GENERATED exists but holds no forecasting output. Run pod_run_crafd_chain.sh to completion (its STATUS must read OK) before this script."
else
    echo "  $PARTNER forecast present under $PARTNER_GENERATED"
fi

# The venv. This script does not build one: phase 1 owns that, and two scripts racing to
# create the same prefix is a failure mode worth not having.
if [ ! -d "$VENV" ]; then
    note "$VENV does not exist. It is built by pod_run_crafd_chain.sh, which must run first anyway for the partner forecast."
fi

# Publish variables must NOT be here. Not a warning — their presence means a publish
# credential reached rented hardware, which guide rule 5 forbids outright.
for V in APPWRITE_ENDPOINT APPWRITE_DATASTORE_PROJECT_ID APPWRITE_DATASTORE_API_KEY; do
    if [ -n "${!V:-}" ]; then
        note "$V is set. This chain publishes nothing and must not hold a publish credential (guide rule 5). Unset it."
    fi
done

# Memory.
# ⚠ UNITS. Both branches compute GiB, not GB: /proc/meminfo's MemTotal is kB divided by
# 1024^2, and cgroup memory.max is bytes divided by 1024^3. The echoed "GB" is therefore wrong
# by 7%, and is left as-is because the comparison is GiB-against-GiB and so correct — the same
# deliberate inconsistency `pod_run_crafd_chain.sh` carries, for the same reason.
CG_MAX=/sys/fs/cgroup/memory.max
if [ -r "$CG_MAX" ]; then
    RAW=$(cat "$CG_MAX" 2>/dev/null)
    if [ "$RAW" = "max" ]; then
        echo "  cgroup memory.max is 'max' (unlimited) — falling back to the host figure"
        MEM_GB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)
    else
        MEM_GB=$(( RAW / 1073741824 ))
    fi
else
    echo "  no cgroup v2 memory.max — falling back to the host figure, which may overstate what this container gets"
    MEM_GB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)
fi
if [ -z "${MEM_GB:-}" ]; then
    note "cannot read a memory limit from either the cgroup or /proc/meminfo"
elif [ "$MEM_GB" -lt "$MEM_FLOOR_GB" ]; then
    note "only ${MEM_GB}GB of RAM visible to this container; need >= ${MEM_FLOOR_GB}GB. $ENSEMBLE's pool peaked at >= 68.2 GiB MEASURED on 2026-10-28 and that figure is a lower bound. Rank machines by RAM first (guide rule 1) and take the 87 GiB class."
else
    echo "  RAM visible to this container: ${MEM_GB}GB (floor ${MEM_FLOOR_GB}GB, from the MEASURED 68.2 GiB peak plus headroom)"
fi

# Disk.
AVAIL_GB=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -z "$AVAIL_GB" ]; then
    note "cannot read free space on $ROOT"
elif [ "$AVAIL_GB" -lt "$DISK_FLOOR_GB" ]; then
    note "only ${AVAIL_GB}GB free on $ROOT; need >= ${DISK_FLOOR_GB}GB (DERIVED not measured — see the arithmetic at DISK_FLOOR_GB)"
else
    echo "  free on $ROOT: ${AVAIL_GB}GB (floor ${DISK_FLOOR_GB}GB, DERIVED not measured)"
fi

[ "$FAIL" = 0 ] || die "preflight failed — fix every MISSING line above and re-run. Nothing was installed and nothing was run."
echo "  preflight OK"

if [ "$PREFLIGHT_ONLY" = 1 ]; then
    echo "### --preflight only: nothing run."
    rm -f "$OUT/STATUS"
    stage preflight_ok
    exit 0
fi

# Everything from here is logged. tee so the pod console and the retained log agree.
exec > >(tee -a "$LOG") 2>&1

# ── verify the venv it inherited ───────────────────────────────────────────────────────

stage verify_install
# The toolz override is the one install whose REVERSION is silent, so it is verified by import
# rather than by version string: `import tlz.curried` is the exact thing C-151 breaks and the
# exact thing every datafactory fetch needs.
"$VENV/bin/python" -c "import tlz.curried" \
  || die "the venv cannot 'import tlz.curried' (C-151). Something re-resolved $VENV after phase 1 built it. Remedy: rm -rf $VENV and re-run pod_run_crafd_chain.sh, which installs toolz>=0.12.1 LAST."
"$VENV/bin/python" -c "import views_baseline, datafactory_query, views_pipeline_core" \
  || die "the venv is missing one of views_baseline / datafactory_query / views_pipeline_core"

PC_VERSION=$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("views-pipeline-core"))' 2>/dev/null)

# ASSERT THE RESOLVED VERSION, NOT A CONSTRAINT. This script installs nothing, so there is no
# resolve to constrain — the version is whatever phase 1 left, and phase 1 only re-resolves
# when `$VENV` is absent. A workspace that survives from before 2026-10-29 therefore carries
# a pipeline-core that cannot read a PredictionFrame cm forecast; this repository's own
# evidence records `views-pipeline-core==3.3.4` in exactly that venv
# (`tools/podrun/evidence/2026-10-28_chrome_box/crafd_chain.out:210`). Verifying that a range
# PERMITS 3.3.5 is not verifying that 3.3.5 is INSTALLED.
"$VENV/bin/python" -c 'import re,sys,importlib.metadata as m; t=lambda v: tuple(int(n) for n in re.findall(r"\d+", v)[:3]); sys.exit(0 if t(m.version("views-pipeline-core")) >= t(sys.argv[1]) else 1)' "$PC_FLOOR" \
  || die "views-pipeline-core $PC_VERSION is below the $PC_FLOOR this phase needs. Below 3.3.5, load_cm_frame cannot read a PredictionFrame cm forecast, so the reconcile dies AFTER ~3h of paid member training (views-pipeline-core#562, published 2026-10-29). Remedy: rm -rf $VENV and re-run pod_run_crafd_chain.sh, which re-resolves."
echo "  views-pipeline-core $PC_VERSION (>= $PC_FLOOR), tlz importable, engine present"

# ── memory sampler ────────────────────────────────────────────────────────────────────
# The 68.2 GiB figure is a LOWER BOUND because its series was never retrieved from a pod that
# no longer exists. This writes the series into deliver/, so the next figure is not.
( while :; do
    C=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
    echo "$(date +%H:%M:%S) $(( C / 1048576 ))MiB $(cat "$OUT/STAGE" 2>/dev/null)" >> "$OUT/memory.log"
    sleep 10
  done ) & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
[ -r /sys/fs/cgroup/memory.peak ] && echo 0 > /sys/fs/cgroup/memory.peak 2>/dev/null

export WANDB_MODE=offline WANDB_SILENT=true

# ── the eight members ─────────────────────────────────────────────────────────────────

START=$(date +%s); N=0; TRAINED=0; SKIPPED=0
for M in $MODELS; do
    N=$((N+1))
    # SKIP A MEMBER THAT HAS ALREADY PRODUCED A FORECAST. Not a hypothetical optimisation:
    # this is the state the 2026-10-28 pod was left in — all eight members trained, then the
    # reconcile died on #562. Re-running to get past it would otherwise retrain 184 minutes
    # to arrive at the same place. The manager itself does the same thing for the ensemble
    # (`prediction_frame_ensemble.py:998` loads an existing PredictionFrame rather than
    # regenerating), so this makes the member loop agree with it.
    GEN=$REPO/models/$M/data/generated
    if [ -n "$(find "$GEN" -mindepth 1 -name '*forecasting*' -print -quit 2>/dev/null)" ]; then
        SKIPPED=$((SKIPPED+1))
        echo "=== [$(date +%H:%M:%S)] member:$M ($N of 8) — SKIPPED, forecast already under $GEN"
        continue
    fi
    stage "member:$M ($N of 8)"
    cd "$REPO/models/$M" || die "cannot enter models/$M"
    T0=$(date +%s)
    "$VENV/bin/python" main.py -r forecasting -t -f || die "member $M exited non-zero at $N of 8"
    TRAINED=$((TRAINED+1))
    echo "  HEARTBEAT $M done in $(( ($(date +%s)-T0)/60 ))min; cgroup_now=$(( $(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)/1048576 ))MiB"
done
echo "  members: $TRAINED trained, $SKIPPED already present"

# ── pool and reconcile ────────────────────────────────────────────────────────────────

stage pool_and_reconcile
cd "$REPO/ensembles/$ENSEMBLE" || die "cannot enter ensembles/$ENSEMBLE"
# -f -sa, NOT -t. With -t the manager calls _train_model_artifact, which shells out to each
# member's run.sh and re-enters the conda lane (#643, #644). -sa with the artifacts on disk
# makes it load them instead. chrome_box additionally reconciles against $PARTNER, read from
# local disk — which the preflight has already confirmed is there.
"$VENV/bin/python" main.py -r forecasting -f -sa || die "$ENSEMBLE pool/reconcile exited non-zero. If this is 'Cannot reconcile: no local forecast found', check views-pipeline-core is >= $PC_FLOOR (it reported $PC_VERSION) and that $PARTNER's forecast is still on disk."
[ -d "$REPO/ensembles/$ENSEMBLE/data/generated" ] || die "$ENSEMBLE reported success but wrote no data/generated — nothing to deliver"
mkdir -p "$OUT/parquet"
cp -r "$REPO/ensembles/$ENSEMBLE/data/generated/." "$OUT/parquet/" || die "cannot copy the pool into $OUT/parquet"

# ── manifest ──────────────────────────────────────────────────────────────────────────

PEAK_FILE=$( [ -r /sys/fs/cgroup/memory.peak ] && echo "$(( $(cat /sys/fs/cgroup/memory.peak)/1048576 ))MiB" || echo "n/a" )
PEAK_LOG=$(awk '{gsub("MiB","",$2); if($2+0>m) m=$2+0} END{printf "%dMiB", m}' "$OUT/memory.log" 2>/dev/null)
stage manifest
{
  echo "ensemble:      $ENSEMBLE (pgm, global land) — phase 2 of 2"
  echo "members:       $MODELS"
  echo "members_run:   $TRAINED trained this invocation, $SKIPPED already present"
  echo "reconciled:    against $PARTNER, read from local disk"
  echo "finished:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "draws:         8 x 128 = 1024 (epic #595 as specified, NOT thinned)"
  echo "runtime_min:   $(( ($(date +%s)-START)/60 ))"
  echo "PEAK cgroup:   memory.peak=$PEAK_FILE  sampled_max=$PEAK_LOG"
  echo "peak_expected: >= 68.2 GiB, MEASURED 2026-10-28 and a LOWER BOUND (30s sampling over"
  echo "               a 4.4-minute stage cannot see a shorter transient). The two figures"
  echo "               above are the ones that replace it; memory.log carries the series."
  echo "mem_floor:     ${MEM_FLOOR_GB}GB enforced, box reported ${MEM_GB}GB"
  echo "disk_floor:    ${DISK_FLOOR_GB}GB enforced (DERIVED) — replace it with pool: below"
  echo "pool:          $(du -sh "$OUT/parquet" 2>/dev/null | cut -f1)"
  echo "pipeline_core: $PC_VERSION (floor $PC_FLOOR)"
  echo "git:           $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
  echo "driver:        tools/podrun/pod_run_chrome_box.sh"
  echo "published:     nothing — this runner uploads nothing and holds no publish credential"
} > "$OUT/MANIFEST"
cat "$OUT/MANIFEST"
echo OK > "$OUT/STATUS"
stage done
echo "### $ENSEMBLE complete. Deliverable: $OUT"
echo "### Publishing to the shelf happens FROM THE LAPTOP, never from here — see pod_run_crafd_chain.sh's header for the remaining steps."

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
#   /workspace/deliver/chrome_box/parquet/   the pooled, reconciled pgm forecast (.npy/.npz — see the copy step)
#   /workspace/deliver/chrome_box/members/   the 8 trained member artefacts + offline wandb
#                                            — THE ONLY IRREPLACEABLE THING HERE
#   /workspace/deliver/chrome_box/FREEZE     the exact package set the run used
#   .../RECONCILIATION_RESIDUAL              the country mass reconciliation could not place
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
# memory.log too, and for the same reason. The sampler appends, and the MANIFEST takes the
# max over the whole file — so without this a resume run (which skips the members and pools
# in minutes) would report the FIRST attempt's peak as its own. The whole purpose of
# retaining this series is to replace a figure whose provenance was uncertain; a peak
# contaminated by an earlier run is that defect again.
rm -f "$OUT/memory.log"

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
PARTNER_STATUS=$ROOT/deliver/$PARTNER/STATUS
if [ ! -d "$PARTNER_GENERATED" ]; then
    note "$PARTNER has produced no forecast on this filesystem ($PARTNER_GENERATED is absent). Run pod_run_crafd_chain.sh first — $ENSEMBLE reconciles against it and cannot proceed without the country totals."
elif [ -z "$(find "$PARTNER_GENERATED" -mindepth 1 -maxdepth 1 -name 'predictions_forecasting_*' -print -quit 2>/dev/null)" ]; then
    note "$PARTNER_GENERATED holds no predictions_forecasting_* output. Run pod_run_crafd_chain.sh to completion before this script."
# Phase 1 writes its own verdict, on this same PODRUN_ROOT. The earlier version of this
# check told the operator "its STATUS must read OK" and then never read it — a message
# describing a check that was not performed. A PredictionFrame directory is created before
# it is filled, so a phase 1 killed mid-save leaves a matching directory and nothing else;
# its STATUS is the only thing that distinguishes finished from interrupted.
elif [ ! -r "$PARTNER_STATUS" ]; then
    note "$PARTNER_STATUS is absent, so phase 1 has not reported a verdict on this filesystem. Run pod_run_crafd_chain.sh to completion."
elif ! grep -q '^OK$' "$PARTNER_STATUS" 2>/dev/null; then
    note "phase 1 did not finish: $PARTNER_STATUS reads '$(cat "$PARTNER_STATUS" 2>/dev/null)'. $ENSEMBLE reconciles against $PARTNER's forecast and a partial one produces a plausible wrong number. Re-run pod_run_crafd_chain.sh."
else
    echo "  $PARTNER forecast present and phase 1 reported OK"
fi

# The datafactory read credential. Phase 1 checks it; phase 2 needs it MORE, because the
# eight `*_sparrow` models are pgm global-land and fetch data phase 1 never touched. Without
# this, a box with no `.netrc` passes `--preflight OK` and dies inside member 1's fetch —
# which is the opposite of the "every problem at once" property this preflight claims.
# /root is local disk on purpose: C-154, /workspace silently ignores chmod.
if [ ! -s /root/.netrc ]; then
    note "/root/.netrc is absent or empty — the datafactory fetch needs it (docs/runpod_run_guide.md Step 2.3). All eight members fetch before anything else happens."
elif [ "$(stat -c %a /root/.netrc 2>/dev/null)" != "600" ]; then
    chmod 600 /root/.netrc 2>/dev/null \
      && echo "  fixed: /root/.netrc was not 600" \
      || note "/root/.netrc is not mode 600 and could not be chmod'd. Put it on /root, never /workspace, which silently ignores chmod (C-154)."
else
    echo "  /root/.netrc present, mode 600"
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
# TWO series, because they differ by 5x and only one of them is the requirement.
#   anon     anonymous pages — the allocation that actually has to fit in RAM. THIS is what
#            MEM_FLOOR_GB answers, and the only figure to size a box from.
#   file     page cache — reclaimable. It climbs monotonically as members write arrays and
#            is not given back until something needs the memory.
#   current  memory.current = anon + file + kernel. MEASURED on this box 2026-10-30: it
#            pinned at the 87GiB cgroup limit while anon peaked at 63.9 GiB — 1.4x high at
#            the peak, and 5x high during the member stage (22.0 GiB vs 4.3 GiB anon).
# ⛔ The withdrawn 19.8 GiB in chrome_box/configs/config_hyperparameters.py and the 68.2 GiB
# "lower bound" below were BOTH memory.current readings taken through this same mistake. The
# floor they produced is high rather than wrong, which is why it survived three sessions.
# The 68.2 figure was also a lower bound because its series was never retrieved from a pod
# that no longer exists; this writes both series into deliver/, so the next one is not.
( while :; do
    # ⛔ `a==""||f==""` is not belt-and-braces. awk skips END only when the file cannot be
    # OPENED; a readable memory.stat that carries no `anon `/`file ` line still runs END, and
    # the old version printed `anon=0MiB file=0MiB` -- a logged zero indistinguishable from
    # "used no memory", which `peak_of` would then report as a measured peak of nothing.
    MS=$(awk '/^anon /{a=$2} /^file /{f=$2} END{if (a=="" || f=="") printf "anon=NA file=NA"; else printf "anon=%dMiB file=%dMiB", a/1048576, f/1048576}' /sys/fs/cgroup/memory.stat 2>/dev/null)
    C=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
    # ⛔ `anon=NA`, never `anon=0MiB`. A sampler that cannot read memory.stat must be
    # visibly unreadable: a logged zero is indistinguishable from "used no memory", and the
    # peak parsers below would report it as a successful measurement of nothing.
    echo "$(date +%H:%M:%S) ${MS:-anon=NA file=NA} current=$(( C / 1048576 ))MiB $(cat "$OUT/STAGE" 2>/dev/null)" >> "$OUT/memory.log"
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
    # ⛔ `predictions_forecasting_*`, NOT `*forecasting*`. The looser glob also matched
    # `forecasting_log.txt`, which the run writes into this same directory BEFORE the
    # predictions are saved — the sibling run type proves the shape on disk today
    # (`models/bronze_kestrel/data/generated/` holds `calibration_log.txt` beside
    # `predictions_calibration_<ts>`). So a member whose training finished and whose
    # FORECAST stage then died — OOM, reclaim, preemption, a #562-class error: exactly the
    # resume case this skip exists for — would have been reported "already present" and
    # silently dropped from the pool. Seven of eight members then reach `-f -sa`, where the
    # best case is an ADR-015 `expected_models: 8` failure after three paid hours and the
    # worst is a plausible misaligned reconcile, because concat ordering is positionally
    # paired against the `*_condor` twins.
    GEN=$REPO/models/$M/data/generated
    if [ -n "$(find "$GEN" -mindepth 1 -maxdepth 1 -name 'predictions_forecasting_*' -print -quit 2>/dev/null)" ]; then
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

# ⛔ COUNT WHAT IS ABOUT TO BE POOLED. Nothing downstream does.
#
# `expected_models: 8` is compared only against `len(config_modelset["models"])` by
# tests/test_ensemble_configs.py — a config-against-config check at CI time. Measured in
# pipeline-core 3.3.4: `grep -rn "expected_models"` over the installed package returns ZERO
# matches. So no run-time guard sees a MISSING constituent:
#   * `_assert_expected_sample_count` raises on a member whose produced sample count is not
#     128 — seven members that each produced 128 pass it;
#   * the #160 balance guard rejects constituents whose counts differ FROM EACH OTHER —
#     seven equal members are balanced.
# A short pool therefore yields 7 x 128 = 896 draws where the config, epic #595 and the
# delivery contract all say 1,024, then reconciles, writes a MANIFEST and reports OK.
#
# This is the backstop the member skip above was wrongly believed to have.
POOLABLE=0
for M in $MODELS; do
    [ -n "$(find "$REPO/models/$M/data/generated" -mindepth 1 -maxdepth 1 -name 'predictions_forecasting_*' -print -quit 2>/dev/null)" ] \
      && POOLABLE=$((POOLABLE+1)) \
      || echo "  NO FORECAST: $M"
done
EXPECTED=$(printf '%s\n' $MODELS | wc -l)
[ "$POOLABLE" = "$EXPECTED" ] || die "only $POOLABLE of $EXPECTED members have a predictions_forecasting_* directory, so the pool would be short. Nothing downstream would catch it: pipeline-core never reads expected_models, and a short pool of equally-sized members passes both the sample-count assert and the #160 balance guard — it would deliver ${POOLABLE} x 128 draws as though it were 1024. Fix the members named above and re-run; already-forecast members are skipped."
echo "  all $POOLABLE of $EXPECTED members have a forecast on disk"

stage pool_and_reconcile
cd "$REPO/ensembles/$ENSEMBLE" || die "cannot enter ensembles/$ENSEMBLE"
# -f -sa, NOT -t. With -t the manager calls _train_model_artifact, which shells out to each
# member's run.sh and re-enters the conda lane (#643, #644). -sa with the artifacts on disk
# makes it load them instead. chrome_box additionally reconciles against $PARTNER, read from
# local disk — which the preflight has already confirmed is there.
"$VENV/bin/python" main.py -r forecasting -f -sa || die "$ENSEMBLE pool/reconcile exited non-zero. If this is 'Cannot reconcile: no local forecast found', check views-pipeline-core is >= $PC_FLOOR (it reported $PC_VERSION) and that $PARTNER's forecast is still on disk."
# ⛔ Look for the OUTPUT, not the directory. `ensembles/chrome_box/data/generated/.gitkeep`
# is TRACKED, so the directory exists in every clone before anything runs and a `[ -d ]`
# test here could never fail: an ensemble that exited 0 having written nothing would have
# copied a `.gitkeep`, reported `pool: 4.0K`, and written STATUS=OK.
POOL_DIR=$(find "$REPO/ensembles/$ENSEMBLE/data/generated" -mindepth 1 -maxdepth 1 -name 'predictions_forecasting_*' -print -quit 2>/dev/null)
[ -n "$POOL_DIR" ] || die "$ENSEMBLE exited 0 but wrote no predictions_forecasting_* under data/generated — there is nothing to deliver, and a run that produces no forecast must not report OK"
# ⚠️ `parquet/` holds NO .parquet files. Measured on the 2026-10-30 run: what lands here is
# `predictions_forecasting_<ts>/<target>/{y_pred.npy,identifiers.npz}` — numpy arrays. The
# name is inaccurate and is kept anyway, deliberately: `parquet/` is the deliverable's name
# in pod_run_crafd_chain.sh (phase 1 of THIS chain), pod_run_darts_calibration.sh,
# pod_run_model.sh and docs/runpod_run_guide.md. pod_run_model.sh may not be edited, so
# renaming here would leave one of five different forever — a divergence every reader of
# either half of the chain has to hold in their head, in exchange for a name. A comment is
# cheaper and lies to nobody. (In the two model runners the name IS accurate; they run a
# collapse step that really writes .parquet.)
mkdir -p "$OUT/parquet"
cp -r "$REPO/ensembles/$ENSEMBLE/data/generated/." "$OUT/parquet/" || die "cannot copy the pool into $OUT/parquet"

# ── collect what cannot be remade ─────────────────────────────────────────────────────
# ⛔ The pool is the product. The member artefacts are the only IRREPLACEABLE thing on the
# box, and `docs/runpod_run_guide.md:429` says so in bold — "`artifact/` is the one that
# cannot be remade" — citing 2026-09-28, when eight models trained, delivered their
# parquets, reported OK, and their weights died with the pods (views-models#549).
#
# This runner had no such stage until the 2026-10-30 teardown audit found 16 trained models
# (363.7 MiB) and 50 offline W&B runs about to die with a box that had just been declared
# safe to destroy. ⚠️ The reason it was missing is the instructive part, so it is recorded
# rather than quietly fixed: the 2026-10-28 decision to copy the pool and NOT the members
# was correct and deliberate — 26.7 GiB of member FORECASTS duplicate what the pool already
# holds — and it silently took `artifacts/` and `wandb/` with it. Those are 542 MB, not
# 30 GB, and they are the half that cannot be regenerated. A size-based exclusion inherited
# a correctness-based one.
#
# Fatal on a missing artefact, and safe to be: the pool is already copied above, so dying
# here costs attention rather than work — and a missing irreplaceable artefact is precisely
# what an operator must learn BEFORE destroying the host, not after.
# ⛔ Paired by the TIMESTAMP both filenames carry, not by mtime against this run's start.
# The first version of this stage keyed on `stat %Y "$PKL" -lt "$START"`, and that made the
# resume path -- the one this whole script is built around, and the one its own die messages
# invite -- fail every time: the member loop SKIPS a member whose forecast already exists,
# so no new .pkl is written, so every artefact predates $START, so all eight land in MISSING
# and the run dies with "no .pkl artefact from THIS run" while all eight sit on disk. And
# `rm -rf "$OUT/members"` ran first, so a previous invocation's collected artefacts -- the
# only irreplaceable thing here -- were destroyed before the check refused to re-collect
# them. Found by `/code-review high` on PR #657 and reproduced before it was accepted.
#
# The invariant that actually holds, verified across all 8 members of the 2026-10-30 run:
# `artifacts/forecasting_model_<ts>.pkl` and `data/generated/predictions_forecasting_<ts>`
# carry the SAME timestamp. So the artefact is paired to the forecast by identity rather
# than by clock, which is C-188's own exit -- key on what only exists on success, and
# compare it to something -- and it is true on a resume as well as a first run.
#
# ⚠️ No `rm -rf` here. Nothing is cleared before a replacement exists; a stale artefact from
# an earlier run is simply not the one the timestamp selects (C-171's concern, answered by
# identity instead of by deletion).
stage collect_members
mkdir -p "$OUT/members"
MISSING=""
for M in $MODELS; do
    GEN=$REPO/models/$M/data/generated
    # Extract the timestamp rather than slicing the directory name: each member writes FOUR
    # directories -- `predictions_forecasting_<ts>` plus one `<ts>_lr_*_best` per target --
    # so `sort | tail -1` would pick `..._lr_sb_best` and the timestamp with it.
    TS=$(ls -1d "$GEN"/predictions_forecasting_* 2>/dev/null \
         | sed -E 's|.*/predictions_forecasting_([0-9]{8}_[0-9]{6}).*|\1|' \
         | sort -u | tail -1)
    if [ -z "$TS" ]; then
        MISSING="$MISSING $M(no-forecast)"
        continue
    fi
    PKL=$REPO/models/$M/artifacts/forecasting_model_$TS.pkl
    if [ ! -f "$PKL" ]; then
        MISSING="$MISSING $M(no-artefact-for-$TS)"
        continue
    fi
    mkdir -p "$OUT/members/$M"
    cp "$PKL" "$OUT/members/$M/" || die "cannot copy $M's artefact into $OUT/members/$M"
    # Offline W&B is the run's own logged record and is written outside $OUT by design,
    # because WANDB_MODE=offline is forced above. C-170's second class; 164 MB, and the only
    # record of what each member logged.
    [ -d "$REPO/models/$M/wandb" ] && cp -r "$REPO/models/$M/wandb" "$OUT/members/$M/" 2>/dev/null
done
[ -d "$REPO/ensembles/$ENSEMBLE/wandb" ] && cp -r "$REPO/ensembles/$ENSEMBLE/wandb" "$OUT/members/_$ENSEMBLE-wandb" 2>/dev/null
[ -z "$MISSING" ] || die "could not pair a trained artefact with a forecast for:$MISSING — '(no-forecast)' means the member wrote no predictions_forecasting_* at all; '(no-artefact-for-<ts>)' means it has that forecast but no artifacts/forecasting_model_<ts>.pkl beside it, so the trained state that produced it is gone and cannot be regenerated without re-running the member (~23 min each, measured). Nothing has been deleted and the pool is intact at $OUT/parquet; do NOT destroy this host until you have decided whether you need those artefacts."
echo "  collected $(find "$OUT/members" -name '*.pkl' | wc -l) member artefacts, $(du -sh "$OUT/members" | cut -f1)"

# ── FREEZE: what this run actually ran on ─────────────────────────────────────────────
# The MANIFEST pins the two packages this runner asserts a floor on. The venv has ~143.
# ⛔ `pip freeze` returns NOTHING in this venv — it is uv-built and has no pip — so the
# obvious way to capture this writes an empty file, which is the absence that reads as
# success. `importlib.metadata` is asked instead: it is the same source the pipeline-core
# version assert above uses, needs neither pip nor uv, and cannot be silently empty.
# Why it matters beyond tidiness: the reconciliation mass deficit this runner reports is a
# property of a specific views-frames, so a reproduction that re-resolves it is not one.
"$VENV/bin/python" - > "$OUT/FREEZE" <<'PYFREEZE'
import importlib.metadata as m
print("\n".join(sorted({f"{d.name}=={d.version}" for d in m.distributions() if d.name})))
PYFREEZE
[ -s "$OUT/FREEZE" ] || die "could not record the package set into $OUT/FREEZE. The run itself succeeded and the pool is intact at $OUT/parquet, but nothing would record WHICH views-frames produced this reconciliation, and that is the one version a reproduction must match."
echo "  recorded $(wc -l < "$OUT/FREEZE") packages in $OUT/FREEZE"

# ── the residual: what reconciliation could NOT place ─────────────────────────────────
# ⛔ Runs on every successful run, and its figures go into the MANIFEST, because the
# 2026-10-30 delivery lost 2.9%-7.3% of the country forecast's fatalities and said only
# `reconciled: against chrome_crate`. Multiplicative scaling cannot lift a country whose
# cells are all zero (0 x anything = 0), so some loss is inherent to the method — but a
# delivery may be approximate, never QUIETLY approximate, least of all to a UN partner.
#
# Two outcomes, deliberately different:
#   rc 0   measured. Unplaced mass is reported and the run continues: it is a property of
#          the method, and the forecast is correct everywhere the grid is non-zero.
#   rc!=0  either the scaling actually failed (country totals are WRONG, which is a defect
#          and not a limit), or the residual could not be measured at all. Both are fatal.
#          The pool is already copied to deliver/ above, so dying here loses nothing but
#          costs a human's attention — and keeps `OK` meaning "fully verified"
#          (ADR-025: a check that cannot see its subject must refuse, not skip).
stage reconciliation_residual
( cd "$REPO" && "$VENV/bin/python" -m reconciliation.residual \
      --ensemble "$REPO/ensembles/$ENSEMBLE" \
      --out "$OUT/RECONCILIATION_RESIDUAL" )
RESIDUAL_RC=$?
[ "$RESIDUAL_RC" = 0 ] || die "the reconciliation residual check exited $RESIDUAL_RC. The pool itself is INTACT at $OUT/parquet and nothing is lost, but this run must not be reported OK: either the scaling failed and the country totals are wrong, or the residual could not be measured. Re-run just this check with: cd $REPO && $VENV/bin/python -m reconciliation.residual --ensemble ensembles/$ENSEMBLE"

# Pulled out of the manifest block so an empty result is visible as a refusal rather than
# as a missing line — `grep` printing nothing is exactly the absence that reads as a pass.
RESIDUAL_LINES=$(grep -E '^[A-Za-z0-9_]+ +unplaced ' "$OUT/RECONCILIATION_RESIDUAL" 2>/dev/null)
[ -n "$RESIDUAL_LINES" ] || RESIDUAL_LINES="⛔ UNREADABLE — RECONCILIATION_RESIDUAL carries no per-target line"

# ── manifest ──────────────────────────────────────────────────────────────────────────

PEAK_FILE=$( [ -r /sys/fs/cgroup/memory.peak ] && echo "$(( $(cat /sys/fs/cgroup/memory.peak)/1048576 ))MiB" || echo "n/a" )
# ⛔ `seen` is not decoration. Without it a log whose format changed, or an unreadable
# memory.stat, yields `0MiB` — which prints in the MANIFEST as a measured peak of nothing
# and would have the next operator size a box from it (ADR-025: a check that cannot see its
# subject must refuse, not return a clean value).
peak_of() {  # peak_of <key> — max of `<key>=<n>MiB` across memory.log, or UNREADABLE
  awk -v k="$1" '{for(i=1;i<=NF;i++) if(index($i,k"=")==1){v=$i; gsub(/[^0-9]/,"",v); if(v!=""){seen=1; if(v+0>m) m=v+0}}} END{if(seen) printf "%dMiB", m; else printf "UNREADABLE"}' "$OUT/memory.log" 2>/dev/null
}
# ⛔ Defaulted, because awk does not run END at all when it cannot open the file -- so a
# missing memory.log yields an EMPTY string rather than the UNREADABLE the function promises,
# and the MANIFEST would print "PEAK anon:" with no value beside it.
PEAK_ANON=$(peak_of anon);    PEAK_ANON=${PEAK_ANON:-UNREADABLE}
PEAK_CUR=$(peak_of current);  PEAK_CUR=${PEAK_CUR:-UNREADABLE}
stage manifest
{
  echo "ensemble:      $ENSEMBLE (pgm, global land) — phase 2 of 2"
  echo "members:       $MODELS"
  echo "members_run:   $TRAINED trained this invocation, $SKIPPED already present"
  echo "members_pooled: $POOLABLE of $EXPECTED verified present before pooling"
  # ⛔ Never print this line unqualified. It used to read "against $PARTNER, read from
  # local disk" and stop — a statement about what was ATTEMPTED, in a field every reader
  # takes as a statement about what was ACHIEVED. Measured on 2026-10-30: 2.9%-7.3% of the
  # country forecast's fatalities were not on the grid and nothing said so.
  echo "reconciled:    against $PARTNER, read from local disk — PARTIALLY. See residual:"
  echo "residual:      country fatalities that could NOT be placed on the grid, because"
  echo "               multiplicative scaling cannot lift a country whose cells are all"
  echo "               zero. Full block in RECONCILIATION_RESIDUAL beside this file."
  echo "$RESIDUAL_LINES" | sed 's/^/               /'
  echo "finished:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # Stated as CONFIGURED, not as observed, and deliberately WITHOUT naming a backstop.
  # ⛔ Never write "enforced by ADR-015 expected_models" here. `grep -rn expected_models`
  # over installed views-pipeline-core 3.3.5 returns ZERO matches — nothing reads it at run
  # time. `expected_samples_per_model` IS enforced (`_assert_expected_sample_count`,
  # prediction_frame_ensemble.py:153) and the #160 balance guard rejects constituents that
  # differ FROM EACH OTHER, but neither of them can see a MISSING constituent: seven members
  # at 128 draws are mutually balanced and individually correct, so a short pool passes both
  # and delivers 896 draws as though they were 1024. `members_pooled:` above is the ONLY
  # check of the count and it is this script's own. That claim was made four times before it
  # was measured — in #649's commit message, in register entry C-188, in a test docstring,
  # and in this line, which printed it into a real delivery artifact on 2026-10-30.
  echo "draws:         8 x 128 = 1024 configured (epic #595, NOT thinned) — see members_pooled: above for the only verification of the count"
  echo "runtime_min:   $(( ($(date +%s)-START)/60 ))"
  echo "PEAK anon:     $PEAK_ANON  <- THE REQUIREMENT. Size a box from this and nothing else."
  echo "PEAK current:  $PEAK_CUR (memory.peak=$PEAK_FILE) — INFLATED by reclaimable page cache."
  echo "               On 2026-10-30 this pinned at the cgroup limit while anon peaked at"
  echo "               63.9 GiB. A box chosen from this number is oversized, not safe."
  echo "peak_history:  the 68.2 GiB 'lower bound' (2026-10-28) and the withdrawn 19.8 GiB in"
  echo "               configs/config_hyperparameters.py were BOTH memory.current readings,"
  echo "               so both are superseded by PEAK anon. memory.log carries both series."
  echo "mem_floor:     ${MEM_FLOOR_GB}GB enforced, box reported ${MEM_GB}GB"
  echo "disk_floor:    ${DISK_FLOOR_GB}GB enforced (DERIVED) — replace it with pool: below"
  echo "pool:          $(du -sh "$OUT/parquet" 2>/dev/null | cut -f1)"
  echo "artefacts:     $(find "$OUT/members" -name '*.pkl' 2>/dev/null | wc -l) trained member artefacts + their offline wandb, in members/ ($(du -sh "$OUT/members" 2>/dev/null | cut -f1))"
  echo "packages:      $(wc -l < "$OUT/FREEZE" 2>/dev/null) recorded in FREEZE beside this file — the venv is uv-built and has no pip"
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

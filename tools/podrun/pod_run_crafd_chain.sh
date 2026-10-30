#!/usr/bin/env bash
# pod_run_crafd_chain.sh [--preflight]
#
# ┌──────────────────────────────────────────────────────────────────────────────────────┐
# │  VERSION 0.1.0 — PROVISIONAL.  NOT part of the monthly production run.                │
# │  No CI runs it. It uploads nothing, and it installs no credential that could.         │
# │  See tools/podrun/__init__.py for what promoting it out of 0.1.0 would require.       │
# └──────────────────────────────────────────────────────────────────────────────────────┘
#
# PHASE 1 of the CRAF'd chain (epic #595, story #640): the eight `*_condor` country-month
# baselines and the `chrome_crate` ensemble that pools them, on a rented pod, from a bare
# image to a pooled country-month forecast on disk.
#
# PHASE 2 — the eight `*_sparrow` pgm models and `chrome_box`, which reconciles against
# chrome_crate — is deliberately NOT here. chrome_box at the epic's 8 x 128 = 1024 draws is
# the shape rusty_bucket's own config comment records as "peaks ~28.6 GB and cannot fit
# production hardware"; the draw count is an open decision, and it decides which machine
# class to rent. Adding it now would mean writing a memory floor for a configuration nobody
# has chosen. The trigger to extend this script is that decision being made.
#
# ## Why this is a sibling and not a flag on pod_run_model.sh
#
# pod_run_model.sh installs `views-hydranet~=0.1.1` at line 172 and imports views_hydranet. It
# is the HydraNet runner and honest about it. These are views-baseline models, so a --library
# flag would branch it at the install set, the import, the artifact check and the deliverable
# shape — four places, both paths harder to read. pod_run_fao_delivery.sh and
# pod_run_darts_calibration.sh already set the precedent of a sibling with its own copy of the
# helpers. This is the fourth occurrence of that pattern, which is when you copy rather than
# abstract.
#
# ## Its own venv, and why that is the point
#
#   VENV=$ROOT/venv-crafd
#
# The other three scripts all use `$ROOT/venv`, so on a pod there is exactly ONE virtualenv and
# its contents are whatever the last script installed. That is the uv-lane twin of register
# C-116 (one environment, several engines, last writer wins), and it is why a baseline chain
# cannot simply reuse what pod_run_model.sh built: it would find views-hydranet and no
# views_baseline. This chain owns its environment so the next one cannot inherit it.
#
# ## Why it never calls run.sh
#
# Every step runs `$VENV/bin/python main.py` directly. `models/*/run.sh` activates a CONDA
# prefix, pip-installs `requirements.txt` into it and does not check pip's exit status
# (views-pipeline-core#560), and on this platform that prefix is shared by 45 tenants whose
# constraints do not all hold at once (views-models#644). Running main.py under an explicitly
# installed venv is what makes this lane immune to all of it — and it is why the ensemble is
# pooled with `-f -sa` below rather than `-t`.
#
# NO publish credential, deliberately. This chain produces a forecast on disk and uploads
# nothing, so the only secret it needs is the datafactory READ credential in /root/.netrc. The
# nine Appwrite publish variables are not merely unnecessary here — placing them on hardware we
# do not control is a cost with no benefit (guide rule 5, ADR-024). The CRAF'd delivery is
# gated separately on views-crafdapi#180 and views-models#599.
#
# Leaves behind:
#   /workspace/deliver/chrome_crate/parquet/   the pooled country-month forecast
#   /workspace/deliver/chrome_crate/members/   each member's forecast, as produced
#   /workspace/deliver/chrome_crate/MANIFEST   versions, timestamps, git sha, what ran
#   /workspace/deliver/chrome_crate/STATUS     OK or FAILED:<stage>
#   /workspace/deliver/chrome_crate/FAILURE    the reason, on failure only
#   /workspace/deliver/chrome_crate/run.log    the whole transcript
#
# Progress is readable from outside at any time:  cat /workspace/deliver/chrome_crate/STAGE

set -uo pipefail

USAGE='usage: pod_run_crafd_chain.sh [--preflight]'
# --preflight runs every check that costs nothing and then STOPS. On a fresh pod it takes
# seconds and reports EVERY problem at once rather than one per attempt, which is the
# difference between one round trip and six.
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

# Honours PODRUN_ROOT for the same reason the other three scripts do, so all four AGREE on
# where deliver/<name>/STATUS lives. On a pod it is /workspace and nothing changes.
ROOT="${PODRUN_ROOT:-/workspace}"
REPO=$ROOT/views-models
VENV=$ROOT/venv-crafd
VENV_MARKER=$VENV/.built-by-pod-run-crafd-chain

ENSEMBLE=chrome_crate
# The eight cm members, in chrome_crate's config_modelset.py order. Listed literally rather
# than parsed out of the config: this script must be readable on a pod with no python yet, and
# a chain that silently follows a reordered config is worse than one that disagrees loudly —
# tests/test_crafd_chain_runner.py compares this list against the config and fails on drift.
MODELS="ashen_condor brisk_condor candid_condor dusky_condor eager_condor faint_condor gentle_condor humble_condor"

OUT=$ROOT/deliver/$ENSEMBLE
LOG=$OUT/run.log
VENV_LOCK=$ROOT/.venv-crafd-build.lock
HELD_VENV_LOCK=0

mkdir -p "$OUT/members" || { echo "!!! cannot create $OUT" >&2; exit 1; }

# A run begins by clearing the previous run's verdict. Without this, STATUS from an earlier
# failed --preflight survives into the next invocation, and both an operator and an automated
# poller read FAILED while the run is in fact progressing — observed on a pod 2026-10-28,
# where a stale `FAILED:preflight` from 11:58:46 was still being served at 12:00 during a
# healthy install. STAGE is the live field; STATUS is a verdict and must not outlive its run
# (`feedback_status_fields_lie`).
rm -f "$OUT/STATUS" "$OUT/FAILURE"

stage() { echo "$1" > "$OUT/STAGE"; echo "=== [$(date +%H:%M:%S)] $1 ==="; }
die()   {
    echo "FAILED:$(cat "$OUT/STAGE" 2>/dev/null)" > "$OUT/STATUS"
    # Written directly, not through the tee subshell: on a hard kill (preemption, OOM) the last
    # buffered log lines can be lost, and that is exactly when the reason matters.
    printf '%s\n' "$1" > "$OUT/FAILURE"
    echo "!!! $1"
    [ "$HELD_VENV_LOCK" = 1 ] && rmdir "$VENV_LOCK" 2>/dev/null
    exit 1
}

# ── preflight ─────────────────────────────────────────────────────────────────────────
# Everything here costs seconds. The failure that matters on rented hardware is discovering
# after an hour of paid time that a credential was missing or a disk was full.

stage preflight
FAIL=0
note() { echo "  MISSING: $*"; FAIL=1; }

[ -d "$REPO/.git" ] || note "$REPO is not a checkout — clone views-models first (docs/runpod_run_guide.md Phase 2)"
for M in $MODELS; do
    [ -f "$REPO/models/$M/main.py" ] || note "no such model: $REPO/models/$M/main.py"
done
[ -f "$REPO/ensembles/$ENSEMBLE/main.py" ] || note "no such ensemble: $REPO/ensembles/$ENSEMBLE/main.py"

# The `[pandas]` extra is unsatisfiable alongside views-pipeline-core 3.x, because
# pipeline-core requires viewser and viewser caps pandas<2 while the extra floors it at >=2
# (views-models#644, views-pipeline-core#561). This lane installs its own set and never reads
# requirements.txt, so the extra cannot break the install here — but a member declaring it is
# a member whose conda lane is broken, and that is worth saying once, cheaply, rather than
# leaving the two declarations to disagree in silence.
PANDAS_EXTRA_SEEN=0
for M in $MODELS; do
    if grep -q 'views-datafactory\[pandas\]' "$REPO/models/$M/requirements.txt" 2>/dev/null; then
        PANDAS_EXTRA_SEEN=$((PANDAS_EXTRA_SEEN + 1))
    fi
done
# ADVISORY, not a refusal. This was a `note` (FAIL=1) in the first version, which would have
# refused every run against a checkout of `development` while the fix was unmerged — a
# refusal for a condition this lane is immune to by construction, with no reachable green
# state. That is the "red when things are fine" failure the guard audits kept finding.
if [ "$PANDAS_EXTRA_SEEN" -gt 0 ]; then
    echo "  ADVISORY: $PANDAS_EXTRA_SEEN of 8 members declare views-datafactory[pandas] in"
    echo "            requirements.txt, which cannot resolve with views-pipeline-core 3.x"
    echo "            (views-models#644). THIS LANE IS UNAFFECTED — it installs its own set"
    echo "            and never reads requirements.txt — but those models' run.sh cannot work."
fi

# The datafactory credential, and the ONLY credential this chain needs. HTTP Basic from
# ~/.netrc in the runtime user's home, resolved at call time. /root is local disk on purpose:
# /workspace is a network filesystem that silently ignores chmod, so 600 does not hold there
# (register C-154, #518).
[ -s /root/.netrc ] || note "/root/.netrc — the datafactory fetch needs it (guide Step 2.3)"
if [ -s /root/.netrc ] && [ "$(stat -c %a /root/.netrc 2>/dev/null)" != "600" ]; then
    chmod 600 /root/.netrc
    echo "  fixed: /root/.netrc was not 600"
fi

# Publish variables must NOT be here. Not a warning — their presence means a publish credential
# was placed on rented hardware, which guide rule 5 forbids, and this chain has no use for one.
for V in APPWRITE_ENDPOINT APPWRITE_DATASTORE_PROJECT_ID APPWRITE_DATASTORE_API_KEY; do
    if [ -n "${!V:-}" ]; then
        note "$V is set. This chain publishes nothing and must not hold a publish credential (guide rule 5). Unset it."
    fi
done

# Memory, read from the cgroup rather than the host (guide rule 2: a machine advertised as
# "16 vCPU" reports cpu.max of 13.6 effective, and memory limits differ from the listing too).
#
# MEASURED 2026-10-28, which is what the earlier version of this comment asked for. It was
# derived (~85 MB for the pooled frame) and said "REPLACE THIS with the observed peak after
# the first run". The first run has happened; the derivation was close.
#
#   chrome_crate pool, on disk                      96 MB   (manifest: `pool: 96M`)
#   the eight cm members, retained                 176 MB   (manifest: `members_kept: 176M`)
#   each member's peak RSS                        ~2.0 MB   (HEARTBEAT rss_peak_kb ~1_990)
#   whole phase, wall clock                          7 min
#   box it ran on                                   87 GiB visible, floor 16, never strained
#
# So 16 stays. It is not a tight floor and does not need to be: this phase is country-month,
# 191 countries x 36 steps, and the arithmetic bounds it below a gigabyte. The floor exists to
# refuse a box too small to hold the venv's working set and the datafactory cache, not the pool.
#
# ⛔ DO NOT RAISE THIS TO chrome_box's FIGURE. `chrome_box` (pgm, global land) peaked at
# 68.2 GiB anon on 2026-10-28 and wants an 87 GiB class. That is PHASE 2, which is
# deliberately not in this script (see the header), so this preflight never guards it. The
# number lives where phase 2 is configured —
# `ensembles/chrome_box/configs/config_hyperparameters.py` — and is pinned by
# `tests/test_chrome_box_memory_is_the_measured_peak.py`. A pgm peak used as a cm floor is
# the same value-for-the-wrong-question mistake that produced views-models#648, which was
# filed against this line and is invalid: the floor was never too low.
#
# ⚠ UNITS. Both branches below compute GiB, not GB: /proc/meminfo's MemTotal is kB and is
# divided by 1024^2, and cgroup memory.max is bytes divided by 1024^3. The variable names and
# the echoed "GB" are therefore wrong by 7%. Left as-is because the comparison is
# GiB-against-GiB and so correct; renaming touches the two echo strings and an external
# guard's regex. Trigger to fix: the next edit that does arithmetic across this boundary.
MEM_FLOOR_GB=16
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
    note "only ${MEM_GB}GB of RAM visible to this container; need >= ${MEM_FLOOR_GB}GB (rank machines by RAM first — guide rule 1)"
else
    echo "  RAM visible to this container: ${MEM_GB}GB (floor ${MEM_FLOOR_GB}GB, DERIVED not measured)"
fi

# Disk. The venv is the bulk of it: views-pipeline-core pulls pyarrow, wandb, scipy,
# scikit-learn and matplotlib; views-baseline and views-datafactory add xarray, zarr and dask.
# Plus the datafactory cache for eight cm querysets and the retained member forecasts.
# ⚠ ALSO DERIVED. Revise from the MANIFEST's measured sizes after the first run.
DISK_FLOOR_GB=25
AVAIL_GB=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -z "$AVAIL_GB" ]; then
    note "cannot read free space on $ROOT"
elif [ "$AVAIL_GB" -lt "$DISK_FLOOR_GB" ]; then
    note "only ${AVAIL_GB}GB free on $ROOT; need >= ${DISK_FLOOR_GB}GB"
else
    echo "  free on $ROOT: ${AVAIL_GB}GB (floor ${DISK_FLOOR_GB}GB, DERIVED not measured)"
fi

if [ "$FAIL" = 1 ]; then
    echo
    echo "!!! preflight found problems above. Nothing has been installed or run."
    echo "FAILED:preflight" > "$OUT/STATUS"
    exit 1
fi
echo "  preflight clean"

if [ "$PREFLIGHT_ONLY" = 1 ]; then
    stage preflight_only_done
    echo "### --preflight only: nothing installed, nothing run."
    exit 0
fi

# Everything from here is logged. tee so the pod console and the retained log agree.
exec > >(tee -a "$LOG") 2>&1

# ── install ───────────────────────────────────────────────────────────────────────────

if [ ! -d "$VENV" ]; then
  if mkdir "$VENV_LOCK" 2>/dev/null; then
    HELD_VENV_LOCK=1
    stage install_python
    uv venv --python 3.11 "$VENV" || die "venv creation failed"
    # The install set, declared HERE and not read from any requirements.txt. That is a fourth
    # place this platform states these dependencies (after the model's requirements.txt, the
    # engine's pyproject and the conda prefix), and the drift is real — pod_run_model.sh:172
    # shows the same shape. The trade is deliberate: an explicit set is the only way this lane
    # can be immune to a requirements.txt that cannot resolve (#644).
    #
    # NO `[pandas]` extra, deliberately and load-bearingly. views-pipeline-core 3.x requires
    # viewser (ADR-063 clause 1), viewser caps pandas<2, and the extra floors pandas at >=2 —
    # ResolutionImpossible, measured 2026-10-28 in a clean venv. pandas arrives anyway through
    # xarray, which is what the proven rusty_bucket stand-ins relied on. See
    # views-pipeline-core#561 for the attempt to remove the inherited viewser.
    #
    # pipeline-core floored at 3.2.0: below it the loader cannot find configs/config_maturity.py
    # and the run dies on a missing config_deployment.py (#594, #635).
    uv pip install --python "$VENV/bin/python" \
        "views-baseline>=1.0.2,<2.0.0" \
        "views-pipeline-core>=3.2.0,<4.0.0" \
        "views-datafactory>=1.13.0,<2.0.0" || die "pip install failed"
    # register C-151: viewser pins toolz<0.12, which cannot import tlz submodules on Python
    # 3.11 and breaks EVERY datafactory fetch. Override after resolution.
    #
    # This MUST stay the last install in this block. Any pip install appended below it
    # re-resolves this venv and can silently pull toolz back under 0.12 — which happened on
    # 2026-09-29, reverting it 1.1.0 -> 0.11.2 with no error.
    uv pip install --python "$VENV/bin/python" "toolz>=0.12.1" || die "toolz override failed"
    touch "$VENV_MARKER" || die "cannot write $VENV_MARKER"
  else
    echo "another run is building $VENV — waiting"
    while [ -d "$VENV_LOCK" ]; do sleep 5; done
  fi
  rmdir "$VENV_LOCK" 2>/dev/null && HELD_VENV_LOCK=0
else
  echo "venv already present at $VENV — reusing"
fi

[ -f "$VENV_MARKER" ] \
  || die "$VENV exists but carries no $(basename "$VENV_MARKER") marker, so this script cannot establish the order it was installed in — rm -rf it and re-run. It may be fine; the point is that nothing here can tell (C-151, #517)."

stage verify_install
# The toolz override is the one install whose REVERSION is silent, so it is verified by import
# rather than by version string: `import tlz.curried` is the exact thing C-151 breaks and the
# exact thing every datafactory fetch needs.
"$VENV/bin/python" -c "import tlz.curried" \
  || die "the venv cannot 'import tlz.curried' (C-151). The toolz override did not hold — something re-resolved this venv after it."
"$VENV/bin/python" -c "import views_baseline, datafactory_query, views_pipeline_core" \
  || die "the venv is missing one of views_baseline / datafactory_query / views_pipeline_core"
PC_VERSION=$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("views-pipeline-core"))' 2>/dev/null)
echo "  views-pipeline-core $PC_VERSION, tlz importable, engine present"

# W&B offline for the whole chain. The ensemble manager opens a W&B run before any member
# trains (pipeline-core managers/ensemble/prediction_frame_ensemble.py:413), so without this a
# box with no key stops at a credential prompt having done no work (views-models#645). Offline
# also keeps a W&B credential off rented hardware, which is guide rule 5 again.
export WANDB_MODE=offline WANDB_SILENT=true

# ── the eight members ─────────────────────────────────────────────────────────────────

START_ALL=$(date +%s)
N=0
for M in $MODELS; do
    N=$((N + 1))
    stage "member:$M ($N of 8)"
    cd "$REPO/models/$M" || die "cannot enter models/$M"
    START_M=$(date +%s)
    # -t -f: train then forecast, in one invocation, so the forecast step reuses what training
    # just cached. No -sa on a member: `saved = args.saved if train else True` already makes
    # the forecast step saved, while passing it would send the TRAINING step looking for local
    # data a fresh pod has never written (#639).
    "$VENV/bin/python" main.py -r forecasting -t -f \
      || die "member $M exited non-zero at $N of 8"
    echo "  HEARTBEAT $M done in $(( ($(date +%s) - START_M) / 60 ))min; rss_peak_kb=$(grep VmHWM /proc/self/status 2>/dev/null | tr -dc '0-9' || echo unknown)"
    # Bring the member's output under deliver/ as it is produced, not at the end. Pod artefacts
    # that exist only in the checkout die with the pod, and the 2026-09-29 FAO run lost exactly
    # this (project_pod_artefact_pattern).
    if [ -d "$REPO/models/$M/data/generated" ]; then
        mkdir -p "$OUT/members/$M"
        cp -r "$REPO/models/$M/data/generated/." "$OUT/members/$M/" 2>/dev/null || true
    fi
done

# ── the pool ──────────────────────────────────────────────────────────────────────────

stage pool
cd "$REPO/ensembles/$ENSEMBLE" || die "cannot enter ensembles/$ENSEMBLE"
# `-f -sa`, NOT `-t`. With -t the manager calls _train_model_artifact, which shells out to
# models/<member>/run.sh and lands back in the conda lane this whole script exists to avoid.
# -f alone takes _forecast_model_artifact, which loads what the members just wrote. -sa pools
# the member forecasts above rather than refetching, so the eight runs are not wasted.
# No -p: this chain publishes nothing (guide rule 5).
"$VENV/bin/python" main.py -r forecasting -f -sa \
  || die "$ENSEMBLE pool exited non-zero"

if [ -d "$REPO/ensembles/$ENSEMBLE/data/generated" ]; then
    mkdir -p "$OUT/parquet"
    cp -r "$REPO/ensembles/$ENSEMBLE/data/generated/." "$OUT/parquet/" 2>/dev/null || true
fi
[ -n "$(ls -A "$OUT/parquet" 2>/dev/null)" ] \
  || die "the pool reported success but $OUT/parquet is empty — nothing was produced"

# ── manifest ──────────────────────────────────────────────────────────────────────────

RUN_MIN=$(( ($(date +%s) - START_ALL) / 60 ))

stage manifest
{
  echo "ensemble:       $ENSEMBLE (country-month)"
  echo "members:        $MODELS"
  echo "finished:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_type:       forecasting"
  echo "engine:         views_baseline $("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("views-baseline"))' 2>/dev/null || echo unknown)"
  echo "pipeline_core:  $PC_VERSION"
  echo "toolz:          $("$VENV/bin/python" -c 'import toolz; print(toolz.__version__)' 2>/dev/null || echo unknown) (C-151 override, verified by importing tlz.curried)"
  echo "venv:           $VENV (its own — not the shared \$ROOT/venv)"
  echo "runtime_min:    $RUN_MIN"
  echo "pool:           $(du -sh "$OUT/parquet" 2>/dev/null | cut -f1)"
  echo "members_kept:   $(du -sh "$OUT/members" 2>/dev/null | cut -f1)"
  echo "git:            $(git -C "$REPO" rev-parse --short HEAD)"
  echo "config:         as committed at the git sha above"
  echo "wandb:          offline (WANDB_MODE=offline), nothing sent"
  echo "published:      nothing — this runner uploads nothing and holds no publish credential"
  echo "phase:          1 of 2 — chrome_box (pgm) is NOT in this script; see the header"
} > "$OUT/MANIFEST"
cat "$OUT/MANIFEST"

echo OK > "$OUT/STATUS"
stage done
echo "### $ENSEMBLE — COMPLETE in ${RUN_MIN}min"
echo "### phase 2 (chrome_box, pgm) is a separate decision — it needs the draw count settled."

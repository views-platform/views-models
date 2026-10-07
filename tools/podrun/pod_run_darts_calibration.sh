#!/usr/bin/env bash
# pod_run_darts_calibration.sh [--preflight] <model_name>
#
# ┌──────────────────────────────────────────────────────────────────────────────────────┐
# │  VERSION 0.1.0 — PROVISIONAL.  NOT part of the monthly production run.                │
# │  No CI runs it. It uploads nothing, and it installs no credential that could.         │
# │  See tools/podrun/__init__.py for what promoting it out of 0.1.0 would require.       │
# └──────────────────────────────────────────────────────────────────────────────────────┘
#
# One r2darts2 model, calibration partition, global pgm, on a rented pod — from a bare
# runpod/pytorch image to the point-prediction parquets the research team consumes
# (views-models#505), without a person watching. Story #534 of epic #532.
#
# The sibling of pod_run_model.sh, which cannot do this: it installs views-hydranet, imports
# views_hydranet, gates on `total_lessons` (a key r2darts2 does not have) and verifies a
# HydraNet-shaped posterior deliverable. A --library flag would have branched that script in
# four places and made both paths harder to read; pod_run_fao_delivery.sh already set the
# precedent of a sibling with its own copy of the helpers.
#
# It is written to fail LOUDLY and EARLY. Every expensive step is preceded by a check that
# costs seconds, because the failure that matters here is discovering after hours of paid GPU
# time that a credential was missing, a disk was full, or the config forbids the run.
#
# Leaves behind:
#   /workspace/deliver/<model>/parquet/    13 point-prediction parquets
#   /workspace/deliver/<model>/MANIFEST    sizes, timestamps, git sha, the gated config values
#   /workspace/deliver/<model>/STATUS      OK or FAILED:<stage>
#   /workspace/deliver/<model>/FAILURE     the reason, on failure only (written directly)
#   /workspace/deliver/<model>/run.log     the whole transcript
#
# NO --rehearsal, deliberately. The HydraNet script needs one because a 300-lesson budget can
# be spent by accident and the only cheap test was to delete the guard. Here the cheap test is
# the sequencing itself: #537 runs ONE model — the cheapest of the eleven — and measures it
# before anything fans out. Adding a patch-and-verify mechanism for `n_epochs` would also be
# patching a number that `early_stopping_patience` already makes approximate. If a cheap
# end-to-end test turns out to be wanted anyway, that is the trigger to add it.
#
# NO publish credential, deliberately. A calibration run uploads nothing, so the only secret
# this needs is the datafactory READ credential in /root/.netrc. The nine Appwrite publish
# variables are not merely unnecessary — placing them on hardware we do not control is a cost
# with no benefit. See ADR-024 and docs/runpod_run_guide.md.
#
# Progress is readable from outside at any time:  cat /workspace/deliver/<model>/STAGE

set -uo pipefail

USAGE='usage: pod_run_darts_calibration.sh [--preflight] <model_name>'
# --preflight runs every check that costs nothing and then STOPS. On a fresh pod it takes
# seconds and reports EVERY problem at once rather than one per attempt, which is the
# difference between one round trip and six.
PREFLIGHT_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --preflight) PREFLIGHT_ONLY=1; shift ;;
        --*) echo "!!! unknown option: $1" >&2; echo "!!! $USAGE" >&2; exit 1 ;;
        *)   break ;;
    esac
done
MODEL="${1:?$USAGE}"
[ $# -le 1 ] || { echo "!!! one model at a time; got: $*" >&2; echo "!!! $USAGE" >&2; exit 1; }

# Honours PODRUN_ROOT for the same reason the other two scripts do, and so all three AGREE on
# where deliver/<model>/STATUS lives. On a pod it is /workspace and nothing changes.
ROOT="${PODRUN_ROOT:-/workspace}"
REPO=$ROOT/views-models
VENV=$ROOT/venv
OUT=$ROOT/deliver/$MODEL
LOG=$OUT/run.log
VENV_LOCK=$ROOT/.venv-build.lock
HELD_VENV_LOCK=0

# Checked, because an unchecked mkdir fails off a pod and the script then runs on into a broken
# tee and a lock check that reports "another run is in progress" for a machine that simply has
# no /workspace.
mkdir -p "$OUT" 2>/dev/null || {
    echo "!!! cannot create $OUT" >&2
    echo "!!! This script runs ON A POD, where $ROOT exists. To exercise it elsewhere, set" >&2
    echo "!!!   PODRUN_ROOT=/some/writable/dir" >&2
    exit 1
}
# A previous attempt on this pod may have left FAILED here, and the header advertises STATUS as
# the way to watch from outside, so a stale one actively misleads.
rm -f "$OUT/STATUS" "$OUT/FAILURE"
exec > >(tee -a "$LOG") 2>&1

stage() { echo "$1" > "$OUT/STAGE"; echo "=== [$(date +%H:%M:%S)] $1 ==="; }
die()   {
    echo "FAILED:$(cat "$OUT/STAGE" 2>/dev/null)" > "$OUT/STATUS"
    # Written directly, not through the tee subshell: on a hard kill (preemption, OOM) the last
    # buffered log lines can be lost, and that is exactly when the reason matters.
    printf '%s\n' "$1" > "$OUT/FAILURE"
    echo "!!! $1"
    exit 1
}

if ! mkdir "$OUT/.lock" 2>/dev/null; then
    # "exists" and "could not be created" need different actions, and reporting the first for
    # the second sends the operator looking for a run that never started.
    if [ -d "$OUT/.lock" ]; then
        echo "!!! $OUT/.lock exists — another run for $MODEL is in progress on this pod."
        echo "!!! If you are certain it is dead: rmdir $OUT/.lock"
    else
        echo "!!! could not create $OUT/.lock — $OUT is not writable."
    fi
    exit 1
fi
# Releases the venv-build lock too, but ONLY if this process is the one holding it — an
# unconditional rmdir here would free a lock another run on this pod is relying on.
trap 'rmdir "$OUT/.lock" 2>/dev/null; [ "$HELD_VENV_LOCK" = 1 ] && rmdir "$VENV_LOCK" 2>/dev/null' EXIT

echo "### $MODEL — darts calibration — started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── 0. preflight: everything knowable before spending money ───────────────────────
stage preflight
FAIL=0
note() { echo "  MISSING: $*"; FAIL=1; }

[ -d "$REPO/.git" ] || note "$REPO is not a checkout — clone views-models first (docs/runpod_run_guide.md Phase 2)"
[ -d "$REPO/models/$MODEL" ] || note "no such model: $REPO/models/$MODEL"
# The deliverable depends on #533. Without it the run trains for hours and then cannot produce
# the parquets the research team asked for — the same late failure as #517, one stage further on.
[ -f "$REPO/tools/collapse/collapse_darts_predictions.py" ] \
  || note "$REPO/tools/collapse/collapse_darts_predictions.py — the converter (#533) is not in this checkout"

# The datafactory credential, and the ONLY credential this run needs. HTTP Basic from ~/.netrc
# in the runtime user's home, resolved at call time; VIEWS_DATAFACTORY is a phantom no code
# reads. /root is local disk on purpose: /workspace is a network filesystem that silently
# ignores chmod, so 600 does not hold there (register C-154, #518).
[ -s /root/.netrc ] || note "/root/.netrc — the datafactory fetch needs it (guide Step 2.3)"
if [ -s /root/.netrc ] && [ "$(stat -c %a /root/.netrc 2>/dev/null)" != "600" ]; then
    chmod 600 /root/.netrc
    echo "  fixed: /root/.netrc was not 600"
fi

nvidia-smi -L || note "no GPU visible — r2darts2 hardcodes accelerator: gpu and fails at model init"

# Disk. Derived for a num_samples:1 model at global pgm: the venv and apt ~5GB; the datafactory
# cache parquet 0.6-14GB depending on whether the model declares 3 covariates or ~71; the
# prediction scratch 13 origins x 3 targets x 36 x 64,818 x 4 bytes ~0.4GB plus a transient
# Zarr of the same size; the 13 run parquets and the 13 delivery parquets ~1.5GB; the artifact.
# 50 leaves room for the 71-covariate class, which nobody has measured.
# REVISE THIS when #536 permits num_samples > 1: the scratch term scales linearly with it.
DISK_FLOOR_GB=50
AVAIL_GB=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -z "$AVAIL_GB" ]; then
    note "cannot read free space on $ROOT"
elif [ "$AVAIL_GB" -lt "$DISK_FLOOR_GB" ]; then
    note "only ${AVAIL_GB}GB free on $ROOT; this run needs >= ${DISK_FLOOR_GB}GB"
else
    echo "  free on $ROOT: ${AVAIL_GB}GB (floor ${DISK_FLOOR_GB}GB)"
fi

# The config gate. config_hyperparameters.py and config_meta.py are plain dicts with no
# imports, so they load under the system python and this check works on a FRESH pod, before
# the venv exists. config_queryset.py imports datafactory_query, so the REGION check can only
# run once the environment is built — it is deferred below rather than skipped silently.
PYCHECK="$VENV/bin/python"
[ -x "$PYCHECK" ] || PYCHECK=$(command -v python3)
if [ -n "$PYCHECK" ] && [ -d "$REPO/models/$MODEL" ]; then
    MODEL="$MODEL" "$PYCHECK" - "$REPO/models/$MODEL" <<'CFGCHECK' || FAIL=1
import os, sys
from pathlib import Path

model = Path(sys.argv[1])


def load(name):
    """Compile and EXECUTE a config from SOURCE. Never pattern-match the file text.

    A text check here would be the defect this repo has already shipped once (#501, "the guard
    that was not one"): a substring assertion satisfied by a COMMENT recording the value's
    history. These configs are executable Python — the value can only be known by running it.

    And not via `spec_from_file_location` either, which is what the sibling script uses:
    `exec_module` reuses `__pycache__` when the cached bytecode's recorded source SIZE and
    mtime still match, and config edits routinely keep the size identical
    (`"num_samples": 1,` and `"num_samples": 5,` are the same length). A pod that has already
    run a model, then pulled a changed config, could be gated on the OLD value with nothing to
    show for it. Registered as C-156; this is the compile-from-source form that cannot.
    """
    path = model / "configs" / (name + ".py")
    ns = {"__file__": str(path), "__name__": "_podrun_" + name}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    return type("Cfg", (), ns)


problems = []
hp = load("config_hyperparameters").get_hp_config()
meta = load("config_meta").get_meta_config()

epochs = hp.get("n_epochs")
print("n_epochs:", epochs, "(read from the config, not the file text)")
if epochs != 300:
    problems.append(
        "n_epochs is %r, expected 300. The eleven target models all declare 300; a different\n"
        "    value means a stale config or a sweep leftover, and this pod would train a model\n"
        "    nobody asked for." % (epochs,)
    )

# The delivery gate. All three are checked TOGETHER because they are one decision: this
# chain delivers point estimates, and the two sample models (little_talks, mister_bluesky)
# are configured for 100 MC-dropout samples with their point metrics commented out.
#
# Measured, not assumed: the engine converts every prediction to a list-in-cell DataFrame and
# the evaluation path materialises ALL 13 rolling origins before releasing any of them
# (darts_forecasting_model_manager.py:353-362). At 100 samples that is ~303 GB of Python
# objects against a pod rule of RAM >= 50 GB. At 1 sample it is ~11 GB.
#
# And `pred_type` is derived from the DATA, not the config (views-evaluation
# native_evaluator.py:258, `"sample" if n_samples > 1 else "point"`), so dropping num_samples
# to 1 without re-activating regression_point_metrics makes the run train to completion and
# THEN raise "No metrics configured for (regression, point)", writing no predictions at all.
# Refusing here costs seconds; discovering it costs the whole run.
samples = hp.get("num_samples")
dropout = hp.get("mc_dropout")
point_metrics = meta.get("regression_point_metrics") or []
print("num_samples:", samples, "| mc_dropout:", dropout,
      "| regression_point_metrics:", len(point_metrics))

if samples != 1:
    problems.append(
        "num_samples is %r, expected 1 for a point delivery. At 100 the list-in-cell\n"
        "    conversion needs ~303 GB of RAM (all 13 origins are held at once), which no\n"
        "    rentable pod has. views-models#536 decides what these models run at; until it\n"
        "    lands this script refuses them rather than discovering it after training."
        % (samples,)
    )
if dropout:
    problems.append(
        "mc_dropout is %r, expected False. With one sample a stochastic pass gives a single\n"
        "    dropout-perturbed value rather than the deterministic point estimate the other\n"
        "    nine models produce, so the delivery would not be comparable across models."
        % (dropout,)
    )
if not point_metrics:
    problems.append(
        "regression_point_metrics is empty. With num_samples=1 the evaluator reads exactly\n"
        "    this list, finds nothing, and raises AFTER the full training run — writing no\n"
        "    predictions. See views-models#536."
    )

if meta.get("prediction_format") != "dataframe":
    problems.append(
        "prediction_format is %r, expected 'dataframe'. The 'prediction_frame' path writes\n"
        "    predictions_<run_type>_<ts>/ DIRECTORIES of numpy, which this script's converter\n"
        "    does not read — collapse_predictions.py does (views-models#492)."
        % (meta.get("prediction_format"),)
    )
if meta.get("level") != "pgm":
    problems.append("level is %r, expected 'pgm'" % (meta.get("level"),))
# #504: the engine defaults the prediction index to country_id whatever the level, and
# CorePredictionSniffer then refuses every origin at pgm. The declaration is load-bearing, and
# a run on fimbulthul already reported PASS while writing nothing because of it.
if meta.get("entity_id") != "priogrid_id":
    problems.append(
        "entity_id is %r, expected 'priogrid_id'. views-r2darts2 defaults it to country_id at\n"
        "    any level, so this declaration is what stops the sniffer refusing all 13 origins\n"
        "    (views-models#504, views-r2darts2#55)." % (meta.get("entity_id"),)
    )

for p in problems:
    print("  MISSING: " + p)
sys.exit(1 if problems else 0)
CFGCHECK
else
    note "cannot run the config check (no python3, or the model directory is absent)"
fi

if [ -x "$VENV/bin/python" ]; then
    REGION=$(cd "$REPO" && "$VENV/bin/python" -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('_q', 'models/$MODEL/configs/config_queryset.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(getattr(m, 'REGION', None))" 2>/dev/null)
    echo "  REGION: ${REGION:-<unreadable>}"
    [ "$REGION" = "land" ] || note "REGION is '${REGION:-<unreadable>}', expected 'land' — this would not be a global-pgm run"
else
    echo "  deferred until the environment exists: REGION (config_queryset imports datafactory_query)"
fi

echo "  this run uploads nothing and reads no publish credential (ADR-024)"

if [ "$FAIL" != "0" ]; then
    die "preflight failed — nothing above costs GPU time to fix."
fi
echo "preflight clean"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
    # PREFLIGHT_OK, not OK. STATUS is the file an orchestrator reads to decide whether a
    # model's output may be used — pod_run_fao_delivery.sh already does exactly that with
    # pod_run_model.sh's STATUS, to refuse pooling a partial roster. A preflight that wrote
    # OK would be indistinguishable from a finished run that produced parquets, and #538 runs
    # ten of these in sequence. Only the end of this script may write OK.
    echo PREFLIGHT_OK > "$OUT/STATUS"
    stage preflight_only_done
    echo "### --preflight only: nothing was run, nothing was written but this status."
    exit 0
fi

# ── 1. environment (skipped if a previous run on this pod built it) ───────────────
if [ ! -x "$VENV/bin/python" ]; then
  # $VENV is shared by every model on this pod, but the lock above is per MODEL — so two
  # models started together on a fresh pod would both enter this block and corrupt each
  # other's build. Serialise it, and re-check after acquiring: the run we waited for has
  # almost certainly built it.
  stage await_environment
  WAITED=0
  while ! mkdir "$VENV_LOCK" 2>/dev/null; do
      [ -x "$VENV/bin/python" ] && break
      [ "$WAITED" -ge 1800 ] && die "waited 30 min for $VENV_LOCK; if you are certain that build is dead: rmdir $VENV_LOCK"
      sleep 10
      WAITED=$((WAITED + 10))
      echo "another run on this pod is building $VENV — waited ${WAITED}s"
  done
  [ -d "$VENV_LOCK" ] && HELD_VENV_LOCK=1

  if [ ! -x "$VENV/bin/python" ]; then
    stage install_system
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq libpq-dev build-essential zstd rsync || die "apt failed"

    stage install_python
    uv venv --python 3.11 "$VENV" || die "venv creation failed"
    # From the GIT TAG, not PyPI. PyPI's latest is 0.2.3, and 0.2.3 NEVER FREES the prediction
    # scratch directory — ~4 000 of them at ~53 GB each filled fimbulthul's 2 TB disk on
    # 2026-09-20 (views-r2darts2#54). 0.2.4 adds PredictionScratch with an atexit backstop and
    # releases it on the dataframe path, and makes a failed device restore raise instead of
    # finishing silently on CPU (#41). It is tagged and deliberately NOT published: releasing
    # it is irreversible and other repos resolve against the range.
    #
    # [manager] is not optional. views-pipeline-core is `optional = true` in r2darts2's
    # pyproject and reaches the environment ONLY through that extra; without it main.py's first
    # import fails with ModuleNotFoundError: No module named 'views_pipeline_core'.
    uv pip install --python "$VENV/bin/python" \
        "views_r2darts2[manager] @ git+https://github.com/views-platform/views-r2darts2@0.2.4" \
        "views-datafactory>=1.9.0,<2.0.0" || die "pip install failed"
    # register C-151: viewser pins toolz<0.12, which cannot import tlz submodules on Python
    # 3.11 and breaks EVERY datafactory fetch. Override after resolution.
    #
    # This MUST stay the last install in this block. Any pip install appended below it
    # re-resolves this prefix and can silently pull toolz back under 0.12 — which happened on
    # 2026-09-29, reverting it 1.1.0 -> 0.11.2 with no error.
    uv pip install --python "$VENV/bin/python" "toolz>=0.12.1" || die "toolz override failed"
  else
    echo "another run built $VENV while we waited — reusing it"
  fi
  rmdir "$VENV_LOCK" 2>/dev/null && HELD_VENV_LOCK=0
else
  echo "venv already present — reusing"
fi

stage verify_env
"$VENV/bin/python" - <<'PY' || die "environment verification failed"
import importlib.metadata as md

import torch, tlz.curried, views_r2darts2, views_pipeline_core, datafactory_query, darts  # noqa: F401

# The version is ASSERTED, not assumed. The install above pins a git tag, and a silent
# fallback to PyPI's 0.2.3 would reintroduce the scratch leak that filled a 2 TB disk — which
# does not fail, it just fills the machine hours later.
version = md.version("views_r2darts2")
assert version == "0.2.4", (
    "views_r2darts2 %s is installed, expected 0.2.4. 0.2.3 never frees the prediction scratch "
    "directory (views-r2darts2#54) and will fill this pod's disk mid-run." % version
)
print("views_r2darts2", version, "| darts", darts.__version__)

# Not just `is_available()`. darts pins torch>=2.0.0 with NO upper bound, so a fresh env can
# resolve a CUDA build newer than the machine's driver (views-models#494). That reports
# available and then fails on the first kernel launch — which, since r2darts2 hardcodes
# accelerator: gpu, happens at model init after the data fetch. Launch a real kernel here.
assert torch.cuda.is_available(), "CUDA not available"
probe = torch.ones(8, device="cuda")
assert float((probe @ probe).item()) == 8.0, "a CUDA kernel ran and gave the wrong answer"
print("torch", torch.__version__, "cap", torch.cuda.get_device_capability(), "— a real kernel ran")

import pandas, numpy
print("pandas", pandas.__version__, "numpy", numpy.__version__)
PY

# ── 2. repo ───────────────────────────────────────────────────────────────────────
if [ ! -d "$REPO/.git" ]; then
  stage clone
  git clone --depth 1 -b development https://github.com/views-platform/views-models.git "$REPO" \
    || die "clone failed"
fi
stage check_checkout
[ -d "$REPO/models/$MODEL" ] || die "no such model: models/$MODEL"
[ -f "$REPO/tools/collapse/collapse_darts_predictions.py" ] \
  || die "tools/collapse/collapse_darts_predictions.py is not in this checkout (#533)"

stage check_config
# Re-run the gate under the venv, which adds the REGION check the preflight had to defer.
REGION=$(cd "$REPO" && "$VENV/bin/python" -c "
import importlib.util
spec = importlib.util.spec_from_file_location('_q', 'models/$MODEL/configs/config_queryset.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(getattr(m, 'REGION', None))") || die "cannot read config_queryset.py"
echo "REGION: $REGION"
[ "$REGION" = "land" ] || die "REGION is '$REGION', expected 'land' — this would not be a global-pgm run"

# ── 3. the run ────────────────────────────────────────────────────────────────────
stage train_and_evaluate
# The prediction scratch honours TMPDIR and otherwise lands in /tmp — the CONTAINER disk,
# while the disk floor above measures $ROOT, the volume. A floor that measures a filesystem
# the workload does not use cannot fire. PredictionScratch() passes no base_dir, so this is
# the only lever (views_r2darts2/transformers/frame_builder.py).
mkdir -p "$ROOT/tmp" || die "cannot create $ROOT/tmp"
export TMPDIR="$ROOT/tmp"
echo "TMPDIR=$TMPDIR (the volume, not the container disk)"

cd "$REPO/models/$MODEL" || die "cannot enter model dir"
# WANDB_MODE=offline is NOT optional. Without it main.py calls wandb.login(), which blocks on
# an interactive prompt no one is watching, and a forecasting run on a pod already died there
# after the environment was built.
export WANDB_MODE=offline WANDB_SILENT=true
START=$(date +%s)
"$VENV/bin/python" main.py -r calibration -t -e || die "main.py exited non-zero"
RUN_MIN=$(( ($(date +%s) - START) / 60 ))
echo "run took ${RUN_MIN} minutes"

# Peak scratch, for the MANIFEST. #537 exists to measure this, and a number nobody wrote down
# is a number the next pod has to rediscover.
SCRATCH_PEAK=$(du -sh "$TMPDIR" 2>/dev/null | cut -f1)

# ── 4. the deliverable ────────────────────────────────────────────────────────────
stage collapse
# Clear a previous attempt first: parquets are named from the SOURCE run's timestamp, so an old
# set and a new set can coexist, and if they happened to sum to 13 the count check would pass
# while the manifest covered two different training runs.
rm -rf "$OUT/parquet"
mkdir -p "$OUT/parquet"
# From the REPO root. `python -m tools.collapse...` resolves `tools` from the current directory
# and nothing is installed — the equivalent slip in the FAO script left the shell in the
# ensemble directory and every tool call raised ModuleNotFoundError while an `|| echo` fallback
# reported that its SUBJECT was broken.
cd "$REPO" || die "cannot enter repo"
"$VENV/bin/python" -m tools.collapse.collapse_darts_predictions \
    "models/$MODEL" --run-type calibration --out-dir "$OUT/parquet" || die "collapse failed (#533)"
N=$(ls -1 "$OUT/parquet"/*.parquet 2>/dev/null | wc -l)
[ "$N" -eq 13 ] || die "expected 13 parquets, got $N"

stage verify_parquet
"$VENV/bin/python" - "$OUT/parquet" <<'PY' || die "parquet verification failed"
import sys, glob, pandas as pd, numpy as np

files = sorted(glob.glob(sys.argv[1] + "/*.parquet"))
assert len(files) == 13, f"{len(files)} parquets, expected 13"
total = 0
for f in files:
    d = pd.read_parquet(f)
    preds = [c for c in d.columns if c.startswith("pred_")]
    assert list(d.columns)[:2] == ["month_id", "priogrid_id"], f"keys are not the first columns: {f}"
    assert d["month_id"].dtype == "int64" and d["priogrid_id"].dtype == "int64", f
    assert len(preds) == 3, f"{len(preds)} pred_* columns in {f}, expected 3"
    assert len(d) == 2_333_448, f"{len(d)} rows in {f}, expected 2,333,448"
    assert not d.duplicated(["month_id", "priogrid_id"]).any(), f"duplicate keys in {f}"
    v = d[preds].to_numpy()
    # The whole point of the converter: one scalar per cell, never a list. A list cell survives
    # to_numpy as dtype=object, and ensemble-updater would silently take its first element.
    assert v.dtype.kind == "f", f"{f} holds {v.dtype}, not floats — a list cell got through"
    assert np.isfinite(v).all(), f"non-finite in {f}"
    assert (v >= 0).all(), f"negative in {f}"
    total += len(d)
print(f"{len(files)} parquets, {total:,} rows, scalar floats, unique keys, finite, non-negative")
PY

stage manifest
{
  echo "model:          $MODEL"
  echo "finished:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_type:       calibration"
  echo "engine:         views_r2darts2 $("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("views_r2darts2"))' 2>/dev/null || echo unknown) (git tag 0.2.4)"
  echo "region:         $REGION"
  echo "runtime_min:    $RUN_MIN"
  echo "scratch_peak:   ${SCRATCH_PEAK:-unknown} (in $TMPDIR)"
  echo "parquets:       $(du -sh "$OUT/parquet" | cut -f1)"
  echo "git:            $(git -C "$REPO" rev-parse --short HEAD)"
  echo "config:         as committed at the git sha above"
  echo "published:      nothing — this runner uploads nothing and holds no publish credential"
} > "$OUT/MANIFEST"
cat "$OUT/MANIFEST"

echo OK > "$OUT/STATUS"
stage done
echo "### $MODEL — COMPLETE"

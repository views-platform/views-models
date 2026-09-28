#!/usr/bin/env bash
# pod_run_model.sh <model_name>
#
# ┌──────────────────────────────────────────────────────────────────────────────────────┐
# │  VERSION 0.1.0 — PROVISIONAL.  NOT part of the monthly production run.                │
# │  Exercised on ONE model family (HydraNet) and ONE partition (calibration), during the │
# │  first rented-GPU deployment (2026-09-28). No CI runs it. It uploads nothing.         │
# │  See tools/podrun/__init__.py for what promoting it out of 0.1.0 would require.       │
# └──────────────────────────────────────────────────────────────────────────────────────┘
#
# One HydraNet, calibration partition, global land, on a rented pod — from a bare
# runpod/pytorch image to parquets a researcher can open, without a person watching.
#
# It is written to fail LOUDLY and EARLY. Every expensive step is preceded by a check
# that costs seconds, because the failure mode that matters here is discovering after
# seven hours of paid GPU time that a credential was missing or a disk was full.
#
# Leaves behind:
#   /workspace/deliver/<model>/parquet/    13 point-prediction parquets  (~9 MB)
#   /workspace/deliver/<model>/draws/      the lr_* posterior, zstd      (~25 MB)
#   /workspace/deliver/<model>/STATUS      OK or FAILED:<stage>
#   /workspace/deliver/<model>/FAILURE     the reason, on failure only (written directly)
#   /workspace/deliver/<model>/run.log     the whole transcript
#
# Progress is readable from outside at any time:  cat /workspace/deliver/<model>/STAGE

set -uo pipefail

MODEL="${1:?usage: pod_run_model.sh <model_name>}"
ROOT=/workspace
REPO=$ROOT/views-models
VENV=$ROOT/venv
OUT=$ROOT/deliver/$MODEL
LOG=$OUT/run.log

mkdir -p "$OUT"
# A previous attempt on this pod may have left FAILED here. Clear it before anything else,
# or `cat STATUS` reports that old failure for the whole of this run -- and the header
# advertises STATUS/STAGE as the way to watch from outside, so a stale one actively misleads.
rm -f "$OUT/STATUS"
exec > >(tee -a "$LOG") 2>&1

stage() { echo "$1" > "$OUT/STAGE"; echo "=== [$(date +%H:%M:%S)] $1 ==="; }
die()   {
    echo "FAILED:$(cat "$OUT/STAGE" 2>/dev/null)" > "$OUT/STATUS"
    # Written directly, not through the tee subshell: on a hard kill (preemption, OOM)
    # the last buffered log lines can be lost, and that is exactly when the reason matters.
    printf '%s\n' "$1" > "$OUT/FAILURE"
    echo "!!! $1"
    exit 1
}

if ! mkdir "$OUT/.lock" 2>/dev/null; then
    echo "!!! $OUT/.lock exists -- another run for $MODEL is in progress on this pod."
    echo "!!! If you are certain it is dead: rmdir $OUT/.lock"
    exit 1
fi
trap 'rmdir "$OUT/.lock" 2>/dev/null' EXIT

echo "### $MODEL — started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── 0. preflight: everything that can be known before spending money ──────────────
stage preflight
[ -s /root/.netrc ] || die "/root/.netrc missing — the datafactory fetch would fail after setup"
[ "$(stat -c %a /root/.netrc)" = "600" ] || chmod 600 /root/.netrc
nvidia-smi -L || die "no GPU visible"
AVAIL_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
[ "$AVAIL_GB" -ge 40 ] || die "only ${AVAIL_GB}GB free on $ROOT; one model needs ~20GB"
echo "free on $ROOT: ${AVAIL_GB}GB"

# ── 1. environment (skipped if a previous run on this pod built it) ───────────────
if [ ! -x "$VENV/bin/python" ]; then
  stage install_system
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq libpq-dev build-essential zstd rsync || die "apt failed"

  stage install_python
  uv venv --python 3.11 "$VENV" || die "venv creation failed"
  uv pip install --python "$VENV/bin/python" \
      "views-hydranet~=0.1.1" "views-datafactory>=1.9.0,<2.0.0" || die "pip install failed"
  # register C-151: viewser pins toolz<0.12, which cannot import tlz submodules on
  # Python 3.11 and breaks EVERY datafactory fetch. Override after resolution.
  uv pip install --python "$VENV/bin/python" "toolz>=0.12.1" || die "toolz override failed"
else
  echo "venv already present — reusing"
fi

stage verify_env
"$VENV/bin/python" - <<'PY' || die "environment verification failed"
import torch, tlz.curried, views_hydranet, views_pipeline_core, datafactory_query
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "cap", torch.cuda.get_device_capability())
PY

# ── 2. repo ───────────────────────────────────────────────────────────────────────
if [ ! -d "$REPO/.git" ]; then
  stage clone
  git clone --depth 1 -b development https://github.com/views-platform/views-models.git "$REPO" \
    || die "clone failed"
fi
stage check_checkout
[ -d "$REPO/models/$MODEL" ] || die "no such model: models/$MODEL"
[ -f "$REPO/tools/collapse/collapse_predictions.py" ] \
  || die "tools/collapse is not in this checkout — copy it to the pod before running (PR #506)"

stage check_config
# Load and CALL the config rather than pattern-matching the file. A text check here would be
# the same defect this repo has already shipped once (#501, "the guard that was not one"): a
# substring assertion satisfied by a COMMENT recording the value's history, so the guard passed
# on the wrong region. A comment cannot satisfy this one.
"$VENV/bin/python" - "$REPO/models/$MODEL" <<'CFGCHECK' || die "config check failed"
import importlib.util, sys
from pathlib import Path

model = Path(sys.argv[1])

def load(name):
    path = model / "configs" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_podrun_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

hp = load("config_hyperparameters").get_hp_config()
lessons = hp["total_lessons"]
print("total_lessons:", lessons, "(read from the config, not the file text)")
if lessons < 300:
    sys.exit("total_lessons is %s, expected >= 300 - this pod would train a throwaway model" % lessons)

region = getattr(load("config_queryset"), "REGION", None)
print("REGION:", region)
if region != "land":
    sys.exit("REGION is %r, expected 'land' - this would not be a global-land run" % region)
CFGCHECK

# ── 3. the run ────────────────────────────────────────────────────────────────────
stage train_and_evaluate
cd "$REPO/models/$MODEL" || die "cannot enter model dir"
export WANDB_MODE=offline WANDB_SILENT=true
START=$(date +%s)
"$VENV/bin/python" main.py -r calibration -t -e || die "main.py exited non-zero"
echo "run took $(( ($(date +%s) - START) / 60 )) minutes"

# ── 4. collapse to the deliverable ────────────────────────────────────────────────
stage collapse
# Clear a previous attempt first. Parquets are named from the SOURCE run's timestamp, so an
# old set and a new set can coexist; if they happened to sum to 13 the count check below
# would pass while the manifest covered two different training runs.
rm -rf "$OUT/parquet"
mkdir -p "$OUT/parquet"
cd "$REPO" || die "cannot enter repo"
"$VENV/bin/python" -m tools.collapse.collapse_predictions \
    "models/$MODEL" --run-type calibration --out-dir "$OUT/parquet" || die "collapse failed"
N=$(ls -1 "$OUT/parquet"/*.parquet 2>/dev/null | wc -l)
[ "$N" -eq 13 ] || die "expected 13 parquets, got $N"

stage verify_parquet
"$VENV/bin/python" - "$OUT/parquet" <<'PY' || die "parquet verification failed"
import sys, glob, pandas as pd, numpy as np
fs = sorted(glob.glob(sys.argv[1] + "/*.parquet"))
total = 0
for f in fs:
    d = pd.read_parquet(f)
    p = [c for c in d.columns if c.startswith("pred_")]
    assert list(d.columns)[:2] == ["month_id", "priogrid_id"], f
    assert len(p) == 3, f
    assert not d.duplicated(["month_id", "priogrid_id"]).any(), f"duplicate keys in {f}"
    v = d[p].to_numpy()
    assert np.isfinite(v).all(), f"non-finite in {f}"
    assert (v >= 0).all(), f"negative in {f}"
    total += len(d)
print(f"{len(fs)} parquets, {total:,} rows, all keys unique, all finite, all non-negative")
PY

# ── 5. compress the posterior (236x on real output — the draws come home too) ─────
stage compress_draws
# Clear any archive from a previous attempt on this pod, so a stale one cannot be counted
# as this run's output.
rm -rf "$OUT/draws"
mkdir -p "$OUT/draws"

SRC=$(ls -d "$REPO/models/$MODEL/data/generated/predictions_calibration_"* 2>/dev/null | tail -1)
[ -n "$SRC" ] || die "no predictions_calibration_* directory under $REPO/models/$MODEL/data/generated"
[ -d "$SRC" ] || die "$SRC is not a directory"
BASE=$(basename "$SRC")

# COUNT FIRST. `find ... -print0 | tar --null -T -` exits 0 and writes a VALID ~22-byte archive
# when find matches nothing, so the obvious pipeline reports success while shipping an empty
# posterior. The only other signal would be a small number in MANIFEST that a human has to
# notice. That is the failure this block exists to make impossible.
N_DRAWS=$(cd "$SRC/.." && find "$BASE" -path '*/lr_*' -name '*.np*' | wc -l)
[ "$N_DRAWS" -gt 0 ] || die "no lr_* draw files under $SRC — the layout is not what this script expects"

( cd "$SRC/.." && find "$BASE" -path '*/lr_*' -name '*.np*' -print0 \
    | tar -I 'zstd -3 -T0' -cf "$OUT/draws/${BASE}_lr.tar.zst" --null -T - ) \
  || die "compressing draws failed"

# And verify the archive actually holds them, rather than trusting tar's exit code.
N_ARCHIVED=$(tar -I zstd -tf "$OUT/draws/${BASE}_lr.tar.zst" | wc -l)
[ "$N_ARCHIVED" -eq "$N_DRAWS" ] \
  || die "archive holds $N_ARCHIVED entries but $N_DRAWS draw files were found"
echo "draws archived: $N_ARCHIVED files"

stage manifest
{
  echo "model:        $MODEL"
  echo "finished:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source:       $(basename "$SRC")"
  echo "raw draws:    $(du -sh "$SRC" | cut -f1)"
  echo "compressed:   $(du -sh "$OUT/draws" | cut -f1)"
  echo "parquets:     $(du -sh "$OUT/parquet" | cut -f1)"
  echo "git:          $(git -C "$REPO" rev-parse --short HEAD)"
  echo "lessons:      $(grep -oE "'total_lessons': [0-9]+" "$REPO/models/$MODEL/configs/config_hyperparameters.py" | head -1)"
} > "$OUT/MANIFEST"
cat "$OUT/MANIFEST"

echo OK > "$OUT/STATUS"
stage done
echo "### $MODEL — COMPLETE"

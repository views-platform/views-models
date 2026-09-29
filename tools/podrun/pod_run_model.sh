#!/usr/bin/env bash
# pod_run_model.sh [--rehearsal] <model_name>
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
#   /workspace/deliver/<model>/REHEARSAL   present ONLY on a --rehearsal run
#
# TWO MODES.
#   (default)          production. total_lessons must be >= 300, and the run refuses otherwise.
#   --rehearsal <n>    exercises the whole chain on a deliberately undertrained model at <n>
#                      lessons. Patches the POD's clone of the config — never a tracked file
#                      — and marks the output as unfit to deliver.
#
# The lesson floor exists because a 300-lesson budget can be spent by ACCIDENT — a config
# left at a sweep value, a stale checkout. It was never meant to forbid a deliberate cheap
# end-to-end test, which is a routine and necessary thing to want, especially after a run
# that failed late. Before --rehearsal existed the only way to get one was to edit the guard
# out, which produced output indistinguishable from a real run: the cheap test and the
# accident looked the same on disk. That is the actual hazard, and it is what REHEARSAL and
# the MANIFEST `mode:` line address. Wasted GPU time is recoverable; an undertrained
# forecast reaching a partner as a real one is not.
#
# Progress is readable from outside at any time:  cat /workspace/deliver/<model>/STAGE

set -uo pipefail

# --rehearsal takes the lesson count rather than reading it from the config, so a rehearsal
# needs NO edit to a tracked file. main.py has no hyperparameter override, so the count has
# to reach the model through its config; this script patches the POD's ephemeral clone and
# verifies the patch (see the config check). The count is REQUIRED — there is exactly one way
# to ask for a rehearsal, and it states the number out loud in the command that starts it.
USAGE='usage: pod_run_model.sh [--rehearsal <lessons>] <model_name>'
REHEARSAL_LESSONS=""
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
        --*) echo "!!! unknown option: $1" >&2; echo "!!! $USAGE" >&2; exit 1 ;;
        *)   break ;;
    esac
done
MODEL="${1:?$USAGE}"
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
# Same reasoning for the rehearsal marker: a production run in a directory left behind by an
# earlier rehearsal must not inherit its "undeliverable" mark, and — far worse — a rehearsal
# must not inherit a previous production run's ABSENCE of one.
rm -f "$OUT/REHEARSAL" "$OUT/.lessons"
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
  # views-pipeline-core[appwrite] is requested EXPLICITLY, with no version, so
  # views-hydranet's own range still decides which pipeline-core is installed. The extra is
  # what carries the Appwrite SDK, and without it `_build_datastore` raises at publish time
  # — which is AFTER the full training run. That is exactly how the first FAO delivery
  # attempt failed on 2026-09-29 (views-models#517); it was fixed by hand on a pod that no
  # longer exists, so the repository never learned it.
  uv pip install --python "$VENV/bin/python" \
      "views-hydranet~=0.1.1" "views-datafactory>=1.9.0,<2.0.0" \
      "views-pipeline-core[appwrite]" || die "pip install failed"
  # register C-151: viewser pins toolz<0.12, which cannot import tlz submodules on
  # Python 3.11 and breaks EVERY datafactory fetch. Override after resolution.
  #
  # This MUST stay the last install in this block. Any pip install appended below it
  # re-resolves this prefix and can silently pull toolz back under 0.12 — which happened on
  # 2026-09-29, when installing the appwrite extra by hand reverted it 1.1.0 -> 0.11.2 with
  # no error. Pinned by tests/test_falsification_40_lesson_run_readiness.py.
  uv pip install --python "$VENV/bin/python" "toolz>=0.12.1" || die "toolz override failed"
else
  echo "venv already present — reusing"
fi

stage verify_env
"$VENV/bin/python" - <<'PY' || die "environment verification failed"
import torch, tlz.curried, views_hydranet, views_pipeline_core, datafactory_query
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "cap", torch.cuda.get_device_capability())

# The publish path is verified HERE, in preflight, not discovered at publish time. Without
# the appwrite extra this import is the only thing between a green-looking pod and a run
# that trains for hours and then cannot hand over its forecasts (#517). Importing the
# client is a weaker check than publishing, but it is the strongest one available before
# there is anything to publish — and it is the check whose absence cost the first delivery.
import appwrite  # noqa: F401  — the SDK itself; views-pipeline-core[appwrite] provides it
from views_pipeline_core.modules.appwrite import file as _appwrite_file  # noqa: F401
print("appwrite client importable — the publish path exists")

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
[ -f "$REPO/tools/collapse/collapse_predictions.py" ] \
  || die "tools/collapse is not in this checkout — copy it to the pod before running (PR #506)"

stage check_config
# Load and CALL the config rather than pattern-matching the file. A text check here would be
# the same defect this repo has already shipped once (#501, "the guard that was not one"): a
# substring assertion satisfied by a COMMENT recording the value's history, so the guard passed
# on the wrong region. A comment cannot satisfy this one.
# REHEARSAL/OUT/MODEL reach the script through the environment: the heredoc is quoted, so
# the shell does not interpolate into it, and that is deliberate — the config values must
# come from the config file, not from string substitution.
REHEARSAL_LESSONS="$REHEARSAL_LESSONS" OUT="$OUT" MODEL="$MODEL" \
"$VENV/bin/python" - "$REPO/models/$MODEL" <<'CFGCHECK' || die "config check failed"
import importlib, importlib.util, os, re, subprocess, sys
from pathlib import Path

model = Path(sys.argv[1])

def load(name):
    path = model / "configs" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_podrun_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cfg_path = model / "configs" / "config_hyperparameters.py"
requested = os.environ.get("REHEARSAL_LESSONS") or ""

lessons = load("config_hyperparameters").get_hp_config()["total_lessons"]
print("total_lessons:", lessons, "(read from the config, not the file text)")

if not requested:
    # A rehearsal patches this pod's clone, and section 2 does NOT re-clone when .git already
    # exists — so a production run started on a pod that has rehearsed reads the LEFTOVER
    # patch. It would refuse (the floor holds), but it would blame the committed config and
    # tell the operator to use --rehearsal, which is the opposite of what they want. Name the
    # real cause instead. Checked before the floor so the accurate message wins.
    dirty = subprocess.run(
        ["git", "-C", str(model), "status", "--porcelain", "--", str(cfg_path)],
        capture_output=True, text=True,
    )
    if dirty.returncode == 0 and dirty.stdout.strip():
        sys.exit(
            "%s is MODIFIED in this checkout, so total_lessons=%s is not what the committed\n"
            "  config says. This pod has almost certainly run --rehearsal already, and that\n"
            "  patch is still in place. A production run must start from the committed config:\n"
            "      git -C %s checkout -- %s\n"
            "  Refusing rather than training on a config neither of us chose."
            % (cfg_path, lessons, model, cfg_path.relative_to(model))
        )

if requested:
    # Patch the POD's clone. This is a throwaway checkout on rented hardware; the tracked
    # config keeps saying 300, which is the production truth and must not be edited to get a
    # cheap test. Substitution is anchored on the same literal the file is known to contain.
    target = int(requested)
    text = cfg_path.read_text()
    patched, n = re.subn(r"('total_lessons'\s*:\s*)\d+", r"\g<1>%d" % target, text)
    if n != 1:
        sys.exit(
            "--rehearsal %d: expected exactly one 'total_lessons': <int> in %s, found %d.\n"
            "  Refusing rather than guessing which one to patch."
            % (target, cfg_path, n)
        )
    cfg_path.write_text(patched)

    # VERIFY by re-importing, not by trusting the substitution. A patch that silently failed
    # would otherwise produce a 300-lesson run wearing a rehearsal label, or the reverse.
    importlib.invalidate_caches()
    lessons = load("config_hyperparameters").get_hp_config()["total_lessons"]
    if lessons != target:
        sys.exit(
            "--rehearsal %d: patched %s but it still reports total_lessons=%s. Not proceeding."
            % (target, cfg_path, lessons)
        )
    print("REHEARSAL: patched the pod's config to %d lessons and re-read it to confirm." % lessons)
    print("           Output will be marked NOT FIT TO DELIVER.")

# Recorded for the MANIFEST, so the manifest reports the value the run was GATED on rather
# than re-deriving it by grepping the file text. Two readings of one number can disagree.
Path(os.environ["OUT"], ".lessons").write_text(str(lessons))

if not requested and lessons < 300:
    sys.exit(
        "total_lessons is %s, expected >= 300 - this pod would train a throwaway model.\n"
        "  If you MEANT a cheap end-to-end test, that is what --rehearsal is for:\n"
        "      pod_run_model.sh --rehearsal %s %s\n"
        "  It takes the lesson count on the command line, patches only the pod's clone, and\n"
        "  marks the output as unfit to deliver — so a deliberate cheap run cannot be\n"
        "  confused with an accidental one, and no tracked config has to be edited."
        % (lessons, lessons, os.environ.get("MODEL", "<model>"))
    )

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
  # The value the run was GATED on, written by the config check — not a second, independent
  # grep of the file text, which could disagree with it and be believed.
  echo "lessons:      $(cat "$OUT/.lessons" 2>/dev/null || echo unknown)"
  if [ -n "$REHEARSAL_LESSONS" ]; then
    echo "mode:         REHEARSAL — NOT FIT TO DELIVER"
    # Stated explicitly because the `git:` line above no longer fully describes the run: the
    # pod's config_hyperparameters.py was patched after checkout, so that sha alone would
    # imply 300 lessons. A manifest that has to be cross-read with a flag is a manifest that
    # will be misread.
    echo "config:       PATCHED after checkout — total_lessons forced to $REHEARSAL_LESSONS"
  else
    echo "mode:         production"
    echo "config:       as committed at the git sha above"
  fi
} > "$OUT/MANIFEST"
cat "$OUT/MANIFEST"

# ── the mark that makes a rehearsal unmistakable downstream ───────────────────────
# A rehearsal's parquets are structurally identical to a production run's: same columns,
# same row counts, same names, same finite non-negative values. Every check in section 4
# passes. Nothing about the FILES says the model behind them is undertrained, which is why
# this has to be a separate artefact that travels with them.
#
# This MARKS; it cannot REFUSE. The publish step is not in this script (the header is
# accurate: this runner uploads nothing), so the refusal has to live wherever the forecast
# is handed to a store. Until it does, this file is the only thing standing between a
# rehearsal and a partner, and that is a weaker guarantee than it should be — tracked as
# the second half of the rehearsal work, not as done.
if [ -n "$REHEARSAL_LESSONS" ]; then
  {
    echo "THIS OUTPUT IS A REHEARSAL. DO NOT DELIVER IT."
    echo
    echo "model:    $MODEL"
    echo "lessons:  $(cat "$OUT/.lessons" 2>/dev/null || echo unknown)"
    echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "It was produced with --rehearsal to exercise the pipeline end to end. The model is"
    echo "deliberately undertrained. The parquets are well-formed and will pass every"
    echo "structural check, including this runner's own — that is precisely why this file"
    echo "exists. Nothing in the data itself will tell you."
  } > "$OUT/REHEARSAL"
  echo "### WROTE $OUT/REHEARSAL — this output is NOT fit to deliver"
fi

echo OK > "$OUT/STATUS"
stage done
echo "### $MODEL — COMPLETE"

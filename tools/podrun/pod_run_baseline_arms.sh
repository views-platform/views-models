#!/usr/bin/env bash
# pod_run_baseline_arms.sh [--preflight] [--root DIR] [--repo DIR] [--args "..."] [source ...]
#
# ┌──────────────────────────────────────────────────────────────────────────────────────┐
# │  VERSION 0.1.0 — PROVISIONAL.  NOT part of the monthly production run.                │
# │  No CI runs it. It uploads nothing, and it installs no credential that could.         │
# └──────────────────────────────────────────────────────────────────────────────────────┘
#
# N views-baseline sources, calibration partition, global land, on ONE rented pod, in sequence.
# For the #728 comparison set: the ten #695 arms and the eleven #717/#730/#731 sources.
#
# ## WHY THIS EXISTS — no existing runner can run a views-baseline source
#
# Measured 2026-11-13, which is why this is a fourth sibling rather than a flag on one of them:
#
#   pod_run_crafd_chain.sh   MODELS hardcoded to the eight *_condor, ENSEMBLE=chrome_crate
#   pod_run_chrome_box.sh    MODELS hardcoded to the eight *_sparrow, ENSEMBLE=chrome_box
#   pod_run_model.sh         takes any MODEL, but mentions views-baseline ZERO times: it installs
#                            views-hydranet and its verify step imports views_hydranet
#
# ⛔ That last one is worse than absence. Point it at a baseline source and `verify_env` reports a
# HEALTHY environment — views_hydranet IS installed — then main.py dies on
# `from views_baseline.manager.baseline_manager import ...`. A green precondition in front of a run
# that cannot work. This script installs views-baseline and verifies THAT import.
#
# ## ⛔ ARCHIVE-AND-DELETE BETWEEN SOURCES IS NOT HOUSEKEEPING, IT IS THE REASON 21 FIT
#
# views-baseline register C-54: ~180 GB written per model at S=256, 7.17 GB per origin (13 origins
# = 93 GB). At the declared S=64 that is ~23-45 GB per source. A pod quota is 100 GB (measured
# 2026-11-12: / = 100G). So ONE source fits and twenty-one do not — 21 x 23..45 GB is 480..945 GB.
# Each source is archived under deliver/ and its data/generated cleared before the next starts, and
# free space is re-checked each time. Without that this dies at source three with two already paid.
#
# ## ONE STORE STATE FOR THE WHOLE SET
#
# The comparison is between sources, so a difference traceable to two different pulls is not a
# result (the P-30 lesson from views-hydranet's EXP-02). Sources fetch individually, so this checks
# the store's FINGERPRINT once before the loop and refuses if it moved — more honest than trusting
# `queryset_digest`, which digests the CONFIG and not the data (measured: eight members, eight
# different digests, byte-identical files).
#
# ## WHAT IT DOES NOT DO
#
# No ensemble, no pooling, no reconciliation — these sources join no ensemble. Nothing is
# published: guide rule 5, and the preflight refuses if any APPWRITE_* variable is set.
#
# ⚠️ THE ONE THING I COULD NOT VERIFY: the exact main.py invocation. The #695 campaign's archives
# hold `predictions_calibration_<ts>/origin_0..12/<target>/{y_pred.npy,identifiers.npz}` and no
# log, so the invocation is not recoverable from them. `-r calibration -t -e` is INFERRED from that
# output shape — per-origin predictions is an evaluate output over the rolling origins — and is
# overridable with --args. Run ONE source first and check the output shape before the full set.

set -uo pipefail

USAGE='usage: pod_run_baseline_arms.sh [--preflight] [--root DIR] [--repo DIR] [--args "..."] [source ...]'

ROOT="${PODRUN_ROOT:-/workspace}"
REPO="${PODRUN_REPO:-$ROOT/views-models}"
VENV="$ROOT/venv-baseline"
PREFLIGHT_ONLY=0
RUN_ARGS="-r calibration -t -e"
# The output shape a correct calibration run produces, MEASURED from the #695 campaign
# archives, not assumed. Overridable only by editing here — a run that writes a different
# shape is a different experiment and should say so out loud.
EXPECT_ORIGINS=13
EXPECT_TARGETS=3

# The #728 comparison set. Written down rather than globbed: the set is the experiment's
# definition, and a glob would silently admit any future models/* directory.
SOURCES_DEFAULT="plain_burlap short_linen plain_linen long_linen sheer_velvet plush_velvet \
flat_velvet light_tweed even_tweed heavy_tweed nested_burlap lone_burlap local_burlap \
civic_burlap global_burlap mapless_burlap stateless_burlap distant_burlap narrow_canvas \
middle_canvas broad_canvas"

while [ $# -gt 0 ]; do
    case "$1" in
        --preflight) PREFLIGHT_ONLY=1; shift ;;
        --root)      ROOT="$2"; VENV="$ROOT/venv-baseline"; REPO="${PODRUN_REPO:-$ROOT/views-models}"; shift 2 ;;
        --repo)      REPO="$2"; shift 2 ;;
        --args)      RUN_ARGS="$2"; shift 2 ;;
        --*)         echo "!!! unknown option: $1" >&2; echo "!!! $USAGE" >&2; exit 1 ;;
        *)           break ;;
    esac
done
SOURCES="${*:-$SOURCES_DEFAULT}"

OUT="$ROOT/deliver/baseline_arms"
mkdir -p "$OUT" || { echo "!!! cannot create $OUT" >&2; exit 1; }

# STATUS is a verdict and must not outlive its run (C-192): clear the previous one up front, so a
# crash leaves no stale OK behind.
rm -f "$OUT/STATUS" "$OUT/FAILURE"

stage() { echo "$1" > "$OUT/STAGE"; echo "=== [$(date +%H:%M:%S)] $1 ==="; }
die()   { echo "FAILED:$(cat "$OUT/STAGE" 2>/dev/null)" > "$OUT/STATUS"
          printf '%s\n' "$1" > "$OUT/FAILURE"; echo "!!! $1" >&2; exit 1; }

FAIL=0
note() { echo "  MISSING: $*"; FAIL=1; }

stage preflight

[ -d "$REPO/.git" ] || note "$REPO is not a checkout — clone views-models first (guide Phase 2)"
for S in $SOURCES; do
    [ -f "$REPO/models/$S/main.py" ]                || note "no such source: models/$S/main.py"
    [ -f "$REPO/models/$S/configs/config_meta.py" ] || note "models/$S has no config_meta.py"
done

command -v uv >/dev/null 2>&1 || note "uv is not on PATH, and this script builds its venv with it. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"

# ⛔ pg_config, i.e. libpq-dev. MEASURED FAILURE 2026-11-13: this runner died at install_python
# with `Error: pg_config executable not found` after a pod reset wiped the apt packages. The chain
# is not obvious — views-pipeline-core depends on viewser, which depends on views-storage, which
# depends on psycopg2, which builds from source and needs pg_config. So a baseline run fails on a
# POSTGRES build tool it never uses, several minutes into a paid install, and the error names
# psycopg2 rather than the missing apt package.
# ⚠️ Checking `zstd` is NOT a proxy for "the apt packages are there": zstd ships in the RunPod base
# image and libpq-dev does not. That is exactly how this bit — one token checked, a whole class
# assumed (ADR-025: an assumption a check cannot see is how a preflight returns clean on a run
# that cannot start).
command -v pg_config >/dev/null 2>&1 || note "pg_config is not on PATH, so psycopg2 cannot build and the install WILL fail minutes in. Install it: apt-get install -y libpq-dev build-essential (guide Step 2.2). A pod reset wipes this; zstd surviving does not mean it did."
command -v gcc >/dev/null 2>&1 || note "gcc is not on PATH, so no sdist in the dependency tree can build. apt-get install -y build-essential (guide Step 2.2)."

[ -s /root/.netrc ] || note "/root/.netrc — the datafactory fetch needs it (guide Step 2.3)"
if [ -s /root/.netrc ] && [ "$(stat -c %a /root/.netrc 2>/dev/null)" != "600" ]; then
    chmod 600 /root/.netrc; echo "  fixed: /root/.netrc was not 600"
fi

# Guide rule 5, as a PATTERN not a roster (#721): any APPWRITE_* on rented hardware is a finding.
for V in $(compgen -v | grep -E '^APPWRITE_' | sort); do
    [ -n "${!V:-}" ] || continue
    case "$V" in
        *_API_KEY) note "$V is set — a PUBLISH CREDENTIAL on rented hardware. Guide rule 5 forbids this outright. Unset it." ;;
        *)         note "$V is set. This chain publishes nothing and has no use for it (guide rule 5). Unset it." ;;
    esac
done

# Memory, from the CGROUP not the host (guide rule 2). Baseline sources are small — the #695
# campaign peaked ~2 GB RSS per member — but evaluation at S=256 once died at 87.3 GB against an
# 87.5 GiB limit (views-baseline C-54), so the floor is not nominal.
CG=/sys/fs/cgroup/memory.max
if [ -r "$CG" ] && [ "$(cat $CG)" != "max" ]; then
    RAM_GB=$(( $(cat $CG) / 1073741824 ))
else
    RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
fi
echo "  RAM visible to this container: ${RAM_GB}GB (floor 16GB)"
[ "$RAM_GB" -ge 16 ] || note "only ${RAM_GB}GB RAM; this needs >= 16GB"

# The floor is a real pod constraint, but it is overridable so this script can be EXERCISED
# off a pod. A runner that cannot be tested without renting hardware gets verified by
# reading, and reading is how the three existing runners came to be unusable for a
# views-baseline source without anyone noticing.
DISK_FLOOR_GB="${PODRUN_DISK_FLOOR_GB:-50}"
DISK_GB=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
echo "  free on $ROOT: ${DISK_GB}GB (floor ${DISK_FLOOR_GB}GB; ~23-45GB per source at S=64, archived and cleared between)"
[ "${DISK_GB:-0}" -ge "$DISK_FLOOR_GB" ] || note "only ${DISK_GB}GB free on $ROOT; need >= ${DISK_FLOOR_GB}GB for one source plus its archive"

echo "  sources to run: $(echo $SOURCES | wc -w)"
echo "  invocation:     main.py $RUN_ARGS"

[ "$FAIL" -eq 0 ] || { echo "FAILED:preflight" > "$OUT/STATUS"; die "preflight failed — fix every MISSING line above"; }
echo "  preflight clean"

if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
    stage preflight_only_done
    echo "### --preflight only: nothing installed, nothing run."
    exit 0
fi

stage install_python
if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python 3.11 "$VENV" || die "venv creation failed (is uv on PATH?)"
    # views-baseline FIRST and floored at 1.4.0: MatryoshkaConflictology needs `dolls` and
    # PooledConflictology exists only from there, and eleven of these sources name them.
    uv pip install --python "$VENV/bin/python" \
        "views-baseline>=1.4.0,<2.0.0" \
        "views-pipeline-core>=3.2.0,<4.0.0" \
        "views-datafactory>=1.17.0,<2.0.0" || die "pip install failed"
    # register C-151: viewser pins toolz<0.12, which cannot import tlz submodules on Python 3.11
    # and breaks EVERY datafactory fetch. ⛔ MUST STAY THE LAST INSTALL — anything appended below
    # re-resolves this venv and can silently pull toolz back under 0.12 (happened 2026-09-29).
    uv pip install --python "$VENV/bin/python" "toolz>=0.12.1" || die "toolz override failed"
else
    echo "venv already present at $VENV — reusing"
fi

stage verify_install
"$VENV/bin/python" - <<'PY' || die "the venv is not what this run needs — see the error above"
import importlib.metadata as m
import tlz.curried                      # C-151: the override is verified by IMPORT, not by version
from views_baseline.manager.baseline_manager import BaselineForecastingModelManager  # noqa
print("  views-baseline", m.version("views-baseline"),
      "| views-pipeline-core", m.version("views-pipeline-core"),
      "| views-datafactory", m.version("views-datafactory"),
      "| toolz", m.version("toolz"), "| tlz importable | baseline manager importable")
PY

# ── one store state for the whole set ────────────────────────────────────────────────────────
# Fingerprint, not queryset_digest: the digest covers the CONFIG. Measured 2026-11-12 on the
# rebuilt store (views-datafactory 1.18.1): priogrid 148759, month 504, ged_sb_best == 240. It was
# 1,782 before #547. If this reads anything else the store moved and the set is not comparable.
stage fingerprint
"$VENV/bin/python" - <<'PY' || die "store fingerprint check failed — see above. Do NOT run the set against a moved store."
from datafactory_query.defaults import DEFAULT_REMOTE
from datafactory_query.dataset import load_dataset
d = load_dataset(region='land', start=504, end=504, features=['ged_sb_best'],
                 output_format='dataframe', data_dir=DEFAULT_REMOTE.zarr_url).reset_index()
v = float(d.loc[d['priogrid_id'] == 148759, 'ged_sb_best'].iloc[0])
print(f"  store fingerprint: priogrid 148759 month 504 ged_sb_best = {v:,.0f} (expect 240)")
raise SystemExit(0 if abs(v - 240) < 1 else 1)
PY

# ── the sources, one at a time, archived and cleared as we go ────────────────────────────────
N=0
TOTAL=$(echo $SOURCES | wc -w)
for S in $SOURCES; do
    N=$((N + 1))
    stage "source:$S ($N of $TOTAL)"

    AVAIL=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
    [ "${AVAIL:-0}" -ge 50 ] || die "only ${AVAIL}GB free before $S ($N of $TOTAL) — archive-and-clear is not keeping up. Sources already archived are in $OUT and are safe."

    cd "$REPO/models/$S" || die "cannot enter models/$S"
    START=$(date +%s)
    WANDB_MODE=offline WANDB_SILENT=true \
      "$VENV/bin/python" main.py $RUN_ARGS || die "source $S exited non-zero at $N of $TOTAL"

    # Archive BEFORE clearing, and read the archive back before trusting it. A pod artefact that
    # exists only in the checkout dies with the pod (project_pod_artefact_pattern, 5 instances).
    GEN="$REPO/models/$S/data/generated"

    # ⛔ ASSERT THE SHAPE, NOT MERELY NON-EMPTINESS. A non-empty check passes on a run that
    # produced ONE origin of thirteen, and because this loop then archives and CLEARS, the wrong
    # output would be archived and the source data deleted — leaving STATUS OK, a MANIFEST
    # claiming N archives, and nothing afterwards able to tell a short run from a good one.
    # The expected numbers are MEASURED, not guessed: the #695 campaign archive
    # (reports/2026-11-04_baseline_phase1/archives_s64/plain_burlap_s64.tar.zst) holds 13 distinct
    # origin_N directories and 3 distinct lr_*_best targets under each.
    [ -d "$GEN" ] || die "$S produced no data/generated at all. The invocation ($RUN_ARGS) is wrong — STOP rather than run the remaining sources."
    N_ORIGIN=$(find "$GEN" -type d -name 'origin_*' | wc -l)
    N_TARGET=$(find "$GEN" -type d -name 'lr_*_best' -printf '%f\n' 2>/dev/null | sort -u | wc -l)
    echo "  shape: $N_ORIGIN origin dir(s), $N_TARGET distinct target(s) (expect $EXPECT_ORIGINS and $EXPECT_TARGETS)"
    [ "$N_ORIGIN" -eq "$EXPECT_ORIGINS" ] || die "$S wrote $N_ORIGIN origin directories, expected $EXPECT_ORIGINS. NOT archiving and NOT clearing — the output is still on disk. Check the invocation ($RUN_ARGS) before re-running."
    [ "$N_TARGET" -eq "$EXPECT_TARGETS" ] || die "$S wrote $N_TARGET distinct targets, expected $EXPECT_TARGETS. NOT archiving and NOT clearing — the output is still on disk."

    tar -I 'zstd -3' -cf "$OUT/${S}_s64.tar.zst" -C "$GEN" . \
      || die "could not archive $S — NOT clearing its output"
    tar -I zstd -tf "$OUT/${S}_s64.tar.zst" >/dev/null \
      || die "archive for $S does not read back — NOT clearing its output"
    SZ=$(du -sh "$OUT/${S}_s64.tar.zst" | cut -f1)
    # ${GEN:?} so the shell refuses rather than running rm if GEN is ever unset or empty.
    # ! -name .gitkeep: that file is TRACKED (7 per source, asserted by
    # tests/test_model_structure.py), and deleting it makes the checkout structurally invalid
    # part-way through a 21-source run.
    find "${GEN:?}" -mindepth 1 ! -name .gitkeep -delete \
      || echo "  WARNING: could not clear $GEN — the next source may run short of disk"
    echo "  HEARTBEAT $S done in $(( ($(date +%s) - START) / 60 ))min; archive $SZ; free now $(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')GB"
done

stage manifest
{
  echo "set:            #728 baseline comparison ($TOTAL sources)"
  echo "sources:        $SOURCES"
  echo "finished:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "invocation:     main.py $RUN_ARGS"
  echo "S:              declared in each config (64) — this runner overrides nothing"
  echo "engine:         views-baseline $("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("views-baseline"))' 2>/dev/null)"
  echo "pipeline_core:  $("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("views-pipeline-core"))' 2>/dev/null)"
  echo "datafactory:    $("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("views-datafactory"))' 2>/dev/null)"
  echo "toolz:          $("$VENV/bin/python" -c 'import toolz; print(toolz.__version__)' 2>/dev/null) (C-151 override, verified by importing tlz.curried)"
  echo "store:          fingerprint checked once before the set — priogrid 148759 month 504 ged_sb_best == 240"
  echo "git:            $(git -C "$REPO" rev-parse --short HEAD)"
  echo "wandb:          offline (WANDB_MODE=offline), nothing sent"
  echo "published:      nothing — this runner uploads nothing and holds no publish credential"
  echo "archives:       $(ls "$OUT"/*.tar.zst 2>/dev/null | wc -l) of $TOTAL"
} > "$OUT/MANIFEST"
cat "$OUT/MANIFEST"

echo OK > "$OUT/STATUS"
stage done
echo "### baseline set — COMPLETE: $(ls "$OUT"/*.tar.zst 2>/dev/null | wc -l)/$TOTAL archived in $OUT"
echo "### bring $OUT home before the pod is destroyed — it is the only copy."

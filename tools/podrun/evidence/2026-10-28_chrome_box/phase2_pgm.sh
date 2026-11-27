#!/usr/bin/env bash
# PROVISIONAL phase-2 driver — chrome_box (pgm). NOT in the repo yet; if this works it gets
# ported into tools/podrun/ properly, with tests. Written to a file rather than typed so that
# what was run is recorded (the defect pod_run_fao_delivery.sh names for 2026-09-29).
set -uo pipefail
ROOT=/workspace
REPO=$ROOT/views-models
VENV=$ROOT/venv-crafd
ENSEMBLE=chrome_box
MODELS="ashen_sparrow brisk_sparrow candid_sparrow dusky_sparrow eager_sparrow faint_sparrow gentle_sparrow humble_sparrow"
OUT=$ROOT/deliver/$ENSEMBLE
mkdir -p "$OUT/members"
rm -f "$OUT/STATUS" "$OUT/FAILURE"
stage(){ echo "$1" > "$OUT/STAGE"; echo "=== [$(date +%H:%M:%S)] $1 ==="; }
die(){ echo "FAILED:$(cat "$OUT/STAGE" 2>/dev/null)" > "$OUT/STATUS"; printf '%s\n' "$1" > "$OUT/FAILURE"; echo "!!! $1"; exit 1; }

# Memory sampler. memory.peak is authoritative where present; otherwise poll memory.current.
# This is the measurement the config's ~53.85 GiB estimate needs checking against.
( while :; do
    C=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
    echo "$(date +%H:%M:%S) $(( C / 1048576 ))MiB $(cat "$OUT/STAGE" 2>/dev/null)" >> "$OUT/memory.log"
    sleep 10
  done ) & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
[ -r /sys/fs/cgroup/memory.peak ] && echo 0 > /sys/fs/cgroup/memory.peak 2>/dev/null

export WANDB_MODE=offline WANDB_SILENT=true
"$VENV/bin/python" -c "import tlz.curried" || die "venv cannot import tlz.curried (C-151)"

START=$(date +%s); N=0
for M in $MODELS; do
  N=$((N+1)); stage "member:$M ($N of 8)"
  cd "$REPO/models/$M" || die "cannot enter models/$M"
  T0=$(date +%s)
  "$VENV/bin/python" main.py -r forecasting -t -f || die "member $M exited non-zero at $N of 8"
  echo "  HEARTBEAT $M done in $(( ($(date +%s)-T0)/60 ))min; cgroup_now=$(( $(cat /sys/fs/cgroup/memory.current)/1048576 ))MiB"
  [ -d "$REPO/models/$M/data/generated" ] && { mkdir -p "$OUT/members/$M"; cp -r "$REPO/models/$M/data/generated/." "$OUT/members/$M/" 2>/dev/null; }
done

stage pool_and_reconcile
cd "$REPO/ensembles/$ENSEMBLE" || die "cannot enter ensembles/$ENSEMBLE"
# -f -sa only: -t would shell out to each member's run.sh and re-enter the conda lane.
# chrome_box additionally reconciles against chrome_crate, read from local disk.
"$VENV/bin/python" main.py -r forecasting -f -sa || die "$ENSEMBLE pool/reconcile exited non-zero"
[ -d "$REPO/ensembles/$ENSEMBLE/data/generated" ] && { mkdir -p "$OUT/parquet"; cp -r "$REPO/ensembles/$ENSEMBLE/data/generated/." "$OUT/parquet/" 2>/dev/null; }

PEAK_FILE=$( [ -r /sys/fs/cgroup/memory.peak ] && echo "$(( $(cat /sys/fs/cgroup/memory.peak)/1048576 ))MiB" || echo "n/a" )
PEAK_LOG=$(awk '{gsub("MiB","",$2); if($2+0>m) m=$2+0} END{printf "%dMiB", m}' "$OUT/memory.log" 2>/dev/null)
stage manifest
{
  echo "ensemble:      $ENSEMBLE (pgm, global land)"
  echo "members:       $MODELS"
  echo "finished:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "draws:         8 x 128 = 1024 (epic #595 as specified, NOT thinned)"
  echo "reconciled:    against chrome_crate, read from local disk"
  echo "runtime_min:   $(( ($(date +%s)-START)/60 ))"
  echo "PEAK cgroup:   memory.peak=$PEAK_FILE  sampled_max=$PEAK_LOG"
  echo "predicted:     ~53.85 GiB per ensembles/chrome_box/configs/config_hyperparameters.py"
  echo "box:           $(( $(cat /sys/fs/cgroup/memory.max)/1073741824 )) GiB"
  echo "pool:          $(du -sh "$OUT/parquet" 2>/dev/null | cut -f1)"
  echo "git:           $(git -C "$REPO" rev-parse --short HEAD)"
  echo "driver:        /workspace/phase2_pgm.sh (PROVISIONAL, not in the repo)"
  echo "published:     nothing"
} > "$OUT/MANIFEST"
cat "$OUT/MANIFEST"
echo OK > "$OUT/STATUS"
stage done

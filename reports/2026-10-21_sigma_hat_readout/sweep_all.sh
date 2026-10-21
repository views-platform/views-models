#!/usr/bin/env bash
# P4 adequacy probe + the real work: sweep each retrieved artifact.
# Runs in an ISOLATED dir so views-hydranet/reports is never written to.
set -u
SP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ISO=$SP/sweep_iso
E=/home/simon/anaconda3/envs/views-hydranet-env
M=/home/simon/Documents/scripts/views_platform/views-models/models
H=/home/simon/Documents/scripts/views_platform/views-hydranet/reports

# The harness is assembled from THREE files that live in TWO unrelated views-hydranet report
# directories, and nothing used to record that. Deleting $ISO therefore made this script
# un-runnable and the published sigma-hat un-reproducible (/falsify round 5, HF-5). Build $ISO
# here, from named sources, so the measurement can be re-run from this directory alone.
#   run_p2.py              <- 2026-10-16_pos_weight_arm_outcome/evidence/   (the truncated_nb shim)
#   vehicle.py             <- 2026-10-12_underfire_diagnosis/               (load_vehicle + sidecar gate)
#   p2_gate_calibration.py <- 2026-10-12_underfire_diagnosis/               (the committed statistic)
# views-hydranet is READ ONLY from here: files are copied out, never written back.
build_iso() {
    mkdir -p "$ISO"
    local src dst missing=0
    for spec in "run_p2.py:$H/2026-10-16_pos_weight_arm_outcome/evidence/run_p2.py" \
                "vehicle.py:$H/2026-10-12_underfire_diagnosis/vehicle.py" \
                "p2_gate_calibration.py:$H/2026-10-12_underfire_diagnosis/p2_gate_calibration.py"; do
        dst=${spec%%:*}
        src=${spec#*:}
        if [ ! -f "$src" ]; then echo "MISSING HARNESS SOURCE: $src"; missing=1; continue; fi
        cp "$src" "$ISO/$dst"
    done
    [ "$missing" = "0" ] || { echo "cannot build \$ISO -- refusing to sweep"; return 1; }
    # record what was ACTUALLY copied -- hashed here, not at the source path. Reading the sha from
    # the source after the fact records a file that may no longer be the one that ran: that exactly
    # happened on 2026-10-21, when views-hydranet's Amendment 4 rewrote vehicle.py 28 minutes after
    # this seat copied it (see heads_iso/HARNESS_SOURCES.md).
    ( cd "$ISO" && sha256sum run_p2.py vehicle.py p2_gate_calibration.py ) > "$SP/harness_provenance.new"
    if [ -f "$SP/harness_provenance.sha256" ] \
       && ! diff -q "$SP/harness_provenance.sha256" "$SP/harness_provenance.new" >/dev/null; then
        echo "⛔ THE HARNESS HAS CHANGED UPSTREAM since these numbers were measured:"
        diff "$SP/harness_provenance.sha256" "$SP/harness_provenance.new" | sed 's/^/    /'
        echo "    A changed harness is a different instrument. Re-measuring with it is fine --"
        echo "    quoting the OLD numbers alongside the new ones is not."
        echo "    Known live example: Amendment 4 removed EXPECTED_SIDECAR['output_distribution'],"
        echo "    so run_p2.py's shim raises KeyError against the current vehicle.py."
        if [ "${HARNESS_DRIFT_OK:-0}" != "1" ]; then
            echo "    Refusing. Re-run with HARNESS_DRIFT_OK=1 to proceed deliberately."
            rm -f "$SP/harness_provenance.new"
            return 1
        fi
        echo "    HARNESS_DRIFT_OK=1 -- proceeding, and the pin is being updated."
    fi
    mv "$SP/harness_provenance.new" "$SP/harness_provenance.sha256"
    echo "built \$ISO from named sources; provenance in harness_provenance.sha256"
}
build_iso || exit 1
mkdir -p $SP/p2_results
run() { # name  weights
  local n=$1 w=$2
  echo "=================== $n ==================="
  echo "weights: $w"
  [ -f "$w" ] || { echo "MISSING WEIGHTS"; return 1; }
  [ -f "$w.config.json" ] || { echo "MISSING SIDECAR - load_vehicle will refuse (views-models#549)"; return 1; }
  ( cd "$ISO" && /usr/bin/time -v $E/bin/python run_p2.py "$w" 2>&1 ) | tail -40
  if [ -f "$ISO/p2_result.npz" ]; then
    mv "$ISO/p2_result.npz" "$SP/p2_results/p2_result_${n}.npz"
    echo ">>> saved p2_result_${n}.npz"
  else
    echo ">>> NO npz PRODUCED for $n"
  fi
}
run bronze "$M/bronze_kestrel/data/generated/pod_delivery/artifact/calibration_model_20261021_010229.pt"
run cobalt "$M/cobalt_kestrel/data/generated/pod_delivery/artifact/calibration_model_20261021_005527.pt"
run indigo "$M/indigo_kestrel/data/generated/pod_training_20261020_113202/artifact/calibration_model_20261020_113202.pt"
echo "=================== ALL SWEEPS DONE ==================="
ls -l $SP/p2_results/

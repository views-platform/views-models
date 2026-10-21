"""/falsify round 5 — pod five (silver_kestrel on views-hydranet 0.1.4), 2026-10-21.

Round 4 asked whether a retrieved artifact was SECURED. Round 5 asked whether the *conclusion
drawn from it* survives, and whether the machinery that retrieved it had ever been exercised.
Five hard and three soft falsifications. Six are now fixed and are guarded here as regressions;
one is open by decision and stays xfail(strict=True).

Fixed in this pass:
  HF-1/HF-2  READOUT.md attributed the sigma-hat excess and the zero-crossing split to the
             0.1.3/0.1.4 library boundary. silver014 refuted both: 190/190 bit-identical weight
             tensors. Corrected in place, struck not deleted, and corrected on #572.
  HF-3       ARTIFACT_METADATA.md described five arms; six are in the backup.
  HF-4       watch_pod.sh pull() had never executed and died under set -u on first use.
  HF-5       the readout was not reproducible from its own directory.
  SF-1       sigma-hat was reported without naming its common-support arm set.
  SF-2       watcher logs resolved through $0, which is /bin when the file is sourced.
"""
import re
from pathlib import Path

import pytest

REPO = Path("/home/simon/Documents/scripts/views_platform/views-models")
SCRATCH = REPO / "scratchpad"
READOUT = REPO / "reports/2026-10-21_sigma_hat_readout"
BACKUP = Path("/home/simon/Documents/kestrel_sigma_backup_20261021")

# The three claims pod five refuted. They may appear ONLY as quoted, marked corrections.
REFUTED = [
    r"excess sits on the 0\.1\.3/0\.1\.4",
    r"is not a seed variance",
    r"two `<= 0\.1\.3` arms cross zero",
]


def test_refuted_claims_appear_only_inside_a_marked_correction():
    """HF-1/HF-2. The claims were published to #572, so deleting them would hide that they were
    made. They are kept and struck. What must never happen is one of them standing as an
    assertion again — so every occurrence has to sit near a CORRECTED/originally marker."""
    text = (READOUT / "READOUT.md").read_text()
    unmarked = []
    for pat in REFUTED:
        for m in re.finditer(pat, text):
            window = text[max(0, m.start() - 700):m.start()]
            if not re.search(r"CORRECTED|originally (read|concluded|reported)", window):
                unmarked.append((pat, m.start()))
    assert unmarked == [], f"refuted claim asserted without a correction marker: {unmarked}"


def test_the_corrected_result_is_stated_with_its_evidence():
    """A correction that does not carry the measurement is just a retraction. The determinism
    evidence (190/190 tensors, array_equal gate cube) must be in the document."""
    text = (READOUT / "READOUT.md").read_text()
    for needle in ("190/190", "np.array_equal", "numerically inert"):
        assert needle in text, f"corrected readout omits its own evidence: {needle!r}"


def test_backup_metadata_describes_every_arm_actually_present():
    """HF-3. An arm on disk but absent from ARTIFACT_METADATA.md is an arm nobody will know to
    restore."""
    present = {p.name for p in BACKUP.iterdir() if p.is_dir()}
    doc = (BACKUP / "ARTIFACT_METADATA.md").read_text()
    undocumented = {a for a in present if a not in doc}
    assert undocumented == set(), f"arms in backup but not in metadata: {undocumented}"


def test_backup_manifest_covers_every_file_in_every_arm():
    """A manifest that omits a file cannot detect that file's corruption."""
    manifest = (BACKUP / "MANIFEST.sha256").read_text()
    missing = [
        f"{d.name}/{f.name}"
        for d in BACKUP.iterdir() if d.is_dir()
        for f in d.iterdir() if f.is_file() and f"{d.name}/{f.name}" not in manifest
    ]
    assert missing == [], f"files present but unmanifested: {missing}"


def test_the_guard_suite_exercises_the_retrieval_path_not_only_the_verifier():
    """HF-4. pull() had never run. Its first invocation — the instant pod five's artifact
    appeared — hit `local tag=... rest=${spec#*:} src=${rest%%:*}`, which bash expands before
    assigning, so `rest` was unset and set -u killed the watcher: nothing retrieved, nothing
    logged. Six guard tests all targeted verify_local. The suite must reach pull()."""
    suite = (SCRATCH / "test_watch_pod_guards.sh").read_text()
    assert "pull >/dev/null" in suite or re.search(r"^\s*pull\b", suite, re.M), \
        "no test in the guard suite invokes pull()"
    assert "rsync_exit=" in suite, \
        "the suite does not assert pull() logged its per-rsync exit codes — the death signature"


def test_the_sigma_hat_readout_is_reproducible_from_its_own_directory():
    """HF-5. sweep_all.sh did `cd $ISO && python run_p2.py` but nothing created $ISO, and the
    three modules it loads live in two unrelated views-hydranet report dirs with nothing
    recording that. Deleting sweep_iso made my own published number un-reproducible."""
    sweep = (READOUT / "sweep_all.sh").read_text()
    missing = [m for m in ("run_p2.py", "vehicle.py", "p2_gate_calibration.py") if m not in sweep]
    assert missing == [], f"sweep_all.sh records no provenance for: {missing}"
    assert "build_iso" in sweep, "sweep_all.sh still assumes $ISO exists rather than building it"


def test_sigma_hat_is_never_reported_without_naming_its_common_support_set():
    """SF-1. The prereg mandates common bin support but says only 'every arm' — cell members, or
    every arm in the experiment? The two readings move sigma-hat by up to 19%."""
    text = (READOUT / "READOUT.md").read_text()
    assert re.search(r"common support over (all six|the three)", text, re.I), \
        "READOUT.md reports sigma-hat without naming which arms define the common support"
    assert "second unpinned parameter" in text, \
        "the support-set ambiguity is used but not disclosed as a free parameter"


def test_watcher_logs_resolve_correctly_when_the_file_is_sourced():
    """SF-2. watch_pod.sh used $(dirname "$0") — /bin when sourced, which is the mode its own
    test harness uses, so every diagnostic append failed with permission denied. The fix for F-4
    reintroduced F-5."""
    src = (SCRATCH / "watch_pod.sh").read_text()
    assert 'dirname "$0"' not in src, "log path still derived from $0; breaks under source"
    assert "BASH_SOURCE[0]" in src, "log path should resolve via BASH_SOURCE"


@pytest.mark.xfail(strict=True, reason="OPEN BY DECISION: pull()'s destination is per-model, so a "
                                       "second run of one model name overwrites the first's loose "
                                       "files. No pod is running; fixing it would move the "
                                       "established delivery layout. Documented, not built.")
def test_a_second_run_of_one_model_cannot_overwrite_the_first_runs_delivery():
    """SF-3. Pod five reused the silver_kestrel name deliberately, to hold the config fixed — so
    its rsync destination collided with the original silver run's tree and overwrote
    calibration_log.txt. The timestamped dirs survived and the original .pt verified bit-intact
    (sha 89b1c804…), so nothing load-bearing was lost. The destination is run-agnostic by
    construction and the next same-name rerun hits it again."""
    code = "\n".join(ln.split("#")[0] for ln in (SCRATCH / "watch_pod.sh").read_text().splitlines())
    dests = re.findall(r'"\$D/[^"]*"', code)
    assert any(re.search(r"\$\{?(RUN_ID|RUNTS|STAMP)\}?", d) for d in dests), \
        f"no rsync destination carries a run identifier; same-model reruns collide: {dests}"

"""Falsification stubs — 2026-11-17, "the pod can be terminated, nothing on it is missing locally".

VERDICT: FALSIFIED. One hard finding, remediated before the pod was destroyed.

What was missed: `rsync` had pulled only `/workspace/deliver/baseline_arms/`, which is what the
runner writes. Three records the 2026-11-13 delivery of the same lane carries were left on the pod
and would have died with it:

  * the full run log (139 KB) — install resolution, per-model runtimes, peak memory, metric tables;
  * the per-model Weights & Biases offline run directories (52 MB raw, 293 KB compressed), which
    hold each run's own resolved `requirements.txt` and its metric tables as JSON;
  * a checksum file computed ON the pod, which is what makes the local copy verifiable at all.

The per-run `requirements.txt` is the part that matters. It is an independent record of the engine
that produced the numbers — it says `views-baseline==1.5.0`, confirming the MANIFEST rather than
repeating it. Version skew between model runs is the error that already cost this project money
once; throwing away the second witness to it is not a cosmetic loss.

All three were pulled and verified against the pod's own checksums before termination, so
`test_the_delivery_carries_the_same_record_types_as_its_predecessor` PASSES. It is kept as a
regression guard, not as a stub.

The second test FAILS and is the real residue: nothing in the runner produces these records. On
2026-11-13 a person thought to collect them; on 2026-11-17 the same person did not, and only an
audit run while the machine was still alive caught it. A completeness rule that depends on someone
remembering is not a rule.
"""

from __future__ import annotations

import pathlib

import pytest

DELIVERIES = pathlib.Path(
    "/home/simon/Documents/scripts/views_platform/pod_deliverables"
)

#: What a baseline-lane delivery must carry beyond the per-model archives. Each name is a record
#: type that cannot be reconstructed once the pod is gone.
#:
#: Matched anywhere beneath the delivery directory, deliberately: the 2026-11-13 delivery puts the
#: provenance bundle in a `run_provenance/` subdirectory and the 2026-11-17 one puts it at the top
#: level, where `sha256sum -c SHA256.pod` resolves it without a `cd`. The layout drifted; the record
#: is present in both, and asserting on the record is the point.
REQUIRED_RECORDS = ("run.log", "run_provenance.tar.zst", "SHA256.pod", "MANIFEST")


def _baseline_deliveries() -> list[pathlib.Path]:
    """Every delivery directory holding `*_s64.tar.zst` archives from this lane."""
    if not DELIVERIES.is_dir():
        pytest.skip(f"{DELIVERIES} not present — cannot see the subject (ADR-025)")
    return sorted(d for d in DELIVERIES.iterdir() if any(d.glob("*_s64.tar.zst")))


def test_the_delivery_carries_the_same_record_types_as_its_predecessor() -> None:
    """A delivery missing its log, provenance or checksums cannot be audited after teardown."""
    found = _baseline_deliveries()
    assert found, "no baseline-lane deliveries found — the check cannot see its subject"

    incomplete: dict[str, list[str]] = {}
    for delivery in found:
        missing = [r for r in REQUIRED_RECORDS if not any(delivery.rglob(r))]
        if missing:
            incomplete[delivery.name] = missing

    assert not incomplete, (
        "baseline deliveries missing records that die with the pod: "
        f"{incomplete}. Collect them BEFORE terminating; none can be rebuilt afterwards."
    )


def test_the_runner_itself_collects_what_must_come_home() -> None:
    """FAILS TODAY. Completeness currently depends on an operator remembering.

    `pod_run_baseline_arms.sh` writes the archives and a MANIFEST into `$OUT` and stops there. It
    never copies its own log into `$OUT`, never bundles the per-model `wandb/` run records, and
    never computes a checksum file on the pod. So the thing that makes a delivery verifiable is
    produced by hand, outside the script, by whoever happens to think of it.

    The remedy is in the runner, not in a checklist: its `manifest` stage already runs last and
    already knows `$OUT`, `$SOURCES` and `$REPO`.
    """
    runner = pathlib.Path(__file__).resolve().parents[2] / "tools" / "podrun" / "pod_run_baseline_arms.sh"
    assert runner.is_file(), f"{runner} not found — cannot see the subject (ADR-025)"
    body = runner.read_text(encoding="utf-8")

    absent = [
        need
        for need, token in (
            ("copies its own run log into $OUT", "run.log"),
            # ⛔ NOT the bare word "wandb": the runner already contains `WANDB_MODE=offline`,
            # which made an earlier version of this check pass while the records were being left
            # behind. The token has to be the directory, not the setting.
            ("bundles the per-model wandb run records", "/wandb"),
            ("computes a checksum file on the pod", "sha256sum"),
        )
        if token not in body
    ]
    assert not absent, (
        "pod_run_baseline_arms.sh does not collect records that die with the pod: "
        + "; ".join(absent)
        + ". Found on 2026-11-17 by auditing a finished run while the machine was still alive."
    )

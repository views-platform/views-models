"""What `pod_run_crafd_chain.sh` leaves on a pod that `pod_run_chrome_box.sh` brings home.

⚠️ **These belong merged into `test_falsification_pod_teardown_completeness.py`.** They are
in their own file because the audit that produced them (`/falsify`, 2026-11-02, the claim
"I can terminate the pod now") may create a failing test file and may not edit an existing
one. Merge on the first commit that touches either file.

**The measurement.** On pod `216.81.151.55:12889`, after both phases reported `STATUS: OK`
and both deliveries had been rsynced home, these existed only on the pod:

* the eight `models/*_condor/artifacts/forecasting_model_<ts>.pkl` — 84 KB to 419 KB,
  1.9 MB in total, the trained state of every chrome_crate member
* `models/*_condor/wandb/offline-run-*` (24 runs) and
  `ensembles/chrome_crate/wandb/offline-run-*` (1 run) — `WANDB_MODE=offline`, so these
  were never sent anywhere and the pod was their only copy

Both deliveries were otherwise complete: 148 files / 29,133,443,584 bytes for chrome_box
and 100 files / 191,958,963 bytes for chrome_crate, byte-identical on pod and laptop, zero
path differences and zero size mismatches.

**The asymmetry.** `pod_run_chrome_box.sh:489-516` runs `stage collect_members`, which
copies each member's `artifacts/forecasting_model_<ts>.pkl`, each member's `wandb/`, and the
ensemble's own `wandb/`. Its failure message (`:517`) ends *"do NOT destroy this host until
you have decided whether you need those artefacts"* — the runner knows this class of loss.
`pod_run_crafd_chain.sh` has no `collect_members` stage at all; its header (`:81`) documents
`members/` as *"each member's forecast, as produced"*, and that is exactly what it holds.
So the forecasts come home and the models that produced them do not.

This is not symmetrical with C-317 (HydraNet training is not bit-reproducible). A condor
member is retrainable. The cost of the loss is ~23 min per member of pod time, measured, and
a box to run it on — not irrecoverability. That is why these are Tier 3, not Tier 1.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CRATE_CHAIN = REPO / "tools" / "podrun" / "pod_run_crafd_chain.sh"
BOX_RUNNER = REPO / "tools" / "podrun" / "pod_run_chrome_box.sh"


@pytest.mark.xfail(reason="no such guard exists; written 2026-11-02", strict=False)
def test_the_crate_chain_brings_home_the_artefact_that_produced_each_forecast():
    """A forecast whose trained model stayed on the pod cannot be explained after teardown.

    The box runner pairs every `predictions_forecasting_<ts>` with its
    `artifacts/forecasting_model_<ts>.pkl` and refuses to continue if any member has a
    forecast with no artefact beside it. The crate chain copies the forecast alone, so the
    pairing it would have refused on is never attempted.
    """
    copies = [
        line.strip()
        for line in CRATE_CHAIN.read_text().splitlines()
        if re.match(r"\s*(cp|rsync)\b", line)
    ]
    assert copies, "no cp or rsync at all in the chain — the parse is wrong, not the runner"
    assert any("artifacts" in line for line in copies), (
        "pod_run_crafd_chain.sh never names a member artefact, so the eight condor "
        "forecasting_model_<ts>.pkl stay in the pod's repo clone and die with the host. "
        "Measured 2026-11-02: 1.9 MB across eight files, present on the pod and absent "
        "from ensembles/chrome_crate/data/generated/pod_delivery_20261102/ after a "
        "complete, byte-verified rsync of deliver/chrome_crate."
    )


@pytest.mark.xfail(reason="no such guard exists; written 2026-11-02", strict=False)
def test_an_offline_wandb_run_is_collected_because_offline_means_nowhere_else():
    """`WANDB_MODE=offline` makes the pod the only copy, which is the opposite of safe.

    The chain sets offline mode and says so in its own summary (`:520`,
    "wandb: offline (WANDB_MODE=offline), nothing sent"). Nothing sent is correct and is
    the rule — no credential on rented hardware. It also means nothing is retrievable
    later, so the run record has to be copied under `deliver/` or it does not exist.
    """
    # Read the COPY STATEMENTS, not the file. "wandb" appears twice in the chain — once in
    # a disk-budget comment, once in the summary line that says nothing is sent — and a
    # substring check over the whole body is satisfied by either. What matters is whether
    # any cp or rsync names a wandb path.
    copies = [
        line.strip()
        for line in CRATE_CHAIN.read_text().splitlines()
        if re.match(r"\s*(cp|rsync)\b", line)
    ]
    assert copies, "no cp or rsync at all in the chain — the parse is wrong, not the runner"
    assert any("wandb" in line for line in copies), (
        "the chain declares WANDB_MODE=offline but no cp/rsync names a wandb path, so "
        "nothing reaches deliver/. Its two copies take data/generated/ only:\n  "
        + "\n  ".join(copies)
        + "\nMeasured 2026-11-02: 24 models/*_condor/wandb/offline-run-* plus "
        "ensembles/chrome_crate/wandb/offline-run-20261102_082152-fsnwipyl existed only "
        "on the pod. The box runner copies both (pod_run_chrome_box.sh:514,516)."
    )


@pytest.mark.xfail(reason="no such guard exists; written 2026-11-02", strict=False)
def test_a_runner_that_collects_artefacts_warns_before_teardown_and_one_that_does_not_also_warns():
    """The teardown warning lives in the runner that has least need of it.

    `pod_run_chrome_box.sh:517` tells the operator not to destroy the host until they have
    decided about the artefacts — on the path that already collected them. The crate chain,
    which collects none, carries no such line. The warning is attached to the safe runner
    and missing from the lossy one.
    """
    box = BOX_RUNNER.read_text()
    crate = CRATE_CHAIN.read_text()
    assert "do NOT destroy this host" in box, "the box runner's warning moved or was reworded"
    assert "destroy this host" in crate or "teardown" in crate, (
        "pod_run_crafd_chain.sh says nothing about what teardown costs, while the runner "
        "that loses less says it explicitly. An operator reading only the chain's output "
        "has no way to learn that eight trained models are about to be discarded."
    )


@pytest.mark.xfail(reason="no such guard exists; written 2026-11-02", strict=False)
def test_delivery_parity_is_measured_in_bytes_not_in_du_output():
    """`du -sh` across MooseFS and ext4 differs enough to invent a 97 MB discrepancy.

    The pod's `/workspace` is `mfs#us-ks-2.runpod.net:9421`; the laptop is ext4 on
    `/dev/nvme0n1p2`. `du` reports disk usage, so the same `members/` tree read 465 MB on
    the pod and 368 MB locally — a 97 MB gap that looks exactly like a truncated rsync.
    File count and apparent bytes showed 133 files on both sides, identical byte-for-byte.
    A teardown check that compares `du` output will either block a complete delivery or,
    with the bias the other way, pass an incomplete one.
    """
    pytest.fail(
        "no teardown check exists at all; when one is written it must compare "
        "find -printf '%s' totals and path sets, never du -sh across two filesystems."
    )

"""Failing stubs from /falsify on "I can terminate the pod now" (2026-11-12, views-models#721).

NOT part of the suite — `reports/` is in norecursedirs, verified by `pytest --collect-only reports/`
returning "no tests collected". Run explicitly:

    conda run -n views_pipeline python -m pytest reports/2026-11-12_falsify_pod_termination -v

Each test encodes one finding. F1 and F2 were HARD falsifications, remediated the same session and
md5-verified, so those two now PASS and exist to stop the gap reopening. OBS1 is expected to FAIL:
it asserts a property `queryset_digest` does not have.
"""

import json
import os
from pathlib import Path

import pytest

ARCHIVE = Path("/home/simon/Documents/scripts/views_platform/pod_deliverables")
BOX = ARCHIVE / "2026-11-11_chrome_box"
CRATE = ARCHIVE / "2026-11-11_chrome_crate"

pytestmark = pytest.mark.skipif(not ARCHIVE.is_dir(), reason="pod archive not on this machine")


def test_f1_a_retrieved_delivery_contains_both_phases():
    """HARD, remediated. chrome_box reconciles AGAINST chrome_crate.

    The runner header says "deliver/ is the only thing retrieved" and I retrieved one SUBDIRECTORY
    of deliver/. Without chrome_crate the "0.000% of country mass unplaced" verdict rests on the
    runner's own RECONCILIATION_RESIDUAL — a self-report, not independent evidence.
    """
    assert BOX.is_dir(), "phase 2 output (chrome_box) missing from the archive"
    assert CRATE.is_dir(), (
        "phase 1 output (chrome_crate) missing. It is the INPUT the phase-2 reconciliation was "
        "measured against, so without it that measurement cannot be re-derived."
    )
    assert (CRATE / "MANIFEST").is_file()
    assert (CRATE / "STATUS").read_text().strip() == "OK"


def test_f2_both_phase_transcripts_are_retrieved():
    """HARD, remediated. Phase 1's transcript lived only on the pod."""
    logs = BOX / "_runlogs"
    assert (logs / "phase1_crafd_chain.log").is_file(), "phase 1 transcript not retrieved"
    assert (logs / "phase2_chrome_box.log").is_file(), "phase 2 transcript not retrieved"


def test_obs1_queryset_digest_should_distinguish_data_not_only_config():
    """EXPECTED TO FAIL. Measured on the pod: all 8 members' pulls were byte-identical
    (md5 2cb79304c61b) while every member's queryset_digest DIFFERED.

    So the digest cannot answer "did these two runs read the same store?", which is exactly the
    question views-datafactory#547 makes urgent: a store that changes under you is invisible to it.
    datafactory stated this; this is the measurement.
    """
    prov = BOX / "_training_pull_provenance" / "forecasting_datafactory_df.parquet.provenance.json"
    if not prov.is_file():
        pytest.skip("training-pull provenance not archived")
    d = json.loads(prov.read_text())
    assert "data_digest" in d or "store_version" in d, (
        "provenance carries queryset_digest (a digest of the CONFIG) and no digest of the DATA. "
        f"Keys present: {sorted(d)}. Two runs against different stores produce the same "
        "queryset_digest, and one run across 8 members produced 8 different ones over identical "
        "bytes — the field cannot support either comparison."
    )


def test_obs2_the_pool_directory_name_matches_its_contents():
    """EXPECTED TO FAIL. The pool dir is named `parquet/` and contains .npy/.npz.

    This defeated a probe in this very audit: a glob for *.parquet returned zero files and would
    have been recorded as "nothing wrong" had the absence not been treated as unreadable input.
    """
    pool = BOX / "parquet"
    if not pool.is_dir():
        pytest.skip("pool not archived")
    exts = {p.suffix for p in pool.rglob("*") if p.is_file()}
    assert ".parquet" in exts, (
        f"directory is named 'parquet' but holds {sorted(exts)}. A name that misdescribes its "
        "contents makes a correct check return an empty result."
    )

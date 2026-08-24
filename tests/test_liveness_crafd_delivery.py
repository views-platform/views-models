"""Guards on the CRAF'd delivery surface (`tools/liveness/crafd_delivery.py`).

The CRAF'd delivery was armed on 2026-08-14 (#399) and had **no instrument at all** until
#413 — the only way to answer "did CRAF'd get its forecast?" was to list the bucket by
hand, which is what we did on the day. A live partner delivery with no monitor is #320
("FAO forecast delivery has been stalled for 145 days and nothing detected it") waiting to
happen to the second partner.

**The fake decodes the module's query and applies it to real file names**, rather than
keying on the constants the module supplies. That distinction is the whole lesson of C-102
/ #411: the FAO suite keyed on its own constants, so it answered whatever the module asked
and stayed green for months while the matcher found nothing in the real bucket. A fixture
that restates the module's constant cannot notice reality moving.

File names here are the real 2026-08-24 contents of `crafd_bucket`: 108 shards + sidecar +
`__manifest.json` + `historical_dataset_*`, trimmed to a few shards because the names are
what is being guarded, not the count.
"""

import json
from datetime import datetime, timezone
from urllib.parse import parse_qs, unquote, urlparse

import pytest

from tools.liveness.appwrite_api import AppwriteCredentials
from tools.liveness.crafd_delivery import (
    CRAFD_BUCKET_ID,
    CheckReport,
    CrafdDeliveryCheck,
    main,
    render,
)
from tools.liveness.report import exit_code_for

pytestmark = pytest.mark.green

NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
SENTINEL_KEY = "SECRET-KEY-NEVER-RENDER"
CREDS = AppwriteCredentials("https://fra.cloud.appwrite.io/v1", "proj", SENTINEL_KEY)

STEM = "rusty_bucket_forecasting_20260727_095355"
MANIFEST = f"{STEM}__manifest.json"

REAL_LISTING = (
    [{"$id": f"s{i}", "$createdAt": "2026-08-14T18:35:53.000+00:00",
      "name": f"{STEM}__lr_ged_os__m{594 + i:06d}.arrow.parquet", "sizeOriginal": 920_000}
     for i in range(4)]
    + [{"$id": "sc", "$createdAt": "2026-08-14T18:35:54.000+00:00",
        "name": f"{STEM}__sidecar.parquet", "sizeOriginal": 850_000},
       {"$id": "mf", "$createdAt": "2026-08-14T18:35:54.000+00:00",
        "name": MANIFEST, "sizeOriginal": 26_544},
       {"$id": "h", "$createdAt": "2026-08-14T18:36:05.000+00:00",
        "name": "historical_dataset_20260814_203554.parquet", "sizeOriginal": 171_838_903}]
)


def _appwrite_like_fetch(listing, fail_with=None):
    def fetch(url, headers):
        assert headers["X-Appwrite-Key"] == SENTINEL_KEY
        if fail_with is not None:
            raise fail_with
        if f"/buckets/{CRAFD_BUCKET_ID}" in url and "/files" not in url:
            return {"$id": CRAFD_BUCKET_ID}
        raw = parse_qs(urlparse(url).query).get("queries[]", [])
        files, limit = list(listing), None
        for q in (json.loads(unquote(r)) for r in raw):
            method, values = q.get("method"), q.get("values") or []
            if method == "startsWith":
                files = [f for f in files if f["name"].startswith(values[0])]
            elif method == "endsWith":
                files = [f for f in files if f["name"].endswith(values[0])]
            elif method == "orderDesc":
                files = sorted(files, key=lambda f: f["$createdAt"], reverse=True)
            elif method == "limit":
                limit = values[0]
        return {"total": len(files), "files": files[:limit] if limit else files}

    return fetch


class TestItSeesWhatIsActuallyInTheBucket:
    def test_a_real_delivery_reads_as_delivering(self):
        report = CrafdDeliveryCheck(
            credentials=CREDS, fetch=_appwrite_like_fetch(REAL_LISTING)
        ).run(now=NOW)
        assert report.verdict == "DELIVERING"
        assert report.forecast_verdict == "DELIVERING"
        assert report.forecast_newest_name == MANIFEST

    def test_the_residual_does_not_count_the_delivery_itself(self):
        report = CrafdDeliveryCheck(
            credentials=CREDS, fetch=_appwrite_like_fetch(REAL_LISTING)
        ).run(now=NOW)
        assert report.other_files == 0, (
            f"{report.other_files} files read as belonging to neither stream, but every "
            f"file in this listing belongs to the forecast run or the historical stream"
        )

    def test_the_forecast_verdict_is_reachable_in_both_directions(self):
        """A verdict with one reachable value asserts nothing — C-102's actual defect."""
        empty = CrafdDeliveryCheck(
            credentials=CREDS, fetch=_appwrite_like_fetch([])
        ).run(now=NOW)
        assert empty.forecast_verdict == "NEVER_DELIVERED"

    def test_an_old_manifest_reads_as_stalled_not_missing(self):
        stale = [dict(f, **{"$createdAt": "2026-01-01T00:00:00.000+00:00"}) for f in REAL_LISTING]
        report = CrafdDeliveryCheck(
            credentials=CREDS, fetch=_appwrite_like_fetch(stale)
        ).run(now=NOW)
        assert report.forecast_verdict == "STALLED"
        assert report.verdict == "DELIVERY_STALLED"


class TestItFailsHonestly:
    @pytest.mark.red
    def test_a_storage_failure_is_unreachable_not_quiet(self):
        report = CrafdDeliveryCheck(
            credentials=CREDS, fetch=_appwrite_like_fetch([], fail_with=RuntimeError("boom"))
        ).run(now=NOW)
        assert report.verdict == "UNREACHABLE"
        assert "RuntimeError" in report.error

    @pytest.mark.red
    def test_without_credentials_it_skips_truthfully_rather_than_reporting_quiet(self):
        report = CrafdDeliveryCheck(credentials=None, fetch=_appwrite_like_fetch([])).run(now=NOW)
        assert report.verdict in ("SKIP_NO_CREDENTIALS", "CREDENTIALS_INCOMPLETE")
        assert report.forecast_verdict is None

    @pytest.mark.red
    def test_the_key_is_never_rendered(self):
        report = CrafdDeliveryCheck(
            credentials=CREDS, fetch=_appwrite_like_fetch(REAL_LISTING)
        ).run(now=NOW)
        # `render` returns a STRING, not lines. The first version of this test wrote
        # `"\n".join(render(report))`, which splices a newline between every CHARACTER —
        # so the key could never be found and the assertion could never fail. Proven
        # vacuous before it was fixed; kept as a comment because the shape is easy to
        # write again.
        assert SENTINEL_KEY not in render(report)


@pytest.mark.beige
class TestStructuralConventions:
    def test_it_honours_the_surface_contract(self):
        """docs/CICs/LivenessChecks.md: a Check, a frozen report, a main returning an int."""
        assert callable(main) and callable(render)
        assert CheckReport.__dataclass_params__.frozen

    def test_every_verdict_it_can_emit_is_classifiable(self):
        """`exit_code_for` raises on an unregistered verdict — by design, so a new one
        cannot slip through as a silent 0."""
        for verdict in ("DELIVERING", "DELIVERY_STALLED", "UNREACHABLE",
                        "SKIP_NO_CREDENTIALS", "CREDENTIALS_INCOMPLETE"):
            assert isinstance(exit_code_for(verdict), int)

    def test_the_bound_comes_from_the_declaration_not_a_constant(self):
        report = CrafdDeliveryCheck(
            credentials=CREDS, fetch=_appwrite_like_fetch(REAL_LISTING)
        ).run(now=NOW)
        from deliveries.status import declared_max_age_days

        from tools.liveness.crafd_delivery import BOUND_SOURCE

        assert BOUND_SOURCE == "deliveries/un_crafd.py"
        assert report.max_age_days == declared_max_age_days("un_crafd")
        # And it is rendered, so an operator can see WHICH declaration decided the bound
        # rather than trusting a number the tool might have invented.
        assert "max_age_declared_in: deliveries/un_crafd.py" in render(report)


@pytest.mark.live
def test_live_crafd_delivery_invariants():
    check = CrafdDeliveryCheck()
    if check.credentials is None:
        pytest.skip("no Appwrite credentials resolvable in this environment")
    try:
        report = check.run()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"appwrite unreachable: {type(e).__name__}: {e}")
    if report.verdict in ("UNREACHABLE", "SKIP_NO_CREDENTIALS", "CREDENTIALS_INCOMPLETE"):
        pytest.skip(f"not observable here: {report.verdict}")
    assert report.total_files and report.total_files > 0
    assert report.forecast_newest_name is not None, (
        "the forecast stream is invisible against the real bucket — the C-102 failure"
    )

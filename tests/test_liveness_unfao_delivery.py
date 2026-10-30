"""Liveness S4: the FAO delivery bucket (unfao_bucket) — issue #242, epic #238.

TDD suite written BEFORE the implementation. Ground truth captured live
2026-07-19: bucket `unfao_bucket`, 18 files, two delivery streams —
the ADR-013 manifest (newest 2026-08-13) and
``historical_dataset_*.parquet`` (newest 2026-03-30, ~20MB). Real monthly
deliveries ran Jan–Mar 2026, then stalled; nobody noticed (register C-99
class). This check makes "when did FAO last receive anything?" a machine
answer with per-stream verdicts.

Unit tests are offline (fake fetch, fake credentials, injected clock).
Credentials resolution is REUSED from tools.liveness.appwrite_store (same
project, same .env — S7 will home it in a shared credentials module).
"""

from datetime import datetime, timezone

import pytest

from tools.liveness.appwrite_store import AppwriteCredentials
from tools.liveness.unfao_delivery import (
    UnfaoDeliveryCheck,
    main,
    render,
)

pytestmark = pytest.mark.green

NOW = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
SENTINEL_KEY = "SECRET-KEY-NEVER-RENDER"
CREDS = AppwriteCredentials("https://fra.cloud.appwrite.io/v1", "proj", SENTINEL_KEY)

#: Captured live 2026-08-24. The forecast stream is judged on the ADR-013 commit marker —
#: written after the shards and the sidecar, so its presence means the run finished.
#: The fixtures below deliberately carry a source model in the run stem
#: (`rusty_bucket_forecasting_...`) that the module never matches on: the module keys on
#: the `__manifest.json` suffix, and a fixture that agreed with a hardcoded model name
#: would be re-creating C-102 in the tests.
STEM = "rusty_bucket_forecasting_20260727_095355"
MANIFEST = f"{STEM}__manifest.json"

REAL_FORECAST_DOC = {
    "total": 1,
    "files": [{"$id": "f1", "$createdAt": "2026-03-10T10:47:48.000+00:00",
               "name": "old_run_20260310_114703__manifest.json", "sizeOriginal": 26_544}],
}
#: How many files the newest manifest's own run owns — shards + sidecar + manifest.
REAL_RUN_DOC = {"total": 9, "files": []}
REAL_HISTORICAL_DOC = {
    "total": 8,
    "files": [{"$id": "h1", "$createdAt": "2026-03-30T09:48:45.000+00:00",
               "name": "historical_dataset_20260330_114835.parquet", "sizeOriginal": 20_500_000}],
}
REAL_OVERALL_DOC = {"total": 18, "files": []}

FRESH_FORECAST_DOC = {
    "total": 1,
    "files": [{"$id": "f", "$createdAt": "2026-07-10T08:00:00.000+00:00",
               "name": MANIFEST, "sizeOriginal": 26_544}],
}
FRESH_RUN_DOC = {"total": 1, "files": []}
FRESH_HISTORICAL_DOC = {
    "total": 1,
    "files": [{"$id": "h", "$createdAt": "2026-07-11T08:00:00.000+00:00",
               "name": "historical_dataset_20260711_100000.parquet", "sizeOriginal": 20_000_000}],
}
FRESH_OVERALL_DOC = {"total": 2, "files": []}

def _responses(forecast, historical, overall, run=None):
    """Keyed on what actually appears in each URL, which is the point of the rewrite.

    The manifest query encodes `endsWith(name, "__manifest.json")`, so its URL carries
    `__manifest.json` and NOT the run stem. The run-count query encodes
    `startsWith(name, "<stem>")`, so it carries the stem and not the suffix. They are
    distinguishable without either fixture restating a constant the module owns — which
    is what made the previous version assert that the tool agreed with itself.
    """
    run = run if run is not None else {"total": 0, "files": []}
    stem = ""
    if forecast.get("files"):
        stem = forecast["files"][0]["name"].removesuffix("__manifest.json")
    return {"__manifest.json": forecast,
            **({stem: run} if stem else {}),
            "historical_dataset_": historical,
            "unfao_bucket/files": overall,
            # The bucket GET that proves the key is accepted before any listing is
            # believed. Last, so the `/files` listings above still match first. Its
            # body is never read — only whether it raises.
            "buckets/unfao_bucket": {"$id": "unfao_bucket"}}

def _fake_fetch(responses):
    def fetch(url, headers):
        assert headers["X-Appwrite-Key"] == SENTINEL_KEY
        for key, value in responses.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"unexpected url: {url}")
    return fetch


@pytest.fixture
def no_machine_credentials(monkeypatch, tmp_path):
    """Cut this test off from the machine it runs on.

    ``credential_gap_report()`` reads process env AND this repository's own
    ``.env``, so a test asserting "nothing is configured" was in fact asserting
    something about the laptop. These passed for as long as the developer's
    ``.env`` happened to carry all twelve Appwrite coordinates, and went red the
    hour those were removed (correctly, per ADR-018) — the removal exposed the
    defect, it did not cause it. Nothing uncommitted may decide a verdict here.
    """
    from tools.liveness import appwrite_api

    for key in appwrite_api.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    empty_root = tmp_path / "repo"
    empty_root.mkdir()
    monkeypatch.setattr(appwrite_api, "REPO_ROOT", empty_root)

# ── verdicts on the real capture ──────────────────────────────────────

def test_both_streams_stalled_on_real_capture():
    fetch = _fake_fetch(_responses(REAL_FORECAST_DOC, REAL_HISTORICAL_DOC, REAL_OVERALL_DOC,
                                   run=REAL_RUN_DOC))
    report = UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.verdict == "DELIVERY_STALLED"
    assert report.forecast_verdict == "STALLED"
    assert report.historical_verdict == "STALLED"
    assert report.forecast_newest_name == "old_run_20260310_114703__manifest.json"
    assert report.forecast_days_since == 131
    assert report.historical_newest_name == "historical_dataset_20260330_114835.parquet"
    assert report.historical_days_since == 111
    # 18 total - 9 in the newest manifest's run - 8 historical = 1 belonging to neither.
    # Counted from the run rather than from the manifest alone: with the manifest as the
    # only forecast match this would read 8, and a residual that counts a healthy
    # delivery's own shards is the same misleading number C-102 produced, relabelled.
    assert report.other_files == 1

def test_delivering_when_both_streams_recent():
    fetch = _fake_fetch(_responses(FRESH_FORECAST_DOC, FRESH_HISTORICAL_DOC, FRESH_OVERALL_DOC))
    report = UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.verdict == "DELIVERING"
    assert report.forecast_verdict == "DELIVERING"
    assert report.historical_verdict == "DELIVERING"

def test_mixed_streams_yield_overall_stalled():
    fetch = _fake_fetch(_responses(FRESH_FORECAST_DOC, REAL_HISTORICAL_DOC, {"total": 2, "files": []}))
    report = UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.forecast_verdict == "DELIVERING"
    assert report.historical_verdict == "STALLED"
    assert report.verdict == "DELIVERY_STALLED"

def test_missing_stream_reported_as_never_delivered():
    fetch = _fake_fetch(_responses({"total": 0, "files": []}, FRESH_HISTORICAL_DOC, {"total": 1, "files": []}))
    report = UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.forecast_verdict == "NEVER_DELIVERED"
    assert report.verdict == "DELIVERY_STALLED"

@pytest.mark.red
def test_unreachable_when_fetch_fails():
    report = UnfaoDeliveryCheck(credentials=CREDS,
                                fetch=_fake_fetch({"unfao_bucket": OSError("dns fail")})).run(now=NOW)
    assert report.verdict == "UNREACHABLE"
    assert "dns fail" in (report.error or "")

def test_skip_when_no_credentials(no_machine_credentials):
    report = UnfaoDeliveryCheck(credentials=None, fetch=_fake_fetch({})).run(now=NOW)
    assert report.verdict == "SKIP_NO_CREDENTIALS"


# ── raw facts + redaction ─────────────────────────────────────────────

def test_render_facts_and_redaction():
    fetch = _fake_fetch(_responses(REAL_FORECAST_DOC, REAL_HISTORICAL_DOC, REAL_OVERALL_DOC,
                                   run=REAL_RUN_DOC))
    text = render(UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW))
    lines = text.strip().splitlines()
    assert all(":" in line for line in lines)
    assert SENTINEL_KEY not in text
    assert any("old_run_20260310_114703__manifest.json" in line for line in lines)
    assert any("DELIVERY_STALLED" in line for line in lines)


# ── exit codes ────────────────────────────────────────────────────────

def test_exit_zero_delivering(capsys):
    fetch = _fake_fetch(_responses(FRESH_FORECAST_DOC, FRESH_HISTORICAL_DOC, FRESH_OVERALL_DOC))
    assert main(check=UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch), now=NOW) == 0

def test_exit_zero_skip(no_machine_credentials, capsys):
    assert main(check=UnfaoDeliveryCheck(credentials=None, fetch=_fake_fetch({})), now=NOW) == 0

def test_exit_one_stalled(capsys):
    fetch = _fake_fetch(_responses(REAL_FORECAST_DOC, REAL_HISTORICAL_DOC, REAL_OVERALL_DOC,
                                   run=REAL_RUN_DOC))
    assert main(check=UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch), now=NOW) == 1

@pytest.mark.red
def test_exit_two_unreachable(capsys):
    fetch = _fake_fetch({"unfao_bucket/files": OSError("boom")})
    assert main(check=UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch), now=NOW) == 2


# ── live integration (creds + network; skips truthfully) ──────────────

@pytest.mark.live
def test_live_unfao_delivery_invariants():
    check = UnfaoDeliveryCheck()
    if check.credentials is None:
        pytest.skip("no Appwrite credentials resolvable in this environment")
    try:
        report = check.run()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"unfao bucket unreachable: {type(e).__name__}: {e}")
    if report.verdict == "UNREACHABLE":
        pytest.skip(f"unfao bucket unreachable: {report.error}")
    assert report.total_files and report.total_files > 0
    assert report.forecast_newest_name is not None


@pytest.mark.beige
def test_structural_conventions_unfao_delivery():
    """ADR-005 beige: surface module conventions — check/render/main exposed,
    and every verdict this surface can emit is registered in the exit map."""
    import tools.liveness.unfao_delivery as module
    from tools.liveness.report import EXIT_CODE_BY_VERDICT

    assert callable(module.main) and callable(module.render)
    assert hasattr(module, "CheckReport")
    for verdict in ('DELIVERING', 'DELIVERY_STALLED', 'SKIP_NO_CREDENTIALS', 'UNREACHABLE'):
        assert verdict in EXIT_CODE_BY_VERDICT, verdict


# ── the rejected-key blind spot (see the sibling suite for the measurement) ──
# Same mechanism, and worse here: this surface's name for "the listing came back
# empty" is DELIVERY_STALLED — indistinguishable, before the fix, from a dead key
# on the bucket the UN FAO is served from.

@pytest.mark.red
def test_rejected_key_is_unreachable_not_stalled():
    def fetch(url, headers):
        if "/files?" in url:
            return {"total": 0, "files": []}
        raise PermissionError("HTTP Error 401: Unauthorized")

    report = UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.verdict == "UNREACHABLE", (
        "a rejected key read as a partner bucket that had gone quiet"
    )
    assert "401" in (report.error or "")


def test_key_is_verified_before_any_listing_is_believed():
    calls = []

    def fetch(url, headers):
        calls.append(url)
        return {"total": 0, "files": []}

    UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert calls and calls[0].endswith("/storage/buckets/unfao_bucket"), (
        f"first call was {calls[0] if calls else '<none>'}"
    )


# ── A1 (#360): one freshness threshold, and it is the declared one ──────────
#
# `DELIVERING_WITHIN_DAYS = 45` used to live in unfao_delivery.py while
# deliveries/un_fao.py declared `max_age = months(2)` ≈ 60 days. Two thresholds,
# two files, disagreeing — the shape epic #342 removed for `ensemble` (#347) and
# `wire_upload_enabled` (#348), still present in the one piece of code that
# actually measures freshness (ADR-019 §8; register C-121).


class TestTheThresholdIsDerivedFromTheDeclaration:
    def test_the_hardcoded_constant_is_gone(self):
        import tools.liveness.unfao_delivery as mod

        assert not hasattr(mod, "DELIVERING_WITHIN_DAYS"), (
            "DELIVERING_WITHIN_DAYS is back. The bound must come from "
            "deliveries/un_fao.py's max_age — one fact, one place."
        )

    def test_the_default_bound_is_what_the_delivery_declares(self):
        from deliveries.un_fao import REQUIRE

        fetch = _fake_fetch(_responses(REAL_FORECAST_DOC, REAL_HISTORICAL_DOC, REAL_OVERALL_DOC,
                                   run=REAL_RUN_DOC))
        report = UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
        assert report.max_age_days == REQUIRE.max_age.count * 30

    def test_the_same_run_flips_verdict_when_the_declared_bound_changes(self):
        """The test that proves the wiring, not merely the constant's absence.

        The forecast stream's newest file is 2026-03-10 and NOW is 2026-07-19 —
        131 days. Under a 200-day bound that is DELIVERING; under 60 it is STALLED.
        Same data, same clock, different declaration.
        """
        fetch = _fake_fetch(_responses(REAL_FORECAST_DOC, REAL_HISTORICAL_DOC, REAL_OVERALL_DOC,
                                   run=REAL_RUN_DOC))
        lenient = UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch, max_age_days=200).run(now=NOW)
        strict = UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch, max_age_days=60).run(now=NOW)

        assert lenient.forecast_verdict == "DELIVERING"
        assert strict.forecast_verdict == "STALLED"
        assert lenient.forecast_days_since == strict.forecast_days_since

    def test_render_says_which_bound_was_used(self):
        """An operator reading the output must be able to tell it is the declared
        one, not a number the tool invented."""
        fetch = _fake_fetch(_responses(REAL_FORECAST_DOC, REAL_HISTORICAL_DOC, REAL_OVERALL_DOC,
                                   run=REAL_RUN_DOC))
        text = render(UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW))
        assert "max_age_days" in text
        assert "deliveries/un_fao.py" in text


class TestItRefusesToInventABound:
    """ADR-003 / ADR-020. A default bound would silently re-create the two-thresholds
    defect with one of the two numbers invisible."""

    def test_a_missing_declaration_raises_and_names_the_file(self):
        """The real message, from the real loader — not one a monkeypatch threw."""
        from deliveries.status import declared_max_age_days

        with pytest.raises(FileNotFoundError) as exc:
            declared_max_age_days("no_such_consumer")
        assert "deliveries/no_such_consumer.py" in str(exc.value)
        assert "will not invent" in str(exc.value)

    def test_a_declaration_without_max_age_raises_and_says_what_to_add(self, monkeypatch):
        """Patched at the loader, because `declared_max_age_days` re-executes the
        file from disk — patching the imported module object would not reach it."""
        import types

        import deliveries.status as status
        from deliveries.vocabulary import Require

        monkeypatch.setattr(
            status, "load_delivery",
            lambda path: types.SimpleNamespace(REQUIRE=Require(targets=("x",))),
        )
        with pytest.raises(ValueError) as exc:
            status.declared_max_age_days("un_fao")
        assert "max_age=months(n)" in str(exc.value)

    def test_the_check_propagates_rather_than_defaulting(self, monkeypatch):
        """If the bound cannot be read, the check must fail — not fall back to a
        number and report a verdict computed against it."""
        import tools.liveness.unfao_delivery as mod

        def _boom():
            raise RuntimeError("bound unavailable")

        monkeypatch.setattr(mod, "_load_declared_max_age_days", _boom)
        with pytest.raises(RuntimeError):
            UnfaoDeliveryCheck(credentials=CREDS, fetch=_fake_fetch({})).run(now=NOW)


class TestCredentialSkipIsUnaffected:
    def test_still_skips_truthfully_without_credentials(self, no_machine_credentials):
        """A silent pass here would report 'delivering' having read nothing (C-75)."""
        report = UnfaoDeliveryCheck(credentials=None, fetch=_fake_fetch({})).run(now=NOW)
        assert report.verdict == "SKIP_NO_CREDENTIALS"


# ── the guard C-102 needed: the module's queries, run against real bucket contents ──


#: A faithful sample of what `unfao_bucket` held on 2026-08-24 — 108 shards, a sidecar,
#: the ADR-013 manifest, and the historical artifact. Trimmed to 6 shards; the count is
#: irrelevant to what this guards, the NAMES are the point.
REAL_BUCKET_LISTING = (
    [{"$id": f"s{i}", "$createdAt": "2026-08-13T06:00:30.000+00:00",
      "name": f"{STEM}__lr_ged_sb__m{559 + i:06d}.arrow.parquet", "sizeOriginal": 920_000}
     for i in range(6)]
    + [{"$id": "sc", "$createdAt": "2026-08-13T06:00:42.000+00:00",
        "name": f"{STEM}__sidecar.parquet", "sizeOriginal": 850_000},
       {"$id": "mf", "$createdAt": "2026-08-13T06:00:43.000+00:00",
        "name": MANIFEST, "sizeOriginal": 26_544},
       {"$id": "h", "$createdAt": "2026-08-13T06:00:55.000+00:00",
        "name": "historical_dataset_20260813_080043.parquet", "sizeOriginal": 171_838_903}]
)


def _appwrite_like_fetch(listing):
    """A fake that PARSES the query and filters, instead of pattern-matching the URL.

    This is the difference between a fixture and a guard. The previous fake keyed on the
    literal prefix constants the module supplies, so it answered whatever the module
    asked and the test asserted that the tool agreed with itself. It stayed green for
    months while `forecast_dataset_` matched nothing in the real bucket, because the
    capture it was built from was faithful to **2026-03** and the naming had moved to the
    ADR-013 manifest (C-102, #411).

    Here the module's real query is decoded and applied to real file names. A matcher that
    does not match reality returns nothing and the verdict goes to NEVER_DELIVERED — which
    is exactly the failure that went unseen.
    """
    import json
    from urllib.parse import parse_qs, unquote, urlparse

    def fetch(url, headers):
        assert headers["X-Appwrite-Key"] == SENTINEL_KEY
        if "/buckets/unfao_bucket" in url and "/files" not in url:
            return {"$id": "unfao_bucket"}
        raw = parse_qs(urlparse(url).query).get("queries[]", [])
        queries = [json.loads(unquote(q)) for q in raw]
        files, limit = list(listing), None
        for q in queries:
            method, values = q.get("method"), q.get("values") or []
            if method == "startsWith":
                files = [f for f in files if f["name"].startswith(values[0])]
            elif method == "endsWith":
                files = [f for f in files if f["name"].endswith(values[0])]
            elif method == "orderDesc":
                files = sorted(files, key=lambda f: f["$createdAt"], reverse=True)
            elif method == "limit":
                limit = values[0]
        total = len(files)
        return {"total": total, "files": files[:limit] if limit else files}

    return fetch


class TestTheSurfaceFindsWhatIsActuallyInTheBucket:
    """Every assertion here is against real 2026-08-24 file names, not a restated constant."""

    def test_a_real_delivery_reads_as_delivering(self):
        report = UnfaoDeliveryCheck(
            credentials=CREDS, fetch=_appwrite_like_fetch(REAL_BUCKET_LISTING)
        ).run(now=datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc))
        assert report.forecast_verdict == "DELIVERING", (
            "the forecast stream is invisible to this surface against real bucket "
            "contents — the C-102 failure, which reported NEVER_DELIVERED over 110 "
            "delivered files"
        )
        assert report.verdict == "DELIVERING"
        assert report.forecast_newest_name == MANIFEST

    def test_the_residual_does_not_count_the_delivery_itself(self):
        """`other_files` must mean "belongs to neither stream", not "is not the manifest"."""
        report = UnfaoDeliveryCheck(
            credentials=CREDS, fetch=_appwrite_like_fetch(REAL_BUCKET_LISTING)
        ).run(now=datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc))
        assert report.other_files == 0, (
            f"{report.other_files} files read as belonging to neither stream, but every "
            f"file in this listing belongs to the forecast run or the historical stream"
        )

    def test_an_empty_bucket_still_reads_as_never_delivered(self):
        """The verdict must remain reachable in both directions, or it asserts nothing."""
        report = UnfaoDeliveryCheck(
            credentials=CREDS, fetch=_appwrite_like_fetch([])
        ).run(now=datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc))
        assert report.forecast_verdict == "NEVER_DELIVERED"

"""Liveness S6: the VPN-only legacy store (gjoll) — issue #244, epic #238.

TDD suite written BEFORE the implementation. The legacy prediction store is
Postgres on ``gjoll.muspelheim.local`` (PRIO-internal, resolvable only on
the PRIO VPN), accessed via ``views_forecasts.db_ops.ViewsMetadata`` whose
constructor connects immediately; ``.get_runs()`` returns [name, description,
min_month, max_month]. Off-VPN the connection dies with
``could not translate host name "gjoll.muspelheim.local"`` — which must be
the truthful verdict VPN_REQUIRED, never a false RED.

Historical receipt encoded here: the store's Postgres schema is literally
``forecasts_metadata`` — the origin of the phantom Appwrite collection ID
that killed the June 2026 run (someone copied the legacy schema name into
the new store's config).

Run-name parsing and freshness reuse S1 (tools.liveness.old_api) — one
parser, one convention, everywhere.
"""


import pytest

from tools.liveness.old_api import FRESHNESS_BUDGET_MONTHS
from tools.liveness.vpn_store import (
    STORE_HOST,
    VpnStoreCheck,
    main,
    render,
)
from tools.partitions.domain import date_to_month_id

pytestmark = pytest.mark.green

CAPTURE_NOW = date_to_month_id(2026, 7)  # 559

RUN_ROWS = [
    {"name": "escwa_2021_02_01", "min_month": 400, "max_month": 460},
    {"name": "fatalities003_2026_04_t01", "min_month": 121, "max_month": 592},
    {"name": "fatalities003_2026_05_t01", "min_month": 121, "max_month": 593},
    {"name": "r_2021_12_01", "min_month": 400, "max_month": 460},
]


def _client(value):
    def list_runs():
        if isinstance(value, Exception):
            raise value
        return value
    return list_runs


# ── verdicts ──────────────────────────────────────────────────────────

def test_fresh_when_latest_within_budget():
    report = VpnStoreCheck(list_runs=_client(RUN_ROWS)).run(now_month_id=CAPTURE_NOW)
    assert report.verdict == "STORE_FRESH"
    assert report.latest_run == "fatalities003_2026_05_t01"
    assert report.months_behind == FRESHNESS_BUDGET_MONTHS == 2
    assert report.latest_max_month == 593

@pytest.mark.red
def test_stale_when_latest_beyond_budget():
    report = VpnStoreCheck(list_runs=_client(RUN_ROWS)).run(now_month_id=CAPTURE_NOW + 3)
    assert report.verdict == "STORE_STALE"
    assert report.months_behind == 5

@pytest.mark.red
def test_vpn_required_on_host_resolution_failure():
    err = OSError('could not translate host name "gjoll.muspelheim.local" to address')
    report = VpnStoreCheck(list_runs=_client(err)).run(now_month_id=CAPTURE_NOW)
    assert report.verdict == "VPN_REQUIRED"
    assert STORE_HOST in (report.error or "")

def test_skip_when_package_missing():
    report = VpnStoreCheck(list_runs=_client(ModuleNotFoundError("No module named 'ingester3'"))).run(
        now_month_id=CAPTURE_NOW
    )
    assert report.verdict == "SKIP_NO_PACKAGE"

@pytest.mark.red
def test_unreachable_on_other_errors():
    report = VpnStoreCheck(list_runs=_client(RuntimeError("password authentication failed"))).run(
        now_month_id=CAPTURE_NOW
    )
    assert report.verdict == "UNREACHABLE"
    assert "password authentication" in (report.error or "")

def test_no_fatalities_runs_is_stale_class():
    rows = [{"name": "escwa_2021_02_01", "min_month": 1, "max_month": 2}]
    report = VpnStoreCheck(list_runs=_client(rows)).run(now_month_id=CAPTURE_NOW)
    assert report.verdict == "STORE_STALE"
    assert report.latest_run is None


# ── raw facts ─────────────────────────────────────────────────────────

def test_render_facts():
    text = render(VpnStoreCheck(list_runs=_client(RUN_ROWS)).run(now_month_id=CAPTURE_NOW))
    lines = text.strip().splitlines()
    assert all(":" in line for line in lines)
    assert any(STORE_HOST in line for line in lines)
    assert any("fatalities003_2026_05_t01" in line for line in lines)
    assert any("forecasts_metadata" in line for line in lines)  # the schema receipt


# ── exit codes ────────────────────────────────────────────────────────

def test_exit_zero_fresh(capsys):
    assert main(check=VpnStoreCheck(list_runs=_client(RUN_ROWS)), now_month_id=CAPTURE_NOW) == 0

def test_exit_zero_vpn_required(capsys):
    err = OSError('could not translate host name "gjoll.muspelheim.local"')
    assert main(check=VpnStoreCheck(list_runs=_client(err)), now_month_id=CAPTURE_NOW) == 0
    assert "VPN_REQUIRED" in capsys.readouterr().out

def test_exit_one_stale(capsys):
    assert main(check=VpnStoreCheck(list_runs=_client(RUN_ROWS)), now_month_id=CAPTURE_NOW + 6) == 1

@pytest.mark.red
def test_exit_two_unreachable(capsys):
    assert main(check=VpnStoreCheck(list_runs=_client(RuntimeError("boom"))), now_month_id=CAPTURE_NOW) == 2


# ── live integration (VPN + package; skips truthfully) ────────────────

@pytest.mark.live
def test_live_vpn_store_invariants():
    try:
        report = VpnStoreCheck().run()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"vpn store not checkable: {type(e).__name__}: {e}")
    if report.verdict in ("VPN_REQUIRED", "SKIP_NO_PACKAGE", "UNREACHABLE"):
        pytest.skip(f"vpn store not reachable here: {report.verdict} {report.error or ''}")
    assert report.latest_run is not None


@pytest.mark.beige
def test_structural_conventions_vpn_store():
    """ADR-005 beige: surface module conventions — check/render/main exposed,
    and every verdict this surface can emit is registered in the exit map."""
    import tools.liveness.vpn_store as module
    from tools.liveness.report import EXIT_CODE_BY_VERDICT

    assert callable(module.main) and callable(module.render)
    assert hasattr(module, "CheckReport")
    for verdict in ('STORE_FRESH', 'STORE_STALE', 'VPN_REQUIRED', 'SKIP_NO_PACKAGE', 'UNREACHABLE'):
        assert verdict in EXIT_CODE_BY_VERDICT, verdict


def test_default_client_flattens_the_metadata_frame(monkeypatch):
    import sys
    import types

    class FakeFrame:
        def reset_index(self):
            return self

        def __getitem__(self, columns):
            return self

        def to_dict(self, orient):
            assert orient == "records"
            return [{"name": "fatalities003_2026_05_t01",
                     "min_month": 409, "max_month": 593}]

    db_ops = types.ModuleType("views_forecasts.db_ops")
    db_ops.ViewsMetadata = type(
        "ViewsMetadata", (), {"get_runs": lambda self: FakeFrame()}
    )
    package = types.ModuleType("views_forecasts")
    package.db_ops = db_ops
    monkeypatch.setitem(sys.modules, "views_forecasts", package)
    monkeypatch.setitem(sys.modules, "views_forecasts.db_ops", db_ops)
    rows = VpnStoreCheck._list_runs_via_views_forecasts()
    assert rows == [{"name": "fatalities003_2026_05_t01",
                     "min_month": 409, "max_month": 593}]

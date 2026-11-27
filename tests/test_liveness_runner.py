"""Liveness S7: shared report utilities + the aggregate runner — issue #245.

TDD for the WET->DRY consolidation. Extraction covers ONLY what >=2 checks
demonstrably duplicated: the fact renderer, the Appwrite API helpers
(credentials/fetch/query builders), and the verdict->exit-code map. The
aggregate runner composes the checks' mains: one raw-facts block per
surface, exit code = worst verdict, SKIP verdicts non-fatal.
"""

import pytest

from tools.liveness.report import (
    EXIT_CODE_BY_VERDICT,
    exit_code_for,
    render_facts,
    worst_exit,
)
from tools.liveness.__main__ import SURFACES, run_all

pytestmark = pytest.mark.green


# ── render_facts ──────────────────────────────────────────────────────

def test_render_facts_skips_none_and_formats_lines():
    text = render_facts([("a", 1), ("b", None), ("c", "x")])
    assert text == "a: 1\nc: x"


# ── the merged verdict map ────────────────────────────────────────────

def test_every_check_verdict_is_classified():
    ok = {"LIVE_FRESH", "INPUT_FRESH", "STORE_ACTIVE", "STORE_FRESH",
          "EXECUTION_CURRENT", "DELIVERING"}
    skip = {"SKIP_NO_CREDENTIALS", "SKIP_NO_PACKAGE", "VPN_REQUIRED"}
    warn = {"LIVE_STALE", "LIVE_NOT_SERVING", "INPUT_STALE", "STORE_IDLE",
            "STORE_STALE", "DELIVERY_STALLED", "EXECUTION_STALE",
            # #298: a `.env` exists but is incomplete. In `warn`, not `skip`, on
            # purpose — "nothing configured" is an honest absence of observation,
            # "half configured" is a fault a human must fix.
            "CREDENTIALS_INCOMPLETE"}
    fail = {"UNREACHABLE"}
    for verdict in ok | skip:
        assert exit_code_for(verdict) == 0, verdict
    for verdict in warn:
        assert exit_code_for(verdict) == 1, verdict
    for verdict in fail:
        assert exit_code_for(verdict) == 2, verdict
    assert set(EXIT_CODE_BY_VERDICT) == ok | skip | warn | fail

def test_unknown_verdict_fails_loud():
    with pytest.raises(KeyError):
        exit_code_for("MADE_UP_VERDICT")

def test_worst_exit():
    assert worst_exit([0, 0, 0]) == 0
    assert worst_exit([0, 1, 0]) == 1
    assert worst_exit([1, 2, 0]) == 2
    assert worst_exit([]) == 0


# ── the aggregate runner ──────────────────────────────────────────────

def test_registry_covers_every_surface():
    """Exact list, in order — a set comparison would not notice a surface going missing
    and being replaced by a new one, which is the regression that matters here.

    `crafd_delivery` joined 2026-08-24 (#413): the CRAF'd delivery was armed on 08-14 and
    had no instrument until then.
    """
    assert [name for name, _ in SURFACES] == [
        "old_api", "datafactory_input", "appwrite_store",
        "unfao_delivery", "crafd_delivery", "wandb_execution", "vpn_store",
    ]

def test_run_all_prints_blocks_and_returns_worst(capsys):
    fakes = [("alpha", lambda: 0), ("beta", lambda: 1), ("gamma", lambda: 0)]
    code = run_all(surfaces=fakes)
    out = capsys.readouterr().out
    assert code == 1
    assert "alpha" in out and "beta" in out and "gamma" in out

@pytest.mark.red
def test_run_all_unreachable_dominates(capsys):
    fakes = [("a", lambda: 1), ("b", lambda: 2)]
    assert run_all(surfaces=fakes) == 2

def test_run_all_all_green(capsys):
    fakes = [("a", lambda: 0), ("b", lambda: 0)]
    assert run_all(surfaces=fakes) == 0

@pytest.mark.red
def test_run_all_survives_a_crashing_check(capsys):
    def boom():
        raise RuntimeError("check exploded")
    fakes = [("a", lambda: 0), ("broken", boom), ("c", lambda: 0)]
    code = run_all(surfaces=fakes)
    out = capsys.readouterr().out
    assert code == 2                      # a crashing check is worst-class
    assert "check exploded" in out        # reported as a fact, not swallowed


# ── behavior preservation: every main still exists and is callable ──

def test_all_surface_mains_importable():
    for name, main_callable in SURFACES:
        assert callable(main_callable), name


@pytest.mark.beige
def test_structural_conventions_runner():
    """ADR-005 beige: registry structure — uniquely named surfaces, all
    runnable, in the canonical order."""
    names = [name for name, _ in SURFACES]
    assert names == ["old_api", "datafactory_input", "appwrite_store",
                     "unfao_delivery", "crafd_delivery", "wandb_execution", "vpn_store"]
    assert all(callable(run_main) for _, run_main in SURFACES)

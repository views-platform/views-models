"""The delivery coherence rules (ADR-019 §4, ADR-017 §5).

Every rule here is answerable **inside this repository, offline**. The two that are not —
`targets` and `coverage` — are deliberately absent; see `TestChecksThatDoNotRunHere`.

Failing cases are built from *real* sources making *wrong claims*, rather than from
invented source directories. A fixture that invents a model can drift from how the repo
actually spells things; a fixture that mis-claims a real one cannot.
"""

import warnings
from datetime import date

import pytest

from deliveries.coherence import (
    CoherenceError,
    check,
    maturity_of,
)
from deliveries.vocabulary import Delivery, Require, cm, live, monthly, months, paused, pgm, prod

pytestmark = pytest.mark.beige


def _delivery(send, *, reconciled=None, max_age=months(2), intent=None):
    return (
        Delivery(
            send=send,
            frequency=monthly,
            tier=prod,
            intent=intent or live(since=date(2026, 8, 4)),
        ),
        Require(reconciled=reconciled, max_age=max_age),
    )


# ── The migration mapping (ADR-017 §3) ─────────────────────────────────────


class TestMaturityMapping:
    """`maturity` does not exist yet — Phase 2 is cross-repo. The rules run against
    today's `deployment_status` with ADR-017's mapping applied in memory."""

    @pytest.mark.parametrize(
        "source,expected",
        [("rusty_bucket", "candidate"), ("skinny_love", "candidate")],
    )
    def test_shadow_maps_to_candidate(self, source, expected):
        assert maturity_of(source) == expected

    def test_deployed_maps_to_candidate_when_r2_would_fail(self):
        """ADR-017 §3: `deployed` → `graduate` **only if R2 holds, else candidate**.

        `white_mustang` is the repo's single `deployed` source and both its members are
        `shadow`. A straight rename would make it a graduate ensemble with candidate
        members — a violation of this ADR's own rule on the day it lands.
        """
        assert maturity_of("white_mustang") == "candidate"

    def test_unknown_source_fails_loudly(self):
        with pytest.raises(CoherenceError) as exc:
            maturity_of("no_such_source_anywhere")
        assert "models/" in str(exc.value) or "ensembles/" in str(exc.value)


# ── Resolution ─────────────────────────────────────────────────────────────


class TestResolution:
    def test_real_delivery_resolves(self):
        delivery, require = _delivery([pgm("rusty_bucket")])
        check(delivery, require, consumer="un_fao")

    def test_unknown_source_is_refused(self):
        delivery, require = _delivery([pgm("no_such_ensemble")])
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_fao")
        assert "no_such_ensemble" in str(exc.value)


# ── Level correspondence ───────────────────────────────────────────────────


class TestLevel:
    def test_matching_claim_passes(self):
        delivery, require = _delivery([pgm("rusty_bucket")])
        check(delivery, require, consumer="un_fao")

    def test_wrong_claim_is_refused(self):
        """`rusty_bucket` declares pgm. Claiming cm must fail, and name its config."""
        delivery, require = _delivery([cm("rusty_bucket")])
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_fao")
        message = str(exc.value)
        assert "config_meta.py" in message
        assert "pgm" in message and "cm" in message


# ── Reconciliation ─────────────────────────────────────────────────────────


class TestReconciliation:
    def test_connected_pair_passes(self):
        """`skinny_love` (pgm) declares reconcile_with `pink_ponyclub` (cm)."""
        delivery, require = _delivery(
            [pgm("skinny_love"), cm("pink_ponyclub")], reconciled=True
        )
        check(delivery, require, consumer="un_fao")

    def test_disconnected_group_is_refused(self):
        """`rude_boy` reconciles with nothing, so the two sources are not one group."""
        delivery, require = _delivery(
            [pgm("skinny_love"), cm("rude_boy")], reconciled=True
        )
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_fao")
        assert "rude_boy" in str(exc.value)

    def test_unreconciled_multi_source_is_a_hard_error(self):
        """ADR-019 §4: not supported — it silently permits a country total that
        disagrees with the sum of its cells."""
        delivery, require = _delivery(
            [pgm("skinny_love"), cm("pink_ponyclub")], reconciled=False
        )
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_fao")
        message = str(exc.value)
        assert "not currently supported" in message
        assert "deliveries/un_fao.py" in message, (
            "ADR-020: the error must name the file the reader has to open"
        )

    def test_single_source_needs_no_reconciliation(self):
        delivery, require = _delivery([pgm("rusty_bucket")], reconciled=False)
        check(delivery, require, consumer="un_fao")

    def test_unset_multi_source_is_the_same_hard_error_as_false(self):
        """`reconciled` is `bool | None` and defaults to `None`, and **no delivery in the
        platform sets it** — `un_fao.py` and `un_crafd.py` both omit the key. So unset is
        the state a two-source delivery lands in by simply not mentioning it, and it was
        the one state with no test (#420 HARD 1).

        `coherence.py` decides it with `if require.reconciled is not True`, so unset has
        always behaved as `False`. ADR-019 §4 now says so; this pins it, so the ADR and
        the checks cannot drift apart again — which is the whole subject of #420.
        """
        delivery, require = _delivery(
            [pgm("skinny_love"), cm("pink_ponyclub")], reconciled=None
        )
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_fao")
        message = str(exc.value)
        assert "not currently supported" in message
        assert "reconciled=None" in message, (
            "the message must show the value it actually found, or a reader who never "
            "wrote `reconciled` cannot tell which key the error is about"
        )

    def test_one_source_ignores_reconciled_whatever_it_says(self):
        """ADR-019 §4: with one source the key is not examined at all.

        Reconciliation is a property of a *combination*. Asserted for all three values
        because `True` being accepted here is the surprising one — it reads as a promise
        the checks never verify.
        """
        for value in (True, False, None):
            delivery, require = _delivery([pgm("rusty_bucket")], reconciled=value)
            check(delivery, require, consumer="un_fao")


# ── Freshness ──────────────────────────────────────────────────────────────


class TestFreshness:
    def test_live_without_max_age_is_refused(self):
        """ADR-019 §4. The absence of this bound is why a partner received nothing
        for 145 days while a complete forecast sat on the shelf (#320, C-121)."""
        delivery, require = _delivery([pgm("rusty_bucket")], max_age=None)
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_fao")
        assert "max_age" in str(exc.value)

    def test_paused_without_max_age_is_allowed(self):
        delivery, require = _delivery(
            [pgm("rusty_bucket")],
            max_age=None,
            intent=paused("shakedown pending", since=date(2026, 8, 4)),
        )
        check(delivery, require, consumer="un_fao")


# ── Tier, and the day-one transition ───────────────────────────────────────


class TestTierWarnsDuringTransition:
    def test_candidate_to_prod_warns_rather_than_blocks(self):
        """ADR-017 §11 "Day-one state": the platform inherits exactly this violation —
        `rusty_bucket` is a candidate delivering to the production-tier FAO consumer.

        It must **warn, not block**, until the real production ensemble graduates. This
        test pins the warning, so a change that turns it into a hard failure is caught
        rather than discovered when the delivery stops.
        """
        delivery, require = _delivery([pgm("rusty_bucket")])
        with pytest.warns(UserWarning, match="candidate"):
            check(delivery, require, consumer="un_fao")

    def test_every_real_delivery_file_still_passes(self):
        """**Every** declaration must be checkable, not just the one we remembered.

        Until 2026-08-11 this asserted `deliveries/un_fao.py` by name, and every other
        `check()` call site in the suite used a synthetic fixture. So a second consumer
        could ship a flatly incoherent declaration and the suite would stay green — the
        gap was invisible for exactly as long as there was only one consumer, which is
        the worst time to notice it (#333).

        Discovery is by `delivery_files()`, the same glob production uses, so a new
        consumer is covered the moment its file exists.
        """
        from deliveries.status import delivery_files, load_delivery

        files = list(delivery_files())
        assert files, "no delivery declarations discovered — this test asserts nothing"

        for path in files:
            module = load_delivery(path)
            consumer = path.stem
            # `rusty_bucket` is `candidate` and both consumers are prod tier, so a
            # UserWarning is expected today (ADR-017 §11 day-one state). What must not
            # happen is a CoherenceError.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    check(module.DELIVERY, module.REQUIRE, consumer=consumer)
                except Exception as exc:  # noqa: BLE001 — any refusal is the finding
                    raise AssertionError(
                        f"deliveries/{consumer}.py is not coherent: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc


# ── What is deliberately not checked here ──────────────────────────────────


class TestChecksThatDoNotRunHere:
    """ADR-020 §4: two stairs end outside this repository. Their absence is a decision."""

    def test_targets_are_not_gated_at_edit_time(self):
        """A `targets` gate today would reject a *correct* delivery file.

        `rusty_bucket` declares `lr_*_best` in `regression_targets` and emits
        `lr_ged_*` (register C-123). Checking the delivery's targets against the
        source config would fail the repo's own FAO ensemble — and the first thing
        this repository would teach a newcomer is that its errors are wrong (C-125).
        """
        delivery, require = _delivery([pgm("rusty_bucket")])
        require = Require(targets=("not_a_real_target",), max_age=months(2))
        with pytest.warns(UserWarning):
            check(delivery, require, consumer="un_fao")  # must not raise

    def test_coverage_is_not_gated_at_edit_time(self):
        """Cell counts live in views-postprocessing, beside the GAUL asset."""
        delivery, _ = _delivery([pgm("rusty_bucket")])
        require = Require(coverage="not_a_real_region", max_age=months(2))
        with pytest.warns(UserWarning):
            check(delivery, require, consumer="un_fao")  # must not raise


class TestCycleGuard:
    def test_self_containing_ensemble_fails_with_a_message(self, tmp_path, monkeypatch):
        """A `deployed` ensemble that contains itself must name the file, not
        RecursionError. ADR-020: no error may end in a stack trace the reader
        cannot act on."""
        import deliveries.coherence as coh

        monkeypatch.setattr(coh, "require_source", lambda name: tmp_path / name)
        monkeypatch.setattr(
            coh, "source_config",
            lambda src, which: {"deployment_status": "deployed"} if which == "deployment"
            else {"models": ["loopy"]} if which == "modelset" else {},
        )
        with pytest.raises(coh.CoherenceError) as exc:
            coh.maturity_of("loopy")
        assert "config_modelset.py" in str(exc.value)
        assert "itself" in str(exc.value)

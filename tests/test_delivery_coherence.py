"""The delivery coherence rules (ADR-019 §4, ADR-017 §5).

Every rule here is answerable **inside this repository, offline**. The two that are not —
whether a target *exists*, and what a coverage region contains — are deliberately absent;
see `TestChecksThatDoNotRunHere`. The coverage *rule* added in #428 is a different thing
with a confusingly similar name: it compares two declarations inside one delivery file.
See `TestTargetCoverage`.

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


def _delivery(send, *, reconciled=None, max_age=months(2), intent=None, targets=()):
    return (
        Delivery(
            send=send,
            frequency=monthly,
            tier=prod,
            intent=intent or live(since=date(2026, 8, 4)),
        ),
        Require(reconciled=reconciled, max_age=max_age, targets=targets),
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
    """ADR-020 §4: two stairs end outside this repository. Their absence is a decision.

    **#428 changed what this class is asserting, and the change is narrow.** There is now
    a rule that reads `REQUIRE.targets` (`_check_target_coverage`, `TestTargetCoverage`) — but it
    compares that tuple against the `provides=` written beside it in the same file. What
    is still not checked, and is what these two tests pin, is anything that would require
    opening a *source config* or a *run*: whether a target exists, and what cells a
    coverage region contains. Both cases below use one source, where the coverage rule
    does not apply at all.
    """

    def test_target_existence_is_not_gated_at_edit_time(self):
        """A gate on whether a target is *real* would reject a *correct* delivery file.

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



# ── Coverage: which source answers for which target ────────────────────────


class TestTargetCoverage:
    """ADR-019 §4, #428. The *other* reason a delivery names several sources.

    Named `TargetCoverage`, not `Coverage`: `Require.coverage` is an unrelated key (a
    GAUL region), and `TestChecksThatDoNotRunHere` is where *that* one is pinned.

    Reconciliation says the sources agree with each other about one target. Coverage
    says that between them they carry the targets asked for. Until `provides` (#427)
    there was no way to say which, so the composition `un_crafd` needs — three targets,
    no reconciling ensemble that carries three — could not be written down.

    These deliveries all set `reconciled=True`, because `_check_reconciliation` still
    refuses any two-source delivery that does not. Splitting the two rules apart is
    #429 (S5); this story only adds the coverage half. The order in `check()` puts
    coverage first, so these cases are decided on their own terms either way.
    """

    def test_a_target_nobody_claims_is_refused(self):
        delivery, require = _delivery(
            [pgm("skinny_love", provides=("lr_ged_sb",)),
             cm("pink_ponyclub", provides=("lr_ged_ns",))],
            reconciled=True,
            targets=("lr_ged_sb", "lr_ged_ns", "lr_ged_os"),
        )
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_crafd")
        message = str(exc.value)
        assert "lr_ged_os" in message, "the message must name the missing target"
        assert "deliveries/un_crafd.py" in message, "ADR-020: name the file to open"
        assert "provides=" in message, "say what to write, not only what is wrong"

    def test_two_sources_claiming_one_target_at_one_level_is_refused(self):
        delivery, require = _delivery(
            [pgm("skinny_love", provides=("lr_ged_sb",)),
             pgm("white_mustang", provides=("lr_ged_sb",))],
            reconciled=True,
            targets=("lr_ged_sb",),
        )
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_crafd")
        message = str(exc.value)
        assert "skinny_love" in message and "white_mustang" in message, (
            "naming the target alone leaves the reader grepping for who else claimed it"
        )
        assert "lr_ged_sb" in message

    def test_the_same_target_at_two_levels_is_the_reconciliation_case_and_passes(self):
        """ADR-017 §3: a pgm source and a cm source answering for the same target is
        exactly what reconciliation *is*. Refusing it here would forbid `un_fao`'s own
        intended shape."""
        delivery, require = _delivery(
            [pgm("skinny_love", provides=("lr_ged_sb",)),
             cm("pink_ponyclub", provides=("lr_ged_sb",))],
            reconciled=True,
            targets=("lr_ged_sb",),
        )
        check(delivery, require, consumer="un_fao")

    def test_a_complete_two_source_split_passes(self):
        delivery, require = _delivery(
            [pgm("skinny_love", provides=("lr_ged_sb", "lr_ged_ns")),
             cm("pink_ponyclub", provides=("lr_ged_os",))],
            reconciled=True,
            targets=("lr_ged_sb", "lr_ged_ns", "lr_ged_os"),
        )
        check(delivery, require, consumer="un_crafd")

    def test_annotating_some_sources_but_not_others_is_refused(self):
        """Not in #428's table — decided here, and the reason is in the message.

        An un-annotated source claims everything it contains, so it overlaps whatever
        the others claim: the duplicate check goes vacuous and the coverage check passes
        for the wrong reason. The realistic slip is adding a second source and only
        annotating the new one.
        """
        delivery, require = _delivery(
            [pgm("skinny_love"),
             cm("pink_ponyclub", provides=("lr_ged_os",))],
            reconciled=True,
            targets=("lr_ged_sb", "lr_ged_os"),
        )
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_crafd")
        message = str(exc.value)
        assert "skinny_love" in message, "name the source that was left un-annotated"
        assert "remove them all" in message, "both ways out, not just one"


class TestTargetCoverageDoesNotApply:
    """The rule must stay invisible to every delivery that exists today."""

    def test_one_source_is_untouched_even_with_provides_narrower_than_targets(self):
        """#428: with one source the rule does not apply — and the reason is not
        deference, it is C-123.

        With nowhere else a target could come from, `provides` narrower than `targets`
        is no longer a claim about the *division of labour*; it is a claim about what
        this one source contains. That is the check the module deliberately does not
        make, because `rusty_bucket` declares `lr_*_best` and emits `lr_ged_*`, so it
        would refuse a correct file.

        Found by mutation: relaxing the guard to `< 1` passed every test here, because
        the one-source case was only ever asserted *without* `provides` and the
        all-omitted early exit was catching it instead.
        """
        delivery, require = _delivery(
            [pgm("rusty_bucket", provides=("lr_ged_sb",))],
            targets=("lr_ged_sb", "lr_ged_ns", "lr_ged_os"),
        )
        with pytest.warns(UserWarning):
            check(delivery, require, consumer="un_fao")

    def test_one_source_with_no_provides_is_the_shape_every_real_delivery_has(self):
        delivery, require = _delivery(
            [pgm("rusty_bucket")],
            targets=("lr_ged_sb", "lr_ged_ns", "lr_ged_os"),
        )
        with pytest.warns(UserWarning):
            check(delivery, require, consumer="un_fao")

    def test_two_sources_with_no_provides_at_all_are_untouched(self):
        """Omitted means "every target this source contains" (ADR-019 §3), so nothing
        is claimed exclusively and there is nothing to be inconsistent about. This is
        the shape every two-source delivery had before #427."""
        delivery, require = _delivery(
            [pgm("skinny_love"), cm("pink_ponyclub")],
            reconciled=True,
            targets=("lr_ged_sb", "lr_ged_ns", "lr_ged_os"),
        )
        check(delivery, require, consumer="un_fao")

    def test_with_no_required_targets_nothing_can_be_missing(self):
        """`Require.targets` defaults to `()` — and both real deliveries set it, but a
        delivery need not. With nothing required, the coverage half has no question to
        ask, and refusing here would make `provides` unusable in such a file."""
        delivery, require = _delivery(
            [pgm("skinny_love", provides=("lr_ged_sb",)),
             cm("pink_ponyclub", provides=("lr_ged_os",))],
            reconciled=True,
        )
        check(delivery, require, consumer="un_fao")

    def test_but_a_same_level_duplicate_is_still_refused_with_no_targets(self):
        """The two halves of this rule are gated differently, and that is deliberate.

        Coverage asks "did anyone answer for what was required?" and needs `targets`.
        Duplication asks "did two sources answer for the same thing?" and does not —
        two sources contradicting each other at one level is wrong whether or not
        anybody asked for that target. Pinned because the tidy-looking simplification
        is to gate the whole rule on `targets`, which would silently drop this half.
        """
        delivery, require = _delivery(
            [pgm("skinny_love", provides=("lr_ged_sb",)),
             pgm("white_mustang", provides=("lr_ged_sb",))],
            reconciled=True,
        )
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_fao")
        assert "lr_ged_sb" in str(exc.value)



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

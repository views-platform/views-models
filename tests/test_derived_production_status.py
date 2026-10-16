""""In production" is derived, never declared (ADR-017 §4e).

    A source is *in production* ⟺ its maturity is `graduate` **and** a delivery ships
    it (directly, or via a composite that contains it) to a **production-tier** consumer.

Two things this file is careful about.

**Transitivity.** ADR-017 §4a: a model inside a delivered ensemble is in production
*transitively*. The derivation has to walk `config_modelset.py`, not just look at what
`send` names.

**Not over-claiming.** The report says what is *declared*. It observes nothing. A
declared-live delivery that nothing ever runs is invisible to it — that is ADR-020 §4's
"hole in the floor" (register C-126), and it is the failure that actually happened
(#320). Wording that implied otherwise would be the truthfulness bug ADR-005 and C-75
exist to prevent, so a test pins that the report says so out loud.
"""

from pathlib import Path

import pytest

from deliveries.status import (
    declared_source,
    delivered_sources,
    in_production,
    report,
)

pytestmark = pytest.mark.beige

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── The derivation ─────────────────────────────────────────────────────────


class TestDerivation:
    def test_candidate_source_is_not_in_production(self):
        """`rusty_bucket` is delivered to FAO but is `candidate` (was `shadow`),
        so the first conjunct fails. It is delivered, not in production."""
        assert in_production("rusty_bucket") is False

    def test_undelivered_source_is_not_in_production(self):
        """`pink_ponyclub` runs monthly and ships to the public API, but no
        `deliveries/*.py` names it, so the second conjunct fails."""
        assert in_production("pink_ponyclub") is False

    def test_both_conjuncts_are_required(self):
        """Neither maturity nor a delivery edge alone is enough."""
        for source in ("rusty_bucket", "pink_ponyclub", "white_mustang"):
            assert in_production(source) is False

    def test_transitive_membership_is_followed(self):
        """ADR-017 §4a: a model inside a delivered ensemble is in production
        *transitively*. Today no ensemble qualifies, so no member does either —
        but the walk must actually happen, not be skipped."""
        members = delivered_sources()
        assert "rusty_bucket" in members, (
            "rusty_bucket is named by deliveries/un_fao.py and must appear"
        )
        assert len(members) > 1, (
            "delivered_sources() returned only the directly-named source — the walk "
            "into config_modelset.py did not happen"
        )

    def test_unknown_source_fails_loudly(self):
        from deliveries.coherence import CoherenceError

        with pytest.raises(CoherenceError):
            in_production("no_such_source_anywhere")


# ── The value is computed, not stored ──────────────────────────────────────



class TestDeclaredSourceRefusesSeveral:
    """#430. A postprocessor config carries exactly one source, and the reason changed.

    The refusal used to say several sources "needs ADR-019 §4's reconciliation rules,
    which the postprocessor does not implement". Since #429 §4 permits several — one
    reconciliation group plus any source present solely to provide targets no other source
    provides. So the message named a rule that no longer forbids anything, while the real
    limit sat in another repository: `views_postprocessing` reads `configs["ensemble"]` as
    a single string and has no key for a list.

    The refusal is right and stays. Picking the first would be a silent choice about which
    forecast reaches an external partner. What it owes the reader is the truth about where
    it is fixed.
    """

    def _two_source_delivery(self, tmp_path):
        """A real delivery file with a second source added, written somewhere harmless."""
        import shutil

        source = REPO_ROOT / "deliveries" / "un_crafd.py"
        body = source.read_text()
        assert 'send      = [pgm("rusty_bucket")],' in body, (
            "deliveries/un_crafd.py no longer has the send line this test rewrites"
        )
        body = body.replace('send      = [pgm("rusty_bucket")],',
                            'send      = [pgm("rusty_bucket"), cm("pink_ponyclub")],', 1)
        body = body.replace("Delivery, Require, pgm, live,", "Delivery, Require, cm, pgm, live,", 1)
        shutil.copytree(REPO_ROOT / "deliveries", tmp_path / "deliveries")
        (tmp_path / "deliveries" / "two_sources.py").write_text(body)
        return tmp_path / "deliveries"

    def test_it_refuses_and_says_where_the_limit_actually_is(self, tmp_path, monkeypatch):
        import deliveries.status as status

        monkeypatch.setattr(status, "DELIVERIES_DIR", self._two_source_delivery(tmp_path))
        with pytest.raises(ValueError) as exc:
            declared_source("two_sources")
        message = str(exc.value)
        assert "rusty_bucket" in message and "pink_ponyclub" in message, (
            "name the sources it found — the reader has to know which two"
        )
        assert "views_postprocessing" in message, (
            "the limit is one repository away; naming ADR-019 §4 instead sends the reader "
            "to a rule that has permitted this since #429"
        )
        assert "ADR-019 §4 permits several sources" in message, (
            "say plainly that the delivery side is not what refused, or the reader "
            "edits the wrong file"
        )
        assert "Ask Simon" in message, (
            "ADR-020 §5: a check that cannot be answered here is a locked door, and a "
            "locked door names the person and supplies the request"
        )

    def test_it_still_refuses_rather_than_picking_the_first(self, tmp_path, monkeypatch):
        """The whole point. A silent choice here delivers a different forecast to an
        external partner and says nothing."""
        import deliveries.status as status

        monkeypatch.setattr(status, "DELIVERIES_DIR", self._two_source_delivery(tmp_path))
        with pytest.raises(ValueError):
            declared_source("two_sources")

    def test_one_source_is_unchanged(self):
        assert declared_source("un_fao") == "rusty_bucket"
        assert declared_source("un_crafd") == "rusty_bucket"



class TestNothingStoresTheAnswer:
    def test_no_config_declares_is_in_production(self):
        """ADR-017 §4e: "it can't lie, because there's no field to lie in."

        If someone adds the field to a config, the derivation stops being the only
        answer and the label can drift from reality — which is the entire defect
        this ADR was written to remove.
        """
        offenders = []
        for base in ("models", "ensembles", "postprocessors", "deliveries"):
            for path in (REPO_ROOT / base).rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(text.splitlines(), 1):
                    if "is_in_production" in line and "def " not in line and "import" not in line:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{i}")
        assert not offenders, (
            f"'is_in_production' appears as data, not a derivation: {offenders}\n"
            f"  ADR-017 §4e: the value is worked out on demand and never stored."
        )


# ── The report ─────────────────────────────────────────────────────────────


class TestReport:
    def test_names_every_declared_field(self):
        text = report()
        for field in ("frequency", "tier", "intent", "un_fao", "rusty_bucket"):
            assert field in text, f"the report does not mention {field!r}"

    def test_frequency_has_a_consumer(self):
        """ADR-019 §3 makes `frequency` required. A required key nothing reads is
        decoration. `monthly_run.sh` is deliberately not changed by this epic, so
        the report is what gives the key a reader."""
        assert "monthly" in report()

    def test_says_plainly_that_it_observed_nothing(self):
        """C-126 / ADR-020 §4: a declared-live delivery that nothing runs produces
        no error, because nothing failed. The report must not imply it checked."""
        text = report().lower()
        assert "declared" in text
        assert any(
            phrase in text
            for phrase in ("not observed", "does not observe", "no observation")
        ), "the report must say it reports declarations, not observations"

    def test_points_at_the_instrument_that_does_observe(self):
        """ADR-017 §7: the declaration side is here, the verification side is
        `tools/liveness`. A reader asking "did it actually ship?" needs the pointer."""
        assert "tools.liveness" in report() or "tools/liveness" in report()

    def test_runs_offline(self):
        """No network, no Appwrite. This reads files only."""
        report()

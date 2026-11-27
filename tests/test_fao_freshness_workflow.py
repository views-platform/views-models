"""The FAO freshness heartbeat must be able to fail (D7, #320).

A workflow nobody tests is how `update_catalogs` accumulated **eight consecutive
failures without anyone noticing** (#336). This one guards a delivery that already
went dark for 145 days, so the ways it could quietly stop working are worth pinning.

Three properties, and the second is the one that matters:

1. It is **scheduled** — a heartbeat that only runs when someone pushes is not a
   heartbeat, and the thing being detected is precisely that nobody looked.
2. It classifies on the **verdict**, never on the exit code. `SKIP_NO_CREDENTIALS`
   maps to exit 0 in `tools/liveness/report.py`, so a job keyed on the exit code
   would report success having read nothing — a dead heartbeat indistinguishable
   from a healthy one. Same defect class as C-113 and C-114.
3. It **fails** on a missing-credentials skip rather than passing.

These are static checks on the workflow file. They cannot prove GitHub runs it, but
they can prove it was not quietly rewritten into something that always passes.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.beige

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "fao_freshness.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    yaml = pytest.importorskip(
        "yaml", reason="PyYAML not installed — cannot parse the workflow (truthful skip)"
    )
    assert WORKFLOW.exists(), (
        f"{WORKFLOW.relative_to(REPO_ROOT)} is missing.\n"
        f"  D7 of the release requires a scheduled check that fails when the FAO "
        f"delivery goes stale."
    )
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def body() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


class TestItRuns:
    def test_it_is_scheduled(self, workflow):
        # PyYAML parses a bare `on:` key as the boolean True.
        triggers = workflow.get("on", workflow.get(True, {}))
        assert "schedule" in triggers, (
            "the heartbeat has no schedule. A check that only runs on push cannot "
            "detect that nobody has run anything — which is the failure it exists for."
        )

    def test_it_can_be_triggered_by_hand(self, workflow):
        """So it can be *proven* to fire, rather than believed to."""
        triggers = workflow.get("on", workflow.get(True, {}))
        assert "workflow_dispatch" in triggers

    def test_it_reads_the_check_this_repo_already_has(self, body):
        assert "tools.liveness.unfao_delivery" in body


class TestItCanActuallyFail:
    """The properties that stop it becoming a green light nobody questions."""

    def test_it_classifies_on_the_verdict_not_the_exit_code(self, body):
        assert "verdict" in body, "the job must read the check's verdict"
        assert "DELIVERING)" in body, (
            "the job must name the passing verdict explicitly — an allow-list, so an "
            "unrecognised verdict fails rather than slipping through"
        )

    def test_a_missing_credential_skip_fails_rather_than_passes(self, body):
        """`SKIP_NO_CREDENTIALS` is exit 0. For this job it must be a failure."""
        assert "SKIP_NO_CREDENTIALS)" in body, (
            "the job does not handle SKIP_NO_CREDENTIALS. It maps to exit 0, so "
            "without an explicit branch the heartbeat passes having read nothing."
        )
        skip_branch = body.split("SKIP_NO_CREDENTIALS)")[1].split(";;")[0]
        assert "exit 1" in skip_branch, (
            "SKIP_NO_CREDENTIALS does not fail the job. A heartbeat that goes green "
            "without reading anything is worse than no heartbeat."
        )

    def test_an_unrecognised_verdict_fails(self, body):
        catchall = body.split("*)")[-1]
        assert "exit 1" in catchall, (
            "an unknown verdict does not fail the job. A gate that cannot tell "
            "'passed' from 'did not run' is not a gate (C-113)."
        )

    def test_the_stale_branch_fails(self, body):
        stale_branch = body.split("DELIVERY_STALLED)")[1].split(";;")[0]
        assert "exit 1" in stale_branch


class TestItDoesNotSetItsOwnBound:
    def test_the_threshold_is_not_written_in_the_workflow(self, body):
        """#360 collapsed two thresholds into one. A number here would make three."""
        import re

        for match in re.finditer(r"\b(\d+)\s*(?:days?|DAYS?)\b", body):
            assert match.group(1) not in ("45", "60"), (
                f"the workflow names a freshness bound ({match.group(0)!r}). The bound "
                f"is declared in deliveries/un_fao.py and read by the check — a copy "
                f"here re-creates the defect #360 removed."
            )

    def test_it_says_where_the_bound_lives(self, body):
        assert "deliveries/un_fao.py" in body, (
            "an operator reading a failure must be told which file declares the bound"
        )


class TestSecrets:
    def test_it_requests_exactly_the_three_appwrite_secrets(self, body):
        for name in (
            "APPWRITE_ENDPOINT",
            "APPWRITE_DATASTORE_PROJECT_ID",
            "APPWRITE_DATASTORE_API_KEY",
        ):
            assert f"secrets.{name}" in body, f"{name} is not passed to the check"

    def test_no_secret_value_is_echoed(self, body):
        """The check redacts its own credentials; the workflow must not undo that."""
        assert "echo \"$APPWRITE" not in body
        assert "${{ secrets." in body  # referenced only through the env block


class TestEveryVerdictHasABranch:
    """The catch-all must be unreachable in practice.

    It still fails, so safety does not depend on this — but an operator who hits a
    real condition deserves a message naming the fix rather than the word
    "inconclusive". A first live run produced CREDENTIALS_INCOMPLETE, which had no
    branch and fell through; this test exists so the next new verdict does not.
    """

    def test_all_verdicts_this_check_can_emit_are_handled(self, body):
        from tools.liveness import unfao_delivery  # noqa: F401
        import inspect
        import re

        source = inspect.getsource(unfao_delivery)
        emitted = set(re.findall(r'verdict\s*=\s*"([A-Z_]+)"', source))
        emitted |= set(re.findall(r'verdict="([A-Z_]+)"', source))
        # Verdicts the shared credential helper can return on this surface.
        emitted |= {"SKIP_NO_CREDENTIALS", "CREDENTIALS_INCOMPLETE"}

        unhandled = sorted(v for v in emitted if f"{v})" not in body)
        assert not unhandled, (
            f"these verdicts have no branch in the workflow: {unhandled}\n"
            f"  They would fall to the catch-all and be reported as 'inconclusive', "
            f"when the check already knows exactly what is wrong."
        )

"""No model may declare the purged `africa_me_legacy` scope, and this check proves it looked.

⛔ WHY THIS IS NOT IN `test_roster_conformance.py`. That file already asserts
`REGION == "land"` — correctly — but it is parametrised over `ROSTER`, the eight HydraNets.
On 2026-11-02 `models/light_strider` was found still declaring `africa_me_legacy`, six weeks
after `455b9674` moved the fleet to global land. It is a `ConflictologyModel`, so it was never
in `ROSTER`, so no check ever looked at it. **The guard did not fail; its subject was out of
scope** — ADR-025's exact shape. A roster-scoped check cannot see a straggler by construction,
so this one is scoped to *every config that declares a REGION* and takes no roster as input.

⛔ WHY `ast` AND NOT A REGEX OR A SUBSTRING. The assertion this replaces was once
`"africa_me_legacy" in text`, which the comment recording the flip satisfied — the defect's
vocabulary and its refutation live in the same file (C-189, seven instances). A regex anchored
to `^REGION` is better but still reads text. `ast` reads the *assignment*, so a comment,
a docstring, or a historical note mentioning the scope cannot satisfy or break it. The
`light_strider` config deliberately still names `africa_me_legacy` in its comment, recording
what it used to be, and this test must pass over that.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PURGED = "africa_me_legacy"

#: Below this, the test is not measuring the fleet — it is measuring its own glob.
#: 58 configs declared a REGION on 2026-11-02 (42 `land`, 16 `land_gaul`); the floor is set
#: well under that so adding or retiring a model does not fail the suite, while a glob that
#: silently stops matching does.
MIN_CONFIGS_WITH_REGION = 40


def _region_of(path: Path) -> str | None:
    """The value of the module-level `REGION = "..."` assignment, or None if there is none."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "REGION":
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    return node.value.value
                # A computed REGION would defeat this check silently, so refuse it.
                pytest.fail(f"{path}: REGION is not a string literal, so it cannot be audited")
    return None


def _declared_regions() -> dict[Path, str]:
    found: dict[Path, str] = {}
    for path in sorted(REPO.glob("models/*/configs/config_queryset.py")):
        region = _region_of(path)
        if region is not None:
            found[path] = region
    return found


def test_the_check_can_see_the_fleet():
    """⛔ ADR-025: a check that cannot read its subject must FAIL, not pass vacuously.

    Without this, a renamed directory or a changed config filename turns every assertion
    below into a no-op over an empty set, and the suite goes green on zero evidence.
    """
    found = _declared_regions()
    assert len(found) >= MIN_CONFIGS_WITH_REGION, (
        f"only {len(found)} configs declare a REGION; expected at least "
        f"{MIN_CONFIGS_WITH_REGION}. The glob, not the fleet, is what this measured."
    )


def test_no_config_declares_the_purged_scope():
    """`africa_me_legacy` was purged on 2026-11-01 and must not come back via a config."""
    offenders = {p.relative_to(REPO).as_posix(): r for p, r in _declared_regions().items() if r == PURGED}
    assert not offenders, (
        "these configs would re-pull the purged africa_me_legacy scope on their next run: "
        f"{sorted(offenders)}"
    )


def test_a_historical_mention_in_a_comment_does_not_trip_the_check():
    """The anti-C-189 property, asserted rather than assumed.

    `light_strider`'s config records `africa_me_legacy` in a comment on purpose. If this test
    ever starts reading text instead of the AST, this assertion fails — which is the point.
    """
    path = REPO / "models/light_strider/configs/config_queryset.py"
    assert path.exists(), f"{path} is gone; update or retire this test"
    assert PURGED in path.read_text(), (
        "this test's subject is a config whose COMMENT names the purged scope; "
        "if that note was removed, point this test at another one or delete it"
    )
    assert _region_of(path) == "land"

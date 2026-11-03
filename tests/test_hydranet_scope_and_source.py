"""Every HydraNet pulls the `land` scope from the VIEWS data factory, and nothing else.

Both values are **policy, not emergent agreement**, which is why this file pins the literals
`"land"` and `"views-datafactory"` rather than deriving them from the majority the way
`test_hydranet_grid_column_names.py` derives the grid column name. The distinction matters:

  - A grid *column name* is whatever the producer emits. If the factory renamed it platform-wide,
    all 27 models should move together and that is not a defect — so the expected value is the
    majority, and the failure is one model drifting away from its siblings.
  - A *scope* and a *data source* are chosen. If all 27 drifted to `land_gaul` or back to viewser
    together, a majority-derived check would still pass while the thing it exists to protect was
    gone. So the expected value is written down.

⛔ Read the dict `generate()` RETURNS, not the module-level `REGION` constant. A config can
declare `REGION = "land"` and return something else in the descriptor; `generate()`'s dict is what
the loader consumes (via `get_queryset()`), so that is where the value has to be right.

⛔ AND READ IT WITH `ast`, NOT BY IMPORTING THE MODULE. Every HydraNet queryset opens with
`from datafactory_query.defaults import DEFAULT_REMOTE`, and CI does not install
`views-datafactory` — the executing version of this file failed **54 times** in CI with
`ModuleNotFoundError: No module named 'datafactory_query'` while passing locally, where the
package happens to be present. `tests/test_no_region_is_the_purged_scope.py` already uses `ast`
and states the reason: it reads the *assignment*, so a comment or docstring naming a value cannot
satisfy it. The same choice buys both properties here — comment-proof, and runnable where the
dependency is absent.

⚠️ The discovery is duplicated from `test_hydranet_grid_column_names.py` on purpose. Two five-line
globs asking different questions are cheaper to read than a shared `tests/helpers.py`, and this
repo has no such module by policy.
"""

from __future__ import annotations

import ast
import glob
from pathlib import Path

import pytest

pytestmark = pytest.mark.beige

REPO = Path(__file__).resolve().parents[1]

#: ⛔ POLICY, not a measurement — see the module docstring for why these are literals.
REQUIRED_REGION = "land"
REQUIRED_SOURCE = "views-datafactory"

#: Below this, the glob has broken rather than the fleet having shrunk (ADR-025).
MIN_HYDRANETS = 20


def _hydranets() -> list[str]:
    return sorted(
        Path(p).parts[-3]
        for p in glob.glob(str(REPO / "models" / "*" / "configs" / "config_meta.py"))
        if '"algorithm": "HydraNet"' in Path(p).read_text(encoding="utf-8")
    )


def _descriptor(name: str) -> dict[str, str]:
    """The string-valued entries of the dict `generate()` returns, read without importing it.

    Walks to the `return {...}` inside `def generate()` and keeps entries whose key and value are
    both plain string literals — which is all this file asserts on. An entry whose value is a name
    or a call is skipped rather than guessed at, so the assertions below report it as absent
    rather than silently passing.
    """
    path = REPO / "models" / name / "configs" / "config_queryset.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    # Module-level `NAME = "literal"` bindings, so a descriptor entry written as
    # `"region": REGION` resolves. Every HydraNet queryset is written that way, and resolving it
    # is stronger than reading REGION directly: it proves the constant actually reaches the
    # descriptor rather than merely existing in the file.
    consts: dict[str, str] = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    def _literal(value: ast.expr) -> str | None:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        if isinstance(value, ast.Name):
            return consts.get(value.id)
        return None

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "generate"):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Dict):
                out: dict[str, str] = {}
                for k, v in zip(inner.value.keys, inner.value.values):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        resolved = _literal(v)
                        if resolved is not None:
                            out[k.value] = resolved
                return out
    raise AssertionError(
        f"no `return {{...}}` dict literal found inside generate() in {path}. ⛔ Failing rather "
        "than reporting a clean result: this check cannot see its subject (ADR-025)."
    )


HYDRANETS = _hydranets()


def test_the_discovery_found_the_fleet():
    """⛔ ADR-025: a check that cannot see its subject must FAIL, not pass over an empty list."""
    assert len(HYDRANETS) >= MIN_HYDRANETS, (
        f"only {len(HYDRANETS)} HydraNets discovered, expected at least {MIN_HYDRANETS}. "
        "The selector is a raw substring match on '\"algorithm\": \"HydraNet\"' in config_meta.py; "
        "a change in spacing or quoting there would empty this list and make every assertion "
        "below vacuously true. Fix the selector, do not lower the floor."
    )


@pytest.mark.parametrize("name", HYDRANETS)
def test_a_hydranet_pulls_the_land_scope(name):
    """Parametrized per model so a failure names the offender, not just a count."""
    region = _descriptor(name).get("region")
    assert region == REQUIRED_REGION, (
        f"{name} pulls region={region!r}, not {REQUIRED_REGION!r}. "
        f"`land` is 64,818 PRIO-GRID land cells; `land_gaul` is the FAO delivery's cut of it and "
        f"is a different fleet of cells, so a model on the wrong one is not comparable with its "
        f"siblings even when every other setting matches."
    )


@pytest.mark.parametrize("name", HYDRANETS)
def test_a_hydranet_reads_the_data_factory(name):
    """The descriptor's own `source` field, which is what the loader dispatches on."""
    source = _descriptor(name).get("source")
    assert source == REQUIRED_SOURCE, (
        f"{name} declares source={source!r}, not {REQUIRED_SOURCE!r}. "
        "The viewser path reaches PRIO's PostgreSQL and the factory path reads the zarr store; "
        "they do not return the same frame shape or the same index names."
    )


@pytest.mark.parametrize("name", HYDRANETS)
def test_a_hydranet_queryset_does_not_import_viewser(name):
    """Declaring the factory while importing viewser is the look-alike case of C-157.

    The `source` field above is a string a config can set while still reaching for the old
    client, so this asserts the import is gone and not merely the label changed.
    """
    path = REPO / "models" / name / "configs" / "config_queryset.py"
    offending = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith(("import viewser", "from viewser"))
    ]
    assert not offending, (
        f"{name}'s queryset still imports viewser while declaring {REQUIRED_SOURCE!r}: {offending}"
    )

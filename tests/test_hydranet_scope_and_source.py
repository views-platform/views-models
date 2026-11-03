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

⛔ Read through `generate()`, not off the module-level `REGION` constant. A config can declare
`REGION = "land"` and return something else in the descriptor; `generate()`'s dict is what the
loader actually consumes (`views_pipeline_core.managers.model` reads it via `get_queryset()`), so
that is where the value has to be right.

⚠️ The discovery is duplicated from `test_hydranet_grid_column_names.py` on purpose. Two five-line
globs asking different questions are cheaper to read than a shared `tests/helpers.py`, and this
repo has no such module by policy.
"""

from __future__ import annotations

import glob
import importlib.util
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


def _descriptor(name: str) -> dict:
    path = REPO / "models" / name / "configs" / "config_queryset.py"
    spec = importlib.util.spec_from_file_location(f"qs_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate()


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

"""Datafactory descriptors must SOURCE raw registry feature names (#163, EPIC #154).

The VIEWS data factory serves raw UCDP column names (`ged_sb_best`, `ged_ns_best`,
`ged_os_best`, …). The renaming to a model-internal name happens **on arrival**,
in each model's own ``config_queryset.py`` (the ``FEATURE_RENAME`` *values*) — never
in shared datafactory infrastructure. The "consumer bridge" that used to rename
``ged_*_best -> lr_*_best`` centrally is being removed from views-datafactory.

This guard locks in "source raw, rename locally" so that removal is provably safe
for views-models: it asserts that no datafactory descriptor *sources* a
consumer-bridge name (``lr_*``) or a viewser variant (``*_sum_nokgi``). It is
**name-agnostic** about which raw name a descriptor sources and what it renames to
— it only forbids sourcing a name that the datafactory itself does not serve.

Static (AST) analysis, so it runs everywhere without importing ``datafactory_query``.
"""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.beige

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIRS = [REPO_ROOT / "models", REPO_ROOT / "postprocessors"]

# Source names a datafactory descriptor must NOT use (it doesn't serve these):
BRIDGE_PREFIX = "lr_"          # consumer-bridge renamed name
VIEWSER_SUFFIX = "_sum_nokgi"  # viewser aggregation variant, not a datafactory column


def _string_keys(node: ast.Dict) -> list[str]:
    return [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]


def _string_elts(node: ast.List) -> list[str]:
    return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]


def _discover_datafactory_descriptors():
    """Yield (label, [source_feature_names]) for every datafactory config_queryset.py.

    A file is a datafactory descriptor if it declares ``FEATURE_RENAME`` /
    ``FACTORY_FEATURES`` or names the ``views-datafactory`` source. The source
    feature names are the ``FEATURE_RENAME`` keys + ``FACTORY_FEATURES`` elements.
    """
    out = []
    for base in CONFIG_DIRS:
        for path in base.glob("*/configs/config_queryset.py"):
            src = path.read_text()
            if "views-datafactory" not in src and "FEATURE_RENAME" not in src and "FACTORY_FEATURES" not in src:
                continue  # viewser model — not a datafactory descriptor
            tree = ast.parse(src)
            source_names: list[str] = []
            for stmt in tree.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                names = {t.id for t in stmt.targets if isinstance(t, ast.Name)}
                if "FEATURE_RENAME" in names and isinstance(stmt.value, ast.Dict):
                    source_names += _string_keys(stmt.value)
                if "FACTORY_FEATURES" in names and isinstance(stmt.value, ast.List):
                    source_names += _string_elts(stmt.value)
            if source_names:
                out.append((path.parent.parent.name, sorted(set(source_names))))
    return out


DESCRIPTORS = _discover_datafactory_descriptors()


def test_datafactory_descriptors_discovered():
    """Sanity: the known datafactory consumers are found (guards against a silent
    discovery break that would make the source check vacuous)."""
    found = {name for name, _ in DESCRIPTORS}
    expected = {"bright_starship", "bold_comet", "blazing_meteor", "un_fao"}
    missing = expected - found
    assert not missing, f"datafactory descriptors not discovered: {missing} (found {sorted(found)})"


@pytest.mark.parametrize("name,source_names", DESCRIPTORS, ids=[d[0] for d in DESCRIPTORS])
def test_datafactory_sources_are_raw_not_bridge(name, source_names):
    """Every datafactory descriptor must source raw registry names — never a
    consumer-bridge (`lr_*`) or viewser (`*_sum_nokgi`) name."""
    bad = [
        s for s in source_names
        if s.startswith(BRIDGE_PREFIX) or s.endswith(VIEWSER_SUFFIX)
    ]
    assert not bad, (
        f"{name}: datafactory descriptor sources non-raw feature name(s) {bad} — "
        f"the datafactory serves raw UCDP names; rename to a model-internal name in "
        f"FEATURE_RENAME *values* instead of sourcing a bridge/viewser name (#163)"
    )

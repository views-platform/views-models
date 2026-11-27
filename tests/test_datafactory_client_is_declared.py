"""A model whose queryset imports ``datafactory_query`` must install it (#489).

``datafactory_query`` ships in ``views-datafactory``. A ``config_queryset.py`` that imports
it and a ``requirements.txt`` that does not declare the package is a model that loads its
config and then dies at data fetch — or, in the catalogs job, is quietly rendered without
a feature description (#483). The eleven pgm r2darts2 models arrived from ``staging_202608``
in exactly that state. Nothing else in the hygiene suite judges the pairing: it checks
bounds and consistency of what IS declared, not what the code imports.

The rule is one-directional on purpose: declaring ``views-datafactory`` without importing
it is waste, not breakage, and is not judged here.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _imports_datafactory(queryset: Path) -> bool:
    tree = ast.parse(queryset.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "datafactory_query":
            return True
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "datafactory_query" for a in node.names):
            return True
    return False


IMPORTERS = sorted(
    q.parent.parent.name
    for q in REPO_ROOT.glob("models/*/configs/config_queryset.py")
    if _imports_datafactory(q)
)


def test_the_check_is_not_vacuous():
    assert len(IMPORTERS) >= 20, IMPORTERS


@pytest.mark.parametrize("name", IMPORTERS)
def test_a_datafactory_queryset_declares_the_client(name):
    lines = (REPO_ROOT / "models" / name / "requirements.txt").read_text(encoding="utf-8").splitlines()
    declared = [line for line in lines if line.strip().lower().startswith("views-datafactory")]
    assert declared, (
        f"{name}/configs/config_queryset.py imports datafactory_query but requirements.txt "
        f"does not declare views-datafactory — the model loads its config and dies at data fetch."
    )

"""Which data source a model's ``config_queryset.py`` reaches — read from the file, by AST.

One reader for the catalog (root README table) and the per-model README (#474), the same
shape as ``deliveries.coherence.maturity_of`` (register C-147: one reader, not three).

The answer is one of:

- ``"viewser"``      — the file imports ``viewser`` (pandas 1, Python ≤ 3.11; #473).
- ``"datafactory"``  — the file imports ``datafactory_query`` (ships in ``views-datafactory``).
- ``"synthetic"``    — ``generate()`` returns a dict literal with ``"source": "synthetic"``.
- ``"none"``         — there is no ``config_queryset.py`` (the baselines that need no data).
- ``"unknown"``      — none of the above, or more than one of them. Never a guess: a file that
                      imports both clients, or neither and is not synthetic, is reported as such
                      so a person looks, rather than counted on one side of the split.

Reads the source; imports nothing from it. A queryset file can import a client that is not
installed here (the catalogs job installs none — #483), and this must still answer.
"""

from __future__ import annotations

import ast
from pathlib import Path

VIEWSER = "viewser"
DATAFACTORY = "datafactory"
SYNTHETIC = "synthetic"
NONE = "none"
UNKNOWN = "unknown"

_CLIENT_MODULES = {"viewser": VIEWSER, "datafactory_query": DATAFACTORY}


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def _declares_synthetic(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant) and key.value == "source"
                and isinstance(value, ast.Constant) and value.value == SYNTHETIC
            ):
                return True
    return False


def data_source_of(queryset_path: Path) -> str:
    """The data source ``queryset_path`` reaches; see the module docstring for the values."""
    if not queryset_path.exists():
        return NONE
    tree = ast.parse(queryset_path.read_text(encoding="utf-8"), filename=str(queryset_path))
    found = {_CLIENT_MODULES[r] for r in _imported_roots(tree) if r in _CLIENT_MODULES}
    if _declares_synthetic(tree):
        found.add(SYNTHETIC)
    return found.pop() if len(found) == 1 else UNKNOWN

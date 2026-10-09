"""Detect a model's / ensemble's data source (EPIC #192 / S1 #193).

The reconciliation geography must use the country-id system of the data being
reconciled — VIEWS `country_id` (viewser) vs `gaul0_code` (views-datafactory),
which do not overlap. So the geography source must be **derived** from the
ensemble's actual data, not assumed. This module is that derivation (SRP:
detection only — it knows nothing about providers).

Discriminator: a model's `config_queryset.py` declares its source. A
datafactory model's `generate()` returns a descriptor dict with
`"source": "views-datafactory"`; a viewser model returns a `Queryset`. We detect
via AST (the string literal `"views-datafactory"` in the module) — import-free
(no constituent configs imported at run start) and comment-proof (comments are
not in the AST). Same discriminator as `tests/test_datafactory_source_names.py`.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

VIEWSER = "viewser"
DATAFACTORY = "views-datafactory"


def detect_model_source(model_dir: Path) -> str:
    """Return ``"viewser"`` or ``"views-datafactory"`` for a model, from its
    config_queryset descriptor. Raises ``FileNotFoundError`` if absent."""
    path = Path(model_dir) / "configs" / "config_queryset.py"
    if not path.exists():
        raise FileNotFoundError(f"config_queryset.py not found for model at {model_dir}")
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == DATAFACTORY:
            return DATAFACTORY
    return VIEWSER


def _load_modelset(ensemble_dir: Path) -> dict:
    path = Path(ensemble_dir) / "configs" / "config_modelset.py"
    spec = importlib.util.spec_from_file_location(
        f"_recon_modelset_{Path(ensemble_dir).name}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_modelset_config()


def detect_ensemble_source(ensemble_dir: Path, models_dir: Path | None = None) -> str:
    """Return the single data source shared by an ensemble's constituents.

    Fails loud (``ValueError``) if the constituents disagree — a mixed-source
    ensemble cannot be reconciled coherently (the grid→country attribution would
    mix country-id systems). See C-49.
    """
    ensemble_dir = Path(ensemble_dir)
    if models_dir is None:
        models_dir = ensemble_dir.parent.parent / "models"
    models = _load_modelset(ensemble_dir).get("models", [])
    if not models:
        raise ValueError(f"{ensemble_dir.name}: config_modelset declares no models")
    sources = {m: detect_model_source(Path(models_dir) / m) for m in models}
    distinct = set(sources.values())
    if len(distinct) != 1:
        raise ValueError(
            f"{ensemble_dir.name}: constituents disagree on data source {sources} — "
            f"a mixed-source ensemble cannot be reconciled coherently (country-id "
            f"systems differ; see C-49)."
        )
    return distinct.pop()

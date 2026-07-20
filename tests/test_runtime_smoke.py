"""Runtime smoke — actually EXECUTE a model (train+forecast) and assert on OUTPUT.

The rest of the suite verifies *declarations* (config, structure, contracts)
exhaustively but verifies *runtime behavior* nowhere in CI (register C-106).
"The config is valid" is proven thousands of ways; "the system actually works"
is proven zero ways — and every silent-failure incident (C-40 return shape,
C-104 config-vs-runtime sample count, reproducibility drift) lives in that gap.

This runs a real views-models model config through the real
config -> BaselineModelCatalog -> model path (the manager's construction seam,
`views_baseline/model/catalog.py`) on a tiny in-memory fixture, and asserts the
produced PredictionFrame honors the config. It catches the execution class no
parse-based test can:
  * does the model actually run end-to-end (not just parse)?
  * does the produced `sample_count` equal what the config declared (C-104)?
  * is the output well-formed (shape, dtype, no NaN/Inf, non-negative)?
  * is it deterministic (same seed -> identical draws)?

Offline by construction: numpy / pandas / views_frames / views_baseline only —
no network, no GPU, no wandb, no pipeline-core. It is **skip-truthful** (C-75):
when `views_baseline` is absent it SKIPs, so it never false-reds the main CI
suite (which does not install views_baseline). The dedicated
`.github/workflows/runtime_smoke.yml` job installs views_baseline and runs this
for real — the "runtime verified at PR" signal C-106 asks for.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    from views_baseline.model.catalog import BaselineModelCatalog

    _HAS_BASELINE = True
except ImportError:  # noqa: BLE001 — absence is a truthful skip, not a failure (C-75)
    _HAS_BASELINE = False

# Runtime/adversarial execution (ADR-005 red); skip truthfully where the model
# package is not installed — NOT `importorskip` on pipeline-core (C-91 anti-pattern).
pytestmark = [
    pytest.mark.red,
    pytest.mark.skipif(
        not _HAS_BASELINE,
        reason="views_baseline not installed — runtime smoke runs in runtime_smoke.yml",
    ),
]

REPO = Path(__file__).resolve().parent.parent

# Baseline algorithms that are (a) offline (numpy/pandas only) and (b) distributional
# — they draw `n_samples` per cell, so the output width is a real config-vs-runtime
# assertion. Point models (Zero/Locf/Average) produce width 1 and are less
# discriminating for the C-104 catch; the catalog handles them but we target the
# distributional ones here.
_OFFLINE_DISTRIBUTIONAL = {"ConflictologyModel", "MixtureBaseline"}


def _load_config(path: Path, getter: str) -> dict:
    """Load a config_*.py and call its getter, returning {} on any load problem
    (a broken config is a different test's concern; it must not break collection)."""
    try:
        spec = importlib.util.spec_from_file_location(
            f"_smoke_{path.parent.parent.name}_{getter}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, getter, None)
        return fn() if callable(fn) else {}
    except Exception:  # noqa: BLE001
        return {}


def _discover_cases():
    """Every views-models model that is an offline distributional baseline with a
    usable config (algorithm + targets + n_samples). Discovery, not a pinned name,
    so the smoke survives any single model being retired."""
    cases = []
    models = REPO / "models"
    if not models.is_dir():
        return cases
    for d in sorted(models.iterdir()):
        cfg = d / "configs"
        meta = _load_config(cfg / "config_meta.py", "get_meta_config")
        if meta.get("algorithm") not in _OFFLINE_DISTRIBUTIONAL:
            continue
        hp = _load_config(cfg / "config_hyperparameters.py", "get_hp_config")
        targets = hp.get("regression_targets") or meta.get("regression_targets")
        if hp.get("n_samples") and targets:
            cases.append((d.name, meta, hp, list(targets)))
    return cases


_CASES = _discover_cases()
_IDS = [c[0] for c in _CASES]


# The model validates the input index names against its declared level (ADR-003):
# pgm -> (month_id, priogrid_id), cm -> (month_id, country_id). The fixture must
# match the model's own `level`, or the model correctly fails loud.
_LEVEL_UNIT = {"pgm": "priogrid_id", "cm": "country_id"}


def _tiny_df(targets, level: str) -> pd.DataFrame:
    """Minimal offline fixture: 2 units over 21 months, integer index named for the
    model's declared level, one column per target. Enough history for the
    climatology window to resample."""
    unit_name = _LEVEL_UNIT.get(level, "priogrid_id")
    idx = pd.MultiIndex.from_product(
        [range(480, 501), [1, 2]], names=["month_id", unit_name]
    )
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {t: rng.integers(0, 5, len(idx)).astype(float) for t in targets}, index=idx
    )


def _run(meta: dict, hp: dict, targets: list):
    """Build the model VIA THE CATALOG (the manager's real config->model seam,
    which reads config['n_samples'] etc.) and run train+forecast on the fixture."""
    level = meta.get("level", "pgm")
    df = _tiny_df(targets, level)
    config = {**hp, "targets": targets}
    catalog = BaselineModelCatalog(
        config=config, partition_dict={"test": (495, 500)}, loa=level
    )
    model = catalog.get_model(meta["algorithm"])
    model.fit(df)
    return model.predict(df=df, sequence_number=0, output_length=3)


@pytest.mark.skipif(not _CASES, reason="no offline distributional-baseline models found")
@pytest.mark.parametrize("name,meta,hp,targets", _CASES, ids=_IDS)
def test_model_executes_and_output_honors_config(name, meta, hp, targets):
    """The core C-106 assertion: a real config, run through the real construction
    path, actually produces a well-formed PredictionFrame whose sample_count is
    what the config declared."""
    out = _run(meta, hp, targets)
    pf = out[targets[0]]
    y = np.asarray(pf.values)

    assert y.ndim == 2 and y.shape[0] > 0, f"{name}: empty/degenerate output {y.shape}"
    # config declares -> runtime honors (the C-104 config-vs-runtime catch)
    assert pf.sample_count == hp["n_samples"], (
        f"{name}: produced sample_count={pf.sample_count} but config declares "
        f"n_samples={hp['n_samples']}"
    )
    assert y.dtype in (np.float32, np.float64), f"{name}: unexpected dtype {y.dtype}"
    assert not np.isnan(y).any(), f"{name}: NaN in forecast"
    assert not np.isinf(y).any(), f"{name}: Inf in forecast"
    assert y.min() >= 0, f"{name}: negative forecast (counts must be >= 0)"
    ids = pf.identifiers
    assert "time" in ids and "unit" in ids, f"{name}: identifiers missing time/unit"
    assert len(ids["time"]) == y.shape[0], f"{name}: identifier length != n_rows"


@pytest.mark.skipif(not _CASES, reason="no offline distributional-baseline models found")
def test_forecast_is_deterministic():
    """Same seed -> identical draws. Nothing else in the suite runs a model twice
    and diffs the arrays; a reproducibility regression would otherwise be silent."""
    name, meta, hp, targets = _CASES[0]
    first = _run(meta, hp, targets)[targets[0]].values
    second = _run(meta, hp, targets)[targets[0]].values
    np.testing.assert_array_equal(
        np.asarray(first), np.asarray(second),
        err_msg=f"{name}: forecast not deterministic under fixed seed",
    )

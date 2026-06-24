import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
ENSEMBLES_DIR = REPO_ROOT / "ensembles"
EXTRACTORS_DIR = REPO_ROOT / "extractors"
POSTPROCESSORS_DIR = REPO_ROOT / "postprocessors"

# Scaffold template models that exist only as scaffolding examples and should
# not be exercised by the production test suite.
_FIXTURE_MODELS = {"fake_model"}


def _collect_model_dirs(base_dir: Path) -> list[Path]:
    """Return sorted list of model directories (dirs containing main.py)."""
    if not base_dir.exists():
        return []
    return sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and (d / "main.py").exists()
        and d.name not in _FIXTURE_MODELS
    )


def _collect_config_dirs(base_dir: Path) -> list[Path]:
    """Return sorted list of directories containing configs/config_partitions.py."""
    if not base_dir.exists():
        return []
    return sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and (d / "configs" / "config_partitions.py").exists()
    )


ALL_MODEL_DIRS = _collect_model_dirs(MODELS_DIR)
ALL_ENSEMBLE_DIRS = _collect_model_dirs(ENSEMBLES_DIR)
ALL_EXTRACTOR_DIRS = _collect_config_dirs(EXTRACTORS_DIR)
ALL_POSTPROCESSOR_DIRS = _collect_config_dirs(POSTPROCESSORS_DIR)

MODEL_NAMES = [d.name for d in ALL_MODEL_DIRS]
ENSEMBLE_NAMES = [d.name for d in ALL_ENSEMBLE_DIRS]
EXTRACTOR_NAMES = [d.name for d in ALL_EXTRACTOR_DIRS]
POSTPROCESSOR_NAMES = [d.name for d in ALL_POSTPROCESSOR_DIRS]

ALL_PARTITION_DIRS = (
    ALL_MODEL_DIRS + ALL_ENSEMBLE_DIRS
    + ALL_EXTRACTOR_DIRS + ALL_POSTPROCESSOR_DIRS
)
ALL_PARTITION_NAMES = (
    MODEL_NAMES + ENSEMBLE_NAMES
    + EXTRACTOR_NAMES + POSTPROCESSOR_NAMES
)


def load_canonical_partitions() -> dict:
    """Load canonical partition boundaries from meta/partitions.json."""
    with open(REPO_ROOT / "meta" / "partitions.json") as f:
        return json.load(f)


def load_config_module(config_path: Path, module_name: str = None):
    """Load a Python config file as a module using importlib (not exec)."""
    if module_name is None:
        module_name = config_path.stem
    # Use a unique module name to avoid collisions across models
    unique_name = f"_cfg_{config_path.parent.parent.name}_{module_name}"
    spec = importlib.util.spec_from_file_location(unique_name, config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Regression targets: single source of truth (EPIC #154 / S1 #155) ───────
# A model may declare ``regression_targets`` in config_meta.py and/or
# config_hyperparameters.py. The pipeline merges both with config_meta taking
# precedence (ConfigurationManager.get_combined_config). These helpers are the
# ONE way views-models code should obtain a model's targets — derive, never
# hardcode a target-name literal.

def _read_regression_targets(config_path: Path, getter_name: str):
    """Return the regression_targets list declared in one config file, or None
    if the file / accessor / key is absent. A bare string is normalized to a list."""
    if not config_path.exists():
        return None
    module = load_config_module(config_path)
    getter = getattr(module, getter_name, None)
    if getter is None:
        return None
    config = getter() or {}
    targets = config.get("regression_targets")
    if targets is None:
        return None
    if isinstance(targets, str):
        targets = [targets]
    return list(targets)


def regression_targets_by_location(model_dir: Path) -> dict:
    """Map each config location that declares regression_targets to its list.

    Keys are a subset of {"meta", "hp"}; a location absent from the dict did not
    declare the key. Used to enforce cross-location agreement.
    """
    config_dir = model_dir / "configs"
    out = {}
    meta = _read_regression_targets(config_dir / "config_meta.py", "get_meta_config")
    if meta is not None:
        out["meta"] = meta
    hp = _read_regression_targets(config_dir / "config_hyperparameters.py", "get_hp_config")
    if hp is not None:
        out["hp"] = hp
    return out


def get_regression_targets(model_dir: Path) -> list[str]:
    """The single source of truth for a model's regression targets.

    Mirrors the pipeline merge precedence (config_meta wins over
    config_hyperparameters; hp is the fallback). Returns ``[]`` if undeclared.
    """
    located = regression_targets_by_location(model_dir)
    return located.get("meta") or located.get("hp") or []


@pytest.fixture(params=ALL_MODEL_DIRS, ids=MODEL_NAMES)
def model_dir(request):
    """Parametrized fixture yielding each model directory."""
    return request.param


@pytest.fixture(params=ALL_ENSEMBLE_DIRS, ids=ENSEMBLE_NAMES)
def ensemble_dir(request):
    """Parametrized fixture yielding each ensemble directory."""
    return request.param


@pytest.fixture(params=ALL_MODEL_DIRS + ALL_ENSEMBLE_DIRS,
                ids=MODEL_NAMES + ENSEMBLE_NAMES)
def any_model_dir(request):
    """Parametrized fixture yielding each model and ensemble directory."""
    return request.param

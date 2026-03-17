import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
ENSEMBLES_DIR = REPO_ROOT / "ensembles"


def _collect_model_dirs(base_dir: Path) -> list[Path]:
    """Return sorted list of model directories (dirs containing main.py)."""
    if not base_dir.exists():
        return []
    return sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and (d / "main.py").exists()
    )


ALL_MODEL_DIRS = _collect_model_dirs(MODELS_DIR)
ALL_ENSEMBLE_DIRS = _collect_model_dirs(ENSEMBLES_DIR)

MODEL_NAMES = [d.name for d in ALL_MODEL_DIRS]
ENSEMBLE_NAMES = [d.name for d in ALL_ENSEMBLE_DIRS]


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

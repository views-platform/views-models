"""S1 (#193) — source detection (AST, import-free)."""
from pathlib import Path

import pytest

from reconciliation.source_detection import (
    DATAFACTORY,
    VIEWSER,
    detect_ensemble_source,
    detect_model_source,
)

pytestmark = pytest.mark.green

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
ENSEMBLES = REPO / "ensembles"


def test_detects_viewser_model():
    assert detect_model_source(MODELS / "car_radio") == VIEWSER


def test_detects_datafactory_model():
    assert detect_model_source(MODELS / "bright_starship") == DATAFACTORY


def test_missing_config_fails_loud():
    with pytest.raises(FileNotFoundError):
        detect_model_source(MODELS / "no_such_model_xyz")


@pytest.mark.parametrize(
    "ensemble", ["skinny_love", "pink_ponyclub", "white_mustang", "cruel_summer"]
)
def test_reconciliation_ensembles_are_viewser_today(ensemble):
    assert detect_ensemble_source(ENSEMBLES / ensemble) == VIEWSER


def _write_model(models_dir: Path, name: str, datafactory: bool):
    cfg = models_dir / name / "configs"
    cfg.mkdir(parents=True)
    if datafactory:
        body = 'def generate():\n    return {"name": "x", "source": "views-datafactory"}\n'
    else:
        body = "def generate():\n    return object()  # stands in for a viewser Queryset\n"
    (cfg / "config_queryset.py").write_text(body)


def test_mixed_source_ensemble_fails_loud(tmp_path):
    models_dir = tmp_path / "models"
    _write_model(models_dir, "vw_model", datafactory=False)
    _write_model(models_dir, "df_model", datafactory=True)
    ens = tmp_path / "ensembles" / "mixed"
    (ens / "configs").mkdir(parents=True)
    (ens / "configs" / "config_modelset.py").write_text(
        "def get_modelset_config():\n    return {'models': ['vw_model', 'df_model']}\n"
    )
    with pytest.raises(ValueError, match="disagree on data source"):
        detect_ensemble_source(ens, models_dir=models_dir)

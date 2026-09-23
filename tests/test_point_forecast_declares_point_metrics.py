"""A source that forecasts a point must declare the metrics a point is scored with (#486).

views-evaluation 2.0 refuses to evaluate a point forecast with no
``regression_point_metrics`` — "No metrics configured for (regression, point)" — and it
does so *after* training. ``bad_romance`` (r2darts2, ``num_samples: 1``, ``mc_dropout:
False``) had the key commented out and burned an 816 s training run on fimbulthul before
being refused. The census that found it: one of 31 r2darts2 sources.

Scope: sources whose hyperparameters declare ``num_samples`` — the r2darts2 vocabulary for
"how many samples per forecast". ``num_samples <= 1`` is a point forecast. Sources that
declare no ``num_samples`` are not judged here; their point-ness is decided elsewhere.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem + path.parent.parent.name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _point_sources():
    for configs in sorted(REPO_ROOT.glob("models/*/configs")):
        hp_path = configs / "config_hyperparameters.py"
        if not hp_path.exists():
            continue
        hp = _load(hp_path).get_hp_config()
        if "num_samples" in hp and hp["num_samples"] <= 1:
            yield configs.parent.name, configs


POINT_SOURCES = list(_point_sources())


def test_the_check_is_not_vacuous():
    assert len(POINT_SOURCES) >= 20, POINT_SOURCES


@pytest.mark.parametrize("name,configs", POINT_SOURCES, ids=[n for n, _ in POINT_SOURCES])
def test_a_point_forecast_declares_point_metrics(name, configs):
    meta = _load(configs / "config_meta.py").get_meta_config()
    assert meta.get("regression_point_metrics"), (
        f"{name} forecasts a point (num_samples <= 1) but config_meta.py declares no "
        f"regression_point_metrics — views-evaluation 2.0 refuses this AFTER training (#486)."
    )

"""A views-r2darts2 model at pgm must declare `entity_id: "priogrid_id"` (#499).

views-r2darts2 reads the index name for its prediction frames from the combined config and
**defaults it to `"country_id"`** regardless of the model's level
(`engines/darts_forecasting_model_manager.py:140`, 0.2.3). That is right for the 31 cm models
and wrong for every pgm one: pipeline-core's `CorePredictionSniffer` expects
`{priogrid_id, month_id}` at pgm and refuses `{month_id, country_id}`.

The refusal cost a day on 2026-09-22 because it was invisible — pipeline-core evaluated in
threads and discarded their exceptions, so the run wrote no predictions and reported PASS
(views-pipeline-core#529). The prediction VALUES were always priogrid cells; only the label
was wrong.

This guard is views-models' half. The upstream half — derive the default from `level` —
is views-r2darts2#55; when it ships, this test still passes and can be retired deliberately.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {"pgm": "priogrid_id", "cm": "country_id"}


def _meta(directory: Path) -> dict:
    path = directory / "configs" / "config_meta.py"
    spec = importlib.util.spec_from_file_location("meta_" + directory.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_meta_config()


def _darts_models():
    for directory in sorted((REPO_ROOT / "models").glob("*")):
        req = directory / "requirements.txt"
        if not req.is_file() or "views-r2darts2" not in req.read_text(encoding="utf-8"):
            continue
        yield directory.name, _meta(directory)


DARTS = list(_darts_models())


def test_the_check_is_not_vacuous():
    assert len(DARTS) >= 40, [n for n, _ in DARTS]


@pytest.mark.parametrize("name,meta", DARTS, ids=[n for n, _ in DARTS])
def test_entity_id_agrees_with_level(name, meta):
    level = meta["level"]
    expected = EXPECTED[level]
    declared = meta.get("entity_id", "country_id")  # the engine's own default
    assert declared == expected, (
        f"{name} is level={level!r} but its predictions would be indexed by {declared!r}; "
        f"pipeline-core's CorePredictionSniffer expects {expected!r}. Declare "
        f'"entity_id": "{expected}" in config_meta.py (views-r2darts2#55).'
    )

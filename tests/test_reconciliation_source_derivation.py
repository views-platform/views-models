"""S2 (#194) — the geography source is derived, with fail-loud guards."""
import pytest

from reconciliation.composition import _derive_source, build_reconciler_for_run

pytestmark = pytest.mark.green


def _model(root, name, datafactory):
    cfg = root / "models" / name / "configs"
    cfg.mkdir(parents=True)
    body = (
        'def generate():\n    return {"source": "views-datafactory"}\n'
        if datafactory
        else "def generate():\n    return object()  # stands in for a Queryset\n"
    )
    (cfg / "config_queryset.py").write_text(body)


def _ensemble(root, name, reconcile_with, constituents):
    ens = root / "ensembles" / name
    (ens / "configs").mkdir(parents=True)
    rw = f"'{reconcile_with}'" if reconcile_with else "None"
    (ens / "configs" / "config_meta.py").write_text(
        f"def get_meta_config():\n    return {{'name': '{name}', 'reconcile_with': {rw}}}\n"
    )
    (ens / "configs" / "config_modelset.py").write_text(
        f"def get_modelset_config():\n    return {{'models': {constituents!r}}}\n"
    )
    (ens / "configs" / "config_partitions.py").write_text(
        "def generate():\n    return {'forecasting': {'train': (121, 557), 'test': (558, 594)}}\n"
    )
    return ens


def test_derives_viewser_for_an_all_viewser_pair(tmp_path):
    _model(tmp_path, "vw1", datafactory=False)
    _model(tmp_path, "vw2", datafactory=False)
    _ensemble(tmp_path, "cm", None, ["vw2"])
    pgm = _ensemble(tmp_path, "pgm", "cm", ["vw1"])
    assert _derive_source(pgm) == "viewser"


def test_pgm_cm_source_mismatch_fails_loud(tmp_path):
    _model(tmp_path, "vw1", datafactory=False)
    _model(tmp_path, "df1", datafactory=True)
    _ensemble(tmp_path, "cm", None, ["df1"])  # CM is datafactory
    pgm = _ensemble(tmp_path, "pgm", "cm", ["vw1"])  # PGM is viewser
    with pytest.raises(ValueError, match="disagree on data source"):
        _derive_source(pgm)


def test_missing_reconcile_with_fails_loud(tmp_path):
    _model(tmp_path, "vw1", datafactory=False)
    pgm = _ensemble(tmp_path, "pgm", None, ["vw1"])
    with pytest.raises(ValueError, match="no reconcile_with"):
        _derive_source(pgm)


def test_datafactory_pair_fails_loud_no_provider(tmp_path):
    # both datafactory -> derives "views-datafactory" -> factory has no provider ->
    # fail loud (never a silent viewser fallback). Fails at the provider lookup,
    # before any viewser/postprocessing is touched.
    _model(tmp_path, "df1", datafactory=True)
    _model(tmp_path, "df2", datafactory=True)
    _ensemble(tmp_path, "cm", None, ["df2"])
    pgm = _ensemble(tmp_path, "pgm", "cm", ["df1"])
    with pytest.raises(ValueError, match="[Uu]nknown reconciliation geography source"):
        build_reconciler_for_run(pgm)

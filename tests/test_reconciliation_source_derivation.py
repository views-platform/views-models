"""S2 (#194) — the geography source is derived, with fail-loud guards."""
import pytest

# reconciliation/__init__.py imports composition, which needs pipeline-core's Reconciler
# port (3.0.0+, unreleased). Guard BEFORE any reconciliation import: placed after one,
# collection ERRORS instead of skipping, which is what turned CI red (ADR-005 §skip).
pytest.importorskip("views_pipeline_core.domain.reconciliation_port")

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


def test_datafactory_pair_resolves_to_the_datafactory_provider(tmp_path):
    """Both datafactory -> derives "views-datafactory" -> selects the GAUL provider.

    ⛔ This test used to assert the opposite — that the pair FAILED LOUD because no
    datafactory provider existed (`test_datafactory_pair_fails_loud_no_provider`). #601
    built the provider, so the refusal it pinned is gone. The property that survives, and
    the one that mattered all along, is **never a silent viewser fallback**: the pair must
    resolve to the GAUL provider, not to the viewser one, because the two country-id
    systems do not overlap (C-49). That is what is asserted now.

    The provider class is checked rather than built: constructing it is harmless, but
    `build()` would fetch from the datafactory, which needs credentials and a network.
    """
    from reconciliation.datafactory_country_mapping_provider import (
        DatafactoryCountryMappingProvider,
    )
    from reconciliation.reconciler_factory import _PROVIDERS
    from reconciliation.viewser_country_mapping_provider import (
        ViewserCountryMappingProvider,
    )

    _model(tmp_path, "df1", datafactory=True)
    _model(tmp_path, "df2", datafactory=True)
    _ensemble(tmp_path, "cm", None, ["df2"])
    pgm = _ensemble(tmp_path, "pgm", "cm", ["df1"])

    source = _derive_source(pgm)
    assert source == "views-datafactory"
    assert _PROVIDERS[source] is DatafactoryCountryMappingProvider
    assert _PROVIDERS[source] is not ViewserCountryMappingProvider


def test_a_source_with_no_provider_still_fails_loud(tmp_path):
    """The fail-loud path the test above used to cover, kept on a source that genuinely
    has no provider. Without this, #601 would have deleted the guard rather than moved it."""
    _model(tmp_path, "vw1", datafactory=False)
    _ensemble(tmp_path, "cm", None, ["vw1"])
    pgm = _ensemble(tmp_path, "pgm", "cm", ["vw1"])
    with pytest.raises(ValueError, match="[Uu]nknown reconciliation geography source"):
        build_reconciler_for_run(pgm, source="gaul")

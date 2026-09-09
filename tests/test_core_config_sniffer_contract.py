"""Every config in this repo is accepted by the pipeline-core that will run it (#371).

**The gap this closes.** No workflow here loaded a config through views-pipeline-core's
``CoreConfigSniffer``, so a config pipeline-core refuses at load was caught only by a
human reading a diff. That is how views-models#367 came to declare
``classification_targets`` on ``rusty_bucket`` with no classification metric key —
``_check_targets_and_metrics`` raises on exactly that pair, and ``sniff_all()`` runs
before any side effect in both ensemble managers, so the ensemble would have died at
config load.

**This runs against the INSTALLED pipeline-core, deliberately.** ``run_tests.yml`` pins
an exact version; the question this file answers is "will the thing that actually runs
accept these configs", which a vendored copy of the rules could not answer. It lives in
the existing test job rather than a new workflow for the same reason — the pin is
already there.

**Two exclusions, both deliberate rather than accidental.**

*Deprecated sources* are skipped. The sniffer refuses to run a model whose
``deployment_status`` is ``deprecated`` — that is the contract working, not a config
defect, and asserting it away would invert the check. Membership is derived from the
config at run time, never from a hardcoded list, so retiring a model does not require
editing this file.

*``KNOWN_REJECTED``* pins the configs that are refused today for reasons that predate
this guard. It is a **set**, so both adding and removing a member is a reviewed edit:
a new rejection fails the suite, and fixing one also fails until the pin is updated.
The alternative — a "≤ N failures" threshold — would let one defect be swapped for
another silently.
"""

from pathlib import Path

import pytest

from tests.conftest import ALL_ENSEMBLE_DIRS, ALL_MODEL_DIRS, load_config_module

pytestmark = [pytest.mark.beige]

# The sniffer and the configuration manager both live in pipeline-core. Without it there
# is nothing to check against; `runtime_smoke.yml` deliberately does not install it.
CoreConfigSniffer = pytest.importorskip(
    "views_pipeline_core.modules.validation.core_config_sniffer",
    reason="views_pipeline_core is not installed — nothing to validate configs against",
).CoreConfigSniffer
ConfigurationManager = pytest.importorskip(
    "views_pipeline_core.managers.configuration.configuration",
    reason="views_pipeline_core is not installed",
).ConfigurationManager

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Configs the installed pipeline-core refuses today, for reasons that predate this
#: guard. Every one of these raises
#: ``evaluation_mode='point' requires aggregate_method to be set``. Six are real
#: baselines; three (``*_dream``) are fixtures listed in ``meta/fixtures.json``.
#: Verified present in the 3.0.0 tag, so this is not an artefact of a newer release.
#:
#: **This set must only ever shrink.** Adding to it means shipping a config that
#: pipeline-core will refuse at load — say why in the commit if you do.
KNOWN_REJECTED = frozenset({
    "average_cmbaseline",
    "average_pgmbaseline",
    "locf_cmbaseline",
    "locf_pgmbaseline",
    "zero_cmbaseline",
    "zero_pgmbaseline",
    "diagonal_dream",
    "horizontal_dream",
    "vertical_dream",
})

#: Run types worth checking. `_check_evaluation_contract` only runs for non-forecasting,
#: so one of each side of that branch is the minimum honest coverage.
RUN_TYPES = ("forecasting", "calibration")


def _config(directory, filename, getter):
    path = directory / "configs" / filename
    if not path.exists():
        return None
    fn = getattr(load_config_module(path), getter, None)
    return fn() if fn is not None else None


def _combined(directory):
    """Build the config the managers build, using pipeline-core's own merge.

    Reimplementing the precedence (partition_dict < hyperparameters < deployment < meta)
    here would be a second copy of a rule owned upstream, free to drift from it. The
    managers reach this through ``ConfigurationManager.get_combined_config``; so does this.
    """
    partitions = _config(directory, "config_partitions.py", "generate") or {}
    manager = ConfigurationManager(
        config_hyperparameters=_config(directory, "config_hyperparameters.py", "get_hp_config") or {},
        config_deployment=_config(directory, "config_deployment.py", "get_deployment_config") or {},
        config_meta=_config(directory, "config_meta.py", "get_meta_config") or {},
        partition_dict=partitions,
    )
    return manager.get_combined_config(), partitions


def _is_deprecated(directory):
    deployment = _config(directory, "config_deployment.py", "get_deployment_config") or {}
    return deployment.get("deployment_status") == "deprecated"


def _subjects():
    """(name, directory, target) for every source the sniffer should accept."""
    for target, directories in (("model", ALL_MODEL_DIRS), ("ensemble", ALL_ENSEMBLE_DIRS)):
        for directory in directories:
            if _is_deprecated(directory):
                continue
            yield directory.name, directory, target


def _rejections(run_type):
    """{name: error} for every non-deprecated source the sniffer refuses."""
    refused = {}
    for name, directory, target in _subjects():
        combined, partitions = _combined(directory)
        try:
            CoreConfigSniffer(combined, partitions, target=target).sniff_all(run_type)
        except Exception as exc:  # noqa: BLE001 — any refusal is the fact
            refused[name] = f"{type(exc).__name__}: {exc}"
    return refused


@pytest.mark.parametrize("run_type", RUN_TYPES)
def test_no_config_is_refused_that_is_not_already_known(run_type):
    """The set of refused configs is exactly ``KNOWN_REJECTED`` — no more, no fewer."""
    refused = _rejections(run_type)
    new = set(refused) - KNOWN_REJECTED
    fixed = KNOWN_REJECTED - set(refused)

    assert not new, (
        f"[{run_type}] pipeline-core refuses {len(new)} config(s) that were fine before. "
        f"These would die at config load, before any side effect:\n"
        + "\n".join(f"  {n}: {refused[n]}" for n in sorted(new))
    )
    assert not fixed, (
        f"[{run_type}] {sorted(fixed)} no longer refused — good. Remove them from "
        f"KNOWN_REJECTED in this file so the pin keeps meaning something."
    )


def test_the_check_is_not_vacuous():
    """Guard against the whole thing passing over an empty or tiny subject list.

    If discovery broke, every assertion above would pass while checking nothing. The
    floor is deliberately well below today's count so ordinary additions and removals
    do not trip it.
    """
    subjects = list(_subjects())
    assert len(subjects) > 100, (
        f"only {len(subjects)} configs discovered — discovery is probably broken, and "
        f"the rejection assertions above would be passing over almost nothing"
    )


def test_the_guard_has_teeth_on_the_367_shape():
    """The defect this file exists for is actually caught.

    ``classification_targets`` with no classification metric key — what views-models#367
    writes for ``rusty_bucket``. Synthetic rather than read from the tree, so the test
    keeps its meaning after that config is fixed.
    """
    bad = {
        "name": "rusty_bucket",
        "level": "pgm",
        "aggregation": "concat",
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample"],
    }
    with pytest.raises(ValueError, match="classification_targets is non-empty"):
        CoreConfigSniffer(bad, {}, target="ensemble")._check_targets_and_metrics()

    # ...and the same config with a valid classification metric is accepted.
    CoreConfigSniffer(
        {**bad, "classification_sample_metrics": ["Brier_cls_sample"]}, {}, target="ensemble"
    )._check_targets_and_metrics()

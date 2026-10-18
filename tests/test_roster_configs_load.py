"""Every tracked config must actually LOAD — not merely carry the right values.

`test_roster_conformance.py` compares 184 individual config values against a reference dict and
passes. It never constructs a `HydraNetConfig`. So a config can satisfy every pinned value and still
be **unloadable**, and the suite stays green.

That is not hypothetical. On 2026-09-07 an emit run over the roster found `bold_comet` and
`heavy_freighter` could not be run at all: both had scheduled sampling active (`ss_schedule='linear'`,
`ss_epsilon_max=0.5`) with `ss_feedback` unset, so it defaulted to `'mean'` and contradicted their own
`rollout_feedback='sample'` — C-259, reported as views-hydranet#295. **Two of eight production
ensemble members had been unrunnable since August and every test passed.**

The validation itself was never missing: `HydraNetConfig` raised correctly, and the platform's
`CoreConfigSniffer` is the wrong layer (it has no knowledge of `ss_feedback`). What was missing was
anything that ran that validation *cheaply, in CI, over every tracked config* instead of only
when someone tried to launch a model.

"Every tracked config" is wider than the roster, and was not always: see `TRACKED_MODELS`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# One declaration of the roster, not three. `rusty_bucket`'s modelset was reordered precisely so
# that the two existing declarations could not drift apart; this file must not become a third.
from tests.test_roster_conformance import (
    EXPERIMENT_ARMS,
    ROSTER_MODELS,
    SEED_REPLICATES,
)

#: Every tracked HydraNet config in the repository, in all three categories that exist.
#:
#: The roster is the eight. An experiment arm is a tracked model directory that exists to be
#: trained against one of the eight as a control; it is deliberately NOT a roster member
#: (`test_roster_conformance.EXPERIMENT_ARMS` says why) and so it was invisible to this file
#: until it was named here. It must be, because the defect in the docstring above is one an
#: arm inherits by construction: an arm is a COPY of its control's config with two or three
#: values changed, which copies every latent config fault along with it, and the arm's whole
#: point is that it is launched on rented hardware where a load failure costs provisioning
#: time rather than a second (views-models#552).
#:
#: A seed replicate inherits that reasoning exactly — it is also a copy of a control's config
#: launched on rented hardware — and it is additionally the case that a replicate failing to
#: load wastes a pod whose only purpose is to measure variance, so the loss is the estimate
#: and not just the provisioning time (views-models#565).
TRACKED_MODELS = (
    list(ROSTER_MODELS) + list(EXPERIMENT_ARMS) + list(SEED_REPLICATES)
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"



def _assemble(model: str) -> dict:
    """Merge the config parts the pipeline merges, plus the `run_type` it supplies at launch."""
    parts: dict = {"run_type": "calibration"}
    configs = MODELS_DIR / model / "configs"
    # ADR-017 Phase 2: a source carries config_maturity.py OR config_deployment.py. The
    # new file wins, as in pipeline-core's load_maturity_config (3.2.0). Read whichever exists.
    maturity_stem = "config_maturity" if (configs / "config_maturity.py").exists() else "config_deployment"
    for stem in ("config_hyperparameters", "config_meta", maturity_stem):
        path = configs / f"{stem}.py"
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        getters = [n for n in dir(mod) if n.startswith("get_")]
        assert getters, f"{path} exposes no get_* function"
        parts.update(getattr(mod, getters[0])())
    return parts


@pytest.mark.parametrize("model", TRACKED_MODELS)
def test_the_tracked_config_can_be_constructed(model):
    """The whole test. If it raises, that model cannot be run — whatever else is green."""
    pytest.importorskip("views_hydranet")
    from views_hydranet.utils.config_initializer import ConfigInitializer

    try:
        ConfigInitializer(_assemble(model)).get_config()
    except Exception as exc:  # noqa: BLE001 - the failure text is the useful part
        pytest.fail(
            f"{model}'s config does not load, so the model cannot be run:\n{exc}"
        )


@pytest.mark.parametrize("model", TRACKED_MODELS)
def test_scheduled_sampling_declares_its_feedback(model):
    """The specific defect, pinned so it cannot return by a different route.

    A model with scheduled sampling ACTIVE must say what it feeds back. Inactive (`eps_max == 0`)
    is fine unset — that is `violet_visitor`, and it is a legitimate difference, not an oversight.

    Parametrized rather than looping internally so that its scope is declared in the same
    place, and in the same form, as the construction test's — which is what
    `test_both_categories_of_tracked_config_are_in_scope` is able to read.
    """
    cfg = _assemble(model)
    active = cfg.get("ss_schedule") and (cfg.get("ss_epsilon_max") or 0) > 0
    assert not (active and not cfg.get("ss_feedback")), (
        f"scheduled sampling is active in {model} but ss_feedback is unset. It defaults to "
        "'mean', which contradicts rollout_feedback='sample' and makes the config unloadable "
        "(C-259, views-hydranet#295)."
    )


def _parametrized_scope(func) -> set[str]:
    """The model names pytest will actually collect for `func`.

    This reads the `parametrize` decorator, not the module-level list the decorator happens
    to be given today. The distinction is the whole point of the test below: the previous
    version of that test compared `TRACKED_MODELS` against the two declarations, so it
    could not see

        -@pytest.mark.parametrize("model", TRACKED_MODELS)
        +@pytest.mark.parametrize("model", ROSTER_MODELS)

    which leaves the arm unconstructed with the full suite green. The only trace that edit
    leaves anywhere is one fewer SKIP, and nobody reads the skip count.
    """
    marks = [m for m in getattr(func, "pytestmark", []) if m.name == "parametrize"]
    assert len(marks) == 1, (
        f"{func.__name__} is no longer parametrized exactly once, so its real scope cannot "
        f"be read here and this guard cannot do its job"
    )
    _argnames, argvalues = marks[0].args
    return set(argvalues)


@pytest.mark.parametrize(
    "func",
    [test_the_tracked_config_can_be_constructed, test_scheduled_sampling_declares_its_feedback],
    ids=lambda f: f.__name__,
)
def test_both_categories_of_tracked_config_are_in_scope(func):
    """The guard on this file's own scope, which is where it failed before.

    Narrowing a parametrization back to the roster is a one-line edit that leaves every
    other test in the repository green — a smaller parametrization is not a failure. That is
    how `amber_kestrel` came to be pinned by four tests and constructed by none
    (views-models#552), and the first attempt at this guard did not close it: it asserted on
    the list, and the cheaper mutation edits the decorator.

    So this asserts on the collected scope, for every test in this file that has one, and
    compares it against the three source-of-truth declarations read independently.
    """
    declared = set(ROSTER_MODELS) | set(EXPERIMENT_ARMS) | set(SEED_REPLICATES)
    scope = _parametrized_scope(func)
    assert scope == declared, (
        f"{func.__name__} runs over {sorted(scope)}, but the tracked configs are "
        f"{sorted(declared)}. Missing: {sorted(declared - scope)}; unexpected: "
        f"{sorted(scope - declared)}. Every model in ROSTER, every experiment arm and every "
        f"seed replicate must be built through HydraNetConfig here; a value pin elsewhere is "
        f"not a substitute (C-259, views-hydranet#295)."
    )
    assert EXPERIMENT_ARMS, (
        "EXPERIMENT_ARMS is empty, so this file is covering the roster only. If the last arm "
        "was genuinely retired, delete this assertion deliberately rather than letting the "
        "next arm arrive unconstructed."
    )

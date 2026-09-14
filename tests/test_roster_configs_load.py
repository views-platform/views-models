"""Every roster config must actually LOAD — not merely carry the right values.

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
anything that ran that validation *cheaply, in CI, over the whole roster* instead of only when
someone tried to launch a model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# One declaration of the roster, not three. `rusty_bucket`'s modelset was reordered precisely so
# that the two existing declarations could not drift apart; this file must not become a third.
from tests.test_roster_conformance import ROSTER_MODELS as ROSTER

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"



def _assemble(model: str) -> dict:
    """Merge the config parts the pipeline merges, plus the `run_type` it supplies at launch."""
    parts: dict = {"run_type": "calibration"}
    for stem in ("config_hyperparameters", "config_meta", "config_deployment"):
        path = MODELS_DIR / model / "configs" / f"{stem}.py"
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        getters = [n for n in dir(mod) if n.startswith("get_")]
        assert getters, f"{path} exposes no get_* function"
        parts.update(getattr(mod, getters[0])())
    return parts


@pytest.mark.parametrize("model", ROSTER)
def test_the_roster_config_can_be_constructed(model):
    """The whole test. If it raises, that model cannot be run — whatever else is green."""
    pytest.importorskip("views_hydranet")
    from views_hydranet.utils.config_initializer import ConfigInitializer

    try:
        ConfigInitializer(_assemble(model)).get_config()
    except Exception as exc:  # noqa: BLE001 - the failure text is the useful part
        pytest.fail(
            f"{model}'s config does not load, so the model cannot be run or contribute to the "
            f"ensemble:\n{exc}"
        )


def test_scheduled_sampling_declares_its_feedback():
    """The specific defect, pinned so it cannot return by a different route.

    A model with scheduled sampling ACTIVE must say what it feeds back. Inactive (`eps_max == 0`)
    is fine unset — that is `violet_visitor`, and it is a legitimate difference, not an oversight.
    """
    offenders = []
    for model in ROSTER:
        cfg = _assemble(model)
        active = cfg.get("ss_schedule") and (cfg.get("ss_epsilon_max") or 0) > 0
        if active and not cfg.get("ss_feedback"):
            offenders.append(model)
    assert not offenders, (
        f"scheduled sampling is active but ss_feedback is unset in: {offenders}. It defaults to "
        "'mean', which contradicts rollout_feedback='sample' and makes the config unloadable "
        "(C-259, views-hydranet#295)."
    )

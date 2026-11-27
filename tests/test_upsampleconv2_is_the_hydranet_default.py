"""`UpsampleConv2` is the default upsampling for a new HydraNet, enforced by exception list.

⛔ WHY AN EXCEPTION LIST AND NOT A TEMPLATE DEFAULT: **there is no default to change.**
Measured 2026-11-04 across all three repositories:

  - views-hydranet  `utils/config_initializer.py:142` is `model: str = Field(...)` — REQUIRED,
    no default. (views-hydranet#546 states this itself.)
  - views-models    `tools/scaffold/build_model_scaffold.py` has no HydraNet template; it
    delegates to `views_pipeline_core.templates.model.template_config_hyperparameters`, which
    is algorithm-agnostic. `git grep HydraBNUNet06_LSTM4 -- tools/` finds only two EXP-02
    docstrings.
  - So a new HydraNet gets its architecture from whoever copies an existing config.

A default that exists nowhere cannot be flipped. The only way this repository can express one is
to make the *old* value the thing that needs justifying — which is this file. A new HydraNet that
declares `HydraBNUNet06_LSTM4` fails until its name is added below with a reason, so the path of
least resistance becomes `UpsampleConv2`.

This is the shape the repo already uses for `KNOWN_GATELESS`, `ARM_FREE_KEYS` and
`DECLARES_PIPELINE_CORE`: an escape hatch that must be widened visibly.

## The evidence for the default (views-hydranet#546, FINDING_EXP03.md §11)

Better on the only thing measured with an interval, worse on nothing:

  B-1 structure share, lag 1, h1   +0.0504, across-seed 90% CI [+0.0229, +0.0780], 5 seeds
  negative controls lags 16/32     do NOT replicate  => sharpening, not rescaling
  parameter count                  +-0 exactly, 905,289 both ways
  CRPS, every truth band           null — not worse
  gate AP                          does not replicate; mean +0.0021 — not worse

⚠️ **B-1 at lag 1 was NOT pre-registered** — `PREREG.md` registered P-23 on gate AP. The adoption
case rests on a statistic chosen after seeing the data: hypothesis-generating, not confirmatory.
Until views-hydranet#529 runs a second, pre-registered arm this must not be called a confirmed
improvement. ⛔ Pre-committed revert condition: any measurement showing a **resolved** CRPS or AP
cost sends the default back.
"""

from __future__ import annotations

import ast
import glob
from pathlib import Path

import pytest

pytestmark = pytest.mark.beige

REPO = Path(__file__).resolve().parents[1]

DEFAULT_ARCHITECTURE = "UpsampleConv2"

#: ⛔ Models permitted to keep `HydraBNUNet06_LSTM4`, each with the reason it is not flipped.
#: Adding a name here is a deliberate, reviewed act. Removing one means that model was flipped.
#: Every entry is a live reason, not a grandfather clause — see views-models#711 for the triggers.
LEGACY_ARCHITECTURE_ALLOWED: dict[str, str] = {
    # ⭐ EMPTY, and that is the point: on 2026-11-04 all 27 HydraNets were flipped to
    # UpsampleConv2 by Simon's decision, so the default is REAL rather than enforced by
    # exception. The three deferrals this list used to hold (the EXP-03 control arm, the
    # kestrel contrast chain, the rusty_bucket members) were all resolved the same way —
    # by flipping everything at once, which is what keeps every one-variable contrast
    # intact: no pair can differ on `model` if no model declares anything else.
    #
    # ⛔ An entry here is a model allowed to keep an architecture the fleet has retired.
    # Adding one needs a reason and a trigger, in an issue, not just a string.
}


def _hydranets() -> list[str]:
    return sorted(
        Path(p).parts[-3]
        for p in glob.glob(str(REPO / "models" / "*" / "configs" / "config_meta.py"))
        if '"algorithm": "HydraNet"' in Path(p).read_text(encoding="utf-8")
    )


def _architecture(model: str) -> str | None:
    """The `model` value from `get_hp_config()`'s returned dict, read via ast.

    ast and not import: a HydraNet's config imports nothing, but its sibling
    `config_queryset.py` needs `datafactory_query`, which CI lacks — and a guard that only runs
    locally is not a guard (54 CI failures, test_hydranet_scope_and_source.py).
    """
    path = REPO / "models" / model / "configs" / "config_hyperparameters.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and k.value == "model"
                        and isinstance(v, ast.Constant) and isinstance(v.value, str)):
                    return v.value
    return None


HYDRANETS = _hydranets()


def test_the_check_can_see_the_fleet():
    """⛔ ADR-025: an empty roster would make every assertion below vacuous."""
    assert len(HYDRANETS) >= 20, (
        f"only {len(HYDRANETS)} HydraNets discovered, expected 20+. The selector is a substring "
        "match on '\"algorithm\": \"HydraNet\"'; fix it rather than lowering this floor."
    )


@pytest.mark.parametrize("model", HYDRANETS)
def test_a_hydranet_uses_the_default_or_is_a_listed_exception(model):
    arch = _architecture(model)
    assert arch is not None, (
        f"{model} declares no `model` key in get_hp_config() — a HydraNet must name its "
        "architecture, because views-hydranet has no default (config_initializer.py:142)."
    )
    if arch == DEFAULT_ARCHITECTURE:
        return
    assert model in LEGACY_ARCHITECTURE_ALLOWED, (
        f"{model} declares {arch!r}, but the default for a new HydraNet is "
        f"{DEFAULT_ARCHITECTURE!r} (views-hydranet#546: better on the only intervalled "
        f"measurement, worse on nothing, +-0 parameters).\n"
        f"If this model genuinely must keep {arch!r}, add it to LEGACY_ARCHITECTURE_ALLOWED in "
        f"this file WITH THE REASON. Widening that list is meant to be a visible act."
    )


def test_the_exception_list_has_no_stale_entries():
    """A name here that no longer needs the exception is a lie about what is pending.

    Fires when a listed model was flipped — or deleted — and the list was not updated. That is
    the drift that turned C-157's "Four instances" into five, and #698's seven stale configs.
    """
    stale = sorted(
        name for name in LEGACY_ARCHITECTURE_ALLOWED
        if name not in HYDRANETS or _architecture(name) == DEFAULT_ARCHITECTURE
    )
    assert not stale, (
        "these names are in LEGACY_ARCHITECTURE_ALLOWED but no longer need to be "
        f"(flipped already, or gone): {stale}. Remove them — the list is the record of what is "
        "still pending, and a stale entry overstates the work left."
    )


def test_every_exception_states_a_reason():
    """An exception without a reason is an exemption, and those accumulate silently."""
    empty = sorted(k for k, v in LEGACY_ARCHITECTURE_ALLOWED.items() if not v.strip())
    assert not empty, f"exceptions with no stated reason: {empty}"

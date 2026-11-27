"""A model's W&B sweep name starts with that model's own name.

⛔ WHY. On 2026-11-04 **21 of 162** models had a `config_sweep.py` whose `name` identified a
DIFFERENT model. Twelve of them were the `ravaging_*` / `roaming_*` / `warring_*` sets, each
carrying the sweep name of the model it was scaffolded from:

    ravaging_fighter  ->  "new_rules_nbeats_shadow_20260508_D"
    wild_rose         ->  "lovely_creature"

The name is what W&B labels the sweep, so launching one files the run under another model's
identity. Nothing read it in a normal train/eval run, which is exactly why it drifted for months
across three whole families at once.

⚠️ The algorithm segment was RIGHT in all sixteen prefixed cases — `ravaging_fighter` really is
`NBEATSModel` and the name really said `nbeats`. Only the leading model name was wrong. That is
what a scaffold copy looks like: everything specific to the algorithm survives the copy because
it was already correct, and the one field identifying the source does not.

⚠️ A PREFIX is asserted, not equality. The fleet has no single suffix convention — measured on
2026-11-04 there were `amber_kestrel_sweep`, `dancing_monkey_tsmixer`, `fancy_feline_grid_test`,
`shining_codex_01` and bare `lovely_creature`, all legitimate. The only thing every correct name
has in common is that it begins with its owner. Asserting more than that would fail 100+ models
for a convention nobody agreed.

⛔ Read via `ast`, not by importing: a sweep config is plain data, and several model packages are
not installed in CI (`tests/test_hydranet_scope_and_source.py` learned this the expensive way —
54 CI failures from a guard that executed a config).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.beige

REPO = Path(__file__).resolve().parents[1]
FIXTURES: set[str] = set(json.loads((REPO / "meta" / "fixtures.json").read_text()))


def _sweep_name(model: str) -> str | None:
    """The top-level `name` of the dict assigned to `sweep_config`, or None if absent.

    Only the TOP-LEVEL name: `config_sweep.py` also carries `metric = {'name': 'MSE'}` and
    per-parameter dicts, and those are not model identities. Reading any `"name"` key at any
    depth is how a first attempt at this check produced 60+ false positives on metric names.
    """
    path = REPO / "models" / model / "configs" / "config_sweep.py"
    if not path.exists():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "sweep_config" not in targets or not isinstance(node.value, ast.Dict):
            continue
        for k, v in zip(node.value.keys, node.value.values):
            if (isinstance(k, ast.Constant) and k.value == "name"
                    and isinstance(v, ast.Constant) and isinstance(v.value, str)):
                return v.value
    return None


MODELS = sorted(
    d.name for d in (REPO / "models").iterdir()
    if d.is_dir() and (d / "main.py").exists() and d.name not in FIXTURES
)
WITH_SWEEP = [m for m in MODELS if _sweep_name(m) is not None]


def test_the_check_can_see_the_fleet():
    """⛔ ADR-025: a parse that silently found nothing would make this file vacuous."""
    assert len(WITH_SWEEP) >= 100, (
        f"only {len(WITH_SWEEP)} of {len(MODELS)} models yielded a top-level sweep name. "
        "The ast walk looks for `sweep_config = {...}` with a literal 'name'; if the fleet "
        "changed shape, fix the walk rather than lowering this floor."
    )


@pytest.mark.parametrize("model", WITH_SWEEP)
def test_a_sweep_name_starts_with_its_own_model(model):
    name = _sweep_name(model)
    assert name == model or name.startswith(model + "_"), (
        f"{model}'s sweep is named {name!r}, which does not start with {model!r}. A sweep "
        f"launched from here would be filed in W&B under another model's identity. If this is a "
        f"scaffold copy, swap the leading model name and keep the rest — the algorithm and "
        f"campaign segments are usually already correct for the owner."
    )

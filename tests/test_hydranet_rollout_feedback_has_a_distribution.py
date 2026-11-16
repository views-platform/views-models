"""A HydraNet that feeds back a DRAW must have a distribution to draw from.

⛔ Ten of thirty-seven set `'rollout_feedback': 'sample'` on a `'standard'` point head, which
has no distribution family. The config loads and the model TRAINS; `HydraNetInference` refuses
the pair when it is constructed, so the run dies at the first forecast — after the expensive
part. Found by the views-hydranet session, 2026-11-16 (views-models#737): `calm`, `keen`,
`mellow`, `nimble`, `quiet`, each `_counter` and `_sorter`. The ranking arm of views-models#729
was among them.

**Why it survived review.** The comment ten lines above the defect states the rule correctly for
a different key: "K IS PINNED AT 1 AND NOT A CHOICE: K > 1 requires a sampleable family
(nb/zinb/mixture_nb/truncated_nb) and `validate` refuses it on a `standard` point head." The
author knew. `n_head_samples` got the rule; `rollout_feedback`, two lines later, did not. Reading
the file is therefore not enough to catch this, which is why it is a test.

**Where the refusal lives.** `views_hydranet/utils/hydranet_inference.py` resolves the family
with `resolve_family(output_distribution)` and raises when `rollout_feedback == "sample"` and the
result is `None`. `views_hydranet/distributions/registry.py` is the single source of the valid
names: `DISTRIBUTION_REGISTRY` holds `nb`, `zinb`, `mixture_nb`, `truncated_nb`. `standard` is
absent from it, so `resolve_family("standard") is None`.

⚠️ `SAMPLEABLE` below MIRRORS that registry's keys. It is a literal here on purpose: this suite
runs without views-hydranet installed, and a check that cannot see its subject must refuse rather
than skip (ADR-025) — so a silent import guard would be worse than a stated mirror. If upstream
adds a family, this set gains a name and the test gets stricter about nothing; the failure mode
of the mirror going stale is a FALSE ALARM naming the new family, not a missed defect.

A model may also omit `rollout_feedback` entirely, which upstream resolves to `'mean'` for a
family-less head. That is legal and this test allows it.
"""
from __future__ import annotations

import glob
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.beige

REPO = Path(__file__).resolve().parents[1]

#: Mirror of ``views_hydranet.distributions.registry.DISTRIBUTION_REGISTRY`` keys — the output
#: distributions that have a family to draw from. Anything else (notably ``standard``) does not.
SAMPLEABLE = frozenset({"nb", "zinb", "mixture_nb", "truncated_nb"})


def _load_hp(name: str) -> dict:
    path = REPO / "models" / name / "configs" / "config_hyperparameters.py"
    spec = importlib.util.spec_from_file_location(f"hp_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_hp_config()


def _hydranets() -> list[str]:
    return sorted(
        Path(p).parts[-3]
        for p in glob.glob(str(REPO / "models" / "*" / "configs" / "config_meta.py"))
        if '"algorithm": "HydraNet"' in Path(p).read_text(encoding="utf-8")
    )


def test_no_hydranet_feeds_back_a_draw_it_cannot_make():
    """`rollout_feedback='sample'` requires a sampleable `output_distribution`."""
    names = _hydranets()
    assert len(names) >= 30, f"only {len(names)} HydraNets found — the discovery is broken"

    offenders = {}
    for name in names:
        hp = _load_hp(name)
        if hp.get("rollout_feedback") == "sample":
            dist = hp.get("output_distribution")
            if dist not in SAMPLEABLE:
                offenders[name] = dist

    assert not offenders, (
        "these models feed back a sampled value from a head that has no distribution family, "
        "so they train and then die at the first forecast:\n"
        + "\n".join(f"  {n}: output_distribution={d!r}" for n, d in sorted(offenders.items()))
        + f"\n\nSet 'rollout_feedback': 'mean' in each, or remove the key. Sampleable heads are "
        f"{sorted(SAMPLEABLE)}."
    )


def test_the_check_above_is_not_vacuous():
    """Both sides of the rule must exist in the roster, or the check proves nothing.

    ⛔ The check is an implication. It passes trivially if nothing samples, and it exercises
    nothing if every head is sampleable. Both held at some point in this repo's history, so
    neither is hypothetical.
    """
    names = _hydranets()
    assert len(names) >= 30, f"only {len(names)} HydraNets found — the discovery is broken"

    configs = {n: _load_hp(n) for n in names}
    samplers = [n for n, hp in configs.items() if hp.get("rollout_feedback") == "sample"]
    familyless = [
        n for n, hp in configs.items() if hp.get("output_distribution") not in SAMPLEABLE
    ]

    assert samplers, (
        "no model sets rollout_feedback='sample', so the rule has no instances to check. "
        "Either the key was renamed upstream or the parsing broke."
    )
    assert familyless, (
        "every model has a sampleable head, so the rule cannot fire. If that is genuinely true "
        "now, this test should be deleted along with the one above, not left passing."
    )

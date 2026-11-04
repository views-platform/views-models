"""The twelve EXP arms pin their queryset `name` to the floor model, and must keep doing so.

⛔ THIS GUARD EXISTS TO PREVENT A FIX, WHICH IS UNUSUAL, SO HERE IS THE REASON (C-194).

An audit of self-naming fields reports these twelve as `name != directory` — the same shape as
the copy-paste defects in C-157 instance (6) and the 21 sweep names corrected in #708. Here the
obvious correction is the wrong move.

The descriptor's `name` is an INPUT to `queryset_digest`. Measured 2026-11-04 as a natural
experiment on digests already on disk:

    violet_visitor / copper_kestrel / indigo_kestrel / silver_kestrel
        byte-identical descriptors — region='land', loa='priogrid_month', same features,
        same zarr_url, pgm, months 121-504, calibration — each deriving `name` from its own path
        -> FOUR DIFFERENT digests (b32a221b…, 1ff83913…, f643329…, 2de3a05f…)

    the twelve arms, `name` pinned to "violet_visitor", plus violet_visitor itself
        -> ONE digest, b32a221b…

Same descriptor + different runtime name => different digest. Same name => same digest.

`tools/podrun/pod_run_exp02_arm.sh:86` pins a single `EXPECTED_DIGEST` and checks every arm's
provenance sidecar against it roughly 430 s into a run, before any GPU spend. Give each arm its
own name and each gets its own digest, so that one pinned value refuses eleven of twelve HEALTHY
arms — and the operator's next move is to widen or delete the gate, which is how a cheap
pre-spend check gets thrown away.

⚠️ This file asserts the arms agree with the FLOOR, not with a hardcoded "violet_visitor". The
floor is read from the arms' own majority, so renaming `violet_visitor` moves all thirteen
together without failing, while one arm drifting away fails. A hardcoded string would itself go
stale — the failure mode that produced #698.
"""

from __future__ import annotations

import ast
import glob
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.beige

REPO = Path(__file__).resolve().parents[1]

#: The arms are the models whose queryset pins a `model_name` literal instead of deriving it.
#: Discovered, not listed, so a thirteenth arm is covered the day it is scaffolded.
PINNED_LITERAL = "model_name"


def _pinned_name(path: Path) -> str | None:
    """The module-level `model_name = "..."` literal, or None when it is derived from the path."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == PINNED_LITERAL for t in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return None


PINNED: dict[str, str] = {}
for _p in sorted(glob.glob(str(REPO / "models" / "*" / "configs" / "config_queryset.py"))):
    _name = _pinned_name(Path(_p))
    if _name is not None:
        PINNED[Path(_p).parts[-3]] = _name


def test_some_model_pins_its_queryset_identity():
    """⛔ ADR-025: if the ast walk found nothing, every assertion below is vacuous.

    A zero here means either the arms were deleted or the walk broke. Both need a human, and
    neither should read as a pass.
    """
    assert len(PINNED) >= 12, (
        f"only {len(PINNED)} model(s) pin a `model_name` literal, expected at least 12 (the EXP "
        "arms). Either the arms are gone — in which case delete this file and C-194's guard "
        "reference — or the ast walk no longer matches how they declare it. Do not lower this "
        "floor to make the suite green."
    )


def test_every_pinned_arm_names_the_same_floor():
    """One arm drifting is the failure; a joint rename of the floor is not."""
    floor, n_floor = Counter(PINNED.values()).most_common(1)[0]
    odd = sorted(name for name, pinned in PINNED.items() if pinned != floor)
    assert not odd, (
        f"{len(PINNED) - n_floor} of {len(PINNED)} pinned arms disagree about the floor model.\n"
        f"  majority pins: {floor!r}\n"
        + "".join(f"  {name} pins: {PINNED[name]!r}\n" for name in odd)
        + "⛔ All arms sharing one pinned EXPECTED_DIGEST must share one queryset identity. "
        "An arm that names a different floor produces a different digest and will be refused by "
        "the gate ~430 s into a paid run. See C-194."
    )


def test_the_floor_model_exists_and_derives_its_own_name():
    """The floor must be a real model, and must NOT itself be pinned — or the chain is circular."""
    floor, _ = Counter(PINNED.values()).most_common(1)[0]
    floor_queryset = REPO / "models" / floor / "configs" / "config_queryset.py"
    assert floor_queryset.exists(), (
        f"the arms pin their identity to {floor!r}, which has no "
        f"models/{floor}/configs/config_queryset.py. The digest they share is then tied to a "
        "model that no longer exists."
    )
    assert _pinned_name(floor_queryset) is None, (
        f"{floor} is itself pinning a `model_name` literal. The floor must derive its name from "
        "its own path, otherwise the identity every arm shares is pinned to something that is "
        "itself pointing elsewhere."
    )

"""A model that declares a loss function must declare a target scaler (#537).

**The failure this prevents wastes a full training run and produces nothing.**

`brave_heart` died on 2026-10-09 after ~90 GPU-minutes:

    RuntimeError: NaN in SpotlightLossLogcosh: per_channel=[nan, nan, nan]

It was the only one of the eleven pgm darts models with no `target_scaler`. Three siblings
run the *same* loss and survived, because they scale their targets first.

**Why the omission is fatal rather than merely sloppy.** views-r2darts2 scales the target
through exactly one key (`dataset/base.py:1192-1223`):

    self._target_scaler = instantiate(target_scaler) if target_scaler is not None else None
    if self._target_scaler is not None:
        targets_ts = self._target_scaler.fit_transform(targets_ts)

So without it the loss sees raw fatality counts, which reach **113,395** in a single
cell-month of the calibration window. `logcosh` evaluates `cosh(x)`, which overflows float32
at about `x > 89`:

    log(cosh(asinh(113395))) = log(cosh(12.33)) = 11.64     finite
    log(cosh(113395))                           = inf    -> NaN

MSELoss would merely have been badly conditioned; logcosh overflows outright. The loss is
what decides whether the omission is survivable, which is why this guard keys on a loss being
declared at all rather than on a list of "dangerous" ones — a list would need updating every
time someone adds a loss, and the next one to overflow will not announce itself.

**What makes this hard to catch by reading.** `brave_heart` carries a `feature_scaler_map`
that explicitly lists `lr_ged_sb`, `lr_ged_ns` and `lr_ged_os` under `AsinhTransform`. It
looks exactly like target scaling. It is not: that map applies to columns used as *features*,
and the target path never consults it. A convincing near-miss, which is presumably how the
omission survived review.

**Scope.** Models declaring `loss_function`. Two darts models (`adolecent_slob`, `hot_stream`)
declare neither a loss nor a scaler and are therefore out of scope here — if they ever gain a
loss, this guard starts covering them, which is the right moment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Read as TEXT, not by importing. These configs are plain dicts, but the point of this guard
#: is the *declaration* — a model that computes its scaler at runtime would still be opaque to
#: a reviewer, and this file is about what the config says it does.
_LOSS = re.compile(r'^\s*"loss_function":\s*"([^"]+)"', re.M)
_TARGET_SCALER = re.compile(r'^\s*"target_scaler":\s*("?[\w.]+"?)', re.M)


def _declarations():
    for directory in sorted((REPO_ROOT / "models").glob("*")):
        path = directory / "configs" / "config_hyperparameters.py"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        loss = _LOSS.search(text)
        if not loss:
            continue
        scaler = _TARGET_SCALER.search(text)
        yield directory.name, loss.group(1), (scaler.group(1) if scaler else None)


DECLARED = list(_declarations())


def test_the_check_is_not_vacuous():
    """A parametrized test over an empty list passes while checking nothing."""
    assert len(DECLARED) >= 40, (
        f"only {len(DECLARED)} models declare a loss_function; this guard was written when 40 "
        "did. If models were retired, lower the floor deliberately."
    )


@pytest.mark.parametrize("name,loss,scaler", DECLARED, ids=[n for n, _, _ in DECLARED])
def test_a_declared_loss_comes_with_a_declared_target_scaler(name, loss, scaler):
    assert scaler is not None, (
        f"{name} declares loss_function={loss!r} and no target_scaler, so the loss will see "
        f"RAW target values. In this roster those reach 113,395 fatalities in one cell-month; "
        f"logcosh overflows float32 above ~89 and the run dies with NaN after training "
        f"completes. views-r2darts2 scales the target through `target_scaler` ALONE "
        f"(dataset/base.py:1192-1223) — a feature_scaler_map listing the target columns does "
        f"NOT do it, which is the trap brave_heart fell into. Declare "
        f'"target_scaler": "AsinhTransform" like its 39 siblings, or state in the config why '
        f"this model's targets are safe unscaled."
    )
    assert scaler.strip('"') != "None", (
        f"{name} declares target_scaler explicitly as None alongside loss_function={loss!r}. "
        f"If that is deliberate, the reason belongs in the config beside it — an explicit None "
        f"and a missing key fail identically at runtime."
    )


def test_the_target_is_scaled_by_target_scaler_alone_not_by_feature_scaler_map():
    """The premise of the guard above, asserted against the engine rather than assumed.

    If a future views-r2darts2 routes the target through the feature map too, this guard
    becomes unnecessary and should be reconsidered rather than left as cargo cult.
    """
    base = (
        REPO_ROOT.parent / "views-r2darts2" / "views_r2darts2" / "dataset" / "base.py"
    )
    if not base.is_file():
        pytest.skip("views-r2darts2 is not checked out beside this repo")
    text = base.read_text(encoding="utf-8")
    assert re.search(r"self\._target_scaler\s*=.*target_scaler is not None", text, re.S), (
        "the engine no longer gates target scaling on `target_scaler is not None`; re-read "
        "dataset/base.py and revisit whether this guard still describes reality"
    )
    assert re.search(
        r"if self\._target_scaler is not None:\s*\n\s*targets_ts = self\._target_scaler", text
    ), "the target transform is no longer applied where this guard assumes it is"

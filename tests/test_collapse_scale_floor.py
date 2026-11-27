"""The collapse scale floor must stay BELOW log1p's own ceiling, or it refuses the impossible.

`_check_scale` asks one question: is this frame still in log1p space? That makes its threshold
a derived quantity, not a preference — the bound is `log1p(largest count the data can hold)`.

⛔ It was 12.0, which is ABOVE that ceiling. Over the 13-month scoring window the largest
observed count is 2,482, and `log1p(2482) = 7.817`, so a frame genuinely in log1p space cannot
present a maximum above ~7.8. The band (7.82, 12.0) was therefore a refusal region in which the
dangerous hypothesis is IMPOSSIBLE and only a legitimate count-space forecast can sit. It
refused `quiet_lander` at 11.31 — a value that proves the frame is NOT log1p, since
`expm1(11.31) = 81,633`, a count no cell on the global-land grid reaches.

These tests pin the derivation rather than the number, so a future change has to restate the
reasoning instead of nudging a constant.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tools.collapse.collapse_predictions import (
    MIN_PLAUSIBLE_MAX,
    CollapseError,
    _check_scale,
)

#: The largest count observed over the 13-month scoring window (`ged_sb_best`), from
#: views-hydranet `reports/2026-10-27_metric_redteam/separation_matrix.out`. The ceiling of
#: log1p space follows from it and from nothing else.
LARGEST_OBSERVED_COUNT = 2482.0
LOG1P_CEILING = math.log1p(LARGEST_OBSERVED_COUNT)  # 7.817


@pytest.mark.green
def test_the_floor_sits_above_log1p_ceiling_but_not_by_much():
    """A floor below the ceiling cannot catch the hypothesis; far above it refuses the safe."""
    assert MIN_PLAUSIBLE_MAX > LOG1P_CEILING, (
        f"the floor {MIN_PLAUSIBLE_MAX} is at or below log1p's ceiling {LOG1P_CEILING:.3f}, so "
        f"a frame genuinely still in log1p space could present a maximum ABOVE the floor and "
        f"pass. The guard would then be unable to catch the one thing it exists for."
    )
    assert MIN_PLAUSIBLE_MAX < 2 * LOG1P_CEILING, (
        f"the floor {MIN_PLAUSIBLE_MAX} is more than twice log1p's ceiling "
        f"{LOG1P_CEILING:.3f}. Everything between the ceiling and the floor is a band where "
        f"the dangerous hypothesis is impossible and only a legitimate count-space forecast "
        f"can sit — the band this guard refused quiet_lander inside, at 11.31."
    )


@pytest.mark.red
def test_a_value_log1p_cannot_reach_is_not_refused():
    """11.31 implies a count of 81,633. That is proof of count space, not evidence against it."""
    frame = pd.DataFrame({"pred_lr_os_best": np.array([0.0, 3.0, 11.31], dtype="float32")})
    _check_scale(frame, "origin_5", MIN_PLAUSIBLE_MAX)  # must not raise
    assert math.expm1(11.31) > LARGEST_OBSERVED_COUNT * 10, (
        "the premise of this test is that 11.31 is far outside log1p's range; if the largest "
        "observed count grows by an order of magnitude, re-derive the floor rather than "
        "deleting this test."
    )


@pytest.mark.red
def test_a_frame_actually_left_in_log1p_space_is_still_refused():
    """The guard must keep catching the thing it exists for, at the top of log1p's range."""
    frame = pd.DataFrame({"pred_lr_sb_best": np.array([0.0, 2.5, LOG1P_CEILING], dtype="float32")})
    with pytest.raises(CollapseError) as e:
        _check_scale(frame, "origin_0", MIN_PLAUSIBLE_MAX)
    assert "log1p" in str(e.value)


@pytest.mark.red
def test_one_corrupted_target_is_caught_although_its_siblings_are_healthy():
    """Per-target, not flattened — a healthy sibling must not carry a corrupted one over."""
    frame = pd.DataFrame({
        "pred_lr_sb_best": np.array([0.0, 814.0], dtype="float32"),   # fine
        "pred_lr_os_best": np.array([0.0, 4.2], dtype="float32"),     # still log1p
    })
    with pytest.raises(CollapseError) as e:
        _check_scale(frame, "origin_3", MIN_PLAUSIBLE_MAX)
    assert "pred_lr_os_best" in str(e.value), (
        "the refusal must name the corrupted column; a combined maximum would have been "
        "carried over the floor by the healthy sibling and shipped a log1p'd target."
    )


@pytest.mark.beige
def test_the_refusal_states_the_count_the_value_implies():
    """A reader must be able to judge the claim without re-deriving expm1 by hand."""
    frame = pd.DataFrame({"pred_lr_ns_best": np.array([0.0, 5.0], dtype="float32")})
    with pytest.raises(CollapseError) as e:
        _check_scale(frame, "origin_1", MIN_PLAUSIBLE_MAX)
    msg = str(e.value)
    assert f"{math.expm1(5.0):,.0f}" in msg, (
        "the message should carry expm1(value) so the reader sees what count the value would "
        "mean if the frame really were log1p'd — that number is the whole argument."
    )

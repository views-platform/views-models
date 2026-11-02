"""Guards for `tools/publish_pooled_forecast.py`.

The script's job is to put an already-pooled, already-reconciled forecast on the
`production_forecasts` shelf — an **irreversible, outward-facing** write whose
`reconciled=True` flag reaches a UN partner through `deliveries/un_crafd.py`'s
`REQUIRE.reconciled`.

So these tests are about the refusals, not the happy path. Each one corresponds to a way the
script could put a wrong forecast on that shelf, and each has a prior in the register:

| refusal | prior |
|---|---|
| no `RECONCILIATION_RESIDUAL` at all | ADR-025 — absence is not a pass |
| a residual that says PARTIALLY | **C-192** (Tier 1), **C-145** |
| a pool missing a target | the `POOLABLE` gate in `pod_run_chrome_box.sh`, one hop later |
| a directory name the run_id cannot be derived from | `source_selection` fails loud *after* upload |

⭐ **None of these tests can publish anything.** They exercise the guards, which all run before
any credential is read and before any store is constructed — the script imports
`PredictionStoreConfig` only inside the `--publish` branch. A test that reached the upload path
would need the nine shelf variables, and this file never sets them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.publish_pooled_forecast import (  # noqa: E402
    POOL_DIR,
    TARGETS,
    Refuse,
    main,
    refuse_a_partial_pool,
    refuse_unless_the_residual_says_fully,
    run_identity,
)

FULLY = "=== RECONCILIATION RESIDUAL ===\n\n✅ FULLY RECONCILED. Every country's mass is on the grid.\n"
PARTIAL = "=== RECONCILIATION RESIDUAL ===\n\n⚠️  UP TO 7.321% OF THE COUNTRY FORECAST'S FATALITIES ARE NOT ON THE\n   GRID.\n"
GARBAGE = "truncated\n"


def _pool(tmp_path: Path, *, targets=TARGETS, rows=4, draws=3, name="predictions_forecasting_20261102_120000") -> Path:
    """A minimally valid pool: one directory per target, each with the two arrays.

    Tiny on purpose — the guards under test read directory structure and one text file, never
    the array contents, so a 4-row frame exercises them exactly as a 2,330,712-row one does.
    """
    pool = tmp_path / "parquet" / name
    for target in targets:
        d = pool / target
        d.mkdir(parents=True)
        np.save(d / "y_pred.npy", np.arange(rows * draws, dtype=np.float32).reshape(rows, draws))
        np.savez(
            d / "identifiers.npz",
            time=np.arange(rows, dtype=np.int64),
            unit=np.arange(rows, dtype=np.int64),
        )
    return pool


# --------------------------------------------------------------------------------------
# The guard that is about truth rather than shape
# --------------------------------------------------------------------------------------


def test_a_missing_residual_refuses_rather_than_assuming_reconciled(tmp_path: Path) -> None:
    """⛔ Absence is not a pass (ADR-025).

    ⚠️ Not hypothetical: the retained 2026-10-30 `chrome_box` delivery carries `STATUS: OK`,
    a MANIFEST, and **no** `RECONCILIATION_RESIDUAL` — because it predates the residual check.
    That is the delivery which was short by 4.023% / 7.321% / 2.943%. Run against it, the
    script refuses here, which is the behaviour this test exists to keep.
    """
    pool = _pool(tmp_path)
    with pytest.raises(Refuse, match="Absence is not a pass"):
        refuse_unless_the_residual_says_fully(pool)


def test_a_partially_reconciled_residual_refuses(tmp_path: Path) -> None:
    """C-192 at Tier 1: the run reports OK and the mass is not on the grid."""
    pool = _pool(tmp_path)
    (pool.parent.parent / "RECONCILIATION_RESIDUAL").write_text(PARTIAL)
    with pytest.raises(Refuse, match="PARTIALLY"):
        refuse_unless_the_residual_says_fully(pool)


def test_an_unreadable_residual_refuses_and_says_so(tmp_path: Path) -> None:
    """A residual whose format changed is UNREADABLE, which is neither verdict.

    It must not fall through to either — the whole point of ADR-025's third verdict.
    """
    pool = _pool(tmp_path)
    (pool.parent.parent / "RECONCILIATION_RESIDUAL").write_text(GARBAGE)
    with pytest.raises(Refuse, match="UNREADABLE"):
        refuse_unless_the_residual_says_fully(pool)


def test_a_fully_reconciled_residual_is_accepted_and_its_path_returned(tmp_path: Path) -> None:
    """The one accepting case, and it returns WHICH file it read.

    Returning the path rather than a bool is deliberate: the script prints it, so the operator
    sees which file the verdict came from rather than taking "checked" on trust (C-191).
    """
    pool = _pool(tmp_path)
    expected = pool.parent.parent / "RECONCILIATION_RESIDUAL"
    expected.write_text(FULLY)
    assert refuse_unless_the_residual_says_fully(pool) == expected


def test_the_residual_is_found_beside_parquet_not_only_inside_the_pool(tmp_path: Path) -> None:
    """⛔ Where the pod actually puts it.

    `pod_run_chrome_box.sh` writes `RECONCILIATION_RESIDUAL` into `$OUT`, beside `parquet/`,
    while the pool is `$OUT/parquet/predictions_forecasting_<ts>/`. A guard that looked only
    inside the pool would never find it and would refuse every real delivery — so the search
    walks upward, and this pins that it does.
    """
    pool = _pool(tmp_path)
    (tmp_path / "RECONCILIATION_RESIDUAL").write_text(FULLY)  # two levels above the pool
    assert refuse_unless_the_residual_says_fully(pool).parent == tmp_path


# --------------------------------------------------------------------------------------
# Shape guards
# --------------------------------------------------------------------------------------


def test_a_pool_missing_a_target_refuses(tmp_path: Path) -> None:
    """All three or none — a short run on the shelf reports success.

    The mirror of `pod_run_chrome_box.sh`'s `POOLABLE` gate, which refuses to pool 7 members
    of 8 because nothing downstream notices. This is that gate one hop later: nothing
    downstream notices a manifest naming two streams where `REQUIRE.targets` declares three.
    """
    pool = _pool(tmp_path, targets=("lr_sb_best", "lr_os_best"))
    with pytest.raises(Refuse, match="lr_ns_best"):
        refuse_a_partial_pool(pool, TARGETS)


def test_a_target_directory_missing_its_identifiers_refuses(tmp_path: Path) -> None:
    """Both arrays, not just `y_pred.npy`.

    `load_pf` needs `identifiers.npz` to build the `SpatioTemporalIndex`; a directory holding
    only the values would pass a `is_dir()` check and fail at load — after the other targets
    may already have been published, had the loads not all been hoisted first.
    """
    pool = _pool(tmp_path)
    (pool / "lr_ns_best" / "identifiers.npz").unlink()
    with pytest.raises(Refuse, match="lr_ns_best"):
        refuse_a_partial_pool(pool, TARGETS)


def test_the_run_id_is_derived_from_the_directory_name(tmp_path: Path) -> None:
    """The shape `prediction_frame_ensemble.py:781` produces, reproduced exactly.

    `f"{name}_{run_type}_{timestamp}"`. views-postprocessing's `source_selection` matches on
    it and the wire's staging directory is named after it, so a divergent id fails loud there
    — after an irreversible upload.
    """
    pool = _pool(tmp_path, name="predictions_forecasting_20261102_143000")
    assert run_identity(pool, "chrome_box") == ("chrome_box_forecasting_20261102_143000", "forecasting")


@pytest.mark.parametrize(
    "bad",
    ["parquet", "pod_delivery_20261102", "predictions_forecasting", "predictions_forecasting_2026"],
)
def test_a_directory_the_run_id_cannot_be_derived_from_refuses(tmp_path: Path, bad: str) -> None:
    """⛔ Including `pod_delivery_*`, which is what an operator is most likely to point at.

    The rsync target in the guide is `pod_delivery_<date>/`; the pool is the
    `predictions_forecasting_<ts>/` directory *inside* it. Pointing `--pool` at the wrong one
    is the obvious mistake, and it must not produce a run_id built from a date that is not the
    run's timestamp.
    """
    pool = _pool(tmp_path, name=bad)
    with pytest.raises(Refuse, match="not a pool directory name"):
        run_identity(pool, "chrome_box")


def test_the_pool_pattern_requires_a_full_timestamp() -> None:
    """Read the regex rather than infer it — the run identity depends on both groups."""
    m = POOL_DIR.match("predictions_calibration_20261102_143000")
    assert m is not None and m["run_type"] == "calibration" and m["ts"] == "20261102_143000"
    assert POOL_DIR.match("predictions_forecasting_20261102") is None


# --------------------------------------------------------------------------------------
# The default must not publish
# --------------------------------------------------------------------------------------


def test_the_default_run_publishes_nothing_and_constructs_no_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """⛔ `--publish` is opt-in, and the dry path must not touch a credential.

    Enforced by poisoning the import: if the default path reaches
    `PredictionStoreConfig.from_environment()`, this fails. That is stronger than asserting on
    the printed text, because the text could say DRY RUN while a store had already been built.
    """
    pool = _pool(tmp_path)
    (tmp_path / "RECONCILIATION_RESIDUAL").write_text(FULLY)

    import views_pipeline_core.configs.prediction_store as ps

    def explode() -> None:
        raise AssertionError("the dry path read a credential")

    monkeypatch.setattr(ps.PredictionStoreConfig, "from_environment", staticmethod(explode))

    assert main(["--pool", str(pool)]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "nothing was sent" in out
    assert "chrome_box_forecasting_20261102_120000" in out


def test_targets_disagreeing_on_the_draw_count_refuse(tmp_path: Path) -> None:
    """One target not from this run would put an unpoolable set on the shelf.

    Checked after loading all three and before publishing any — the loads are hoisted for
    exactly this reason, so a disagreement is caught while the shelf is still untouched.
    """
    pool = _pool(tmp_path)
    (tmp_path / "RECONCILIATION_RESIDUAL").write_text(FULLY)
    d = pool / "lr_os_best"
    np.save(d / "y_pred.npy", np.zeros((4, 7), dtype=np.float32))  # 7 draws, not 3
    with pytest.raises(Refuse, match="disagree on the draw count"):
        main(["--pool", str(pool)])


# --------------------------------------------------------------------------------------
# ⛔ THE WIRING. The tests above prove the guards WORK; these prove main() CALLS them.
#
# Added after a mutation pass caught the gap: deleting
# `refuse_unless_the_residual_says_fully(pool)` and `refuse_a_partial_pool(pool, TARGETS)`
# from `main()` left all fifteen tests green, because every refusal was exercised as a unit
# and the only tests reaching `main()` were ones set up to pass. A guard that exists and is
# not called is not a guard, and the unit tests could not see the difference.
#
# This is the same defect as C-182 one level up: the subject set was right and the unit of
# attribution was wrong. The audit question that finds it — "if exactly one call were
# dropped, which test fails?" — had no answer. Now each of these is that answer.
# --------------------------------------------------------------------------------------


def test_main_refuses_a_pool_whose_residual_is_missing(tmp_path: Path) -> None:
    """Through main(), not the helper — pins that main() calls the residual guard."""
    pool = _pool(tmp_path)
    with pytest.raises(Refuse, match="Absence is not a pass"):
        main(["--pool", str(pool)])


def test_main_refuses_a_pool_whose_residual_says_partially(tmp_path: Path) -> None:
    """Through main() — the C-192 case, wired."""
    pool = _pool(tmp_path)
    (tmp_path / "RECONCILIATION_RESIDUAL").write_text(PARTIAL)
    with pytest.raises(Refuse, match="PARTIALLY"):
        main(["--pool", str(pool)])


def test_main_refuses_a_partial_pool(tmp_path: Path) -> None:
    """Through main() — pins that main() calls the partial-pool guard.

    ⚠️ The residual is FULLY here on purpose. If main() stopped calling
    `refuse_a_partial_pool`, the missing target would instead surface as a `load_pf`
    FileNotFoundError, which is a different exception and a worse message — so this asserts
    on `Refuse` and the target's name, not merely that something raised.
    """
    pool = _pool(tmp_path, targets=("lr_sb_best", "lr_os_best"))
    (tmp_path / "RECONCILIATION_RESIDUAL").write_text(FULLY)
    with pytest.raises(Refuse, match="lr_ns_best"):
        main(["--pool", str(pool)])


def test_main_refuses_a_directory_the_run_id_cannot_come_from(tmp_path: Path) -> None:
    """Through main() — pins the run_identity call, with everything else valid."""
    pool = _pool(tmp_path, name="pod_delivery_20261102")
    (tmp_path / "RECONCILIATION_RESIDUAL").write_text(FULLY)
    with pytest.raises(Refuse, match="not a pool directory name"):
        main(["--pool", str(pool)])


def test_main_checks_the_pool_before_reading_anything_expensive(tmp_path: Path) -> None:
    """Order matters: the refusals must precede the loads.

    A guard that runs after `load_pf` has mmapped three 27 GiB arrays still refuses, but it
    refuses minutes later and after touching the data. `main()` resolves the run identity and
    both refusals before it imports `load_pf` at all, and the cheapest way to pin that is to
    make the load impossible: the target directories hold no arrays, so if any load were
    attempted first this would raise something other than `Refuse`.
    """
    pool = tmp_path / "parquet" / "predictions_forecasting_20261102_120000"
    for target in TARGETS:
        (pool / target).mkdir(parents=True)  # directories, no arrays
    (tmp_path / "RECONCILIATION_RESIDUAL").write_text(FULLY)
    with pytest.raises(Refuse, match="no y_pred.npy"):
        main(["--pool", str(pool)])


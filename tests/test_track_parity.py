"""Track A/B parity tests — verify numpy (.npy) and parquet (.parquet) predictions match.

Track A: PredictionFrame format — y_pred.npy + identifiers.npz per origin/target
Track B: DataFrame delivery — .parquet with list-column of posterior samples per origin/target

These tracks are produced simultaneously by the prediction delivery pipeline.
This test verifies they contain identical values so Track B can be safely retired.

See risk register C-47.
"""

import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
_FIXTURE_MODELS = {"fake_model"}

ALL_MODEL_DIRS = sorted(
    d for d in MODELS_DIR.iterdir()
    if d.is_dir() and (d / "main.py").exists()
    and d.name not in _FIXTURE_MODELS
) if MODELS_DIR.exists() else []

pytestmark = [pytest.mark.green]


def _find_prediction_runs(model_dir: Path) -> list[tuple[str, str]]:
    """Find all (run_type, timestamp) pairs that have both Track A dirs and Track B parquets."""
    gen_dir = model_dir / "data" / "generated"
    if not gen_dir.exists():
        return []

    # Track A directories: predictions_{run_type}_{timestamp}/
    track_a = {}
    for d in gen_dir.iterdir():
        if d.is_dir() and d.name.startswith("predictions_"):
            match = re.match(r"predictions_(\w+?)_(\d{8}_\d{6})$", d.name)
            if match:
                run_type, timestamp = match.groups()
                track_a[(run_type, timestamp)] = d

    # Track B parquets: predictions_{run_type}_{timestamp}_{target}_{origin}.parquet
    track_b_timestamps = set()
    for f in gen_dir.iterdir():
        if f.is_file() and f.name.startswith("predictions_") and f.suffix == ".parquet":
            match = re.match(r"predictions_(\w+?)_(\d{8}_\d{6})_", f.name)
            if match:
                track_b_timestamps.add((match.group(1), match.group(2)))

    return sorted(track_a.keys() & track_b_timestamps)


def _get_track_a_targets_origins(pred_dir: Path) -> list[tuple[str, int]]:
    """List all (target, origin) pairs available in Track A."""
    pairs = []
    for origin_dir in sorted(pred_dir.iterdir()):
        if not origin_dir.is_dir() or not origin_dir.name.startswith("origin_"):
            continue
        origin = int(origin_dir.name.split("_")[1])
        for target_dir in sorted(origin_dir.iterdir()):
            if target_dir.is_dir() and (target_dir / "y_pred.npy").exists():
                pairs.append((target_dir.name, origin))
    return pairs


def _load_track_b(gen_dir: Path, run_type: str, timestamp: str, target: str, origin: int) -> np.ndarray:
    """Load Track B parquet and extract the posterior samples array."""
    import pandas as pd

    fname = f"predictions_{run_type}_{timestamp}_{target}_{origin:02d}.parquet"
    fpath = gen_dir / fname
    if not fpath.exists():
        return None
    df = pd.read_parquet(fpath)
    pred_col = [c for c in df.columns if c.startswith("pred_")]
    if not pred_col:
        return None
    samples = np.stack(df[pred_col[0]].values)
    return samples


_PARITY_CASES = []
for model_dir in ALL_MODEL_DIRS:
    for run_type, timestamp in _find_prediction_runs(model_dir):
        pred_dir = model_dir / "data" / "generated" / f"predictions_{run_type}_{timestamp}"
        for target, origin in _get_track_a_targets_origins(pred_dir):
            _PARITY_CASES.append((model_dir, run_type, timestamp, target, origin))


@pytest.mark.skipif(not _PARITY_CASES, reason="No models with both Track A and Track B predictions")
class TestTrackParity:
    """Verify Track A (numpy) and Track B (parquet) contain identical prediction values."""

    @pytest.fixture(params=_PARITY_CASES[:20] if len(_PARITY_CASES) > 20 else _PARITY_CASES,
                    ids=[f"{c[0].name}/{c[1]}/{c[3]}/o{c[4]}" for c in
                         (_PARITY_CASES[:20] if len(_PARITY_CASES) > 20 else _PARITY_CASES)])
    def parity_case(self, request):
        return request.param

    def test_values_match(self, parity_case):
        """Track A .npy and Track B .parquet must contain identical posterior samples."""
        model_dir, run_type, timestamp, target, origin = parity_case
        gen_dir = model_dir / "data" / "generated"

        track_a = np.load(
            gen_dir / f"predictions_{run_type}_{timestamp}" / f"origin_{origin}" / target / "y_pred.npy"
        )

        track_b = _load_track_b(gen_dir, run_type, timestamp, target, origin)
        if track_b is None:
            pytest.skip(f"Track B parquet not found for {target} origin {origin}")

        assert track_a.shape == track_b.shape, (
            f"Shape mismatch: Track A {track_a.shape} vs Track B {track_b.shape}"
        )
        np.testing.assert_array_equal(
            track_a, track_b,
            err_msg=f"Value mismatch between Track A and Track B for {model_dir.name}/{target}/origin_{origin}",
        )

    def test_row_ordering_matches_identifiers(self, parity_case):
        """Track B row ordering must match Track A identifiers (time, unit)."""
        import pandas as pd

        model_dir, run_type, timestamp, target, origin = parity_case
        gen_dir = model_dir / "data" / "generated"

        ids = np.load(
            gen_dir / f"predictions_{run_type}_{timestamp}" / f"origin_{origin}" / target / "identifiers.npz"
        )

        fname = f"predictions_{run_type}_{timestamp}_{target}_{origin:02d}.parquet"
        fpath = gen_dir / fname
        if not fpath.exists():
            pytest.skip("Track B parquet not found")

        df = pd.read_parquet(fpath)

        if "month_id" in df.columns and "priogrid_id" in df.columns:
            np.testing.assert_array_equal(
                df["month_id"].values, ids["time"],
                err_msg="month_id ordering mismatch between Track B and Track A identifiers",
            )
            np.testing.assert_array_equal(
                df["priogrid_id"].values, ids["unit"],
                err_msg="priogrid_id ordering mismatch between Track B and Track A identifiers",
            )

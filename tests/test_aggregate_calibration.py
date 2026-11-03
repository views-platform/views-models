"""Guards for the aggregate-calibration tool.

Built against synthetic posteriors whose answer is known by construction: one where cells are
independent across draws, one where every draw is a coherent joint scenario, and one where the
predictive systematically under-shoots the observed totals. A diagnostic that cannot tell those
three apart is not a diagnostic.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

N_CELLS, N_MONTHS, N_DRAWS = 25, 2, 128


def _write(tmp_path: Path, draws: np.ndarray) -> Path:
    root = tmp_path / "predictions_forecasting_20260930_000000" / "lr_sb_best"
    root.mkdir(parents=True)
    n = draws.shape[0]
    np.save(root / "y_pred.npy", draws.astype(np.float32))
    np.savez(
        root / "identifiers.npz",
        time=np.repeat(np.arange(561, 561 + N_MONTHS), n // N_MONTHS).astype(np.int64),
        unit=np.tile(np.arange(1000, 1000 + n // N_MONTHS), N_MONTHS).astype(np.int64),
    )
    return root.parent


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "tools.prereg.aggregate_calibration", *[str(a) for a in args]],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )


def _independent(rng):
    """Every cell drawn on its own. Var(sum) == sum Var by construction, so ratio -> 1."""
    return rng.gamma(2.0, 3.0, size=(N_CELLS * N_MONTHS, N_DRAWS))


def _joint(rng):
    """Every draw is a scenario: one shared multiplier moves all cells together. Positive
    dependence, so Var(sum) must exceed sum Var."""
    base = rng.gamma(2.0, 3.0, size=(N_CELLS * N_MONTHS, 1))
    scenario = rng.gamma(2.0, 0.5, size=(1, N_DRAWS))
    return base * scenario


class TestDependenceIsDetected:
    def test_independent_cells_give_a_ratio_near_one(self, tmp_path):
        pooled = _write(tmp_path, _independent(np.random.default_rng(0)))
        out = tmp_path / "r.json"
        r = _run(pooled, "--json-out", out)
        assert r.returncode == 0, r.stdout + r.stderr
        ratios = [x["dependence_ratio"] for x in json.loads(out.read_text())["lr_sb_best"]]
        assert all(0.7 < x < 1.4 for x in ratios), (
            f"independent cells should give Var(sum) ~ sum Var; got {ratios}"
        )

    def test_joint_scenarios_give_a_ratio_well_above_one(self, tmp_path):
        pooled = _write(tmp_path, _joint(np.random.default_rng(0)))
        out = tmp_path / "r.json"
        _run(pooled, "--json-out", out)
        ratios = [x["dependence_ratio"] for x in json.loads(out.read_text())["lr_sb_best"]]
        assert all(x > 2.0 for x in ratios), (
            "draws that move together must show positive dependence — this is the property that "
            f"makes an aggregate interval wide enough to be honest; got {ratios}"
        )

    def test_the_independent_case_is_called_out_in_the_output(self, tmp_path):
        """F1 is the pre-registered falsifier. It has to be visible, not only in the JSON."""
        pooled = _write(tmp_path, _independent(np.random.default_rng(1)))
        assert "F1" in _run(pooled).stdout


class TestTheForecastCaseIsHonestAboutWhatItCannotScore:
    def test_without_actuals_it_says_so(self, tmp_path):
        pooled = _write(tmp_path, _independent(np.random.default_rng(2)))
        out = _run(pooled).stdout
        assert "FORECASTING case" in out and "do not" in out.lower(), (
            "a forecast has no observations yet; the tool must say that rather than imply a "
            f"comparison it did not make.\n{out}"
        )

    def test_without_actuals_no_row_claims_a_pit(self, tmp_path):
        pooled = _write(tmp_path, _independent(np.random.default_rng(3)))
        out = tmp_path / "r.json"
        _run(pooled, "--json-out", out)
        assert all("pit" not in x for x in json.loads(out.read_text())["lr_sb_best"])


class TestUnderPredictionIsCaught:
    @staticmethod
    def _actuals(tmp_path, pooled, multiplier):
        """Observed totals set to a multiple of the predictive mean, so the expected verdict is
        known before the tool runs."""
        import pandas as pd
        out = tmp_path / "pre.json"
        _run(pooled, "--json-out", out)
        rows = json.loads(out.read_text())["lr_sb_best"]
        df = pd.DataFrame([
            {"month_id": r["month_id"], "unit": r["unit"],
             "observed_total": r["total_mean"] * multiplier}
            for r in rows
        ])
        path = tmp_path / "actuals.parquet"
        df.to_parquet(path)
        return path

    def test_a_timid_model_trips_F2_and_F4(self, tmp_path):
        """Observed totals ten times the predictive mean: PIT pinned at 1, MCR ~0.1."""
        pooled = _write(tmp_path, _independent(np.random.default_rng(4)))
        actuals = self._actuals(tmp_path, pooled, multiplier=10.0)
        out = _run(pooled, "--actuals", actuals).stdout
        assert "F2" in out, f"under-prediction not flagged\n{out}"
        assert "F4" in out, f"MCR not flagged\n{out}"

    def test_a_well_centred_model_trips_neither(self, tmp_path):
        """The control. Without it, a tool that printed F2/F4 unconditionally would pass above."""
        pooled = _write(tmp_path, _independent(np.random.default_rng(5)))
        actuals = self._actuals(tmp_path, pooled, multiplier=1.0)
        out = _run(pooled, "--actuals", actuals).stdout
        assert "F2" not in out and "F4" not in out, f"false alarm on a centred model\n{out}"
        assert "MCR median" in out

    def test_mcr_is_reported_as_a_ratio_to_observed(self, tmp_path):
        pooled = _write(tmp_path, _independent(np.random.default_rng(6)))
        actuals = self._actuals(tmp_path, pooled, multiplier=4.0)
        out = tmp_path / "r.json"
        _run(pooled, "--actuals", actuals, "--json-out", out)
        mcrs = [x["mcr"] for x in json.loads(out.read_text())["lr_sb_best"]]
        assert all(abs(m - 0.25) < 0.01 for m in mcrs), (
            f"observed set to 4x the predictive mean must give MCR 0.25; got {mcrs}"
        )

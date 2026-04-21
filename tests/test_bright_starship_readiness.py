"""Failing test stubs from falsification audit: "ready to run bright_starship"

Generated: 2026-04-21
Source: /falsify audit
Findings: F-1 (hard), F-2 (hard), F-3 (soft)

These tests document pre-flight blockers discovered before first run.
They SHOULD fail until the blockers are resolved — that's the point.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIGHT_STARSHIP = REPO_ROOT / "models" / "bright_starship"


class TestF1_DatafactoryQueryDependency:
    """F-1 (hard): datafactory_query must be importable in the run environment.

    Without it, any cache-miss fetch in _ensure_data() crashes at:
        from datafactory_query import load_dataset
    """

    def test_datafactory_query_importable(self):
        """datafactory_query must be installed for bright_starship data fetching."""
        result = subprocess.run(
            ["conda", "run", "-n", "views-hydranet-env",
             "python", "-c", "import datafactory_query"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            "datafactory_query is not installed in views-hydranet-env. "
            "Install via: conda run -n views-hydranet-env pip install "
            "'views-datafactory @ git+https://github.com/views-platform/"
            "views-datafactory.git@development'"
        )


class TestF2_CalibrationParquetCached:
    """F-2 (hard): calibration parquet must exist if datafactory_query is unavailable.

    bright_starship has validation and forecasting parquets cached, but
    calibration is missing. The standard first run (-r calibration -t -e)
    will trigger _ensure_data() → fetch_data() → datafactory_query import → crash.
    """

    def test_calibration_parquet_exists(self):
        """calibration_viewser_df.parquet must be cached for offline runs."""
        parquet = BRIGHT_STARSHIP / "data" / "raw" / "calibration_viewser_df.parquet"
        assert parquet.exists(), (
            f"Missing: {parquet.relative_to(REPO_ROOT)}. "
            "Either fetch calibration data first (requires datafactory_query) "
            "or copy from a machine that has it cached."
        )


class TestF3_RunShEnvironment:
    """F-3 (soft): run.sh expects conda env at envs/views-hydranet.

    The env doesn't exist locally. run.sh will create it from scratch
    (~10 min), which would install datafactory via requirements.txt.
    Not "ready to run" — more "ready to bootstrap then run."
    """

    def test_run_sh_conda_env_exists(self):
        """The conda env expected by run.sh must exist."""
        env_path = REPO_ROOT / "envs" / "views-hydranet"
        assert env_path.is_dir(), (
            f"Missing conda env at {env_path.relative_to(REPO_ROOT)}. "
            "run.sh will attempt to create it from scratch. "
            "Either run `run.sh` once to bootstrap, or use "
            "`conda run -n views-hydranet-env` with datafactory_query installed."
        )

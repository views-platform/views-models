"""Readiness pre-flight + static dependency contracts for the datafactory models.

History: born 2026-04-21 as falsification stubs for "ready to run
bright_starship" (F-1 hard, F-2 hard). F-1 (datafactory_query importable)
is resolved on the workstation — views-hydranet-env is provisioned and the
probe passes there.

Re-scoped 2026-06-12 (issue #122, register C-75): the conda probe used to
false-red in CI — its only guard was `which("conda")`, truthy on runners
that have miniconda but not the workstation env. The probe is now a
workstation pre-flight that SKIPS truthfully when the target env is absent,
and the CI-meaningful contract is covered by static checks that need no
datafactory install (the queryset imports `datafactory_query` at module
level, so executing it on CI is impossible until views-datafactory has a
pinned release — see the C-73 lesson and the hybrid follow-up issue).

Coverage is parametrized over BOTH datafactory models — bright_starship and
shining_codex (register C-41: shining_codex previously had none).
"""
import ast
import functools
import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

BRIGHT_STARSHIP = MODELS_DIR / "bright_starship"

# model -> conda env (matched by basename, so both named envs like
# 'views-hydranet-env' and run.sh path envs like 'envs/views-r2darts2' resolve)
DATAFACTORY_MODELS = {
    "bright_starship": "views-hydranet-env",
    "shining_codex": "views-r2darts2",
}


@functools.lru_cache(maxsize=None)
def _conda_env_path(name):
    """Full path of the conda env whose directory name is `name`, else None.

    None on any conda absence/failure — callers skip, never false-red (C-75).
    """
    if not shutil.which("conda"):
        return None
    try:
        out = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True, text=True, timeout=120,
        )
        envs = json.loads(out.stdout).get("envs", [])
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None
    for env in envs:
        if Path(env).name == name:
            return env
    return None


@pytest.mark.red
class TestF1_DatafactoryQueryDependency:
    """Workstation pre-flight: datafactory_query must be importable in the env
    that runs each datafactory model — without it, any cache-miss fetch in
    `_ensure_data()` crashes at `from datafactory_query import load_dataset`.
    Skips (truthfully) wherever the env doesn't exist, e.g. CI."""

    @pytest.mark.parametrize("model,env_name", DATAFACTORY_MODELS.items())
    def test_datafactory_query_importable(self, model, env_name):
        env_path = _conda_env_path(env_name)
        if env_path is None:
            pytest.skip(
                f"{model}: conda env '{env_name}' not present on this machine "
                "(workstation pre-flight; static contract checks below still run)"
            )
        result = subprocess.run(
            ["conda", "run", "-p", env_path, "python", "-c", "import datafactory_query"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"{model}: datafactory_query is not installed in {env_name}. "
            f"Install via: conda run -p {env_path} pip install "
            "'views-datafactory @ git+https://github.com/views-platform/"
            f"views-datafactory.git@development'\nstderr: {result.stderr[-300:]}"
        )


@pytest.mark.green
class TestEnvGuardSanity:
    """The skip guard itself must be trustworthy (a guard that errors or lies
    recreates the C-75 false red)."""

    def test_nonexistent_env_reports_absent(self):
        assert _conda_env_path("definitely-not-a-real-env-xyz") is None


@pytest.mark.beige
@pytest.mark.parametrize("model", list(DATAFACTORY_MODELS))
class TestDatafactoryContract:
    """CI-runnable static contract: each datafactory model declares and wires
    its dependency. Text/AST checks only — config_queryset.py imports
    datafactory_query at module level, so it cannot be imported here without
    views-datafactory installed (no pinned release exists yet)."""

    def test_requirements_declare_views_datafactory(self, model):
        req = (MODELS_DIR / model / "requirements.txt").read_text()
        assert "views-datafactory" in req, (
            f"{model}/requirements.txt does not declare views-datafactory — "
            "a fresh run.sh env could not fetch data"
        )

    def test_queryset_imports_datafactory_query(self, model):
        source = (MODELS_DIR / model / "configs" / "config_queryset.py").read_text()
        assert "datafactory_query" in source, (
            f"{model}/configs/config_queryset.py no longer references "
            "datafactory_query — the declared dependency and the code disagree"
        )

    def test_queryset_defines_generate(self, model):
        source = (MODELS_DIR / model / "configs" / "config_queryset.py").read_text()
        tree = ast.parse(source)
        assert any(
            isinstance(node, ast.FunctionDef) and node.name == "generate"
            for node in ast.walk(tree)
        ), f"{model}/configs/config_queryset.py has no generate() function"


@pytest.mark.red
@pytest.mark.xfail(reason="F-2 pre-flight blocker: calibration_viewser_df.parquet not yet cached", strict=False)
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

"""The catalogs job must not abort on a model whose data client is not installed (#478).

pipeline-core 3.3.0 made ``ModelPathManager.get_queryset`` raise ``ImportError`` (with an
install hint) when ``config_queryset.py`` imports a client that is absent; 3.2.0 returned
``None``. ``tools/catalogs/update_readme.py`` calls it for every model in a job that installs
no data client, so without per-model isolation the job dies on the first datafactory model
and no README is regenerated. This runs the real script, as the job does (``cwd`` is the
repo), against a temporary repo holding one model whose queryset imports a module that
cannot exist — under both loader behaviours the script must finish and write the README.

Not mocked: the script is monolithic (register C-81/C-93), so the honest test is the script.

The fixture's client is a module name pipeline-core's install-hint table has never heard of,
on purpose: ``datafactory_query`` or ``viewser`` would import fine on a developer machine
and the test would prove nothing there. Both of the loader's paths — the hint-carrying
``ImportError`` for a known client and the bare re-raise for an unknown one — are
``ImportError``s, and the script's ``except`` catches the class. What this test does NOT
cover is C-81 itself: ``ModelPathManager(configs_dir)`` at construction (``validate=True``)
still aborts the job on a model directory missing a standard subfolder, before the
queryset is ever reached. That defect is registered and open; this guard is narrower.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "tools" / "catalogs" / "update_readme.py"
# Any real model with a config_queryset.py; its queryset import is replaced below.
DONOR = REPO_ROOT / "models" / "purple_alien"


@pytest.fixture
def tmp_repo(tmp_path):
    for d in ("tools", "deliveries", "meta"):
        shutil.copytree(REPO_ROOT / d, tmp_path / d, ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "ensembles").mkdir()
    (tmp_path / "models").mkdir()
    # ModelPathManager finds the repo root by pyprojroot (.git) and then by a .gitignore file.
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("")
    shutil.copy(REPO_ROOT / "models" / "README_scaffold.md", tmp_path / "models" / "README_scaffold.md")
    model = tmp_path / "models" / DONOR.name

    def _dirs_configs_and_readme(directory, names):
        # The donor's directory TREE (ModelPathManager validates it), its configs, its README —
        # nothing else: no artifacts, data, logs or notebooks come along.
        keep_files = Path(directory).name == "configs" or Path(directory) == DONOR
        return [n for n in names if not (Path(directory) / n).is_dir()
                and not (keep_files and (n.endswith(".py") or n == "README.md"))]

    shutil.copytree(DONOR, model, ignore=_dirs_configs_and_readme)
    (model / "configs" / "config_queryset.py").write_text(
        "import definitely_not_an_installed_data_client_478 as client  # noqa: F401\n"
        "def generate():\n    return client.Queryset()\n"
    )
    return tmp_path


def test_the_script_finishes_and_writes_the_readme_when_the_data_client_is_absent(tmp_repo):
    before = (tmp_repo / "models" / DONOR.name / "README.md").read_text()
    result = subprocess.run(
        [sys.executable, str(tmp_repo / "tools" / "catalogs" / "update_readme.py")],
        cwd=tmp_repo, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        "update_readme.py aborted on a model whose data client is not installed — the "
        f"catalogs job would regenerate nothing:\n{result.stderr[-2000:]}"
    )
    after = (tmp_repo / "models" / DONOR.name / "README.md").read_text()
    assert "No description provided" in after, after[:600]
    assert after != before or "No description provided" in before

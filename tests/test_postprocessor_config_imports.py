"""Every postprocessor's configs import cleanly (views-postprocessing C-83).

**Why an import failure is worse than a wrong value here.** pipeline-core's
`get_queryset()` swallows any exception raised while importing `config_queryset.py`
(`views_pipeline_core/data/model_path.py:783-785` — `except Exception: … self._queryset =
None`), and `declared_data_format(None)` then defaults to `"dataframe"`. So a file that
fails to import is indistinguishable from one that declares the wrong format, and the
symptom is a manager complaining that the queryset says `dataframe` while the file plainly
says `feature_frame`.

At global-land scale that is not cosmetic: the pandas path OOM-kills at ~24 GB on 64,742
cells × ~438 months.

This became a live risk in this repository on 2026-08-11, when ADR-021 gave
`config_queryset.py` an import of `deliveries.status`. The import resolves through a
`sys.path` bootstrap based on `Path(__file__).resolve().parents[3]`, so it does not depend
on the caller's working directory — but nothing asserted that, and the failure mode is
silent by construction.

Imports are exercised **from a different working directory** on purpose: a config that
only resolves when you happen to be at the repository root is the same defect wearing a
disguise.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.green]

REPO_ROOT = Path(__file__).resolve().parents[1]
POSTPROCESSORS = REPO_ROOT / "postprocessors"

CONFIGS = ("config_meta.py", "config_queryset.py", "config_partitions.py")

CONSUMERS = sorted(p.name for p in POSTPROCESSORS.iterdir() if (p / "configs").is_dir())


def test_there_is_at_least_one_postprocessor_to_check():
    """Guard the parametrisation: an empty list would pass every case below."""
    assert CONSUMERS, "no postprocessors discovered — the checks below assert nothing"


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("config", CONFIGS)
def test_config_imports_from_an_unrelated_working_directory(consumer, config, tmp_path):
    """The import must not depend on where the process happens to be standing."""
    path = POSTPROCESSORS / consumer / "configs" / config
    if not path.exists():
        pytest.skip(f"{consumer} declares no {config}")

    # `config_queryset.py` imports datafactory_query at module scope; CI does not install
    # views-datafactory. Skipping truthfully is right — re-deriving the value by parsing
    # the file would rebuild the thing ADR-021 deleted.
    if config == "config_queryset.py":
        pytest.importorskip(
            "datafactory_query",
            reason="views-datafactory not installed; config_queryset cannot be imported",
        )

    source = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('c', {str(path)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ""},
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, (
        f"{consumer}/configs/{config} does not import from {tmp_path}.\n"
        f"  get_queryset() would swallow this and default data_format to 'dataframe' "
        f"(C-83), so the symptom appears far from the cause.\n"
        f"  stderr: {proc.stderr[-600:]}"
    )


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_the_queryset_declares_the_frame_path(consumer):
    """`feature_frame`, not pandas — asserted on the imported value, not the source text."""
    path = POSTPROCESSORS / consumer / "configs" / "config_queryset.py"
    if not path.exists():
        pytest.skip(f"{consumer} declares no config_queryset.py")
    pytest.importorskip("datafactory_query", reason="views-datafactory not installed")

    spec = importlib.util.spec_from_file_location(f"_qs_{consumer}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.generate()["data_format"] == "feature_frame", (
        f"{consumer} does not declare the frame path; the pandas path OOM-kills at "
        f"global-land scale"
    )

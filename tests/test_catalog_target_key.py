"""The catalog generator reads a target key that models actually declare (#336).

`tools/catalogs/update_readme.py` read `configs['targets']` at two call sites. **No
model has ever declared that key** — `targets` was synthesised by views-pipeline-core,
and 3.0.0 retired it outright (pipeline-core #381). So the "Update Model Catalogs"
workflow raised `KeyError: 'targets'` on the first model it reached, and had failed on
**every run since 2026-06-26** — eight consecutive failures across `main` and
`development`.

It went unnoticed because that workflow triggers only on `push` with a path filter, so
it never appears in a pull request's check list. Meanwhile it holds a write token and
commits the regenerated catalogs, so the README catalogs sat five weeks stale while a
job with repo-write access failed unattended.

Measured across all 128 model + ensemble configs on 2026-08-03:

    "regression_targets"    87 configs
    (no target key)         28 configs
    "targets"                0 configs

These are static checks. They do not import `update_readme.py`, because that module is
a script with no `__main__` guard — importing it runs the whole catalog build.
"""

from pathlib import Path
import re

import pytest

pytestmark = pytest.mark.green

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOGS = REPO_ROOT / "tools" / "catalogs"

RETIRED_KEY = "targets"
DECLARED_KEY = "regression_targets"


def test_no_catalog_script_reads_the_retired_targets_key():
    """`configs['targets']` cannot succeed — nothing declares it."""
    offenders = []
    for path in sorted(CATALOGS.glob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"""configs\[\s*['"]targets['"]\s*\]""", line):
                offenders.append(f"  {path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    assert not offenders, (
        f"a catalog script reads configs['{RETIRED_KEY}'], which no model declares and "
        f"which pipeline-core 3.0.0 retired. Read '{DECLARED_KEY}' and tolerate its "
        f"absence — 28 configs have no target key at all:\n" + "\n".join(offenders)
    )


def test_every_config_either_declares_the_key_or_is_tolerated():
    """The generator must survive both shapes present in the tree.

    Not an assertion that every config SHOULD declare a target — 28 legitimately do
    not, and standardising that is separate work (#151). This pins the fact the
    generator has to cope with, so a future 'just read the key' rewrite fails here
    rather than in a workflow nobody watches.
    """
    declaring, silent = [], []
    for kind in ("models", "ensembles"):
        for cfg in sorted((REPO_ROOT / kind).glob("*/configs/config_meta.py")):
            text = cfg.read_text(encoding="utf-8")
            (declaring if f'"{DECLARED_KEY}"' in text else silent).append(cfg.parent.parent.name)
    assert declaring, f"no config declares '{DECLARED_KEY}' — has the schema changed?"
    assert silent, (
        "every config now declares a target key. If that is deliberate, the generator's "
        "fallback is dead code and this test should be replaced by a strict assertion."
    )


def test_the_catalog_workflow_pins_pipeline_core():
    """It commits to the repo, so its inputs must not move underneath it."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "update_catalogs.yml").read_text(encoding="utf-8")
    assert re.search(r'pip install\s+"?views_pipeline_core==', workflow), (
        "update_catalogs.yml installs views_pipeline_core unpinned, and it pushes the "
        "regenerated catalogs with a write token — the committed content could change "
        "because a dependency released, with no commit here to explain it."
    )

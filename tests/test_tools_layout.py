"""`tools/` groups by responsibility — enforced, not merely documented (C-60).

C-60 replaced a flat pile of scripts with `tools/{catalogs,partitions,scaffold}` on
2026-06-07 and was marked **Resolved**. By 2026-07-31 the root had regressed from 2 loose
files to 6, and the register still said Resolved. Nothing had counted.

The rule already existed in prose — `tools/README.md` opens with *"Each subdirectory
handles one responsibility"* — which is precisely the problem: a structural rule stated
in a document decays silently, because documents do not fail. This file is the tripwire
that prose could not be.

It deliberately does **not** check what is inside each subdirectory. Grouping is the rule;
how a group organises itself is that group's business.
"""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.beige]

TOOLS = Path(__file__).resolve().parent.parent / "tools"

# `__init__.py` makes `tools` a package — it is structure, not a tool.
_ALLOWED_AT_ROOT = {"__init__.py", "README.md"}


def test_no_loose_tools_at_the_root_of_tools():
    loose = sorted(
        path.name
        for path in TOOLS.iterdir()
        if path.is_file()
        and path.name not in _ALLOWED_AT_ROOT
        and not path.name.startswith(".")
    )
    assert not loose, (
        f"{len(loose)} file(s) sit loose at tools/ root: {loose}. Each tool belongs in a "
        f"directory named for its responsibility (tools/README.md, line 1). This is how "
        f"C-60 regressed from 2 loose files to 6 while its register entry read "
        f"'Resolved' — if no existing group fits, that is a signal, not a reason to drop "
        f"the file here."
    )


def test_every_tool_group_declares_what_it_is_for():
    """A directory named for a responsibility should say what that responsibility is."""
    undocumented = []
    for group in sorted(p for p in TOOLS.iterdir() if p.is_dir()):
        if group.name.startswith((".", "__")):
            continue
        init = group / "__init__.py"
        readme = group / "README.md"
        has_docstring = init.exists() and init.read_text(encoding="utf-8").strip().startswith('"""')
        if not (has_docstring or readme.exists()):
            undocumented.append(group.name)
    assert not undocumented, (
        f"tool groups with no stated responsibility: {undocumented}. Add a module "
        f"docstring to __init__.py or a README.md — a directory whose purpose must be "
        f"inferred from its filenames is the flat layout again, one level down."
    )

"""The FAO config derives its source from the delivery declaration (#347, ADR-019).

Before this, `postprocessors/un_fao/configs/config_meta.py` carried the line
`"ensemble": "rusty_bucket"` — one hand-typed string deciding which forecast reaches
the UN, inside a file whose own docstring said modifying it *"will not affect the
model"*. That sentence was false for the life of the file.

**The key still exists; it is no longer typed.** `views_postprocessing`'s FAO manager
reads `self.configs["ensemble"]` at `unfao/managers/unfao.py:140` and `:195`, one
repository away. Deleting the key would raise `KeyError` there. So the config now
*derives* the value from `deliveries/un_fao.py` instead of declaring it — the decision
moves, the interface does not. ADR-017's principle applied literally: a thing that is
never typed cannot lie.

The tests that matter here are the failure paths. A config that silently fell back to a
default source would deliver the *wrong forecast to the UN* and say nothing, which is
worse than any error.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import load_config_module

REPO_ROOT = Path(__file__).resolve().parents[1]
FAO_CONFIG = REPO_ROOT / "postprocessors" / "un_fao" / "configs" / "config_meta.py"


def _meta() -> dict:
    module = load_config_module(FAO_CONFIG, module_name="fao_meta_under_test")
    return module.get_meta_config()


def _string_literals_excluding_docs(path: Path) -> set[str]:
    """Every string constant in the file that is not a docstring or a comment.

    Inspecting raw text would flag the module docstring, which *describes* the old
    hand-typed line in order to explain why it is gone. Prose about a name is not a
    declaration of one, and a test that cannot tell the difference teaches people to
    stop explaining themselves.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
    return {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    }


# ── The derivation ─────────────────────────────────────────────────────────


@pytest.mark.beige
class TestConfigDerivesFromTheDeclaration:
    def test_the_key_still_exists_for_views_postprocessing(self):
        """`unfao/managers/unfao.py:195` does `self.configs["ensemble"]`. If this key
        disappears, the FAO delivery raises KeyError in a repo this epic does not
        touch."""
        assert "ensemble" in _meta(), (
            "views_postprocessing reads configs['ensemble']; removing it breaks the "
            "delivery one repo away (unfao/managers/unfao.py:195)."
        )

    def test_it_matches_what_the_delivery_declares(self):
        from deliveries.un_fao import DELIVERY

        declared = [source.name for source in DELIVERY.send]
        assert _meta()["ensemble"] in declared

    def test_the_name_is_not_hand_written_in_the_config(self):
        """The point of the story: the decision lives in deliveries/un_fao.py now.

        A literal source name in this file would mean two places state one fact, with
        nothing reconciling them — ADR-019 §8's rejected alternative.
        """
        from deliveries.un_fao import DELIVERY

        literals = _string_literals_excluding_docs(FAO_CONFIG)
        for source in DELIVERY.send:
            assert source.name not in literals, (
                f"'{source.name}' is still hand-written in "
                f"postprocessors/un_fao/configs/config_meta.py.\n"
                f"  It must be derived from deliveries/un_fao.py, not typed twice."
            )

    def test_changing_the_declaration_changes_the_config(self):
        """The invariant that replaces #343's parity test: the config is downstream
        of the declaration, not merely equal to it today."""
        import deliveries.un_fao as declaration
        from deliveries.vocabulary import Delivery, monthly, pgm, prod

        original = declaration.DELIVERY
        try:
            declaration.DELIVERY = Delivery(
                send=[pgm("skinny_love")],
                frequency=monthly,
                tier=prod,
                intent=original.intent,
            )
            assert _meta()["ensemble"] == "skinny_love"
        finally:
            declaration.DELIVERY = original
        assert _meta()["ensemble"] != "skinny_love"


# ── The failure paths, which are the ones that matter ──────────────────────


@pytest.mark.red
class TestItFailsLoudlyRatherThanFallingBack:
    def test_no_default_source_appears_anywhere_in_the_config(self):
        """ADR-003. A fallback here would deliver the wrong forecast to the UN and
        say nothing — worse than any error this file could raise."""
        text = FAO_CONFIG.read_text(encoding="utf-8")
        code = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("#")
        )
        for pattern in ('get("ensemble",', "get('ensemble',", "or \"rusty", "or 'rusty"):
            assert pattern not in code, (
                f"{pattern!r} looks like a fallback source in {FAO_CONFIG.name}.\n"
                f"  A silent default delivers the wrong forecast; fail loud instead."
            )

    def test_a_missing_declaration_names_the_file(self):
        """Run in a subprocess so the real import machinery is exercised, not a mock."""
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "import importlib.util, pathlib\n"
            "sys.modules['deliveries.un_fao'] = None\n"
            "spec = importlib.util.spec_from_file_location('m', %r)\n"
            "m = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(m)\n"
            "m.get_meta_config()\n"
        ) % (str(REPO_ROOT), str(FAO_CONFIG))
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert proc.returncode != 0, (
            "a broken delivery declaration must fail, not fall back to a default"
        )
        assert "un_fao" in proc.stderr or "deliveries" in proc.stderr, (
            f"the failure names no file, so a reader cannot act on it (ADR-020).\n"
            f"  stderr: {proc.stderr[-400:]}"
        )


# ── The guard from #343 has done its job and must now be retired ───────────


@pytest.mark.beige
class TestTheAdditiveGuardIsCorrectlyRetired:
    def test_the_postprocessor_now_reads_deliveries(self):
        """#343 asserted that nothing under postprocessors/ referenced deliveries/,
        because that story was additive. This story is the one that changes it, so
        the guard is inverted here rather than deleted — the fact stays checked, its
        expected value flips."""
        hits = subprocess.run(
            ["git", "grep", "-l", "deliveries", "--", "postprocessors/"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert hits.returncode in (0, 1), (
            f"git grep could not run (rc={hits.returncode}); this guard proved nothing."
        )
        assert "postprocessors/un_fao/configs/config_meta.py" in hits.stdout, (
            "the FAO config does not reference deliveries/ — the derivation is not "
            "wired up."
        )

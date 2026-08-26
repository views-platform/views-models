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


def _config_in_a_copy(*, edit=None, delete_declaration: bool = False) -> subprocess.CompletedProcess:
    """Load the FAO config in a throwaway repo copy, optionally with a rewritten or
    deleted declaration.

    **Why a copy rather than patching `sys.modules` (#430).** These two tests used to
    monkeypatch `deliveries.un_fao` in the importing process — assigning `DELIVERY`, or
    setting `sys.modules['deliveries.un_fao'] = None`. That worked only because the config
    carried a private accessor doing `from deliveries.un_fao import DELIVERY`. It now calls
    `deliveries.status.declared_source`, which re-executes the declaration **from disk**
    via `spec_from_file_location` and so cannot be reached by either trick.

    The guarantee is unchanged — a missing or altered declaration still decides the config,
    and still fails loudly. What changed is that simulating one by patching an import is no
    longer faithful. A copy is: it exercises the real filesystem and the real import
    machinery, which is what the subprocess in the second test was already reaching for.

    Same shape as `_armed_in_a_copy` in `test_intent_arms_the_delivery.py`, deliberately
    not shared. That one rewrites `intent` and reads `wire_upload_enabled`; this one
    rewrites `send` or removes the file and reads `ensemble`. Two copies that are
    understood beat one parameterised helper that is guessed.
    """
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        shutil.copytree(
            REPO_ROOT, repo, symlinks=True,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc", "models", "ensembles",
                "envs", "wandb", "data", "artifacts", "logs", "reports", "docs",
            ),
        )
        declaration = repo / "deliveries" / "un_fao.py"
        if delete_declaration:
            declaration.unlink()
        elif edit is not None:
            body = declaration.read_text()
            assert edit[0] in body, f"{edit[0]!r} is no longer in deliveries/un_fao.py"
            declaration.write_text(body.replace(edit[0], edit[1], 1))
        return subprocess.run(
            [sys.executable, "-c",
             "import importlib.util,sys;"
             f"sys.path.insert(0,{str(repo)!r});"
             "s=importlib.util.spec_from_file_location('m',"
             f"{str(repo / 'postprocessors/un_fao/configs/config_meta.py')!r});"
             "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
             "print(m.get_meta_config()['ensemble'])"],
            capture_output=True, text=True, cwd=repo,
        )


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
        of the declaration, not merely equal to it today.

        Rewrites `send` in a repo copy rather than reassigning `DELIVERY` on the imported
        module, because the config now reads the declaration from disk (#430) — see
        `_config_in_a_copy`. The copy is the stronger test: it changes the file the
        production path actually reads.
        """
        proc = _config_in_a_copy(
            edit=('send      = [pgm("rusty_bucket")]', 'send      = [pgm("skinny_love")]')
        )
        assert proc.returncode == 0, proc.stderr[-500:]
        assert proc.stdout.strip() == "skinny_love", (
            f"the config did not follow the declaration: {proc.stdout!r}"
        )
        assert _meta()["ensemble"] == "rusty_bucket", (
            "the real repo must be untouched — the edit belonged to the copy"
        )


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
        """The declaration is actually deleted, in a copy, and the config must refuse.

        This used to set `sys.modules['deliveries.un_fao'] = None` in a subprocess, which
        stopped simulating anything once the config began reading the file from disk
        (#430). Deleting the file is what the test was always describing.
        """
        proc = _config_in_a_copy(delete_declaration=True)
        assert proc.returncode != 0, (
            "a missing delivery declaration must fail, not fall back to a default.\n"
            f"  it printed: {proc.stdout!r}"
        )
        assert "un_fao" in proc.stderr and "deliveries" in proc.stderr, (
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

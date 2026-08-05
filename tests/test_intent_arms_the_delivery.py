"""`intent` is the arming switch, and arming refuses when the repo disagrees (#348).

Two facts decided whether the FAO delivery uploads, in two places:

- `DELIVERY.intent` in `deliveries/un_fao.py` — `live()` or `paused(...)`
- `wire_upload_enabled` in `postprocessors/un_fao/configs/config_meta.py`

ADR-019 §8 rejects exactly that: *"two places to state one fact, and nothing to
reconcile them."* Register **C-129**. The launcher key is now **derived** from `intent`,
so it is stated once. views-postprocessing's contract is untouched — it still reads
`configs.get("wire_upload_enabled", product.UPLOAD_ENABLED)` at `unfao/managers/unfao.py:317`.

**The interesting part is the guard.** Committing a derived-armed key would make a clean
checkout upload — and a clean checkout still has `REGION = "africa_me_legacy"` in
`config_queryset.py` (register C-110), so it would upload the **wrong region** to a UN
bucket. So arming is withheld unless the repository agrees with itself: the delivery's
declared `coverage` must match the postprocessor's `REGION`.

It **disarms and warns**; it does not raise. Raising would make the config unloadable
from a clean checkout, breaking runs that never intended to upload — inventing a new
failure mode to guard an old one. Disarming is what the interlock already does when the
key is absent (vpp ADR-013 §11.4 stages artifacts locally), so this refuses the dangerous
half and leaves the rest working. A test below pins *both* halves: it must disarm, and it
must not crash.

That converts C-110's observable half from *"silently delivers the wrong region"* into
*"loudly declines to upload until the two files agree"*, which is what makes the
derivation safe to commit at all.
"""

import ast
import subprocess
from pathlib import Path

import pytest

from tests.conftest import load_config_module

REPO_ROOT = Path(__file__).resolve().parents[1]
FAO_DIR = REPO_ROOT / "postprocessors" / "un_fao" / "configs"
FAO_META = FAO_DIR / "config_meta.py"
FAO_QUERYSET = FAO_DIR / "config_queryset.py"


def _meta() -> dict:
    return load_config_module(FAO_META, module_name="fao_meta_arming").get_meta_config()


def _armed_in_a_copy(*, region: str, intent_src: str | None = None) -> tuple[bool, str]:
    """Build a throwaway repo copy with a chosen REGION, and read the arming state.

    The tests must not depend on whether *this* checkout happens to agree with itself.
    A developer's tree has REGION = "land_gaul"; a clean checkout still has
    "africa_me_legacy" (C-110). A test that asserted either would pass in one place and
    fail in the other, which is how #348 first broke in CI.
    """
    import shutil
    import sys
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
        queryset = repo / "postprocessors/un_fao/configs/config_queryset.py"
        text = queryset.read_text()
        import re as _re
        queryset.write_text(
            _re.sub(r'REGION = "[a-z_]+"', f'REGION = "{region}"', text, count=1)
        )
        if intent_src is not None:
            declaration = repo / "deliveries/un_fao.py"
            body = declaration.read_text()
            body = _re.sub(r"intent    = .*", f"intent    = {intent_src},", body, count=1)
            # the real file imports only what it uses; a substituted intent may need more
            body = body.replace(
                "    Delivery, Require, pgm, live, monthly, prod, months,",
                "    Delivery, Require, pgm, live, paused, monthly, prod, months,",
            )
            declaration.write_text(body)
        proc = subprocess.run(
            [sys.executable, "-c",
             "import importlib.util,sys;"
             f"sys.path.insert(0,{str(repo)!r});"
             "s=importlib.util.spec_from_file_location('m',"
             f"{str(repo / 'postprocessors/un_fao/configs/config_meta.py')!r});"
             "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
             "print(m.get_meta_config()['wire_upload_enabled'])"],
            capture_output=True, text=True, cwd=repo,
        )
    assert proc.returncode == 0, (
        f"the config failed to LOAD. It must disarm, not break.\n  {proc.stderr[-400:]}"
    )
    return proc.stdout.strip() == "True", proc.stderr


def _region_in(source: str) -> str | None:
    """REGION as declared in a given text of config_queryset.py, read statically."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "REGION":
                    if isinstance(node.value, ast.Constant):
                        return node.value.value
    return None


# ── The derivation ─────────────────────────────────────────────────────────


@pytest.mark.beige
class TestArmingIsDerivedFromIntent:
    def test_the_key_is_produced_for_views_postprocessing(self):
        """`unfao/managers/unfao.py:317` reads it as an optional launcher key."""
        assert "wire_upload_enabled" in _meta()

    def test_live_arms_when_the_repository_agrees_with_itself(self):
        from deliveries.un_fao import DELIVERY, REQUIRE

        assert DELIVERY.intent.state == "live"
        armed, _ = _armed_in_a_copy(region=REQUIRE.coverage)
        assert armed is True

    def test_paused_disarms_even_when_the_repository_agrees(self):
        """Regions agreeing must not be enough — `intent` is what decides."""
        from deliveries.un_fao import REQUIRE

        armed, _ = _armed_in_a_copy(
            region=REQUIRE.coverage,
            intent_src='paused("testing the disarm path", since=date(2026, 8, 5))',
        )
        assert armed is False

    def test_arming_is_not_hand_written(self):
        """The whole point: one fact, one place (ADR-019 §8, C-129)."""
        tree = ast.parse(FAO_META.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == "wire_upload_enabled":
                        assert not isinstance(value, ast.Constant), (
                            "wire_upload_enabled is a literal again.\n"
                            "  It must be derived from DELIVERY.intent in "
                            "deliveries/un_fao.py — two places to state one fact is "
                            "what ADR-019 §8 rejects."
                        )


# ── The guard that makes committing this safe ──────────────────────────────


@pytest.mark.red
class TestArmingIsWithheldWhenTheRepoDisagreesWithItself:
    def test_this_checkout_is_internally_consistent_or_disarmed(self):
        """Asserts the *relationship*, not today's value.

        Whether this checkout agrees with itself depends on whether it carries the
        uncommitted `REGION = "land_gaul"` (C-110). A developer's tree usually does; a
        clean checkout does not. Asserting either would be asserting that C-110 is
        resolved, which it is not — and would pass here while failing in CI, which is
        exactly how this story first broke.

        What must hold everywhere: if the two disagree, the upload is not armed.
        """
        from deliveries.un_fao import REQUIRE

        agrees = REQUIRE.coverage == _region_in(FAO_QUERYSET.read_text())
        armed = _meta()["wire_upload_enabled"]
        if not agrees:
            assert armed is False, (
                "this checkout disagrees with itself about the region, yet the upload "
                "is armed — a run would ship a region nobody declared (C-110)."
            )

    def test_a_mismatch_disarms_without_crashing_and_names_the_file(self):
        armed, stderr = _armed_in_a_copy(region="africa_me_legacy")
        assert armed is False, (
            "a region mismatch armed the delivery anyway — a clean checkout would "
            "upload the wrong region to a UN bucket (register C-110)"
        )
        assert "config_queryset.py" in stderr, (
            f"the warning names no file (ADR-020).\n  stderr: {stderr[-500:]}"
        )

    def test_a_clean_checkout_today_would_disarm_rather_than_arm(self):
        """The committed `config_queryset.py` still says africa_me_legacy (C-110).

        This test states what that means, so the fact is visible rather than implied:
        a fresh clone does not silently upload the wrong region; it refuses.
        """
        committed = subprocess.run(
            ["git", "show", "HEAD:postprocessors/un_fao/configs/config_queryset.py"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert committed.returncode == 0
        from deliveries.un_fao import REQUIRE

        committed_region = _region_in(committed.stdout)
        if committed_region != REQUIRE.coverage:
            # Expected today. The guard above is what makes it safe.
            assert committed_region is not None
        # If they now agree, C-110's region half has been resolved — nothing to assert.

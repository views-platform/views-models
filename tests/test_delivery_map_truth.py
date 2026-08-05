"""`docs/forecast_delivery_map.md` must not rot silently (#349, epic #342).

The map is deliberately **not an ADR**: it describes how a forecast physically reaches a
consumer *today*, and it is expected to change — *"a shrinking list, by design."* That
is exactly what makes it dangerous. A page designed to change, that nothing checks,
becomes confidently wrong.

It has already been wrong once. Its predecessor (ADR-017 §1) claimed the live FAO leg
could never see `rusty_bucket`'s output; in fact the legacy leg was retired in #149 and
the contract leg is the only one. The same pass found **C-97 marked Resolved on a claim
that is not true**.

**Assertions are on the underlying fact, never the prose.** A test that matched sentences
would fail on every edit and teach the next person to delete it. These check that a file
exists, a count matches, a key is absent — the things the sentences are *about*.

Claims are split by where they can be answered:

- **offline, in this repo** — `beige`, always run
- **another repository** — `live`, skipped truthfully when it is not importable (ADR-005)
- **the live buckets** — `live`, and covered by `python -m tools.liveness`, not here
"""

import ast
import re
import subprocess
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP = REPO_ROOT / "docs" / "forecast_delivery_map.md"


def _map_text() -> str:
    return MAP.read_text(encoding="utf-8")


# ── What the map is ────────────────────────────────────────────────────────


@pytest.mark.beige
class TestTheMapIsNotAnADR:
    """ADR-000 gained the containment rule when #341 split ADR-017. This pins it."""

    def test_it_exists_where_the_adrs_cite_it(self):
        assert MAP.exists(), (
            "docs/forecast_delivery_map.md is missing, but ADR-017, ADR-019 and "
            "ADR-020 all cite it for today's state."
        )

    def test_it_does_not_live_in_the_adr_directory(self):
        strays = list((REPO_ROOT / "docs" / "ADRs").glob("*forecast_delivery_map*"))
        assert not strays, (
            f"{strays} — the map is not an ADR. ADR-000: decisions are 'never "
            f"deleted… superseded, not erased', and this page is designed to shrink."
        )

    def test_it_says_so_itself(self):
        assert "not an ADR" in _map_text(), (
            "the map must state that it is not an ADR, or a reader will apply ADR-000's "
            "never-delete rule to a page built to shrink."
        )

    def test_it_carries_a_re_traced_date(self):
        match = re.search(r"re-traced against the code and the live buckets:\s*(\d{4}-\d{2}-\d{2})", _map_text())
        assert match, "the map must say when it was last re-traced against reality"
        traced = date.fromisoformat(match.group(1))
        assert traced <= date.today(), f"the map claims a future re-trace date: {traced}"


# ── Offline claims ─────────────────────────────────────────────────────────


@pytest.mark.beige
class TestOfflineClaims:
    def test_the_maturity_distribution_is_what_the_map_says(self):
        """Counts **both quote styles**. A double-quote-only pattern matched 47 of 128
        files and reported the rest as absent, which is how the map carried a wrong
        figure for a week (register C-127). This test exists so that is unrepeatable.
        """
        pattern = re.compile(r"""deployment_status['"]?\s*[:=]\s*['"]([a-z_]+)""")
        counts: dict[str, int] = {}
        files = list((REPO_ROOT / "models").rglob("config_deployment.py")) + \
                list((REPO_ROOT / "ensembles").rglob("config_deployment.py"))
        for path in files:
            for value in pattern.findall(path.read_text(encoding="utf-8")):
                counts[value] = counts.get(value, 0) + 1

        text = _map_text()
        for value, count in counts.items():
            assert f"{count} `{value}`" in text, (
                f"the map does not say there are {count} `{value}` sources "
                f"(found {sorted(counts.items())} across {len(files)} files).\n"
                f"  Open docs/forecast_delivery_map.md and correct the figure."
            )
        assert f"{len(files)} files" in text, (
            f"the map does not say {len(files)} files."
        )

    def test_the_reconciling_ensembles_are_what_the_map_says(self):
        declared = sorted(
            path.parent.parent.name
            for path in (REPO_ROOT / "ensembles").glob("*/configs/config_meta.py")
            if '"reconciliation": "pgm_cm_point"' in path.read_text(encoding="utf-8")
        )
        text = _map_text()
        for name in declared:
            assert name in text, (
                f"'{name}' declares reconciliation but the map does not mention it."
            )

    def test_the_fao_source_is_no_longer_a_buried_literal(self):
        """#347 moved it. The map used to call that line 'the smell, exactly'."""
        config = REPO_ROOT / "postprocessors/un_fao/configs/config_meta.py"
        literals = {
            node.value for node in ast.walk(ast.parse(config.read_text(encoding="utf-8")))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        from deliveries.un_fao import DELIVERY

        for source in DELIVERY.send:
            assert source.name not in literals, (
                f"'{source.name}' is a literal in the FAO config again — the map's "
                f"description of where the delivery is declared would be wrong."
            )

    def test_the_map_does_not_still_call_the_fao_docstring_false(self):
        """#347 rewrote it. A map that keeps reporting a fixed defect trains readers
        to distrust it, which is worse than saying nothing."""
        text = _map_text()
        stale = "modifying it will not affect the model" in text and "**false**" in text
        assert not stale, (
            "the map still describes the FAO config docstring as false. #347 rewrote "
            "it.\n  Open docs/forecast_delivery_map.md and update that entry."
        )

    def test_c110_residual_is_described_accurately(self):
        """C-110 has narrowed. `wire_upload_enabled` is committed and derived since
        #348; only `wire_contract` and `region` remain working-tree only."""
        committed = subprocess.run(
            ["git", "show", "HEAD:postprocessors/un_fao/configs/config_meta.py"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert committed.returncode == 0
        if "wire_upload_enabled" in committed.stdout:
            text = _map_text()
            assert "absent from git" not in text or "wire_upload_enabled" not in text.split("absent from git")[0][-400:], (
                "the map still says wire_upload_enabled is absent from git. Since #348 "
                "it is committed and derived from DELIVERY.intent.\n"
                "  Open docs/forecast_delivery_map.md and narrow C-110's residual to "
                "wire_contract and region."
            )


# ── Claims answered in another repository ──────────────────────────────────


def _views_postprocessing_dir() -> Path:
    """Where views-postprocessing's source is, or a truthful skip.

    `views_postprocessing` may import as a **namespace package** — `__file__` is None
    while `__path__` points at a sibling source checkout on `sys.path`. That is not an
    installed package, and `pytest.importorskip` alone would let a test claim it
    verified an installation it never saw (the C-75 failure mode).

    So: resolve through `__path__` when `__file__` is absent, and say which was used.
    """
    module = pytest.importorskip(
        "views_postprocessing",
        reason="views_postprocessing not importable — cannot verify the map's "
               "cross-repo claims (truthful skip, C-75)",
    )
    if module.__file__:
        return Path(module.__file__).parent
    paths = list(getattr(module, "__path__", []))
    if not paths:
        pytest.skip(
            "views_postprocessing imports as an empty namespace package — there is no "
            "source to read, so nothing here was verified (truthful skip, C-75)"
        )
    return Path(paths[0])


@pytest.mark.live
class TestCrossRepoClaims:
    """Skipped truthfully when views-postprocessing is not importable (ADR-005, C-75).

    A silent pass here would be worse than no test: it would report that the wire
    contract was verified when nothing was read.
    """

    def test_the_legacy_pandas_reader_is_gone(self):
        source = _views_postprocessing_dir() / "unfao" / "managers" / "unfao.py"
        if not source.exists():
            pytest.skip(f"{source} not present in the installed package")
        text = source.read_text(encoding="utf-8")
        assert "LEGACY_FORECAST_FILTERS" not in text, (
            "LEGACY_FORECAST_FILTERS is back in views-postprocessing's FAO manager. "
            "The map says the legacy pandas reader was retired in #149."
        )

    def test_the_upload_is_interlocked_off_by_default(self):
        product = _views_postprocessing_dir() / "unfao" / "product.py"
        if not product.exists():
            pytest.skip(f"{product} not present in the installed package")
        text = product.read_text(encoding="utf-8")
        # Allow an annotation: the real line is `UPLOAD_ENABLED: bool = False`.
        # A pattern that demanded a bare `=` reported the constant as missing — the
        # same defect class as C-127, in the test written to prevent C-127.
        assert re.search(r"UPLOAD_ENABLED\s*(?::\s*\w+\s*)?=\s*False", text), (
            "vpp ADR-013 §11.4's interlock is no longer off by default. The map, and "
            "this repo's derived arming state (#348), both assume it is."
        )

"""Guards for the two checks added after the 2026-09-30 rehearsal.

That delivery passed EVERY structural check — valid manifest, right coverage, findability guard
green — while serving a posterior whose point estimate was zero for all 2,333,448 cells. It was
found by hand at 4am, by someone computing anchors for an unrelated probe.

These tests run the real tools against synthetic posteriors built to be degenerate, sparse and
healthy. They do not read source text: the previous round of guards in this repository went
22/22 green with every fix reverted, because they asserted on strings a comment could satisfy.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
pytest.importorskip("views_frames")
pytest.importorskip("views_frames_summarize")


def _make_pooled(tmp_path: Path, draws_by_target: dict) -> Path:
    """A pooled ensemble output in the layout the ensemble actually writes.

    y_pred.npy + identifiers.npz — NOT PredictionFrame.save's values.npy. The tools must read
    what is on disk, and this fixture is what proves they do.
    """
    root = tmp_path / "predictions_forecasting_20260930_024838"
    for target, draws in draws_by_target.items():
        d = root / target
        d.mkdir(parents=True)
        n = draws.shape[0]
        np.save(d / "y_pred.npy", draws)
        np.savez(
            d / "identifiers.npz",
            time=np.repeat(np.arange(561, 561 + 4), n // 4).astype(np.int64)[:n],
            unit=np.tile(np.arange(1000, 1000 + n // 4), 4).astype(np.int64)[:n],
        )
    return root


def _run(module: str, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *[str(a) for a in args]],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )


ALL_ZERO = np.zeros((40, 8), dtype=np.float32)


def _sparse_with_spike():
    """The 2026-09-30 shape: almost everything zero, a few enormous draws in one cell."""
    a = np.zeros((40, 8), dtype=np.float32)
    a[7, :3] = [301274.9, 15000.0, 8000.0]
    return a


def _healthy():
    rng = np.random.default_rng(1)
    return rng.gamma(2.0, 2.0, size=(40, 8)).astype(np.float32)


class TestPosteriorHealthDetectsWhatEveryOtherCheckMissed:
    def test_an_all_zero_posterior_is_DEGENERATE(self, tmp_path):
        pooled = _make_pooled(tmp_path, {"lr_sb_best": ALL_ZERO})
        r = _run("tools.prereg.posterior_health", pooled, "--mode", "rehearsal")
        assert "DEGENERATE" in r.stdout, (
            "a posterior of all zeros must be called degenerate. Every structural check passes "
            f"on it, which is exactly why this one has to fire.\n{r.stdout}\n{r.stderr}"
        )

    def test_a_degenerate_production_run_exits_non_zero(self, tmp_path):
        pooled = _make_pooled(tmp_path, {"lr_sb_best": ALL_ZERO})
        assert _run("tools.prereg.posterior_health", pooled, "--mode", "production").returncode != 0

    def test_a_degenerate_rehearsal_exits_zero(self, tmp_path):
        """A rehearsal at low lesson count is EXPECTED to look like this. Failing the run it was
        designed to produce would teach the operator to ignore the exit code, which would cost
        more than it saves."""
        pooled = _make_pooled(tmp_path, {"lr_sb_best": ALL_ZERO})
        r = _run("tools.prereg.posterior_health", pooled, "--mode", "rehearsal")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "EXPECTED for a rehearsal" in r.stdout

    def test_the_two_modes_say_different_things_about_identical_numbers(self, tmp_path):
        """The numbers cannot distinguish a rehearsal from a broken production run. The runner
        knows which it is, and that is the only reason this can be reported usefully."""
        pooled = _make_pooled(tmp_path, {"lr_sb_best": ALL_ZERO})
        reh = _run("tools.prereg.posterior_health", pooled, "--mode", "rehearsal").stdout
        prod = _run("tools.prereg.posterior_health", pooled, "--mode", "production").stdout
        assert reh != prod
        assert "investigate the" in prod and "investigate the" not in reh

    def test_a_healthy_posterior_is_not_flagged(self, tmp_path):
        """The control. Without it, a tool that printed DEGENERATE unconditionally would pass
        every test above."""
        pooled = _make_pooled(tmp_path, {"lr_sb_best": _healthy()})
        r = _run("tools.prereg.posterior_health", pooled, "--mode", "production")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "DEGENERATE" not in r.stdout, r.stdout

    def test_the_json_report_carries_the_numbers_not_only_a_verdict(self, tmp_path):
        pooled = _make_pooled(tmp_path, {"lr_sb_best": _sparse_with_spike()})
        out = tmp_path / "health.json"
        _run("tools.prereg.posterior_health", pooled, "--mode", "rehearsal", "--json-out", out)
        rec = json.loads(out.read_text())
        s = rec["targets"][0]
        for k in ("rows", "samples", "draws_nonzero_fraction", "rows_all_zero_fraction",
                  "max_draw", "point_nonzero"):
            assert k in s, f"{k} missing — a verdict without its evidence cannot be argued with"
        assert s["max_draw"] == pytest.approx(301274.9, rel=1e-4)


class TestAnchorsAreRecordedAndDiscriminating:
    def test_it_records_the_raw_draws_not_only_the_point_estimate(self, tmp_path):
        """The reason this matters: on 2026-09-30 every point estimate was 0.0, so comparing
        point estimates would have passed against an unrelated all-zero dataset. The draws are
        what discriminate."""
        pooled = _make_pooled(tmp_path, {"lr_sb_best": _sparse_with_spike()})
        out = tmp_path / "anchors.json"
        r = _run("tools.prereg.capture_anchors", pooled, "--mode", "rehearsal", "--out", out)
        assert r.returncode == 0, r.stdout + r.stderr
        rec = json.loads(out.read_text())
        cells = rec["anchors"]["lr_sb_best"]["cells"]
        assert cells, "no anchor cells recorded"
        assert all("draws" in c and len(c["draws"]) == 8 for c in cells), (
            "every anchor must carry its full draw vector"
        )
        assert any(float(v) > 0 for c in cells for v in c["draws"]), (
            "the spike cell must be among the anchors — the top-N selection is by point estimate "
            "and must surface the cells that carry signal"
        )

    def test_the_selection_is_reproducible(self, tmp_path):
        """Seeded, so the choice of cells provably was not steered by what the API returned."""
        pooled = _make_pooled(tmp_path, {"lr_sb_best": _healthy()})
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        _run("tools.prereg.capture_anchors", pooled, "--mode", "rehearsal", "--out", a)
        _run("tools.prereg.capture_anchors", pooled, "--mode", "rehearsal", "--out", b)
        ka = [(c["month_id"], c["priogrid_id"]) for c in json.loads(a.read_text())["anchors"]["lr_sb_best"]["cells"]]
        kb = [(c["month_id"], c["priogrid_id"]) for c in json.loads(b.read_text())["anchors"]["lr_sb_best"]["cells"]]
        assert ka == kb and len(ka) > 0

    def test_it_records_the_estimator_version(self, tmp_path):
        """P8 claimed to remove 'compared the wrong thing' BY CONSTRUCTION because both sides run
        the same version. They float independently inside `views-frames>=1.10.2,<2`, and the
        SERVING version is not observable from outside. So it is written down."""
        pooled = _make_pooled(tmp_path, {"lr_sb_best": _healthy()})
        out = tmp_path / "anchors.json"
        _run("tools.prereg.capture_anchors", pooled, "--mode", "rehearsal", "--out", out)
        rec = json.loads(out.read_text())
        assert rec.get("views_frames_version") not in (None, "", "unknown"), (
            "the estimator version must be recorded, or a later mismatch is mysterious "
            "rather than diagnosable"
        )
        assert "tower_point" in rec["estimator"]

    def test_full_precision_is_preserved(self, tmp_path):
        """repr(), not round(): the comparison is made across a boundary where rounding is the
        difference between a real disagreement and a spurious one."""
        pooled = _make_pooled(tmp_path, {"lr_sb_best": _sparse_with_spike()})
        out = tmp_path / "anchors.json"
        _run("tools.prereg.capture_anchors", pooled, "--mode", "rehearsal", "--out", out)
        rec = json.loads(out.read_text())
        vals = [v for c in rec["anchors"]["lr_sb_best"]["cells"] for v in c["draws"]]
        assert all(isinstance(v, str) for v in vals), "values must be repr strings, not floats"
        assert any("." in v for v in vals)

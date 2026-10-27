"""The launch documented for the chrome pair must work on a box that has never run them.

#639. The command recorded on #598 was:

    python -m ensembles.chrome_crate.main --run_type forecasting --saved
    python -m ensembles.chrome_box.main   --run_type forecasting --saved

and it does not start. `ForecastingModelArgs._validate` refuses it with *"Run type is
forecasting but neither --train, --evaluate, nor --sweep flag is set. Nothing to do..."*, so it
never reaches a member — the audit on #639 predicted a `FileNotFoundError` from the first
member's `resolve_artifact_path`, which is one layer further in than the run actually gets. Both
readings agree on the cause: there is no `-t`, and none of the sixteen members has an artifact.

⛔ These guards read the DOCUMENTATION and run it through the REAL parser. That is deliberate.
The audit's stub asserted that every member has a forecasting artifact, which can only pass on a
machine that has already done the run — it would be red in CI forever and green only by accident
of local state. The defect was never "this box lacks artifacts"; it was "the written command
cannot run". So the subject here is the written command, and the oracle is
`views_pipeline_core.cli.ForecastingModelArgs`, which is what rejected it. The stub's either/or
survives intact in `test_documented_launch_does_not_presume_member_artifacts`: a launch without
`-t` is allowed only where the artifacts it presumes actually exist.
"""

import io
import re
import shlex
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from views_pipeline_core.cli import ForecastingModelArgs

from tests.conftest import ENSEMBLES_DIR, MODELS_DIR

#: The pair #639 is about. Named, not globbed: a glob that stopped matching would report fewer
#: passes instead of failing, and a cardinality may never stand in for an identity (C-182).
CHROME_ENSEMBLES = ("chrome_box", "chrome_crate")

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_LAUNCH_RE = re.compile(r"^\s*python\s+-m\s+ensembles\.(\w+)\.main\b(.*)$")


def _fenced_launches(readme: Path):
    """Return `(ensemble, argv)` for every `python -m ensembles.X.main` line in a code fence.

    Only fenced blocks count. Prose may quote a broken command — this file's own module
    docstring does, and both chrome READMEs quote the refused one to say why it is refused —
    and a guard that read prose would forbid documenting the defect it exists to prevent.
    """
    found = []
    for fence in _FENCE_RE.findall(readme.read_text()):
        for line in fence.splitlines():
            match = _LAUNCH_RE.match(line)
            if match:
                # `comments=True` drops a trailing `# FIRST, same machine` annotation.
                found.append((match.group(1), shlex.split(match.group(2), comments=True)))
    return found


def _all_documented_launches():
    """Every documented launch across both chrome READMEs, as `(readme_name, ensemble, argv)`.

    A launch for `chrome_crate` is documented in BOTH READMEs, because `chrome_box` cannot run
    without it. Each occurrence is checked on its own: a correct command in one file does not
    excuse a stale copy in the other.
    """
    launches = []
    for name in CHROME_ENSEMBLES:
        readme = ENSEMBLES_DIR / name / "README.md"
        for ensemble, argv in _fenced_launches(readme):
            launches.append((name, ensemble, argv))
    return launches


_LAUNCHES = _all_documented_launches()
_IDS = [f"{readme}:{ensemble}:{' '.join(argv)}" for readme, ensemble, argv in _LAUNCHES]


def _parse(argv):
    """Run `argv` through the real CLI. Returns the args, or the refusal message."""
    saved_argv = sys.argv
    buffer = io.StringIO()
    try:
        sys.argv = ["main.py", *argv]
        with redirect_stdout(buffer), redirect_stderr(buffer):
            return ForecastingModelArgs.parse_args(), None
    except SystemExit:
        lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
        return None, " ".join(lines[:1]) or "refused with no message"
    finally:
        sys.argv = saved_argv


def test_both_chrome_ensembles_document_a_launch():
    """Neither README may go quiet. Absence is not a pass.

    Every other guard here is parametrized over what was found, so a README that stops
    documenting its launch would silently reduce the number of checks that run rather than fail
    one (C-182, ADR-025). This is the census that makes the rest of the file mean something.
    """
    documented = {ensemble for _, ensemble, _ in _LAUNCHES}
    assert documented == set(CHROME_ENSEMBLES), (
        "the chrome READMEs no longer document a launch for each ensemble in a code fence; "
        f"found {sorted(documented)}, expected {sorted(CHROME_ENSEMBLES)}. #639 exists because "
        "the recorded command did not run, and an undocumented command cannot be checked at all."
    )


@pytest.mark.parametrize("readme,ensemble,argv", _LAUNCHES, ids=_IDS)
def test_the_documented_launch_is_accepted_by_the_cli(readme, ensemble, argv):
    """The documented command must survive `ForecastingModelArgs._validate`.

    This is the guard that fails on #639's actual defect. `--run_type forecasting --saved` is
    refused by the rule *"neither --train, --evaluate, nor --sweep flag is set"*, which is why
    nobody would have reached a member on the rented box.
    """
    args, refusal = _parse(argv)
    assert args is not None, (
        f"ensembles/{readme}/README.md documents a launch the CLI refuses:\n"
        f"  python -m ensembles.{ensemble}.main {' '.join(argv)}\n"
        f"  {refusal}"
    )
    assert args.run_type == "forecasting", (
        f"ensembles/{readme}/README.md documents a {args.run_type} launch for {ensemble}; "
        "the chrome pair exists to produce a forecast (#595)."
    )
    assert args.forecast, (
        f"ensembles/{readme}/README.md documents a launch for {ensemble} without `-f`. "
        "`_execute_model_tasks` forecasts only under `if ctx.args.forecast:`, so this command "
        "would train and then produce nothing."
    )


@pytest.mark.parametrize("readme,ensemble,argv", _LAUNCHES, ids=_IDS)
def test_documented_launch_does_not_presume_member_artifacts(readme, ensemble, argv):
    """The stub's either/or, from the #639 audit: train, or have the artifacts already.

    Without `-t`, every member resolves `get_latest_model_artifact_path("forecasting")`, which
    raises `FileNotFoundError` when the directory holds no artifact. So a documented launch
    without `-t` is a claim about the filesystem, and this asserts that claim is true.
    """
    args, _ = _parse(argv)
    if args is not None and args.train:
        return

    members = _members(ensemble)
    assert members, f"could not read the modelset of {ensemble}"
    missing = [
        member
        for member in members
        if not any((MODELS_DIR / member / "artifacts").glob("*forecasting*"))
    ]
    assert not missing, (
        f"ensembles/{readme}/README.md documents a launch for {ensemble} without `-t`, which "
        f"presumes a forecasting artifact for every member. These have none: {missing}. "
        "Either train in the documented command (`-t`) or do not document a launch that "
        "depends on artifacts this repository does not carry."
    )


@pytest.mark.parametrize("readme,ensemble,argv", _LAUNCHES, ids=_IDS)
def test_the_documented_launch_does_not_pass_saved(readme, ensemble, argv):
    """`-t --saved` parses cleanly and still dies on a fresh box, so the string matters.

    `_create_model_args` computes `saved = ctx.args.saved if train else True`. The member
    FORECAST step therefore always runs saved, reusing what training cached — but the member
    TRAINING step inherits `--saved` from here, and on a fresh box there is no cached data for
    it to read. The parser cannot catch this, which is why it is asserted separately.
    """
    args, _ = _parse(argv)
    assert args is None or not args.saved, (
        f"ensembles/{readme}/README.md documents `--saved` for {ensemble}. The member training "
        "step then looks for locally stored data, which a freshly rented box does not have "
        "(#639). Drop it: the member forecast step is saved regardless of this flag."
    )


@pytest.mark.parametrize("readme,ensemble,argv", _LAUNCHES, ids=_IDS)
def test_the_documented_launch_does_not_publish(readme, ensemble, argv):
    """No `-p`. The chrome pair produces; serving is gated elsewhere.

    Rule 5 of `docs/runpod_run_guide.md` keeps publish credentials off rented hardware, and the
    real delivery is gated on views-crafdapi#180 and #599. A documented `-p` would put the
    Appwrite secrets on the box as a side effect of following the README.
    """
    args, _ = _parse(argv)
    assert args is None or not args.prediction_store, (
        f"ensembles/{readme}/README.md documents `-p` for {ensemble}, which would need publish "
        "credentials on the box. The chrome run produces a forecast; the delivery is gated on "
        "views-crafdapi#180 and #599."
    )


def test_chrome_box_documents_the_cm_run_first():
    """`chrome_box` cannot run before `chrome_crate`, on the same filesystem.

    `load_cm_frame` resolves `EnsemblePathManager("chrome_crate")
    ._get_generated_predictions_data_file_paths("forecasting")` and fails loud when it finds
    nothing. The order is a precondition of the run, so the README that a person follows
    top-to-bottom must show the cm launch above the pgm one.
    """
    launches = [ensemble for ensemble, _ in _fenced_launches(ENSEMBLES_DIR / "chrome_box" / "README.md")]
    assert "chrome_crate" in launches, (
        "ensembles/chrome_box/README.md does not document the chrome_crate run. chrome_box reads "
        "chrome_crate's forecast from local disk (load_cm_frame), so running it alone fails with "
        "'no local forecast found for country model chrome_crate'."
    )
    assert launches.index("chrome_crate") < launches.index("chrome_box"), (
        "ensembles/chrome_box/README.md documents its own launch before chrome_crate's. The cm "
        f"run is a precondition of the pgm run, not a parallel option. Order found: {launches}"
    )


def _members(ensemble: str) -> list[str]:
    """The ensemble's modelset, read from its own config rather than hard-coded here.

    `tests/test_chrome_box_pairing.py` owns the assertion about WHICH members these are and in
    what order. This file only needs the names, so it reads them and does not restate them — a
    second copy of the roster would drift and would not be the copy anyone checks.
    """
    text = (ENSEMBLES_DIR / ensemble / "configs" / "config_modelset.py").read_text()
    block = text[text.index('"models"') :]
    return re.findall(r'"([a-z]+_[a-z]+)"', block[: block.index("]")])

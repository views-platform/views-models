"""The launch documented for the chrome pair must work on a box that has never run them.

#639. The command recorded on #598 was:

    python -m ensembles.chrome_crate.main --run_type forecasting --saved
    python -m ensembles.chrome_box.main   --run_type forecasting --saved

and it does not start. `ForecastingModelArgs._validate` refuses it with *"Run type is
forecasting but neither --train, --evaluate, nor --sweep flag is set. Nothing to do..."*, so it
never reaches a member — the audit on #639 predicted a `FileNotFoundError` from the first
member's `resolve_artifact_path`, which is one layer further in than the run actually gets. Both
readings agree on the cause: there is no `-t`, and none of the sixteen members has an artifact.

The subject of these guards is the WRITTEN COMMAND, and the oracle is
`views_pipeline_core.cli.ForecastingModelArgs`, which is the thing that rejected it. The audit's
own stub asserted instead that every member has a forecasting artifact, which can only pass on a
machine that has already done the run — red in CI forever and green only by accident of local
state. The defect was never "this box lacks artifacts"; it was "the written command cannot run".

⛔ This file was rewritten after an independent mutation audit of its first version, in which
**12 of 20 mutations survived**. Three habits produced all twelve, and each is now closed by
construction rather than by another value assertion. They are written down because the first
version read as though it covered them:

1. **A union stood in for a conjunction.** The census pooled both READMEs into one set of names,
   so `chrome_crate/README.md` could document nothing at all — or document `chrome_box`'s launch
   instead of its own — and stay green on the strength of the other file. That is C-182's
   mechanism one level up: a cardinality over the union standing in for an identity per file.
   Everything here is now asserted **per README**, and each must document **its own** ensemble.
2. **A notation stood in for the command a person copies.** The guard recognised exactly
   `python -m ensembles.X.main` inside a backtick fence. A `$ ` prompt, a `~~~` fence, a
   four-space indented block, or the `python main.py` form the generated "Usage" section already
   uses two paragraphs above — each left the census silently. A command this guard cannot read is
   now a **failure, not an omission** (ADR-025: a check that cannot see its subject must refuse).
3. **A position in the file stood in for a position in the instructions.** The ordering guard
   searched the whole README, so a mention inside the ASCII directory tree satisfied "cm first"
   while the runnable block said the opposite. Scope is now the `<!-- manual -->` block only —
   which is also the only content `update_catalogs.yml` preserves, so a launch documented outside
   it would be deleted by the next regeneration.
"""

import io
import re
import shlex
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import pytest

from views_pipeline_core.cli import ForecastingModelArgs

from tests.conftest import ENSEMBLES_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.catalogs.readme_preserve import extract_manual_blocks  # noqa: E402

#: The pair #639 is about. Named, not globbed: a glob that stopped matching would report fewer
#: passes instead of failing, and a cardinality may never stand in for an identity (C-182).
CHROME_ENSEMBLES = ("chrome_box", "chrome_crate")

#: Exactly the flags a fresh-box forecasting launch may carry. A WHITELIST, not a blacklist of
#: `--saved` and `-p`: the first version enumerated the two flags it knew were hostile, and
#: `-wn -re -dd -o 3` walked straight past it — `-wn` wants a W&B credential, `-dd` runs a drift
#: self-test at fetch, `-o` overrides the timestep. A documented fresh-box launch is a short,
#: fully specified command, so anything outside this set has to be argued for in a review.
FRESH_BOX_TOKENS = frozenset(
    {"-r", "--run_type", "forecasting", "-t", "--train", "-f", "--forecast"}
)

_BACKTICK_FENCE_RE = re.compile(r"^[ \t]*```[^\n]*\n(.*?)^[ \t]*```", re.DOTALL | re.MULTILINE)
_TILDE_FENCE_RE = re.compile(r"^[ \t]*~~~", re.MULTILINE)
_PROMPT_RE = re.compile(r"^[ \t]*(?:[$>]\s+)?")

#: Every way this repository's own documentation invokes a model or ensemble. The point is not to
#: bless them all — it is that a command written in any of them is SEEN. `python main.py` and
#: `./run.sh` are the two forms the generated "Usage" section uses, so they are house style, not
#: an exotic evasion.
_COMMAND_RE = re.compile(
    r"""^
    (?:cd\s+ensembles/(?P<cd>\w+)\s*&&\s*)?
    (?:
        python3?\s+-m\s+ensembles\.(?P<module>\w+)\.main
      | python3?\s+ensembles/(?P<script>\w+)/main\.py
      | python3?\s+main\.py
      | (?:\./|bash\s+|sh\s+)run\.sh
    )
    (?P<argv>\s.*|$)
    """,
    re.VERBOSE,
)

#: A line that is trying to be a command. Used only to decide whether an unparseable line is a
#: silent escape (which must fail) or ordinary prose inside a fence (which must not).
_LOOKS_LIKE_A_COMMAND_RE = re.compile(r"\b(?:python3?|run\.sh|main\.py|ensembles\.)\b")
# `WANDB_MODE=offline`, however it is exported: bare, `export`-prefixed, or beside other
# assignments on the same line. Quoted forms count too — `WANDB_MODE="offline"` is equivalent.
_WANDB_OFFLINE_RE = re.compile(r"""\bWANDB_MODE\s*=\s*["']?offline["']?(?!\w)""")


@dataclass(frozen=True)
class Launch:
    """One documented command, and which README documented it."""

    readme: str
    ensemble: str
    line: str
    argv: tuple[str, ...]

    def __str__(self) -> str:
        return f"ensembles/{self.readme}/README.md: {self.line.strip()}"


class UnreadableDocumentation(Exception):
    """A documented command this guard cannot read. Never swallowed — see ADR-025."""


def _manual_block(ensemble: str) -> str:
    """The `<!-- manual -->` content of `ensemble`'s README.

    Scope is this block and nothing else, for two reasons that happen to coincide. It is the only
    content `tools/catalogs/readme_preserve.py` carries across a regeneration, so a launch
    documented outside it is deleted the next time `update_catalogs.yml` runs; and it is the
    block a person reads as instructions, which is what the ordering guard needs to be about.
    """
    readme = ENSEMBLES_DIR / ensemble / "README.md"
    blocks = extract_manual_blocks(readme.read_text())
    return "\n".join(blocks)


def _launches(ensemble: str) -> list[Launch]:
    """Every command documented in `ensemble`'s manual block.

    Raises:
        UnreadableDocumentation: a code notation or a command line this guard cannot read. It
            refuses rather than skipping: every survivor of the first audit that hid a command
            did so by leaving this function's predecessor with nothing to return.
    """
    block = _manual_block(ensemble)
    if not block:
        raise UnreadableDocumentation(
            f"ensembles/{ensemble}/README.md has no <!-- manual --> block, so any launch it "
            "documents would be deleted by the next update_catalogs.yml regeneration (C-78)."
        )

    if _TILDE_FENCE_RE.search(block):
        raise UnreadableDocumentation(
            f"ensembles/{ensemble}/README.md's manual block uses a ~~~ fence. GitHub renders it "
            "as code and this guard does not read it, so a command inside it would be "
            "documented and unchecked. Use a ``` fence."
        )

    fences = _BACKTICK_FENCE_RE.findall(block)
    if not fences:
        raise UnreadableDocumentation(
            f"ensembles/{ensemble}/README.md's manual block documents no command in a ``` "
            "fence. #639 exists because the recorded command did not run; an undocumented "
            "command cannot be checked at all."
        )

    outside = _BACKTICK_FENCE_RE.sub("", block)
    indented = [
        line
        for line in outside.splitlines()
        if re.match(r"^(?: {4}|\t)\s*\S", line) and _LOOKS_LIKE_A_COMMAND_RE.search(line)
    ]
    if indented:
        raise UnreadableDocumentation(
            f"ensembles/{ensemble}/README.md's manual block has an indented code block that "
            "looks like a command. GitHub renders it as code and this guard does not read it:\n"
            + "\n".join(f"  {line}" for line in indented)
        )

    launches = []
    for fence in fences:
        for raw in fence.splitlines():
            line = _PROMPT_RE.sub("", raw, count=1)
            if not line.strip():
                continue
            match = _COMMAND_RE.match(line)
            if not match:
                if _LOOKS_LIKE_A_COMMAND_RE.search(line):
                    raise UnreadableDocumentation(
                        f"ensembles/{ensemble}/README.md documents a command this guard cannot "
                        f"parse, so none of its flags are checked:\n  {raw.strip()}\n"
                        "Write it as `python -m ensembles.<name>.main <flags>`."
                    )
                continue
            try:
                argv = shlex.split(match.group("argv"), comments=True)
            except ValueError as error:
                # A `\` line-continuation makes shlex raise. Caught HERE so the failure names
                # the README; the first version let it escape and take the whole file down at
                # import with a message that named no file.
                raise UnreadableDocumentation(
                    f"ensembles/{ensemble}/README.md documents a command that does not lex "
                    f"({error}), so its flags are unchecked:\n  {raw.strip()}"
                ) from error
            target = match.group("module") or match.group("script") or match.group("cd")
            launches.append(
                Launch(
                    readme=ensemble,
                    #: No explicit name means the README's own ensemble — `python main.py` and
                    #: `./run.sh` are run from the ensemble's own directory.
                    ensemble=target or ensemble,
                    line=raw,
                    argv=tuple(argv),
                )
            )
    return launches


def _parse(argv) -> tuple[object, str | None]:
    """Run `argv` through the real CLI. Returns `(args, None)` or `(None, refusal)`."""
    saved_argv = sys.argv
    buffer = io.StringIO()
    try:
        sys.argv = ["main.py", *argv]
        with redirect_stdout(buffer), redirect_stderr(buffer):
            return ForecastingModelArgs.parse_args(), None
    except SystemExit:
        lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
        return None, (lines[0] if lines else "refused with no message")
    finally:
        sys.argv = saved_argv


@pytest.mark.parametrize("ensemble", CHROME_ENSEMBLES)
def test_the_documented_launch_is_written_in_a_notation_this_guard_can_read(ensemble):
    """Refuse rather than skip. This is the guard that makes every other one mean something.

    Of the twelve mutations that survived the first audit, five hid a command in plain sight: a
    `$ ` prompt, a `~~~` fence, a four-space indented block, the `python main.py` form, or a
    module that does not exist. Each left the old census with nothing to iterate, and a census
    with nothing to iterate reports fewer passes rather than failing (C-182, ADR-025).
    """
    try:
        launches = _launches(ensemble)
    except UnreadableDocumentation as error:
        pytest.fail(str(error))

    assert launches, (
        f"ensembles/{ensemble}/README.md's manual block has a ``` fence but no command in it."
    )

    missing_entry_points = [
        str(launch)
        for launch in launches
        if not (ENSEMBLES_DIR / launch.ensemble / "main.py").is_file()
    ]
    assert not missing_entry_points, (
        "a documented launch names an entry point that does not exist, so following the README "
        "fails with `No module named ...` before any flag matters:\n"
        + "\n".join(f"  {m}" for m in missing_entry_points)
    )


@pytest.mark.parametrize("ensemble", CHROME_ENSEMBLES)
def test_each_readme_documents_its_own_ensembles_launch(ensemble):
    """Per README, not per union.

    The first version compared a set of names pooled across both files, so `chrome_crate`'s
    README could document no launch at all, or document `chrome_box`'s instead of its own, and
    stay green because the other file named both. An operator reads one file.
    """
    documented = {launch.ensemble for launch in _launches(ensemble)}
    assert ensemble in documented, (
        f"ensembles/{ensemble}/README.md documents launches for {sorted(documented)} but not "
        f"for {ensemble} itself. Someone following this file would never run this ensemble."
    )


@pytest.mark.parametrize("ensemble", CHROME_ENSEMBLES)
def test_the_documented_launch_is_accepted_by_the_cli(ensemble):
    """Every documented command must survive `ForecastingModelArgs._validate`.

    This is the guard that fails on #639's actual defect: `--run_type forecasting --saved` is
    refused by the rule *"neither --train, --evaluate, nor --sweep flag is set"*, which is why
    nobody would have reached a member on the rented box.
    """
    offenders = []
    for launch in _launches(ensemble):
        args, refusal = _parse(launch.argv)
        if args is None:
            offenders.append(f"  {launch}\n      {refusal}")
        elif args.run_type != "forecasting":
            offenders.append(f"  {launch}\n      documents a {args.run_type} run, not a forecast")
        elif not args.forecast:
            offenders.append(
                f"  {launch}\n      has no `-f`, so `_execute_model_tasks` would train and then "
                "produce nothing"
            )
    assert not offenders, (
        f"ensembles/{ensemble}/README.md documents a launch that would not forecast:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("ensemble", CHROME_ENSEMBLES)
def test_documented_launch_does_not_presume_member_artifacts(ensemble):
    """#639's acceptance item: the documented launch must train.

    ⛔ The audit's stub asserted this by globbing `models/*/artifacts` for every member, and that
    form is abandoned here rather than kept alongside. Two measurements killed it. It cannot fail
    alone: the real parser refuses any non-`--train` run that lacks `--saved`
    (`if (not self.train and not self.sweep) and not self.saved`), so every command that reaches
    its branch is already red in `test_the_documented_launch_is_accepted_by_the_cli` or in
    `test_the_documented_launch_does_not_pass_saved`. And where it could have spoken it was
    machine-state dependent: eight empty files touched into `models/*_condor/artifacts/`
    silenced it completely. A guard whose subject is local state cannot certify a claim about a
    box that does not exist yet.

    Asserted on the argv rather than on the parsed args, so a command the parser refuses fails
    here too. The first version's `args is None or not args.saved` let the refused command
    through this class of check entirely.
    """
    offenders = [
        f"  {launch}"
        for launch in _launches(ensemble)
        if not {"-t", "--train"} & set(launch.argv)
    ]
    assert not offenders, (
        f"ensembles/{ensemble}/README.md documents a launch without `-t`. Each member then "
        "resolves `get_latest_model_artifact_path('forecasting')`, which raises "
        "FileNotFoundError when the directory holds no artifact — and no member of either "
        f"chrome ensemble has one:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("ensemble", CHROME_ENSEMBLES)
def test_the_documented_launch_does_not_pass_saved(ensemble):
    """`-t --saved` parses cleanly and still dies on a fresh box, so the token matters.

    `_create_model_args` computes `saved = ctx.args.saved if train else True`. The member
    FORECAST step is therefore saved regardless of this flag, reusing what training cached — but
    the member TRAINING step inherits it and goes looking for data a rented box has never
    written. The parser cannot catch this, which is why it is asserted separately.
    """
    offenders = [
        f"  {launch}"
        for launch in _launches(ensemble)
        if {"-sa", "--saved"} & set(launch.argv)
    ]
    assert not offenders, (
        f"ensembles/{ensemble}/README.md documents `--saved`. The member training step then "
        "looks for locally stored data, which a freshly rented box does not have (#639):\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("ensemble", CHROME_ENSEMBLES)
def test_the_documented_launch_carries_no_flag_beyond_a_fresh_box_forecast(ensemble):
    """Whitelist. `-p` is not the only flag that makes a documented launch wrong.

    Rule 5 of `docs/runpod_run_guide.md` keeps publish credentials off rented hardware, and the
    delivery is gated on views-crafdapi#180 and #599 — so `-p` must not appear. But the first
    version enumerated only `-p` and `--saved`, and `-wn -re -dd -o 3` passed: `-wn` needs a W&B
    credential on the box, `-dd` runs a drift self-test at fetch, `-o` overrides the timestep.
    Enumerating harm requires imagining it, so this enumerates the four tokens that are known to
    be right and refuses the rest.
    """
    offenders = []
    for launch in _launches(ensemble):
        extra = [token for token in launch.argv if token not in FRESH_BOX_TOKENS]
        if extra:
            offenders.append(f"  {launch}\n      unexpected: {extra}")
    assert not offenders, (
        f"ensembles/{ensemble}/README.md documents a launch carrying flags beyond a fresh-box "
        f"forecast (allowed: {sorted(FRESH_BOX_TOKENS)}). `-p` would need publish credentials "
        "on the box; `-wn` a W&B credential; `-dd` and `-o` change what the run does:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("ensemble", CHROME_ENSEMBLES)
def test_the_documented_launch_sets_wandb_offline_in_the_block_it_documents(ensemble):
    """The fence an operator copies must set `WANDB_MODE=offline`, or the run stops at a login.

    `_execute_model_training` wraps `_train_ensemble` in
    `self._wandb_module.initialize_run(...)` (pipeline-core
    `managers/ensemble/prediction_frame_ensemble.py:413`), so W&B is contacted before the first
    member trains. On a box with no credential that is `UsageError: api_key not configured
    (no-tty)`, or an interactive prompt in a terminal — the run does no work either way.
    `--wandb_notifications` is not the lever: it gates alerts, not `initialize_run`, so the
    whitelist in `test_the_documented_launch_carries_no_flag_beyond_a_fresh_box_forecast`
    cannot express this and it is asserted here.

    Asserted **per fence**, not per manual block, because the failure mode is an operator
    copying the code block: an export mentioned only in the surrounding prose is an export that
    does not get run. Only `WANDB_MODE` is required — `WANDB_SILENT` merely quiets output, so
    demanding it would be asserting cosmetics.

    #645. The reason #642 shipped without this: the verification run used a wrapper that set
    `WANDB_MODE=offline` while the README did not, so a launch was verified working for a
    reason the documentation never stated.
    """
    block = _manual_block(ensemble)
    fences = _BACKTICK_FENCE_RE.findall(block)
    # _launches() already refuses an unreadable block; this reaches the fences directly because
    # the export is deliberately NOT a command (_LOOKS_LIKE_A_COMMAND_RE ignores it), so it
    # never appears in a Launch.
    offenders = [
        fence
        for fence in fences
        if _LOOKS_LIKE_A_COMMAND_RE.search(fence) and not _WANDB_OFFLINE_RE.search(fence)
    ]
    assert not offenders, (
        f"ensembles/{ensemble}/README.md documents a launch in a ``` fence that does not set "
        "WANDB_MODE=offline. The ensemble manager opens a W&B run before any member trains, so "
        "on a fresh box this stops at a credential prompt having done no work (#645). Add "
        "`export WANDB_MODE=offline WANDB_SILENT=true` inside the fence, as "
        "pod_run_fao_delivery.sh:272 does:\n"
        + "\n".join(f"  ---\n{fence.rstrip()}\n  ---" for fence in offenders)
    )


def test_chrome_box_documents_the_cm_run_first():
    """`chrome_box` cannot run before `chrome_crate`, on the same filesystem.

    `load_cm_frame` resolves `EnsemblePathManager("chrome_crate")
    ._get_generated_predictions_data_file_paths("forecasting")` and fails loud when it finds
    nothing. The order is a precondition of the run, so the block a person follows must show the
    cm launch above the pgm one.

    Scoped to the manual block. The first version searched the whole README and was satisfied by
    a `chrome_crate` mention inside the ASCII directory tree, while the runnable fence was
    reversed — the recurring `src.index(a) < src.index(b)` survivor shape.
    """
    order = [launch.ensemble for launch in _launches("chrome_box")]
    assert "chrome_crate" in order, (
        "ensembles/chrome_box/README.md's manual block does not document the chrome_crate run. "
        "chrome_box reads chrome_crate's forecast from local disk (load_cm_frame), so running it "
        "alone fails with 'no local forecast found for country model chrome_crate'."
    )
    assert order.index("chrome_crate") < order.index("chrome_box"), (
        "ensembles/chrome_box/README.md documents its own launch before chrome_crate's. The cm "
        f"run is a precondition of the pgm run, not a parallel option. Order found: {order}"
    )

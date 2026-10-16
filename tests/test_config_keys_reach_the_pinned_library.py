"""A config key the pinned library does not have is accepted, ignored, and paid for.

`tools/podrun/pod_run_model.sh` installs `views-hydranet` FROM PyPI under a range pin, and
clones views-models at `-b development`. So the config a rented pod trains is whatever is on
`development`, resolved against whatever that range admits. Those two facts combine badly:

  - `HydraNetConfig` sets `extra="allow"`, and its `reject_near_miss_keys` only refuses
    unknown keys within one edit of a real field. A genuinely new key — `body_nll_denominator`
    is the worked example — is not near-miss to anything, so an older library **accepts it
    and ignores it**.
  - The run then completes. It trains the arm's control, reports success, and comes back
    byte-identical to the model it was supposed to be contrasted against. A ~$2.25 arm
    produces a confounded "null" that reads as a real negative result (C-324 / C-331).

This file is the loud failure that case otherwise lacks. It PARSES the pin out of the runner
and never writes to it: `pod_run_model.sh` is production infrastructure, and raising a floor
there is an operator decision, not a side effect of a test run.

**It is live but quiet today**, which is the correct resting state and is asserted below
rather than left to inspection — a guard whose subject never occurs is indistinguishable from
a guard that cannot fire (C-159/C-160).
"""

import re
from pathlib import Path

import pytest

from tests.conftest import ALL_MODEL_DIRS, load_config_module

pytestmark = pytest.mark.beige

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "tools" / "podrun" / "pod_run_model.sh"

#: Config key -> the views-hydranet release that first IMPLEMENTS it.
#:
#: A key belongs here when an older library would accept it silently rather than raise. Keys
#: that are refused by an older library do not need an entry: that failure is already loud.
#:
#: `trajectory_log_path` is deliberately ABSENT. The key itself predates 0.1.3; what 0.1.3
#: added is the parent-directory creation an absolute path needs, and
#: `models/copper_kestrel/configs/config_hyperparameters.py` creates that directory itself
#: precisely so the config carries no version dependency. If that mkdir is ever removed, add
#: `"trajectory_log_path": (0, 1, 3)` here in the same change.
KEY_REQUIRES: dict[str, tuple[int, int, int]] = {
    # views-hydranet 0.1.3 (PyPI 2026-10-16). The zero-truncated NB body's NLL denominator as
    # a config axis. Under 0.1.1/0.1.2 it is accepted and dropped.
    "body_nll_denominator": (0, 1, 3),
}

#: Keys that no config sets yet. Pinned so that the day one does, the author is told that
#: this file is the thing that will check it — and so that removing a key from KEY_REQUIRES
#: while a config still sets it cannot pass unnoticed.
EXPECTED_UNUSED = frozenset({"body_nll_denominator"})


def _pinned_specifier() -> str:
    """The `views-hydranet` requirement specifier the runner installs, verbatim."""
    text = RUNNER.read_text(encoding="utf-8")
    # A requirement, not prose: the name must be followed by a version operator. An earlier
    # version of this matched any quoted string starting "views-hydranet" and picked up a
    # COMMENT ("views-hydranet rebuilds the model from that sidecar..."), which made the
    # duplicate check below fire on a file with exactly one pin.
    matches = re.findall(
        r'"(views-hydranet(?:\[[^\]]*\])?\s*(?:~=|==|>=|>|<=|<)[^"]*)"', text
    )
    assert matches, (
        f"no quoted views-hydranet requirement found in {RUNNER.relative_to(REPO_ROOT)}. "
        f"The pin moved or the install line was rewritten — this file cannot answer its "
        f"question without it, and must fail rather than report a clean result (C-160)."
    )
    assert len(set(matches)) == 1, (
        f"{RUNNER.relative_to(REPO_ROOT)} names views-hydranet more than once with different "
        f"specifiers: {sorted(set(matches))}. Which one a pod resolves then depends on order."
    )
    return matches[0]


def _floor_and_ceiling(spec: str) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    """Lowest version the specifier admits, and the exclusive upper bound if it has one."""
    body = spec[len("views-hydranet"):].strip()
    compatible = re.fullmatch(r"~=\s*(\d+)\.(\d+)\.(\d+)", body)
    if compatible:
        major, minor, patch = (int(g) for g in compatible.groups())
        # PEP 440: ~=X.Y.Z is >=X.Y.Z, <X.(Y+1).0
        return (major, minor, patch), (major, minor + 1, 0)
    lower = re.search(r">=\s*(\d+(?:\.\d+)*)", body)
    upper = re.search(r"<\s*(\d+(?:\.\d+)*)", body)
    assert lower, (
        f"cannot read a lower bound out of {spec!r}. Extend this parser in the same change "
        f"that changes the pin; an unparsed pin must not read as satisfied."
    )
    to_tuple = lambda s: tuple(int(p) for p in s.split("."))  # noqa: E731
    return to_tuple(lower.group(1)), (to_tuple(upper.group(1)) if upper else None)


def _pad(version: tuple[int, ...]) -> tuple[int, int, int]:
    return (version + (0, 0, 0))[:3]


def _models_setting(key: str) -> list[str]:
    """Every model whose merged hyperparameter config declares `key`."""
    out = []
    for model_dir in ALL_MODEL_DIRS:
        config_path = model_dir / "configs" / "config_hyperparameters.py"
        if not config_path.is_file():
            continue
        module = load_config_module(config_path)
        getter = getattr(module, "get_hp_config", None)
        if getter is None:
            continue
        if key in getter():
            out.append(model_dir.name)
    return sorted(out)


def test_the_runner_pin_is_readable():
    """Read it once, loudly. Everything below is vacuous if this cannot be parsed."""
    spec = _pinned_specifier()
    floor, ceiling = _floor_and_ceiling(spec)
    assert floor >= (0, 1, 0), f"implausible floor {floor} parsed out of {spec!r}"
    assert ceiling is None or ceiling > floor, f"{spec!r} admits no version at all"


@pytest.mark.parametrize("key", sorted(KEY_REQUIRES))
def test_a_key_the_pin_cannot_reach_is_refused(key):
    """⛔ The silent no-op case: a config sets a key no admitted version implements.

    This is the hard failure. If the pin's ceiling excludes the release that implements the
    key, then NO resolution can produce a library that honours it, and every pod run of that
    config trains the control while reporting success.
    """
    users = _models_setting(key)
    if not users:
        pytest.skip(f"no config sets {key!r}")
    required = KEY_REQUIRES[key]
    spec = _pinned_specifier()
    floor, ceiling = _floor_and_ceiling(spec)
    assert ceiling is None or required < _pad(ceiling), (
        f"{users} set {key!r}, which views-hydranet implements from "
        f"{'.'.join(map(str, required))}, but the runner pin {spec!r} excludes it. No "
        f"resolution can honour that key: the library accepts it (extra='allow'), drops it, "
        f"and the run trains the control and reports OK (C-324/C-331). Raise the pin in "
        f"tools/podrun/pod_run_model.sh — an operator decision, deliberately not automated — "
        f"or remove the key."
    )
    assert _pad(floor) >= required, (
        f"{users} set {key!r}, which views-hydranet implements from "
        f"{'.'.join(map(str, required))}, but the runner pin {spec!r} still ADMITS versions "
        f"below that ({'.'.join(map(str, floor))} upward). A resolver takes the newest match "
        f"today, so this is latent rather than broken — and latent on rented hardware that "
        f"costs money per run. Raise the floor, or have the config not depend on the "
        f"version (see KEY_REQUIRES' note on trajectory_log_path for how)."
    )


def test_the_unused_set_is_honest():
    """The resting state, asserted rather than assumed.

    Every key in KEY_REQUIRES that no config sets must be listed in EXPECTED_UNUSED, and
    every key in EXPECTED_UNUSED must really be unset. Without this, the parametrised test
    above skips its way to green and nobody can tell whether it is quiet because the
    repository is clean or because the key names drifted.
    """
    unused = {key for key in KEY_REQUIRES if not _models_setting(key)}
    assert unused == set(EXPECTED_UNUSED), (
        f"keys in KEY_REQUIRES that no config sets: {sorted(unused)}; EXPECTED_UNUSED says "
        f"{sorted(EXPECTED_UNUSED)}. If a config now sets one, remove it from EXPECTED_UNUSED "
        f"in the same commit — the check above is then live and must be seen to pass."
    )


def test_every_tracked_key_is_spelled_the_way_the_library_spells_it():
    """A typo'd key here would make this whole file assert nothing.

    `extra="allow"` means a misspelled key in a CONFIG is silently ignored; a misspelled key
    in KEY_REQUIRES is worse, because it also makes the guard silently vacuous. There is no
    way to check a spelling against the library from here without importing it, so what is
    checked is the weaker, still useful thing: the names are lowercase snake_case with no
    whitespace, which is the convention every key in every config follows.
    """
    for key in KEY_REQUIRES:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", key), (
            f"{key!r} is not snake_case; it cannot be a views-hydranet config key"
        )

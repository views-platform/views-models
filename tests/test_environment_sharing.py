"""Which conda environment each launcher uses, and who it shares it with (C-115, C-116).

131 `requirements.txt` resolve into **11 environments**. `run.sh` pip-installs only
its own file into that shared prefix and never uninstalls, so an environment's
contents depend on which tenant ran last. Measured consequences, both directions:

    declared but absent   `ensembles/skinny_love` declared views-frames>=1.7.0,<2.0.0
                          while `envs/views_ensemble` had it not installed — and the
                          model completed a run in that state (2026-07-22, wandb
                          atomic-jazz-101). The declaration was removed in PR #325.
    present, undeclared   27 models receive views-datafactory because a co-tenant
                          declares it; in `envs/views-baseline` only 10 of 37 do.

The tests here guard the one case where that sharing is not merely misleading but
**silently wrong**, and they exist because the trap is invisible: it looks like a typo.
"""

from pathlib import Path
import re
import subprocess

import pytest

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

pytestmark = pytest.mark.green

REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = re.compile(r'^env_path="\$project_path/envs/(?P<env>[^"]+)"', re.M)

# ── the trap ──────────────────────────────────────────────────────────
# These two directory names differ by ONE character, and that character is
# load-bearing. Their tenants declare mutually unsatisfiable versions:
#
#   envs/views_r2darts2  (22 tenants)  views-r2darts2==0.1.0  x12
#                                      views-r2darts2>=0.1.0  x10
#   envs/views-r2darts2  ( 9 tenants)  views-r2darts2>=1.0.0,<2.0.0
#
# `==0.1.0` and `>=1.0.0,<2.0.0` cannot both be satisfied. They do not collide
# today ONLY because they resolve to separately-named directories. Merging the
# names — the obvious tidy-up, and what a regenerated run.sh would do — puts both
# in one prefix, where run.sh installs each tenant's file with no uninstall and
# the resolved version becomes whatever ran last. There is no error: pip reports
# success, the models train, and half of them forecast with the wrong algorithm
# version. Registered as **C-115, Tier 1**.
#
# This is enforced by test_no_environment_holds_mutually_unsatisfiable_specs below,
# which asserts the invariant rather than the directory names — a single model moved
# between the two environments is as dangerous as a rename, and a name-based check
# passes it.

# pip accepts these as requirements.txt lines; PEP 508 does not parse them.
_BARE_URL_PREFIXES = ("git+", "http://", "https://", "-e ", "-r ", "--")


def _env_by_launcher():
    out = subprocess.run(
        ["git", "ls-files", "-z", "*run.sh"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    mapping = {}
    for name in out.split("\0"):
        if not name:
            continue
        match = _ENV_PATH.search((REPO_ROOT / name).read_text(encoding="utf-8"))
        if match:
            mapping[name] = match.group("env")
    return mapping


def _candidate_versions(specifier_sets):
    """Versions worth testing: every version named, plus its immediate successors.

    Enough to decide satisfiability for the range shapes this repo uses (`==`,
    `>=`, `<`), without pretending to solve it in general.
    """
    seen = set()
    for spec_set in specifier_sets:
        for spec in spec_set:
            try:
                base = Version(spec.version)
            except InvalidVersion:
                continue
            major, minor, micro = base.major, base.minor, base.micro
            seen.update({
                str(base),
                f"{major}.{minor}.{micro + 1}",
                f"{major}.{minor + 1}.0",
                f"{major + 1}.0.0",
            })
    return seen


def test_no_environment_holds_mutually_unsatisfiable_specs():
    """The real invariant: co-tenants of one environment must be able to agree.

    Two models sharing a conda prefix while declaring specs with no version in
    common cannot both get what they asked for. `run.sh` installs each tenant's
    file into the shared prefix and never uninstalls, so pip reports success to
    both and the resolved version is whichever ran last — no error, wrong output.

    This is asserted over the actual (environment, spec) pairs rather than over
    directory *names*, because the dangerous change is not only the full rename:
    moving a single `==0.1.0` model into the `>=1.0.0,<2.0.0` environment breaks
    it just as thoroughly and would leave both names in place. An earlier draft of
    this test checked the names and passed that scenario. See **C-115**.
    """
    launcher_env = _env_by_launcher()
    by_env = {}
    for launcher, env in launcher_env.items():
        req = REPO_ROOT / Path(launcher).parent / "requirements.txt"
        if not req.is_file():
            continue
        for raw in req.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(_BARE_URL_PREFIXES):
                continue
            try:
                parsed = Requirement(line)
            except InvalidRequirement:
                continue
            if parsed.url:
                continue
            by_env.setdefault(env, {}).setdefault(parsed.name, {}).setdefault(
                str(parsed.specifier), []
            ).append(launcher)

    conflicts = []
    for env, packages in sorted(by_env.items()):
        for package, variants in sorted(packages.items()):
            if len(variants) < 2:
                continue
            spec_sets = [SpecifierSet(s) for s in variants]
            combined = SpecifierSet(",".join(variants))
            candidates = _candidate_versions(spec_sets)
            if not any(combined.contains(v, prereleases=True) for v in candidates):
                detail = "; ".join(
                    f"{spec or '(none)'} <- {len(files)} model(s), e.g. {files[0]}"
                    for spec, files in sorted(variants.items())
                )
                conflicts.append(f"  envs/{env}: {package}: {detail}")

    assert not conflicts, (
        "an environment's tenants declare specs with no version in common — pip will "
        "report success to each and the resolved version will be whichever ran last:\n"
        + "\n".join(conflicts)
        + "\n\nIf this appeared after renaming or merging environment directories, "
        "revert that: envs/views_r2darts2 and envs/views-r2darts2 are kept apart on "
        "purpose (C-115). Reconcile the specs before unifying the names."
    )


def test_every_launcher_names_an_environment():
    """A launcher with no `env_path` would install into whatever is active."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "*run.sh"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    all_launchers = {n for n in out.split("\0") if n and n != "monthly_run.sh"}
    resolved = set(_env_by_launcher())
    missing = sorted(all_launchers - resolved)
    assert not missing, (
        "run.sh with no `env_path=\"$project_path/envs/...\"` line — it would install "
        "into whichever environment happened to be active:\n"
        + "\n".join(f"  {n}" for n in missing)
    )


def test_environment_sharing_is_recorded_not_discovered():
    """Pin the tenant counts, so a change to who-shares-what is a visible diff.

    This is a characterization test: it asserts nothing about what the numbers
    *should* be, only that changing them is deliberate. Adding a model to a
    37-tenant environment is currently a silent act; this makes it a review
    comment. Update the expected mapping when the change is intended.
    """
    counts = {}
    for env in _env_by_launcher().values():
        counts[env] = counts.get(env, 0) + 1

    expected = {
        "views-baseline": 37,
        "views_stepshifter": 32,
        "views_r2darts2": 22,
        "views_ensemble": 13,
        "views-r2darts2": 9,
        "views-hydranet": 8,
        "views-stepshifter": 7,
        "views-seldon": 1,
        "views-postprocessing": 1,
        "views-graphdb": 1,
        "views-faoapi": 1,
    }
    assert counts == expected, (
        "the environment -> tenant mapping changed.\n"
        f"  expected: {dict(sorted(expected.items()))}\n"
        f"  actual:   {dict(sorted(counts.items()))}\n"
        "If deliberate, update `expected` in this test. If you are adding a model, "
        "note that it inherits every package its co-tenants install (C-116)."
    )

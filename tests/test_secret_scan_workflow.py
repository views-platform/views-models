"""Structural guards on the secret-scan workflow (#300).

The workflow itself cannot be executed here, but the two properties that decide
whether it is *capable of failing* are static and are pinned below.

Why this file exists. A secret scan that cannot fail is worse than no scan: the þing-02
verdict struck a contract clause and named this job as its replacement, so a false green
here is a false assurance platform-wide. Two settings decide it, and both were verified
empirically on 2026-07-31 by cloning this repo with `--depth 1`:

    gitleaks git --log-opts="--all --full-history"   ->  "1 commits scanned"
                                                        "no leaks found"
                                                        exit 0

A naive job calls that green forever. Worse, a ratio check ("did we scan most of the
commits?") ALSO passes there, because `git rev-list --count --all` in a shallow clone is
likewise 1 — both numbers come from the same truncated repository and lie together. Only
`git rev-parse --is-shallow-repository` is an independent witness.
"""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.beige]

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "secret_scan.yml"
ALLOWLIST = REPO_ROOT / ".gitleaksignore"


def _workflow_text() -> str:
    assert WORKFLOW.exists(), f"{WORKFLOW.name} must exist — it is D5's replacement"
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_is_tracked_and_not_gitignored():
    """The job must exist in git, not merely on someone's disk.

    `.gitignore` carries blanket `*.json` / `*.yaml` / `*.yml` rules for wandb run
    outputs. They also swallow anything new under `.github/` — the four workflows
    tracked before this one survive only because git does not ignore what is already
    tracked. Adding this file hit it directly: `git add` reported the path ignored and
    the commit went through WITHOUT it, leaving a PR that looked complete and contained
    no workflow. A `!.github/**` negation now prevents that; this test prevents its
    removal.
    """
    import subprocess

    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(WORKFLOW.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
    ).returncode == 0
    assert not ignored, (
        "the secret-scan workflow is gitignored — it would silently not exist in the "
        "repository. Restore the `!.github/**` negation in .gitignore (#300)"
    )

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(WORKFLOW.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT, capture_output=True,
    ).returncode == 0
    assert tracked, "the secret-scan workflow exists on disk but is not tracked in git"


def test_checkout_fetches_full_history():
    """`fetch-depth: 0` — without it the scan walks one commit and passes green."""
    assert "fetch-depth: 0" in _workflow_text(), (
        "actions/checkout defaults to depth 1. Without `fetch-depth: 0` the "
        "`--all --full-history` scan sees a single commit, finds nothing, and exits 0 "
        "— a permanent false green (#300)"
    )


def test_shallow_repository_guard_is_present():
    """The independent witness. A ratio check alone cannot detect a shallow clone."""
    assert "--is-shallow-repository" in _workflow_text(), (
        "the shallow-repository check is the only guard not derived from the scan "
        "itself; a commits-scanned ratio passes in a shallow clone because both sides "
        "of the ratio are truncated (#300)"
    )


def test_scan_covers_all_refs_and_full_history():
    text = _workflow_text()
    assert "--all --full-history" in text, (
        "the scan must cover every ref, not just the checked-out branch — this repo's "
        "only real finding lives on a branch, in a file deleted from HEAD"
    )


def test_gitleaks_version_is_pinned_with_a_checksum():
    """'latest' would let the verdict change without a commit."""
    text = _workflow_text()
    assert "GITLEAKS_VERSION:" in text and "latest" not in text.lower().split("gitleaks_version:")[1][:40]
    assert "sha256sum -c" in text, (
        "a pinned version without a checksum pins a name, not an artifact"
    )


def test_findings_are_redacted_in_ci_logs():
    assert "--redact" in _workflow_text(), (
        "CI logs on a public repo are public; a scanner that prints the secret it "
        "found has published it a second time"
    )


def test_every_allowlist_fingerprint_carries_a_justification():
    """An undocumented allowlist entry is indistinguishable from a hidden leak."""
    assert ALLOWLIST.exists(), ".gitleaksignore must exist"
    lines = ALLOWLIST.read_text(encoding="utf-8").splitlines()

    undocumented = []
    for index, line in enumerate(lines):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        # Walk up past any SIBLING fingerprints to the comment block that documents
        # them. Grouping related findings under one justification is correct — the two
        # README placeholders are literally the same string in two copied files — so
        # requiring a comment immediately above every line would punish good practice.
        # What must not exist is a fingerprint reachable only from blank space.
        documented = False
        for j in range(index - 1, -1, -1):
            above = lines[j].strip()
            if not above:
                break            # blank line: the entry stands alone, undocumented
            if above.startswith("#"):
                documented = True
                break
            # else: a sibling fingerprint — keep walking up
        if not documented:
            undocumented.append(entry)

    assert not undocumented, (
        "these .gitleaksignore fingerprints have no comment above them explaining why "
        f"the finding is benign: {undocumented}"
    )

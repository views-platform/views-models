"""This repo cites the Appwrite Seam Contract by its current name (#304).

The contract, homed in views-appwrite, was renamed from ``PLATFORM-001`` by that
repo's ADR-011. The rename is not cosmetics: an opaque identifier hides staleness.
This repository's register cited the contract at ``60674b2`` — v1.0.0, two ratified
versions behind — and nobody noticed, because nothing about ``60674b2`` signals age.
The same argument applies to the name: ``PLATFORM-001`` costs every reader a lookup,
and a lookup that is skipped is a citation that is never checked.

These guards are deliberately narrow. They cannot detect a *stale pin* — that needs
the other repository — so they do not pretend to. What they detect is the cheap,
mechanical regression: a new citation written under the retired identity.

The old name is still permitted as a **signpost** ("formerly ``PLATFORM-001``"), which
is exactly what the contract's own header does for a reader arriving with it.
"""

from pathlib import Path
import subprocess

import pytest

pytestmark = pytest.mark.green

REPO_ROOT = Path(__file__).resolve().parents[1]

RETIRED_NAME = "PLATFORM-001"
CURRENT_NAME = "Appwrite Seam Contract"
RETIRED_FILENAME = "PLATFORM-001_identity_secrets_configuration_contract.md"

# This file necessarily contains every string it forbids, so it excludes itself.
# The alternative — matching only outside string literals — is the C-57 mistake:
# a regex cannot reliably tell code from the text that describes it.
SELF = Path(__file__).name

TEXT_SUFFIXES = {".md", ".py", ".sh", ".toml", ".yml", ".yaml", ".txt", ".cfg"}


def _tracked_text_files():
    """Tracked files only — an untracked file is not something this repo says."""
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    for name in out.split("\0"):
        if not name or Path(name).name == SELF:
            continue
        path = REPO_ROOT / name
        if path.suffix.lower() in TEXT_SUFFIXES and path.is_file():
            yield name, path


def _lines_mentioning(needle):
    hits = []
    for name, path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if needle not in text:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if needle in line:
                hits.append((name, number, line.strip()))
    return hits


def _paragraphs_mentioning(needle):
    """(file, first line number, text) per blank-line-delimited block containing needle.

    Scope is the paragraph, not the line, because prose wraps: the canonical
    CLAUDE.md sentence carries the signpost and the current name on opposite
    sides of a line break. That text is propagated verbatim across six repos, so
    a line-scoped rule would demand reflowing a file this repo does not own the
    formatting of.
    """
    blocks = []
    for name, path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if needle not in text:
            continue
        start = 1
        for para in text.split("\n\n"):
            if needle in para:
                blocks.append((name, start, " ".join(para.split())))
            start += para.count("\n") + 2
    return blocks


def test_no_file_cites_the_retired_contract_filename():
    """The old path resolves at old commits, but a NEW citation must not use it.

    A link written today against the retired filename can only be pinned to a
    commit from before the rename — which is to say, pinned to a version that is
    already superseded on the day it is written.
    """
    hits = _lines_mentioning(RETIRED_FILENAME)
    assert not hits, "retired contract filename cited:\n" + "\n".join(
        f"  {n}:{ln}: {text[:120]}" for n, ln, text in hits
    )


def test_every_mention_of_the_retired_name_is_a_signpost():
    """``PLATFORM-001`` may appear only in a paragraph that also names the contract.

    That keeps the old identifier findable for anyone who arrives with it, while
    making a bare citation — one that leaves the reader with the lookup — fail.
    """
    offenders = [
        (name, number, text)
        for name, number, text in _paragraphs_mentioning(RETIRED_NAME)
        if CURRENT_NAME not in text
    ]
    assert not offenders, (
        f"{RETIRED_NAME} cited without naming {CURRENT_NAME!r} in the same paragraph:\n"
        + "\n".join(f"  {n}:{ln}: {text[:120]}" for n, ln, text in offenders)
    )

#!/usr/bin/env python3
"""Derive a delivery note's checksum from the facts on its own title page.

The readable identifier says what a delivery IS. The checksum says it has not
drifted: it is a digest of every specification field, so changing any one of
them changes the number. Quoting the pair ``<identifier> / <checksum>`` names
a delivery unambiguously and proves the note still describes it.

    python3 checksum.py <note-dir>            print the computed checksum
    python3 checksum.py <note-dir> --check    exit 1 if it disagrees with the
                                              \\deliverychecksum in main.tex

Eight decimal digits: ~10^8 values, which for a project producing tens of
deliveries a year leaves collision risk negligible while staying short enough
to read aloud.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

# Order is part of the definition. Appending a new field changes every
# checksum computed afterwards, which is correct -- it is a different spec.
FIELDS = (
    "deliveryid",
    "deliverylibrary",
    "deliverypartition",
    "deliveryspatialres",
    "deliveryspatialextent",
    "deliverytemporalres",
    "deliverytemporalextent",
    "deliverystructure",
    "deliverytargets",
    "deliverybackend",
    "deliveryframework",
    "deliverycompute",
    "deliverysha",
    "deliverydate",
)


def _strip_latex(value: str) -> str:
    """Reduce a field to its semantic content.

    Markup is presentation, not fact: ``\\texttt{pgm}`` and ``pgm`` describe the
    same delivery and must not produce different checksums.
    """
    value = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", value)
    value = value.replace("---", "-").replace("--", "-")
    value = re.sub(r"\\[a-zA-Z]+", "", value)
    value = value.replace("{", "").replace("}", "").replace("\\", "")
    return " ".join(value.split())


def read_fields(main_tex: Path) -> dict[str, str]:
    source = main_tex.read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for name in FIELDS:
        match = re.search(
            r"^\\newcommand\{\\" + name + r"\}\{(.*)\}\s*$", source, re.M
        )
        if not match:
            raise SystemExit(f"{main_tex}: no \\{name} defined")
        found[name] = _strip_latex(match.group(1))
    return found


def checksum(fields: dict[str, str]) -> str:
    canonical = "\n".join(f"{k}={fields[k]}" for k in FIELDS)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{int(digest[:16], 16) % 10**8:08d}"


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)
    note = Path(argv[0])
    main_tex = note if note.is_file() else note / "main.tex"
    fields = read_fields(main_tex)
    computed = checksum(fields)

    if "--check" not in argv:
        print(computed)
        return 0

    declared = re.search(
        r"^\\newcommand\{\\deliverychecksum\}\{(\d{8})\}\s*$",
        main_tex.read_text(encoding="utf-8"),
        re.M,
    )
    if declared is None:
        print(f"no \\deliverychecksum in {main_tex}; should be {computed}")
        return 1
    if declared.group(1) != computed:
        print(
            f"checksum stale: declared {declared.group(1)}, "
            f"computed {computed}. A specification field changed."
        )
        return 1
    print(f"checksum {computed} matches the specification fields")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

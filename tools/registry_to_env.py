#!/usr/bin/env python3
"""Emit ``NAME=value`` env lines for the NON-SECRET coordinates in a PLATFORM-001
coordinate registry (þing-01 #275 / #287, verdict D2).

Reads the OWNED registry file (``views-appwrite/docs/ADRs/platform/coordinate_registry.toml``)
and prints only the **connection** and **target** classes — the non-secret identifiers a
consumer needs. It NEVER emits a secret: secret entries in the registry are *slots* (name +
required scopes) that carry no value, and the operator supplies the actual key separately. This
is the read-don't-copy discipline that retires the laptop-``.env`` copy-chain: the registry is
read and its values emitted; the file is never copied into a repo, and no value is ever baked
into code (PLATFORM-001 §4).

Mirrors ``views-faoapi/deployment/registry_to_env.py`` — the same canonical reader. A tiny
per-consumer copy of a stdlib-only reader is the þing-01 pattern; the single source of truth is
the DATA (the registry), never the reader. Requires Python ≥ 3.11 (``tomllib``); run it with the
launcher's activated env python, not the system interpreter.

Usage:  python registry_to_env.py <registry.toml>
"""

from __future__ import annotations

import sys
import tomllib

# The declared classes that hold non-secret coordinate values. `secret` (slots, no value),
# `excluded`, `meta`, and `test_environment` are deliberately NOT emitted.
_COORDINATE_CLASSES = ("connection", "target")


def coordinates(registry_path: str) -> list[str]:
    """Return ``NAME=value`` lines for every connection/target coordinate in the registry.

    Raises on a malformed registry or a coordinate entry missing its ``value`` — fail loud
    rather than emit a half-built environment (verdict D5)."""
    with open(registry_path, "rb") as fh:
        registry = tomllib.load(fh)
    lines: list[str] = []
    for cls in _COORDINATE_CLASSES:
        for name, entry in registry.get(cls, {}).items():
            if "value" not in entry:
                raise ValueError(f"registry coordinate {name!r} (class {cls!r}) has no value")
            lines.append(f"{name}={entry['value']}")
    return lines


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: registry_to_env.py <registry.toml>")
    print("\n".join(coordinates(sys.argv[1])))

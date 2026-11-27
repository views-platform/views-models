#!/usr/bin/env python3
"""Emit ``NAME=value`` env lines for the NON-SECRET coordinates in an Appwrite Seam
Contract coordinate registry (þing-01 #275 / #287, verdict D2).

Reads the OWNED registry file (``views-appwrite/docs/ADRs/platform/coordinate_registry.toml``)
and prints only the **connection** and **target** classes — the non-secret identifiers a
consumer needs. It NEVER emits a secret: secret entries in the registry are *slots* (name +
required scopes) that carry no value, and the operator supplies the actual key separately. This
is the read-don't-copy discipline that retires the laptop-``.env`` copy-chain: the registry is
read and its values emitted; the file is never copied into a repo, and no value is ever baked
into code (The Appwrite Seam Contract §4 — the contract formerly called `PLATFORM-001`).

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


def _is_planned(entry: dict) -> bool:
    """A coordinate declared for a consumer that does not exist yet.

    The registry uses ``status = "planned — …"`` to reserve a name before the container
    exists. Such an entry has no ``value`` **by design**, and that is a declaration of
    intent, not a malformed registry.
    """
    return str(entry.get("status", "")).strip().lower().startswith("planned")


def coordinates(registry_path: str) -> list[str]:
    """Return ``NAME=value`` lines for every connection/target coordinate in the registry.

    Raises on a malformed registry or a coordinate entry missing its ``value`` — fail loud
    rather than emit a half-built environment (verdict D5). **Planned entries are skipped,
    not fatal**: see below.

    Why the skip exists. On 2026-07-31 views-appwrite added
    ``[target.APPWRITE_CRAFD_BUCKET_ID]`` with ``status = "planned — views-crafdapi"`` and
    (NOTE: those four CRAFD slots graduated to valued ``[target]`` entries on 2026-08-02,
    registry v1.4.0, so the example below is history rather than the current registry)
    no value, reserving the name for a consumer that does not exist yet. This function
    raised on it — and because it raises for the *whole registry*, the un_fao launcher
    lost **every** coordinate, not just the planned one. A neighbouring repository adding
    a placeholder should not be able to unconfigure the FAO delivery path. Fail-loud is
    still right for a coordinate that *ought* to have a value; a reservation is a
    different thing and the registry already distinguishes them.
    """
    with open(registry_path, "rb") as fh:
        registry = tomllib.load(fh)
    lines: list[str] = []
    for cls in _COORDINATE_CLASSES:
        for name, entry in registry.get(cls, {}).items():
            if "value" not in entry:
                if _is_planned(entry):
                    continue  # reserved name, consumer not built yet
                raise ValueError(f"registry coordinate {name!r} (class {cls!r}) has no value")
            lines.append(f"{name}={entry['value']}")
    return lines


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: registry_to_env.py <registry.toml>")
    print("\n".join(coordinates(sys.argv[1])))

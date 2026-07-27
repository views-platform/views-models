#!/usr/bin/env python3
"""Self-diagnosing credential presence check for views-models.

One command answers "which credentials am I missing?" without reading code —
the fix for the recurring "the creds are a mystery" problem
(see reports/security/appwrite_credentials_audit.md).

    python tools/check_credentials.py

The required key NAMES are the single-sourced schema in `.env.example`. This tool
reports, per key, whether it is filled in the local (gitignored) `.env` or already
present in the process environment, and exits non-zero if any is missing. It reads
NO secret values into its output — only names and present/missing status.

Stdlib only; no dependency on python-dotenv.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_env_names(path: Path) -> list[str]:
    """Key names declared in an env file (lines like `KEY=` / `KEY=value`)."""
    names: list[str] = []
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        names.append(s.split("=", 1)[0].strip())
    return names


def _parse_env_filled(path: Path) -> set[str]:
    """Key names that have a NON-EMPTY value in an env file (values never printed)."""
    filled: set[str] = set()
    if not path.exists():
        return filled
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        if v.strip():
            filled.add(k.strip())
    return filled


def main() -> int:
    example = REPO_ROOT / ".env.example"
    if not example.exists():
        print("ERROR: .env.example not found — cannot determine the required key schema.")
        return 2

    required = _parse_env_names(example)
    dotenv = REPO_ROOT / ".env"
    filled_in_dotenv = _parse_env_filled(dotenv)

    missing: list[str] = []
    print(f"Credential check — schema: {example.relative_to(REPO_ROOT)}"
          f" ({len(required)} keys) | local .env: {'present' if dotenv.exists() else 'ABSENT'}\n")
    for key in required:
        if filled_in_dotenv and key in filled_in_dotenv:
            where = "filled in .env"
        elif os.getenv(key):
            where = "set in environment"
        else:
            where = "MISSING"
            missing.append(key)
        print(f"  {'ok  ' if where != 'MISSING' else 'MISS'}  {key:<42} {where}")

    print()
    if missing:
        print(f"INCOMPLETE — {len(missing)} of {len(required)} missing: {', '.join(missing)}")
        print("Fill them in .env (copy from .env.example; values from the Appwrite console).")
        return 1
    print(f"OK — all {len(required)} credentials present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared Appwrite API helpers for the liveness checks (S7 extraction, #245).

One home for what the appwrite_store and unfao_delivery checks demonstrably
duplicated: credentials (dataclass, .env parsing, resolution order), the
server-side query builders (the 2026-07-19 pagination-bug cure), and the
default stdlib fetch.

Credentials resolution order (values never rendered anywhere):
    1. process env vars (APPWRITE_ENDPOINT / _DATASTORE_PROJECT_ID / _DATASTORE_API_KEY)
    2. known platform .env files, discovered by walking this file's ancestors
       (robust from the main checkout AND from git worktrees).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

FETCH_TIMEOUT_SECONDS = 25

_ENV_LINE = re.compile(
    r"^export\s+(APPWRITE_ENDPOINT|APPWRITE_DATASTORE_PROJECT_ID|"
    r"APPWRITE_DATASTORE_API_KEY)\s*=\s*(.+?)\s*$"
)

FetchJson = Callable[[str, Dict[str, str]], object]


@dataclass(frozen=True)
class AppwriteCredentials:
    endpoint: str
    project_id: str
    api_key: str


def load_credentials_from_env_file(path: Path) -> Optional[AppwriteCredentials]:
    """Parse an export-style .env; None unless all three keys are present."""
    try:
        text = path.read_text()
    except OSError:
        return None
    found: Dict[str, str] = {}
    for line in text.splitlines():
        match = _ENV_LINE.match(line.strip())
        if match:
            found[match.group(1)] = match.group(2).strip("\"'")
    try:
        return AppwriteCredentials(
            endpoint=found["APPWRITE_ENDPOINT"],
            project_id=found["APPWRITE_DATASTORE_PROJECT_ID"],
            api_key=found["APPWRITE_DATASTORE_API_KEY"],
        )
    except KeyError:
        return None


def _known_env_files():
    """Candidate platform .env files, discovered by walking ancestors — robust
    to running from the main checkout AND from a git worktree under
    .claude/worktrees/ (where fixed parent-counting breaks)."""
    return tuple(
        ancestor / "views-faoapi" / ".env"
        for ancestor in Path(__file__).resolve().parents
        if (ancestor / "views-faoapi" / ".env").exists()
    )


def resolve_credentials() -> Optional[AppwriteCredentials]:
    """Env vars first, then the known platform .env files."""
    import os

    env = {k: os.environ.get(k) for k in (
        "APPWRITE_ENDPOINT", "APPWRITE_DATASTORE_PROJECT_ID", "APPWRITE_DATASTORE_API_KEY",
    )}
    if all(env.values()):
        return AppwriteCredentials(
            env["APPWRITE_ENDPOINT"],              # type: ignore[arg-type]
            env["APPWRITE_DATASTORE_PROJECT_ID"],  # type: ignore[arg-type]
            env["APPWRITE_DATASTORE_API_KEY"],     # type: ignore[arg-type]
        )
    for candidate in _known_env_files():
        credentials = load_credentials_from_env_file(candidate)
        if credentials is not None:
            return credentials
    return None


def newest_first_query(limit: int = 5) -> str:
    """Appwrite query string: order by $createdAt descending, capped.

    Encoded once so every bucket listing is immune to the 25-per-page
    default that produced the 2026-07-19 false-idle verdict.
    """
    import json as _json
    from urllib.parse import quote as _quote

    queries = (
        {"method": "orderDesc", "attribute": "$createdAt"},
        {"method": "limit", "values": [limit]},
    )
    return "&".join("queries[]=" + _quote(_json.dumps(q)) for q in queries)


def stream_newest_query(prefix: str) -> str:
    """Appwrite query string: newest file whose name starts with ``prefix``."""
    import json as _json
    from urllib.parse import quote as _quote

    queries = (
        {"method": "startsWith", "attribute": "name", "values": [prefix]},
        {"method": "orderDesc", "attribute": "$createdAt"},
        {"method": "limit", "values": [1]},
    )
    return "&".join("queries[]=" + _quote(_json.dumps(q)) for q in queries)


def fetch_json(url: str, headers: Dict[str, str]) -> object:
    """Default fetch: stdlib urllib, lazy import, explicit timeout."""
    import json
    import urllib.request

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        return json.load(response)

"""Shared Appwrite API helpers for the liveness checks (S7 extraction, #245).

One home for what the appwrite_store and unfao_delivery checks demonstrably
duplicated: credentials (dataclass, .env parsing, resolution order), the
server-side query builders (the 2026-07-19 pagination-bug cure), and the
default stdlib fetch.

Credentials resolution order (values never rendered anywhere):
    1. process env vars (APPWRITE_ENDPOINT / _DATASTORE_PROJECT_ID / _DATASTORE_API_KEY)
    2. **this repository's own** ``.env``, at the repo root.

#298 — why it is only its own file now. This module used to walk ancestors looking for
``views-faoapi/.env``, a hardcoded FOREIGN repository, and its line regex matched only
``export``-prefixed lines. views-models' own ``.env`` is bare ``KEY=`` style, so the regex
excluded it and the path never looked at it. Net effect: views-models observed its own
internal shelf **under the FAO service's identity** — the answer it got was "the FAO key
can see this", reported as "the shelf is healthy". Those are different propositions, and
they diverge the moment the two keys' scopes differ, which is the entire point of the
least-privilege work. A warning line could not have fixed it: what consumers read is the
exit code, and an exit code carries no warning. So the foreign path is gone, not annotated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

FETCH_TIMEOUT_SECONDS = 25

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_KEYS = (
    "APPWRITE_ENDPOINT",
    "APPWRITE_DATASTORE_PROJECT_ID",
    "APPWRITE_DATASTORE_API_KEY",
)

# Both line styles, because the format is not this module's business (#298): this repo's
# .env is bare `KEY=value`, views-faoapi's is `export KEY=value`, and a credential parser
# that only understands one of them is asserting a habit, not a contract.
_ENV_LINE = re.compile(
    r"^(?:export\s+)?(APPWRITE_ENDPOINT|APPWRITE_DATASTORE_PROJECT_ID|"
    r"APPWRITE_DATASTORE_API_KEY)\s*=\s*(.+?)\s*$"
)

FetchJson = Callable[[str, Dict[str, str]], object]


@dataclass(frozen=True)
class AppwriteCredentials:
    endpoint: str
    project_id: str
    api_key: str


def _clean(value: str) -> str:
    """Strip surrounding quotes, or an unquoted trailing ` # comment`.

    Both forms occur in real files: dotenv writes bare values, and the bash-sourceable
    `.env` this repo uses carries trailing comments on the coordinate lines.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value.split(" #", 1)[0].rstrip()


def parse_env_file(path: Path) -> Dict[str, str]:
    """The APPWRITE_* keys present in an env file. Missing file → empty mapping.

    One job: read a file, return what it declares. It does not decide whether that is
    enough — `load_credentials_from_env_file` and `missing_credentials` do, and keeping
    those separate is what lets the caller tell "no file" from "incomplete file" (#298).
    """
    try:
        text = path.read_text()
    except OSError:
        return {}
    found: Dict[str, str] = {}
    for line in text.splitlines():
        match = _ENV_LINE.match(line.strip())
        if match:
            found[match.group(1)] = _clean(match.group(2))
    return found


def load_credentials_from_env_file(path: Path) -> Optional[AppwriteCredentials]:
    """Parse a `.env` (either line style); None unless all three keys are present."""
    found = parse_env_file(path)
    if not all(found.get(key) for key in REQUIRED_KEYS):
        return None
    return AppwriteCredentials(
        endpoint=found["APPWRITE_ENDPOINT"],
        project_id=found["APPWRITE_DATASTORE_PROJECT_ID"],
        api_key=found["APPWRITE_DATASTORE_API_KEY"],
    )


def own_env_file() -> Path:
    """This repository's own `.env`. The only file this module reads (#298)."""
    return REPO_ROOT / ".env"


def missing_credentials() -> tuple:
    """Which required variables are absent, considering process env then own `.env`.

    Exists so a caller can distinguish **nothing configured** (a truthful skip, exit 0)
    from **configured but incomplete** (a misconfiguration a human must fix — loud, and
    it names the variables). Returning `None` from `resolve_credentials()` conflates the
    two, and that conflation is half of what #298 is about.
    """
    import os

    from_file = parse_env_file(own_env_file())
    return tuple(
        key for key in REQUIRED_KEYS
        if not (os.environ.get(key) or from_file.get(key))
    )


def credential_gap_report() -> tuple:
    """`(verdict, error)` describing why credentials could not be resolved.

    Lives here, not in the surfaces: describing a credential state is the credentials
    module's job, while mapping a verdict to an exit code stays `report.py`'s. Both
    surfaces call it rather than each carrying the same six lines — this is one concept
    in one place, not a premature generalisation of two different things.
    """
    missing = missing_credentials()
    own = own_env_file()
    if not missing:
        # Nothing is missing, yet the caller has no credentials — so they were injected
        # as None deliberately (a test, or a surface constructed without them). Do not
        # report a configuration fault we cannot see; say what is actually true.
        return (
            "SKIP_NO_CREDENTIALS",
            "no credentials supplied to this check (none were resolved for it)",
        )
    if len(missing) == len(REQUIRED_KEYS):
        return (
            "SKIP_NO_CREDENTIALS",
            f"no Appwrite credentials in the environment and none in {own} "
            f"(this repo's own .env; other repos' .env files are deliberately not read — #298)",
        )
    return (
        "CREDENTIALS_INCOMPLETE",
        f"credentials are partially configured — missing {', '.join(missing)}. "
        f"Set them in the environment or in {own}",
    )


def resolve_credentials() -> Optional[AppwriteCredentials]:
    """Process env first, then **this repository's own** `.env`. Never another repo's."""
    import os

    env = {key: os.environ.get(key) for key in REQUIRED_KEYS}
    if all(env.values()):
        return AppwriteCredentials(
            env["APPWRITE_ENDPOINT"],              # type: ignore[arg-type]
            env["APPWRITE_DATASTORE_PROJECT_ID"],  # type: ignore[arg-type]
            env["APPWRITE_DATASTORE_API_KEY"],     # type: ignore[arg-type]
        )
    return load_credentials_from_env_file(own_env_file())


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


def assert_bucket_reachable(
    endpoint: str,
    bucket_id: str,
    headers: Dict[str, str],
    fetch: Callable[[str, Dict[str, str]], object],
) -> None:
    """Prove the key is accepted and the bucket resolves, before reading either.

    **Appwrite answers a REJECTED key on the file-listing endpoint with HTTP 200
    and ``total: 0``** — measured 2026-08-02 against Appwrite 1.9.5, with a real
    key (200, total=461), a garbage key (200, total=0) and an empty key (200,
    total=0). Listing files is the only call these surfaces made, so a dead
    credential was indistinguishable from an empty bucket, and the store
    reported ``STORE_IDLE`` / *"bucket contains no files"* — exit 1, "attention" —
    while in fact nothing was authenticated at all.

    That matters beyond tidiness: both Appwrite keys expire around 2026-11-30,
    the write path reports that expiry as success, and these surfaces are the
    detector. A detector that renders the failure it exists to catch as mild
    staleness is not one.

    Every other endpoint tested returns 401 for the same key — bucket get, bucket
    list, database get, collection list, and ``/health``. Getting the **bucket
    itself** is used because it answers two questions in one call: the key is
    accepted (401 if not) and the bucket coordinate still resolves (404 if not).
    A wrong bucket id would otherwise also surface as emptiness, for exactly the
    same reason.

    Raises whatever ``fetch`` raises; callers already turn that into
    ``UNREACHABLE`` (exit 2). Returns nothing — the body is not the point.
    """
    fetch(f"{endpoint}/storage/buckets/{bucket_id}", headers)


def fetch_json(url: str, headers: Dict[str, str]) -> object:
    """Default fetch: stdlib urllib, lazy import, explicit timeout."""
    import json
    import urllib.request

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        return json.load(response)

"""Green tests for the credential schema (.env.example) and the presence checker.

Validates the *mechanism* (schema completeness + the checker's missing/present logic)
deterministically — it does NOT read the ambient environment, so it is stable in CI.
Guards `reports/security/appwrite_credentials_audit.md`'s remediation from regressing.
"""
from pathlib import Path

import pytest

from tools import check_credentials

pytestmark = pytest.mark.green

REPO_ROOT = Path(__file__).resolve().parent.parent

# The credentials run-0 (rusty_bucket --prediction_store) + the un_fao postprocessor need.
_CRITICAL_KEYS = {
    "APPWRITE_ENDPOINT",
    "APPWRITE_DATASTORE_PROJECT_ID",
    "APPWRITE_DATASTORE_API_KEY",
    "APPWRITE_PROD_FORECASTS_BUCKET_ID",
    "APPWRITE_PROD_FORECASTS_BUCKET_NAME",
    "APPWRITE_PROD_FORECASTS_COLLECTION_ID",
    "APPWRITE_PROD_FORECASTS_COLLECTION_NAME",
    "APPWRITE_METADATA_DATABASE_ID",
    "APPWRITE_METADATA_DATABASE_NAME",
    "APPWRITE_UNFAO_BUCKET_ID",
    "APPWRITE_UNFAO_BUCKET_NAME",
    "APPWRITE_UNFAO_COLLECTION_ID",
    "APPWRITE_UNFAO_COLLECTION_NAME",
}


def test_env_example_exists_and_declares_the_canonical_keys():
    example = REPO_ROOT / ".env.example"
    assert example.exists(), ".env.example (the credential schema) must exist"
    declared = set(check_credentials._parse_env_names(example))
    missing = _CRITICAL_KEYS - declared
    assert not missing, f".env.example is missing canonical keys: {sorted(missing)}"


# --- #299: the .gitignore gaps -------------------------------------------------
# This repo is PUBLIC. Before #299 the only literal was `.env`; `.env.bak` and
# `.env.local` were covered incidentally by `*.bak`/`*.local`, and the two shapes most
# likely to exist on a rotation day were not covered at all. These tests pin both
# halves of the fix — the broadening AND the negation that keeps the schema tracked,
# because a `.env.*` rule without `!.env.example` would silently un-track the file the
# checker above reads.

def _is_ignored(name: str) -> bool:
    import subprocess

    return subprocess.run(
        ["git", "check-ignore", "-q", name], cwd=REPO_ROOT
    ).returncode == 0


@pytest.mark.red
@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.faoapi",      # the exact filename the production server uses
        ".env.20260728",    # the shape of a file made on a rotation day
        ".env.save",
        ".env.bak",
        ".env.local",
    ],
)
def test_credential_file_shapes_are_gitignored(name):
    assert _is_ignored(name), (
        f"{name} is NOT gitignored — on a public repo that is one careless `git add` "
        f"from a published credential (#299)"
    )


@pytest.mark.red
def test_env_example_is_NOT_gitignored():
    assert not _is_ignored(".env.example"), (
        ".env.example must stay tracked — it is the credential schema "
        "tools/check_credentials.py reads. If a broad `.env.*` rule is added without "
        "the `!.env.example` negation, this fails (#299)"
    )


def test_parse_env_filled_ignores_comments_blanks_and_empties(tmp_path):
    f = tmp_path / ".env"
    f.write_text(
        "# a comment\n"
        "\n"
        "FILLED=somevalue\n"
        "EMPTY=\n"
        "SPACED =  val \n",
        encoding="utf-8",
    )
    filled = check_credentials._parse_env_filled(f)
    assert filled == {"FILLED", "SPACED"}  # EMPTY (blank value) is not "filled"


def test_checker_flags_missing_and_passes_when_complete(tmp_path, monkeypatch):
    (tmp_path / ".env.example").write_text("KEY_A=\nKEY_B=  # note\n", encoding="utf-8")
    monkeypatch.setattr(check_credentials, "REPO_ROOT", tmp_path)
    # ensure the checker can't be rescued by ambient env vars named KEY_A/KEY_B
    monkeypatch.delenv("KEY_A", raising=False)
    monkeypatch.delenv("KEY_B", raising=False)

    # one filled, one blank -> INCOMPLETE (exit 1)
    (tmp_path / ".env").write_text("KEY_A=value\nKEY_B=\n", encoding="utf-8")
    assert check_credentials.main() == 1

    # both filled -> OK (exit 0)
    (tmp_path / ".env").write_text("KEY_A=value\nKEY_B=value\n", encoding="utf-8")
    assert check_credentials.main() == 0

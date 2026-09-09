"""Guards on the Appwrite Seam Contract coordinate reader (`tools/credentials/registry_to_env.py`).

This file had no tests until 2026-07-31, and the day it went without them a neighbouring
repository unconfigured the FAO delivery path by adding four lines of TOML.

What happened. views-appwrite added `[target.APPWRITE_CRAFD_BUCKET_ID]` with
`status = "planned — views-crafdapi"` and no value, reserving a name for a consumer that
does not exist yet. `coordinates()` raised on the missing value — and because it raises
for the whole registry, `postprocessors/un_fao/run.sh` lost **every** coordinate, not
just the planned one. Since #293 that failure is at least loud; before it, it was
swallowed by `2>/dev/null`.

The distinction these tests pin: a coordinate that *ought* to have a value and does not
is a malformed registry and must fail loud (verdict D5). A coordinate explicitly marked
as a reservation is a declaration of intent and must be skipped. The registry already
distinguishes them; the reader now does too.
"""
import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.green]

MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "tools" / "credentials" / "registry_to_env.py"
)

tomllib = pytest.importorskip(
    "tomllib", reason="registry_to_env needs Python 3.11+ for tomllib (it says so itself)"
)


def _load():
    spec = importlib.util.spec_from_file_location("registry_to_env", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registry(tmp_path: Path, body: str) -> str:
    path = tmp_path / "coordinate_registry.toml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_emits_connection_and_target_coordinates(tmp_path):
    module = _load()
    registry = _registry(tmp_path, """
[connection.APPWRITE_ENDPOINT]
class = "connection"
value = "https://example.test/v1"

[target.APPWRITE_UNFAO_BUCKET_ID]
class = "target"
value = "unfao_bucket"
""")
    assert module.coordinates(registry) == [
        "APPWRITE_ENDPOINT=https://example.test/v1",
        "APPWRITE_UNFAO_BUCKET_ID=unfao_bucket",
    ]


def test_planned_reservation_is_skipped_not_fatal(tmp_path):
    """The live regression: one planned entry must not unconfigure everything else."""
    module = _load()
    registry = _registry(tmp_path, """
[connection.APPWRITE_ENDPOINT]
class = "connection"
value = "https://example.test/v1"

[target.APPWRITE_CRAFD_BUCKET_ID]
class = "target"
status = "planned — views-crafdapi (consumer does not exist yet)"
consumer = "views-crafdapi"
""")
    emitted = module.coordinates(registry)
    assert emitted == ["APPWRITE_ENDPOINT=https://example.test/v1"], (
        "a reserved-but-unbuilt coordinate must be skipped, and the coordinates that DO "
        "have values must still be emitted — a neighbouring repo adding a placeholder "
        "must not be able to unconfigure the FAO delivery path"
    )


@pytest.mark.red
def test_valueless_coordinate_without_planned_status_still_fails_loud(tmp_path):
    """The skip must not become a blanket excuse. Fail-loud is still the default."""
    module = _load()
    registry = _registry(tmp_path, """
[target.APPWRITE_UNFAO_BUCKET_ID]
class = "target"
consumer = "views-postprocessing"
""")
    with pytest.raises(ValueError, match="APPWRITE_UNFAO_BUCKET_ID"):
        module.coordinates(registry)


@pytest.mark.red
def test_secret_slots_are_never_emitted(tmp_path):
    """The reader emits coordinates only; secrets stay operator slots (The Appwrite Seam Contract §5)."""
    module = _load()
    registry = _registry(tmp_path, """
[connection.APPWRITE_ENDPOINT]
class = "connection"
value = "https://example.test/v1"

[secret.APPWRITE_DATASTORE_API_KEY]
class = "secret"
value = "must-never-be-emitted"
""")
    emitted = module.coordinates(registry)
    assert emitted == ["APPWRITE_ENDPOINT=https://example.test/v1"]
    assert not any("API_KEY" in line for line in emitted)


def test_the_real_platform_registry_reads_cleanly_if_present(tmp_path):
    """Integration: the actual sibling registry, if this checkout has one beside it."""
    module = _load()
    real = (
        Path(__file__).resolve().parents[2]
        / "views-appwrite" / "docs" / "ADRs" / "platform" / "coordinate_registry.toml"
    )
    if not real.exists():
        pytest.skip(f"no sibling views-appwrite checkout at {real}")
    emitted = module.coordinates(str(real))
    assert emitted, "the real registry emitted no coordinates at all"
    assert all("=" in line for line in emitted)
    assert not any("API_KEY" in line for line in emitted), "a secret leaked into the output"

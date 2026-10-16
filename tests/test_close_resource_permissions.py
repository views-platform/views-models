"""Guards on the Appwrite resource-permission auditor
(`tools/credentials/close_resource_permissions.py`).

The script closed a live hole: `unfao` and `production_forecasts` were readable, writable
and deletable by any unauthenticated caller holding the project ID. It stays as the
regression guard, because nothing else on the platform inspects a resource's permission
list — views-pipeline-core's C-292 says so in as many words: *"No test inspects the
argument."*

**So this file exists to keep the guard itself honest.** These tests do not talk to
Appwrite. They pin the one piece of logic that can do damage — the read-modify-write in
`_close_collection` — because `PUT /databases/{db}/collections/{id}` resets every optional
parameter it is not given. A PUT that carries only `permissions` renames the collection
and flips `documentSecurity`, which is a worse outcome than the exposure being fixed.

The refusal case is the reason this file was written rather than assumed. `None` is not a
safe stand-in for "unchanged": if the GET ever stops returning `enabled`, the naive code
sends `"enabled": None` and *writes* the configuration change the function exists to
prevent. That branch had never executed against anything when it was added.
"""
import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.green]

MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tools" / "credentials" / "close_resource_permissions.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("close_resource_permissions", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def _healthy_before():
    """What `_audit_collection` returns for a collection that is safe to rewrite."""
    return {
        "id": "unfao",
        "name": "UNFAO File Metadata",
        "permissions": ['read("any")', 'update("any")'],
        "documentSecurity": False,
        "enabled": True,
        "keyed_total": 111,
        "anonymous": "HTTP 200, total=111",
    }


class TestTheWriteIsRefusedRatherThanGuessed:
    """A field the GET did not return must stop the write, not be sent as `None`."""

    @pytest.mark.parametrize("absent", ["name", "documentSecurity", "enabled"])
    def test_a_missing_preserved_field_refuses_without_calling_the_api(self, mod, absent):
        before = _healthy_before()
        before[absent] = None

        def explode(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("the API was called despite a missing preserved field")

        mod._call = explode
        try:
            wrote, problems = mod._close_collection("https://ep/v1", "db", {}, before)
        finally:
            mod._call = _load()._call

        assert wrote is False, "a refusal must not report itself as a write"
        assert problems and absent in problems[0]

    def test_documentSecurity_false_is_a_value_and_not_an_absence(self, mod):
        """`False` and `None` are different answers.

        A falsy check here would refuse every collection on this platform — all three run
        with `documentSecurity: false` — turning the safety guard into a total outage of
        the tool. The distinction is `is None`, not truthiness.
        """
        before = _healthy_before()
        before["documentSecurity"] = False
        before["enabled"] = True
        sent = {}

        def capture(method, url, headers, body=None):
            sent["method"], sent["body"] = method, body
            return {**{f: before[f] for f in mod.PRESERVED_FIELDS}, "$permissions": []}

        mod._call = capture
        try:
            wrote, problems = mod._close_collection("https://ep/v1", "db", {}, before)
        finally:
            mod._call = _load()._call

        assert wrote is True and problems == []
        assert sent["method"] == "PUT"


class TestTheWriteCarriesEverythingThePutWouldReset:
    def test_every_preserved_field_is_sent_back_unchanged(self, mod):
        before = _healthy_before()
        sent = {}

        def capture(method, url, headers, body=None):
            sent.update(body or {})
            return {**{f: before[f] for f in mod.PRESERVED_FIELDS}, "$permissions": []}

        mod._call = capture
        try:
            mod._close_collection("https://ep/v1", "db", {}, before)
        finally:
            mod._call = _load()._call

        assert sent["permissions"] == [], "the one field this tool exists to change"
        for field in mod.PRESERVED_FIELDS:
            assert sent[field] == before[field], f"{field} was not preserved"


class TestDriftIsDetectedRatherThanAssumedAway:
    """A 200 is not proof the write did what was asked."""

    @pytest.mark.parametrize("field,corrupted", [
        ("name", "renamed-by-the-put"),
        ("documentSecurity", True),
        ("enabled", False),
    ])
    def test_a_changed_field_in_the_response_is_reported_as_drift(
        self, mod, field, corrupted
    ):
        before = _healthy_before()

        def capture(method, url, headers, body=None):
            echoed = {f: before[f] for f in mod.PRESERVED_FIELDS}
            echoed[field] = corrupted
            return {**echoed, "$permissions": []}

        mod._call = capture
        try:
            wrote, problems = mod._close_collection("https://ep/v1", "db", {}, before)
        finally:
            mod._call = _load()._call

        assert wrote is True, "the write happened; that is why it needs repairing"
        assert any(field in p for p in problems)

    def test_permissions_that_survive_the_write_are_reported(self, mod):
        before = _healthy_before()

        def capture(method, url, headers, body=None):
            return {**{f: before[f] for f in mod.PRESERVED_FIELDS},
                    "$permissions": ['read("any")']}

        mod._call = capture
        try:
            wrote, problems = mod._close_collection("https://ep/v1", "db", {}, before)
        finally:
            mod._call = _load()._call

        assert wrote is True
        assert any("not emptied" in p for p in problems)

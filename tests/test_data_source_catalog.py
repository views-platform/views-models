"""The catalog says which data source each model reaches (#474), and the split is what #473 measured.

``tools/catalogs/data_source.py`` is the one reader (C-147 shape). Two kinds of test:

1. **Each branch on a synthetic file** — a queryset importing viewser; one importing
   datafactory_query; the synthetic descriptor; both clients (``unknown``, never a guess);
   neither (``unknown``); no file (``none``).
2. **The fleet pin** — a characterisation of today's split, updated deliberately when a model
   migrates, like ``test_environment_sharing``'s tenant counts. #473 §2 measured 77 viewser /
   29 datafactory-or-synthetic on 2026-09-17; #491 added eleven pgm datafactory models.
"""

from collections import Counter
import pytest

from tests.conftest import ALL_MODEL_DIRS
from tools.catalogs.data_source import DATAFACTORY, NONE, SYNTHETIC, UNKNOWN, VIEWSER, data_source_of

@pytest.mark.parametrize(
    "source,expected",
    [
        ("from viewser import Queryset, Column\ndef generate():\n    return Queryset('x', 'cm')\n", VIEWSER),
        ("import viewser\n", VIEWSER),
        ("from datafactory_query.defaults import DEFAULT_REMOTE\ndef generate():\n    return {}\n", DATAFACTORY),
        ("def generate():\n    return {'source': 'synthetic', 'pattern': 'diagonal_gradient'}\n", SYNTHETIC),
        ("from viewser import Queryset\nfrom datafactory_query.defaults import DEFAULT_REMOTE\n", UNKNOWN),
        ("import os\ndef generate():\n    return {'source': 'somewhere_else'}\n", UNKNOWN),
        ("from .viewser import x\n", UNKNOWN),  # relative import is not the client
        # a stray dict elsewhere in the file is not generate()'s answer
        ("from datafactory_query.defaults import DEFAULT_REMOTE\nNOTE = {'source': 'synthetic'}\ndef generate():\n    return {'source': 'views-datafactory'}\n", DATAFACTORY),
    ],
    ids=["viewser-from", "viewser-import", "datafactory", "synthetic", "both-clients", "neither", "relative", "stray-dict"],
)
def test_each_branch_on_a_synthetic_file(tmp_path, source, expected):
    q = tmp_path / "config_queryset.py"
    q.write_text(source)
    assert data_source_of(q) == expected


def test_no_file_is_none(tmp_path):
    assert data_source_of(tmp_path / "config_queryset.py") == NONE


def _fleet():
    return {d.name: data_source_of(d / "configs" / "config_queryset.py") for d in ALL_MODEL_DIRS}


def test_no_model_is_unknown():
    fleet = _fleet()
    unknown = sorted(n for n, s in fleet.items() if s == UNKNOWN)
    assert not unknown, f"the classifier could not place these — look, do not guess: {unknown}"


def test_the_split_is_what_the_catalog_says():
    """Characterisation pin. When a model migrates, change the numbers here on purpose and
    say so in the commit (#473 §7-8 is the plan for the 77)."""
    counts = Counter(_fleet().values())
    assert dict(counts) == {VIEWSER: 77, DATAFACTORY: 34, SYNTHETIC: 6}, dict(counts)

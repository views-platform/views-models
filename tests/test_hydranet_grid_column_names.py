"""Every HydraNet must name the grid column the data factory actually emits.

⛔ Seven of twenty-seven named `priogrid_gid` while the factory emits `priogrid_id`, and all
seven were members of `rusty_bucket`, the FAO delivery ensemble. It went unnoticed because
every one of them last ran in September, before the naming settled; the twenty that run
regularly were all correct (views-models#698).

**The key is read, not decorative.** `views_hydranet/utils/data_sniffer.py:165` takes
`config["id_col"]` and `:168-169` indexes the frame with it, and `DataSniffer` is constructed
at `hydranet_manager.py:134` — it ran 20 times in a real 2026-11-03 run log.

**Measured, on a delivered raw pull** (`models/calm_orbiter/data/raw/*.parquet`,
24,890,112 rows): `index.names == ['month_id', 'priogrid_id']`, columns
`['lr_sb_best','lr_ns_best','lr_os_best','c_id','row','col']`. `priogrid_gid` appears nowhere.

⚠️ This asserts AGREEMENT ACROSS MODELS, not the literal string. A platform-wide rename would
move all twenty-seven together and should not fail here; one model drifting away from the rest
is the failure this exists to catch. The expected value is derived from the majority rather
than hardcoded, so the test states the invariant instead of a constant that would itself go
stale — which is how the seven got here.
"""
from __future__ import annotations

import glob
import importlib.util
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.beige

REPO = Path(__file__).resolve().parents[1]


def _load(path: Path, fn: str):
    spec = importlib.util.spec_from_file_location(f"m{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, fn)()


def _hydranets() -> list[str]:
    return sorted(
        Path(p).parts[-3]
        for p in glob.glob(str(REPO / "models" / "*" / "configs" / "config_meta.py"))
        if '"algorithm": "HydraNet"' in Path(p).read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("key", ["id_col", "index_names", "identity_cols"])
def test_every_hydranet_names_the_grid_column_the_same_way(key):
    """One model drifting from the other twenty-six is the failure; a joint rename is not."""
    names = _hydranets()
    assert len(names) >= 20, f"only {len(names)} HydraNets found — the discovery is broken"

    values = {
        n: repr(_load(REPO / "models" / n / "configs" / "config_hyperparameters.py",
                      "get_hp_config").get(key))
        for n in names
    }
    counts = Counter(values.values())
    majority, n_majority = counts.most_common(1)[0]
    odd = sorted(n for n, v in values.items() if v != majority)

    assert not odd, (
        f"{len(odd)} HydraNet(s) declare a different {key!r} from the other {n_majority}:\n"
        f"  the {n_majority}: {majority}\n"
        + "".join(f"  {n}: {values[n]}\n" for n in odd)
        + "\n`id_col` is read at data_sniffer.py:165 and used to index the frame at :168-169, "
        "so a name the factory does not emit is a KeyError at startup — after the clone and "
        "the install, on a rented pod. The factory emits index.names == "
        "['month_id', 'priogrid_id'] (measured 2026-11-03, views-models#698)."
    )


def test_the_agreed_name_is_one_the_factory_actually_emits():
    """Agreement is necessary and not sufficient — all twenty-seven could agree on a wrong name.

    Pinned against a delivered raw pull when one is on disk. ⚠️ It SKIPS when none is, rather
    than passing: a check that cannot see its subject must not report success (ADR-025).
    """
    pulls = sorted(glob.glob(str(REPO / "models" / "*" / "data" / "raw" / "*datafactory_df.parquet")))
    if not pulls:
        pytest.skip("no delivered raw pull on disk to compare against")

    import pandas as pd

    index_names = list(pd.read_parquet(pulls[0]).index.names)
    names = _hydranets()
    declared = _load(REPO / "models" / names[0] / "configs" / "config_hyperparameters.py",
                     "get_hp_config")["index_names"]
    assert list(declared) == index_names, (
        f"every HydraNet declares index_names={declared}, but the delivered pull at "
        f"{Path(pulls[0]).relative_to(REPO)} carries {index_names}. They agree with each "
        f"other and disagree with the data."
    )

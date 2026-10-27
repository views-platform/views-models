"""The member-order pairing that epic #595's reconciliation rests on (S3, #598).

`aggregation: "concat"` lays member *k*'s draws at sample-axis positions 128(k-1)..128k-1,
and `reconcile_frames` runs in ALIGNED_DRAWS mode, which scales grid draw *i* to country
draw *i*. Composing those two facts: **position k in chrome_box's modelset must hold the
same setting as position k in chrome_crate's**, or draws are reconciled against unrelated
models' country totals. Nothing fails at run time when they diverge — the shapes still
match, the sums still balance per draw, and a plausible wrong number comes out. The order
is therefore a correctness invariant with no runtime guard, which is why it has this one.

⛔ **#598's text says the pairing key is `(family, window_months, lambda_mix)`. There is no
`family` key in this repo.** The family is `algorithm` in `config_meta.py`;
`window_months` and `lambda_mix` live in `config_hyperparameters.py`; and `lambda_mix` is
**absent** on the four `ConflictologyModel` members, so it is read with `.get()` and
compared as `None`. The triple therefore spans two files per model, and this module reads
the real configs rather than restating them — except for `PAIRS`, which is the declared
table the configs are checked against.

Reading discipline, learned from this repo's own surviving mutations:

- **Never through `set()`.** `EXPERIMENT_ARMS` was only ever read as `set(...)` or indexed
  by name, so re-pointing a control stayed green. Order is the property here, so every
  comparison is a list or an indexed pair.
- **Fail, do not skip, on a missing config.** `pytest.skip` reports green-by-silence at
  exactly the moment the signal matters (C-113). `tests/test_falsification_synthetic.py`
  skips when a modelset is absent and would go green if one vanished; this does not.
"""
from __future__ import annotations

import pytest

from tests.conftest import ENSEMBLES_DIR, MODELS_DIR, load_config_module

pytestmark = pytest.mark.green

PGM_ENSEMBLE = "chrome_box"
CM_ENSEMBLE = "chrome_crate"

#: Epic #595's table: the eight settings, in the order the two modelsets must list them.
#: `(pgm member, cm member, algorithm, window_months, lambda_mix)`. `None` for lambda_mix
#: means the key is absent, which is what the four Conflictology members declare — not
#: that it is present and null.
PAIRS: list[tuple[str, str, str, int, float | None]] = [
    ("ashen_sparrow", "ashen_condor", "ConflictologyModel", 12, None),
    ("brisk_sparrow", "brisk_condor", "ConflictologyModel", 18, None),
    ("candid_sparrow", "candid_condor", "ConflictologyModel", 24, None),
    ("dusky_sparrow", "dusky_condor", "ConflictologyModel", 36, None),
    ("eager_sparrow", "eager_condor", "MixtureBaseline", 18, 0.1),
    ("faint_sparrow", "faint_condor", "MixtureBaseline", 36, 0.1),
    ("gentle_sparrow", "gentle_condor", "MixtureBaseline", 18, 0.3),
    ("humble_sparrow", "humble_condor", "MixtureBaseline", 36, 0.3),
]

PGM_MEMBERS = [pgm for pgm, _cm, _a, _w, _lm in PAIRS]
CM_MEMBERS = [cm for _pgm, cm, _a, _w, _lm in PAIRS]


def _config(directory, filename, getter):
    """Load one config and call its getter. Fails loud when the file is missing: a member
    that has been renamed or half-applied must turn the suite red, not skip."""
    path = directory / "configs" / filename
    if not path.exists():
        pytest.fail(f"{path} does not exist — an epic #595 member or ensemble is missing")
    return getattr(load_config_module(path), getter)()


def _setting(model_name):
    """The `(algorithm, window_months, lambda_mix)` triple, read from the two files that
    actually hold it. `lambda_mix` via `.get()` — absent on the Conflictology four."""
    model_dir = MODELS_DIR / model_name
    meta = _config(model_dir, "config_meta.py", "get_meta_config")
    hp = _config(model_dir, "config_hyperparameters.py", "get_hp_config")
    return (meta["algorithm"], hp["window_months"], hp.get("lambda_mix"))


def _modelset(ensemble_name):
    return _config(ENSEMBLES_DIR / ensemble_name, "config_modelset.py", "get_modelset_config")["models"]


def _level(name, base_dir):
    return _config(base_dir / name, "config_meta.py", "get_meta_config")["level"]


class TestTheDeclaredTable:
    def test_the_table_is_exactly_epic_595s_eight_settings(self):
        """Pinned in both directions, so editing the table is a reviewed edit rather than
        a silent one. The settings are the epic's: four Conflictology windows, then a 2x2
        of MixtureBaseline over {18, 36} x {0.1, 0.3}."""
        assert [(a, w, lm) for _p, _c, a, w, lm in PAIRS] == [
            ("ConflictologyModel", 12, None),
            ("ConflictologyModel", 18, None),
            ("ConflictologyModel", 24, None),
            ("ConflictologyModel", 36, None),
            ("MixtureBaseline", 18, 0.1),
            ("MixtureBaseline", 36, 0.1),
            ("MixtureBaseline", 18, 0.3),
            ("MixtureBaseline", 36, 0.3),
        ]

    def test_the_eight_settings_are_distinct(self):
        """⛔ Without this, element-wise equality is weaker than it reads: if two pairs
        shared a setting, swapping those two members in one modelset would leave every
        per-pair comparison equal and the pairing still wrong. Distinctness is what makes
        the triple a usable key."""
        settings = [(a, w, lm) for _p, _c, a, w, lm in PAIRS]
        assert len(set(settings)) == len(settings), f"duplicate settings: {settings}"

    def test_the_table_names_sixteen_distinct_members(self):
        assert len(set(PGM_MEMBERS + CM_MEMBERS)) == 16


class TestTheModelsetsMatchTheTable:
    def test_the_pgm_ensemble_lists_the_pgm_members_in_order(self):
        """List equality, never set equality — the order IS the property."""
        assert _modelset(PGM_ENSEMBLE) == PGM_MEMBERS

    def test_the_cm_ensemble_lists_the_cm_members_in_order(self):
        assert _modelset(CM_ENSEMBLE) == CM_MEMBERS

    def test_the_two_modelsets_are_the_same_length(self):
        assert len(_modelset(PGM_ENSEMBLE)) == len(_modelset(CM_ENSEMBLE)) == 8


class TestEachPositionIsPairedInTheRealConfigs:
    @pytest.mark.parametrize("pgm,cm", [(p, c) for p, c, _a, _w, _lm in PAIRS])
    def test_the_twin_at_this_position_has_the_same_setting(self, pgm, cm):
        """Read from the configs on disk, not from PAIRS: this is the guard that catches a
        member whose `window_months` or `lambda_mix` was edited on one side only."""
        assert _setting(pgm) == _setting(cm), (
            f"{pgm} and {cm} occupy the same position in the two modelsets but declare "
            f"different settings: {_setting(pgm)} vs {_setting(cm)}. ALIGNED_DRAWS would "
            f"scale this grid model's draws to an unrelated country model's total."
        )

    @pytest.mark.parametrize("pgm,cm,algorithm,window,lam", PAIRS)
    def test_the_configs_declare_what_the_table_says(self, pgm, cm, algorithm, window, lam):
        """The other direction: the table is not free to drift from the configs either."""
        assert _setting(pgm) == (algorithm, window, lam)
        assert _setting(cm) == (algorithm, window, lam)

    @pytest.mark.parametrize("pgm,cm", [(p, c) for p, c, _a, _w, _lm in PAIRS])
    def test_the_pair_sits_at_the_same_index_in_both_modelsets(self, pgm, cm):
        """Position, stated as position. The two guards above would both pass if both
        lists were permuted the same way; this one is about the index itself."""
        pgm_set, cm_set = _modelset(PGM_ENSEMBLE), _modelset(CM_ENSEMBLE)
        assert pgm_set.index(pgm) == cm_set.index(cm)


class TestTheLevelsAreNotCrossed:
    def test_every_pgm_member_is_pgm(self):
        assert [_level(m, MODELS_DIR) for m in PGM_MEMBERS] == ["pgm"] * 8

    def test_every_cm_member_is_cm(self):
        """⛔ `_sparrow` is pgm and `_condor` is cm. Read from `config_meta.py`, not
        inferred from the suffix — the suffix convention is a naming habit and this is
        the thing that would make the reconciliation meaningless if it were crossed."""
        assert [_level(m, MODELS_DIR) for m in CM_MEMBERS] == ["cm"] * 8

    def test_the_ensembles_declare_the_levels_their_members_do(self):
        assert _level(PGM_ENSEMBLE, ENSEMBLES_DIR) == "pgm"
        assert _level(CM_ENSEMBLE, ENSEMBLES_DIR) == "cm"


class TestTheSweepConfigsDoNotDriftFromTheHyperparameters:
    """`config_sweep.py` carries a THIRD copy of `window_months` and `lambda_mix`, under
    `parameters[k]["value"]`. Nothing covered it before this, and the members' own sweep
    docstring states the risk: *"A sweep over a member of a reconciled ensemble would
    break the pairing the ALIGNED_DRAWS mode relies on."* A sweep that moved one side's
    window would do exactly that, and the pairing guards above read the hp file only.
    """

    @pytest.mark.parametrize("model", PGM_MEMBERS + CM_MEMBERS)
    def test_the_sweep_pins_the_same_window_and_lambda_as_the_hyperparameters(self, model):
        hp = _config(MODELS_DIR / model, "config_hyperparameters.py", "get_hp_config")
        sweep = _config(MODELS_DIR / model, "config_sweep.py", "get_sweep_config")
        params = sweep["parameters"]
        for key in ("window_months", "lambda_mix"):
            if key not in hp:
                assert key not in params, (
                    f"{model}: config_sweep declares {key!r} that config_hyperparameters "
                    f"does not — a sweep would introduce a setting the pairing never saw"
                )
                continue
            assert key in params, f"{model}: config_sweep drops {key!r}"
            assert params[key] == {"value": hp[key]}, (
                f"{model}: config_sweep pins {key}={params[key]} but config_hyperparameters "
                f"says {hp[key]}. A sweep run would break the #595 pairing silently."
            )


def test_every_pair_guard_runs_over_every_declared_pair():
    """⛔ A meta-guard, because narrowing a `parametrize` decorator was a surviving
    mutation in this repository before (`e874be0f`): the guards stay green while covering
    fewer pairs. Discovers the per-pair guards by their argnames and pins the count, so
    dropping a pair from any decorator fails here even if every remaining case passes.
    """
    import inspect

    expected = len(PAIRS)
    checked = {}
    for cls in (TestEachPositionIsPairedInTheRealConfigs,):
        for name, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith("test_"):
                continue
            marks = [m for m in getattr(fn, "pytestmark", []) if m.name == "parametrize"]
            assert marks, f"{name} is not parametrized — it cannot be per-pair"
            (argnames, argvalues), *_ = [(m.args[0], m.args[1]) for m in marks]
            checked[name] = len(argvalues)

    assert checked, "no per-pair guards were discovered — this meta-guard found nothing"
    assert all(n == expected for n in checked.values()), (
        f"a per-pair guard does not cover all {expected} pairs: {checked}"
    )
    assert len(checked) == 3, (
        f"expected three per-pair guards, found {sorted(checked)} — a guard was added or "
        f"removed without updating this count, so 'every guard' is no longer pinned"
    )

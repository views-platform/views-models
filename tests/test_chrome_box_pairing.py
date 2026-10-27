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

import ast

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


def _classes_instantiated_in_main(ensemble_dir):
    """The names actually CALLED in `main.py`, read from the AST.

    A comment, a docstring and an import all mention a class without calling it, and all
    three satisfy a text search. Only a `Call` node is an instantiation.
    """
    main_py = ensemble_dir / "main.py"
    if not main_py.exists():
        pytest.fail(f"{main_py} does not exist — an epic #595 ensemble is missing")
    tree = ast.parse(main_py.read_text(encoding="utf-8"))
    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            called.add(func.id)
        elif isinstance(func, ast.Attribute):
            called.add(func.attr)
    return called


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
    """`config_sweep.py` carries a THIRD copy of the settings, under
    `parameters[k]["value"]`. Nothing covered it before this, and the members' own sweep
    docstring states the risk: *"A sweep over a member of a reconciled ensemble would
    break the pairing the ALIGNED_DRAWS mode relies on."* A sweep that moved one side's
    window would do exactly that, and the pairing guards above read the hp file only.

    ⛔ **Every shared key, not a named list of two.** This enumerated
    `("window_months", "lambda_mix")`, which left `n_samples` free to drift: dropping a
    member's sweep `n_samples` to 64 against an hp 128 was a surviving mutation on the
    whole suite. `n_samples` is load-bearing for the same reason the window is —
    `chrome_crate/main.py` says it in its own comment: *"the mode is derived from this
    frame's sample_count, so thinning here silently changes chrome_box's mode"*, i.e.
    ALIGNED_DRAWS degrades to POINT_BROADCAST. A list of keys written by hand covers the
    keys its author thought of; the intersection covers the ones that exist.
    """

    #: Sweep-only keys, which have no hyperparameter counterpart and are not drift.
    SWEEP_ONLY = frozenset({"method", "name", "metric", "parameters"})

    @pytest.mark.parametrize("model", PGM_MEMBERS + CM_MEMBERS)
    def test_the_sweep_pins_every_shared_setting_as_the_hyperparameters_do(self, model):
        hp = _config(MODELS_DIR / model, "config_hyperparameters.py", "get_hp_config")
        sweep = _config(MODELS_DIR / model, "config_sweep.py", "get_sweep_config")
        params = sweep["parameters"]

        shared = sorted(set(hp) & set(params))
        assert shared, (
            f"{model}: config_sweep's parameters and config_hyperparameters share no key "
            f"at all — this guard would be checking nothing. sweep={sorted(params)}, "
            f"hp={sorted(hp)}"
        )
        for key in shared:
            assert params[key] == {"value": hp[key]}, (
                f"{model}: config_sweep pins {key}={params[key]!r} but "
                f"config_hyperparameters says {hp[key]!r}. A sweep run would break the "
                f"#595 pairing silently."
            )

        # The other direction: a sweep that introduces a setting the hp file never
        # declared, so the pairing was never checked against it.
        introduced = sorted(set(params) - set(hp) - self.SWEEP_ONLY)
        assert not introduced, (
            f"{model}: config_sweep declares {introduced} that config_hyperparameters "
            f"does not — a sweep would introduce settings the pairing never saw"
        )


class TestTheManagerIsTheDrawAwareOne:
    """⛔ Both ensembles must instantiate `PredictionFrameEnsembleManager`, and this is
    read from the AST, not from the source text.

    The repository already believed it was covered and was not. `chrome_box`'s own
    `config_meta` says the DataFrame path *"REFUSES `pgm_cm` rather than implement it"*
    and its `main.py` says publishing there *"would return unreconciled predictions"* —
    yet switching both ensembles to `EnsembleManager` left the entire suite green. The
    only thing that looked at the manager choice was
    `tests/test_pfe_production_readiness.py::_discover_pfe_ensembles`, which **finds its
    subjects by grepping `main.py` for the class name** — so an ensemble that stops using
    the class also stops being checked, and the ⛔ comment naming the class in the source
    was what held `chrome_box` in scope. Deleting that comment removed it from the census
    and nothing went red.

    A comment is invisible to `ast`, so the string cannot satisfy this guard, and the
    absence assertion means leaving a dead `PredictionFrameEnsembleManager(...)` call
    beside a live `EnsembleManager(...)` does not either.
    """

    @pytest.mark.parametrize("ensemble", [PGM_ENSEMBLE, CM_ENSEMBLE])
    def test_main_instantiates_the_prediction_frame_manager(self, ensemble):
        instantiated = _classes_instantiated_in_main(ENSEMBLES_DIR / ensemble)
        assert "PredictionFrameEnsembleManager" in instantiated, (
            f"{ensemble}/main.py does not call PredictionFrameEnsembleManager(...) — it "
            f"calls {sorted(instantiated)}. The DataFrame path refuses `pgm_cm` and would "
            f"publish unreconciled predictions; draw-aware reconciliation lives only on "
            f"the PredictionFrame path."
        )

    @pytest.mark.parametrize("ensemble", [PGM_ENSEMBLE, CM_ENSEMBLE])
    def test_main_does_not_also_instantiate_the_dataframe_manager(self, ensemble):
        instantiated = _classes_instantiated_in_main(ENSEMBLES_DIR / ensemble)
        assert "EnsembleManager" not in instantiated, (
            f"{ensemble}/main.py instantiates EnsembleManager as well as the "
            f"PredictionFrame one. Which manager actually runs is then a question about "
            f"statement order, and the DataFrame one refuses this ensemble's `pgm_cm`."
        )


def test_every_pair_guard_runs_over_every_declared_pair():
    """⛔ A meta-guard, because narrowing a `parametrize` decorator was a surviving
    mutation in this repository before (`e874be0f`): the guards stay green while covering
    fewer pairs. Discovers the per-pair guards by their argnames and compares what each
    decorator actually carries against `PAIRS`.

    ⛔ **Content, and marks — not a length.** The first form of this guard asserted
    `len(argvalues) == len(PAIRS)`, which is a count, which is the same weakness it was
    written to catch. Two mutations survived it:

    - `[...][:7] + [(PAIRS[0][0], PAIRS[0][1])]` — eight argvalues, seven distinct pairs,
      `humble_sparrow`'s pair silently uncovered and `ashen_sparrow`'s run twice.
    - `pytest.param(..., marks=pytest.mark.skip("flaky on the pod image"))` — eight
      argvalues, one pair **skipped**, the file reporting "49 passed, 1 skipped".

    The second is the sharper one: this module's own reading discipline says *"Fail, do
    not skip… `pytest.skip` reports green-by-silence at exactly the moment the signal
    matters (C-113)"*, and a mark is exactly how that is done without touching the body.
    So the values are compared element-wise and in order, and any mark on a param is a
    failure here rather than a silent subtraction from coverage.
    """
    import inspect

    #: What a decorator must carry, keyed by how many argnames it declares. A guard with
    #: a shape not listed here FAILS: an unrecognised arity means this meta-guard cannot
    #: tell what full coverage would be, and a check that cannot see its subject refuses
    #: (ADR-025) rather than waving the guard through.
    expected_by_arity = {
        2: [(pgm, cm) for pgm, cm, _a, _w, _lm in PAIRS],
        5: [tuple(row) for row in PAIRS],
    }

    checked = {}
    for cls in (TestEachPositionIsPairedInTheRealConfigs,):
        for name, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith("test_"):
                continue
            marks = [m for m in getattr(fn, "pytestmark", []) if m.name == "parametrize"]
            assert marks, f"{name} is not parametrized — it cannot be per-pair"
            assert len(marks) == 1, (
                f"{name} carries {len(marks)} parametrize decorators; this meta-guard "
                f"reads one and would not see the others' coverage"
            )
            argnames, argvalues = marks[0].args[0], marks[0].args[1]
            arity = len(argnames.split(","))
            assert arity in expected_by_arity, (
                f"{name} parametrizes {argnames!r}, a shape this meta-guard does not "
                f"know how to check against PAIRS — add it to expected_by_arity rather "
                f"than leaving the guard unpinned"
            )

            values = []
            for item in argvalues:
                # pytest.param(...) wraps the values and carries the marks. A skip or
                # xfail mark removes the case from coverage without changing the length.
                # ParameterSet is itself a NamedTuple, so this test must come before the
                # plain-tuple one. Read by attribute rather than by importing pytest's
                # private class — and if those attributes were ever renamed, the
                # `tuple(item)` fallback yields its 3 fields and the comparison below
                # goes red, which is the safe direction for a guard to fail in.
                if hasattr(item, "marks") and hasattr(item, "values"):
                    assert not item.marks, (
                        f"{name} marks the param {item.values} with "
                        f"{[m.name for m in item.marks]} — that pair is no longer "
                        f"covered, and this module fails rather than skips (C-113)"
                    )
                    values.append(tuple(item.values))
                else:
                    values.append(tuple(item) if isinstance(item, (tuple, list)) else (item,))

            assert values == expected_by_arity[arity], (
                f"{name}'s parametrize does not carry PAIRS. It covers {values}, "
                f"expected {expected_by_arity[arity]} — a pair was dropped, duplicated "
                f"or reordered, and the remaining cases can all still pass."
            )
            checked[name] = arity

    assert checked, "no per-pair guards were discovered — this meta-guard found nothing"
    assert len(checked) == 3, (
        f"expected three per-pair guards, found {sorted(checked)} — a guard was added or "
        f"removed without updating this count, so 'every guard' is no longer pinned"
    )

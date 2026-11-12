"""A model's `num_samples` and its declared metric kind must agree (#536).

**The failure this prevents costs a whole training run and writes nothing.**

views-evaluation chooses which metric list to read from the **data**, not from the config:

    pred_type = "sample" if ef.is_sample else "point"     # native_evaluator.py:258
    metrics_list = self.config.get(f"{task}_{pred_type}_metrics", [])

where `is_sample` is `n_samples > 1`. If the list it lands on is empty it raises:

    No metrics configured for (regression, point). The frame for target 'lr_ged_sb' has
    1 sample(s) per row, so it is a 'point' evaluation and requires a non-empty
    'regression_point_metrics' list in the config.

That raise happens **after** training completes and after every rolling origin has been
predicted — the same shape of late failure as #517, where a missing Appwrite extra was
discovered only at publish time, hours in. On a rented pod that is the whole cost of the run.

**Why a guard rather than care.** `little_talks` and `mister_bluesky` declare
`num_samples: 100` with their point metrics commented out. That pairing is correct. But #536
lowers them to 1 for the point-prediction delivery (epic #532), and lowering the count without
re-activating the point metrics produces exactly the raise above. Nothing in this suite noticed
that relationship before this file: the sample count lives in `config_hyperparameters.py` and
the metric lists in `config_meta.py`, so no single-file check can see it.

**Scope.** Every model declaring `num_samples`, whatever engine. The invariant belongs to the
hyperparameter, not to r2darts2 — it would hold for any engine whose predictions carry a
sample axis. It happens to select the same 42 models as
`test_darts_entity_id_matches_level._darts_models()` today; that is a coincidence of the
current roster, not a shared predicate, so this is **not** the "fourth caller" that file names
as the trigger for extracting a shared helper.

**What this does not check:** whether the metrics named are *implemented*, or whether a
sample metric over a handful of draws is statistically meaningful. views-evaluation raises on
an unknown metric name (ADR-013), and the second question is a modelling judgement.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _config(directory: Path, name: str, getter: str) -> dict:
    """Compile the config from SOURCE, deliberately not via `spec_from_file_location`.

    The idiomatic loader in this suite is
    `importlib.util.spec_from_file_location(...)` + `exec_module`, and it reads
    **`__pycache__`** when the cached bytecode's recorded size and mtime still match the
    source. Config edits routinely keep the size identical — `"num_samples": 1,` and
    `"num_samples": 5,` are the same length, as are `"level": "pgm"` and `"level": "cm"` —
    so an edit made within the same mtime granularity can be invisible to every test that
    loads configs this way.

    This was not theoretical. Mutation-testing this file set `dark_river`'s `num_samples`
    to 5, restored the source, and verified the restoration by sha256 — and the next run
    still saw 5, because the stale `.pyc` was reused. The source was byte-identical and the
    test was reading something else.

    `compile()` on the text we just read cannot do that. Safe here because these two
    configs are plain dicts with no imports; a config that imports (`config_queryset`
    reaches `datafactory_query`) would need the module machinery and the cache risk with it.
    """
    path = directory / "configs" / f"config_{name}.py"
    namespace: dict = {"__file__": str(path), "__name__": f"_cfg_{name}_{directory.name}"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)  # noqa: S102
    return namespace[getter]()


def _models_with_a_sample_count():
    for directory in sorted((REPO_ROOT / "models").glob("*")):
        hp_path = directory / "configs" / "config_hyperparameters.py"
        meta_path = directory / "configs" / "config_meta.py"
        if not (hp_path.is_file() and meta_path.is_file()):
            continue
        try:
            hp = _config(directory, "hyperparameters", "get_hp_config")
            meta = _config(directory, "meta", "get_meta_config")
        except Exception:  # noqa: BLE001 — reported by the config-completeness suite
            continue
        if "num_samples" not in hp:
            continue
        yield directory.name, hp["num_samples"], meta


SAMPLED = list(_models_with_a_sample_count())


def test_the_check_is_not_vacuous():
    """A parametrized test over an empty list passes while checking nothing."""
    assert len(SAMPLED) >= 40, (
        f"only {len(SAMPLED)} models declare num_samples; this guard was written when 42 did. "
        "If models were retired, lower the floor deliberately."
    )


@pytest.mark.parametrize(
    "name,num_samples,meta", SAMPLED, ids=[n for n, _, _ in SAMPLED]
)
def test_the_declared_metrics_match_the_sample_count(name, num_samples, meta):
    point = meta.get("regression_point_metrics") or []
    sample = meta.get("regression_sample_metrics") or []

    assert isinstance(num_samples, int) and num_samples >= 1, (
        f"{name}: num_samples is {num_samples!r}; it is passed straight to predict() and "
        f"must be a positive integer"
    )

    if num_samples == 1:
        assert point, (
            f"{name}: num_samples is 1, so views-evaluation will classify this as a 'point' "
            f"evaluation and read `regression_point_metrics` — which is empty. The run will "
            f"train to completion and THEN raise, writing no predictions. Either declare "
            f"point metrics or raise num_samples above 1. "
            f"(regression_sample_metrics={'set' if sample else 'empty'}, and it is never "
            f"read at one sample.)"
        )
    else:
        assert sample, (
            f"{name}: num_samples is {num_samples}, so views-evaluation will classify this as "
            f"a 'sample' evaluation and read `regression_sample_metrics` — which is empty. "
            f"The run will train to completion and THEN raise. Either declare sample metrics "
            f"or set num_samples to 1. "
            f"(regression_point_metrics={'set' if point else 'empty'}, and it is not read "
            f"above one sample.)"
        )


def test_the_pgm_delivery_batch_agrees_on_whether_its_estimate_is_stochastic():
    """The eleven pgm models ship as ONE batch, so they must not differ silently here.

    Scoped to the batch on purpose. An earlier version of this test asserted, for every model
    in the repo, that `num_samples: 1` implies `mc_dropout: False` — and it was red for
    `emerging_principles` and `preliminary_directives`, two cm models that deliberately take a
    single MC-dropout draw. That is an unusual choice but a defensible one, and a test that
    calls it a defect is asserting a preference. The claim that is actually true is narrower:
    **models handed over together must be comparable with each other.**

    `num_samples` and `mc_dropout` are both mandatory and go straight to `predict()`
    (`darts_forecasting_model_manager.py::_get_predict_kwargs`). At one sample with
    `mc_dropout: True` the forward pass is still stochastic, so the "point" estimate is one
    draw from the dropout distribution rather than the deterministic prediction — and nothing
    in the delivered parquet records which it was. A researcher comparing eleven models would
    be comparing nine expectations against two single draws, with no way to tell.

    This test is **red before #536 and green after**: today nine are deterministic and two are
    not, which is the whole content of that issue.
    """
    batch = {}
    for directory in sorted((REPO_ROOT / "models").glob("*")):
        hp_path = directory / "configs" / "config_hyperparameters.py"
        meta_path = directory / "configs" / "config_meta.py"
        if not (hp_path.is_file() and meta_path.is_file()):
            continue
        try:
            hp = _config(directory, "hyperparameters", "get_hp_config")
            meta = _config(directory, "meta", "get_meta_config")
        except Exception:  # noqa: BLE001
            continue
        if "num_samples" not in hp or meta.get("level") != "pgm":
            continue
        batch[directory.name] = (hp["num_samples"], hp.get("mc_dropout"))

    assert len(batch) == 11, (
        f"expected the 11 pgm models of epic #532, found {len(batch)}: {sorted(batch)}. "
        "If the roster changed, update this count deliberately."
    )

    stochastic = {n: v for n, v in batch.items() if v[0] > 1 or v[1] is True}
    deterministic = {n: v for n, v in batch.items() if n not in stochastic}
    assert not (stochastic and deterministic), (
        "the pgm batch is split on whether its point estimate is stochastic, so the eleven "
        "are not comparable with each other:\n"
        f"  deterministic ({len(deterministic)}): {sorted(deterministic)}\n"
        f"  stochastic    ({len(stochastic)}): "
        + ", ".join(f"{n} (num_samples={v[0]}, mc_dropout={v[1]})"
                    for n, v in sorted(stochastic.items()))
        + "\nSee #536. Either align them, or deliver them as separate batches and say so."
    )

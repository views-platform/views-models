"""Turn a model's predictions into the point-prediction parquets researchers consume.

``ensemble-updater`` and the qualitative-analysis work want **one number per cell**. Getting there
depends on what the model wrote, and the two families write different things:

**HydraNet** (``prediction_format: "prediction_frame"``) writes a posterior cube — one row per
(cell, month), one column per draw — under
``models/<model>/data/generated/predictions_<run_type>_<ts>/origin_i/<target>/``.

    python -m tools.collapse.collapse_predictions models/<model> --run-type calibration
    python -m tools.collapse.plot_collapse_audit <parquet> --draws-dir <origin> --out audit.png

**r2darts2** (``prediction_format: "dataframe"``) writes 13 parquets directly,
``predictions_<run_type>_<ts>_<NN>.parquet``, already one file per origin — but with a Python
**list in every cell** and the keys as an index.

    python -m tools.collapse.collapse_darts_predictions models/<model> --run-type calibration

Two converters rather than one: the inputs share no format, no target naming and no draw-count
contract, and the HydraNet reader's "a posterior cube always carries D x K >= 2" is false for nine
of the eleven darts models. What they share is the output schema and the reasons for each refusal.

House rules: a converter refuses rather than guesses — a misaligned target, a non-finite draw, a
negative value, a duplicate ``(month_id, priogrid_id)`` or a log1p-scaled field is an error, never
something to average away. The specification both implement is views-models#505; the darts half is
views-models#533 under epic #532.
"""

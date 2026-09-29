"""Turn HydraNet posterior draws into the point-prediction parquets researchers consume.

The models write their predictions as a posterior cube — one row per (cell, month), one column
per draw — under ``models/<model>/data/generated/predictions_<run_type>_<ts>/origin_i/<target>/``.
``ensemble-updater`` and the qualitative-analysis work want one number per cell. This package
does that conversion, on Simon's laptop, from the numpy that comes down off the run.

    python -m tools.collapse.collapse_predictions models/<model> --run-type calibration
    python -m tools.collapse.plot_collapse_audit <parquet> --draws-dir <origin> --out audit.png

House rules: the converter refuses rather than guesses — a misaligned target, a non-finite draw,
a negative value or a log1p-scaled field is an error, never something to average away. The
specification it implements is views-models#505.
"""

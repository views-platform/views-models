"""Parity comparison between viewser and datafactory model/ensemble predictions.

Compares prediction arrays pair-by-pair and generates a structured report.
Designed for the 12-comparison parity matrix (9 model pairs + 3 ensemble pairs).

Usage:
    python scripts/compare_parity.py --run calibration
    python scripts/compare_parity.py --run forecasting --pair purple_alien bright_starship
    python scripts/compare_parity.py --run calibration --ensemble
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

MODEL_PAIRS = [
    ("purple_alien", "bright_starship", "shrinkage"),
    ("blue_stranger", "bold_comet", "basu_dpd"),
    ("violet_visitor", "blazing_meteor", "lognormal_nll"),
]

ENSEMBLE_PAIR = ("golden_hour", "stellar_horizon")

TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]


def find_prediction_dir(name, kind, run_type):
    if kind == "model":
        base = REPO / "models" / name / "data" / "generated"
    else:
        base = REPO / "ensembles" / name / "data" / "generated"
    pred_dirs = sorted(
        [d for d in base.glob(f"predictions_{run_type}_*") if d.is_dir()]
    )
    return pred_dirs[-1] if pred_dirs else None


def load_origins(pred_dir, target):
    origins = sorted(pred_dir.glob("origin_*"), key=lambda p: int(p.name.split("_")[1]))
    preds, ids_list = [], []
    for origin in origins:
        yf = origin / target / "y_pred.npy"
        idf = origin / target / "identifiers.npz"
        if yf.exists():
            preds.append(np.load(yf))
            ids_list.append(dict(np.load(idf)))
    return preds, ids_list


def compute_metrics(pred_a, pred_b):
    mean_a = pred_a.mean(axis=1)
    mean_b = pred_b.mean(axis=1)
    std_a = pred_a.std(axis=1)
    std_b = pred_b.std(axis=1)

    mask = (mean_a > 0) | (mean_b > 0)
    n_nonzero = mask.sum()

    pearson = np.corrcoef(mean_a, mean_b)[0, 1]
    pearson_nonzero = np.corrcoef(mean_a[mask], mean_b[mask])[0, 1] if n_nonzero > 1 else np.nan

    mae = np.abs(mean_a - mean_b).mean()
    rmse = np.sqrt(((mean_a - mean_b) ** 2).mean())

    rel_diff = np.where(
        (np.abs(mean_a) + np.abs(mean_b)) > 0,
        2 * np.abs(mean_a - mean_b) / (np.abs(mean_a) + np.abs(mean_b)),
        0.0,
    )
    median_rel_diff = np.median(rel_diff[mask]) if n_nonzero > 0 else 0.0

    std_corr = np.corrcoef(std_a, std_b)[0, 1]

    median_a = np.median(pred_a, axis=1)
    median_b = np.median(pred_b, axis=1)
    median_corr = np.corrcoef(median_a, median_b)[0, 1]

    q95_a = np.percentile(pred_a, 97.5, axis=1)
    q95_b = np.percentile(pred_b, 97.5, axis=1)
    q95_corr = np.corrcoef(q95_a, q95_b)[0, 1]

    return {
        "n_rows": pred_a.shape[0],
        "n_samples": pred_a.shape[1],
        "n_nonzero": int(n_nonzero),
        "mean_corr": float(pearson),
        "mean_corr_nonzero": float(pearson_nonzero),
        "median_corr": float(median_corr),
        "std_corr": float(std_corr),
        "q97.5_corr": float(q95_corr),
        "mae_of_means": float(mae),
        "rmse_of_means": float(rmse),
        "median_relative_diff": float(median_rel_diff),
        "mean_a_global": float(mean_a.mean()),
        "mean_b_global": float(mean_b.mean()),
        "std_a_global": float(std_a.mean()),
        "std_b_global": float(std_b.mean()),
    }


def grade(metrics):
    r = metrics["mean_corr"]
    if r > 0.99:
        return "EXCELLENT"
    elif r > 0.95:
        return "GOOD"
    elif r > 0.80:
        return "FAIR"
    elif r > 0.50:
        return "POOR"
    else:
        return "DIVERGENT"


def compare_pair(name_a, name_b, kind, run_type):
    dir_a = find_prediction_dir(name_a, kind, run_type)
    dir_b = find_prediction_dir(name_b, kind, run_type)

    if not dir_a:
        return {"status": "MISSING", "detail": f"{name_a} has no {run_type} predictions"}
    if not dir_b:
        return {"status": "MISSING", "detail": f"{name_b} has no {run_type} predictions"}

    results = {"status": "OK", "dir_a": str(dir_a.name), "dir_b": str(dir_b.name), "targets": {}}

    for target in TARGETS:
        preds_a, ids_a = load_origins(dir_a, target)
        preds_b, ids_b = load_origins(dir_b, target)

        if not preds_a or not preds_b:
            results["targets"][target] = {"status": "MISSING"}
            continue

        n_origins_a, n_origins_b = len(preds_a), len(preds_b)
        if n_origins_a != n_origins_b:
            results["targets"][target] = {
                "status": "ORIGIN_MISMATCH",
                "origins_a": n_origins_a,
                "origins_b": n_origins_b,
            }
            continue

        origin_metrics = []
        for i, (pa, pb, ia, ib) in enumerate(zip(preds_a, preds_b, ids_a, ids_b)):
            if pa.shape != pb.shape:
                origin_metrics.append({"origin": i, "status": "SHAPE_MISMATCH", "shape_a": pa.shape, "shape_b": pb.shape})
                continue

            ids_match = np.array_equal(ia["time"], ib["time"]) and np.array_equal(ia["unit"], ib["unit"])
            m = compute_metrics(pa, pb)
            m["origin"] = i
            m["ids_match"] = ids_match
            m["grade"] = grade(m)
            origin_metrics.append(m)

        agg_corrs = [m["mean_corr"] for m in origin_metrics if "mean_corr" in m]
        results["targets"][target] = {
            "status": "OK",
            "n_origins": n_origins_a,
            "avg_mean_corr": float(np.mean(agg_corrs)) if agg_corrs else None,
            "min_mean_corr": float(np.min(agg_corrs)) if agg_corrs else None,
            "grade": grade({"mean_corr": np.mean(agg_corrs)}) if agg_corrs else "N/A",
            "origins": origin_metrics,
        }

    return results


def print_report(name_a, name_b, kind, run_type, results):
    print(f"\n{'='*72}")
    print(f"PARITY COMPARISON: {name_a} ↔ {name_b}")
    print(f"Type: {kind}  |  Run: {run_type}  |  Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*72}")

    if results["status"] == "MISSING":
        print(f"  SKIPPED — {results['detail']}")
        return

    print(f"  Dirs: {results['dir_a']}  ↔  {results['dir_b']}")

    for target, tres in results["targets"].items():
        print(f"\n  --- {target} ---")
        if tres["status"] != "OK":
            print(f"  Status: {tres['status']}")
            continue

        print(f"  Origins: {tres['n_origins']}  |  Grade: {tres['grade']}")
        print(f"  Avg mean correlation: {tres['avg_mean_corr']:.6f}")
        print(f"  Min mean correlation: {tres['min_mean_corr']:.6f}")

        print(f"\n  {'Origin':>6} {'Grade':>10} {'r(mean)':>10} {'r(med)':>10} {'r(std)':>10} {'r(q97.5)':>10} {'MAE':>10} {'MedRelDiff':>10} {'IDs':>5}")
        print(f"  {'-'*6:>6} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*5:>5}")
        for m in tres["origins"]:
            if "mean_corr" not in m:
                print(f"  {m['origin']:>6}  {m.get('status', 'ERROR')}")
                continue
            print(
                f"  {m['origin']:>6} {m['grade']:>10} {m['mean_corr']:>10.6f} {m['median_corr']:>10.6f} "
                f"{m['std_corr']:>10.6f} {m['q97.5_corr']:>10.6f} {m['mae_of_means']:>10.4f} "
                f"{m['median_relative_diff']:>10.4f} {'✓' if m['ids_match'] else '✗':>5}"
            )

        o0 = tres["origins"][0]
        if "mean_a_global" in o0:
            print("\n  Origin 0 scale check:")
            print(f"    {name_a}: mean={o0['mean_a_global']:.4f}, std={o0['std_a_global']:.4f}")
            print(f"    {name_b}: mean={o0['mean_b_global']:.4f}, std={o0['std_b_global']:.4f}")


def print_summary(all_results):
    print(f"\n{'='*72}")
    print("PARITY SUMMARY")
    print(f"{'='*72}")
    print(f"\n{'Pair':<35} {'Run':<14} {'lr_sb':>8} {'lr_ns':>8} {'lr_os':>8} {'Verdict':>10}")
    print(f"{'-'*35:<35} {'-'*14:<14} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*10:>10}")
    for (na, nb, kind, run_type), res in all_results:
        label = f"{na} ↔ {nb}"
        if res["status"] == "MISSING":
            print(f"{label:<35} {run_type:<14} {'—':>8} {'—':>8} {'—':>8} {'MISSING':>10}")
            continue
        grades = []
        for t in TARGETS:
            g = res["targets"].get(t, {}).get("grade", "—")
            grades.append(g)
        worst = "MISSING"
        rank = {"EXCELLENT": 5, "GOOD": 4, "FAIR": 3, "POOR": 2, "DIVERGENT": 1, "N/A": 0, "—": 0}
        valid = [g for g in grades if g in rank and rank[g] > 0]
        if valid:
            worst = min(valid, key=lambda g: rank[g])
        print(f"{label:<35} {run_type:<14} {grades[0]:>8} {grades[1]:>8} {grades[2]:>8} {worst:>10}")


def main():
    parser = argparse.ArgumentParser(description="Parity comparison between viewser and datafactory predictions")
    parser.add_argument("--run", required=True, choices=["calibration", "validation", "forecasting"])
    parser.add_argument("--pair", nargs=2, metavar=("MODEL_A", "MODEL_B"), help="Compare a specific pair")
    parser.add_argument("--ensemble", action="store_true", help="Compare ensemble pair only")
    parser.add_argument("--all", action="store_true", help="Compare all pairs + ensemble")
    args = parser.parse_args()

    all_results = []

    if args.pair:
        res = compare_pair(args.pair[0], args.pair[1], "model", args.run)
        print_report(args.pair[0], args.pair[1], "model", args.run, res)
        all_results.append(((args.pair[0], args.pair[1], "model", args.run), res))

    elif args.ensemble:
        na, nb = ENSEMBLE_PAIR
        res = compare_pair(na, nb, "ensemble", args.run)
        print_report(na, nb, "ensemble", args.run, res)
        all_results.append(((na, nb, "ensemble", args.run), res))

    else:
        for na, nb, loss in MODEL_PAIRS:
            res = compare_pair(na, nb, "model", args.run)
            print_report(na, nb, "model", args.run, res)
            all_results.append(((na, nb, "model", args.run), res))

        if args.all:
            na, nb = ENSEMBLE_PAIR
            res = compare_pair(na, nb, "ensemble", args.run)
            print_report(na, nb, "ensemble", args.run, res)
            all_results.append(((na, nb, "ensemble", args.run), res))

    if len(all_results) > 1:
        print_summary(all_results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
04. Supervised prediction of 6-hour CE worsening (Fig. 3, Fig. 6A,
    Supp. Fig. 4-5).

Feature sets (see config.FEATURE_SETS_TO_RUN)
    12-Protein + Clinical (RFECV)  fixed 12-protein panel; clinical candidates
                                   selected by RFECV (random-forest estimator)
                                   inside each training fold; selection
                                   frequencies are logged
    12-Protein + Clinical          fixed 12-protein panel + CLINICAL_FEATURES_FINAL
    Clinical Only                  complete candidate clinical set
    60-Protein Only                elastic-net panel alone

Per feature set (results/04_classification/<set>/)
    results.csv                    all metrics with bootstrap CIs
    oof_predictions.csv            pooled out-of-fold probabilities
    fold_metrics.csv               fold-level accuracy/precision/recall (0.5)
    rfecv_fold_log.csv, rfecv_selection_frequency.csv  (RFECV set only)
    roc_curves.png, roc_auc_forest.png, pr_auc_forest.png,
    youden_precision_recall_forest.png, best_model_pr_curve.png,
    calibration_all_models.png, best_model_calibration.png, kde_best_model.png

Across sets (results/04_classification/)
    composite_roc_panel.png, best_models_roc_auc_forest.png,
    best_models_pr_auc_forest.png, delong_tests.csv, summary_best_models.csv

Usage
    python scripts/04_classification.py [--n-bootstrap 1000] [--n-repeats 1]
        [--feature-sets "12-Protein + Clinical" "Clinical Only"]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ce_proteomics import config  # noqa: E402
from ce_proteomics import classification_plots as cplots  # noqa: E402
from ce_proteomics.classification import (  # noqa: E402
    build_classifiers, cross_validate_feature_set, delong_roc_test, get_best_model_by_auc,
    make_cv, oof_predictions_frame, rfecv_selection_frequency, summarize_performance,
)
from ce_proteomics.io import (  # noqa: E402
    ensure_dir, load_analysis_dataset, read_protein_list, validate_panel, worsening_cohort,
)
from ce_proteomics.plotting import save_figure, set_style  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=config.ANALYSIS_DATASET_FILE)
    parser.add_argument("--panel-12", type=Path, default=config.PANEL_12_FILE)
    parser.add_argument("--panel-60", type=Path, default=config.PANEL_60_FILE)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS)
    parser.add_argument("--feature-sets", nargs="+", default=config.FEATURE_SETS_TO_RUN)
    parser.add_argument("--n-bootstrap", type=int, default=config.N_BOOTSTRAP_CI)
    parser.add_argument("--n-splits", type=int, default=config.CV_N_SPLITS)
    parser.add_argument("--n-repeats", type=int, default=config.CV_N_REPEATS)
    parser.add_argument("--no-scale-within-folds", action="store_true",
                        help="Legacy behavior: fit the scaler once on the full cohort.")
    parser.add_argument("--rfe-trees", type=int, default=config.RFE_RF_N_ESTIMATORS,
                        help="Random-forest size inside RFECV (lower for quick tests).")
    return parser.parse_args()


def define_feature_sets(df: pd.DataFrame, proteins_12: list[str], proteins_60: list[str]) -> dict:
    candidates = [c for c in config.CLINICAL_FEATURES_CANDIDATE if c in df.columns]
    missing = [c for c in config.CLINICAL_FEATURES_CANDIDATE if c not in df.columns]
    if missing:
        print(f"  Note: candidate clinical variables not in the dataset: {', '.join(missing)}")
    final = [c for c in config.CLINICAL_FEATURES_FINAL if c in df.columns]
    return {
        "12-Protein + Clinical (RFECV)": {"fixed": proteins_12, "candidates": candidates, "rfecv": True},
        "12-Protein + Clinical": {"fixed": proteins_12 + final, "candidates": [], "rfecv": False},
        "Clinical Only": {"fixed": candidates, "candidates": [], "rfecv": False},
        "60-Protein Only": {"fixed": proteins_60, "candidates": [], "rfecv": False},
        "12-Protein Only": {"fixed": proteins_12, "candidates": [], "rfecv": False},
        "60-Protein + Clinical": {"fixed": proteins_60 + final, "candidates": [], "rfecv": False},
    }


def run_feature_set(name: str, spec: dict, df: pd.DataFrame, args, out_root: Path) -> dict:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    out_dir = ensure_dir(out_root / name.lower().replace(" ", "_").replace("+", "plus")
                         .replace("(", "").replace(")", ""))
    X = df.set_index(config.COL_ID)
    y = X[config.COL_WORSE]
    cv = make_cv(n_splits=args.n_splits, n_repeats=args.n_repeats)
    classifiers = build_classifiers()

    cv_out = cross_validate_feature_set(
        X, y, fixed_cols=spec["fixed"], candidate_cols=spec["candidates"], use_rfecv=spec["rfecv"],
        classifiers=classifiers, cv=cv, scale_within_folds=not args.no_scale_within_folds,
    )
    results_df, roc_curves, pr_curves = summarize_performance(
        cv_out["y"], cv_out["pooled_probs"], cv_out["fold_metrics"], n_bootstrap=args.n_bootstrap,
    )
    results_df.insert(0, "Feature Set", name)
    results_df.to_csv(out_dir / "results.csv", index=False)
    oof_predictions_frame(cv_out["y"], cv_out["pooled_probs"]).to_csv(out_dir / "oof_predictions.csv", index=False)
    cv_out["fold_metrics"].to_csv(out_dir / "fold_metrics.csv", index=False)

    if cv_out["rfecv_log"] is not None:
        cv_out["rfecv_log"].to_csv(out_dir / "rfecv_fold_log.csv", index=False)
        freq = rfecv_selection_frequency(cv_out["rfecv_log"], spec["candidates"])
        freq.to_csv(out_dir / "rfecv_selection_frequency.csv", index=False)
        print("\nRFECV clinical selection frequency across outer folds:")
        print(freq.to_string(index=False))

    show = ["Classifier", "AUC (Pooled)", "AUC 95% CI", "PR-AUC (AP)", "PR-AUC 95% CI",
            "Recall (Youden)", "Brier Score", "Cal. Slope", "Cal. Intercept"]
    print(results_df[show].to_string(index=False))

    y = cv_out["y"]
    prevalence = float(np.mean(y))
    best = get_best_model_by_auc(results_df)
    best_row = results_df.loc[results_df["Classifier"] == best].iloc[0]
    print(f"\nBest model by ROC-AUC: {best} ({best_row['AUC (Pooled)']:.3f} {best_row['AUC 95% CI']})")

    save_figure(cplots.plot_roc_curves(roc_curves, f"{name} ROC curves"), out_dir / "roc_curves.png")
    save_figure(cplots.plot_auc_forest(results_df, "roc", f"ROC-AUC with 95% CI: {name}"),
                out_dir / "roc_auc_forest.png")
    save_figure(cplots.plot_auc_forest(results_df, "pr", f"PR-AUC with 95% CI: {name}", prevalence=prevalence),
                out_dir / "pr_auc_forest.png")
    save_figure(cplots.plot_youden_precision_recall_forest(results_df, name, prevalence),
                out_dir / "youden_precision_recall_forest.png")
    save_figure(cplots.plot_best_model_pr_curve(
        y, cv_out["pooled_probs"][best], best, name, best_row["PR-AUC (AP)"],
        best_row["PR-AUC CI Lower"], best_row["PR-AUC CI Upper"], best_row["Thresh (Youden)"]),
        out_dir / "best_model_pr_curve.png")
    save_figure(cplots.plot_calibration_curves(cv_out["pooled_probs"], y, name), out_dir / "calibration_all_models.png")
    save_figure(cplots.plot_calibration_curves({best: cv_out["pooled_probs"][best]}, y, name),
                out_dir / "best_model_calibration.png")
    save_figure(cplots.plot_kde_for_best_model(cv_out["pooled_probs"], y, best, name, best_row["Thresh (Youden)"]),
                out_dir / "kde_best_model.png")
    save_figure(cplots.plot_accuracy_barplot(cv_out["fold_metrics"], name), out_dir / "fold_accuracy.png")

    return {"results_df": results_df, "roc_curves": roc_curves, "pr_curves": pr_curves,
            "pooled_probs": cv_out["pooled_probs"], "y": y, "best": best}


def delong_comparisons(all_results: dict, out_root: Path) -> None:
    """DeLong tests between the reference and comparison feature sets."""
    ref, cmp_ = config.DELONG_REFERENCE_SET, config.DELONG_COMPARISON_SET
    if ref not in all_results or cmp_ not in all_results:
        print(f"\nDeLong comparison skipped ({ref} vs {cmp_} not both run).")
        return
    a, b = all_results[ref], all_results[cmp_]
    common = a["y"].index.intersection(b["y"].index)
    y = a["y"].loc[common]
    rows = []
    pairs = [(a["best"], b["best"], "best vs best")] + \
            [(clf, clf, "same classifier") for clf in a["pooled_probs"] if clf in b["pooled_probs"]]
    for clf_a, clf_b, kind in pairs:
        pa = pd.Series(a["pooled_probs"][clf_a], index=a["y"].index).loc[common]
        pb = pd.Series(b["pooled_probs"][clf_b], index=b["y"].index).loc[common]
        res = delong_roc_test(y, pa, pb)
        rows.append({"Comparison": kind, "Model A": f"{ref}: {clf_a}", "Model B": f"{cmp_}: {clf_b}",
                     "N": len(common), **res})
    delong_df = pd.DataFrame(rows)
    delong_df.to_csv(out_root / "delong_tests.csv", index=False)
    print("\nDeLong tests (reference vs comparison feature set):")
    print(delong_df.to_string(index=False))


def main() -> None:
    args = parse_args()
    set_style()
    config.RFE_RF_N_ESTIMATORS = args.rfe_trees
    out_root = ensure_dir(args.results_dir / "04_classification")

    df = worsening_cohort(load_analysis_dataset(args.dataset))
    proteins_12 = validate_panel(read_protein_list(args.panel_12), df, "12-protein")
    proteins_60 = validate_panel(read_protein_list(args.panel_60), df, "60-protein")
    print(f"Worsening cohort: n = {len(df)}; 12-protein panel: {len(proteins_12)}; "
          f"60-protein panel: {len(proteins_60)}")

    feature_sets = define_feature_sets(df, proteins_12, proteins_60)
    unknown = set(args.feature_sets) - set(feature_sets)
    if unknown:
        raise ValueError("Unknown feature set(s): " + ", ".join(sorted(unknown)))

    all_results = {}
    for name in args.feature_sets:
        all_results[name] = run_feature_set(name, feature_sets[name], df, args, out_root)

    all_roc = {name: r["roc_curves"] for name, r in all_results.items()}
    save_figure(cplots.plot_composite_roc_panel(all_roc), out_root / "composite_roc_panel.png")
    save_figure(cplots.plot_best_model_forest(all_results, "roc"), out_root / "best_models_roc_auc_forest.png")
    save_figure(cplots.plot_best_model_forest(all_results, "pr"), out_root / "best_models_pr_auc_forest.png")

    summary = pd.concat(
        [r["results_df"][r["results_df"]["Classifier"] == r["best"]] for r in all_results.values()],
        ignore_index=True,
    )
    summary.to_csv(out_root / "summary_best_models.csv", index=False)
    print("\nBest model per feature set:")
    print(summary[["Feature Set", "Classifier", "AUC (Pooled)", "AUC 95% CI", "PR-AUC (AP)",
                   "PR-AUC 95% CI", "Brier Score", "Recall (Youden)"]].to_string(index=False))

    delong_comparisons(all_results, out_root)
    print(f"\nWritten to {out_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
05. Goal 2 mechanistic analysis of the 60-protein panel.

Stages
    A. Differential expression and volcano plot (Fig. 4). The outcome is
       configurable: 'worsening' (6-hour worsening, n = 116) or 'baseline'
       (CE on the admission CT, n = 123).
    B. PCA on z-scored NPX for all subjects with an admission CT (n = 123):
       PC scatter by baseline CE (Fig. 5A), PC scores by 6-hour worsening
       (Fig. 5B), syndromic correlation-loading plots for PC1 and PC2
       (Fig. 5C-D), scree and eigenvector bar plots (Supp. Fig. 6), and
       bootstrap loading stability.
    C. Random-forest interpretation on the worsening cohort: MDI and
       permutation importance (Supp. Fig. 7), SHAP bar and beeswarm
       (Fig. 6B-C), SHAP direction protein lists for enrichment (Fig. 7),
       pairwise TreeSHAP interactions (Supp. Fig. 8).
    D. Optional STRING enrichment of the SHAP direction lists (--string).

Outputs go to results/05_mechanistic/.

Usage
    python scripts/05_mechanistic_analysis.py [--volcano-outcome worsening|baseline]
        [--pca-bootstrap 500] [--string]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ce_proteomics import config  # noqa: E402
from ce_proteomics import interpretation as interp  # noqa: E402
from ce_proteomics import pca as pcamod  # noqa: E402
from ce_proteomics.enrichment import enrich_shap_directions, export_shap_direction_lists  # noqa: E402
from ce_proteomics.io import (  # noqa: E402
    baseline_cohort, ensure_dir, load_analysis_dataset, read_protein_list, validate_panel,
    worsening_cohort,
)
from ce_proteomics.plotting import save_figure, set_style  # noqa: E402
from ce_proteomics.univariate import differential_expression, plot_volcano  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=config.ANALYSIS_DATASET_FILE)
    parser.add_argument("--panel-60", type=Path, default=config.PANEL_60_FILE)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS)
    parser.add_argument("--volcano-outcome", choices=["worsening", "baseline"], default="worsening")
    parser.add_argument("--pca-bootstrap", type=int, default=config.PCA_BOOTSTRAP_N)
    parser.add_argument("--string", action="store_true", help="Query STRING for enrichment (needs network).")
    return parser.parse_args()


def stage_differential_expression(df_base, df_worse, proteins, outcome, out_dir):
    print("\nA. Differential expression")
    if outcome == "worsening":
        data, col, names, contrast = df_worse, config.COL_WORSE, ("NonWorsening", "Worsening"), \
            "CE Worsening / Non-Worsening"
    else:
        data, col, names, contrast = df_base, config.COL_CE_BASELINE, ("NoCE", "CE"), "CE / No CE"
    de = differential_expression(data[proteins], data[col], group_names=names)
    de.to_csv(out_dir / f"differential_expression_{outcome}.csv", index=False)
    print(f"  {int(de['Significant'].sum())} of {len(de)} proteins significant at FDR < {config.FDR_ALPHA} "
          f"({outcome}, n = {len(data)})")
    save_figure(plot_volcano(de, contrast_label=contrast), out_dir / f"fig4_volcano_{outcome}.png")
    return de


def stage_pca(df_base, proteins, pca_bootstrap, out_dir):
    print("\nB. PCA on the 60-protein panel")
    X = df_base.set_index(config.COL_ID)[proteins].astype(float).dropna()
    X_z = pcamod.zscore_matrix(X)
    ce_base = df_base.set_index(config.COL_ID).loc[X.index, config.COL_CE_BASELINE]
    worsening = df_base.set_index(config.COL_ID).loc[X.index, config.COL_WORSE]

    pca, scores, loadings = pcamod.run_pca(X_z)
    corr_loadings = pcamod.compute_correlation_loadings(pca, list(X_z.columns))
    evr = pca.explained_variance_ratio_
    print("  Variance explained: " + ", ".join(f"PC{i + 1} {v * 100:.1f}%" for i, v in enumerate(evr)))

    tests_base = pcamod.pc_group_tests(scores, ce_base)
    tests_worse = pcamod.pc_group_tests(scores, worsening)
    tests_base.to_csv(out_dir / "pc_tests_baseline_ce.csv", index=False)
    tests_worse.to_csv(out_dir / "pc_tests_worsening.csv", index=False)
    print("  PC scores by 6-hour worsening (Welch t):")
    print(tests_worse[["PC", "N0", "N1", "p_value", "Cohens_d"]].to_string(index=False))

    scores.to_csv(out_dir / "pca_scores.csv")
    loadings.to_csv(out_dir / "pca_eigenvector_loadings.csv")
    corr_loadings.to_csv(out_dir / "pca_correlation_loadings.csv")

    save_figure(pcamod.create_pca_panel(scores, corr_loadings, pca, ce_base, worsening, tests_worse),
                out_dir / "fig5_pca_panel.png")
    save_figure(pcamod.plot_pca_scatter(scores, ce_base, pca), out_dir / "fig5a_pca_scatter_baseline_ce.png")
    save_figure(pcamod.plot_pc_boxplots(scores, worsening, ["PC1", "PC2"], tests=tests_worse),
                out_dir / "fig5b_pc_boxplots_worsening.png")
    for pc_index in (1, 2):
        save_figure(pcamod.plot_syndromic_loadings(corr_loadings, pc_index, evr[pc_index - 1]),
                    out_dir / f"fig5_syndromic_pc{pc_index}.png")
        save_figure(pcamod.plot_loadings_bar(loadings, f"PC{pc_index}"),
                    out_dir / f"supp_fig6_pc{pc_index}_eigenvector_loadings.png")
    save_figure(pcamod.plot_scree(pca), out_dir / "supp_fig6a_scree.png")

    if pca_bootstrap > 0:
        stability, _ = pcamod.bootstrap_pca_stability(X_z.values, list(X_z.columns), n_boot=pca_bootstrap)
        stability.to_csv(out_dir / "pca_bootstrap_stability.csv")
        save_figure(pcamod.plot_pca_stability_heatmap(stability), out_dir / "pca_bootstrap_stability_heatmap.png")
        for pc in ("PC1", "PC2"):
            axis = pcamod.get_axis_proteins(stability, pc)
            print(f"  {pc} axis-defining proteins ({len(axis)}): {', '.join(axis[:12])}"
                  + (" ..." if len(axis) > 12 else ""))
    return corr_loadings


def stage_interpretation(df_worse, proteins, out_dir):
    print("\nC. Random-forest interpretation (worsening cohort)")
    X = df_worse.set_index(config.COL_ID)[proteins].astype(float).dropna()
    X_z = pcamod.zscore_matrix(X)
    y = df_worse.set_index(config.COL_ID).loc[X.index, config.COL_WORSE].astype(int)

    rf, mdi, cv_results = interp.fit_interpretation_rf(X_z, y)
    mdi.to_csv(out_dir / "rf_mdi_importance.csv")
    pd.DataFrame([cv_results]).to_csv(out_dir / "rf_cv_performance.csv", index=False)
    save_figure(interp.plot_importance_bar(mdi, "RF_importance", top_n=20,
                                           xlabel="Mean decrease in impurity",
                                           title="Random forest importance (MDI)"),
                out_dir / "rf_mdi_importance.png")

    perm = interp.get_rf_permutation_importance(X_z, y)
    perm.to_csv(out_dir / "rf_permutation_importance.csv")
    save_figure(interp.plot_importance_bar(perm, "perm_importance_mean", top_n=20,
                                           error_column="perm_importance_std",
                                           xlabel="Decrease in ROC-AUC when permuted",
                                           title="Random forest permutation importance"),
                out_dir / "supp_fig7_rf_permutation_importance.png")

    shap_values, explainer = interp.compute_shap_values(rf, X_z)
    summary = interp.shap_direction_summary(shap_values, X_z)
    summary.to_csv(out_dir / "shap_summary.csv", index=False)
    pd.DataFrame(shap_values, index=X_z.index, columns=X_z.columns).to_csv(out_dir / "shap_values.csv")
    print("  Top proteins by mean |SHAP|:")
    print(summary.head(config.SHAP_TOP_N)[["Protein", "Mean_|SHAP|", "Direction_corr", "Direction"]]
          .to_string(index=False))

    save_figure(interp.create_shap_panel(summary, shap_values, X_z), out_dir / "fig6bc_shap_panel.png")
    save_figure(interp.plot_shap_bar(summary), out_dir / "fig6b_shap_bar.png")
    save_figure(interp.plot_shap_beeswarm(shap_values, X_z), out_dir / "fig6c_shap_beeswarm.png")

    lists = interp.shap_protein_lists(summary)
    export_shap_direction_lists(lists, out_dir)

    _, pairs, _, matrix = interp.compute_shap_interactions(explainer, X_z, shap_values)
    matrix.to_csv(out_dir / "shap_interaction_matrix_top15.csv")
    pairs.to_csv(out_dir / "shap_interaction_pairs_top15.csv", index=False)
    save_figure(interp.plot_shap_interaction_heatmap(matrix), out_dir / "supp_fig8_shap_interaction_heatmap.png")
    return mdi, summary, lists


def main() -> None:
    args = parse_args()
    set_style()
    out_dir = ensure_dir(args.results_dir / "05_mechanistic")

    df = load_analysis_dataset(args.dataset)
    df_base = baseline_cohort(df)
    df_worse = worsening_cohort(df)
    proteins = validate_panel(read_protein_list(args.panel_60), df, "60-protein")
    print(f"Baseline cohort n = {len(df_base)}; worsening cohort n = {len(df_worse)}; "
          f"panel proteins = {len(proteins)}")

    de = stage_differential_expression(df_base, df_worse, proteins, args.volcano_outcome, out_dir)
    corr_loadings = stage_pca(df_base, proteins, args.pca_bootstrap, out_dir)
    mdi, summary, lists = stage_interpretation(df_worse, proteins, out_dir)

    ranks = interp.compare_importance_ranks(corr_loadings, mdi, de, summary)
    ranks.to_csv(out_dir / "method_rank_comparison.csv")

    if args.string:
        print("\nD. STRING enrichment of SHAP direction lists")
        strict = interp.shap_protein_lists(summary, top_n=len(summary),
                                           direction_corr=config.SHAP_ENRICH_DIRECTION_CORR)
        for direction, table in enrich_shap_directions(strict).items():
            table.to_csv(out_dir / f"string_enrichment_{direction}.csv", index=False)

    print(f"\nWritten to {out_dir}")
    print("For Fig. 7, upload results/05_mechanistic/shap_proteins_toward_worsening.txt to the "
          "Gladstone Interactive Enrichment Analysis tool.")


if __name__ == "__main__":
    main()

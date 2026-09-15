#!/usr/bin/env python3
"""
02. Goal 1 discovery: proteome-wide adjusted logistic screen.

One logistic regression per protein (CE on the admission CT ~ NPX + age +
sex + time to baseline CT), Benjamini-Hochberg FDR across all proteins.
Proteins with q < FDR_ALPHA form the 12-protein panel.

Outputs (results/02_univariate/)
    logistic_screen_all_proteins.csv   full screen with beta, OR, 95% CI, p, q
    supp_table_2_panel_associations.csv  the panel rows (Supplementary Table 2)
    panel_boxplot_tests.csv            t-tests behind the Fig. 2 asterisks
    fig2_panel_boxplots.png            Figure 2
results/panels/panel_12_fdr_logistic.txt  the panel, one protein per line

Usage
    python scripts/02_univariate_screen.py [--dataset ...] [--results-dir ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ce_proteomics import config  # noqa: E402
from ce_proteomics.io import (  # noqa: E402
    baseline_cohort, ensure_dir, get_protein_columns, load_analysis_dataset, write_protein_list,
)
from ce_proteomics.plotting import save_figure, set_style  # noqa: E402
from ce_proteomics.univariate import (  # noqa: E402
    adjusted_logistic_screen, panel_group_tests, plot_panel_boxplots, select_fdr_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=config.ANALYSIS_DATASET_FILE)
    parser.add_argument("--assays", type=Path, default=config.OLINK_ASSAY_LIST_FILE)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS)
    parser.add_argument("--fdr-alpha", type=float, default=config.FDR_ALPHA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    out_dir = ensure_dir(args.results_dir / "02_univariate")
    panel_dir = ensure_dir(args.results_dir / "panels")

    df = baseline_cohort(load_analysis_dataset(args.dataset))
    proteins = get_protein_columns(df, args.assays)
    print(f"Baseline cohort: n = {len(df)}; proteins screened: {len(proteins)}")

    print("Running the adjusted logistic screen ...")
    screen = adjusted_logistic_screen(df, proteins, fdr_alpha=args.fdr_alpha)
    screen.to_csv(out_dir / "logistic_screen_all_proteins.csv", index=False)

    panel = select_fdr_panel(screen)
    write_protein_list(panel, panel_dir / config.PANEL_12_FILE.name)
    print(f"FDR-significant panel ({len(panel)} proteins): {', '.join(panel)}")

    supp_table = screen[screen["Significant"]][
        ["Protein", "N", "Beta", "SE", "OR", "OR_CI_lower", "OR_CI_upper", "p_value", "q_value"]
    ]
    supp_table.to_csv(out_dir / "supp_table_2_panel_associations.csv", index=False)

    if panel:
        tests = panel_group_tests(df, panel)
        tests.to_csv(out_dir / "panel_boxplot_tests.csv", index=False)
        fig = plot_panel_boxplots(df, panel, tests)
        save_figure(fig, out_dir / "fig2_panel_boxplots.png")
    print(f"Written to {out_dir}")


if __name__ == "__main__":
    main()

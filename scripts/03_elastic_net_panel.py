#!/usr/bin/env python3
"""
03. Goal 2 panel selection: bootstrapped elastic-net logistic regression.

Proteins are residualized on age, sex, and time to the baseline CT,
standardized, and entered into elastic-net logistic regression (CE on the
admission CT as the outcome) over EN_N_BOOTSTRAP stratified bootstrap
resamples. Proteins selected in more than EN_SELECTION_FREQUENCY of resamples
form the stable (60-protein) panel.

Outputs (results/03_elastic_net/)
    elastic_net_stability.csv        selection count/frequency and mean coefficient
    elastic_net_stability_vs_effect.png  frequency vs effect, univariate hits highlighted
results/panels/panel_60_elastic_net.txt  the panel, one protein per line

Usage
    python scripts/03_elastic_net_panel.py [--n-bootstrap 1000] [--n-jobs -1]

Runtime: with ~5,400 proteins this step takes hours at 1,000 resamples.
Use --n-bootstrap with a small value to test the pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ce_proteomics import config  # noqa: E402
from ce_proteomics.elastic_net import (  # noqa: E402
    bootstrap_elastic_net_selection, plot_stability_vs_effect, prepare_elastic_net_inputs,
    select_stable_panel,
)
from ce_proteomics.io import (  # noqa: E402
    baseline_cohort, ensure_dir, get_protein_columns, load_analysis_dataset, read_protein_list,
    write_protein_list,
)
from ce_proteomics.plotting import save_figure, set_style  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=config.ANALYSIS_DATASET_FILE)
    parser.add_argument("--assays", type=Path, default=config.OLINK_ASSAY_LIST_FILE)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS)
    parser.add_argument("--n-bootstrap", type=int, default=config.EN_N_BOOTSTRAP)
    parser.add_argument("--frequency-threshold", type=float, default=config.EN_SELECTION_FREQUENCY)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    out_dir = ensure_dir(args.results_dir / "03_elastic_net")
    panel_dir = ensure_dir(args.results_dir / "panels")

    df = baseline_cohort(load_analysis_dataset(args.dataset))
    proteins = get_protein_columns(df, args.assays)
    X_scaled, y, feature_names = prepare_elastic_net_inputs(df, proteins)
    print(f"Baseline cohort: n = {len(y)}; proteins: {len(feature_names)}; "
          f"bootstraps: {args.n_bootstrap}")

    stability = bootstrap_elastic_net_selection(
        X_scaled, y, feature_names, n_bootstrap=args.n_bootstrap, n_jobs=args.n_jobs,
    )
    stability.to_csv(out_dir / "elastic_net_stability.csv", index=False)

    panel = select_stable_panel(stability, args.frequency_threshold)
    write_protein_list(panel, panel_dir / config.PANEL_60_FILE.name)
    print(f"Stable panel (> {args.frequency_threshold:.0%} of resamples): {len(panel)} proteins")

    panel_12_path = panel_dir / config.PANEL_12_FILE.name
    highlight = read_protein_list(panel_12_path) if panel_12_path.exists() else []
    fig = plot_stability_vs_effect(stability, highlight)
    save_figure(fig, out_dir / "elastic_net_stability_vs_effect.png")
    print(f"Written to {out_dir}")


if __name__ == "__main__":
    main()

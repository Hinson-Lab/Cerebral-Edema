#!/usr/bin/env python3
"""
06. Admission glucose and blood alcohol diagnostics (Supplementary Fig. 3).

Descriptive figures assessing whether the selection of glucose and alcohol as
clinical predictors could be driven by extreme values: glucose distributions
by baseline CE and 6-hour trajectory with a Q3 + 3xIQR reference line, robust
modified z-scores for detectable alcohol, worsening rates by detectable
alcohol with Wilson CIs, and outcome counts above the alcohol cutoff.

Outputs go to results/06_diagnostics/.

Usage
    python scripts/06_glucose_alcohol_diagnostics.py [--dataset ...] [--results-dir ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ce_proteomics import config  # noqa: E402
from ce_proteomics.diagnostics import run_diagnostics  # noqa: E402
from ce_proteomics.io import ensure_dir, load_analysis_dataset  # noqa: E402
from ce_proteomics.plotting import set_style  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=config.ANALYSIS_DATASET_FILE)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    out_dir = ensure_dir(args.results_dir / "06_diagnostics")
    df = load_analysis_dataset(args.dataset)
    summary = run_diagnostics(df, out_dir)
    print(summary.to_string(index=False))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()

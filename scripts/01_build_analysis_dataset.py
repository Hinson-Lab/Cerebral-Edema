#!/usr/bin/env python3
"""
01. Build the analysis dataset.

Inputs (data/raw/, see config.py)
    Olink Explore HT long-format NPX export
    REDCap clinical export (DATA_LABELS)
    REDCap CT export (DATA_LABELS; one row per subject per scan)

Outputs (data/processed/)
    olink_npx_wide.csv     one row per sample, one column per assay
    olink_assays.txt       assay (protein) column names
    analysis_dataset.csv   one row per subject: outcomes, covariates,
                           clinical variables, proteins
    missingness_report.csv missing-value counts for outcomes and covariates

Usage
    python scripts/01_build_analysis_dataset.py
    python scripts/01_build_analysis_dataset.py --olink path/to/npx.csv \
        --clinical path/to/clinical.csv --ct path/to/ct.csv --out-dir data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ce_proteomics import config  # noqa: E402
from ce_proteomics.io import ensure_dir, write_protein_list  # noqa: E402
from ce_proteomics.preprocessing import (  # noqa: E402
    build_analysis_dataset,
    load_redcap_export,
    missingness_report,
    pivot_olink_long_to_wide,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--olink", type=Path, default=config.OLINK_LONG_FILE)
    parser.add_argument("--clinical", type=Path, default=config.REDCAP_CLINICAL_FILE)
    parser.add_argument("--ct", type=Path, default=config.REDCAP_CT_FILE)
    parser.add_argument("--out-dir", type=Path, default=config.DATA_PROCESSED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.olink, args.clinical, args.ct):
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
    out_dir = ensure_dir(args.out_dir)

    print("1. Pivoting Olink NPX data to wide format")
    olink_long = pd.read_csv(args.olink)
    olink_wide, assays = pivot_olink_long_to_wide(olink_long)
    olink_wide.to_csv(out_dir / config.OLINK_WIDE_FILE.name, index=False)
    write_protein_list(assays, out_dir / config.OLINK_ASSAY_LIST_FILE.name)

    print("2. Loading REDCap exports")
    clinical = load_redcap_export(args.clinical)
    ct = load_redcap_export(args.ct)

    print("3. Building the analysis dataset")
    df = build_analysis_dataset(olink_wide, clinical, ct, assays)
    df.to_csv(out_dir / config.ANALYSIS_DATASET_FILE.name, index=False)

    report_cols = list(dict.fromkeys(
        [config.COL_CE_BASELINE, config.COL_CE_6HR, config.COL_WORSE] + config.ADJUSTMENT_COVARIATES
        + [c for c in config.CLINICAL_FEATURES_CANDIDATE if c in df.columns]
    ))
    report = missingness_report(df, report_cols)
    report.to_csv(out_dir / "missingness_report.csv", index=False)
    print("\nMissingness report:")
    print(report.to_string(index=False))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()

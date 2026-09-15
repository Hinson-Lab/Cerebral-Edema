#!/usr/bin/env python3
"""
Generate a synthetic Olink + REDCap data set with the same file layout and
column labels as the PROTIPS exports, so the pipeline can be run end to end
without access to patient data (used by tests/test_smoke_pipeline.py and
useful for reviewers).

The data are random with a planted signal (a block of proteins shifted in
subjects with baseline CE and, more weakly, in those who worsen). Nothing in
these files corresponds to a real participant.

Usage
    python tests/make_synthetic_data.py --out-dir data/synthetic_raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ce_proteomics import config  # noqa: E402


def make_synthetic(out_dir: Path, n_subjects: int = 123, n_follow_up: int = 116,
                   n_assays: int = 300, n_signal: int = 15, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [f"PT-{i + 1:03d}" for i in range(n_subjects)]

    # Outcomes
    ce_base = rng.binomial(1, 0.31, size=n_subjects)
    has_follow = np.zeros(n_subjects, dtype=bool)
    has_follow[rng.choice(n_subjects, size=n_follow_up, replace=False)] = True
    p_worse = np.where(ce_base == 1, 0.60, 0.17)
    worse = rng.binomial(1, p_worse)
    ce_6hr = np.clip(ce_base + worse, 0, 1)

    # Proteins: planted signal in the first n_signal assays
    assays = [f"PROT{i + 1:04d}" for i in range(n_assays)]
    npx = rng.normal(0, 1, size=(n_subjects, n_assays))
    latent = rng.normal(0, 1, size=n_subjects)
    for j in range(n_signal):
        npx[:, j] += 0.9 * ce_base + 0.5 * worse + 0.4 * latent + rng.normal(0, 0.3, n_subjects)
    npx += rng.normal(1.0, 0.2, size=(1, n_assays))  # assay-specific offsets

    # Olink long export (plus a few control-assay rows that must be removed)
    long = pd.DataFrame({
        config.OLINK_SAMPLE_COL: np.repeat(ids, n_assays),
        "OlinkID": np.tile([f"OID{i + 1:05d}" for i in range(n_assays)], n_subjects),
        config.OLINK_ASSAY_COL: np.tile(assays, n_subjects),
        config.OLINK_ASSAY_TYPE_COL: "assay",
        config.OLINK_NPX_COL: npx.ravel(),
        "SampleQC": "PASS",
        "Hemolysis": rng.choice([0, 20, 50, 100], size=n_subjects * n_assays),
    })
    controls = pd.DataFrame({
        config.OLINK_SAMPLE_COL: np.repeat(ids, 2),
        "OlinkID": np.tile(["OIDCTRL1", "OIDCTRL2"], n_subjects),
        config.OLINK_ASSAY_COL: np.tile(["Incubation control 1", "Extension control"], n_subjects),
        config.OLINK_ASSAY_TYPE_COL: np.tile(["inc_ctrl", "ext_ctrl"], n_subjects),
        config.OLINK_NPX_COL: rng.normal(0, 1, n_subjects * 2),
        "SampleQC": "PASS",
        "Hemolysis": 0,
    })
    pd.concat([long, controls], ignore_index=True).to_csv(out_dir / config.OLINK_LONG_FILE.name, index=False)

    # Clinical REDCap export
    age = rng.integers(18, 85, n_subjects)
    sex = rng.choice(["Cis Male", "Cis Female"], size=n_subjects, p=[0.7, 0.3])
    gcs = np.clip(rng.integers(3, 13, n_subjects) - ce_base, 3, 12)
    clinical = pd.DataFrame({
        "Record ID": ids,
        "Age upon admission:": age,
        "Gender:": sex,
        "Admission GCS": gcs,
        "GCS Motor": np.clip(gcs - rng.integers(2, 5, n_subjects), 1, 6),
        "Pupil Reactivity": rng.choice(["Both reactive", "One reactive", "Neither reactive"],
                                       size=n_subjects, p=[0.8, 0.15, 0.05]),
        "Glucose": np.round(rng.lognormal(np.log(140), 0.25, n_subjects)),
        "Alcohol": np.where(rng.random(n_subjects) < 0.4, np.round(rng.uniform(10, 350, n_subjects)), 0.0),
        "ISS": rng.integers(9, 45, n_subjects),
        "AIS": rng.integers(3, 6, n_subjects),
        "Hypotension": rng.choice(["Yes", "No"], size=n_subjects, p=[0.15, 0.85]),
        "Hypoxia": rng.choice(["Yes", "No"], size=n_subjects, p=[0.1, 0.9]),
        "Hemoglobin": np.round(rng.normal(13, 1.8, n_subjects), 1),
        "Platelets": np.round(rng.normal(245, 70, n_subjects)),
        "INR": np.round(rng.normal(1.08, 0.12, n_subjects), 2),
        "Fibrinogen": np.round(rng.normal(270, 60, n_subjects)),
    })
    clinical.to_csv(out_dir / config.REDCAP_CLINICAL_FILE.name, index=False)

    # CT REDCap export (baseline row for everyone; 6-hour row for those with follow-up)
    yes_no = lambda v: np.where(v == 1, "Yes", "No")  # noqa: E731
    base_rows = pd.DataFrame({
        "Record ID": ids,
        "Head CT Type": "Baseline",
        "Elapsed time between trauma and Head CT Type ": np.round(rng.uniform(0.5, 3.0, n_subjects), 2),
        "Cerebral edema:": np.where(ce_base == 1, "Present", "Absent"),
        "If previous scan, cerebral edema worse?": np.nan,
    })
    six_rows = pd.DataFrame({
        "Record ID": np.array(ids)[has_follow],
        "Head CT Type": "6 Hour",
        "Elapsed time between trauma and Head CT Type ": np.round(rng.uniform(5.0, 7.0, has_follow.sum()), 2),
        "Cerebral edema:": np.where(ce_6hr[has_follow] == 1, "Present", "Absent"),
        "If previous scan, cerebral edema worse?": yes_no(worse[has_follow]),
    })
    pd.concat([base_rows, six_rows], ignore_index=True).to_csv(out_dir / config.REDCAP_CT_FILE.name, index=False)
    print(f"Synthetic data written to {out_dir}: {n_subjects} subjects, {n_assays} assays, "
          f"{has_follow.sum()} with follow-up, {ce_base.sum()} with baseline CE, {worse[has_follow].sum()} worsened")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=config.REPO_ROOT / "data" / "synthetic_raw")
    parser.add_argument("--n-subjects", type=int, default=123)
    parser.add_argument("--n-assays", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    make_synthetic(args.out_dir, n_subjects=args.n_subjects, n_assays=args.n_assays, seed=args.seed)


if __name__ == "__main__":
    main()

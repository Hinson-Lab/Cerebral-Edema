"""
Loading and saving helpers shared across the pipeline.

All functions take explicit paths (defaulting to the locations in
``config``) so that the same code can be pointed at a different data
directory, for example the synthetic data used by the smoke test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config


def read_protein_list(path: Path | str) -> list[str]:
    """Read a one-protein-per-line text file, ignoring blank lines."""
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_protein_list(proteins: Iterable[str], path: Path | str) -> None:
    """Write a one-protein-per-line text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for protein in proteins:
            handle.write(f"{protein}\n")


def load_assay_list(path: Path | str = config.OLINK_ASSAY_LIST_FILE) -> list[str]:
    """Return the list of Olink assay (protein) column names."""
    return read_protein_list(path)


def load_analysis_dataset(
    path: Path | str = config.ANALYSIS_DATASET_FILE,
) -> pd.DataFrame:
    """
    Load the one-row-per-subject analysis dataset built by
    ``scripts/01_build_analysis_dataset.py``.
    """
    df = pd.read_csv(path)
    df[config.COL_ID] = df[config.COL_ID].astype(str).str.strip()
    return df


def get_protein_columns(
    df: pd.DataFrame,
    assay_list_path: Path | str = config.OLINK_ASSAY_LIST_FILE,
) -> list[str]:
    """
    Protein columns present in ``df``, taken from the saved assay list so that
    clinical columns can never be mistaken for proteins.
    """
    assays = load_assay_list(assay_list_path)
    present = [a for a in assays if a in df.columns]
    missing = len(assays) - len(present)
    if missing:
        print(f"  Note: {missing} assays from the assay list are not in the dataset.")
    return present


def validate_panel(proteins: list[str], df: pd.DataFrame, name: str) -> list[str]:
    """Keep only proteins present in ``df``; warn about any that are missing."""
    present = [p for p in proteins if p in df.columns]
    dropped = sorted(set(proteins) - set(present))
    if dropped:
        print(f"  Warning: {len(dropped)} proteins in the {name} panel are not in the "
              f"dataset and were dropped: {', '.join(dropped[:10])}")
    return present


def baseline_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Subjects with an admission CT and Olink data (n = 123 in the manuscript)."""
    return df[df[config.COL_CE_BASELINE].notna()].copy()


def worsening_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Subjects with a 6-hour follow-up CT (n = 116 in the manuscript)."""
    return df[df[config.COL_WORSE].notna()].copy()


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (and parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

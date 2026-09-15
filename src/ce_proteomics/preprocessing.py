"""
Preprocessing: from raw Olink and REDCap exports to one analysis table.

Steps
-----
1. ``pivot_olink_long_to_wide``: remove control assays and pivot the Olink
   long export to one row per sample with one column per assay (NPX).
2. ``load_redcap_export``: read a REDCap DATA_LABELS export, strip label
   whitespace, and standardize the subject identifier column.
3. ``build_analysis_dataset``: join the admission (baseline) CT row, the
   6-hour CT row, the clinical variables, and the proteomic data into a single
   one-row-per-subject table with explicit outcome columns:

   ==================  =============================================
   Column              Meaning
   ==================  =============================================
   CE Baseline         CE on the admission CT (0/1)
   CE 6hr              CE on the 6-hour CT (0/1; NaN if no follow-up)
   Edema Worse         CE worsened admission -> 6 hr (0/1; NaN if no follow-up)
   Time to Scan        Hours from injury to the admission CT
   ==================  =============================================

   Keeping baseline and 6-hour CE in separate, explicitly named columns avoids
   the ambiguity of a single "Cerebral Edema" column whose meaning depends on
   which scan row it came from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# =============================================================================
# Olink
# =============================================================================

def pivot_olink_long_to_wide(df_long: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Pivot an Olink long-format NPX export to a wide sample x assay matrix.

    Control assays (``AssayType`` ending in ``config.OLINK_CONTROL_SUFFIX``)
    are removed before pivoting. Duplicate sample x assay entries are averaged
    (``pivot_table`` default), which is a no-op when each sample was run once.

    Returns
    -------
    df_wide : DataFrame with a ``SampleID`` column followed by one column per assay
    assays : list of assay names in column order
    """
    df = df_long.copy()
    if config.OLINK_ASSAY_TYPE_COL in df.columns:
        is_control = (
            df[config.OLINK_ASSAY_TYPE_COL].astype(str).str.endswith(config.OLINK_CONTROL_SUFFIX)
        )
        n_control = int(is_control.sum())
        df = df[~is_control]
        print(f"  Removed {n_control} control-assay rows.")
    else:
        print(f"  Note: no '{config.OLINK_ASSAY_TYPE_COL}' column; no control rows removed.")

    df_wide = df.pivot_table(
        index=config.OLINK_SAMPLE_COL,
        columns=config.OLINK_ASSAY_COL,
        values=config.OLINK_NPX_COL,
    )
    df_wide.columns.name = None
    assays = [str(c) for c in df_wide.columns]
    df_wide = df_wide.reset_index().rename(columns={config.OLINK_SAMPLE_COL: config.COL_ID})
    df_wide[config.COL_ID] = df_wide[config.COL_ID].astype(str).str.strip()
    print(f"  Pivoted to {df_wide.shape[0]} samples x {len(assays)} assays.")
    return df_wide, assays


# =============================================================================
# REDCap
# =============================================================================

def _strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _standardize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whichever identifier column is present to ``config.COL_ID``."""
    for candidate in config.REDCAP_ID_CANDIDATES:
        if candidate in df.columns:
            df = df.rename(columns={candidate: config.COL_ID})
            break
    else:
        raise ValueError(
            "No subject identifier column found. Expected one of: "
            + ", ".join(config.REDCAP_ID_CANDIDATES)
        )
    df[config.COL_ID] = df[config.COL_ID].astype(str).str.strip()
    return df


def load_redcap_export(path: Path | str) -> pd.DataFrame:
    """Read a REDCap DATA_LABELS export with cleaned column names and IDs."""
    df = pd.read_csv(path)
    df = _strip_column_names(df)
    df = _standardize_id_column(df)
    return df


def code_binary(series: pd.Series) -> pd.Series:
    """
    Recode a REDCap label column (Present/Absent, Yes/No, 1/0) to float 1/0.

    Numeric input is passed through. Unrecognized labels become NaN and are
    reported so that unexpected coding is not silently dropped.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[text.isin(config.BINARY_POSITIVE_LABELS)] = 1.0
    out[text.isin(config.BINARY_NEGATIVE_LABELS)] = 0.0
    unrecognized = sorted(
        set(text[out.isna() & series.notna() & ~text.isin({"nan", ""})])
    )
    if unrecognized:
        print(f"  Warning: unrecognized labels in '{series.name}' set to NaN: "
              f"{', '.join(unrecognized[:10])}")
    return out


def code_sex(series: pd.Series) -> pd.Series:
    """Recode sex labels using ``config.SEX_MAP`` (numeric input passes through)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    mapped = series.astype(str).str.strip().map(config.SEX_MAP)
    unknown = sorted(set(series[mapped.isna() & series.notna()].astype(str)))
    if unknown:
        print(f"  Warning: sex labels not in SEX_MAP set to NaN: {', '.join(unknown[:10])}")
    return mapped.astype(float)


def _apply_column_map(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """Rename REDCap labels to analysis names (keys are matched after strip)."""
    stripped_map = {k.strip(): v for k, v in column_map.items()}
    return df.rename(columns={c: stripped_map[c] for c in df.columns if c in stripped_map})


# =============================================================================
# Analysis dataset
# =============================================================================

def _split_ct_rows(ct: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ct = _apply_column_map(ct, config.CT_COLUMN_MAP)
    if config.CT_TYPE_COL not in ct.columns:
        raise ValueError(f"CT export is missing the '{config.CT_TYPE_COL}' column.")
    scan_type = ct[config.CT_TYPE_COL].astype(str).str.strip().str.lower()

    baseline = ct[scan_type == config.CT_BASELINE_LABEL].copy()
    six_hr = ct[scan_type == config.CT_6HR_LABEL].copy()

    for name, frame in [("baseline", baseline), ("6-hour", six_hr)]:
        if frame[config.COL_ID].duplicated().any():
            dupes = frame.loc[frame[config.COL_ID].duplicated(), config.COL_ID].unique()
            raise ValueError(
                f"Duplicate subject IDs among {name} CT rows: {', '.join(map(str, dupes[:10]))}"
            )

    baseline_out = pd.DataFrame({
        config.COL_ID: baseline[config.COL_ID].values,
        config.COL_CE_BASELINE: code_binary(baseline["Cerebral Edema"]).values,
        config.COL_TIME_TO_CT: pd.to_numeric(baseline["Time to Scan"], errors="coerce").values,
    })
    six_hr_out = pd.DataFrame({
        config.COL_ID: six_hr[config.COL_ID].values,
        config.COL_CE_6HR: code_binary(six_hr["Cerebral Edema"]).values,
        config.COL_WORSE: code_binary(six_hr["Edema Worse"]).values,
    })
    return baseline_out, six_hr_out


def _prepare_clinical(clinical: pd.DataFrame) -> pd.DataFrame:
    clinical = _apply_column_map(clinical, config.CLINICAL_COLUMN_MAP)
    if clinical[config.COL_ID].duplicated().any():
        dupes = clinical.loc[clinical[config.COL_ID].duplicated(), config.COL_ID].unique()
        raise ValueError(
            f"Duplicate subject IDs in the clinical export: {', '.join(map(str, dupes[:10]))}"
        )
    if config.COL_SEX in clinical.columns:
        clinical[config.COL_SEX] = code_sex(clinical[config.COL_SEX])
    if config.COL_AGE in clinical.columns:
        clinical[config.COL_AGE] = pd.to_numeric(clinical[config.COL_AGE], errors="coerce")

    # Keep only the analysis columns we know how to use.
    keep = [config.COL_ID] + [
        c for c in [config.COL_AGE, config.COL_SEX] + config.CLINICAL_FEATURES_CANDIDATE
        if c in clinical.columns
    ]
    keep = list(dict.fromkeys(keep))
    clinical = clinical[keep].copy()

    # Coerce clinical columns to numeric where possible; binary and ordinal
    # labels are recoded.
    for col in keep[1:]:
        if col in (config.COL_AGE, config.COL_SEX):
            continue
        if pd.api.types.is_numeric_dtype(clinical[col]):
            continue
        text = clinical[col].astype(str).str.strip().str.lower()
        if col in config.ORDINAL_MAPS:
            mapping = {k.lower(): v for k, v in config.ORDINAL_MAPS[col].items()}
            recoded = text.map(mapping)
            unknown = sorted(set(text[recoded.isna() & clinical[col].notna()]))
            if unknown:
                print(f"  Warning: '{col}' labels not in ORDINAL_MAPS set to NaN: {', '.join(unknown[:10])}")
            clinical[col] = recoded.astype(float)
            continue
        if text[clinical[col].notna()].isin(
            config.BINARY_POSITIVE_LABELS | config.BINARY_NEGATIVE_LABELS
        ).all():
            clinical[col] = code_binary(clinical[col])
        else:
            coerced = pd.to_numeric(clinical[col], errors="coerce")
            if coerced.notna().sum() >= clinical[col].notna().sum() * 0.9:
                clinical[col] = coerced
            else:
                print(f"  Note: clinical column '{col}' left as text (non-numeric labels); "
                      "recode it in CLINICAL_COLUMN_MAP / preprocessing if it is a model feature.")
    return clinical


def build_analysis_dataset(
    olink_wide: pd.DataFrame,
    clinical: pd.DataFrame,
    ct: pd.DataFrame,
    assays: list[str],
) -> pd.DataFrame:
    """
    Build the one-row-per-subject analysis table.

    The cohort is defined by subjects who have BOTH an admission CT row and
    Olink data (inner join); the 6-hour CT and clinical variables are then
    left-joined so that subjects without a follow-up scan are retained with
    NaN outcomes.
    """
    baseline, six_hr = _split_ct_rows(ct)
    clinical = _prepare_clinical(clinical)

    df = baseline.merge(olink_wide, on=config.COL_ID, how="inner", validate="one_to_one")
    df = df.merge(six_hr, on=config.COL_ID, how="left", validate="one_to_one")
    df = df.merge(clinical, on=config.COL_ID, how="left", validate="one_to_one")

    # Required columns
    required = [config.COL_CE_BASELINE, config.COL_AGE, config.COL_SEX, config.COL_TIME_TO_CT]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Analysis dataset is missing required columns: " + ", ".join(missing)
            + ". Check CLINICAL_COLUMN_MAP and CT_COLUMN_MAP in config.py."
        )
    missing_clinical = [c for c in config.CLINICAL_FEATURES_FINAL if c not in df.columns]
    if missing_clinical:
        raise ValueError(
            "The following clinical model features are not in the dataset: "
            + ", ".join(missing_clinical)
            + ". Add their REDCap labels to CLINICAL_COLUMN_MAP in config.py."
        )

    # Column order: id, outcomes, covariates, clinical, proteins
    clinical_cols = [c for c in config.CLINICAL_FEATURES_CANDIDATE
                     if c in df.columns and c not in (config.COL_AGE, config.COL_SEX)]
    ordered = (
        [config.COL_ID, config.COL_CE_BASELINE, config.COL_CE_6HR, config.COL_WORSE,
         config.COL_AGE, config.COL_SEX, config.COL_TIME_TO_CT]
        + clinical_cols
        + [a for a in assays if a in df.columns]
    )
    df = df[ordered]

    n_base = int(df[config.COL_CE_BASELINE].notna().sum())
    n_follow = int(df[config.COL_WORSE].notna().sum())
    print(f"  Analysis dataset: {len(df)} subjects with baseline CT + Olink; "
          f"{n_base} with baseline CE coded; {n_follow} with 6-hour worsening coded.")
    return df


def missingness_report(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Count missing values per column (used to document the missingness check)."""
    rows = [{"Column": c, "N missing": int(df[c].isna().sum()), "N": int(len(df))}
            for c in columns if c in df.columns]
    return pd.DataFrame(rows)

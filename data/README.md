# Data directory

Patient-level data are not distributed with this repository (see the Data
Availability statement in the article). Nothing under `data/` is tracked by
git except this file and the `.gitkeep` placeholders.

## data/raw/ (inputs; place your local copies here)

| File (default name in `config.py`) | Content |
| --- | --- |
| `NEW_HE_olink_only.csv` | Olink Explore HT long-format NPX export: one row per sample x assay with `SampleID`, `Assay`, `AssayType`, `NPX` (plus `OlinkID`, `SampleQC`, ... ). Control assays have `AssayType` ending in `_ctrl`. |
| `PredictorsOfLowriskP-CerebralEdemaVariabl_DATA_LABELS_....csv` | REDCap clinical export (labels): one row per subject with `Record ID`, `Age upon admission:`, `Gender:`, and the admission clinical variables. |
| `PredictorsOfLowriskP-CerebralEdemaCTsForH_DATA_LABELS_....csv` | REDCap CT export (labels): one row per subject per scan with `Head CT Type` (`Baseline` / `6 Hour`), `Elapsed time between trauma and Head CT Type`, `Cerebral edema:` (`Present` / `Absent`), `If previous scan, cerebral edema worse?` (`Yes` / `No`). |

File names and column labels are set in `src/ce_proteomics/config.py`
(`OLINK_*`, `REDCAP_*`, `CT_COLUMN_MAP`, `CLINICAL_COLUMN_MAP`).

## data/processed/ (created by `scripts/01_build_analysis_dataset.py`)

| File | Content |
| --- | --- |
| `olink_npx_wide.csv` | one row per sample, one column per assay (NPX) |
| `olink_assays.txt` | assay (protein) column names, one per line |
| `analysis_dataset.csv` | one row per subject: `SampleID`, `CE Baseline`, `CE 6hr`, `Edema Worse`, `Age`, `Sex`, `Time to Scan`, clinical variables, proteins |
| `missingness_report.csv` | missing-value counts for outcomes, covariates, and clinical variables |

## Synthetic data

`python tests/make_synthetic_data.py --out-dir data/synthetic_raw` writes
random data in the same layout (no real participants) so the pipeline can be
exercised without access to the cohort.

# Cerebral-Edema

Analysis code for:

> Radabaugh HL, Abdelhak A, Ning K, Jha RM, Rowell SE, Pollock JM, Mendoza E,
> Rojas-Valencia LM, Ferguson AR, Hinson HE. **Protein Predictors of Worsening
> Cerebral Edema in Traumatic Brain Injury in Critically Ill Patients.**
> *In review* (2026).

Plasma proteomics (Olink Explore HT, 5,394 proteins) measured within 3 hours
of moderate-to-severe TBI in the PROTIPS cohort (n = 123) were used to
(1) derive a parsimonious admission-time biomarker panel plus a supervised
model for radiographic cerebral edema (CE) worsening at 6 hours, and
(2) characterize the mechanistic architecture of CE worsening through a
broader elastic-net-derived protein panel, PCA, random-forest SHAP
attribution, and functional network analysis.

Patient-level data are available from the corresponding author on reasonable
request and are not distributed here. A synthetic data generator is included so
the full pipeline can be run without access to the cohort.

## Repository layout

```
Cerebral-Edema/
  scripts/                 numbered entry points, run in order
    01_build_analysis_dataset.py     Olink + REDCap -> one row per subject
    02_univariate_screen.py          12-protein panel (Fig. 2, Supp. Table 2)
    03_elastic_net_panel.py          60-protein panel
    04_classification.py             CE worsening prediction (Fig. 3, 6A; Supp. Fig. 4, 5)
    05_mechanistic_analysis.py       DE, PCA, SHAP, interactions (Fig. 4, 5, 6B-C; Supp. Fig. 6-8)
    06_glucose_alcohol_diagnostics.py  Supp. Fig. 3
    run_all.sh
  R/
    01_olink_qc.R                    Example Olink QC plots (Supp. Fig. 1, 2; OlinkAnalyze)
  src/ce_proteomics/       analysis package (imported by the scripts)
    config.py                        paths, column labels, every tunable constant
    preprocessing.py, univariate.py, elastic_net.py, pca.py,
    classification.py, classification_plots.py, interpretation.py,
    enrichment.py, diagnostics.py, plotting.py, io.py, compat.py
  extras/                  exploratory analyses not used in the article
  tests/                   synthetic data generator and end-to-end smoke test
  data/                    raw inputs and processed dataset (not tracked)
  results/                 all outputs (not tracked)
```

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/Hinson-Lab/Cerebral-Edema.git
cd Cerebral-Edema
pip install -r requirements.txt
```

The scripts add `src/` to `sys.path` themselves, so no install step is
required. `pip install -e .` also works if you prefer to import
`ce_proteomics` from elsewhere (for example from Spyder).

The R script needs R 4.x with `OlinkAnalyze`, `dplyr`, `ggplot2`, and `readr`.

## Running the pipeline

1. Place the three raw exports in `data/raw/` (see `data/README.md`).
2. Open `src/ce_proteomics/config.py` and make sure the file names and the
   REDCap column labels match your exports. `CLINICAL_COLUMN_MAP` must map
   the REDCap labels of the admission clinical variables (GCS, pupil
   reactivity, glucose, alcohol, and the other Table 1 variables) to the
   analysis names used in `CLINICAL_FEATURES_FINAL` and
   `CLINICAL_FEATURES_CANDIDATE`; the dataset builder stops with a clear
   message if a required variable is missing.
3. Run the scripts in order, or `bash scripts/run_all.sh`.

Each script has `--help`. Every script accepts `--dataset` and
`--results-dir`, and the resampling-heavy steps accept overrides
(`--n-bootstrap`, `--pca-bootstrap`, `--n-repeats`, `--rfe-trees`) for quick
test runs. The manuscript values are the defaults in `config.py`.

Step 03 (1,000 bootstrap elastic-net fits over ~5,400 proteins) takes hours
on a laptop; everything else finishes in minutes.

### Try it on synthetic data

```bash
python tests/make_synthetic_data.py --out-dir data/synthetic_raw
python scripts/01_build_analysis_dataset.py \
    --olink data/synthetic_raw/NEW_HE_olink_only.csv \
    --clinical "data/synthetic_raw/PredictorsOfLowriskP-CerebralEdemaVariabl_DATA_LABELS_2025-05-02_1324.csv" \
    --ct "data/synthetic_raw/PredictorsOfLowriskP-CerebralEdemaCTsForH_DATA_LABELS_2025-05-02_1323.csv"
python scripts/02_univariate_screen.py
python scripts/03_elastic_net_panel.py --n-bootstrap 20
python scripts/04_classification.py --n-bootstrap 200
python scripts/05_mechanistic_analysis.py
python scripts/06_glucose_alcohol_diagnostics.py
```

`pytest tests/` runs the same sequence with small resampling counts as a
smoke test.

## Where each figure and table comes from

| Article item | Script | Output (under `results/`) |
| --- | --- | --- |
| Table 1, Supp. Table 1 | not in this repository (descriptive tables) | |
| Fig. 2 | `02_univariate_screen.py` | `02_univariate/fig2_panel_boxplots.png` |
| Supp. Table 2 | `02_univariate_screen.py` | `02_univariate/supp_table_2_panel_associations.csv` |
| 12-protein panel | `02_univariate_screen.py` | `panels/panel_12_fdr_logistic.txt` |
| 60-protein panel | `03_elastic_net_panel.py` | `panels/panel_60_elastic_net.txt`, `03_elastic_net/elastic_net_stability.csv` |
| Fig. 3A-B | `04_classification.py` | `04_classification/12-protein_plus_clinical/roc_curves.png`, `roc_auc_forest.png` |
| Fig. 3C-D | `04_classification.py` | `04_classification/clinical_only/roc_curves.png`, `roc_auc_forest.png` |
| DeLong test (Results) | `04_classification.py` | `04_classification/delong_tests.csv` |
| RFECV clinical selection (Methods) | `04_classification.py` | `04_classification/12-protein_plus_clinical_rfecv/rfecv_selection_frequency.csv` |
| Supp. Fig. 4 | `04_classification.py` | `.../pr_auc_forest.png` (both feature sets) |
| Supp. Fig. 5 | `04_classification.py` | `04_classification/12-protein_plus_clinical/calibration_all_models.png` |
| Fig. 6A | `04_classification.py` | `04_classification/60-protein_only/roc_curves.png` |
| Fig. 4 | `05_mechanistic_analysis.py` | `05_mechanistic/fig4_volcano_worsening.png` |
| Fig. 5 | `05_mechanistic_analysis.py` | `05_mechanistic/fig5_pca_panel.png` (and single panels) |
| Supp. Fig. 6 | `05_mechanistic_analysis.py` | `05_mechanistic/supp_fig6a_scree.png`, `supp_fig6_pc1/pc2_eigenvector_loadings.png` |
| Fig. 6B-C | `05_mechanistic_analysis.py` | `05_mechanistic/fig6bc_shap_panel.png` |
| Supp. Fig. 7 | `05_mechanistic_analysis.py` | `05_mechanistic/supp_fig7_rf_permutation_importance.png` |
| Supp. Fig. 8 | `05_mechanistic_analysis.py` | `05_mechanistic/supp_fig8_shap_interaction_heatmap.png` (+ CSVs of the values) |
| Fig. 7 (inputs) | `05_mechanistic_analysis.py` | `05_mechanistic/shap_proteins_toward_worsening.txt`, `..._toward_non_worsening.txt` |
| Supp. Fig. 3 | `06_glucose_alcohol_diagnostics.py` | `06_diagnostics/glucose_by_edema_trajectory.png`, `alcohol_modified_z_by_trajectory.png` |
| Supp. Fig. 1, 2 | `R/01_olink_qc.R` | `01_olink_qc/` |

Fig. 7 itself was produced with the Gladstone Bioinformatics Core
[Interactive Enrichment Analysis](https://github.com/gladstone-institutes/Interactive-Enrichment-Analysis)
tool from the exported SHAP-direction protein lists.

## Methods summary (what the code does)

**Analysis dataset.** Control assays are removed from the Olink long export,
which is pivoted to one row per sample. REDCap CT rows are split into the
admission scan (`CE Baseline`, `Time to Scan`) and the 6-hour scan
(`CE 6hr`, `Edema Worse`), joined with the clinical variables and proteomics,
and written as one row per subject. Subjects without a follow-up scan keep
`NaN` outcomes; they are used in the baseline analyses (n = 123) and excluded
from the worsening analyses (n = 116).

**Goal 1, 12-protein panel.** One logistic regression per protein with CE on
the admission CT as the outcome, adjusted for age, sex, and time to the
baseline CT; Benjamini-Hochberg FDR across all proteins; q < 0.05.

**Goal 2, 60-protein panel.** Proteins are residualized on the same
covariates and standardized; elastic-net logistic regression (l1_ratio 0.5,
`LogisticRegressionCV` over 10 C values with 5-fold inner CV) is fit on 1,000
stratified bootstrap resamples; proteins with non-zero coefficients in more
than 70 percent of resamples are retained.

**Supervised classification.** Six classifiers (ridge logistic regression,
elastic-net logistic regression, LDA, random forest, RBF SVM, XGBoost) with
fixed hyperparameters, stratified 5-fold cross-validation (repeatable through
`CV_N_REPEATS`). Standardization, XGBoost class weighting, and clinical
feature selection are all fit inside the training folds. Clinical selection
uses recursive feature elimination with a random-forest estimator and inner
5-fold CV, with the 12-protein panel held fixed and only clinical candidates
eliminated; per-fold selections and selection frequencies are written out.
The reported clinical-proteomic models use the fixed final clinical set
(`CLINICAL_FEATURES_FINAL`); the clinical-only comparison uses the complete
candidate set. Performance comes from pooled out-of-fold probabilities:
ROC-AUC, PR-AUC (average precision, no-skill = prevalence), subject-level
bootstrap 95% CIs, the post-hoc Youden operating point with bootstrap CIs,
Brier score, calibration slope and intercept by logistic recalibration, and
DeLong tests between the clinical-proteomic and clinical-only models.

**Mechanistic analysis.** Welch's t-test differential expression with BH
adjustment and a volcano plot; PCA on z-scored NPX (all 123 subjects) with
correlation loadings, syndromic loading plots, Welch's t-tests of PC scores by
6-hour worsening, and bootstrap loading stability; a random forest on the
60-protein panel (worsening outcome) with MDI and permutation importance,
TreeSHAP values, a direction summary (correlation of protein level with SHAP
value), SHAP-direction protein lists for enrichment, and pairwise TreeSHAP
interaction values from the same fitted model restricted for display to the
top 15 proteins by mean |SHAP|.

All thresholds and resampling counts are named constants in
`src/ce_proteomics/config.py`.

## Reproducibility notes

* Random seeds are fixed (`RANDOM_STATE = 42`); bootstrap and CV seeds are
  derived from it. Results will still differ slightly across library
  versions (XGBoost, shap, scikit-learn).
* `compat.py` hides the scikit-learn 1.8 API change for penalized logistic
  regression so the code runs on scikit-learn 1.3 through 1.8.
* Setting `SCALE_WITHIN_FOLDS = False` (or `--no-scale-within-folds`)
  reproduces an earlier version of the classification in which the scaler
  was fit once on the full analytic cohort; tree-based models are unaffected
  by this choice.

## Citation

See `CITATION.cff`. Please cite the article when using this code.

## License

MIT (see `LICENSE`).

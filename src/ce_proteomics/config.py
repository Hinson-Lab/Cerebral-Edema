"""
Central configuration for the PROTIPS cerebral edema proteomics analysis.

Everything that you might need to change lives here: file locations,
column names in the raw exports, and every tunable analysis constant.
Analysis modules import these values rather than hard-coding them, so a
change made here propagates through the whole pipeline.

Sections
--------
1. Paths
2. Raw input file names
3. Column names in the raw exports and the analysis dataset
4. Feature sets used for supervised classification
5. Analysis constants (statistics, resampling, model hyperparameters)
6. Plot constants
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# 1. PATHS
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "results"

# =============================================================================
# 2. RAW INPUT FILES (place these in data/raw/ ...)

# Olink Explore HT long-format NPX export (one row per sample x assay).
OLINK_LONG_FILE = DATA_RAW / "Proteomics_Data_Raw.csv"

# REDCap "DATA_LABELS" exports.
# Check that variable names match analysis script naming convention (below).
REDCAP_CLINICAL_FILE = (
    DATA_RAW / "REDCAP_Data_Clinical.csv"
)
REDCAP_CT_FILE = (
    DATA_RAW / "REDCAP_Data_Imaging.csv"
)

# =============================================================================
# PROCESSED FILES AND PANEL LISTS (created by the pipeline)
# =============================================================================
OLINK_WIDE_FILE = DATA_PROCESSED / "olink_npx_wide.csv"
OLINK_ASSAY_LIST_FILE = DATA_PROCESSED / "olink_assays.txt"
ANALYSIS_DATASET_FILE = DATA_PROCESSED / "analysis_dataset.csv"

PANEL_DIR = RESULTS / "panels"
PANEL_12_FILE = PANEL_DIR / "panel_12_fdr_logistic.txt"
PANEL_60_FILE = PANEL_DIR / "panel_60_elastic_net.txt"

# =============================================================================
# 3. COLUMN NAMES
# =============================================================================
# --- Olink long export ------------------------------------------------------
OLINK_SAMPLE_COL = "SampleID"
OLINK_ASSAY_COL = "Assay"          # gene symbol; OlinkID would also work
OLINK_NPX_COL = "NPX"
OLINK_ASSAY_TYPE_COL = "AssayType"
OLINK_CONTROL_SUFFIX = "_ctrl"     # rows whose AssayType ends with this are removed

# --- REDCap exports ----------------------------------------------------------
# Subject identifier used to join the REDCap exports to the Olink data.
# The REDCap exports may name this column "Record ID"; both are accepted.
REDCAP_ID_CANDIDATES = ("SampleID", "Record ID")

# CT export: one row per subject per scan.
CT_TYPE_COL = "Head CT Type"
CT_BASELINE_LABEL = "baseline"     # compared after strip().lower()
CT_6HR_LABEL = "6 hour"

# REDCap label -> analysis column name. Keys are matched after stripping
# surrounding whitespace (the raw export contains a trailing space in the
# elapsed-time label).
CT_COLUMN_MAP = {
    "Elapsed time between trauma and Head CT Type": "Time to Scan",
    "Cerebral edema:": "Cerebral Edema",
    "If previous scan, cerebral edema worse?": "Edema Worse",
}

# Clinical export. The outcome labels (above) are defined as such.
# The remaining REDCap labels (below)
# All MUST be filled in to match the local export before running
# scripts/01_build_analysis_dataset.py (the builder validates them).
CLINICAL_COLUMN_MAP = {
    "Age upon admission:": "Age",
    "Gender:": "Sex",
    # "<REDCap label for admission total GCS>": "Admission GCS",
    # "<REDCap label for GCS motor>": "GCS Motor",
    # "<REDCap label for pupil reactivity>": "Pupil Reactivity",
    # "<REDCap label for admission glucose>": "Glucose",
    # "<REDCap label for admission blood alcohol>": "Alcohol",
    # "<REDCap label for ISS>": "ISS",
    # "<REDCap label for AIS head>": "AIS",
    # "<REDCap label for hypotension>": "Hypotension",
    # "<REDCap label for hypoxia>": "Hypoxia",
    # "<REDCap label for hemoglobin>": "Hemoglobin",
    # "<REDCap label for platelets>": "Platelets",
    # "<REDCap label for INR>": "INR",
    # "<REDCap label for fibrinogen>": "Fibrinogen",
}

# Value recoding for categorical clinical variables. Update with local variable name.
SEX_MAP = {"Cis Male": 0, "Cis Female": 1}
# Ordinal recoding for label-valued clinical variables (matched case-insensitively
# after stripping whitespace). Columns already numeric are left unchanged.
ORDINAL_MAPS = {
    "Pupil Reactivity": {"both reactive": 2, "one reactive": 1, "neither reactive": 0},
}
BINARY_POSITIVE_LABELS = {"present", "yes", "true", "1", "1.0"}
BINARY_NEGATIVE_LABELS = {"absent", "no", "false", "0", "0.0"}

# --- Analysis dataset (one row per subject) ---------------------------------
COL_ID = "SampleID"
COL_CE_BASELINE = "CE Baseline"     # CE on the admission CT (0/1), n = 123
COL_CE_6HR = "CE 6hr"               # CE on the 6-hour CT (0/1)
COL_WORSE = "Edema Worse"           # CE worsening admission -> 6 hr (0/1), n = 116
COL_AGE = "Age"
COL_SEX = "Sex"
COL_TIME_TO_CT = "Time to Scan"     # hours from injury to the baseline CT

# Covariates used for adjustment in the univariate screen and for
# residualization before elastic-net selection.
ADJUSTMENT_COVARIATES = [COL_AGE, COL_SEX, COL_TIME_TO_CT]

# =============================================================================
# 4. FEATURE SETS FOR SUPERVISED CLASSIFICATION
# =============================================================================
# Clinical variables retained after RFECV (random-forest estimator, run within
# training folds) as reported in the manuscript Methods.
CLINICAL_FEATURES_FINAL = [
    "Age",
    "Sex",
    "Admission GCS",
    "Pupil Reactivity",
    "Glucose",
    "Alcohol",
]

# Complete candidate clinical set (Table 1) used for the clinical-only models.
# Only columns present in the analysis dataset are used; missing names are
# reported by the classification script.
CLINICAL_FEATURES_CANDIDATE = [
    "Age",
    "Sex",
    "Admission GCS",
    "GCS Motor",
    "ISS",
    "AIS",
    "Pupil Reactivity",
    "Hypotension",
    "Hypoxia",
    "Glucose",
    "Hemoglobin",
    "Alcohol",
    "Platelets",
    "INR",
    "Fibrinogen",
]

# Feature sets to evaluate, in the order they are run. Definitions live in
# scripts/04_classification.py:
#   "12-Protein + Clinical (RFECV)"  fixed 12-protein panel + clinical candidates
#                                    selected by RFECV inside each training fold
#                                    (model development; logs selection frequency)
#   "12-Protein + Clinical"          fixed 12-protein panel + CLINICAL_FEATURES_FINAL
#                                    (the reported clinical-proteomic models, Fig. 3A-B)
#   "Clinical Only"                  complete candidate clinical set (Fig. 3C-D)
#   "60-Protein Only"                elastic-net panel alone (Fig. 6A)
FEATURE_SETS_TO_RUN = [
    "12-Protein + Clinical (RFECV)",
    "12-Protein + Clinical",
    "Clinical Only",
    "60-Protein Only",
]

# =============================================================================
# 5. ANALYSIS CONSTANTS
# =============================================================================
RANDOM_STATE = 42

# Multiple-testing control (Benjamini-Hochberg) used throughout.
FDR_ALPHA = 0.05

# --- Goal 1: univariate logistic screen -------------------------------------
# Per-protein logistic regression adjusted for ADJUSTMENT_COVARIATES;
# proteins with BH q < FDR_ALPHA form the 12-protein panel.
PANEL_BOXPLOT_EQUAL_VAR = True   # Student's t-test for the Fig. 2 asterisks

# --- Goal 2: bootstrapped elastic-net selection -----------------------------
EN_N_BOOTSTRAP = 1000            # manuscript: 1,000 resamples
EN_SELECTION_FREQUENCY = 0.70    # manuscript: retained if selected in > 70%
EN_L1_RATIO = 0.5
EN_N_CS = 10                     # LogisticRegressionCV regularization grid size
EN_INNER_CV = 5
EN_SCORING = "accuracy"
EN_MAX_ITER = 10000
EN_COEF_TOLERANCE = 1e-6         # |coef| above this counts as "selected"

# --- Supervised classification ----------------------------------------------
CV_N_SPLITS = 5
CV_N_REPEATS = 1                 # set > 1 for repeated stratified CV
SCALE_WITHIN_FOLDS = True        # StandardScaler fit on training folds only
N_BOOTSTRAP_CI = 1000            # subject-level bootstrap of pooled OOF predictions
BOOTSTRAP_CI_LEVEL = 0.95
N_CALIBRATION_BINS = 5
YOUDEN_METHOD = "youden"
DELONG_REFERENCE_SET = "12-Protein + Clinical"
DELONG_COMPARISON_SET = "Clinical Only"

# Clinical feature selection: recursive feature elimination with a random-forest
# estimator and inner cross-validation, run inside each outer training fold with
# the 12-protein panel held fixed (only clinical candidates are eliminated).
RFE_RF_N_ESTIMATORS = 200
RFE_INNER_CV = 5
RFE_SCORING = "roc_auc"
RFE_MIN_CLINICAL = 0             # elimination continues down to this many candidates

# Classifier hyperparameters (fixed; no nested tuning in this code base).
CLF_LOGREG_MAX_ITER = 1000
CLF_EN_MAX_ITER = 2000
CLF_EN_L1_RATIO = 0.5
CLF_RF_N_ESTIMATORS = 500
CLF_XGB_N_ESTIMATORS = 100
CLF_XGB_MAX_DEPTH = 4
CLF_XGB_LEARNING_RATE = 0.1

# --- PCA on the 60-protein panel ---------------------------------------------
PCA_N_COMPONENTS = 5
PCA_BOOTSTRAP_N = 500
PCA_LOADING_THRESHOLD = 0.3      # |correlation loading| for "axis-defining"
PCA_STABILITY_THRESHOLD = 0.3    # fraction of bootstraps above threshold
SYNDROMIC_LOADING_CUTOFF = 0.4   # loadings shown in the Fig. 5C-D syndromic plots

# --- Random forest used for interpretation (Fig. 6B-C, Supp. Fig. 7-8) ------
INTERP_RF_N_ESTIMATORS = 500
INTERP_RF_MAX_DEPTH = 10
INTERP_RF_MIN_SAMPLES_LEAF = 5
INTERP_RF_CV_FOLDS = 5
PERMUTATION_N_REPEATS = 20
PERMUTATION_TEST_SIZE = 0.3
SHAP_TOP_N = 15                  # proteins shown in Fig. 6B-C and Supp. Fig. 8
SHAP_LIST_TOP_N = 20             # proteins ranked before splitting by direction
SHAP_LIST_DIRECTION_CORR = 0.15  # |corr(feature, SHAP)| to assign a direction
SHAP_ENRICH_DIRECTION_CORR = 0.30  # stricter cut used for STRING enrichment
STRING_SPECIES = 9606
STRING_FDR = 0.05

# --- Supplementary Fig. 3 (glucose / alcohol diagnostics) -------------------
GLUCOSE_UNIT = "mg/dL"
ALCOHOL_UNIT = "mg/dL"
EXTREME_VALUE_IQR_MULTIPLIER = 3.0   # Q3 + 3 x IQR
MODIFIED_Z_THRESHOLD = 3.5
MODIFIED_Z_CONSTANT = 0.6744897501960817
ALCOHOL_HIGH_CUTOFF = 300
DIAGNOSTIC_JITTER_SEED = 20260804

# =============================================================================
# 6. PLOT CONSTANTS
# =============================================================================
FIG_DPI = 300
FIG_FORMAT = "png"

# Colorblind-friendly classifier colors (Okabe-Ito subset plus dark gray).
# Green is intentionally omitted so no overlaid plot pairs red with green.
CLASSIFIER_COLORS = {
    "Logistic Regression": "#0072B2",
    "Elastic Net": "#E69F00",
    "LDA": "#56B4E9",
    "Random Forest": "#CC79A7",
    "SVM (RBF)": "#D55E00",
    "XGBoost": "#4D4D4D",
}

# Group colors
COLOR_CE_ABSENT = "#66C2A5"      # seaborn Set2 green
COLOR_CE_PRESENT = "#FC8D62"     # seaborn Set2 orange
COLOR_NO_WORSENING = "#66C2A5"
COLOR_WORSENING = "#D9895B"

# SHAP direction colors (Fig. 6B)
COLOR_SHAP_WORSENING = "#F39C3A"       # orange
COLOR_SHAP_NON_WORSENING = "#5DA9A6"   # teal

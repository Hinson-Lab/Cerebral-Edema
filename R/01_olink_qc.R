# ---------------------------------------------------------------------------
# 01. Olink Explore HT sample quality control (Supplementary Fig. 1 and 2)
#
# Sample-level NPX distributions colored by Olink sample QC status and by
# hemolysis index (Supp. Fig. 1A-B), and a PCA of the complete NPX matrix
# colored by QC status with outlier lines and labels (Supp. Fig. 2), using
# the OlinkAnalyze package.
#
# Inputs
#   data/raw/<Olink long-format NPX export>   (see OLINK_LONG_FILE in
#                                             src/ce_proteomics/config.py)
#   A per-sample hemolysis index. If it is not already a column of the NPX
#   export, supply a two-column CSV (SampleID, Hemolysis) via HEMOLYSIS_FILE.
#
# Outputs (results/01_olink_qc/)
#   supp_fig1a_npx_distribution_by_sampleqc.png
#   supp_fig1b_npx_distribution_by_hemolysis.png
#   supp_fig2_pca_by_sampleqc.png
#   sample_qc_summary.csv
#
# Usage:  Rscript R/01_olink_qc.R
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(OlinkAnalyze)
  library(dplyr)
  library(ggplot2)
  library(readr)
})

NPX_FILE       <- "data/raw/NEW_HE_olink_only.csv"
HEMOLYSIS_FILE <- NA_character_   # e.g. "data/raw/hemolysis_index.csv"; NA if already in NPX_FILE
OUT_DIR        <- "results/01_olink_qc"
PCA_OUTLIER_SD <- 3               # outlierDefX / outlierDefY for olink_pca_plot
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# read_NPX handles Olink's native export formats; a plain CSV also works.
npx <- tryCatch(read_NPX(NPX_FILE), error = function(e) read_csv(NPX_FILE, show_col_types = FALSE))

# Remove control assays (AssayType ending in "_ctrl"), mirroring the Python pipeline.
if ("AssayType" %in% names(npx)) {
  npx <- npx %>% filter(!grepl("_ctrl$", AssayType))
}

if (!is.na(HEMOLYSIS_FILE)) {
  hemolysis <- read_csv(HEMOLYSIS_FILE, show_col_types = FALSE)
  npx <- npx %>% left_join(hemolysis, by = "SampleID")
}
if (!"Hemolysis" %in% names(npx)) {
  stop("No 'Hemolysis' column found. Provide HEMOLYSIS_FILE or add the column to the NPX export.")
}
npx <- npx %>% mutate(Hemolysis = factor(Hemolysis))

# Supplementary Fig. 1A: NPX distributions by sample QC status
p1 <- olink_dist_plot(npx, color_g = "SampleQC")
ggsave(file.path(OUT_DIR, "supp_fig1a_npx_distribution_by_sampleqc.png"), p1, width = 14, height = 6, dpi = 300)

# Supplementary Fig. 1B: NPX distributions by hemolysis index
p2 <- olink_dist_plot(npx, color_g = "Hemolysis")
ggsave(file.path(OUT_DIR, "supp_fig1b_npx_distribution_by_hemolysis.png"), p2, width = 14, height = 6, dpi = 300)

# Supplementary Fig. 2: PCA of the complete NPX matrix colored by QC status
p3 <- olink_pca_plot(
  npx,
  color_g = "SampleQC",
  outlierDefX = PCA_OUTLIER_SD,
  outlierDefY = PCA_OUTLIER_SD,
  outlierLines = TRUE,
  label_outliers = TRUE
)
ggsave(file.path(OUT_DIR, "supp_fig2_pca_by_sampleqc.png"), p3[[1]], width = 9, height = 8, dpi = 300)

# Sample-level QC summary
qc_summary <- npx %>%
  group_by(SampleID) %>%
  summarise(
    n_assays = n(),
    n_qc_warning = sum(SampleQC != "PASS" & SampleQC != "Pass", na.rm = TRUE),
    median_npx = median(NPX, na.rm = TRUE),
    hemolysis = first(Hemolysis),
    .groups = "drop"
  )
write_csv(qc_summary, file.path(OUT_DIR, "sample_qc_summary.csv"))
cat("QC figures written to", OUT_DIR, "\n")

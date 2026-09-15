#!/usr/bin/env bash
# Run the full pipeline in order. Assumes raw exports are in data/raw/ and the
# REDCap column labels in src/ce_proteomics/config.py match the local exports.
# Step 03 is the slow one (bootstrapped elastic net); see its --n-bootstrap flag.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/01_build_analysis_dataset.py "$@"
python scripts/02_univariate_screen.py
python scripts/03_elastic_net_panel.py
python scripts/04_classification.py
python scripts/05_mechanistic_analysis.py
python scripts/06_glucose_alcohol_diagnostics.py

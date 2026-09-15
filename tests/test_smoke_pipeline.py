"""
End-to-end smoke test: generate synthetic data and run every pipeline script
with small resampling counts. Verifies that the code runs and produces the
expected files; it makes no claim about results.

Run with
    pytest tests/                      (or)
    python tests/test_smoke_pipeline.py
Takes a few minutes (the RFECV feature set dominates).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO / "src"))

from ce_proteomics import config  # noqa: E402


def _run(*args: str) -> None:
    result = subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"Command failed: {' '.join(args)}\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")


def test_pipeline_runs_end_to_end(tmp_path: Path | None = None) -> None:
    tmp_path = tmp_path or Path(tempfile.mkdtemp())
    raw, proc, res = tmp_path / "raw", tmp_path / "processed", tmp_path / "results"

    _run(str(REPO / "tests" / "make_synthetic_data.py"), "--out-dir", str(raw), "--n-assays", "200")
    _run(str(SCRIPTS / "01_build_analysis_dataset.py"),
         "--olink", str(raw / config.OLINK_LONG_FILE.name),
         "--clinical", str(raw / config.REDCAP_CLINICAL_FILE.name),
         "--ct", str(raw / config.REDCAP_CT_FILE.name),
         "--out-dir", str(proc))
    dataset = proc / config.ANALYSIS_DATASET_FILE.name
    assays = proc / config.OLINK_ASSAY_LIST_FILE.name
    assert dataset.exists() and assays.exists()

    _run(str(SCRIPTS / "02_univariate_screen.py"), "--dataset", str(dataset), "--assays", str(assays),
         "--results-dir", str(res))
    panel_12 = res / "panels" / config.PANEL_12_FILE.name
    assert panel_12.exists() and (res / "02_univariate" / "fig2_panel_boxplots.png").exists()

    _run(str(SCRIPTS / "03_elastic_net_panel.py"), "--dataset", str(dataset), "--assays", str(assays),
         "--results-dir", str(res), "--n-bootstrap", "4", "--n-jobs", "2")
    panel_60 = res / "panels" / config.PANEL_60_FILE.name
    assert panel_60.exists()

    _run(str(SCRIPTS / "04_classification.py"), "--dataset", str(dataset), "--panel-12", str(panel_12),
         "--panel-60", str(panel_60), "--results-dir", str(res), "--n-bootstrap", "25", "--rfe-trees", "20")
    assert (res / "04_classification" / "delong_tests.csv").exists()
    assert (res / "04_classification" / "12-protein_plus_clinical_rfecv" / "rfecv_selection_frequency.csv").exists()

    _run(str(SCRIPTS / "05_mechanistic_analysis.py"), "--dataset", str(dataset), "--panel-60", str(panel_60),
         "--results-dir", str(res), "--pca-bootstrap", "10")
    for name in ("fig4_volcano_worsening.png", "fig5_pca_panel.png", "fig6bc_shap_panel.png",
                 "supp_fig8_shap_interaction_heatmap.png", "shap_proteins_toward_worsening.txt"):
        assert (res / "05_mechanistic" / name).exists(), name

    _run(str(SCRIPTS / "06_glucose_alcohol_diagnostics.py"), "--dataset", str(dataset), "--results-dir", str(res))
    assert (res / "06_diagnostics" / "glucose_by_edema_trajectory.png").exists()


if __name__ == "__main__":
    test_pipeline_runs_end_to_end()
    print("Smoke test passed.")

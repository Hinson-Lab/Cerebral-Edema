# extras/

Exploratory analyses that were run during development but are not used for
any figure or table in the article. They are kept for transparency and reuse
and are not part of the numbered pipeline in `scripts/`.

`exploratory_structure_analysis.py` contains, essentially verbatim from the
original analysis script:

* protein co-expression modules (k-means and hierarchical, several cut
  strategies), module scores, and module-vs-outcome tests
* PCA bootstrap-stability plots and a PCA biplot
* module and PC-axis pathway enrichment through gseapy / Enrichr and STRING
* STRING and correlation-based protein networks, SHAP-direction network plots
* Human Protein Atlas tissue-expression lookups
* spectral clustering of patients with cluster-defining proteins
* panel figures for the clustering analyses and a method-rank comparison

Optional dependencies: `gseapy`, `networkx`, `requests` (STRING and HPA need
network access). Import with the package on the path, for example:

```python
import sys
sys.path.insert(0, "src")
from extras.exploratory_structure_analysis import detect_modules, compute_module_scores
```

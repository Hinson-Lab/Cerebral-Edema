"""
Pathway enrichment inputs and an optional STRING query (Fig. 7).

The manuscript's functional network annotation (Fig. 7) was produced with
the Gladstone Bioinformatics Core Interactive Enrichment Analysis tool
(https://github.com/gladstone-institutes/Interactive-Enrichment-Analysis)
using the SHAP-prioritized protein lists exported here. ``run_string_enrichment``
is an optional, network-dependent convenience that queries the STRING API for
the same lists; it is not required to reproduce any manuscript figure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .io import write_protein_list


def export_shap_direction_lists(lists: dict[str, list[str]], output_dir: Path | str) -> list[Path]:
    """
    Write one text file per SHAP direction (toward_worsening, toward_non_worsening)
    for upload to external enrichment tools. Returns the paths written.
    """
    output_dir = Path(output_dir)
    written = []
    for direction, proteins in lists.items():
        if direction == "unassigned" or not proteins:
            continue
        path = output_dir / f"shap_proteins_{direction}.txt"
        write_protein_list(proteins, path)
        written.append(path)
        print(f"  Exported {len(proteins)} proteins to {path.name}")
    return written


def run_string_enrichment(
    proteins: list[str],
    species: int = config.STRING_SPECIES,
    fdr_threshold: float = config.STRING_FDR,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Query the STRING functional enrichment API for a protein list.

    Returns a DataFrame (category, term, description, fdr, number_of_genes)
    filtered to fdr < ``fdr_threshold``; empty if the request fails or nothing
    is significant. Requires network access and the ``requests`` package.
    """
    try:
        import requests
    except ImportError:
        print("  requests is not installed; skipping STRING enrichment.")
        return pd.DataFrame()

    url = "https://string-db.org/api/json/enrichment"
    params = {"identifiers": "%0d".join(proteins), "species": species,
              "caller_identity": "ce_proteomics"}
    try:
        response = requests.post(url, data=params, timeout=timeout)
    except Exception as exc:  # network errors
        print(f"  STRING request failed: {exc}")
        return pd.DataFrame()
    if response.status_code != 200:
        print(f"  STRING API returned status {response.status_code}")
        return pd.DataFrame()
    data = response.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df = df[df["fdr"] < fdr_threshold].sort_values("fdr")
    return df[["category", "term", "description", "fdr", "number_of_genes"]] if len(df) else pd.DataFrame()


def enrich_shap_directions(lists: dict[str, list[str]], min_proteins: int = 3) -> dict[str, pd.DataFrame]:
    """STRING enrichment for each SHAP direction list with at least ``min_proteins``."""
    results = {}
    for direction, proteins in lists.items():
        if direction == "unassigned":
            continue
        if len(proteins) < min_proteins:
            print(f"  {direction}: only {len(proteins)} proteins; skipping enrichment.")
            continue
        print(f"  {direction}: querying STRING with {len(proteins)} proteins ...")
        results[direction] = run_string_enrichment(proteins)
        print(f"    {len(results[direction])} significant terms")
    return results

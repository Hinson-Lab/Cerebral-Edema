"""
Exploratory structure analyses NOT used for any manuscript figure.

These functions were developed alongside the mechanistic analysis for 
earlier versions of the manuscript. They are retained here for 
transparency and reuse: protein co-expression modules (k-means and
hierarchical), module scores and outcome tests, module and PCA-axis pathway
enrichment via gseapy/Enrichr, STRING and correlation-based protein networks,
Human Protein Atlas tissue lookups, spectral clustering of patients, and the
associated panel figures. 

Optional dependencies: gseapy (Enrichr/GSEA), networkx (network plots),
requests (STRING, Human Protein Atlas). Network access is required for the
STRING and HPA queries.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from ce_proteomics.pca import compute_correlation_loadings, get_axis_proteins
from ce_proteomics.enrichment import run_string_enrichment

try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class AnalysisConfig:
    """Configuration used by the exploratory functions in this module.
        Update ontologies as necessary for new analyses."""
    random_state: int = 42
    p_threshold: float = 0.05
    fdr_method: str = "bh"
    max_modules: int = 10
    module_linkage: str = "average"
    n_pca_components: int = 5
    bootstrap_n: int = 500
    loading_threshold: float = 0.3
    stability_threshold: float = 0.3
    enrichment_databases: list = field(default_factory=lambda: [
        "GO_Biological_Process_2023",
        "GO_Molecular_Function_2023",
        "GO_Cellular_Component_2023",
        "KEGG_2021_Human",
        "Reactome_2022",
    ])
    spectral_k_range: tuple = (2, 7)
    spectral_n_neighbors: int = 10


CONFIG = AnalysisConfig()


# =============================================================================
# CO-EXPRESSION MODULES
# =============================================================================

def detect_modules_kmeans(
    X_z: pd.DataFrame,
    k_range: tuple = (2, 10),
    n_init: int = 20,
    fixed_k: int = None,
) -> tuple[pd.Series, dict]:
    """
    K-means clustering of proteins based on correlation profiles.
    
    This can outperform hierarchical for smaller panels because it
    optimizes globally rather than making greedy local merges.
    
    Parameters
    ----------
    X_z : DataFrame
        Z-scored expression matrix (samples x proteins)
    k_range : tuple
        Range of k values to test (ignored if fixed_k is set)
    fixed_k : int, optional
        If provided, use this k directly (e.g., to match hierarchical)
    
    Returns
    -------
    modules : Series
        Module assignment for each protein (1-indexed)
    metrics : dict
        Silhouette scores for each k tested
    """
    from sklearn.cluster import KMeans
    
    corr = X_z.corr()
    features = corr.values  # Each protein's correlation profile as its feature vector
    
    if fixed_k is not None:
        # Use specified k directly
        km = KMeans(n_clusters=fixed_k, random_state=CONFIG.random_state, n_init=n_init)
        labels = km.fit_predict(features)
        sil = silhouette_score(features, labels)
        
        modules = pd.Series(labels + 1, index=corr.index, name="Module")
        metrics = {fixed_k: {"silhouette": sil, "has_singleton": any(np.bincount(labels) == 1)}}
        
        print(f"K-means modules: k={fixed_k} (fixed, silhouette={sil:.3f})")
        _print_module_sizes(modules)
        return modules, metrics
    
    # Search for optimal k
    metrics = {}
    best_k, best_score, best_labels = None, -1, None
    
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=CONFIG.random_state, n_init=n_init)
        labels = km.fit_predict(features)
        
        # Check for singletons - penalize if present
        unique, counts = np.unique(labels, return_counts=True)
        has_singleton = any(c == 1 for c in counts)
        
        score = silhouette_score(features, labels)
        if has_singleton:
            score *= 0.8  # Penalize singleton modules
        
        metrics[k] = {"silhouette": score, "has_singleton": has_singleton}
        
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    
    modules = pd.Series(best_labels + 1, index=corr.index, name="Module")
    
    print(f"K-means modules: k={best_k} (silhouette={best_score:.3f})")
    _print_module_sizes(modules)
    
    return modules, metrics


def detect_modules_hierarchical(
    X_z: pd.DataFrame,
    method: str = "dynamic",
    max_modules: int = None,
    min_module_size: int = 3,
    linkage_method: str = None,
    corr_threshold: float = None,
) -> tuple[pd.Series, np.ndarray, dict]:
    """
    Hierarchical clustering with multiple cutting strategies.
    
    method : str
        'silhouette' - optimize silhouette (tends to make large modules)
        'dynamic' - variable height cutting, avoids singletons
        'fixed' - use max_modules directly
        'correlation' - cut at specified correlation threshold
    max_modules : int
        Maximum number of modules (for silhouette/fixed methods)
    min_module_size : int
        Minimum proteins per module (for dynamic cutting)
    corr_threshold : float, optional
        Correlation cutoff for 'correlation' method (e.g., 0.5 means 
        proteins must correlate r > 0.5 to be in same module).
        Can also be used with other methods to set initial cut.
    
    Returns
    -------
    modules : Series
        Module assignment for each protein (1-indexed)
    Z : ndarray
        Linkage matrix for dendrogram plotting
    metrics : dict
        Evaluation metrics
    """
    from scipy.cluster.hierarchy import inconsistent
    
    max_modules = max_modules or CONFIG.max_modules
    linkage_method = linkage_method or CONFIG.module_linkage
    
    corr = X_z.corr()
    dist = 1 - corr
    
    # Condensed distance matrix (upper triangle)
    dist_condensed = dist.values[np.triu_indices(len(dist), k=1)]
    Z = linkage(dist_condensed, method=linkage_method)
    
    if method == "silhouette":
        modules, metrics = _hier_silhouette_cut(Z, dist, corr.index, max_modules)
        
    elif method == "dynamic":
        modules, metrics = _hier_dynamic_cut(Z, dist, corr.index, min_module_size, max_modules)
        
    elif method == "fixed":
        labels = fcluster(Z, t=max_modules, criterion="maxclust")
        modules = pd.Series(labels, index=corr.index, name="Module")
        metrics = {"k": max_modules, "method": "fixed"}
    
    elif method == "correlation":
        if corr_threshold is None:
            corr_threshold = 0.5  # Default: proteins must correlate > 0.5
        # Distance threshold = 1 - correlation threshold
        dist_threshold = 1 - corr_threshold
        labels = fcluster(Z, t=dist_threshold, criterion="distance")
        modules = pd.Series(labels, index=corr.index, name="Module")
        
        # Calculate silhouette for reporting
        if modules.nunique() > 1:
            sil = silhouette_score(dist, modules.values, metric="precomputed")
        else:
            sil = np.nan
            
        metrics = {
            "method": "correlation",
            "corr_threshold": corr_threshold,
            "dist_threshold": dist_threshold,
            "k": modules.nunique(),
            "silhouette": sil
        }
        print(f"  Correlation threshold r > {corr_threshold} → {modules.nunique()} modules")
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Hierarchical modules ({method}): k={modules.nunique()}")
    _print_module_sizes(modules)
    
    return modules, Z, metrics


def _hier_silhouette_cut(Z, dist, index, max_modules):
    metrics = {}
    for k in range(2, min(max_modules + 1, len(dist))):
        labels = fcluster(Z, t=k, criterion="maxclust")
        metrics[k] = silhouette_score(dist, labels, metric="precomputed")
    
    best_k = max(metrics, key=metrics.get)
    labels = fcluster(Z, t=best_k, criterion="maxclust")
    modules = pd.Series(labels, index=index, name="Module")
    
    return modules, {"silhouette_scores": metrics, "best_k": best_k}


def _hier_dynamic_cut(Z, dist, index, min_module_size, max_modules):
    """
    Dynamic tree cutting - varies cut height to avoid singletons
    and maintain minimum module sizes.
    
    Strategy: Start with many clusters, then iteratively merge
    small modules into their nearest neighbor until all modules
    meet the minimum size requirement.
    """
    n_proteins = len(index)
    
    # Start with more clusters than we probably need
    initial_k = min(max_modules * 2, n_proteins // 2)
    labels = fcluster(Z, t=initial_k, criterion="maxclust")
    
    # Iteratively merge small modules
    modules = pd.Series(labels, index=index)
    
    for _ in range(50):  # Max iterations
        sizes = modules.value_counts()
        small_modules = sizes[sizes < min_module_size].index.tolist()
        
        if not small_modules:
            break
        
        # Merge smallest module into nearest neighbor
        smallest = sizes.idxmin()
        small_prots = modules[modules == smallest].index
        
        # Find nearest module by mean correlation
        corr_matrix = pd.DataFrame(dist.values, index=index, columns=index)
        other_modules = [m for m in modules.unique() if m != smallest]
        
        best_target, best_dist = None, np.inf
        for target in other_modules:
            target_prots = modules[modules == target].index
            mean_dist = corr_matrix.loc[small_prots, target_prots].mean().mean()
            if mean_dist < best_dist:
                best_target, best_dist = target, mean_dist
        
        if best_target is not None:
            modules[modules == smallest] = best_target
    
    # Renumber modules 1 to k
    unique_modules = sorted(modules.unique())
    remap = {old: new + 1 for new, old in enumerate(unique_modules)}
    modules = modules.map(remap)
    
    # Calculate final silhouette
    final_sil = silhouette_score(dist, modules.values, metric="precomputed")
    
    return modules, {
        "method": "dynamic", 
        "final_k": modules.nunique(),
        "silhouette": final_sil,
        "min_module_size": min_module_size
    }


def _print_module_sizes(modules):
    """Print module size distribution."""
    sizes = modules.value_counts().sort_index()
    size_str = ", ".join([f"M{m}:{n}" for m, n in sizes.items()])
    print(f"  Module sizes: {size_str}")


def detect_modules(
    X_z: pd.DataFrame,
    method: str = "kmeans",
    match_k: bool = False,
    hier_method: str = "dynamic",
    corr_threshold: float = None,
    **kwargs
) -> tuple[pd.Series, np.ndarray | None, dict]:
    """
    Unified interface for module detection.
    
    method : str
        'kmeans' - recommended for smaller panels, matches PCA geometry
        'hierarchical' - use hier_method to determine cutting strategy
        'both' - run both, return comparison
    match_k : bool
        If True and method='both', k-means uses same k as hierarchical.
        If False (default), k-means optimizes k independently via silhouette.
    hier_method : str
        Hierarchical cutting strategy:
        'silhouette' - optimize silhouette score
        'dynamic' - avoid singletons, merge small modules
        'correlation' - cut at correlation threshold (most interpretable)
        'fixed' - use max_modules directly
    corr_threshold : float, optional
        For hier_method='correlation': proteins must correlate above this
        value to be in the same module. E.g., 0.5 means r > 0.5.
        Higher = more/smaller modules. Lower = fewer/larger modules.
        Typical values: 0.3 (lenient), 0.5 (moderate), 0.7 (strict)
    
    Returns
    -------
    modules : Series (or dict if method='both')
    Z : linkage matrix (None for kmeans)
    metrics : dict
    """
    if method == "kmeans":
        modules, metrics = detect_modules_kmeans(X_z, **kwargs)
        return modules, None, metrics
    
    elif method == "hierarchical":
        return detect_modules_hierarchical(
            X_z, method=hier_method, corr_threshold=corr_threshold, **kwargs
        )
    
    elif method == "both":
        print(f"\n--- Hierarchical modules ({hier_method}) ---")
        hier_modules, Z, hier_metrics = detect_modules_hierarchical(
            X_z, method=hier_method, corr_threshold=corr_threshold, **kwargs
        )
        
        hier_k = hier_modules.nunique()
        
        print("\n--- K-means modules ---")
        if match_k:
            # Match hierarchical k
            km_modules, km_metrics = detect_modules_kmeans(X_z, fixed_k=hier_k, **kwargs)
        else:
            # Optimize k independently
            km_modules, km_metrics = detect_modules_kmeans(X_z, **kwargs)
        
        # Compare agreement
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(km_modules.values, hier_modules.values)
        print(f"\nAgreement (Adjusted Rand Index): {ari:.3f}")
        
        return {
            "kmeans": km_modules,
            "hierarchical": hier_modules,
            "ari": ari,
        }, Z, {"kmeans": km_metrics, "hierarchical": hier_metrics}
    
    else:
        raise ValueError(f"Unknown method: {method}")


def get_module_members(modules: pd.Series) -> dict[str, list[str]]:
    """Return dict mapping module names to protein lists."""
    return {
        f"Module_{m}": modules[modules == m].index.tolist()
        for m in sorted(modules.unique())
    }


def compare_modules_to_pca(
    modules: pd.Series,
    loadings: pd.DataFrame,
    pcs: list[str] = ["PC1", "PC2", "PC3"]
) -> pd.DataFrame:
    """
    Quantify how well modules align with PCA axes.
    
    Returns DataFrame for interpretation.
    """
    results = []
    
    for m in sorted(modules.unique()):
        members = modules[modules == m].index.tolist()
        module_loadings = loadings.loc[members]
        
        row = {"Module": f"Module_{m}", "N_proteins": len(members)}
        
        for pc in pcs:
            if pc in module_loadings.columns:
                pc_loads = module_loadings[pc]
                row[f"{pc}_mean"] = pc_loads.mean()
                row[f"{pc}_abs_mean"] = pc_loads.abs().mean()
                # Coherence: are loadings all same sign?
                row[f"{pc}_coherence"] = abs(pc_loads.mean()) / pc_loads.abs().mean() if pc_loads.abs().mean() > 0 else 0
        
        # Which PC does this module load on most?
        abs_means = {pc: row.get(f"{pc}_abs_mean", 0) for pc in pcs}
        row["Primary_PC"] = max(abs_means, key=abs_means.get)
        row["Primary_PC_loading"] = row[f"{row['Primary_PC']}_mean"]
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_module_scores(
    X_z: pd.DataFrame, 
    modules: pd.Series
) -> pd.DataFrame:
    """
    Compute module eigengene-like scores (mean z-score of member proteins).
    """
    scores = {}
    for m in sorted(modules.unique()):
        members = modules[modules == m].index
        scores[f"Module_{m}"] = X_z[members].mean(axis=1)
    return pd.DataFrame(scores, index=X_z.index)


def test_modules_vs_outcome(
    module_scores: pd.DataFrame, 
    y: pd.Series
) -> pd.DataFrame:
    """T-tests comparing module scores between outcome groups."""
    results = []
    for col in module_scores.columns:
        g0 = module_scores.loc[y == 0, col].dropna()
        g1 = module_scores.loc[y == 1, col].dropna()
        t, p = ttest_ind(g1, g0, equal_var=False)
        d = (g1.mean() - g0.mean()) / np.sqrt((g0.var() + g1.var()) / 2)
        results.append({"Module": col, "t_stat": t, "p_value": p, "Cohens_d": d})
    return pd.DataFrame(results).sort_values("p_value")


def plot_module_boxplots(
    module_scores: pd.DataFrame,
    y: pd.Series,
    test_df: pd.DataFrame = None,
    title: str = "Module-level Expression vs Worsening",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Boxplots of module/PC scores by outcome with p-value annotations.
    
    Parameters
    ----------
    module_scores : DataFrame
        Columns are modules/PCs, rows are samples
    y : Series
        Binary outcome (0/1), aligned to module_scores index
    test_df : DataFrame, optional
        Pre-computed test results with 'Module' and 'p_value' columns.
        If None, will compute t-tests.
    title : str
        Plot title
    """
    # Prepare data
    plot_df = module_scores.copy()
    plot_df["Edema_Worse"] = y.loc[plot_df.index].values
    
    melted = plot_df.melt(
        id_vars="Edema_Worse", var_name="Module", value_name="Score"
    )
    
    # Compute tests if not provided
    if test_df is None:
        test_df = test_modules_vs_outcome(module_scores, y)
    
    # Ensure test_df has consistent column name
    if "p" in test_df.columns and "p_value" not in test_df.columns:
        test_df = test_df.rename(columns={"p": "p_value"})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get module order from the data
    module_order = sorted(melted["Module"].unique())
    
    sns.boxplot(
        data=melted, 
        x="Module", 
        y="Score", 
        hue="Edema_Worse",
        order=module_order,
        ax=ax
    )
    
    ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
    
    # Add p-value annotations
    ymax = melted["Score"].max()
    ymin = melted["Score"].min()
    y_range = ymax - ymin
    y_offset = y_range * 0.05
    
    for i, mod in enumerate(module_order):
        p_row = test_df[test_df["Module"] == mod]
        if len(p_row) > 0:
            p_val = p_row["p_value"].values[0]
            # Format p-value
            if p_val < 0.001:
                p_text = "p<0.001"
            else:
                p_text = f"p={p_val:.3f}"
            
            ax.text(
                i,
                ymax + y_offset,
                p_text,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if p_val < 0.05 else "normal"
            )
    
    # Expand y-axis to make room for p-values and title
    ax.set_ylim(ymin - y_range * 0.05, ymax + y_range * 0.20)
    
    # Set title with padding to avoid overlap with p-values
    ax.set_title(title, pad=20, fontsize=11)
    
    # Adjust legend
    ax.legend(title="Edema Worse", loc="upper right")
    
    fig.tight_layout()
    return fig


def plot_pc_boxplots(
    scores_df: pd.DataFrame,
    y: pd.Series,
    pcs: list = ["PC1", "PC2"],
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Boxplots of PC scores by outcome with p-value annotations.
    
    Parameters
    ----------
    scores_df : DataFrame
        PCA scores with PC columns
    y : Series
        Binary outcome
    pcs : list
        Which PCs to plot
    """
    # Compute tests
    test_results = []
    for pc in pcs:
        g0 = scores_df.loc[y == 0, pc].dropna()
        g1 = scores_df.loc[y == 1, pc].dropna()
        t, p = ttest_ind(g1, g0, equal_var=False)
        test_results.append({"Module": pc, "p_value": p, "t_stat": t})
    test_df = pd.DataFrame(test_results)
    
    # Use the module boxplot function
    pc_scores = scores_df[pcs].copy()
    
    return plot_module_boxplots(
        pc_scores, 
        y, 
        test_df=test_df,
        title="PC Scores by Edema Worsening",
        figsize=figsize
    )


def plot_module_heatmap(
    X_z: pd.DataFrame, 
    modules: pd.Series,
    figsize: tuple = (12, 10),
    title: str = None,
) -> plt.Figure:
    """
    Correlation heatmap ordered by module membership.
    
    Proteins are ordered by:
    1. Module assignment
    2. Within each module: mean correlation with other module members (centrality)
    """
    corr = X_z.corr()
    
    # Order proteins: by module, then by centrality within module
    ordered_proteins = []
    for m in sorted(modules.unique()):
        module_prots = modules[modules == m].index.tolist()
        if len(module_prots) > 1:
            # Order by mean correlation with other module members
            module_corr = corr.loc[module_prots, module_prots]
            # Mean correlation with others (excluding self)
            centrality = (module_corr.sum(axis=1) - 1) / (len(module_prots) - 1)
            module_prots = centrality.sort_values(ascending=False).index.tolist()
        ordered_proteins.extend(module_prots)
    
    corr_ord = corr.loc[ordered_proteins, ordered_proteins]
    
    # Module color sidebar
    palette = sns.color_palette("husl", n_colors=modules.nunique())
    color_map = {m: palette[i] for i, m in enumerate(sorted(modules.unique()))}
    row_colors = modules[ordered_proteins].map(color_map)
    
    g = sns.clustermap(
        corr_ord,
        row_cluster=False, col_cluster=False,
        row_colors=row_colors, col_colors=row_colors,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        figsize=figsize,
        dendrogram_ratio=0.1,
        xticklabels=True,
        yticklabels=True,
    )
    
    # Rotate x labels for readability
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
    
    if title is None:
        title = "Protein Co-expression (Ordered by Module)"
    g.fig.suptitle(title, y=1.02)
    return g.fig


def plot_dendrogram(
    Z: np.ndarray, 
    modules: pd.Series,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """Dendrogram with leaves colored by module assignment."""
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = modules.index.tolist()
    palette = sns.color_palette("husl", n_colors=modules.nunique())
    color_map = {m: palette[i] for i, m in enumerate(sorted(modules.unique()))}
    
    dn = dendrogram(Z, labels=labels, leaf_rotation=90, ax=ax, leaf_font_size=8)
    
    # Color labels by module
    for lbl in ax.get_xticklabels():
        prot = lbl.get_text()
        if prot in modules.index:
            lbl.set_color(color_map[modules[prot]])
    
    ax.set_ylabel("Distance (1 - correlation)")
    ax.set_title("Hierarchical Clustering of Proteins")
    fig.tight_layout()
    return fig


# ===========================================
# PCA STABILITY PLOTS AND BIPLOT


def plot_pca_stability(
    stability_df: pd.DataFrame,
    pc: str = "PC1",
    top_n: int = 20,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Visualize bootstrap stability of PCA loadings.
    
    Shows mean loading ± SD for top proteins, with markers indicating
    whether they meet the axis-defining threshold.

    """
    mean_col = f"{pc}_mean"
    sd_col = f"{pc}_sd"
    axis_col = f"{pc}_axis_defining"
    
    if mean_col not in stability_df.columns:
        raise ValueError(f"{pc} not found in stability results")
    
    # Sort by absolute mean loading
    df = stability_df.copy()
    df["abs_mean"] = df[mean_col].abs()
    df = df.sort_values("abs_mean", ascending=False).head(top_n)
    
    # Reverse for horizontal bar plot (top at top)
    df = df.iloc[::-1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(df))
    means = df[mean_col].values
    sds = df[sd_col].values
    is_axis = df[axis_col].values
    
    # Color by axis-defining status
    colors = ["crimson" if a else "gray" for a in is_axis]
    
    # Plot bars with error bars
    ax.barh(y_pos, means, xerr=sds, color=colors, alpha=0.7, 
            error_kw={"ecolor": "black", "capsize": 3, "capthick": 1})
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index)
    ax.axvline(0, color="black", linewidth=0.5)
    
    # Add threshold lines
    thresh = CONFIG.loading_threshold
    ax.axvline(thresh, color="red", linestyle="--", alpha=0.5, label=f"Threshold (±{thresh})")
    ax.axvline(-thresh, color="red", linestyle="--", alpha=0.5)
    
    ax.set_xlabel(f"{pc} Loading (mean ± SD across {CONFIG.bootstrap_n} bootstraps)")
    ax.set_title(f"{pc} Loading Stability\n(Red = axis-defining proteins)")
    ax.legend(loc="lower right")
    
    fig.tight_layout()
    return fig


def plot_pca_stability_heatmap(
    stability_df: pd.DataFrame,
    pcs: list = ["PC1", "PC2", "PC3"],
    top_n: int = 25,
    figsize: tuple = (8, 10),
) -> plt.Figure:
    """
    Heatmap showing loading stability across multiple PCs.
    
    Rows = proteins (sorted by max absolute loading across PCs)
    Columns = PCs
    Color = mean bootstrap loading
    Annotation = proportion of bootstraps exceeding threshold
    """
    # Get mean loadings for each PC
    mean_cols = [f"{pc}_mean" for pc in pcs if f"{pc}_mean" in stability_df.columns]
    prop_cols = [f"{pc}_prop_stable" for pc in pcs if f"{pc}_prop_stable" in stability_df.columns]
    
    means_df = stability_df[mean_cols].copy()
    means_df.columns = [c.replace("_mean", "") for c in mean_cols]
    
    props_df = stability_df[prop_cols].copy()
    props_df.columns = [c.replace("_prop_stable", "") for c in prop_cols]
    
    # Sort by max absolute loading across PCs
    max_abs = means_df.abs().max(axis=1)
    top_proteins = max_abs.sort_values(ascending=False).head(top_n).index
    
    plot_means = means_df.loc[top_proteins]
    plot_props = props_df.loc[top_proteins]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Heatmap of mean loadings
    vmax = plot_means.abs().max().max()
    sns.heatmap(
        plot_means,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=plot_props,  # Annotate with stability proportion
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Mean Loading"}
    )
    
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Protein")
    ax.set_title(f"PCA Loading Stability\n(annotations = proportion of bootstraps |loading| ≥ {CONFIG.loading_threshold})")
    
    fig.tight_layout()
    return fig




def plot_pca_biplot(
    scores_df: pd.DataFrame,
    loadings: pd.DataFrame,
    y: pd.Series,
    pca: PCA,
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    top_n_loadings: int = 10,
    arrow_scale: float = 3.0,
    figsize: tuple = (10, 9),
) -> plt.Figure:
    """
    PCA biplot showing both sample scores and protein loadings.
    
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot samples colored by outcome
    y_aligned = y.loc[scores_df.index]
    colors = y_aligned.map({0: "steelblue", 1: "darkorange"})
    labels = y_aligned.map({0: "No Worsening", 1: "Worsening"})
    
    for outcome, color, label in [(0, "steelblue", "No Worsening"), (1, "darkorange", "Worsening")]:
        mask = y_aligned == outcome
        ax.scatter(scores_df.loc[mask, pc_x], scores_df.loc[mask, pc_y],
                  c=color, label=label, alpha=0.6, s=50, edgecolor="white", linewidth=0.5)
    
    # Get top loading proteins for arrows
    pc_x_idx = int(pc_x.replace("PC", "")) - 1
    pc_y_idx = int(pc_y.replace("PC", "")) - 1
    
    loading_magnitude = np.sqrt(loadings[pc_x]**2 + loadings[pc_y]**2)
    top_proteins = loading_magnitude.nlargest(top_n_loadings).index
    
    # Plot loading arrows
    for prot in top_proteins:
        x_load = loadings.loc[prot, pc_x] * arrow_scale
        y_load = loadings.loc[prot, pc_y] * arrow_scale
        
        ax.annotate("", xy=(x_load, y_load), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5, alpha=0.7))
        
        # Label at arrow tip
        ax.text(x_load * 1.05, y_load * 1.05, prot, fontsize=8, ha="center", va="center",
               color="darkred", fontweight="bold")
    
    # Axis labels with variance explained
    var_x = pca.explained_variance_ratio_[pc_x_idx] * 100
    var_y = pca.explained_variance_ratio_[pc_y_idx] * 100
    ax.set_xlabel(f"{pc_x} ({var_x:.1f}%)")
    ax.set_ylabel(f"{pc_y} ({var_y:.1f}%)")
    
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    
    ax.legend(loc="best")
    ax.set_title("PCA Biplot: Samples + Top Loading Proteins")
    
    fig.tight_layout()
    return fig



# PATHWAY ENRICHMENT (gseapy / Enrichr, STRING) AND TBI CATEGORIES


def run_enrichment(
    gene_list: list[str],
    databases: list[str] = None,
    organism: str = "human",
    background: list[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Run over-representation analysis (ORA) using Enrichr databases.
    
    Parameters
    ----------
    gene_list : list
        Proteins/genes to test for enrichment
    databases : list
        Enrichr database names (default: GO + KEGG + Reactome)
    background : list, optional
        Background gene set (default: all genes in database)
    
    Returns
    -------
    DataFrame with enrichment results, or None if gseapy unavailable
    """
    if not GSEAPY_AVAILABLE:
        print("gseapy not available. Install with: pip install gseapy")
        return None
    
    databases = databases or CONFIG.enrichment_databases
    
    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=databases,
            organism=organism,
            background=background,
            outdir=None,  # Don't save files
            no_plot=True,
        )
        results = enr.results
        results = results[results["Adjusted P-value"] < 0.1].sort_values("Adjusted P-value")
        return results
    except Exception as e:
        print(f"Enrichment analysis failed: {e}")
        return None


def run_prerank_gsea(
    ranked_genes: pd.Series,
    databases: list[str] = None,
    permutations: int = 1000,
) -> Optional[pd.DataFrame]:
    """
    Run preranked GSEA using PC loadings as ranking metric.
    
    Parameters
    ----------
    ranked_genes : Series
        Index = gene names, values = ranking metric (e.g., PC loading)
    """
    if not GSEAPY_AVAILABLE:
        print("gseapy not available.")
        return None
    
    databases = databases or CONFIG.enrichment_databases
    
    try:
        pre_res = gp.prerank(
            rnk=ranked_genes.sort_values(ascending=False),
            gene_sets=databases,
            permutation_num=permutations,
            outdir=None,
            no_plot=True,
            seed=CONFIG.random_state,
        )
        results = pre_res.res2d
        results = results[results["FDR q-val"] < 0.25].sort_values("NES", ascending=False)
        return results
    except Exception as e:
        print(f"Preranked GSEA failed: {e}")
        return None


def enrich_pc_axes(
    stability_df: pd.DataFrame,
    loadings: pd.DataFrame,
    pcs: list[str] = ["PC1", "PC2"],
) -> dict[str, pd.DataFrame]:
    """
    Run enrichment for each PC's axis-defining proteins.
    
    Returns dict mapping PC name to enrichment results DataFrame.
    """
    results = {}
    
    for pc in pcs:
        axis_prots = get_axis_proteins(stability_df, pc)
        if len(axis_prots) < 3:
            print(f"{pc}: Only {len(axis_prots)} axis proteins, skipping enrichment")
            continue
            
        print(f"\n{pc}: Enriching {len(axis_prots)} axis-defining proteins...")
        enr = run_enrichment(axis_prots)
        if enr is not None and len(enr) > 0:
            results[pc] = enr
            print(f"  Top terms: {enr['Term'].head(5).tolist()}")
        else:
            print(f"  No significant enrichment found")
    
    return results


def enrich_modules(
    modules: pd.Series,
) -> dict[str, pd.DataFrame]:
    """
    Run enrichment for each co-expression module.
    """
    module_members = get_module_members(modules)
    results = {}
    
    for mod_name, proteins in module_members.items():
        if len(proteins) < 3:
            continue
        print(f"\n{mod_name}: Enriching {len(proteins)} proteins...")
        enr = run_enrichment(proteins)
        if enr is not None and len(enr) > 0:
            results[mod_name] = enr
            print(f"  Top terms: {enr['Term'].head(3).tolist()}")
    
    return results


def summarize_enrichment(
    enrichment_results: dict[str, pd.DataFrame],
    top_n: int = 5
) -> pd.DataFrame:
    """Create summary table of top enrichment terms per group."""
    rows = []
    for group, df in enrichment_results.items():
        if df is None or len(df) == 0:
            continue
        for _, row in df.head(top_n).iterrows():
            rows.append({
                "Group": group,
                "Term": row["Term"],
                "P_adj": row.get("Adjusted P-value", row.get("FDR q-val", np.nan)),
                "Genes": row.get("Genes", row.get("Lead_genes", "")),
            })
    return pd.DataFrame(rows)






# Curated TBI-relevant protein categories for manual annotation
# Change manual annotation as needed or informed by relevant literature.
TBI_PROTEIN_CATEGORIES = {
    "synaptic_vesicle": ["SNAP25", "CPLX3", "RPH3A", "CHGB", "SYT1", "VAMP2", "STX1A"],
    "postsynaptic": ["HOMER1", "SHANK1", "SHANK2", "SHANK3", "DLG4", "GRIN1"],
    "neuronal_injury": ["NEFL", "NEFM", "NEFH", "CEND1", "UCHL1", "ENO2"],
    "astrocyte": ["GFAP", "S100B", "AQP4", "ALDH1L1", "SLC1A2", "SLC1A3"],
    "calcium_binding": ["HPCAL4", "NECAB1", "NECAB2", "CALB1", "CALB2", "PVALB"],
    "rna_binding_neuronal": ["ELAVL4", "ELAVL2", "ELAVL3", "RBFOX1", "RBFOX3"],
    "immune_neutrophil": ["CD177", "MPO", "ELANE", "PRTN3"],
    "metabolic": ["ACAD9", "MRRF", "ATP5F1A", "TXNRD3NB"],
    "cytoskeletal": ["ARVCF", "RAC3"],
    "transcription_regulation": ["ZNF75A", "YY1AP1", "PHC2"],
    "cell_cycle": ["NEK2", "DZANK1"],
}


def categorize_proteins_tbi(protein_list: list) -> pd.DataFrame:
    """
    Assign proteins to curated TBI-relevant categories.
    
    Useful when formal enrichment tools return nothing for small lists.
    """
    results = []
    for prot in protein_list:
        category = "other/unknown"
        for cat, members in TBI_PROTEIN_CATEGORIES.items():
            if prot.upper() in [m.upper() for m in members]:
                category = cat
                break
        results.append({"Protein": prot, "TBI_Category": category})
    
    return pd.DataFrame(results)


def enrich_shap_directions(
    shap_summary_full: pd.DataFrame,
    method: str = "string",
) -> dict:
    """
    Run pathway enrichment on SHAP-derived protein lists.
    
    More meaningful than PCA enrichment because proteins are grouped
    by their discriminative direction (predicts worsening vs not).
    
    Parameters
    ----------
    shap_summary_full : DataFrame
        Full SHAP summary with all proteins
    method : str
        'string' - STRING database (recommended)
        'enrichr' - Enrichr via gseapy
        'manual' - curated TBI categories
    
    Returns
    -------
    dict with enrichment results for each direction
    """
    # Split by direction
    toward_worsening = shap_summary_full[
        shap_summary_full["Direction_corr"] > 0.3
    ]["Protein"].tolist()
    
    toward_non_worsening = shap_summary_full[
        shap_summary_full["Direction_corr"] < -0.3
    ]["Protein"].tolist()
    
    enrichment_results = {}
    
    for direction, proteins in [("toward_worsening", toward_worsening), 
                                 ("toward_non_worsening", toward_non_worsening)]:
        if len(proteins) < 3:
            print(f"  {direction}: Too few proteins ({len(proteins)}), skipping enrichment")
            continue
        
        print(f"  {direction}: Enriching {len(proteins)} proteins via {method}...")
        
        if method == "string":
            result = run_string_enrichment(proteins)
        elif method == "enrichr" and GSEAPY_AVAILABLE:
            result = run_enrichment(proteins)
        elif method == "manual":
            result = categorize_proteins_tbi(proteins)
        else:
            result = pd.DataFrame()
        
        if len(result) > 0:
            enrichment_results[direction] = result
            if method != "manual":
                print(f"    Found {len(result)} significant terms")
        else:
            print(f"    No significant enrichment found")
            # Fall back to manual categorization
            enrichment_results[direction] = categorize_proteins_tbi(proteins)
            print(f"    Using manual TBI categories instead")
    
    return enrichment_results


def plot_shap_enrichment_comparison(
    enrichment_results: dict,
    top_n: int = 10,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Compare enrichment results between worsening and non-worsening directions.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    directions = ["toward_worsening", "toward_non_worsening"]
    titles = ["Higher → MORE Worsening", "Higher → LESS Worsening"]
    colors = ["darkorange", "steelblue"]
    
    for ax, direction, title, color in zip(axes, directions, titles, colors):
        if direction not in enrichment_results or len(enrichment_results[direction]) == 0:
            ax.text(0.5, 0.5, "No significant\nenrichment", ha="center", va="center", fontsize=12)
            ax.set_title(title)
            ax.axis("off")
            continue
        
        df = enrichment_results[direction].head(top_n)
        
        # Handle different column names (STRING vs Enrichr vs manual)
        if "description" in df.columns:
            terms = df["description"].tolist()
            pvals = df["fdr"].tolist()
        elif "Term" in df.columns:
            terms = df["Term"].tolist()
            pvals = df["Adjusted P-value"].tolist()
        elif "TBI_Category" in df.columns:
            # Manual categorization - show category counts
            category_counts = df["TBI_Category"].value_counts()
            terms = category_counts.index.tolist()
            pvals = [0.01] * len(terms)  # Placeholder for manual
        else:
            ax.text(0.5, 0.5, "Unknown format", ha="center", va="center")
            continue
        
        # Truncate long term names
        terms = [t[:50] + "..." if len(t) > 50 else t for t in terms]
        
        y_pos = np.arange(len(terms))
        ax.barh(y_pos, -np.log10(np.array(pvals) + 1e-10), color=color, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=9)
        ax.set_xlabel("-log10(FDR)")
        ax.set_title(f"Proteins: {title}")
        ax.invert_yaxis()
    
    fig.suptitle("Pathway Enrichment by SHAP Direction", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# =============================================================================
# PROTEIN NETWORKS AND HUMAN PROTEIN ATLAS


def get_string_network(
    proteins: list,
    species: int = 9606,
    score_threshold: float = 400,
) -> Optional[pd.DataFrame]:
    """
    Query STRING database for protein-protein interaction network.
    
    """
    if not REQUESTS_AVAILABLE:
        print("requests library not available")
        return None
    
    string_api_url = "https://string-db.org/api/json/network"
    
    params = {
        "identifiers": "%0d".join(proteins),
        "species": species,
        "required_score": int(score_threshold),
        "caller_identity": "proteomics_analysis"
    }
    
    try:
        response = requests.post(string_api_url, data=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                if len(df) > 0 and "preferredName_A" in df.columns:
                    return df[["preferredName_A", "preferredName_B", "score"]].rename(
                        columns={"preferredName_A": "Protein_1", 
                                "preferredName_B": "Protein_2",
                                "score": "Score"}
                    )
        else:
            print(f"  STRING API returned status {response.status_code}")
            
    except Exception as e:
        print(f"  STRING network request failed: {e}")
    
    return pd.DataFrame()


def build_correlation_network(
    X_z: pd.DataFrame,
    proteins: list,
    corr_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Build a correlation-based network for a subset of proteins.

    """
    # Subset to proteins in the data
    valid_prots = [p for p in proteins if p in X_z.columns]
    if len(valid_prots) < 2:
        return pd.DataFrame(columns=["Protein_1", "Protein_2", "Correlation"])
    
    corr_mat = X_z[valid_prots].corr()
    
    # Extract edges above threshold
    edges = []
    for i, p1 in enumerate(valid_prots):
        for j, p2 in enumerate(valid_prots):
            if i < j:  # Upper triangle only
                r = corr_mat.loc[p1, p2]
                if abs(r) >= corr_threshold:
                    edges.append({"Protein_1": p1, "Protein_2": p2, "Correlation": r})
    
    return pd.DataFrame(edges)


def plot_protein_network(
    edges: pd.DataFrame,
    node_info: pd.DataFrame = None,
    title: str = "Protein Interaction Network",
    node_color_col: str = None,
    node_size_col: str = None,
    layout: str = "spring",
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """
    Plot protein interaction network.
    
    Parameters
    ----------
    edges : DataFrame
        Edges with columns Protein_1, Protein_2, and optionally weight/score column
    node_info : DataFrame, optional
        Node attributes indexed by protein name. Can include SHAP values,
        direction, importance, etc.
    node_color_col : str, optional
        Column in node_info to use for node coloring
    node_size_col : str, optional
        Column in node_info to use for node sizing
    layout : str
        'spring' (default), 'circular', 'kamada_kawai', 'spectral'
    
    Returns
    -------
    Figure
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx not installed. Install with: pip install networkx")
        return None
    
    if len(edges) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No interactions found\nabove threshold", 
               ha="center", va="center", fontsize=14)
        ax.set_title(title)
        ax.axis("off")
        return fig
    
    # Build graph
    G = nx.Graph()
    
    # Add edges
    weight_col = [c for c in edges.columns if c not in ["Protein_1", "Protein_2"]]
    weight_col = weight_col[0] if weight_col else None
    
    for _, row in edges.iterrows():
        weight = row[weight_col] if weight_col else 1.0
        G.add_edge(row["Protein_1"], row["Protein_2"], weight=abs(weight))
    
    # Add isolated nodes if node_info provided
    if node_info is not None:
        for node in node_info.index:
            if node not in G.nodes():
                G.add_node(node)
    
    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(len(G.nodes())))
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Node colors
    if node_info is not None and node_color_col and node_color_col in node_info.columns:
        # Map node colors from node_info
        color_vals = [node_info.loc[n, node_color_col] if n in node_info.index else 0 
                     for n in G.nodes()]
        vmin, vmax = min(color_vals), max(color_vals)
        cmap = plt.cm.RdBu_r
        node_colors = [cmap((v - vmin) / (vmax - vmin + 1e-10)) for v in color_vals]
    else:
        node_colors = "steelblue"
    
    # Node sizes
    if node_info is not None and node_size_col and node_size_col in node_info.columns:
        sizes = [300 + 500 * node_info.loc[n, node_size_col] if n in node_info.index else 300 
                for n in G.nodes()]
    else:
        sizes = 500
    
    # Draw edges
    edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + 3 * (w / max_weight) for w in edge_weights]
    
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, width=edge_widths, 
                          edge_color="gray")
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=sizes, alpha=0.9)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
    
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    
    # Add colorbar if using color mapping
    if node_info is not None and node_color_col:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(node_color_col)
    
    fig.tight_layout()
    return fig


def plot_shap_direction_networks(
    shap_summary: pd.DataFrame,
    X_z: pd.DataFrame,
    network_method: str = "correlation",
    corr_threshold: float = 0.4,
    string_threshold: float = 400,
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """
    Create side-by-side network plots for proteins pushing toward worsening
    vs proteins pushing toward non-worsening.
    
    Parameters
    ----------
    shap_summary : DataFrame
        SHAP summary with Protein, Mean_|SHAP|, Direction_corr columns
    X_z : DataFrame
        Z-scored expression matrix
    network_method : str
        'correlation' - use expression correlation
        'string' - use STRING protein interactions
    corr_threshold : float
        For correlation method: minimum |r| for edge
    string_threshold : float
        For STRING method: minimum score
    
    Returns
    -------
    Figure with two network subplots
    """
    # Split proteins by direction
    toward_worsening = shap_summary[
        shap_summary["Direction_corr"] > 0.3
    ].set_index("Protein")
    
    toward_non_worsening = shap_summary[
        shap_summary["Direction_corr"] < -0.3
    ].set_index("Protein")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for ax, df, title, color in [
        (axes[0], toward_worsening, "Proteins → WORSENING\n(higher = worse outcome)", "Reds"),
        (axes[1], toward_non_worsening, "Proteins → NON-WORSENING\n(higher = better outcome)", "Blues"),
    ]:
        if len(df) < 2:
            ax.text(0.5, 0.5, f"Only {len(df)} proteins\nin this direction", 
                   ha="center", va="center", fontsize=12)
            ax.set_title(title)
            ax.axis("off")
            continue
        
        proteins = df.index.tolist()
        
        # Get network edges
        if network_method == "correlation":
            edges = build_correlation_network(X_z, proteins, corr_threshold)
        elif network_method == "string":
            edges = get_string_network(proteins, score_threshold=string_threshold)
            if edges is None or len(edges) == 0:
                # Fallback to correlation
                print(f"  STRING returned no edges for {title[:20]}..., using correlation")
                edges = build_correlation_network(X_z, proteins, corr_threshold)
        else:
            edges = pd.DataFrame()
        
        # Build graph
        try:
            import networkx as nx
        except ImportError:
            ax.text(0.5, 0.5, "networkx not installed", ha="center", va="center")
            continue
        
        G = nx.Graph()
        for prot in proteins:
            G.add_node(prot)
        
        if len(edges) > 0:
            weight_col = [c for c in edges.columns if c not in ["Protein_1", "Protein_2"]]
            weight_col = weight_col[0] if weight_col else None
            
            for _, row in edges.iterrows():
                if row["Protein_1"] in proteins and row["Protein_2"] in proteins:
                    weight = abs(row[weight_col]) if weight_col else 1.0
                    G.add_edge(row["Protein_1"], row["Protein_2"], weight=weight)
        
        # Layout
        if len(G.edges()) > 0:
            pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(len(G.nodes())))
        else:
            pos = nx.circular_layout(G)
        
        # Node sizes based on SHAP importance
        sizes = [200 + 1500 * df.loc[n, "Mean_|SHAP|"] for n in G.nodes()]
        
        # Node colors based on direction correlation strength
        if "Direction_corr" in df.columns:
            color_vals = [abs(df.loc[n, "Direction_corr"]) for n in G.nodes()]
            cmap = plt.cm.get_cmap(color)
            node_colors = [cmap(v) for v in color_vals]
        else:
            node_colors = "steelblue" if "NON" in title else "darkorange"
        
        # Draw
        if len(G.edges()) > 0:
            edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
            max_weight = max(edge_weights) if edge_weights else 1
            edge_widths = [0.5 + 2 * (w / max_weight) for w in edge_weights]
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=edge_widths, edge_color="gray")
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=sizes, alpha=0.85)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")
        
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
        
        # Stats annotation
        n_edges = len(G.edges())
        n_nodes = len(G.nodes())
        ax.text(0.02, 0.02, f"Nodes: {n_nodes}\nEdges: {n_edges}", 
               transform=ax.transAxes, fontsize=9, va="bottom")
    
    fig.suptitle("SHAP-Based Protein Modules by Outcome Direction", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig

def query_human_protein_atlas(
    proteins: list[str],
    atlas_type: Literal["tissue", "brain", "blood", "single_cell"] = "tissue"
) -> Optional[pd.DataFrame]:
    """
    Query Human Protein Atlas for tissue/cell expression profiles.
    
    Note: This queries the HPA API. For large queries, consider downloading
    the full dataset from https://www.proteinatlas.org/about/download
    
    """
    if not REQUESTS_AVAILABLE:
        print("requests library not available")
        return None
    
    base_url = "https://www.proteinatlas.org/api/search_download.php"
    
    # HPA uses specific column names for different atlases
    atlas_columns = {
        "tissue": "Tissue expression overview",
        "brain": "Brain regional expression",
        "blood": "Blood cell expression",
        "single_cell": "Single cell type expression"
    }
    
    results = []
    for prot in proteins:
        try:
            # Query HPA API
            params = {"search": prot, "format": "json", "columns": "g,eg"}
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    results.append({"Protein": prot, "Data": data[0] if isinstance(data, list) else data})
        except Exception as e:
            print(f"  Query failed for {prot}: {e}")
            continue
    
    if results:
        return pd.DataFrame(results)
    return None


def load_hpa_tissue_data(filepath: str) -> pd.DataFrame:
    """
    Load pre-downloaded HPA tissue expression data.
    
    Download from: https://www.proteinatlas.org/about/download
    File: 'normal_tissue.tsv.zip'
    """
    df = pd.read_csv(filepath, sep="\t")
    return df


def summarize_tissue_expression(
    hpa_data: pd.DataFrame,
    proteins: list[str],
    level_col: str = "Level",
    tissue_col: str = "Tissue",
    gene_col: str = "Gene name"
) -> pd.DataFrame:
    """
    Summarize which tissues express a set of proteins.
    
    Returns pivot table: proteins x tissues with expression levels.
    """
    subset = hpa_data[hpa_data[gene_col].isin(proteins)]
    
    # Convert levels to numeric
    level_map = {"Not detected": 0, "Low": 1, "Medium": 2, "High": 3}
    subset = subset.copy()
    subset["Level_num"] = subset[level_col].map(level_map)
    
    pivot = subset.pivot_table(
        index=gene_col, 
        columns=tissue_col, 
        values="Level_num",
        aggfunc="max"
    )
    return pivot


# =============================================================================
# SPECTRAL CLUSTERING OF PATIENTS
# ===========================================

def run_spectral_clustering(
    X_z: pd.DataFrame,
    y: pd.Series,
    k_range: tuple = None,
    n_neighbors: int = None,
    n_pca_denoise: int = 10,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Spectral clustering of patients based on protein profiles.
    
    Returns
    -------
    cluster_labels : Series
        Cluster assignment per patient
    metrics_df : DataFrame  
        Silhouette and chi-square metrics for each k
    """
    k_range = k_range or CONFIG.spectral_k_range
    n_neighbors = n_neighbors or CONFIG.spectral_n_neighbors
    
    X = X_z.dropna()
    y_aligned = y.loc[X.index].astype(int)
    
    # Denoise with PCA
    n_comp = min(n_pca_denoise, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=CONFIG.random_state)
    X_pca = pca.fit_transform(X)
    
    metrics = []
    best_score, best_labels = -np.inf, None
    
    for k in range(k_range[0], k_range[1] + 1):
        sc = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
            random_state=CONFIG.random_state,
        )
        labels = sc.fit_predict(X_pca)
        
        sil = silhouette_score(X_pca, labels)
        ct = pd.crosstab(labels, y_aligned)
        chi2, p_chi, _, _ = chi2_contingency(ct)
        
        metrics.append({"k": k, "silhouette": sil, "chi2_stat": chi2, "chi2_p": p_chi})
        
        if sil > best_score:
            best_score = sil
            best_labels = pd.Series(labels, index=X.index, name="Cluster")
    
    metrics_df = pd.DataFrame(metrics)
    print(f"Spectral clustering: best k={best_labels.nunique()} (silhouette={best_score:.3f})")
    
    return best_labels, metrics_df


def plot_cluster_outcomes(
    cluster_labels: pd.Series,
    y: pd.Series,
    figsize: tuple = (8, 5)
) -> plt.Figure:
    """Bar plot of outcome proportions per cluster."""
    fig, ax = plt.subplots(figsize=figsize)
    
    ct = pd.crosstab(cluster_labels, y, normalize="index") * 100
    ct.columns = ["No Worsening", "Worsening"]
    
    ct.plot(kind="bar", stacked=True, ax=ax, 
            color=["steelblue", "darkorange"], alpha=0.8)
    
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Patient Cluster")
    ax.set_title("Outcome Distribution by Cluster")
    ax.legend(title="Outcome")
    plt.xticks(rotation=0)
    fig.tight_layout()
    return fig


def get_cluster_defining_proteins(
    X_z: pd.DataFrame,
    cluster_labels: pd.Series,
    top_n: int = 20,
    method: str = "spread"
) -> tuple[pd.DataFrame, list]:
    """
    Identify proteins that best differentiate patient clusters.
    
    """
    X = X_z.loc[cluster_labels.index]
    cluster_means = X.groupby(cluster_labels).mean()
    
    if method == "spread":
        # Between-cluster spread per protein
        spread = cluster_means.max() - cluster_means.min()
        spread = spread.sort_values(ascending=False)
        top_proteins = spread.head(top_n).index.tolist()
    
    elif method == "ftest":
        from scipy.stats import f_oneway
        f_stats = {}
        for prot in X.columns:
            groups = [X.loc[cluster_labels == c, prot].dropna() for c in cluster_labels.unique()]
            if all(len(g) >= 2 for g in groups):
                f, p = f_oneway(*groups)
                f_stats[prot] = f
        f_series = pd.Series(f_stats).sort_values(ascending=False)
        top_proteins = f_series.head(top_n).index.tolist()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return cluster_means, top_proteins


def plot_cluster_protein_heatmap(
    X_z: pd.DataFrame,
    cluster_labels: pd.Series,
    top_n: int = 20,
    figsize: tuple = None,
) -> plt.Figure:
    """
    Heatmap showing proteins that differentiate patient clusters.
    
    Rows = top differentiating proteins
    Columns = patient clusters
    Values = mean z-score expression
    """
    cluster_means, top_proteins = get_cluster_defining_proteins(
        X_z, cluster_labels, top_n=top_n
    )
    
    # Subset to top proteins
    plot_data = cluster_means[top_proteins].T
    
    # Auto-size figure based on number of proteins
    if figsize is None:
        figsize = (3 + len(cluster_means), max(8, 0.35 * len(top_proteins)))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        plot_data,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Mean Z-score"}
    )
    
    ax.set_xlabel("Patient Cluster")
    ax.set_ylabel("Protein")
    ax.set_title(f"Top {top_n} Proteins Differentiating Patient Clusters")
    
    # Ensure all labels are shown
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    fig.tight_layout()
    return fig



# METHOD COMPARISON


def compare_all_importances(
    pca_loadings: pd.DataFrame,
    rf_importance: pd.DataFrame,
    de_stats: pd.DataFrame,
    pc: str = "PC1",
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Compare protein rankings across methods (PCA, RF, DE).
    
    This reveals which proteins are:
    - Consistently important (agree across methods) → robust biomarkers
    - Method-specific (important in only one) → method-dependent findings
    
    Note: LDA removed since RF significantly outperforms.
    """
    # Normalize to ranks
    common_prots = (pca_loadings.index
                    .intersection(rf_importance.index)
                    .intersection(de_stats.set_index("Protein").index))
    
    ranks = pd.DataFrame(index=common_prots)
    
    # PCA rank (by absolute loading)
    pca_abs = pca_loadings.loc[common_prots, pc].abs()
    ranks["PCA_rank"] = pca_abs.rank(ascending=False)
    
    # RF rank
    rf_imp = rf_importance.loc[common_prots, "RF_importance"]
    ranks["RF_rank"] = rf_imp.rank(ascending=False)
    
    # DE rank (by p-value)
    de_indexed = de_stats.set_index("Protein")
    de_p = de_indexed.loc[common_prots, "p_value"]
    ranks["DE_rank"] = de_p.rank(ascending=True)  # Lower p = better rank
    
    # Average rank
    ranks["Mean_rank"] = ranks.mean(axis=1)
    ranks = ranks.sort_values("Mean_rank")
    
    # Add original values for interpretation
    ranks["PCA_loading"] = pca_loadings.loc[common_prots, pc]
    ranks["RF_importance"] = rf_importance.loc[common_prots, "RF_importance"]
    ranks["DE_pvalue"] = de_indexed.loc[common_prots, "p_value"]
    
    print(f"\nTop {top_n} proteins by average rank across methods:")
    print(ranks.head(top_n)[["Mean_rank", "PCA_rank", "RF_rank", "DE_rank"]].to_string())
    
    return ranks


def plot_importance_comparison_heatmap(
    ranks_df: pd.DataFrame,
    top_n: int = 20,
    figsize: tuple = (8, 12)
) -> plt.Figure:
    """
    Heatmap comparing protein ranks across methods.
    
    Reveals agreement/disagreement between PCA, RF, and DE.
    """
    top_prots = ranks_df.head(top_n).index
    
    # Normalize ranks to 0-1 for visualization (1 = most important)
    rank_cols = ["PCA_rank", "RF_rank", "DE_rank"]
    plot_df = ranks_df.loc[top_prots, rank_cols].copy()
    
    # Convert rank to importance (invert so high = important)
    max_rank = plot_df.max().max()
    plot_df = 1 - (plot_df / max_rank)
    plot_df.columns = ["PCA", "RF", "DE"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        plot_df,
        cmap="YlOrRd",
        annot=ranks_df.loc[top_prots, rank_cols].astype(int).values,
        fmt="d",
        ax=ax,
        cbar_kws={"label": "Normalized Importance"}
    )
    
    ax.set_xlabel("Method")
    ax.set_ylabel("Protein")
    ax.set_title(f"Protein Importance Ranks Across Methods\n(annotations = rank, color = importance)")
    
    fig.tight_layout()
    return fig


# =============================================================================
# EXPORTS AND SUMMARY REPORT
# =============================================================================

def export_axis_proteins(
    stability_df: pd.DataFrame,
    output_dir: str = "results",
    pcs: list[str] = ["PC1", "PC2", "PC3"]
):
    """Export axis-defining proteins to text files for external tools."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for pc in pcs:
        proteins = get_axis_proteins(stability_df, pc)
        if proteins:
            filepath = out_path / f"{pc}_axis_proteins.txt"
            pd.Series(proteins).to_csv(filepath, index=False, header=False)
            print(f"Exported {len(proteins)} {pc} proteins to {filepath}")


def export_module_proteins(
    modules: pd.Series,
    output_dir: str = "results"
):
    """Export module memberships for external enrichment tools."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for mod_name, proteins in get_module_members(modules).items():
        filepath = out_path / f"{mod_name}_proteins.txt"
        pd.Series(proteins).to_csv(filepath, index=False, header=False)
        print(f"Exported {len(proteins)} proteins to {filepath}")


def create_summary_report(
    de_stats: pd.DataFrame,
    modules: pd.Series,
    stability_df: pd.DataFrame,
    module_enrichment: dict,
    pc_enrichment: dict,
) -> str:
    """Generate markdown summary of all analyses."""
    lines = ["# Proteomics Analysis Summary\n"]
    
    # DE summary
    n_sig = de_stats["Significant"].sum()
    lines.append(f"## Differential Expression\n")
    lines.append(f"- {n_sig} proteins significant at FDR < {CONFIG.p_threshold}\n")
    if n_sig > 0:
        top3 = de_stats[de_stats["Significant"]].head(3)["Protein"].tolist()
        lines.append(f"- Top hits: {', '.join(top3)}\n")
    
    # Module summary
    lines.append(f"\n## Co-expression Modules\n")
    lines.append(f"- {modules.nunique()} modules detected\n")
    for mod, prots in get_module_members(modules).items():
        lines.append(f"- {mod}: {len(prots)} proteins\n")
    
    # PCA summary
    lines.append(f"\n## PCA Axes\n")
    for pc in ["PC1", "PC2"]:
        axis_prots = get_axis_proteins(stability_df, pc)
        lines.append(f"- {pc}: {len(axis_prots)} axis-defining proteins\n")
    
    # Enrichment summaries
    if module_enrichment:
        lines.append(f"\n## Module Pathway Enrichment\n")
        for mod, enr in module_enrichment.items():
            if len(enr) > 0:
                top_term = enr.iloc[0]["Term"]
                lines.append(f"- {mod}: {top_term}\n")
    
    if pc_enrichment:
        lines.append(f"\n## PC Axis Pathway Enrichment\n")
        for pc, enr in pc_enrichment.items():
            if len(enr) > 0:
                top_term = enr.iloc[0]["Term"]
                lines.append(f"- {pc}: {top_term}\n")
    
    return "".join(lines)


# =============================================================================
# PANEL FIGURES (clustering, spectral)
# =============================================================================

def create_hierarchical_panel(
    X_z: pd.DataFrame,
    modules_hier: pd.Series,
    module_scores_hier: pd.DataFrame,
    module_tests_hier: pd.DataFrame,
    y: pd.Series,
    figsize: tuple = (16, 7),
) -> plt.Figure:
    """
    Panel figure: Hierarchical clustering heatmap + module boxplots.
    """
    fig = plt.figure(figsize=figsize)
    
    # Left: Heatmap (takes more space)
    ax1 = fig.add_axes([0.05, 0.1, 0.55, 0.8])  # [left, bottom, width, height]
    
    # Create heatmap manually (can't use clustermap in subplots easily)
    corr = X_z.corr()
    
    # Order proteins by module
    ordered_proteins = []
    for m in sorted(modules_hier.unique()):
        module_prots = modules_hier[modules_hier == m].index.tolist()
        if len(module_prots) > 1:
            module_corr = corr.loc[module_prots, module_prots]
            centrality = (module_corr.sum(axis=1) - 1) / (len(module_prots) - 1)
            module_prots = centrality.sort_values(ascending=False).index.tolist()
        ordered_proteins.extend(module_prots)
    
    corr_ord = corr.loc[ordered_proteins, ordered_proteins]
    
    im = ax1.imshow(corr_ord.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax1.set_xticks(range(len(ordered_proteins)))
    ax1.set_yticks(range(len(ordered_proteins)))
    ax1.set_xticklabels(ordered_proteins, rotation=90, fontsize=7)
    ax1.set_yticklabels(ordered_proteins, fontsize=7)
    ax1.set_title("Hierarchical Module Co-expression", fontsize=11, fontweight="bold")
    
    # Colorbar
    cbar_ax = fig.add_axes([0.05, 0.02, 0.55, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Correlation")
    
    # Right: Boxplots
    ax2 = fig.add_axes([0.68, 0.15, 0.30, 0.75])
    
    plot_df = module_scores_hier.copy()
    plot_df["Edema_Worse"] = y.loc[plot_df.index].values
    melted = plot_df.melt(id_vars="Edema_Worse", var_name="Module", value_name="Score")
    module_order = sorted(melted["Module"].unique())
    
    sns.boxplot(data=melted, x="Module", y="Score", hue="Edema_Worse",
               order=module_order, ax=ax2, palette={0: "steelblue", 1: "darkorange"})
    
    ax2.axhline(0, color="k", linestyle="--", linewidth=0.5)
    
    # Add p-values
    ymax = melted["Score"].max()
    ymin = melted["Score"].min()
    y_range = ymax - ymin
    
    for i, mod in enumerate(module_order):
        p_row = module_tests_hier[module_tests_hier["Module"] == mod]
        if len(p_row) > 0:
            p_val = p_row["p_value"].values[0]
            p_text = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
            ax2.text(i, ymax + y_range * 0.05, p_text, ha="center", va="bottom",
                    fontsize=8, fontweight="bold" if p_val < 0.05 else "normal")
    
    ax2.set_ylim(ymin - y_range * 0.05, ymax + y_range * 0.20)
    ax2.set_title("Module Scores vs Worsening", fontsize=11, fontweight="bold", pad=15)
    ax2.legend(title="Edema Worse", loc="upper right", fontsize=8)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    
    fig.suptitle("Hierarchical Clustering Analysis", fontsize=13, fontweight="bold", y=0.98)
    
    return fig


def create_kmeans_panel(
    X_z: pd.DataFrame,
    modules_km: pd.Series,
    module_scores_km: pd.DataFrame,
    module_tests_km: pd.DataFrame,
    y: pd.Series,
    figsize: tuple = (16, 7),
) -> plt.Figure:
    """
    Panel figure: K-means clustering heatmap + module boxplots.
    """
    fig = plt.figure(figsize=figsize)
    
    # Left: Heatmap
    ax1 = fig.add_axes([0.05, 0.1, 0.55, 0.8])
    
    corr = X_z.corr()
    
    # Order proteins by module
    ordered_proteins = []
    for m in sorted(modules_km.unique()):
        module_prots = modules_km[modules_km == m].index.tolist()
        if len(module_prots) > 1:
            module_corr = corr.loc[module_prots, module_prots]
            centrality = (module_corr.sum(axis=1) - 1) / (len(module_prots) - 1)
            module_prots = centrality.sort_values(ascending=False).index.tolist()
        ordered_proteins.extend(module_prots)
    
    corr_ord = corr.loc[ordered_proteins, ordered_proteins]
    
    im = ax1.imshow(corr_ord.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax1.set_xticks(range(len(ordered_proteins)))
    ax1.set_yticks(range(len(ordered_proteins)))
    ax1.set_xticklabels(ordered_proteins, rotation=90, fontsize=7)
    ax1.set_yticklabels(ordered_proteins, fontsize=7)
    ax1.set_title("K-means Module Co-expression", fontsize=11, fontweight="bold")
    
    # Colorbar
    cbar_ax = fig.add_axes([0.05, 0.02, 0.55, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Correlation")
    
    # Right: Boxplots
    ax2 = fig.add_axes([0.68, 0.15, 0.30, 0.75])
    
    plot_df = module_scores_km.copy()
    plot_df["Edema_Worse"] = y.loc[plot_df.index].values
    melted = plot_df.melt(id_vars="Edema_Worse", var_name="Module", value_name="Score")
    module_order = sorted(melted["Module"].unique())
    
    sns.boxplot(data=melted, x="Module", y="Score", hue="Edema_Worse",
               order=module_order, ax=ax2, palette={0: "steelblue", 1: "darkorange"})
    
    ax2.axhline(0, color="k", linestyle="--", linewidth=0.5)
    
    # Add p-values
    ymax = melted["Score"].max()
    ymin = melted["Score"].min()
    y_range = ymax - ymin
    
    for i, mod in enumerate(module_order):
        p_row = module_tests_km[module_tests_km["Module"] == mod]
        if len(p_row) > 0:
            p_val = p_row["p_value"].values[0]
            p_text = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
            ax2.text(i, ymax + y_range * 0.05, p_text, ha="center", va="bottom",
                    fontsize=8, fontweight="bold" if p_val < 0.05 else "normal")
    
    ax2.set_ylim(ymin - y_range * 0.05, ymax + y_range * 0.20)
    ax2.set_title("Module Scores vs Worsening", fontsize=11, fontweight="bold", pad=15)
    ax2.legend(title="Edema Worse", loc="upper right", fontsize=8)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    
    fig.suptitle("K-means Clustering Analysis", fontsize=13, fontweight="bold", y=0.98)
    
    return fig




def create_spectral_panel(
    cluster_labels: pd.Series,
    y: pd.Series,
    X_z: pd.DataFrame,
    top_n_proteins: int = 20,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Panel figure: Spectral cluster outcomes + cluster-defining proteins heatmap.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Cluster outcomes bar plot
    ax1 = axes[0]
    ct = pd.crosstab(cluster_labels, y, normalize="index") * 100
    ct.columns = ["No Worsening", "Worsening"]
    ct.plot(kind="bar", stacked=True, ax=ax1, color=["steelblue", "darkorange"], alpha=0.8)
    ax1.set_ylabel("Percentage")
    ax1.set_xlabel("Patient Cluster")
    ax1.set_title("Outcome Distribution by Cluster", fontsize=11, fontweight="bold")
    ax1.legend(title="Outcome", loc="upper right")
    ax1.tick_params(axis='x', rotation=0)
    
    # Add sample counts
    cluster_counts = cluster_labels.value_counts().sort_index()
    for i, (cluster, count) in enumerate(cluster_counts.items()):
        ax1.text(i, 102, f"n={count}", ha="center", va="bottom", fontsize=9)
    
    # Right: Protein heatmap
    ax2 = axes[1]
    
    # Get cluster-defining proteins
    X = X_z.loc[cluster_labels.index]
    cluster_means = X.groupby(cluster_labels).mean()
    spread = cluster_means.max() - cluster_means.min()
    top_proteins = spread.sort_values(ascending=False).head(top_n_proteins).index.tolist()
    
    plot_data = cluster_means[top_proteins].T
    
    im = ax2.imshow(plot_data.values, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)
    
    ax2.set_xticks(range(len(cluster_means)))
    ax2.set_xticklabels([f"Cluster {c}" for c in cluster_means.index], fontsize=9)
    ax2.set_yticks(range(len(top_proteins)))
    ax2.set_yticklabels(top_proteins, fontsize=8)
    ax2.set_xlabel("Patient Cluster")
    ax2.set_ylabel("Protein")
    ax2.set_title(f"Top {top_n_proteins} Cluster-Defining Proteins", fontsize=11, fontweight="bold")
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
    cbar.set_label("Mean Z-score")
    
    fig.suptitle("Spectral Clustering Analysis", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    
    return fig

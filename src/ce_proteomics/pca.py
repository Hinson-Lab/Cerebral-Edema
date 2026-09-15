"""
Principal component analysis of the 60-protein panel (Fig. 5, Supp. Fig. 6).

* PCA is fit on z-scored NPX for all subjects with an admission CT (n = 123).
* Fig. 5A: PC1 x PC2 scores colored by CE on the admission CT.
* Fig. 5B: PC scores compared between subjects who did and did not show
  CE worsening on the 6-hour CT (n = 116), Welch's t-test.
* Fig. 5C-D: "syndromic" plots of correlation loadings (eigenvector x sqrt of
  eigenvalue) for PC1 and PC2, showing proteins with |loading| above
  ``SYNDROMIC_LOADING_CUTOFF``.
* Supp. Fig. 6: scree plot and eigenvector coefficient bar plots.
* ``bootstrap_pca_stability`` identifies axis-defining proteins whose
  correlation loadings are stable across bootstrap resamples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA

from . import config
from .plotting import format_p_value


# =============================================================================
# Core computations
# =============================================================================

def zscore_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Z-score each column (mean 0, SD 1 with ddof = 1)."""
    return (X - X.mean()) / X.std(ddof=1)


def run_pca(
    X_z: pd.DataFrame,
    n_components: int = config.PCA_N_COMPONENTS,
    random_state: int = config.RANDOM_STATE,
) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """
    Fit PCA and return (pca, scores, eigenvector loadings).

    ``scores`` is samples x PCs (index preserved); ``loadings`` is
    proteins x PCs with the raw eigenvector coefficients.
    """
    n_components = min(n_components, X_z.shape[1], X_z.shape[0])
    pca = PCA(n_components=n_components, random_state=random_state)
    scores = pca.fit_transform(X_z)
    pc_names = [f"PC{i + 1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, columns=pc_names, index=X_z.index)
    loadings_df = pd.DataFrame(pca.components_.T, index=X_z.columns, columns=pc_names)
    return pca, scores_df, loadings_df


def compute_correlation_loadings(pca: PCA, feature_names: list[str]) -> pd.DataFrame:
    """
    Correlation loadings: eigenvector x sqrt(eigenvalue). For standardized
    inputs these are the correlations between each protein and the PC scores.
    """
    corr = pca.components_.T * np.sqrt(pca.explained_variance_)
    pc_names = [f"PC{i + 1}" for i in range(pca.n_components_)]
    return pd.DataFrame(corr, index=feature_names, columns=pc_names)


def bootstrap_pca_stability(
    X: np.ndarray,
    feature_names: list[str],
    n_components: int = config.PCA_N_COMPONENTS,
    n_boot: int = config.PCA_BOOTSTRAP_N,
    loading_thresh: float = config.PCA_LOADING_THRESHOLD,
    prop_thresh: float = config.PCA_STABILITY_THRESHOLD,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap PCA to identify stable axis-defining proteins.

    A protein is axis-defining for a PC if its mean bootstrap correlation
    loading magnitude is >= ``loading_thresh`` and it exceeds that threshold
    in >= ``prop_thresh`` of resamples. Bootstrap loadings are sign-aligned to
    the full-data reference solution before averaging.

    Returns
    -------
    stability_df : per-protein mean, SD, proportion stable, axis-defining flag per PC
    reference_loadings : correlation loadings from the full-data PCA
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    n_samples, n_features = X.shape
    n_components = min(n_components, n_features, n_samples)

    pca_ref = PCA(n_components=n_components, random_state=random_state).fit(X)
    ref_loadings = compute_correlation_loadings(pca_ref, feature_names)
    pcs = list(ref_loadings.columns)
    boot = {pc: np.zeros((n_boot, n_features)) for pc in pcs}

    for b in range(n_boot):
        idx = rng.integers(0, n_samples, size=n_samples)
        pca_b = PCA(n_components=n_components, random_state=random_state + b).fit(X[idx, :])
        loadings_b = compute_correlation_loadings(pca_b, feature_names)
        for pc in pcs:
            vec = loadings_b[pc].values
            if np.dot(vec, ref_loadings[pc].values) < 0:
                vec = -vec
            boot[pc][b, :] = vec

    results = {}
    for pc, mat in boot.items():
        mean_loading = mat.mean(axis=0)
        prop_above = (np.abs(mat) >= loading_thresh).mean(axis=0)
        results[f"{pc}_mean"] = mean_loading
        results[f"{pc}_sd"] = mat.std(axis=0)
        results[f"{pc}_prop_stable"] = prop_above
        results[f"{pc}_axis_defining"] = (
            (np.abs(mean_loading) >= loading_thresh) & (prop_above >= prop_thresh)
        )
    return pd.DataFrame(results, index=feature_names), ref_loadings


def get_axis_proteins(stability_df: pd.DataFrame, pc: str = "PC1") -> list[str]:
    """Proteins flagged as axis-defining for ``pc``."""
    col = f"{pc}_axis_defining"
    if col not in stability_df.columns:
        raise ValueError(f"'{pc}' not found in stability results")
    return stability_df.index[stability_df[col]].tolist()


def pc_group_tests(
    scores: pd.DataFrame,
    y: pd.Series,
    pcs: list[str] | None = None,
) -> pd.DataFrame:
    """
    Welch's t-test of PC scores between outcome groups (1 vs 0), with
    Cohen's d. ``y`` may cover a subset of ``scores`` (e.g. the 116 subjects
    with a follow-up CT); only overlapping, non-missing rows are tested.
    """
    pcs = pcs or list(scores.columns)
    y = y.dropna().astype(int)
    common = scores.index.intersection(y.index)
    rows = []
    for pc in pcs:
        g0 = scores.loc[common[y.loc[common] == 0], pc]
        g1 = scores.loc[common[y.loc[common] == 1], pc]
        t, p = ttest_ind(g1, g0, equal_var=False)
        d = (g1.mean() - g0.mean()) / np.sqrt((g0.var() + g1.var()) / 2)
        rows.append({"PC": pc, "N0": len(g0), "N1": len(g1), "Mean0": g0.mean(),
                     "Mean1": g1.mean(), "t_stat": t, "p_value": p, "Cohens_d": d})
    return pd.DataFrame(rows)


# =============================================================================
# Plots
# =============================================================================

def plot_pca_scatter(
    scores: pd.DataFrame,
    groups: pd.Series,
    pca: PCA,
    group_labels: tuple[str, str] = ("No CE (Baseline)", "CE (Baseline)"),
    colors: tuple[str, str] = (config.COLOR_CE_ABSENT, config.COLOR_CE_PRESENT),
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    title: str = "60-Protein Panel PCA",
    ax: plt.Axes | None = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """PC scatter colored by a binary grouping, with n and p-values in the legend."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    groups = groups.loc[scores.index].astype(int)
    tests = pc_group_tests(scores, groups, [pc_x, pc_y]).set_index("PC")

    for g, label, color in [(0, group_labels[0], colors[0]), (1, group_labels[1], colors[1])]:
        mask = groups == g
        ax.scatter(scores.loc[mask, pc_x], scores.loc[mask, pc_y], c=color, s=40,
                   alpha=0.85, edgecolor="white", linewidth=0.5,
                   label=f"{label}; n={int(mask.sum())}")
    for pc in (pc_x, pc_y):
        ax.plot([], [], " ", label=f"{pc} {format_p_value(tests.loc[pc, 'p_value'])}")

    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"{pc_x} ({evr[int(pc_x[2:]) - 1] * 100:.1f}%)")
    ax.set_ylabel(f"{pc_y} ({evr[int(pc_y[2:]) - 1] * 100:.1f}%)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.set_title(title)
    return fig


def plot_scree(pca: PCA, figsize: tuple = (8, 5)) -> plt.Figure:
    """Scree plot of variance explained per PC with the cumulative curve."""
    fig, ax = plt.subplots(figsize=figsize)
    evr = pca.explained_variance_ratio_ * 100
    pcs = np.arange(1, len(evr) + 1)
    ax.bar(pcs, evr, alpha=0.7, label="Individual")
    ax.plot(pcs, np.cumsum(evr), "o-", color="crimson", label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_xticks(pcs)
    ax.legend()
    ax.set_title("Scree plot")
    fig.tight_layout()
    return fig


def plot_loadings_bar(
    loadings: pd.DataFrame,
    pc: str = "PC1",
    top_n: int | None = None,
    ylabel: str = "Eigenvector coefficient",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Bar plot of loadings for one PC (positive red, negative blue)."""
    fig, ax = plt.subplots(figsize=figsize)
    ordered = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index)
    if top_n:
        ordered = ordered.head(top_n)
    colors = ["lightcoral" if v > 0 else "lightblue" for v in ordered.values]
    ax.bar(ordered.index, ordered.values, color=colors)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Protein")
    ax.set_title(f"{pc} loadings")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return fig


def draw_syndromic_loadings(
    ax: plt.Axes,
    corr_loadings: pd.DataFrame,
    pc_index: int,
    vaf: float,
    cutoff: float = config.SYNDROMIC_LOADING_CUTOFF,
    arrow_length: float = 4.5,
    text_size: int = 10,
    cmap_name: str = "coolwarm",
) -> ScalarMappable | None:
    """
    Draw a syndromic-style loading plot on ``ax``: one arrow per protein with
    |correlation loading| >= ``cutoff``, pointing at the PC, colored by the
    signed loading on a fixed [-1, 1] scale. Returns the ScalarMappable so a
    shared colorbar can be added by the caller.
    """
    pc_label = f"PC{pc_index}"
    data = corr_loadings[[pc_label]].rename(columns={pc_label: "loading"})
    data = data[data["loading"].abs() >= cutoff]
    data = data.reindex(data["loading"].abs().sort_values(ascending=False).index)

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=-1, vmax=1)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    vaf_pct = vaf * 100 if vaf <= 1 else vaf
    ax.text(0, 0, f"{pc_label}\n({vaf_pct:.1f}%)", fontsize=14, ha="center", va="center",
            weight="bold", zorder=4)
    ax.add_artist(plt.Circle((0, 0), 1.5, color="lightgrey", ec="black", zorder=3))

    if len(data) == 0:
        ax.text(0, -3, f"No proteins with |loading| >= {cutoff}", ha="center", fontsize=9)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(data), endpoint=False)
    for (protein, row), angle in zip(data.iterrows(), angles):
        value = float(row["loading"])
        color = cmap(norm(value))
        width = 0.3 + 0.6 * abs(value)
        x_end, y_end = 1.8 * np.cos(angle), 1.8 * np.sin(angle)
        x_start, y_start = (arrow_length + 1.0) * np.cos(angle), (arrow_length + 1.0) * np.sin(angle)
        ax.add_patch(patches.FancyArrow(
            x_start, y_start, x_end - x_start, y_end - y_start, width=width,
            head_width=width * 2, head_length=0.8, length_includes_head=True,
            color=color, alpha=0.95,
        ))
        lx, ly = (arrow_length + 2.6) * np.cos(angle), (arrow_length + 2.6) * np.sin(angle)
        ax.text(lx, ly, f"{protein}\n{value:.3f}", fontsize=text_size, ha="center", va="center")

    lim = arrow_length * 2.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    return sm


def plot_syndromic_loadings(
    corr_loadings: pd.DataFrame,
    pc_index: int,
    vaf: float,
    cutoff: float = config.SYNDROMIC_LOADING_CUTOFF,
    figsize: tuple = (9, 9),
) -> plt.Figure:
    """Stand-alone syndromic loading plot for one PC with a colorbar."""
    fig, ax = plt.subplots(figsize=figsize)
    sm = draw_syndromic_loadings(ax, corr_loadings, pc_index, vaf, cutoff=cutoff)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02, shrink=0.6)
    cbar.set_label("PC loadings")
    ax.set_title(f"PC{pc_index} correlation loadings (|loading| >= {cutoff})")
    return fig


def plot_pc_boxplots(
    scores: pd.DataFrame,
    y: pd.Series,
    pcs: list[str] = ("PC1", "PC2"),
    tests: pd.DataFrame | None = None,
    legend_title: str = "CE Worse (6 hr)",
    colors: tuple[str, str] = (config.COLOR_NO_WORSENING, config.COLOR_WORSENING),
    title: str = "PC Scores x CE Worsening",
    ax: plt.Axes | None = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Boxplots of PC scores by outcome with p-value annotations (Fig. 5B)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    pcs = list(pcs)
    y = y.dropna().astype(int)
    common = scores.index.intersection(y.index)
    if tests is None:
        tests = pc_group_tests(scores, y, pcs)
    tests = tests.set_index("PC")

    plot_df = scores.loc[common, pcs].copy()
    plot_df["Outcome"] = y.loc[common].values
    melted = plot_df.melt(id_vars="Outcome", var_name="PC", value_name="Score")
    sns.boxplot(data=melted, x="PC", y="Score", hue="Outcome", order=pcs,
                palette={0: colors[0], 1: colors[1]}, ax=ax)

    ymax, ymin = melted["Score"].max(), melted["Score"].min()
    span = ymax - ymin
    for i, pc in enumerate(pcs):
        ax.text(i, ymax + span * 0.04, format_p_value(tests.loc[pc, "p_value"]),
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(ymin - span * 0.05, ymax + span * 0.2)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("PC Score")
    ax.set_title(title)
    ax.legend(title=legend_title, loc="lower right", fontsize=9)
    return fig


def plot_pca_stability_heatmap(
    stability_df: pd.DataFrame,
    pcs: list[str] = ("PC1", "PC2", "PC3", "PC4", "PC5"),
    figsize: tuple = (8, 14),
) -> plt.Figure:
    """Heatmap of mean bootstrap correlation loadings with stable cells outlined."""
    pcs = [pc for pc in pcs if f"{pc}_mean" in stability_df.columns]
    means = stability_df[[f"{pc}_mean" for pc in pcs]].copy()
    means.columns = pcs
    stable = stability_df[[f"{pc}_axis_defining" for pc in pcs]].to_numpy()
    means = means.reindex(means.abs().max(axis=1).sort_values(ascending=False).index)
    stable = pd.DataFrame(stable, index=stability_df.index).loc[means.index].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(means, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"label": "Mean bootstrap correlation loading"})
    for i in range(stable.shape[0]):
        for j in range(stable.shape[1]):
            if stable[i, j]:
                ax.add_patch(patches.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=1.2))
    ax.set_title("Bootstrap loading stability (outlined = axis-defining)")
    fig.tight_layout()
    return fig


def create_pca_panel(
    scores: pd.DataFrame,
    corr_loadings: pd.DataFrame,
    pca: PCA,
    ce_baseline: pd.Series,
    worsening: pd.Series,
    pc_tests_worsening: pd.DataFrame,
    cutoff: float = config.SYNDROMIC_LOADING_CUTOFF,
    figsize: tuple = (16, 15),
) -> plt.Figure:
    """
    Fig. 5 four-panel layout: (A) PC1 x PC2 by baseline CE, (B) PC scores by
    6-hour worsening, (C) PC1 syndromic loadings, (D) PC2 syndromic loadings.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_pca_scatter(scores, ce_baseline, pca, ax=ax_a, title="60-Protein Panel PCA")
    plot_pc_boxplots(scores, worsening, ["PC1", "PC2"], tests=pc_tests_worsening, ax=ax_b)
    evr = pca.explained_variance_ratio_
    draw_syndromic_loadings(ax_c, corr_loadings, 1, evr[0], cutoff=cutoff)
    sm = draw_syndromic_loadings(ax_d, corr_loadings, 2, evr[1], cutoff=cutoff)

    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], "ABCD"):
        ax.text(-0.05, 1.05, f"{letter}.", transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="bottom", ha="right")
    cbar = fig.colorbar(sm, ax=[ax_c, ax_d], orientation="horizontal", pad=0.02, shrink=0.5)
    cbar.set_label("PC Loadings")
    return fig

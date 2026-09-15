"""
Univariate analyses.

Goal 1 discovery (Fig. 2, Supplementary Table 2)
    ``adjusted_logistic_screen``: one logistic regression per protein with CE on
    the admission CT as the outcome, adjusted for age, sex, and time to the
    baseline CT, followed by Benjamini-Hochberg FDR control across the full
    proteome-wide screen. Proteins with q < FDR_ALPHA form the 12-protein panel.

Panel description (Fig. 2)
    ``panel_group_tests`` and ``plot_panel_boxplots``: NPX boxplots for the
    panel proteins by CE status, with asterisks from FDR-adjusted t-tests.

Differential expression of the 60-protein panel (Fig. 4)
    ``differential_expression`` and ``plot_volcano``: Welch's t-test per protein
    with the difference in mean NPX as the log2 fold change (NPX is already on a
    log2 scale), Cohen's d, and BH-adjusted p-values.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from . import config
from .plotting import significance_stars


# =============================================================================
# Adjusted logistic screen
# =============================================================================

def adjusted_logistic_screen(
    df: pd.DataFrame,
    proteins: list[str],
    outcome_col: str = config.COL_CE_BASELINE,
    covariates: list[str] = config.ADJUSTMENT_COVARIATES,
    fdr_alpha: float = config.FDR_ALPHA,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fit one covariate-adjusted logistic regression per protein.

    Returns a DataFrame (sorted by q-value) with, per protein: the log-odds
    coefficient (Beta) and its standard error, odds ratio with 95% CI,
    p-value, BH-adjusted q-value, a Significant flag, and N. Proteins whose
    model fails to converge are recorded with NaN statistics and excluded from
    the FDR adjustment.
    """
    data = df[[outcome_col] + covariates + proteins].dropna(subset=[outcome_col] + covariates)
    y = data[outcome_col].astype(int)
    base = sm.add_constant(data[covariates].astype(float), has_constant="add")

    rows = []
    n_failed = 0
    for protein in proteins:
        x = data[protein].astype(float)
        mask = x.notna()
        X = base[mask].copy()
        X.insert(1, "Protein", x[mask].values)
        try:
            fit = sm.Logit(y[mask], X).fit(disp=0, maxiter=200)
            beta = float(fit.params["Protein"])
            se = float(fit.bse["Protein"])
            rows.append({
                "Protein": protein,
                "N": int(mask.sum()),
                "Beta": beta,
                "SE": se,
                "OR": float(np.exp(beta)),
                "OR_CI_lower": float(np.exp(beta - 1.959963984540054 * se)),
                "OR_CI_upper": float(np.exp(beta + 1.959963984540054 * se)),
                "p_value": float(fit.pvalues["Protein"]),
                "Converged": bool(fit.mle_retvals.get("converged", True)),
            })
        except Exception:  # singular matrix, perfect separation, etc.
            n_failed += 1
            rows.append({"Protein": protein, "N": int(mask.sum()), "Beta": np.nan, "SE": np.nan,
                         "OR": np.nan, "OR_CI_lower": np.nan, "OR_CI_upper": np.nan,
                         "p_value": np.nan, "Converged": False})

    results = pd.DataFrame(rows)
    valid = results["p_value"].notna()
    results["q_value"] = np.nan
    results.loc[valid, "q_value"] = multipletests(results.loc[valid, "p_value"], method="fdr_bh")[1]
    results["Significant"] = results["q_value"] < fdr_alpha
    results = results.sort_values(["q_value", "p_value"]).reset_index(drop=True)

    if verbose:
        print(f"  Screened {len(proteins)} proteins ({n_failed} failed fits); "
              f"{int(results['Significant'].sum())} with q < {fdr_alpha}.")
    return results


def select_fdr_panel(screen_results: pd.DataFrame) -> list[str]:
    """Proteins flagged Significant in the screen, ordered by q-value."""
    return screen_results.loc[screen_results["Significant"], "Protein"].tolist()


# =============================================================================
# Panel boxplots (Fig. 2)
# =============================================================================

def panel_group_tests(
    df: pd.DataFrame,
    proteins: list[str],
    outcome_col: str = config.COL_CE_BASELINE,
    equal_var: bool = config.PANEL_BOXPLOT_EQUAL_VAR,
) -> pd.DataFrame:
    """
    Two-sample t-test per panel protein (group 0 vs group 1) with BH
    adjustment across the panel, used only for the boxplot asterisks.
    """
    rows = []
    for protein in proteins:
        g0 = df.loc[df[outcome_col] == 0, protein].dropna()
        g1 = df.loc[df[outcome_col] == 1, protein].dropna()
        t, p = ttest_ind(g0, g1, equal_var=equal_var)
        rows.append({"Protein": protein, "t_stat": t, "p_value": p})
    out = pd.DataFrame(rows)
    out["q_value"] = multipletests(out["p_value"], method="fdr_bh")[1]
    out["Stars"] = out["q_value"].apply(significance_stars)
    return out


def plot_panel_boxplots(
    df: pd.DataFrame,
    proteins: list[str],
    tests: pd.DataFrame,
    outcome_col: str = config.COL_CE_BASELINE,
    group_labels: tuple[str, str] = ("Absent", "Present"),
    legend_title: str = "Cerebral Edema at Baseline",
    colors: tuple[str, str] = (config.COLOR_CE_ABSENT, config.COLOR_CE_PRESENT),
    title: str = "Admission-time plasma protein levels by cerebral edema status",
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """Boxplots of NPX per panel protein, split by outcome, with asterisks."""
    plot_df = df[[outcome_col] + proteins].dropna(subset=[outcome_col]).melt(
        id_vars=outcome_col, var_name="Protein", value_name="NPX"
    )
    stars = dict(zip(tests["Protein"], tests["Stars"]))

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=plot_df, x="Protein", y="NPX", hue=outcome_col, hue_order=[0, 1],
        order=proteins, palette={0: colors[0], 1: colors[1]}, ax=ax,
    )
    for i, protein in enumerate(proteins):
        marker = stars.get(protein, "")
        if marker:
            y_max = plot_df.loc[plot_df["Protein"] == protein, "NPX"].max()
            ax.text(i, y_max + 0.5, marker, ha="center", va="bottom", fontsize=12)

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, list(group_labels), title=legend_title)
    ax.set_ylabel("NPX")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90)
    ax.set_title(title, fontsize=16)
    fig.tight_layout()
    return fig


# =============================================================================
# Differential expression and volcano (Fig. 4)
# =============================================================================

def differential_expression(
    X: pd.DataFrame,
    y: pd.Series,
    method: Literal["ttest", "mannwhitney"] = "ttest",
    data_is_log: bool = True,
    fdr_alpha: float = config.FDR_ALPHA,
    group_names: tuple[str, str] = ("Group0", "Group1"),
) -> pd.DataFrame:
    """
    Per-protein association with a binary outcome.

    Parameters
    ----------
    X : samples x proteins matrix (NPX)
    y : binary outcome aligned to X
    method : 'ttest' (Welch) or 'mannwhitney'
    data_is_log : if True (Olink NPX), log2 fold change is the mean difference
    group_names : labels used for the two mean columns

    Returns
    -------
    DataFrame with Protein, Mean_<g0>, Mean_<g1>, Log2_FC, Cohens_d, p_value,
    p_adj (BH), Significant.
    """
    from scipy.stats import mannwhitneyu

    y = y.loc[X.index].astype(int)
    rows = []
    for protein in X.columns:
        vals = X[protein]
        g0 = vals[y == 0].dropna()
        g1 = vals[y == 1].dropna()
        if len(g0) < 3 or len(g1) < 3:
            continue
        if method == "ttest":
            _, pval = ttest_ind(g1, g0, equal_var=False)
        else:
            _, pval = mannwhitneyu(g1, g0, alternative="two-sided")
        mean0, mean1 = g0.mean(), g1.mean()
        pooled_sd = np.sqrt(((len(g0) - 1) * g0.var() + (len(g1) - 1) * g1.var())
                            / (len(g0) + len(g1) - 2))
        cohens_d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan
        log2fc = (mean1 - mean0) if data_is_log else np.log2((mean1 + 1e-8) / (mean0 + 1e-8))
        rows.append({
            "Protein": protein,
            f"Mean_{group_names[0]}": mean0,
            f"Mean_{group_names[1]}": mean1,
            "Log2_FC": log2fc,
            "Cohens_d": cohens_d,
            "p_value": pval,
        })
    de = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    de["p_adj"] = multipletests(de["p_value"], method="fdr_bh")[1]
    de["Significant"] = de["p_adj"] < fdr_alpha
    return de


def plot_volcano(
    de_stats: pd.DataFrame,
    contrast_label: str = "Group1 / Group0",
    label_top_n: int = 15,
    fdr_alpha: float = config.FDR_ALPHA,
    title: str = "Differential expression of the 60-protein panel",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Volcano plot (log2 fold change vs -log10 BH q) with labeled hits."""
    fig, ax = plt.subplots(figsize=figsize)
    de = de_stats.copy()
    de["neg_log10_q"] = -np.log10(de["p_adj"].clip(lower=1e-300))

    nonsig = de[~de["Significant"]]
    sig = de[de["Significant"]]
    ax.scatter(nonsig["Log2_FC"], nonsig["neg_log10_q"], alpha=0.5, color="gray", s=30,
               label="Not significant")
    ax.scatter(sig["Log2_FC"], sig["neg_log10_q"], alpha=0.85, color="crimson", s=50,
               label=f"FDR < {fdr_alpha}")
    ax.axhline(-np.log10(fdr_alpha), color="gray", linestyle="--", alpha=0.6)

    to_label = sig.nsmallest(label_top_n, "p_adj")
    texts = [ax.text(r["Log2_FC"], r["neg_log10_q"], r["Protein"], fontsize=8,
                     ha="center", va="bottom") for _, r in to_label.iterrows()]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5))
    except ImportError:
        pass

    ax.set_xlabel(f"log2 fold change ({contrast_label})")
    ax.set_ylabel("-log10(FDR-adjusted p)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig

"""
Bootstrapped elastic-net panel selection (Goal 2; 60-protein panel).

Procedure (manuscript Methods, "Mechanistic Biomarker Panel Selection")
-----------------------------------------------------------------------
1. Residualize every protein on the adjustment covariates (age, sex, time to
   the baseline CT) by ordinary least squares, because scikit-learn's
   penalized logistic regression has no unpenalized covariate block.
2. Standardize the residuals.
3. Draw ``EN_N_BOOTSTRAP`` stratified bootstrap resamples; on each, fit an
   elastic-net logistic regression (``LogisticRegressionCV``, l1_ratio = 0.5)
   with CE on the admission CT as the outcome.
4. Count how often each protein receives a non-zero coefficient. Proteins
   selected in more than ``EN_SELECTION_FREQUENCY`` of resamples form the
   stable panel.

Runtime note
------------
With ~5,400 proteins and 1,000 resamples this step is slow (hours on a
laptop). ``scripts/03_elastic_net_panel.py --n-bootstrap`` lowers the count
for testing; the manuscript value is in ``config.EN_N_BOOTSTRAP``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from . import config
from .compat import elastic_net_logistic_cv


def residualize_on_covariates(X: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    """
    Regress each column of ``X`` on ``covariates`` (with intercept) and return
    the residual matrix. Solved once for all proteins with a single least-squares
    fit, which is numerically identical to fitting one OLS model per protein.
    """
    C = np.column_stack([np.ones(len(covariates)), covariates.to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(C, X.to_numpy(dtype=float), rcond=None)
    residuals = X.to_numpy(dtype=float) - C @ beta
    return pd.DataFrame(residuals, index=X.index, columns=X.columns)


def bootstrap_elastic_net_selection(
    X_scaled: np.ndarray,
    y: pd.Series | np.ndarray,
    feature_names: list[str],
    n_bootstrap: int = config.EN_N_BOOTSTRAP,
    l1_ratio: float = config.EN_L1_RATIO,
    n_cs: int = config.EN_N_CS,
    inner_cv: int = config.EN_INNER_CV,
    scoring: str = config.EN_SCORING,
    max_iter: int = config.EN_MAX_ITER,
    coef_tolerance: float = config.EN_COEF_TOLERANCE,
    random_state: int = config.RANDOM_STATE,
    n_jobs: int = -1,
    verbose_every: int = 50,
) -> pd.DataFrame:
    """
    Stability selection with bootstrapped elastic-net logistic regression.

    Returns a DataFrame with one row per protein that was selected at least
    once: Selection Count, Selection Frequency, Mean Coefficient (over the
    resamples in which it was non-zero), sorted by frequency.
    """
    y = np.asarray(y).astype(int)
    selection_counts: dict[str, int] = defaultdict(int)
    coefficients: dict[str, list[float]] = defaultdict(list)

    for b in range(n_bootstrap):
        X_boot, y_boot = resample(
            X_scaled, y, stratify=y, replace=True, random_state=random_state + b
        )
        model = elastic_net_logistic_cv(
            l1_ratio,
            Cs=n_cs,
            cv=inner_cv,
            scoring=scoring,
            max_iter=max_iter,
            n_jobs=n_jobs,
            random_state=random_state + b,
        ).fit(X_boot, y_boot)

        coef = model.coef_[0]
        selected = np.abs(coef) > coef_tolerance
        for name, is_selected, value in zip(feature_names, selected, coef):
            if is_selected:
                selection_counts[name] += 1
                coefficients[name].append(float(value))

        if verbose_every and (b + 1) % verbose_every == 0:
            print(f"    bootstrap {b + 1}/{n_bootstrap} complete")

    stability = pd.DataFrame({
        "Protein": list(selection_counts.keys()),
        "Selection Count": list(selection_counts.values()),
    })
    stability["Selection Frequency"] = stability["Selection Count"] / n_bootstrap
    stability["Mean Coefficient"] = [np.mean(coefficients[p]) for p in stability["Protein"]]
    stability = stability.sort_values("Selection Frequency", ascending=False).reset_index(drop=True)
    stability.attrs["n_bootstrap"] = n_bootstrap
    return stability


def select_stable_panel(
    stability: pd.DataFrame,
    frequency_threshold: float = config.EN_SELECTION_FREQUENCY,
) -> list[str]:
    """Proteins selected in more than ``frequency_threshold`` of resamples."""
    return stability.loc[
        stability["Selection Frequency"] > frequency_threshold, "Protein"
    ].tolist()


def prepare_elastic_net_inputs(
    df: pd.DataFrame,
    proteins: list[str],
    outcome_col: str = config.COL_CE_BASELINE,
    covariates: list[str] = config.ADJUSTMENT_COVARIATES,
) -> tuple[np.ndarray, pd.Series, list[str]]:
    """Residualize, standardize, and return (X_scaled, y, feature_names)."""
    data = df.dropna(subset=[outcome_col] + covariates).copy()
    X = data[proteins].astype(float)
    X = X.dropna(axis=1)  # proteins with any missing NPX cannot be residualized
    residuals = residualize_on_covariates(X, data[covariates].astype(float))
    X_scaled = StandardScaler().fit_transform(residuals)
    return X_scaled, data[outcome_col].astype(int), list(residuals.columns)


def plot_stability_vs_effect(
    stability: pd.DataFrame,
    highlight: list[str],
    highlight_label: str = "FDR-significant in univariate screen",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Selection frequency vs mean coefficient, highlighting the univariate hits."""
    n_boot = stability.attrs.get("n_bootstrap", None)
    plot_df = stability.copy()
    plot_df["Highlight"] = plot_df["Protein"].isin(highlight)

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=plot_df, x="Selection Frequency", y="Mean Coefficient", hue="Highlight",
        palette={True: "crimson", False: "gray"}, edgecolor="black", ax=ax,
    )
    ax.axhline(0, linestyle="--", color="black", linewidth=0.7)
    ax.axvline(config.EN_SELECTION_FREQUENCY, linestyle=":", color="black", linewidth=0.8,
               label=f"Selection threshold ({config.EN_SELECTION_FREQUENCY:.0%})")
    ax.set_title("Elastic-net feature stability and effect size")
    ax.set_xlabel("Selection frequency" + (f" ({n_boot} bootstraps)" if n_boot else ""))
    ax.set_ylabel("Mean non-zero coefficient")
    handles, labels = ax.get_legend_handles_labels()
    labels = [highlight_label if l == "True" else ("Other" if l == "False" else l) for l in labels]
    ax.legend(handles, labels, title="")

    texts = [ax.text(r["Selection Frequency"], r["Mean Coefficient"], r["Protein"], fontsize=8)
             for _, r in plot_df[plot_df["Highlight"]].iterrows()]
    try:
        from adjustText import adjust_text
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    except ImportError:
        pass
    fig.tight_layout()
    return fig

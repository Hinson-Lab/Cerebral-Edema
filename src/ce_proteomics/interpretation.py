"""
Model interpretation for the 60-protein random forest (Goal 2; Fig. 6B-C,
Supp. Fig. 7-8, inputs to Fig. 7).

* ``fit_interpretation_rf``: the random forest fit on z-scored NPX of the
  60-protein panel with 6-hour CE worsening as the outcome (with a
  cross-validated performance check).
* Mean-decrease-in-impurity importance and permutation importance
  (Supp. Fig. 7).
* TreeSHAP values for the worsening class, a direction summary (correlation
  between protein level and SHAP value), protein lists by direction for
  enrichment tools (Fig. 7), bar and beeswarm plots (Fig. 6B-C).
* Pairwise TreeSHAP interaction values from the same fitted model, restricted
  for display to the top proteins by mean |SHAP| (Supp. Fig. 8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import config

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    SHAP_AVAILABLE = False


# =============================================================================
# Random forest and importance
# =============================================================================

def _interpretation_rf(random_state: int = config.RANDOM_STATE) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.INTERP_RF_N_ESTIMATORS,
        max_depth=config.INTERP_RF_MAX_DEPTH,
        min_samples_leaf=config.INTERP_RF_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=random_state,
    )


def fit_interpretation_rf(
    X_z: pd.DataFrame,
    y: pd.Series,
    random_state: int = config.RANDOM_STATE,
) -> tuple[RandomForestClassifier, pd.DataFrame, dict]:
    """
    Fit the interpretation random forest and return (model, MDI importance,
    cross-validated performance summary).
    """
    X = X_z.dropna()
    y = y.loc[X.index].astype(int)
    rf = _interpretation_rf(random_state).fit(X, y)
    importance = pd.DataFrame({"RF_importance": rf.feature_importances_}, index=X.columns) \
        .sort_values("RF_importance", ascending=False)

    cv = StratifiedKFold(n_splits=config.INTERP_RF_CV_FOLDS, shuffle=True, random_state=random_state)
    acc = cross_val_score(_interpretation_rf(random_state), X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(_interpretation_rf(random_state), X, y, cv=cv, scoring="roc_auc")
    cv_results = {"cv_accuracy_mean": acc.mean(), "cv_accuracy_std": acc.std(),
                  "cv_auc_mean": auc.mean(), "cv_auc_std": auc.std()}
    print(f"  Interpretation RF: CV accuracy = {acc.mean():.3f} +/- {acc.std():.3f}; "
          f"CV AUC = {auc.mean():.3f} +/- {auc.std():.3f}")
    return rf, importance, cv_results


def get_rf_permutation_importance(
    X_z: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = config.PERMUTATION_N_REPEATS,
    test_size: float = config.PERMUTATION_TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """
    Permutation importance (ROC-AUC decrease) on a stratified hold-out split,
    which is less biased than MDI for correlated features.
    """
    X = X_z.dropna()
    y = y.loc[X.index].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y,
                                              random_state=random_state)
    rf = _interpretation_rf(random_state).fit(X_tr, y_tr)
    perm = permutation_importance(rf, X_te, y_te, n_repeats=n_repeats, random_state=random_state,
                                  scoring="roc_auc")
    return pd.DataFrame({"perm_importance_mean": perm.importances_mean,
                         "perm_importance_std": perm.importances_std}, index=X.columns) \
        .sort_values("perm_importance_mean", ascending=False)


def plot_importance_bar(
    importance: pd.DataFrame,
    column: str,
    top_n: int = 20,
    xlabel: str = "Feature importance",
    title: str = "Random forest feature importance",
    error_column: str | None = None,
    figsize: tuple = (9, 8),
) -> plt.Figure:
    """Horizontal bar plot of the top features by an importance column."""
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(top))
    xerr = top[error_column] if error_column else None
    ax.barh(y_pos, top[column], xerr=xerr, color="forestgreen", alpha=0.8, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top.index)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# =============================================================================
# SHAP
# =============================================================================

def compute_shap_values(rf: RandomForestClassifier, X: pd.DataFrame):
    """
    TreeSHAP values for the worsening class (shape n_samples x n_features),
    plus the explainer for interaction values. Handles both the list and the
    3-D array output formats of different shap versions.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is required for this step: pip install shap")
    explainer = shap.TreeExplainer(rf)
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        values = np.asarray(raw[1])
    else:
        values = np.asarray(raw)
        if values.ndim == 3:
            values = values[:, :, 1]
    if values.shape != (len(X), X.shape[1]):
        raise ValueError(f"Unexpected SHAP shape {values.shape}; expected {(len(X), X.shape[1])}")
    return values, explainer


def shap_direction_summary(shap_values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """
    Per-protein mean |SHAP| and the Pearson correlation between protein level
    and SHAP value (positive: higher level pushes toward worsening).
    """
    rows = []
    for i, col in enumerate(X.columns):
        s = shap_values[:, i]
        v = X[col].to_numpy(dtype=float)
        ok = ~(np.isnan(s) | np.isnan(v))
        corr = np.corrcoef(v[ok], s[ok])[0, 1] if ok.sum() > 3 and np.std(s[ok]) > 0 else np.nan
        rows.append({"Protein": col, "Mean_|SHAP|": float(np.abs(s).mean()), "Direction_corr": corr})
    df = pd.DataFrame(rows).sort_values("Mean_|SHAP|", ascending=False).reset_index(drop=True)
    df["Direction"] = np.where(df["Direction_corr"] > 0, "Higher -> worsening", "Higher -> non-worsening")
    return df


def shap_protein_lists(
    summary: pd.DataFrame,
    top_n: int = config.SHAP_LIST_TOP_N,
    direction_corr: float = config.SHAP_LIST_DIRECTION_CORR,
) -> dict[str, list[str]]:
    """
    Split the top ``top_n`` proteins by mean |SHAP| into those pushing
    predictions toward worsening and toward non-worsening, using the
    correlation between level and SHAP value.
    """
    top = summary.head(top_n)
    return {
        "toward_worsening": top.loc[top["Direction_corr"] > direction_corr, "Protein"].tolist(),
        "toward_non_worsening": top.loc[top["Direction_corr"] < -direction_corr, "Protein"].tolist(),
        "unassigned": top.loc[top["Direction_corr"].abs() <= direction_corr, "Protein"].tolist(),
    }


def plot_shap_bar(
    summary: pd.DataFrame,
    top_n: int = config.SHAP_TOP_N,
    ax: plt.Axes | None = None,
    figsize: tuple = (7, 7),
) -> plt.Figure:
    """Mean |SHAP| per protein, colored by direction of association (Fig. 6B)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    top = summary.head(top_n)
    colors = [config.COLOR_SHAP_WORSENING if d > 0 else config.COLOR_SHAP_NON_WORSENING
              for d in top["Direction_corr"]]
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["Mean_|SHAP|"], color=colors, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["Protein"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance")
    ax.legend(handles=[Patch(facecolor=config.COLOR_SHAP_WORSENING, label="Higher -> CE worsening"),
                       Patch(facecolor=config.COLOR_SHAP_NON_WORSENING, label="Higher -> CE non-worsening")],
              loc="lower right", fontsize=8)
    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    top_n: int = config.SHAP_TOP_N,
    ax: plt.Axes | None = None,
    figsize: tuple = (7, 7),
    random_state: int = config.RANDOM_STATE,
) -> plt.Figure:
    """
    Beeswarm of per-subject SHAP values for the top proteins, colored by the
    subject's relative (min-max scaled) protein level (Fig. 6C).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    rng = np.random.default_rng(random_state)
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    scatter = None
    for row, idx in enumerate(top_idx):
        s = shap_values[:, idx]
        v = X.iloc[:, idx].to_numpy(dtype=float)
        rel = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-12)
        jitter = rng.uniform(-0.25, 0.25, size=len(s))
        scatter = ax.scatter(s, np.full(len(s), row) + jitter, c=rel, cmap="RdBu_r", s=16,
                             alpha=0.7, vmin=0, vmax=1)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(X.columns[top_idx])
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xlabel("SHAP value")
    ax.set_title("SHAP Feature Value Influence")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02, ticks=[0, 1])
        cbar.ax.set_yticklabels(["Low", "High"])
        cbar.set_label("Relative Protein Level")
    return fig


def create_shap_panel(
    summary: pd.DataFrame,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    top_n: int = config.SHAP_TOP_N,
    figsize: tuple = (15, 7),
) -> plt.Figure:
    """Fig. 6B-C side by side: SHAP bar plot and beeswarm."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_shap_bar(summary, top_n=top_n, ax=axes[0])
    plot_shap_beeswarm(shap_values, X, top_n=top_n, ax=axes[1])
    for ax, letter in zip(axes, "BC"):
        ax.text(-0.15, 1.02, f"{letter}.", transform=ax.transAxes, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


# =============================================================================
# SHAP interactions (Supp. Fig. 8)
# =============================================================================

def compute_shap_interactions(
    explainer,
    X: pd.DataFrame,
    shap_values: np.ndarray,
    top_n_features: int = config.SHAP_TOP_N,
) -> tuple[np.ndarray, pd.DataFrame, list[str], pd.DataFrame]:
    """
    TreeSHAP pairwise interaction values from the same fitted model, computed
    on all features and then restricted for display to the top proteins by
    mean |SHAP| (same ranking as Fig. 6B-C).

    Returns
    -------
    interaction_subset : (n_samples, top_n, top_n) array
    pairs : unique pairs ranked by mean |interaction|
    top_features : proteins in display order
    matrix : mean |interaction| matrix (diagonal main effects set to NaN)
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][: min(top_n_features, X.shape[1])]
    top_features = X.columns[top_idx].tolist()

    raw = explainer.shap_interaction_values(X)
    if isinstance(raw, list):
        full = np.asarray(raw[1])
    else:
        full = np.asarray(raw)
        if full.ndim == 4:
            full = full[:, :, :, 1]
    expected = (len(X), X.shape[1], X.shape[1])
    if full.shape != expected:
        raise ValueError(f"Unexpected interaction shape {full.shape}; expected {expected}")

    subset = full[:, top_idx, :][:, :, top_idx]
    mean_abs = np.abs(subset).mean(axis=0)
    np.fill_diagonal(mean_abs, np.nan)
    matrix = pd.DataFrame(mean_abs, index=top_features, columns=top_features)

    pairs = [{"Protein_1": top_features[i], "Protein_2": top_features[j],
              "Interaction_Strength": mean_abs[i, j]}
             for i in range(len(top_features)) for j in range(i + 1, len(top_features))]
    pairs_df = pd.DataFrame(pairs).sort_values("Interaction_Strength", ascending=False) \
        .reset_index(drop=True)
    return subset, pairs_df, top_features, matrix


def plot_shap_interaction_heatmap(
    matrix: pd.DataFrame,
    show_values: bool = True,
    figsize: tuple = (11, 9),
) -> plt.Figure:
    """Lower-triangle heatmap of mean |TreeSHAP interaction| (diagonal omitted)."""
    n = matrix.shape[0]
    values = matrix.to_numpy(dtype=float)
    mask = np.triu(np.ones((n, n), dtype=bool), k=0)
    off_diag = values[~np.eye(n, dtype=bool)]
    off_diag = off_diag[np.isfinite(off_diag)]
    vmax = float(off_diag.max()) if len(off_diag) else 1.0

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(values, xticklabels=matrix.columns, yticklabels=matrix.index, cmap="viridis",
                ax=ax, mask=mask, annot=show_values, fmt=".4f", annot_kws={"size": 7},
                vmin=0, vmax=vmax, square=True, linewidths=0.35, linecolor="white",
                cbar_kws={"label": "Mean absolute TreeSHAP interaction value", "shrink": 0.78})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xlabel("Protein")
    ax.set_ylabel("Protein")
    ax.set_title(f"Pairwise SHAP interactions among the top {n} protein contributors\n"
                 "Ordered by mean |SHAP|; diagonal main effects omitted", fontsize=11, pad=12)
    fig.tight_layout()
    return fig


def compare_importance_ranks(
    corr_loadings: pd.DataFrame,
    rf_importance: pd.DataFrame,
    de_stats: pd.DataFrame,
    shap_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rank every protein under each method for a side-by-side comparison."""
    ranks = pd.DataFrame(index=rf_importance.index)
    ranks["PC1_rank"] = corr_loadings["PC1"].abs().rank(ascending=False).reindex(ranks.index)
    ranks["PC2_rank"] = corr_loadings["PC2"].abs().rank(ascending=False).reindex(ranks.index)
    ranks["RF_rank"] = rf_importance["RF_importance"].rank(ascending=False)
    de = de_stats.set_index("Protein")
    ranks["DE_rank"] = de["p_value"].rank().reindex(ranks.index)
    if shap_summary is not None:
        s = shap_summary.set_index("Protein")
        ranks["SHAP_rank"] = s["Mean_|SHAP|"].rank(ascending=False).reindex(ranks.index)
    ranks["Mean_rank"] = ranks.mean(axis=1)
    return ranks.sort_values("Mean_rank")

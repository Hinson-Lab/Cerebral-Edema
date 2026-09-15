"""
Figures for the supervised classification results (Fig. 3, Fig. 6A,
Supp. Fig. 4-5). Every function returns a matplotlib Figure; saving is done by
the calling script through ``plotting.save_figure``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

from . import config
from .classification import calibration_bin_summary, compute_calibration_metrics, evaluate_at_threshold

COLORS = config.CLASSIFIER_COLORS


def plot_roc_curves(roc_curves: dict, title: str, figsize: tuple = (8, 7)) -> plt.Figure:
    """ROC curves for all classifiers with AUC in the legend."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, (fpr, tpr, auc_val) in roc_curves.items():
        ax.plot(fpr, tpr, color=COLORS[name], linewidth=2.0, label=f"{name} (AUC = {auc_val:.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_auc_forest(
    results_df: pd.DataFrame,
    metric: str,
    title: str,
    prevalence: float | None = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Forest plot of per-classifier ROC-AUC or PR-AUC with bootstrap 95% CIs."""
    if metric == "roc":
        est, lo_col, hi_col = "AUC (Pooled)", "AUC CI Lower", "AUC CI Upper"
        xlabel, reference, ref_label = "ROC-AUC", 0.5, "No skill = 0.50"
    elif metric == "pr":
        est, lo_col, hi_col = "PR-AUC (AP)", "PR-AUC CI Lower", "PR-AUC CI Upper"
        xlabel, reference = "PR-AUC (average precision)", prevalence
        ref_label = None if prevalence is None else f"No skill = {prevalence:.3f}"
    else:
        raise ValueError("metric must be 'roc' or 'pr'")

    plot_df = results_df.sort_values(est, ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=figsize)
    for i, row in plot_df.iterrows():
        c = COLORS[row["Classifier"]]
        ax.errorbar(row[est], i,
                    xerr=[[max(0, row[est] - row[lo_col])], [max(0, row[hi_col] - row[est])]],
                    fmt="o", color=c, ecolor=c, markerfacecolor=c, markeredgecolor="black",
                    markeredgewidth=0.8, capsize=4, linewidth=2, markersize=9)
        ax.text(min(1.01, row[hi_col] + 0.02), i,
                f"{row[est]:.2f} ({row[lo_col]:.2f}-{row[hi_col]:.2f})", va="center", fontsize=11)
    if reference is not None:
        ax.axvline(reference, color="black", linestyle="--", linewidth=0.9, label=ref_label)
        ax.legend(loc="lower left", fontsize=10)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["Classifier"], fontsize=13)
    ax.set_xlim(0, 1.10)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.grid(axis="x", linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_pr_curves(pr_curves: dict, y, title: str, figsize: tuple = (8, 7)) -> plt.Figure:
    """Step-style precision-recall curves for all classifiers."""
    prevalence = float(np.mean(y))
    fig, ax = plt.subplots(figsize=figsize)
    for name, (recall, precision, ap) in pr_curves.items():
        ax.step(recall, precision, where="post", color=COLORS[name], linewidth=1.8,
                label=f"{name} (AP = {ap:.3f})")
    ax.axhline(prevalence, color="black", linestyle="--", linewidth=1.0,
               label=f"No-skill prevalence = {prevalence:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_best_model_pr_curve(
    y, y_prob, model_name: str, title: str,
    ap_value: float, ap_lower: float, ap_upper: float, youden_threshold: float,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """PR curve for one model with its AP CI, no-skill line, and Youden point."""
    y_arr = np.asarray(y, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    precision, recall, _ = precision_recall_curve(y_arr, y_prob)
    prevalence = float(y_arr.mean())
    op = evaluate_at_threshold(y_arr, y_prob, youden_threshold)

    fig, ax = plt.subplots(figsize=figsize)
    c = COLORS[model_name]
    ax.step(recall, precision, where="post", color=c, linewidth=2.0,
            label=f"{model_name}: AP = {ap_value:.3f} ({ap_lower:.3f}-{ap_upper:.3f})")
    ax.axhline(prevalence, color="black", linestyle="--", linewidth=1.0,
               label=f"No-skill prevalence = {prevalence:.3f}")
    ax.plot(op["Recall"], op["Precision"], marker="o", markersize=9, linestyle="none", color=c,
            markeredgecolor="black", markeredgewidth=0.8,
            label=(f"Youden operating point: threshold={youden_threshold:.3f}, "
                   f"precision={op['Precision']:.2f}, recall={op['Recall']:.2f}"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_youden_precision_recall_forest(
    results_df: pd.DataFrame, title: str, prevalence: float, figsize: tuple = (14, 5.8),
) -> plt.Figure:
    """Precision and recall (with CIs) at each classifier's Youden operating point."""
    plot_df = results_df.sort_values("Recall (Youden)", ascending=True).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    panels = [
        (axes[0], "Precision (Youden)", "Precision (Youden) CI Lower", "Precision (Youden) CI Upper",
         "Precision at Youden operating point"),
        (axes[1], "Recall (Youden)", "Recall (Youden) CI Lower", "Recall (Youden) CI Upper",
         "Recall at Youden operating point"),
    ]
    for ax, est, lo_col, hi_col, panel_title in panels:
        for i, row in plot_df.iterrows():
            c = COLORS[row["Classifier"]]
            ax.errorbar(row[est], i,
                        xerr=[[max(0, row[est] - row[lo_col])], [max(0, row[hi_col] - row[est])]],
                        fmt="o", color=c, ecolor=c, markerfacecolor=c, markeredgecolor="black",
                        markeredgewidth=0.6, capsize=4, linewidth=1.5, markersize=7)
            ax.text(min(1.01, row[hi_col] + 0.015), i,
                    f"{row[est]:.2f} ({row[lo_col]:.2f}-{row[hi_col]:.2f})", va="center", fontsize=8)
        ax.set_xlim(0, 1.10)
        ax.set_xlabel("Proportion")
        ax.set_title(panel_title)
        ax.grid(axis="x", linestyle="--", linewidth=0.5)
    axes[0].axvline(prevalence, color="black", linestyle="--", linewidth=0.9,
                    label=f"No-skill precision = prevalence ({prevalence:.3f})")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_yticks(range(len(plot_df)))
    axes[0].set_yticklabels(plot_df["Classifier"])
    fig.suptitle(f"Precision and recall at Youden operating point: {title}", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    return fig


def plot_calibration_curves(
    pooled_probs: dict, y, title: str, n_bins: int = config.N_CALIBRATION_BINS,
) -> plt.Figure:
    """
    Calibration plots (equal-frequency bins, Wilson CIs, Brier/slope/intercept
    inset, and a faint histogram of predicted probabilities), one panel per model.
    """
    n_clf = len(pooled_probs)
    n_cols = min(3, max(1, n_clf))
    n_rows = int(np.ceil(n_clf / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows), squeeze=False)
    axes = axes.flatten()
    y_arr = np.asarray(y, dtype=int)

    for idx, (name, prob) in enumerate(pooled_probs.items()):
        ax = axes[idx]
        prob = np.asarray(prob, dtype=float)
        bins = calibration_bin_summary(y_arr, prob, n_bins=n_bins)
        x, obs = bins["Mean Predicted"].to_numpy(), bins["Observed"].to_numpy()
        lo, hi = bins["CI Lower"].to_numpy(), bins["CI Upper"].to_numpy()
        c = COLORS.get(name, "#4D4D4D")
        ax.errorbar(x, obs, yerr=np.vstack([np.maximum(0, obs - lo), np.maximum(0, hi - obs)]),
                    fmt="o-", color=c, ecolor=c, markerfacecolor=c, markeredgecolor="white",
                    markeredgewidth=0.7, capsize=4, linewidth=1.5, markersize=6,
                    label=f"{n_bins} quantile bins", zorder=3)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, label="Perfect calibration", zorder=1)
        ax_hist = ax.twinx()
        ax_hist.hist(prob, bins=20, range=(0, 1), color="gray", alpha=0.2, edgecolor="none")
        ax_hist.set_yticks([])
        ax_hist.set_ylim(0, ax_hist.get_ylim()[1] * 4)
        m = compute_calibration_metrics(y_arr, prob)
        ax.text(0.04, 0.96, f"Brier = {m['Brier Score']:.3f}\nSlope = {m['Cal. Slope']:.2f}\n"
                f"Intercept = {m['Cal. Intercept']:.2f}", transform=ax.transAxes, fontsize=9,
                va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability", fontsize=10)
        ax.set_ylabel("Observed proportion", fontsize=10)
        ax.set_title(name, fontsize=13)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_box_aspect(1)
    for j in range(n_clf, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Calibration: {title}", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


def plot_kde_for_best_model(
    pooled_probs: dict, y, best_model: str, title: str, thresh_youden: float | None = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Predicted-probability densities by true class for the best model."""
    prob = np.asarray(pooled_probs[best_model])
    y_arr = np.asarray(y, dtype=int)
    fig, ax = plt.subplots(figsize=figsize)
    sns.kdeplot(prob[y_arr == 0], label="No worsening", fill=True, alpha=0.5, ax=ax)
    sns.kdeplot(prob[y_arr == 1], label="Worsening", fill=True, alpha=0.5, ax=ax)
    ax.axvline(0.5, color="#666666", linestyle="--", label="Threshold = 0.5")
    if thresh_youden is not None:
        ax.axvline(thresh_youden, color=COLORS[best_model], linestyle=":", linewidth=2.0,
                   label=f"Youden = {thresh_youden:.2f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\n{best_model}: probability distributions")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_accuracy_barplot(fold_metrics: pd.DataFrame, title: str, figsize: tuple = (12, 6)) -> plt.Figure:
    """Fold-level accuracy (mean and SD) per classifier at the 0.5 threshold."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=fold_metrics, x="Classifier", y="Accuracy", errorbar="sd", hue="Classifier",
                palette=COLORS, dodge=False, legend=False, ax=ax)
    ax.set_title(f"Fold accuracy at threshold 0.5: {title}", fontsize=14)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_composite_roc_panel(all_roc_curves: dict, figsize_per_panel: tuple = (6, 5)) -> plt.Figure:
    """One ROC panel per feature set."""
    n_sets = len(all_roc_curves)
    n_cols = min(3, n_sets)
    n_rows = int(np.ceil(n_sets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
                             squeeze=False)
    axes = axes.flatten()
    for ax, (set_name, roc_curves) in zip(axes, all_roc_curves.items()):
        for clf_name, (fpr, tpr, auc_val) in roc_curves.items():
            ax.plot(fpr, tpr, color=COLORS[clf_name], linewidth=1.8,
                    label=f"{clf_name} (AUC = {auc_val:.2f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(set_name)
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend(loc="lower right", fontsize=7)
    for j in range(n_sets, len(axes)):
        axes[j].axis("off")
    fig.suptitle("ROC curves by feature set", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_best_model_forest(all_results: dict, metric: str, figsize: tuple = (10, 6)) -> plt.Figure:
    """Best classifier per feature set, with bootstrap CIs, on one axis."""
    if metric == "roc":
        est, lo_col, hi_col, xlabel, ref = "AUC (Pooled)", "AUC CI Lower", "AUC CI Upper", "ROC-AUC", 0.5
    elif metric == "pr":
        est, lo_col, hi_col, xlabel, ref = "PR-AUC (AP)", "PR-AUC CI Lower", "PR-AUC CI Upper", \
            "PR-AUC (average precision)", None
    else:
        raise ValueError("metric must be 'roc' or 'pr'")
    rows = []
    for set_name, data in all_results.items():
        r = data["results_df"].loc[data["results_df"][est].idxmax()]
        rows.append({"Feature Set": set_name, "Classifier": r["Classifier"], "Estimate": r[est],
                     "Lower": r[lo_col], "Upper": r[hi_col]})
    plot_df = pd.DataFrame(rows).sort_values("Estimate").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=figsize)
    for i, row in plot_df.iterrows():
        c = COLORS[row["Classifier"]]
        ax.errorbar(row["Estimate"], i,
                    xerr=[[max(0, row["Estimate"] - row["Lower"])], [max(0, row["Upper"] - row["Estimate"])]],
                    fmt="o", color=c, ecolor=c, markerfacecolor=c, markeredgecolor="black",
                    markeredgewidth=0.6, capsize=4, linewidth=1.5, markersize=7)
        ax.text(min(1.02, row["Upper"] + 0.02), i,
                f"{row['Estimate']:.3f} ({row['Lower']:.3f}-{row['Upper']:.3f})", va="center", fontsize=9)
    if ref is not None:
        ax.axvline(ref, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["Feature Set"] + ": " + plot_df["Classifier"])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(xlabel)
    ax.set_title(f"Best model per feature set: {xlabel} with bootstrap 95% CIs")
    ax.grid(axis="x", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig

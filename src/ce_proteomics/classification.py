"""
Supervised prediction of 6-hour cerebral edema worsening (Goal 1; Fig. 3,
Fig. 6A, Supp. Fig. 4-5).

Design
------
* Six classifiers (ridge logistic regression, elastic-net logistic
  regression, LDA, random forest, RBF SVM, XGBoost) with fixed
  hyperparameters from ``config``.
* Stratified k-fold cross-validation (optionally repeated). Every
  data-dependent step is fit on the training folds only:
  - standardization (``StandardScaler`` inside a ``Pipeline``),
  - XGBoost class weighting (``scale_pos_weight`` from the training fold),
  - clinical feature selection by recursive feature elimination with a
    random-forest estimator and inner cross-validation (RFECV), with the
    12-protein panel held fixed and only clinical candidates eliminated
    (``rfecv_fixed_block``).
* Performance is summarized from the pooled out-of-fold (OOF) predicted
  probabilities: ROC-AUC, PR-AUC (average precision), subject-level bootstrap
  95% CIs, the post-hoc Youden operating point with bootstrap CIs, Brier score,
  and calibration slope/intercept by logistic recalibration.
* ``delong_roc_test`` compares correlated ROC curves (DeLong et al. 1988,
  fast implementation of Sun and Xu 2014).

Setting ``config.SCALE_WITHIN_FOLDS = False`` reproduces the earlier version of
this analysis in which the scaler was fit once on the full analytic cohort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .compat import elastic_net_logistic, unpenalized_logistic

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False


# =============================================================================
# Classifiers and cross-validation objects
# =============================================================================

def build_classifiers(random_state: int = config.RANDOM_STATE) -> dict:
    """The six classifiers compared in the manuscript, with fixed hyperparameters."""
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=config.CLF_LOGREG_MAX_ITER, random_state=random_state,
            class_weight="balanced",
        ),
        "Elastic Net": elastic_net_logistic(
            config.CLF_EN_L1_RATIO, max_iter=config.CLF_EN_MAX_ITER, random_state=random_state,
            class_weight="balanced",
        ),
        "LDA": LinearDiscriminantAnalysis(),
        "Random Forest": RandomForestClassifier(
            n_estimators=config.CLF_RF_N_ESTIMATORS, random_state=random_state,
            class_weight="balanced",
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, random_state=random_state,
            class_weight="balanced",
        ),
    }
    if XGBOOST_AVAILABLE:
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=config.CLF_XGB_N_ESTIMATORS, max_depth=config.CLF_XGB_MAX_DEPTH,
            learning_rate=config.CLF_XGB_LEARNING_RATE, random_state=random_state,
            scale_pos_weight=1.0, eval_metric="logloss",
        )
    else:
        print("  Warning: xgboost is not installed; the XGBoost classifier is skipped.")
    missing_colors = set(classifiers).difference(config.CLASSIFIER_COLORS)
    if missing_colors:
        raise ValueError("Missing CLASSIFIER_COLORS entries for: " + ", ".join(sorted(missing_colors)))
    return classifiers


def make_model(classifier, scale_within_folds: bool = config.SCALE_WITHIN_FOLDS):
    """Wrap a classifier in a scaler + classifier pipeline when requested."""
    if scale_within_folds:
        return Pipeline([("scaler", StandardScaler()), ("clf", classifier)])
    return classifier


def make_cv(
    n_splits: int = config.CV_N_SPLITS,
    n_repeats: int = config.CV_N_REPEATS,
    random_state: int = config.RANDOM_STATE,
):
    """Stratified k-fold (repeated when ``n_repeats`` > 1)."""
    if n_repeats > 1:
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=random_state)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# =============================================================================
# Recursive feature elimination with a fixed protein block
# =============================================================================

def rfecv_fixed_block(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fixed_cols: list[str],
    candidate_cols: list[str],
    n_estimators: int | None = None,
    inner_cv_splits: int = config.RFE_INNER_CV,
    scoring: str = config.RFE_SCORING,
    min_candidates: int = config.RFE_MIN_CLINICAL,
    random_state: int = config.RANDOM_STATE,
) -> tuple[list[str], pd.DataFrame]:
    """
    Recursive feature elimination with cross-validation, random-forest
    estimator, and a fixed feature block.

    Starting from ``fixed_cols + candidate_cols``, the candidate with the
    lowest random-forest importance is removed at each step and the inner
    cross-validated score of the remaining set is recorded. The candidate
    subset with the highest inner CV score is returned (ties resolved toward
    fewer features). ``fixed_cols`` are never eliminated. Only training-fold
    data must be passed in, so held-out observations never inform selection.

    Returns
    -------
    selected : list of retained candidate columns
    path : DataFrame of the elimination path (n_candidates, cv_score, removed)
    """
    n_estimators = n_estimators or config.RFE_RF_N_ESTIMATORS
    remaining = list(candidate_cols)
    inner_cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=random_state)
    path = []
    while True:
        cols = list(fixed_cols) + remaining
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state,
                                    class_weight="balanced")
        score = cross_val_score(rf, X_train[cols], y_train, cv=inner_cv, scoring=scoring).mean()
        path.append({"n_candidates": len(remaining), "cv_score": float(score),
                     "candidate_set": list(remaining)})
        if len(remaining) <= min_candidates:
            break
        rf.fit(X_train[cols], y_train)
        importances = pd.Series(rf.feature_importances_, index=cols)
        weakest = importances[remaining].idxmin()
        remaining.remove(weakest)
        path[-1]["removed_next"] = weakest

    path_df = pd.DataFrame(path)
    best = path_df.sort_values(["cv_score", "n_candidates"], ascending=[False, True]).iloc[0]
    return list(best["candidate_set"]), path_df


# =============================================================================
# Metric helpers
# =============================================================================

def bootstrap_metric_ci(y_true, y_prob, metric_func, n_boot=config.N_BOOTSTRAP_CI,
                        ci=config.BOOTSTRAP_CI_LEVEL, seed=config.RANDOM_STATE):
    """
    Subject-level bootstrap CI for a probability-based metric. Resamples that
    contain a single outcome class are skipped.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    estimates = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        estimates.append(metric_func(y_true[idx], y_prob[idx]))
    if not estimates:
        raise RuntimeError("No bootstrap resample contained both outcome classes.")
    alpha = 1 - ci
    return (np.percentile(estimates, 100 * alpha / 2),
            np.percentile(estimates, 100 * (1 - alpha / 2)))


def wilson_interval(events: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion."""
    if total <= 0:
        return np.nan, np.nan
    p = events / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return center - half, center + half


def find_optimal_threshold(y_true, y_prob, method: str = "youden", target_recall: float = 0.80) -> float:
    """
    Post-hoc classification threshold.

    'youden' maximizes sensitivity + specificity - 1; 'target_recall' returns
    the highest threshold achieving ``target_recall``; 'f1' maximizes F1.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    if method == "youden":
        finite = np.isfinite(thresholds)
        idx = np.where(finite)[0]
        return float(thresholds[idx[np.argmax((tpr - fpr)[finite])]])
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    if method == "target_recall":
        valid = np.where(recall[:-1] >= target_recall)[0]
        return float(thr[0] if len(valid) == 0 else thr[valid[-1]])
    if method == "f1":
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
        return float(thr[np.argmax(f1)])
    raise ValueError(f"Unknown method: {method}")


def evaluate_at_threshold(y_true, y_prob, threshold: float) -> dict:
    """Accuracy, precision, recall, and specificity at a fixed threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    total_neg = int((y_true == 0).sum())
    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "Specificity": tn / total_neg if total_neg else 0.0,
    }


def bootstrap_youden_operating_point_ci(y_true, y_prob, n_boot=config.N_BOOTSTRAP_CI,
                                        ci=config.BOOTSTRAP_CI_LEVEL, seed=config.RANDOM_STATE) -> dict:
    """
    Bootstrap CIs for the Youden threshold and its precision, recall, and
    specificity. The threshold is re-estimated within each resample, so the
    intervals reflect uncertainty in the operating point itself.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    store = {"Threshold": [], "Precision": [], "Recall": [], "Specificity": []}
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        thr = find_optimal_threshold(y_true[idx], y_prob[idx], method="youden")
        m = evaluate_at_threshold(y_true[idx], y_prob[idx], thr)
        store["Threshold"].append(thr)
        for key in ("Precision", "Recall", "Specificity"):
            store[key].append(m[key])
    if not store["Threshold"]:
        raise RuntimeError("No bootstrap resample contained both outcome classes.")
    alpha = 1 - ci
    out = {}
    for key, values in store.items():
        out[f"{key} CI Lower"] = float(np.percentile(values, 100 * alpha / 2))
        out[f"{key} CI Upper"] = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return out


def calibration_bin_summary(y_true, y_prob, n_bins: int = config.N_CALIBRATION_BINS) -> pd.DataFrame:
    """Equal-frequency calibration bins with Wilson CIs for the observed proportion."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    ranks = pd.Series(y_prob).rank(method="first")
    bin_ids = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop")
    rows = []
    for bin_id in sorted(pd.Series(bin_ids).dropna().unique()):
        mask = np.asarray(bin_ids == bin_id)
        n_bin = int(mask.sum())
        events = int(y_true[mask].sum())
        lower, upper = wilson_interval(events, n_bin)
        rows.append({"Bin": int(bin_id) + 1, "Mean Predicted": float(y_prob[mask].mean()),
                     "Observed": events / n_bin, "CI Lower": lower, "CI Upper": upper,
                     "N": n_bin, "Events": events})
    return pd.DataFrame(rows)


def compute_calibration_metrics(y_true, y_prob) -> dict:
    """Brier score plus calibration slope/intercept from logistic recalibration."""
    eps = 1e-8
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    recal = unpenalized_logistic(max_iter=1000).fit(logit, y_true)
    return {
        "Brier Score": brier_score_loss(y_true, y_prob),
        "Cal. Slope": float(recal.coef_[0][0]),
        "Cal. Intercept": float(recal.intercept_[0]),
    }


# =============================================================================
# DeLong test for correlated ROC curves
# =============================================================================

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n)
    out[order] = t + 1
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(positive[r])
        ty[r] = _compute_midrank(negative[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    return aucs, sx / m + sy / n


def delong_roc_test(y_true, prob_a, prob_b) -> dict:
    """
    Two-sided DeLong test comparing the ROC-AUCs of two models evaluated on
    the same subjects. Returns both AUCs, their difference, z, and p.
    """
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-y_true, kind="mergesort")  # positives first
    label_1_count = int(y_true.sum())
    preds = np.vstack([np.asarray(prob_a, float), np.asarray(prob_b, float)])[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else np.nan
    p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
    return {"AUC_A": float(aucs[0]), "AUC_B": float(aucs[1]), "Difference": float(diff),
            "z": float(z), "p_value": float(p)}


# =============================================================================
# Cross-validation driver
# =============================================================================

def cross_validate_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    fixed_cols: list[str],
    candidate_cols: list[str] | None = None,
    use_rfecv: bool = False,
    classifiers: dict | None = None,
    cv=None,
    scale_within_folds: bool = config.SCALE_WITHIN_FOLDS,
    random_state: int = config.RANDOM_STATE,
    verbose: bool = True,
) -> dict:
    """
    Run all classifiers through cross-validation for one feature set.

    Parameters
    ----------
    X : features (index = subject ID); rows with missing values are dropped
    y : binary outcome aligned to X
    fixed_cols : columns always included
    candidate_cols : columns subject to RFECV (ignored when use_rfecv is False)
    use_rfecv : select clinical candidates by RFECV inside each training fold

    Returns
    -------
    dict with keys: y (Series), pooled_probs (dict), oof (dict of arrays with
    shape [n_repeats, n]), fold_metrics (DataFrame), rfecv_log (DataFrame or
    None), columns_used (list)
    """
    classifiers = classifiers or build_classifiers(random_state)
    cv = cv or make_cv(random_state=random_state)
    candidate_cols = list(candidate_cols or [])
    cols_all = list(fixed_cols) + [c for c in candidate_cols if c not in fixed_cols]

    data = pd.concat([X[cols_all], y.rename("_y")], axis=1).dropna()
    X = data[cols_all].astype(float)
    y = data["_y"].astype(int)
    n = len(y)
    if verbose:
        print(f"  N = {n} | positives = {int(y.sum())} | features = {len(cols_all)}"
              + (" (RFECV over clinical candidates)" if use_rfecv else ""))

    if not scale_within_folds:  # legacy behavior: scaler fit on the full cohort
        X = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)

    n_repeats = getattr(cv, "n_repeats", 1)
    n_splits = cv.get_n_splits() // n_repeats
    oof = {name: np.full((n_repeats, n), np.nan) for name in classifiers}
    fold_rows, rfecv_rows = [], []

    for k, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        repeat, fold = divmod(k, n_splits)
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        if use_rfecv:
            selected, path = rfecv_fixed_block(X_tr, y_tr, list(fixed_cols), candidate_cols,
                                               random_state=random_state + k)
            cols = list(fixed_cols) + selected
            rfecv_rows.append({"Repeat": repeat + 1, "Fold": fold + 1, "N selected": len(selected),
                               "Selected": "; ".join(selected),
                               "Inner CV score": float(path["cv_score"].max())})
        else:
            cols = cols_all

        pos, neg = int(y_tr.sum()), int((y_tr == 0).sum())
        for name, clf in classifiers.items():
            estimator = clone(clf)
            if name == "XGBoost":
                estimator.set_params(scale_pos_weight=neg / max(pos, 1))
            model = make_model(estimator, scale_within_folds)
            model.fit(X_tr[cols], y_tr)
            prob = model.predict_proba(X_te[cols])[:, 1]
            oof[name][repeat, test_idx] = prob
            pred = (prob >= 0.5).astype(int)
            fold_rows.append({"Repeat": repeat + 1, "Fold": fold + 1, "Classifier": name,
                              "Accuracy": accuracy_score(y_te, pred),
                              "Precision": precision_score(y_te, pred, zero_division=0),
                              "Recall": recall_score(y_te, pred)})

    pooled = {name: np.nanmean(mat, axis=0) for name, mat in oof.items()}
    return {
        "y": y,
        "pooled_probs": pooled,
        "oof": oof,
        "fold_metrics": pd.DataFrame(fold_rows),
        "rfecv_log": pd.DataFrame(rfecv_rows) if use_rfecv else None,
        "columns_used": cols_all,
    }


def summarize_performance(
    y: pd.Series,
    pooled_probs: dict,
    fold_metrics: pd.DataFrame,
    n_bootstrap: int = config.N_BOOTSTRAP_CI,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Per-classifier performance table from pooled OOF predictions, plus ROC and
    PR curve coordinates for plotting.
    """
    y_arr = np.asarray(y, dtype=int)
    results, roc_curves, pr_curves = [], {}, {}
    for i, (name, prob) in enumerate(pooled_probs.items()):
        auc_val = roc_auc_score(y_arr, prob)
        auc_lo, auc_hi = bootstrap_metric_ci(y_arr, prob, roc_auc_score, n_boot=n_bootstrap,
                                             seed=random_state + i)
        fpr, tpr, _ = roc_curve(y_arr, prob)
        roc_curves[name] = (fpr, tpr, auc_val)

        ap = average_precision_score(y_arr, prob)
        pr_lo, pr_hi = bootstrap_metric_ci(y_arr, prob, average_precision_score,
                                           n_boot=n_bootstrap, seed=random_state + 1000 + i)
        precision, recall, _ = precision_recall_curve(y_arr, prob)
        pr_curves[name] = (recall, precision, ap)

        thr = find_optimal_threshold(y_arr, prob, method=config.YOUDEN_METHOD)
        at_thr = evaluate_at_threshold(y_arr, prob, thr)
        youden_ci = bootstrap_youden_operating_point_ci(y_arr, prob, n_boot=n_bootstrap,
                                                        seed=random_state + 1000 + i)
        cal = compute_calibration_metrics(y_arr, prob)
        folds = fold_metrics[fold_metrics["Classifier"] == name]

        results.append({
            "Classifier": name,
            "Accuracy": folds["Accuracy"].mean(),
            "Precision": folds["Precision"].mean(),
            "Recall": folds["Recall"].mean(),
            "AUC (Pooled)": auc_val,
            "AUC 95% CI": f"({auc_lo:.3f}-{auc_hi:.3f})",
            "AUC CI Lower": auc_lo, "AUC CI Upper": auc_hi,
            "PR-AUC (AP)": ap,
            "PR-AUC 95% CI": f"({pr_lo:.3f}-{pr_hi:.3f})",
            "PR-AUC CI Lower": pr_lo, "PR-AUC CI Upper": pr_hi,
            "Brier Score": cal["Brier Score"],
            "Cal. Slope": cal["Cal. Slope"],
            "Cal. Intercept": cal["Cal. Intercept"],
            "Thresh (Youden)": thr,
            "Thresh (Youden) CI Lower": youden_ci["Threshold CI Lower"],
            "Thresh (Youden) CI Upper": youden_ci["Threshold CI Upper"],
            "Recall (Youden)": at_thr["Recall"],
            "Recall (Youden) CI Lower": youden_ci["Recall CI Lower"],
            "Recall (Youden) CI Upper": youden_ci["Recall CI Upper"],
            "Precision (Youden)": at_thr["Precision"],
            "Precision (Youden) CI Lower": youden_ci["Precision CI Lower"],
            "Precision (Youden) CI Upper": youden_ci["Precision CI Upper"],
            "Specificity (Youden)": at_thr["Specificity"],
            "Specificity (Youden) CI Lower": youden_ci["Specificity CI Lower"],
            "Specificity (Youden) CI Upper": youden_ci["Specificity CI Upper"],
        })
    return pd.DataFrame(results), roc_curves, pr_curves


def get_best_model_by_auc(results_df: pd.DataFrame) -> str:
    """Classifier with the highest pooled ROC-AUC."""
    return results_df.loc[results_df["AUC (Pooled)"].idxmax(), "Classifier"]


def oof_predictions_frame(y: pd.Series, pooled_probs: dict) -> pd.DataFrame:
    """Pooled OOF probabilities as a tidy frame (one column per classifier)."""
    out = pd.DataFrame({config.COL_ID: y.index, "y_true": y.values})
    for name, prob in pooled_probs.items():
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out[f"prob_{safe}"] = prob
    return out


def rfecv_selection_frequency(rfecv_log: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """How often each clinical candidate was retained across outer folds."""
    counts = {c: 0 for c in candidates}
    for selected in rfecv_log["Selected"]:
        for c in [s for s in selected.split("; ") if s]:
            counts[c] = counts.get(c, 0) + 1
    n_folds = len(rfecv_log)
    out = pd.DataFrame({"Clinical variable": list(counts), "Folds selected": list(counts.values())})
    out["Selection frequency"] = out["Folds selected"] / n_folds
    return out.sort_values("Selection frequency", ascending=False).reset_index(drop=True)

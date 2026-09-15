"""
Version-safe constructors for scikit-learn logistic regression models.

scikit-learn 1.8 deprecated the ``penalty`` argument in favor of ``l1_ratio``
(``LogisticRegression``) and ``l1_ratios`` (``LogisticRegressionCV``), and
``penalty=None`` in favor of ``C=numpy.inf``. Older releases need the
``penalty`` argument to fit an elastic net at all. These helpers hide that
difference so the analysis modules read the same under either API.
"""

from __future__ import annotations

import warnings

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


def _version_tuple() -> tuple[int, int]:
    parts = sklearn.__version__.split(".")
    return int(parts[0]), int(parts[1].split("rc")[0].split("a")[0].split("b")[0])


SKLEARN_NEW_API = _version_tuple() >= (1, 8)


def elastic_net_logistic(l1_ratio: float, **kwargs) -> LogisticRegression:
    """Elastic-net penalized logistic regression (saga solver)."""
    if SKLEARN_NEW_API:
        return LogisticRegression(solver="saga", l1_ratio=l1_ratio, **kwargs)
    return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, **kwargs)


def elastic_net_logistic_cv(l1_ratio: float, **kwargs) -> LogisticRegressionCV:
    """Elastic-net LogisticRegressionCV over a grid of C values (saga solver)."""
    if SKLEARN_NEW_API:
        warnings.filterwarnings("ignore", message=".*use_legacy_attributes.*", category=FutureWarning)
        return LogisticRegressionCV(solver="saga", l1_ratios=[l1_ratio], **kwargs)
    return LogisticRegressionCV(penalty="elasticnet", solver="saga", l1_ratios=[l1_ratio], **kwargs)


def unpenalized_logistic(**kwargs) -> LogisticRegression:
    """Plain (unpenalized) logistic regression, used for calibration slope/intercept."""
    if SKLEARN_NEW_API:
        return LogisticRegression(C=np.inf, solver="lbfgs", **kwargs)
    return LogisticRegression(penalty=None, solver="lbfgs", **kwargs)

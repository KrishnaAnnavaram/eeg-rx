"""scikit-learn pipelines. Scaling and feature selection are steps of the pipeline, so they are
fit only on the training subjects of each fold. ``clone`` gives each fold a fresh, unfitted model.
"""

from __future__ import annotations

import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSIFIERS = ("logreg", "svm", "mlp")
SELECTORS = ("anova", "l1", "none")
K_GRID = (8, 16, 32, 64, 128)


def make_classifier(name: str, seed: int):
    if name == "logreg":
        return LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)
    if name == "svm":
        # Platt scaling on 3 internal folds of the training data (SVC(probability=True) is deprecated).
        return CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
                                      method="sigmoid", cv=3, ensemble=False)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=800, random_state=seed)
    raise ValueError(f"unknown classifier {name!r}. Use one of {CLASSIFIERS}")


def _l1_logreg() -> LogisticRegression:
    """An L1 logistic regression. scikit-learn 1.8 replaced ``penalty="l1"`` with ``l1_ratio=1``."""
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, solver="liblinear", C=0.5, class_weight="balanced")
    return LogisticRegression(penalty="l1", solver="liblinear", C=0.5, class_weight="balanced")


def make_selector(name: str):
    if name == "anova":
        return SelectKBest(f_classif, k=32)
    if name == "l1":
        # Rank by |L1 coefficient| and keep the top k. A fixed threshold can keep zero features.
        return SelectFromModel(_l1_logreg(), threshold=-np.inf, max_features=32)
    if name == "none":
        return "passthrough"
    raise ValueError(f"unknown selector {name!r}. Use one of {SELECTORS}")


def make_pipeline(classifier: str = "logreg", selector: str = "anova", seed: int = 42) -> Pipeline:
    return Pipeline([
        ("variance", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("select", make_selector(selector)),
        ("clf", make_classifier(classifier, seed)),
    ])


def param_grid(selector: str, n_features: int) -> dict:
    """The inner-loop grid. It selects the feature count (or the L1 strength) inside the folds."""
    if selector == "anova":
        ks = [k for k in K_GRID if k < n_features] + ["all"]
        return {"select__k": ks}
    if selector == "l1":
        return {"select__max_features": [k for k in K_GRID if k < n_features] or [n_features]}
    return {}

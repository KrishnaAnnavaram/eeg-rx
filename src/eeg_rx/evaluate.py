"""Nested, subject-wise evaluation with subject-level metrics, intervals and a permutation test.

- Outer loop: StratifiedGroupKFold over subjects (or leave one subject out). No subject is in a
  training fold and a test fold at the same time. ``check_no_subject_overlap`` enforces this.
- Inner loop: GridSearchCV with group folds on the training subjects only. It selects the
  feature count. Scaling and selection are pipeline steps, so they see only training subjects.
- Each subject gets one score: the mean epoch probability. All metrics are subject-level.
- The permutation test shuffles labels between SUBJECTS and repeats the full nested loop.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

from .config import Settings
from .features import FeatureSet
from .model import make_pipeline, param_grid


class LeakageError(AssertionError):
    pass


def check_no_subject_overlap(train_groups, test_groups) -> None:
    both = set(train_groups) & set(test_groups)
    if both:
        raise LeakageError(f"subjects in train and test: {sorted(both)[:5]}")


def subject_labels(fs: FeatureSet) -> dict[str, int]:
    table: dict[str, int] = {}
    for s, y in zip(fs.subjects, fs.labels):
        if table.setdefault(s, int(y)) != int(y):
            raise ValueError(f"subject {s} has two labels")
    return table


def outer_splitter(n_folds: int, seed: int):
    return LeaveOneGroupOut() if n_folds == 0 else StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)


@dataclass
class SubjectPredictions:
    subjects: list[str]
    y: np.ndarray
    p: np.ndarray
    fold_params: list[dict] = field(default_factory=list)
    epoch_auroc: float = float("nan")


@contextmanager
def expected_warnings():
    """Hide two expected, harmless warnings of small folds.

    A small inner fold can hold one class only. Its AUROC is NaN, and GridSearchCV ranks it last.
    A histogram bin can be constant inside one fold. SelectKBest then gives it a NaN score.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        warnings.filterwarnings("ignore", message=".*non-finite.*")
        warnings.filterwarnings("ignore", message="Features .* are constant")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="(invalid value|divide by zero) encountered")
        yield


def nested_cv(fs: FeatureSet, s: Settings, labels: np.ndarray | None = None) -> SubjectPredictions:
    """Return one out-of-fold probability for each subject."""
    with expected_warnings():
        return _nested_cv(fs, s, labels)


def _nested_cv(fs: FeatureSet, s: Settings, labels: np.ndarray | None) -> SubjectPredictions:
    y = fs.labels if labels is None else labels
    groups = fs.subjects
    base = make_pipeline(s.classifier, s.selector, s.seed)
    grid = param_grid(s.selector, fs.X.shape[1])
    epoch_p = np.full(len(y), np.nan)
    fold_params = []
    for train, test in outer_splitter(s.outer_folds, s.seed).split(fs.X, y, groups):
        check_no_subject_overlap(groups[train], groups[test])
        inner = StratifiedGroupKFold(s.inner_folds, shuffle=True, random_state=s.seed)
        model = clone(base)  # a fresh, unfitted model for each fold
        if grid:
            model = GridSearchCV(model, grid, cv=inner, scoring="roc_auc", refit=True)
            model.fit(fs.X[train], y[train], groups=groups[train])
            fold_params.append(model.best_params_)
        else:
            model.fit(fs.X[train], y[train])
        epoch_p[test] = model.predict_proba(fs.X[test])[:, 1]
    subjects = sorted(set(groups))
    p = np.array([epoch_p[groups == sid].mean() for sid in subjects])
    ys = np.array([y[groups == sid][0] for sid in subjects])
    return SubjectPredictions(subjects, ys, p, fold_params, float(roc_auc_score(y, epoch_p)))


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def ratio(a, b):
        return a / b if b else float("nan")

    return {
        "auroc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn),
        "mcc": float(matthews_corrcoef(y, pred)), "brier": float(brier_score_loss(y, p)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def fast_auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    from scipy.stats import rankdata

    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fast_balanced_accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    pred = p >= threshold
    return float(((pred[y == 1]).mean() + (~pred[y == 0]).mean()) / 2)


FAST = {"auroc": fast_auroc, "balanced_accuracy": fast_balanced_accuracy}


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric: str, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Stratified bootstrap over subjects (the class sizes stay fixed)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    fn = FAST.get(metric) or (lambda a, b: binary_metrics(a, b)[metric])
    vals = np.array([fn(y[idx], p[idx]) for idx in (
        np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n_boot))], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def permutation_test(fs: FeatureSet, s: Settings, observed_auroc: float, n_perm: int) -> dict:
    """Shuffle labels between subjects and repeat the nested loop. p = (1 + hits) / (1 + n_perm)."""
    table = subject_labels(fs)
    sids = sorted(table)
    rng = np.random.default_rng(s.seed + 1)
    null = []
    for _ in range(n_perm):
        shuffled = dict(zip(sids, rng.permutation([table[x] for x in sids])))
        y_perm = np.array([shuffled[x] for x in fs.subjects])
        sp = nested_cv(fs, s, labels=y_perm)
        null.append(fast_auroc(sp.y, sp.p))
    null = np.array(null)
    return {
        "n_permutations": n_perm,
        "p_value": float((1 + (null >= observed_auroc).sum()) / (1 + n_perm)),
        "null_auroc_mean": float(null.mean()) if n_perm else float("nan"),
        "null_auroc_95th": float(np.quantile(null, 0.95)) if n_perm else float("nan"),
    }


def leaky_epoch_cv(fs: FeatureSet, s: Settings, k: int = 32) -> float:
    """THE WRONG PROTOCOL, kept only to show the bias: selection on all epochs, then epoch-wise
    folds that put epochs of one subject in train and test. Returns the epoch accuracy."""
    with expected_warnings():
        X = SelectKBest(f_classif, k=min(k, fs.X.shape[1])).fit_transform(fs.X, fs.labels)
    model = make_pipeline(s.classifier, "none", s.seed)
    accs = []
    for train, test in StratifiedKFold(10, shuffle=True, random_state=s.seed).split(X, fs.labels):
        m = clone(model).fit(X[train], fs.labels[train])
        accs.append((m.predict(X[test]) == fs.labels[test]).mean())
    return float(np.mean(accs))


def evaluate(fs: FeatureSet, s: Settings, n_perm: int | None = None, n_boot: int = 2000) -> dict:
    sp = nested_cv(fs, s)
    m = binary_metrics(sp.y, sp.p)
    ci = {k: bootstrap_ci(sp.y, sp.p, k, n_boot, s.seed) for k in ("auroc", "balanced_accuracy")}
    out = {
        "protocol": "LOSO" if s.outer_folds == 0 else f"StratifiedGroupKFold({s.outer_folds}) x inner({s.inner_folds})",
        "classifier": s.classifier, "selector": s.selector, "features": s.features, "seed": s.seed,
        "n_subjects": len(sp.subjects), "n_responders": int(sp.y.sum()), "n_epochs": int(len(fs.labels)),
        "n_features": int(fs.X.shape[1]),
        "subject_metrics": m, "ci95": ci, "epoch_auroc": sp.epoch_auroc,
        "fold_params": sp.fold_params,
        "subjects": [{"id": a, "label": int(b), "p": round(float(c), 4)} for a, b, c in zip(sp.subjects, sp.y, sp.p)],
    }
    n_perm = s.permutations if n_perm is None else n_perm
    if n_perm > 0:
        out["permutation"] = permutation_test(fs, s, m["auroc"], n_perm)
    return out

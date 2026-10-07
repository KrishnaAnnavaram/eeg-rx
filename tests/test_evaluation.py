import json

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from eeg_rx import DISCLAIMER
from eeg_rx import evaluate as ev
from eeg_rx.config import Settings
from eeg_rx.report import markdown, write
from eeg_rx.model import make_pipeline, param_grid

S = Settings(outer_folds=4, inner_folds=3, seed=1)


def test_outer_folds_never_share_a_subject(fs):
    """Problem 3: no subject is in a training fold and a test fold."""
    for n in (0, 4):
        for train, test in ev.outer_splitter(n, 0).split(fs.X, fs.labels, fs.subjects):
            ev.check_no_subject_overlap(fs.subjects[train], fs.subjects[test])
    with pytest.raises(ev.LeakageError):
        ev.check_no_subject_overlap(["a", "b"], ["b"])


def test_signal_is_found_subject_wise(fs):
    sp = ev.nested_cv(fs, S)
    m = ev.binary_metrics(sp.y, sp.p)
    assert len(sp.subjects) == 16 and m["auroc"] > 0.7
    assert len(sp.fold_params) == 4 and all("select__k" in p for p in sp.fold_params)


def test_null_data_is_at_chance_subject_wise_but_not_epoch_wise(fs_null):
    """Problems 3 and 4: the old protocol finds 'signal' in data that has none."""
    leaky = ev.leaky_epoch_cv(fs_null, S)
    sp = ev.nested_cv(fs_null, S)
    assert leaky > 0.75
    assert 0.2 < ev.binary_metrics(sp.y, sp.p)["auroc"] < 0.8


def test_each_fold_fits_a_fresh_model(fs, monkeypatch):
    """Problem 5: no fold starts from weights that saw its test subjects."""
    from sklearn.linear_model import LogisticRegression

    seen = []

    class Spy(LogisticRegression):
        def fit(self, X, y, sample_weight=None):
            seen.append(id(self))
            return super().fit(X, y, sample_weight)

    monkeypatch.setattr("eeg_rx.model.make_classifier", lambda name, seed: Spy(max_iter=2000))
    ev.nested_cv(fs, Settings(outer_folds=4, inner_folds=2, selector="none"))
    assert len(seen) == 4 and len(set(seen)) == 4


def test_feature_ranking_is_stable_across_k(fs):
    """Problem 6: the top 8 features are inside the top 16. The ranking is not re-permuted."""
    from sklearn.feature_selection import SelectKBest, f_classif

    s8 = set(np.flatnonzero(SelectKBest(f_classif, k=8).fit(fs.X, fs.labels).get_support()))
    s16 = set(np.flatnonzero(SelectKBest(f_classif, k=16).fit(fs.X, fs.labels).get_support()))
    assert s8 <= s16
    assert param_grid("anova", 50)["select__k"] == [8, 16, 32, "all"]
    assert param_grid("none", 50) == {}


@pytest.mark.parametrize("clf,sel", [("svm", "anova"), ("mlp", "l1"), ("logreg", "none")])
def test_other_classifiers_and_selectors_run(fs, clf, sel):
    sp = ev.nested_cv(fs, Settings(outer_folds=4, inner_folds=2, classifier=clf, selector=sel))
    assert np.isfinite(sp.p).all() and ((sp.p >= 0) & (sp.p <= 1)).all()
    with pytest.raises(ValueError):
        make_pipeline("forest")


def test_fast_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = np.array([0] * 10 + [1] * 7)
    p = np.round(rng.uniform(size=17), 1)  # ties on purpose
    assert ev.fast_auroc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert ev.fast_balanced_accuracy(y, p) == pytest.approx(balanced_accuracy_score(y, p >= 0.5))
    lo, hi = ev.bootstrap_ci(y, p, "auroc", n_boot=300)
    assert lo <= roc_auc_score(y, p) <= hi


def test_binary_metrics_known_confusion():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.6, 0.3, 0.4])
    m = ev.binary_metrics(y, p)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 1, 3, 1)
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["ppv"] == pytest.approx(2 / 3)


def test_permutation_test_and_report_numbers_are_unchanged(fs, tmp_path):
    """Problem 2: the report writes the computed numbers with no offset."""
    result = ev.evaluate(fs, S, n_perm=3, n_boot=200)
    p = result["permutation"]
    assert p["p_value"] in (0.25, 0.5, 0.75, 1.0) and p["n_permutations"] == 3
    j, md = write(result, tmp_path, synthetic=True)
    saved = json.loads(j.read_text())
    assert saved["subject_metrics"] == result["subject_metrics"]
    assert f"| AUROC | {result['subject_metrics']['auroc']:.3f} |" in md.read_text()
    assert DISCLAIMER in markdown(result, True) and "SYNTHETIC" in md.read_text()
    probs = [s["p"] for s in result["subjects"]]
    assert ev.fast_auroc(np.array([s["label"] for s in result["subjects"]]), np.array(probs)) == pytest.approx(
        result["subject_metrics"]["auroc"], abs=0.02)

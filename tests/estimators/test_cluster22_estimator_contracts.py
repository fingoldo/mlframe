"""Cluster 22 regressions: mixed-type label sets, name-based decorrelation, and the weighted-scorer retry (audit 2026-11-20, findings 22-F19, 22-F21, 22-F22).

np.unique sorts, so an object label array holding None next to strings raised before the sorted-key fallback could run. The decorrelator fixed its dropped
columns as positions, so a reordered frame lost the wrong ones. The dummy baseline re-ran any TypeError from the scorer without weights.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

from mlframe.estimators.baselines import get_best_dummy_score
from mlframe.estimators.custom import IdentityEstimator, MyDecorrelator


class _IdentityClassifier(ClassifierMixin, IdentityEstimator):
    """The identity estimator acting as a classifier, so ``fit`` records ``classes_``."""


def test_object_labels_with_none_and_strings_fit_without_sorting_error():
    """A mixed None/str label array gives classes_ in the str-key order, with None last, instead of a TypeError (22-F19)."""
    y = np.array(["b", None, "a", "b"], dtype=object)
    est = _IdentityClassifier().fit(np.zeros((4, 2)), y)
    assert list(est.classes_) == ["a", "b", None]


def test_decorrelator_drops_by_name_when_the_frame_is_reordered():
    """Fit on [a, b, c] with b a copy of a: a reordered frame [b, a, c] still loses b, not whatever sits at position 1 (22-F21)."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    train = pd.DataFrame({"a": a, "b": a, "c": rng.normal(size=100)})
    dec = MyDecorrelator(threshold=0.9).fit(train)
    out = dec.transform(train[["b", "a", "c"]])
    assert list(out.columns) == ["a", "c"]


def _data():
    """A small separable classification problem."""
    X = pd.DataFrame({"x": np.arange(20.0)})
    y = (np.arange(20) >= 10).astype(int)
    return X, y


def test_scorer_without_sample_weight_is_retried_unweighted():
    """A scorer that does not accept sample_weight still scores, unweighted, as before (22-F22)."""
    X, y = _data()

    def scorer(model, Xt, yt):
        """Accept no sample_weight."""
        return 0.5

    out = get_best_dummy_score(LogisticRegression(), X, y, X, y, scorer, test_sample_weight=np.ones(20))
    assert out == 0.5


def test_other_type_errors_from_the_scorer_are_not_swallowed():
    """A TypeError unrelated to sample_weight propagates instead of being re-run unweighted (22-F22)."""
    X, y = _data()

    def scorer(model, Xt, yt, sample_weight=None):
        """Fail for a reason that has nothing to do with the weights."""
        raise TypeError("unsupported operand in the metric")

    with pytest.raises(TypeError, match="unsupported operand"):
        get_best_dummy_score(LogisticRegression(), X, y, X, y, scorer, test_sample_weight=np.ones(20))

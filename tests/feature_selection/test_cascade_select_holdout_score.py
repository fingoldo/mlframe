"""cascade_select reports an honest holdout score when given untouched rows, and nothing pretending to be one otherwise."""

import numpy as np
import pandas as pd
import pytest

import importlib

cs = importlib.import_module("mlframe.feature_selection.cascade_select")  # the package re-exports a function of the same name


class _Model:
    def fit(self, X, y):
        self.cols_ = list(X.columns)
        return self

    def score(self, X, y):
        return 0.5


def test_holdout_score_uses_the_final_subset_on_the_caller_rows():
    X = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    score = cs._holdout_score(_Model, X, np.zeros(10), ["a"], X.iloc[:4], np.zeros(4), None)
    assert score == pytest.approx(0.5)


def test_without_a_holdout_there_is_no_score():
    X = pd.DataFrame({"a": np.arange(10.0)})
    assert cs._holdout_score(_Model, X, np.zeros(10), ["a"], None, None, None) is None

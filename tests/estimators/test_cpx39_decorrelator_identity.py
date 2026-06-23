"""CPX39: MyDecorrelator.fit vectorized np.triu(k=1) must drop the IDENTICAL set as the prior double-loop.

The drop rule is: a column is dropped when its absolute correlation with any EARLIER column exceeds the
threshold (the LATER column of each correlated pair). These tests pin that exact set + the transform output.
"""

import numpy as np, pandas as pd, pytest

from mlframe.estimators.custom import MyDecorrelator


def _reference_double_loop(X: pd.DataFrame, threshold: float) -> set:
    """The pre-CPX39 O(p^2) drop rule, kept here as the identity oracle."""
    correlated_features = set()
    corr_matrix = pd.DataFrame(X).corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                correlated_features.add(corr_matrix.columns[i])
    return correlated_features


def _make(n=2000, p=60, n_corr=15, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    for _ in range(n_corr):
        i = int(rng.integers(0, p // 2))
        j = int(rng.integers(p // 2, p))
        X[:, j] = X[:, i] + 0.03 * rng.standard_normal(n)
    return pd.DataFrame(X)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("threshold", [0.8, 0.9, 0.95])
def test_dropped_set_identical_to_double_loop(seed, threshold):
    X = _make(seed=seed)
    dec = MyDecorrelator(threshold=threshold).fit(X)
    assert dec.correlated_features_ == _reference_double_loop(X, threshold)


def test_drops_later_column_of_known_pair():
    # Column 3 is an exact copy of column 1; the LATER index (3) must be dropped, not 1.
    rng = np.random.default_rng(7)
    base = rng.standard_normal((500, 5))
    base[:, 3] = base[:, 1]
    X = pd.DataFrame(base)
    dec = MyDecorrelator(threshold=0.95).fit(X)
    assert dec.correlated_features_ == {3}
    assert list(dec.transform(X).columns) == [0, 1, 2, 4]


def test_zero_variance_columns_not_dropped():
    # corr() yields NaN for constant columns; abs(NaN) > thr is False -> never dropped (both paths agree).
    rng = np.random.default_rng(11)
    base = rng.standard_normal((300, 4))
    base[:, 2] = 5.0  # constant
    X = pd.DataFrame(base)
    dec = MyDecorrelator(threshold=0.9).fit(X)
    assert dec.correlated_features_ == _reference_double_loop(X, 0.9)


def test_transform_output_matches_drop_set():
    X = _make(seed=3)
    dec = MyDecorrelator(threshold=0.9).fit(X)
    out = dec.transform(X)
    expected_keep = [c for c in X.columns if c not in dec.correlated_features_]
    assert list(out.columns) == expected_keep
    pd.testing.assert_frame_equal(out, X[expected_keep])

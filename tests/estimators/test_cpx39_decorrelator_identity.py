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
                # POSITIONS, not labels. Production canonicalises ``correlated_features_`` to integer
                # positions so a fit on a DataFrame and a transform on an ndarray agree; emitting
                # ``corr_matrix.columns[i]`` here matched only because ``_make`` returns a default
                # RangeIndex, where labels happen to equal positions. On a named-column frame the two
                # disagreed outright ({3} vs {'d'}), so this test could not be extended to named columns
                # and did not guard the canonicalisation at all.
                correlated_features.add(i)
    return correlated_features


def _make(n=2000, p=60, n_corr=15, seed=0):
    """Builds seeded synthetic test data; returns ``pd.DataFrame(X)``."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    for _ in range(n_corr):
        i = int(rng.integers(0, p // 2))
        j = int(rng.integers(p // 2, p))
        X[:, j] = X[:, i] + 0.03 * rng.standard_normal(n)
    return pd.DataFrame(X)


def _make_named(n=2000, p=60, n_corr=15, seed=0):
    """The same frame with STRING column labels, where labels can no longer stand in for positions."""
    df = _make(n=n, p=p, n_corr=n_corr, seed=seed)
    df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("threshold", [0.8, 0.9, 0.95])
def test_dropped_set_identical_to_double_loop(seed, threshold):
    """Dropped set identical to double loop."""
    X = _make(seed=seed)
    dec = MyDecorrelator(threshold=threshold).fit(X)
    assert dec.correlated_features_ == _reference_double_loop(X, threshold)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("threshold", [0.8, 0.95])
def test_dropped_set_is_positions_on_a_named_column_frame(seed, threshold):
    """Named columns, where a label-emitting reference would disagree with production outright.

    This is the case the RangeIndex fixture could not distinguish: ``correlated_features_`` must be integer
    positions whatever the frame's labels are, so that a fit on a DataFrame still applies to a transform on
    an ndarray.
    """
    X = _make_named(seed=seed)
    dec = MyDecorrelator(threshold=threshold).fit(X)
    dropped = dec.correlated_features_
    assert dropped == _reference_double_loop(X, threshold)
    assert all(isinstance(c, (int, np.integer)) for c in dropped), f"correlated_features_ carries labels rather than positions: {sorted(dropped)[:5]}"


def test_a_fit_on_named_columns_transforms_a_bare_ndarray():
    """The reason positions are canonical: the fitted set has to survive losing the labels."""
    X = _make_named(n=500, p=12, n_corr=3, seed=1)
    dec = MyDecorrelator(threshold=0.95).fit(X)
    kept_from_frame = dec.transform(X).shape[1]
    kept_from_array = dec.transform(X.to_numpy()).shape[1]
    assert kept_from_frame == kept_from_array, (
        f"transform kept {kept_from_frame} columns from the labelled frame but {kept_from_array} from the "
        "same data as a bare ndarray; the fitted drop set is not position-based"
    )


def test_drops_later_column_of_known_pair():
    # Column 3 is an exact copy of column 1; the LATER index (3) must be dropped, not 1.
    """Drops later column of known pair."""
    rng = np.random.default_rng(7)
    base = rng.standard_normal((500, 5))
    base[:, 3] = base[:, 1]
    X = pd.DataFrame(base)
    dec = MyDecorrelator(threshold=0.95).fit(X)
    assert dec.correlated_features_ == {3}
    assert list(dec.transform(X).columns) == [0, 1, 2, 4]


def test_zero_variance_columns_not_dropped():
    # corr() yields NaN for constant columns; abs(NaN) > thr is False -> never dropped (both paths agree).
    """Zero variance columns not dropped."""
    rng = np.random.default_rng(11)
    base = rng.standard_normal((300, 4))
    base[:, 2] = 5.0  # constant
    X = pd.DataFrame(base)
    dec = MyDecorrelator(threshold=0.9).fit(X)
    assert dec.correlated_features_ == _reference_double_loop(X, 0.9)


def test_transform_output_matches_drop_set():
    """Transform output matches drop set."""
    X = _make(seed=3)
    dec = MyDecorrelator(threshold=0.9).fit(X)
    out = dec.transform(X)
    expected_keep = [c for c in X.columns if c not in dec.correlated_features_]
    assert list(out.columns) == expected_keep
    pd.testing.assert_frame_equal(out, X[expected_keep])

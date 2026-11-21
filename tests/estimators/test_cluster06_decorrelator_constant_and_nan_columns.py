"""Audit 2026-11-20 finding 06-F5: the decorrelator's dropped set on constant and all-NaN columns matches the loop.

A constant column and a column with NaNs both have NaN correlation with everything, so neither can be dropped by the
threshold rule. The vectorised production fit must agree with the double-loop reference on exactly those inputs, and must
not drop a column because its correlation is undefined.
"""

import numpy as np
import pandas as pd

from mlframe.estimators.custom import MyDecorrelator


def _loop_reference(X: pd.DataFrame, threshold: float) -> set:
    """Positions flagged by the pre-vectorisation rule: a later column correlating above threshold with an earlier one."""
    corr = pd.DataFrame(X).corr()
    flagged = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                flagged.add(i)
    return flagged


def _frame():
    """Random columns, a near-copy pair, a constant column and a column with NaN where the copy would have been."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((500, 6))
    X[:, 4] = X[:, 1] + 0.01 * rng.standard_normal(500)
    X[:, 2] = 7.0
    X[::3, 5] = np.nan
    return pd.DataFrame(X)


def test_constant_and_nan_columns_follow_the_loop_rule():
    """The production dropped set equals the loop's, including the undefined-correlation columns."""
    X = _frame()
    dec = MyDecorrelator(threshold=0.9).fit(X)
    assert dec.correlated_features_ == _loop_reference(X, 0.9)


def test_undefined_correlation_is_never_a_drop():
    """A constant column (position 2) and a NaN column (position 5) are kept, since NaN > threshold is False."""
    dec = MyDecorrelator(threshold=0.9).fit(_frame())
    assert 2 not in dec.correlated_features_
    assert 5 not in dec.correlated_features_
    assert 4 in dec.correlated_features_

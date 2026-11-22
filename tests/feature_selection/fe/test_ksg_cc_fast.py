"""The continuous-target Kraskov kernel reproduces sklearn's ``mutual_info_regression`` and is what ``_ksg_mi_batch`` calls for a continuous target."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._ksg_cc_fast import ksg_mi_continuous_target


def _frame(n, p, seed):
    """Columns with a nonlinear dependence on the target, a rounded column (ties) and noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 0] * X[:, 1] + 0.5 * X[:, 2] + 0.3 * rng.normal(size=n)
    X[:, 3] = np.round(X[:, 3], 1)
    return X, y


@pytest.mark.parametrize("n,p,k", [(120, 6, 3), (600, 8, 3), (2000, 5, 5)])
def test_matches_sklearn_mutual_info_regression(n, p, k):
    """The same MI per column as sklearn, to rounding."""
    from sklearn.feature_selection import mutual_info_regression

    X, y = _frame(n, p, seed=n)
    expected = mutual_info_regression(X, y, discrete_features=False, n_neighbors=k, random_state=7)
    got = ksg_mi_continuous_target(X, y, n_neighbors=k, random_state=7)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)


def test_orthogonal_batch_routes_a_continuous_target_to_the_kernel(monkeypatch):
    """``_ksg_mi_batch`` with a continuous target returns the kernel's values."""
    from mlframe.feature_selection.filters import _orthogonal_ksg_mi_fe as mod

    X, y = _frame(300, 4, seed=3)
    calls = []
    real = ksg_mi_continuous_target

    def spy(*a, **k):
        """Record the call and delegate."""
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr("mlframe.feature_selection.filters._ksg_cc_fast.ksg_mi_continuous_target", spy)
    out = mod._ksg_mi_batch(X, y, n_neighbors=3, random_state=0)
    assert calls and out.shape == (4,)
    assert out[1] >= 0.0 and np.isfinite(out).all()


def test_degenerate_inputs_do_not_crash():
    """Fewer rows than neighbours and a constant column give zeros, not errors."""
    X = np.zeros((3, 2))
    assert ksg_mi_continuous_target(X, np.arange(3.0), n_neighbors=3, random_state=0).tolist() == [0.0, 0.0]
    X2 = np.column_stack([np.ones(50), np.arange(50.0)])
    out = ksg_mi_continuous_target(X2, np.arange(50.0), n_neighbors=3, random_state=0)
    assert out[0] < out[1]


def test_columns_call_equals_one_call_per_column():
    """The all-columns call is bit-identical to calling the one-column kernel on each column (every column draws the same noise)."""
    from mlframe.feature_selection.filters._ksg_cc_fast import ksg_mi_continuous_target_columns

    X, y = _frame(500, 7, seed=11)
    together = ksg_mi_continuous_target_columns(X, y, n_neighbors=3, random_state=5)
    one_by_one = np.array([ksg_mi_continuous_target(X[:, [j]], y, n_neighbors=3, random_state=5)[0] for j in range(X.shape[1])])
    np.testing.assert_array_equal(together, one_by_one)

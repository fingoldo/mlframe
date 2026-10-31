"""The batched tail-dependence kernel equals the scalar scorer (max of the two floored tails) column by column."""

import numpy as np

from mlframe.feature_selection.filters._orth_auto_scorer_fe import _score_tail_dep
from mlframe.feature_selection.filters._orthogonal_tail_dependence_fe import tail_dependence_columns


def test_kernel_equals_scalar_scorer_with_real_tail_dependence():
    """Heavy-tailed columns coupled to the target in the upper tail, a negatively coupled one, rounded (ties) and constant columns."""
    rng = np.random.default_rng(0)
    n, k = 5000, 10
    X = rng.standard_t(3, size=(n, k))
    X[:, 1] = np.round(X[:, 1], 1)
    X[:, 2] = 1.0
    y = (X[:, 0] > np.quantile(X[:, 0], 0.93)).astype(np.int64)
    X[:, 3] = 0.7 * X[:, 0] + rng.standard_t(3, size=n)
    X[:, 4] = -X[:, 0] + 0.3 * rng.standard_t(3, size=n)
    got = tail_dependence_columns(np.ascontiguousarray(X.T), y)
    ref = np.array([_score_tail_dep(X[:, j], y) for j in range(k)])
    np.testing.assert_allclose(got, ref, atol=1e-12)
    assert got.max() > 0.3

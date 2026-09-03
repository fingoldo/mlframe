"""The batched AUC assigned ranks straight off `argsort`, with no tie correction.

Within any tied block the rank a row received was decided by its position in the input, not by its value. On a
binary indicator, a count column with a large zero mass, or any low-cardinality feature, the Mann-Whitney sum
over the positives then landed on either side of 0.5 essentially at random -- and `align_feature_direction`
turns that into a sign flip (`sign = -1 if auc < 0.5 else 1`), while `check_feature_direction_stability` reports
the resulting per-fold churn as if it were sampling noise. A genuinely constant column has a true AUC of exactly
0.5 and got whatever the positive rows' positions dictated.

Second, independent defect on the same line: a NaN sorts last under `argsort` and so takes the maximal rank, so
a column whose missingness correlates with the positive class scored a spuriously inflated AUC.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from mlframe.preprocessing.align_feature_direction import _midranks, batch_univariate_auc

N = 4000


@pytest.fixture
def tied():
    """Columns at several cardinalities, plus a constant one and a continuous control."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, N)
    cols = [rng.integers(0, k, N).astype(np.float64) for k in (2, 3, 5, 50)]
    return np.column_stack([*cols, np.ones(N), rng.normal(0, 1, N)]), y


class TestItAgreesWithRocAucScore:
    """`roc_auc_score` uses mid-ranks; the batched form claimed to match it "closely enough"."""

    def test_every_column_matches_sklearn(self, tied):
        """The whole point of the function is to be a faster `roc_auc_score`, not a different statistic."""
        X, y = tied
        ref = np.array([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])])
        assert np.abs(batch_univariate_auc(X, y) - ref).max() < 1e-12

    def test_a_constant_column_scores_exactly_half(self, tied):
        """The sharpest case: no ordering information exists, so the answer cannot depend on row positions."""
        X, y = tied
        assert batch_univariate_auc(X, y)[4] == 0.5

    def test_the_answer_does_not_depend_on_row_order(self, tied):
        """Tie-broken-by-position means shuffling the rows moves the AUC."""
        X, y = tied
        rng = np.random.default_rng(1)
        perm = rng.permutation(N)
        assert np.allclose(batch_univariate_auc(X, y), batch_univariate_auc(X[perm], y[perm]))

    def test_a_binary_indicator_is_not_pushed_across_the_flip_threshold(self):
        """The consequence the module cares about: `sign = -1 if auc < 0.5`."""
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, N)
        x = rng.integers(0, 2, N).astype(np.float64)[:, None]  # independent of y, true AUC ~= 0.5
        assert abs(float(batch_univariate_auc(x, y)[0]) - roc_auc_score(y, x[:, 0])) < 1e-12

    def test_the_ranks_are_scipys_average_ranks(self):
        """Pinned against the reference implementation, not just against the AUC it feeds."""
        rng = np.random.default_rng(3)
        X = rng.integers(0, 8, (2000, 20)).astype(np.float64)
        assert np.array_equal(_midranks(X), stats.rankdata(X, axis=0))


class TestNonFiniteInputIsRefused:
    """A NaN took the maximal rank and inflated the column it appeared in."""

    def test_a_nan_raises(self, tied):
        """Refusing matches the single-class guard this function already chose over a silent wrong answer."""
        X, y = tied
        X = X.copy()
        X[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            batch_univariate_auc(X, y)

    def test_an_infinity_raises(self, tied):
        """`np.isfinite`, not `np.isnan`: an inf sorts last too."""
        X, y = tied
        X = X.copy()
        X[3, 1] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            batch_univariate_auc(X, y)

    def test_clean_input_still_passes(self, tied):
        """The guard must not reject ordinary data."""
        X, y = tied
        assert np.isfinite(batch_univariate_auc(X, y)).all()


class TestTheExistingGuardsSurvive:
    """Neither fix may weaken what was already correct."""

    def test_single_class_y_still_raises(self, tied):
        """Pre-existing guard."""
        X, _ = tied
        with pytest.raises(ValueError, match="single-class"):
            batch_univariate_auc(X, np.ones(N, dtype=int))

    def test_a_perfectly_separating_column_scores_one(self):
        """Sanity at the extreme."""
        y = np.array([0] * 50 + [1] * 50)
        x = np.arange(100, dtype=np.float64)[:, None]
        assert batch_univariate_auc(x, y)[0] == 1.0

    def test_a_perfectly_inverted_column_scores_zero(self):
        """The case the sign flip exists for."""
        y = np.array([0] * 50 + [1] * 50)
        x = np.arange(100, 0, -1, dtype=np.float64)[:, None]
        assert batch_univariate_auc(x, y)[0] == 0.0

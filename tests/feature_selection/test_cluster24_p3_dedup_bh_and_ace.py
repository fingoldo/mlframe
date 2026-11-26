"""Cluster 24 P3 regressions: one Benjamini-Hochberg implementation, and one fit-and-split block in ACE (audit 2026-11-20, findings 24-F60 and 24-F65).

general.py and ace.py each carried the same BH step-up code; both now call the shared ``_fdr_bh`` function, and these controls pin that all three agree
on random p-values, ties and the empty input. The held-out and in-sample branches of ACE's per-replicate importance share one helper, which still
refuses a joint fit whose importance vector is not twice the real width.
"""

import numpy as np
import pytest

from mlframe.feature_selection import ace, general
from mlframe.feature_selection._fdr_bh import benjamini_hochberg_reject


def _reference(p_values, alpha):
    """The textbook BH step-up, written independently of the shared code."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    out = np.zeros(m, dtype=bool)
    if m == 0:
        return out
    order = np.argsort(p)
    passing = [i for i, idx in enumerate(order) if p[idx] <= (i + 1) / m * alpha]
    if passing:
        out[order[: max(passing) + 1]] = True
    return out


@pytest.mark.parametrize("seed", range(5))
def test_all_three_implementations_agree_on_random_p_values(seed):
    """The shared function, general's wrapper and ACE's wrapper return the textbook mask on random p-values (24-F60)."""
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.uniform(0, 0.02, 15), rng.uniform(0, 1, 60)])
    expected = _reference(p, 0.05)
    np.testing.assert_array_equal(benjamini_hochberg_reject(p, 0.05), expected)
    np.testing.assert_array_equal(general._benjamini_hochberg_reject(p, 0.05), expected)
    np.testing.assert_array_equal(ace._benjamini_hochberg_reject(p, 0.05), expected)


def test_ties_and_empty_input_agree():
    """Tied p-values and an empty vector behave the same in every wrapper (24-F60)."""
    tied = np.array([0.01, 0.01, 0.01, 0.4, 0.9])
    for fn in (benjamini_hochberg_reject, general._benjamini_hochberg_reject, ace._benjamini_hochberg_reject):
        np.testing.assert_array_equal(fn(tied, 0.05), _reference(tied, 0.05))
        assert fn(np.array([]), 0.05).shape == (0,)


class _StubModel:
    """A fitted-looking model with a fixed native importance vector."""

    def __init__(self, width):
        self.feature_importances_ = np.arange(1.0, width + 1.0)

    def fit(self, X, y):
        """Accept the fit rows and return self, like an estimator."""
        self.n_fit_ = len(X)
        return self


def test_fit_and_split_returns_real_and_contrast_halves():
    """The shared block fits on the fit rows and splits the importances into two halves of p columns (24-F65)."""
    p = 3
    model = _StubModel(2 * p)
    X = np.zeros((10, 2 * p))
    y = np.zeros(10)
    real, contrast = ace._fit_and_split_importances(model, X[:7], y[:7], X[7:], y[7:], "native", 2, 0, p)
    assert model.n_fit_ == 7
    np.testing.assert_array_equal(real, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(contrast, [4.0, 5.0, 6.0])


def test_fit_and_split_refuses_a_wrong_width_importance_vector():
    """A joint fit whose importance vector is not 2p wide raises, as both old branches did (24-F65)."""
    model = _StubModel(5)
    with pytest.raises(ValueError, match="expected 6 importances"):
        ace._fit_and_split_importances(model, np.zeros((4, 6)), np.zeros(4), np.zeros((4, 6)), np.zeros(4), "native", 2, 0, 3)

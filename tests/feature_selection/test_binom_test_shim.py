"""Tests for ``mlframe.feature_selection.boruta_shap._binom_test_shim.binom_test`` -- a forward-compat shim
matching the removed ``scipy.stats.binom_test`` signature/return (bare p-value) on top of SciPy 1.7+'s
``binomtest`` (a result object). Previously had zero test coverage despite being consolidated specifically
to prevent two independent copies of the same shim from silently drifting apart.
"""

from __future__ import annotations

import pytest
from scipy.stats import binomtest

from mlframe.feature_selection.boruta_shap._binom_test_shim import binom_test


class TestMatchesScipyBinomtest:
    """Groups tests pinning the shim's return value against scipy's own binomtest.pvalue."""

    @pytest.mark.parametrize(
        "x,n,p",
        [
            (5, 10, 0.5),
            (0, 10, 0.5),
            (10, 10, 0.5),
            (3, 20, 0.1),
            (17, 20, 0.9),
        ],
    )
    def test_two_sided_matches_binomtest_pvalue(self, x, n, p):
        """Two sided matches binomtest pvalue."""
        got = binom_test(x, n, p)
        expected = binomtest(x, n=n, p=p, alternative="two-sided").pvalue
        assert got == expected

    @pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
    def test_alternative_argument_is_forwarded(self, alternative):
        """Every ``alternative`` value must be forwarded through to scipy's binomtest unchanged."""
        got = binom_test(7, 10, 0.5, alternative=alternative)
        expected = binomtest(7, n=10, p=0.5, alternative=alternative).pvalue
        assert got == expected

    def test_returns_a_bare_float_not_a_result_object(self):
        """The whole point of the shim: callers written against the old scipy.stats.binom_test API expect
        a bare p-value, not scipy 1.7+'s BinomTestResult object."""
        got = binom_test(5, 10, 0.5)
        assert isinstance(got, float)


class TestBoundaryAndFloatInputs:
    """Groups tests covering edge cases specific to this shim's coercion behaviour."""

    def test_exact_match_p_is_one(self):
        """When x exactly matches the expected mean under H0, the two-sided p-value is 1.0."""
        assert binom_test(5, 10, 0.5) == 1.0

    def test_extreme_count_gives_small_p_value(self):
        """An extreme outcome under H0 should yield a small p-value."""
        assert binom_test(10, 10, 0.5) < 0.01

    def test_float_hit_count_is_coerced_to_int(self):
        """boruta_shap's caller passes a float hit-count vector (np.zeros-derived); the shim must coerce
        via int(x) rather than raising scipy's own "k must be an integer" error."""
        got = binom_test(5.0, 10, 0.5)
        expected = binomtest(5, n=10, p=0.5).pvalue
        assert got == expected

    def test_zero_hits(self):
        """Zero hits against a non-trivial p should be a low p-value (extreme under H0)."""
        got = binom_test(0, 20, 0.5)
        assert got < 0.001

"""Regression tests for DCD's pairwise_cache_max auto-scaling with column count (2026-07-09 fix).

Before this fix, ``pairwise_cache_max`` defaulted to a fixed 50_000 regardless of column count. At
p=423, C(423,2)=89_253 already exceeds that cap, so a long-running DCD pass over this many columns
could evict and later recompute previously-cached pairwise SU values (LRU thrash). ``make_dcd_state``
now RAISES (never lowers) the cap to ``min(C(p,2), _DCD_PAIRWISE_CACHE_HARD_CAP)`` when the configured
value would be smaller than that.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
    make_dcd_state,
    _DCD_PAIRWISE_CACHE_HARD_CAP,
)


def _fd_fn(n_cols, n_rows=200, seed=0):
    """Fd fn."""
    rng = np.random.default_rng(seed)
    fd = rng.integers(0, 4, size=(n_rows, n_cols)).astype(np.int32)
    fn = np.array([4] * n_cols, dtype=np.int64)
    return fd, fn


def test_default_cap_raised_above_C_n_2_for_wide_pool():
    """At p=423 (C(423,2)=89_253 > the fixed 50_000 default), the cap must be raised to cover all pairs."""
    n_cols = 423
    fd, fn = _fd_fn(n_cols)
    st = make_dcd_state(
        X_raw=None,
        factors_data=fd,
        factors_nbins=fn,
        cols=[f"c{i}" for i in range(n_cols)],
        nbins=fn,
        target_indices=None,
    )
    expected_pairs = n_cols * (n_cols - 1) // 2
    assert expected_pairs == 89_253
    assert st.pairwise_cache_max >= expected_pairs


def test_narrow_pool_keeps_the_configured_default():
    """At small p (C(p,2) well under 50_000), the cap must NOT be needlessly inflated."""
    n_cols = 50  # C(50,2) = 1225, far under the 50_000 default
    fd, fn = _fd_fn(n_cols)
    st = make_dcd_state(
        X_raw=None,
        factors_data=fd,
        factors_nbins=fn,
        cols=[f"c{i}" for i in range(n_cols)],
        nbins=fn,
        target_indices=None,
    )
    assert st.pairwise_cache_max == 50_000  # unchanged from the class default


def test_explicit_user_override_above_formula_is_never_lowered():
    """Explicit user override above formula is never lowered."""
    n_cols = 100  # C(100,2) = 4950
    fd, fn = _fd_fn(n_cols)
    st = make_dcd_state(
        X_raw=None,
        factors_data=fd,
        factors_nbins=fn,
        cols=[f"c{i}" for i in range(n_cols)],
        nbins=fn,
        target_indices=None,
        pairwise_cache_max=999_999,  # user explicitly set something huge
    )
    assert st.pairwise_cache_max == 999_999  # never lowered toward the small formula value


def test_extreme_p_clamped_to_hard_cap():
    """At very large p, the formula must clamp to the hard ceiling, not grow unbounded toward C(p,2)."""
    n_cols = 5000  # C(5000,2) = 12_497_500, well above the hard cap
    fd, fn = _fd_fn(n_cols, n_rows=50)
    st = make_dcd_state(
        X_raw=None,
        factors_data=fd,
        factors_nbins=fn,
        cols=[f"c{i}" for i in range(n_cols)],
        nbins=fn,
        target_indices=None,
    )
    assert st.pairwise_cache_max == _DCD_PAIRWISE_CACHE_HARD_CAP


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])

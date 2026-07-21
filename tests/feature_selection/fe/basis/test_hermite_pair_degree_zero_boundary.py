"""mrmr_audit_2026-07-20 edge_cases.md #211: degree=0 for a polynomial/orthogonal-degree family.
``optimise_hermite_pair(min_degree=0, max_degree=0)`` degenerates the polynomial basis to its
degree-0 (constant) term -- a genuinely meaningless polynomial basis. Confirmed empirically before
writing this test: the function does NOT crash and does NOT return a degenerate constant-column
result that would silently pass MI(constant;y)==0 through the relevance gate; it returns None
(the documented 'no admissible engineered form' outcome), exactly like any other regime where no
candidate clears the baseline-uplift threshold."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._hermite_fe_optimise_pair import optimise_hermite_pair


def test_degree_zero_does_not_crash_and_returns_none():
    """A degree-0-only search space must degrade to None, not raise or return a bogus result."""
    rng = np.random.default_rng(0)
    n = 500
    x_a = rng.standard_normal(n)
    x_b = rng.standard_normal(n)
    y = (x_a * x_b > 0).astype(np.int64)
    result = optimise_hermite_pair(x_a, x_b, y, max_degree=0, min_degree=0, n_trials=20, seed=1)
    assert result is None, "degree=0 (constant-only basis) must never be admitted as a usable engineered form"

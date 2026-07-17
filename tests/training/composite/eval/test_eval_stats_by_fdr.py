"""Regression tests for the Benjamini-Yekutieli arbitrary-dependence FDR control.

The candidate bootstrap p-values are correlated (shared base columns + resample
structure), so the family control must use BY, not BH. BY applies a harmonic
``c(m)`` penalty making its rejection threshold strictly stricter than BH for
m >= 2 -- that stricter behaviour is what these tests pin.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery._eval_stats import (
    benjamini_hochberg_reject,
    benjamini_yekutieli_reject,
    apply_fdr_control_to_candidates,
)


def test_by_threshold_strictly_stricter_than_bh():
    # p-values where BH rejects more than BY at the same alpha (the c(m) penalty bites).
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.20, 0.40, 0.80])
    alpha = 0.10
    bh = benjamini_hochberg_reject(p, alpha)
    by = benjamini_yekutieli_reject(p, alpha)
    # BY can never reject more than BH (stricter threshold); here it rejects strictly fewer.
    assert by.sum() <= bh.sum()
    assert by.sum() < bh.sum(), "this fixture is chosen so BY is strictly stricter than BH"
    # Anything BY rejects, BH must also reject (monotone in the threshold).
    assert np.all(bh[by])


def test_by_harmonic_penalty_value():
    # With m=4 finite p-values, c(m) = 1 + 1/2 + 1/3 + 1/4 ~= 2.0833.
    # rank-1 BH threshold = alpha/m = 0.025; rank-1 BY threshold = alpha/(m*c(m)) ~= 0.012.
    # A p of 0.02 sits BETWEEN them: BH rejects it, BY (stricter under dependence) does NOT.
    p = np.array([0.02, 0.5, 0.6, 0.7])
    alpha = 0.10
    c_m = 1.0 + 0.5 + 1.0 / 3 + 0.25
    bh_thresh = alpha / 4.0
    by_thresh = alpha / (4.0 * c_m)
    assert by_thresh < 0.02 < bh_thresh
    by = benjamini_yekutieli_reject(p, alpha)
    bh = benjamini_hochberg_reject(p, alpha)
    assert bh[0]  # 0.02 < 0.025 -> rejected by BH
    assert not by[0]  # 0.02 > 0.012 -> NOT rejected by BY (the dependence penalty bites)
    assert by.sum() < bh.sum()


def test_apply_fdr_control_uses_by_and_stamps_reason():
    # Build candidate entries; with correlated noise-level p-values BY drops the marginal ones.
    cands = [
        {"spec": object(), "bootstrap_p_value": 0.01},
        {"spec": object(), "bootstrap_p_value": 0.045},
        {"spec": object(), "bootstrap_p_value": 0.30},
        {"spec": object(), "bootstrap_p_value": 0.60},
        {"spec": object(), "bootstrap_p_value": float("nan")},  # no bootstrap -> ignored
    ]
    n_dropped = apply_fdr_control_to_candidates(cands, alpha=0.10)
    assert n_dropped >= 1
    dropped = [c for c in cands if c.get("fdr_dropped")]
    assert dropped, "at least one marginal spec must be dropped"
    for c in dropped:
        assert "BY-FDR" in c["reason"], "reason must name BY (arbitrary-dependence FDR), not BH"
    # The NaN-p spec is never scored/dropped.
    assert "fdr_dropped" not in cands[-1]


def test_empty_and_all_nan_inputs():
    assert benjamini_yekutieli_reject(np.array([]), 0.05).size == 0
    out = benjamini_yekutieli_reject(np.array([np.nan, np.nan]), 0.05)
    assert out.dtype == bool and not out.any()

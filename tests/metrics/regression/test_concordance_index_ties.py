"""Regression tests for fast_concordance_index under ties in y_pred.

Pre-fix, fast_concordance_index returned (Kendall tau-b + 1) / 2, an identity that only holds
when y_pred has NO ties. With ties (routine for tree-ensemble risk scores with repeated leaf
outputs), that formula is measurably wrong because tau-b's denominator is the geometric mean of
pairs-not-tied-in-y_true and pairs-not-tied-in-y_pred, while the true (Harrell) C-index
denominator is pairs-not-tied-in-y_true alone.
"""

from __future__ import annotations

import numpy as np

from mlframe.metrics.regression._regression_corr import fast_concordance_index, fast_kendall_tau


def _brute_force_cindex(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """O(N^2) reference implementation of Harrell's C-index, ties in y_pred included."""
    n = len(y_true)
    concordant = discordant = tied_y = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            dx = y_true[i] - y_true[j]
            dy = y_pred[i] - y_pred[j]
            if dy == 0:
                tied_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant + tied_y
    return (concordant + 0.5 * tied_y) / comparable


def test_concordance_index_matches_brute_force_under_heavy_ties():
    """fast_concordance_index must match the O(N^2) brute-force reference under heavy ties."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(5, 60)
        y_true = rng.integers(0, 10, n).astype(np.float64)
        y_pred = rng.integers(0, 5, n).astype(np.float64)  # coarse -> heavy ties
        got = fast_concordance_index(y_true, y_pred)
        want = _brute_force_cindex(y_true, y_pred)
        assert abs(got - want) < 1e-9, f"n={n}: {got} != {want}"


def test_concordance_index_diverges_from_naive_tau_derivation_under_ties():
    """C-index must diverge from the buggy (tau_b + 1) / 2 identity once y_pred has ties."""
    # Manually constructed example (per the audit finding): predictions tied in groups, targets
    # distinct. The pre-fix (tau_b + 1) / 2 formula and the true C-index must disagree here.
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_pred = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])  # 3 tied pairs among 6 comparable-in-y_true rows

    c_index = fast_concordance_index(y_true, y_pred)
    tau = fast_kendall_tau(y_true, y_pred)
    naive = (tau + 1.0) / 2.0
    want = _brute_force_cindex(y_true, y_pred)

    assert abs(c_index - want) < 1e-9
    assert abs(c_index - naive) > 1e-3, "C-index must diverge from the buggy tau-derived identity under ties"


def test_concordance_index_tie_free_case_still_matches_tau_identity():
    """In the fully tie-free case, the (tau_b + 1) / 2 identity still holds to float precision."""
    # Sanity: in the fully tie-free case, the closed-form identity still holds (to float precision).
    rng = np.random.default_rng(1)
    n = 300
    y_true = rng.standard_normal(n)
    y_pred = y_true + 0.2 * rng.standard_normal(n)
    c_index = fast_concordance_index(y_true, y_pred)
    tau = fast_kendall_tau(y_true, y_pred)
    assert abs(c_index - (tau + 1.0) / 2.0) < 1e-9

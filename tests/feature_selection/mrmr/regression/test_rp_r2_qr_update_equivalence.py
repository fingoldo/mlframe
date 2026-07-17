"""Regression test for the QR-update optimization in MRMR's raw-feature floor-drop protection
(``_fit_impl_core.py``'s ``_rp_r2`` closure, 2026-07-10 perf fix).

The protection re-solves a held-out R^2 of ``[base | one candidate column]`` once per raw-feature
candidate, with ``base`` (the already-selected columns) FIXED across all candidates in one pass. At
production scale (p~120 base columns) cProfile attributed 36.2s of a 100k-row run's wall time to
``np.linalg.lstsq`` re-solving this from scratch per candidate (O(n*p^2) per call, only one column
differing between calls). Fixed by computing the base design's QR decomposition ONCE and extending it
per candidate via ``scipy.linalg.qr_insert`` (an O(n*p) column-insert update) -- mathematically
equivalent to a fresh least-squares solve, not an approximation.

These tests pin the numerical equivalence directly (the exact operations the fix uses, side by side
with the original lstsq-based approach) since ``_rp_r2`` itself is a closure buried inside a very large
function and not independently importable. A dedicated end-to-end smoke test exercises the real
``MRMR.fit()`` path that contains this code, confirming it runs and produces valid output at a shape
that exercises the protection block (fit with several pre-selectable raw columns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.linalg as sla


def _lstsq_r2(base_tr, base_va, extra, tr, va, y_tr, yv, ss):
    if extra is None:
        A_tr, A_va = base_tr, base_va
    else:
        A_tr = np.column_stack((base_tr, extra[tr]))
        A_va = np.column_stack((base_va, extra[va]))
    coef, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
    return 1.0 - float(np.sum((yv - A_va @ coef) ** 2)) / ss


def _qr_insert_r2(Q, R, base_va, extra, tr, va, y_tr, yv, ss, coef_base=None):
    if extra is None:
        return 1.0 - float(np.sum((yv - base_va @ coef_base) ** 2)) / ss
    q1, r1 = sla.qr_insert(Q, R, extra[tr], Q.shape[1], which="col")
    coef = sla.solve_triangular(r1, q1.T @ y_tr)
    A_va = np.column_stack((base_va, extra[va]))
    return 1.0 - float(np.sum((yv - A_va @ coef) ** 2)) / ss


def _make_scenario(seed, n=6000, p_base=25):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    va = (idx % 3) == 0
    tr = ~va
    y = rng.standard_normal(n) + 0.3 * rng.standard_normal(n)
    base_cols = [np.ones(n)] + [rng.standard_normal(n) for _ in range(p_base - 1)]
    base_mat = np.column_stack(base_cols)
    y_tr, yv = y[tr], y[va]
    base_tr, base_va = base_mat[tr], base_mat[va]
    ss = float(np.sum((yv - yv.mean()) ** 2))
    return rng, n, tr, va, y, y_tr, yv, base_cols, base_tr, base_va, ss


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_qr_insert_r2_matches_lstsq_r2_baseline(seed):
    _rng, _n, tr, va, _y, y_tr, yv, _base_cols, base_tr, base_va, ss = _make_scenario(seed)
    old_base = _lstsq_r2(base_tr, base_va, None, tr, va, y_tr, yv, ss)

    Q, R = sla.qr(base_tr, mode="economic")
    coef_base = sla.solve_triangular(R, Q.T @ y_tr)
    new_base = _qr_insert_r2(Q, R, base_va, None, tr, va, y_tr, yv, ss, coef_base=coef_base)

    assert new_base == pytest.approx(old_base, abs=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_qr_insert_r2_matches_lstsq_r2_across_candidates(seed):
    """Stress test: plain-random, base-correlated, and target-correlated candidates -- the accept/reject
    threshold decision (the only thing production code acts on) must match for every one."""
    rng, n, tr, va, y, y_tr, yv, base_cols, base_tr, base_va, ss = _make_scenario(seed)
    old_base = _lstsq_r2(base_tr, base_va, None, tr, va, y_tr, yv, ss)
    Q, R = sla.qr(base_tr, mode="economic")
    coef_base = sla.solve_triangular(R, Q.T @ y_tr)
    new_base = _qr_insert_r2(Q, R, base_va, None, tr, va, y_tr, yv, ss, coef_base=coef_base)

    THRESH = 0.005
    max_diff = 0.0
    mismatches = []
    for i in range(150):
        if i % 3 == 0:
            extra = base_cols[1 + (i % (len(base_cols) - 1))] + rng.standard_normal(n) * 1e-6
        elif i % 3 == 1:
            extra = rng.standard_normal(n) + y * 0.15
        else:
            extra = rng.standard_normal(n) * (3 if i % 5 == 0 else 1)

        old_r2 = _lstsq_r2(base_tr, base_va, extra, tr, va, y_tr, yv, ss)
        new_r2 = _qr_insert_r2(Q, R, base_va, extra, tr, va, y_tr, yv, ss)

        max_diff = max(max_diff, abs(old_r2 - new_r2))
        old_decision = (old_r2 - old_base) >= THRESH
        new_decision = (new_r2 - new_base) >= THRESH
        if old_decision != new_decision:
            mismatches.append((i, old_r2, new_r2))

    assert not mismatches, f"threshold-decision mismatches: {mismatches}"
    assert max_diff < 1e-6, f"max R^2 diff {max_diff:.2e} exceeds tolerance"


def test_qr_insert_raises_on_exact_duplicate_matching_lstsq_reject_outcome():
    """A candidate that exactly duplicates an existing base column carries zero new information --
    ``qr_insert`` raises (caught -> -inf -> rejected); the original lstsq path finds a degenerate but
    finite minimum-norm solution with ~zero R^2 increment (also rejected). Both reach the same practical
    decision via different mechanisms."""
    _rng, _n, tr, va, _y, y_tr, yv, base_cols, base_tr, base_va, ss = _make_scenario(seed=7)
    dup = base_cols[1].copy()

    old_base = _lstsq_r2(base_tr, base_va, None, tr, va, y_tr, yv, ss)
    old_dup_r2 = _lstsq_r2(base_tr, base_va, dup, tr, va, y_tr, yv, ss)
    assert old_dup_r2 - old_base < 0.005, "exact duplicate must not clear the R^2 floor via lstsq"

    Q, R = sla.qr(base_tr, mode="economic")
    with pytest.raises(np.linalg.LinAlgError):
        sla.qr_insert(Q, R, dup[tr], Q.shape[1], which="col")


@pytest.mark.slow
def test_mrmr_fit_with_raw_protection_path_exercised_runs_clean():
    """End-to-end smoke test: a real MRMR.fit() with FE enabled and several raw columns, shaped to reach
    the raw-feature floor-drop protection block (needs selected_vars non-empty and X a DataFrame).
    Confirms the QR-based _rp_r2 path runs without exception and produces a valid, non-degenerate fit --
    not a test of the protection's SELECTION correctness (covered by the underselection.py suite), just
    that the optimized numerical path is exercised end-to-end without crashing."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(20)})
    y = X["x0"] * 2.0 + X["x1"] * 1.5 + rng.standard_normal(n) * 0.5

    m = MRMR(fe_max_steps=1, fe_unary_preset="minimal", fe_binary_preset="minimal", verbose=0, random_seed=0)
    m.fit(X, y)

    assert m.support_ is not None
    assert np.asarray(m.support_).sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])

"""Evaluating the distinct legs of a cross-basis matrix in one batched call per degree, and forming the products in one launch, give the per-leg loop's matrix bit for bit."""

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _gpu_resident_cross_basis as cb


def test_batched_legs_equal_per_leg(monkeypatch):
    """Pair, triplet and repeated-leg specs over five columns, mixed degrees and one pinned basis: same matrix with the prefetch and without it."""
    r = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({f"c{i}": r.standard_normal(n).astype(np.float32) * (1 + i) for i in range(5)})
    X["c4"] = np.exp(X["c4"] / 5).astype(np.float32)  # a skewed column the router may send to another basis
    specs = [
        {"legs": [("c0", 1), ("c1", 2)]},
        {"legs": [("c0", 2), ("c1", 2), ("c2", 1)]},
        {"legs": [("c3", 1), ("c4", 3)]},
        {"legs": [("c0", 1), ("c3", 1), ("c2", 2), ("c4", 1)]},
        {"legs": [("c1", 3, "laguerre"), ("c2", 2)]},
        {"legs": [("c0", 1), ("c1", 2)]},
    ]
    got = cp.asnumpy(cb.build_leg_product_matrix_gpu(cp, X, specs))
    # the reference: no batched leg evaluation and no fused product, i.e. one evaluation per leg and one multiplication per factor
    from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands

    clear_fe_resident_operands()  # the fit-wide leg cache would hand the first call's legs to the reference
    monkeypatch.setattr(cb, "_prefetch_legs", lambda *a, **k: None)
    monkeypatch.setattr(cb, "_fused_leg_products", lambda *a, **k: None)
    want = cp.asnumpy(cb.build_leg_product_matrix_gpu(cp, X, specs))
    np.testing.assert_array_equal(got, want)


def test_second_call_takes_legs_from_the_fit_wide_cache(monkeypatch):
    """A second matrix over the same operand values evaluates no leg again and equals the first bit for bit; changed values are evaluated afresh."""
    from mlframe.feature_selection.filters import _gpu_resident_basis as gb
    from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands

    r = np.random.default_rng(1)
    X = pd.DataFrame({f"c{i}": r.standard_normal(2000).astype(np.float32) for i in range(3)})
    specs = [{"legs": [("c0", 1), ("c1", 2)]}, {"legs": [("c1", 1), ("c2", 2)]}]
    clear_fe_resident_operands()
    first = cp.asnumpy(cb.build_leg_product_matrix_gpu(cp, X, specs))
    calls = []
    real = gb._gpu_evaluate_basis_matrix
    monkeypatch.setattr(gb, "_gpu_evaluate_basis_matrix", lambda *a, **k: calls.append(1) or real(*a, **k))
    again = cp.asnumpy(cb.build_leg_product_matrix_gpu(cp, X, specs))
    assert not calls
    np.testing.assert_array_equal(first, again)
    X2 = X.copy()
    X2["c0"] = X2["c0"] * 2
    cb.build_leg_product_matrix_gpu(cp, X2, specs)
    assert calls


def test_raw_baseline_mi_is_scored_once_per_column_and_matches_the_block_score(monkeypatch):
    """A column scored by an earlier call is served from the table; a subset or reordered call gives the same per-column values as scoring the whole block."""
    from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands

    r = np.random.default_rng(2)
    n = 4000
    X = pd.DataFrame({f"c{i}": r.standard_normal(n).astype(np.float32) for i in range(4)})
    y = (X["c0"] * X["c1"] + r.standard_normal(n) > 0).astype(np.int64).to_numpy()
    clear_fe_resident_operands()
    full = cb._raw_baseline_mi(cp, X.to_numpy(np.float32), list(X.columns), y, 10, np.float32)
    scored = []
    real = cb._resident_mi
    monkeypatch.setattr(cb, "_resident_mi", lambda c, m, *a, **k: scored.append(m.shape[1]) or real(c, m, *a, **k))
    sub = ["c3", "c1"]
    part = cb._raw_baseline_mi(cp, X[sub].to_numpy(np.float32), sub, y, 10, np.float32)
    assert not scored
    assert part == {k: full[k] for k in sub}
    X["c9"] = r.standard_normal(n).astype(np.float32)
    cb._raw_baseline_mi(cp, X[["c1", "c9"]].to_numpy(np.float32), ["c1", "c9"], y, 10, np.float32)
    assert scored == [1]

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
    monkeypatch.setattr(cb, "_prefetch_legs", lambda *a, **k: None)
    monkeypatch.setattr(cb, "_fused_leg_products", lambda *a, **k: None)
    want = cp.asnumpy(cb.build_leg_product_matrix_gpu(cp, X, specs))
    np.testing.assert_array_equal(got, want)

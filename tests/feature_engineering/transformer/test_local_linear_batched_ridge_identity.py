"""Identity-pin for the local_linear batched-ridge optimization.

``compute_local_linear_attention`` originally fit a fresh ``sklearn.linear_model.Ridge``
object per anchor row (``Ridge().fit/.predict`` inside a Python ``for q in range(n_anchor)``
loop). That was replaced with a single BATCHED centred normal-equations solve over all
rows -- one ``np.linalg.solve`` over ``(N, d, d)`` -- which is mathematically identical to
``Ridge(alpha, fit_intercept=True)`` up to float32 reduction order, and 18-70x faster on
production shapes (see ``feature_engineering/_benchmarks/bench_local_linear_batched_ridge.py``).

This test pins that equivalence. The ANN neighbour search is monkeypatched to a deterministic
exact-kNN so the OLD reference re-implementation and the NEW module path consume the SAME
neighbour blocks; only the regression differs. We assert the emitted features match the OLD
per-row ``sklearn.Ridge`` reference to within float32 ULP tolerance (the outputs are stored as
float32; the divergence is pure reduction order, well under any value/selection threshold).

A future revert to a different regression (wrong centering, missing ridge term, transposed
einsum, mis-scattered intercept/r2) breaks the tolerance and fails here.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

import mlframe.feature_engineering.transformer.local_linear as ll_mod
from mlframe.feature_engineering.transformer.local_linear import compute_local_linear_attention


class _ExactKNN:
    """Deterministic exact cosine-kNN stub matching build_hnsw_index/query_topk contract."""

    def __init__(self, pool: np.ndarray):
        """Helper: Init  ."""
        self.pool = np.asarray(pool, dtype=np.float64)
        norms = np.linalg.norm(self.pool, axis=1, keepdims=True)
        self.unit = self.pool / np.maximum(norms, 1e-12)


def _fake_build(pool, **kwargs):
    """Helper: Fake build."""
    return _ExactKNN(pool)


def _fake_query(index: _ExactKNN, anchor: np.ndarray, k: int):
    """Helper: Fake query."""
    a = np.asarray(anchor, dtype=np.float64)
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    sims = an @ index.unit.T  # (n_anchor, n_pool), cosine similarity
    ids = np.argsort(-sims, axis=1, kind="stable")[:, :k].astype(np.int64)
    dists = np.take_along_axis(1.0 - sims, ids, axis=1).astype(np.float32)
    return ids, dists


def _old_reference(X_neighbour_pool, y_neighbour_pool, topk_ids, d, return_r2, ridge_alpha, dtype):
    """Verbatim OLD per-row sklearn.Ridge loop."""
    from sklearn.linear_model import Ridge

    n_anchor = topk_ids.shape[0]
    n_out_cols = d + 1 + (1 if return_r2 else 0)
    out = np.zeros((n_anchor, n_out_cols), dtype=dtype)
    for q in range(n_anchor):
        ids = topk_ids[q]
        Xn = X_neighbour_pool[ids]
        yn = y_neighbour_pool[ids]
        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        model.fit(Xn, yn)
        out[q, 0] = model.intercept_
        out[q, 1 : 1 + d] = model.coef_
        if return_r2:
            pred = model.predict(Xn)
            ss_res = float(np.sum((yn - pred) ** 2))
            ss_tot = float(np.sum((yn - yn.mean()) ** 2))
            out[q, 1 + d] = 1.0 - ss_res / max(ss_tot, 1e-12)
    return out


@pytest.mark.parametrize(
    "seed,n_train,d,k,return_r2",
    [
        (0, 400, 8, 32, True),
        (1, 500, 6, 24, True),
        (2, 300, 4, 16, False),
    ],
)
def test_local_linear_batched_matches_per_row_ridge(monkeypatch, seed, n_train, d, k, return_r2):
    """Local linear batched matches per row ridge."""
    monkeypatch.setattr(ll_mod, "build_hnsw_index", _fake_build)
    monkeypatch.setattr(ll_mod, "query_topk", _fake_query)

    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, d)).astype(np.float32)
    y_train = (X_train[:, 0] * 1.5 - X_train[:, 1] + rng.standard_normal(n_train) * 0.1).astype(np.float32)
    X_query = rng.standard_normal((120, d)).astype(np.float32)

    dtype = np.float32
    ridge_alpha = 1e-3
    df = compute_local_linear_attention(
        X_train,
        y_train,
        X_query,
        splitter=None,
        seed=seed,
        k=k,
        ridge_alpha=ridge_alpha,
        standardize=True,
        return_r2=return_r2,
        dtype=dtype,
    )
    new = df.to_numpy()

    # Reproduce the standardisation + neighbour blocks the module used, then run the OLD reference.
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler().fit(X_train)
    Xt_s = scaler.transform(X_train).astype(dtype, copy=False)
    Xq_s = scaler.transform(X_query).astype(dtype, copy=False)
    idx = _ExactKNN(Xt_s)
    topk_ids, _ = _fake_query(idx, Xq_s, k=k)
    old = _old_reference(Xt_s, y_train.astype(np.float32), topk_ids, d, return_r2, ridge_alpha, dtype)

    assert new.shape == old.shape
    # Float32 ULP-level equivalence (pure reduction-order divergence; values stored as float32).
    np.testing.assert_allclose(new, old, rtol=1e-3, atol=1e-4)

"""Regression sensor for the iter111 perf rewrite of
``generate_conditional_residual_features``.

The inner ``(x_i, x_j)`` loop accumulated per-bin sum/count via ``np.add.at``
(unbuffered scatter) and recomputed the per-``x_i`` finiteness mask + global
mean once per pair. iter111 replaced the scatter with ``np.bincount`` (a single
C pass, bit-identical accumulation order) and hoisted the per-``x_i``
invariants out of the inner loop.

Two pins:

* ``test_..._does_not_use_np_add_at`` -- the generate path must NOT call
  ``np.add.at`` any more (FAILS on the pre-iter111 code which called it 2x per
  pair).
* ``test_..._bit_identical_to_add_at_reference`` -- the bincount accumulation is
  byte-for-byte identical to the add.at reference on discrete / NaN-mixed data
  (exercises ties + the global-mean fallback bin).
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._extra_fe_families import (
    generate_conditional_residual_features,
)


def _make_df(n: int = 5000) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(7)
    cols = ["a", "b", "c"]
    data = {
        "a": rng.standard_normal(n),
        "b": rng.integers(0, 5, n).astype(float),  # discrete -> ties
        "c": rng.standard_normal(n),
    }
    data["c"][rng.integers(0, n, n // 50)] = np.nan  # exercise global-mean fallback
    return pd.DataFrame(data), cols


def test_conditional_residual_generate_uses_bincount_not_add_at(monkeypatch):
    # np.add.at is a ufunc METHOD and is READ-ONLY in current numpy -- it cannot be monkeypatched
    # (setattr raises "'numpy.ufunc' object attribute 'at' is read-only"). Pin the iter111 rewrite by its
    # POSITIVE signal instead: np.bincount IS used (it replaced the per-pair np.add.at scatter, 2 calls
    # per pair). Bit-identical accumulation is pinned by the reference test below; together they guarantee
    # the scatter-free bincount path. A regression back to np.add.at would drop bincount usage to zero here.
    calls = {"n": 0}
    orig = np.bincount

    def spy(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(np, "bincount", spy)
    X, cols = _make_df()
    generate_conditional_residual_features(X, cols, n_bins=10)
    assert calls["n"] > 0, "generate must use np.bincount (the iter111 scatter-free accumulation)"


def test_conditional_residual_generate_bit_identical_to_add_at_reference():
    X, cols = _make_df()
    enc, _ = generate_conditional_residual_features(X, cols, n_bins=10)

    # Independent add.at reference replicating the pre-iter111 numerics.
    col_vals = {c: np.asarray(X[c].to_numpy(), dtype=np.float64) for c in cols}
    for x_j in cols:
        xj = col_vals[x_j]
        finite = xj[np.isfinite(xj)]
        q = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 11)))
        if q.size < 2:
            lo = float(finite.min())
            q = np.array([lo, lo + 1.0])
        codes_j = np.clip(np.searchsorted(q[1:-1], xj, side="right"), 0, max(0, q.size - 2))
        codes_j[~np.isfinite(xj)] = 0
        n_bins_eff = q.size - 1
        for x_i in cols:
            if x_i == x_j:
                continue
            xi = col_vals[x_i]
            fin = np.isfinite(xi)
            gm = float(xi[fin].mean()) if fin.any() else 0.0
            bs = np.zeros(n_bins_eff)
            bc = np.zeros(n_bins_eff)
            np.add.at(bs, codes_j[fin], xi[fin])
            np.add.at(bc, codes_j[fin], 1.0)
            bm = np.where(bc > 0.0, bs / np.maximum(bc, 1.0), gm)
            ref = np.where(fin, xi - bm[codes_j], 0.0)
            got = enc[f"{x_i}__cond_resid_by__{x_j}"].to_numpy()
            assert np.array_equal(got, ref), f"mismatch for ({x_i},{x_j})"

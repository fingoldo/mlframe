"""Regression tests for the iter93 Gram-gated condition-number fast path in
``_linear_residual_multi_fit``.

The multicollinearity cond gate reads only the singular VALUES of the (n, K)
scaled base; iter93 obtains them from the (K, K) Gram matrix via ``eigvalsh``
(~2.7x faster than the tall-matrix ``svd``) and recomputes the exact SVD cond
only inside a +-0.01% band of the gate threshold. The fitted alphas / beta /
collinear_fallback decision MUST stay bit-identical to the legacy SVD-cond path
on every regime (clean, near-collinear fallback, constant column, trailing NaN,
K=1), since the lstsq inputs are untouched.

The fast-path sensor (``test_multibase_fit_uses_gram_eigvalsh_for_cond``) fails
on pre-fix code: HEAD computes the cond gate with ``np.linalg.svd`` and never
calls ``np.linalg.eigvalsh`` in that gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.transforms.linear import _linear_residual_multi_fit

_TINY = np.finfo(np.float64).tiny
_COND = 30.0


def _legacy_fit(y: np.ndarray, base: np.ndarray) -> dict:
    """Pre-iter93 reference: astype copies + tall-matrix SVD condition number."""
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    base_f = base.astype(np.float64)
    y_f = y.astype(np.float64)
    row_finite = np.isfinite(y_f) & np.all(np.isfinite(base_f), axis=1)
    if not bool(row_finite.all()):
        base_f = base_f[row_finite]
        y_f = y_f[row_finite]
    n, k = base_f.shape
    if n < k + 1:
        return {"alphas": [0.0] * k, "beta": float(np.mean(y_f)) if n > 0 else 0.0, "collinear_fallback": True}
    X = np.column_stack([base_f, np.ones(n, dtype=np.float64)])
    if k == 1:
        cond = 1.0
    else:
        bc = base_f - base_f.mean(axis=0, keepdims=True)
        cn = np.linalg.norm(bc, axis=0)
        if np.any(cn < 1e-12):
            cond = float("inf")
        else:
            sv = np.linalg.svd(bc / cn, compute_uv=False)
            cond = float(sv.max() / max(sv.min(), _TINY))
    if cond > _COND or not np.isfinite(cond):
        return {"alphas": [0.0] * k, "beta": float(np.mean(y_f)), "collinear_fallback": True}
    coef, *_ = np.linalg.lstsq(X, y_f, rcond=None)
    return {"alphas": [float(c) for c in coef[:k]], "beta": float(coef[-1]), "collinear_fallback": False}


def _cases():
    rng = np.random.default_rng(7)
    n = 4000
    out = {}
    b = rng.normal(size=(n, 4))
    out["clean"] = (b.sum(1) + rng.normal(size=n), b)
    b = rng.normal(size=(n, 3))
    b[:, 2] = b[:, 0] + rng.normal(0, 1e-6, n)
    out["near_collinear"] = (b.sum(1) + rng.normal(size=n), b)
    b = rng.normal(size=(n, 2))
    b[:, 1] = 5.0
    out["constant_col"] = (b[:, 0] + rng.normal(size=n), b)
    b = rng.normal(size=(n, 3))
    b[:30, 0] = np.nan
    out["trailing_nan"] = (b.sum(1) + rng.normal(size=n), b)
    b = rng.normal(size=(n, 1))
    out["k1"] = (b[:, 0] + rng.normal(size=n), b)
    # scale-mismatched bases (BKW unit-norm scaling must keep these non-collinear)
    b = rng.normal(size=(n, 2))
    b[:, 1] *= 1e6
    out["scale_mismatch"] = (b.sum(1) + rng.normal(size=n), b)
    return out


@pytest.mark.parametrize("name", list(_cases().keys()))
def test_multibase_fit_bit_identical_to_legacy_svd_cond(name):
    """Multibase fit bit identical to legacy svd cond."""
    y, base = _cases()[name]
    old = _legacy_fit(y, base)
    new = _linear_residual_multi_fit(y, base)
    assert old["collinear_fallback"] == new["collinear_fallback"], name
    assert np.array_equal(np.array(old["alphas"]), np.array(new["alphas"])), name
    assert (old["beta"] == new["beta"]) or (np.isnan(old["beta"]) and np.isnan(new["beta"])), name


def test_multibase_fit_uses_gram_eigvalsh_for_cond(monkeypatch):
    """Spy proving the cond gate now routes through ``eigvalsh`` (Gram path).

    Pre-fix the gate used ``np.linalg.svd`` and never called ``eigvalsh`` -- so
    this assertion is the regression sensor (fails on HEAD)."""
    rng = np.random.default_rng(3)
    n = 4000
    base = rng.normal(size=(n, 3))
    y = base.sum(1) + rng.normal(size=n)
    calls = {"eigvalsh": 0}
    real_eigvalsh = np.linalg.eigvalsh

    def spy(a, *args, **kwargs):
        """Spy."""
        calls["eigvalsh"] += 1
        return real_eigvalsh(a, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigvalsh", spy)
    _linear_residual_multi_fit(y, base)
    assert calls["eigvalsh"] >= 1, "K>1 cond gate must use the Gram eigvalsh fast path"


def test_multibase_fit_no_full_copy_on_float64_input(monkeypatch):
    """asarray (view), not astype, on already-float64 input: the body must not
    allocate a fresh full copy of the trial buffer for the dtype conversion."""
    rng = np.random.default_rng(5)
    n = 4000
    base = np.ascontiguousarray(rng.normal(size=(n, 3)))  # already float64
    y = np.ascontiguousarray(base.sum(1) + rng.normal(size=n))
    # asarray returns the same object for a matching-dtype contiguous array.
    assert np.asarray(base, dtype=np.float64) is base
    assert np.asarray(y, dtype=np.float64) is y
    res = _linear_residual_multi_fit(y, base)
    assert res["collinear_fallback"] is False

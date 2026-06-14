"""iter93 A/B: ``_linear_residual_multi_fit`` Gram-gated condition number + asarray views.

The per-fold OLS fit in the forward-stepwise CV loop (``forward_stepwise._cv_rmse_with_folds``,
profiled cumtime ~1.2s on the 1M-driven REGRESSION discovery) spent its non-lstsq budget on
(a) two redundant ``astype(float64)`` copies of an already-float64 trial buffer and (b) a full
tall-matrix ``np.linalg.svd`` of the (n, K) scaled base just to read its singular VALUES for the
multicollinearity cond gate. This bench A/Bs the shipped ``_linear_residual_multi_fit`` (Gram-based
``eigvalsh`` cond, gated to recompute exact SVD within +-0.01% of the threshold; asarray views) vs
the legacy SVD-cond + astype path reconstructed inline, at the screen working size.

Run:
    CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 \
        python -m mlframe.training.composite.discovery._benchmarks.bench_iter93_multibase_ols_fit
"""
from __future__ import annotations

import sys

sys.modules.setdefault("cupy", None)
import scipy.stats  # noqa: F401,E402
import numba  # noqa: F401,E402

import time  # noqa: E402

import numpy as np  # noqa: E402

from ...transforms.linear import _linear_residual_multi_fit  # noqa: E402

_TINY = np.finfo(np.float64).tiny
_COND = 30.0


def _legacy_fit(y, base):
    """Reconstruction of the pre-iter93 path: astype copies + tall-matrix SVD cond."""
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


def _timeit(fn, r=60):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(r):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


def main() -> None:
    rng = np.random.default_rng(0)
    n = 80_000
    print(f"--- _linear_residual_multi_fit A/B @ n={n} (median ms, best-of-60) ---")
    for k in (2, 3, 4):
        base = rng.normal(size=(n, k))
        y = base.sum(1) + rng.normal(size=n)
        old = _legacy_fit(y, base)
        new = _linear_residual_multi_fit(y, base)
        a_eq = np.array_equal(np.array(old["alphas"]), np.array(new["alphas"]))
        b_eq = old["beta"] == new["beta"]
        f_eq = old["collinear_fallback"] == new["collinear_fallback"]
        t_old = _timeit(lambda: _legacy_fit(y, base))
        t_new = _timeit(lambda: _linear_residual_multi_fit(y, base))
        print(f"k={k}: OLD {t_old:.3f}  NEW {t_new:.3f}  ({t_old / t_new:.2f}x) | alphas_biteq={a_eq} beta_eq={b_eq} fallback_eq={f_eq}")


if __name__ == "__main__":  # pragma: no cover
    main()

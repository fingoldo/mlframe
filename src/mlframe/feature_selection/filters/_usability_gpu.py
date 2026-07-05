"""GPU-resident (cupy) scoring primitives for the usability-aware selection / pure-form retention
passes (gated, default OFF). The CPU numpy/sklearn path stays the DEFAULT + FALLBACK.

WHY THIS MODULE EXISTS
----------------------
The two usability passes (``build_usability_candidate_pool`` /
``_usability_aware_selection._abscorr``-driven diversity+shortlist, and ``_fe_pure_form_retention``'s
residual-after-additive-basis gate) score candidates on the CPU with numpy ``corrcoef`` and an sklearn
``LinearRegression`` residual. Those primitives -- |corr(candidate, continuous y)|, |corr(candidate,
residual)|, and the additive-basis residual itself -- are array-elementwise/GEMV work that is GPU-able
with cupy / cupyx. This module hosts the cupy twins so the two files import ONE home (no duplication;
the per-candidate MI is already GPU-routed via ``_usability_njit_pool._pair_combo_mi_cupy`` -- reused,
not re-implemented here).

SELECTION-EQUIVALENCE IS THE BAR (non-negotiable). The clean-form demotion uses |corr| with the
CONTINUOUS y and is ULP-sensitive: a cupy-vs-numpy float drift can flip the canonical clean-compound
recovery. Every helper here therefore:

  * computes in float64 (the heavy-tail precision the CPU path uses),
  * mirrors numpy ``corrcoef``'s exact estimator (centered dot / sqrt(ss_u*ss_v); the SAME std<1e-12
    guard the CPU ``_abscorr`` applies, on the SAME float64 inputs),
  * raises on ANY cupy/device error so the caller falls back to the exact CPU path.

The gate (``MLFRAME_FE_GPU_USABILITY``) defaults OFF. It is enabled ONLY after a host has VERIFIED the
GPU path returns the SAME selection as the CPU path (the gate-on pytest must stay 11 passed). On a host
where cupy corr drifts a pin (the GTX 1050 Ti reassociates the last bits of a reduction), this path is
NOT selection-safe and stays gated OFF -- the win is real on stronger cards / larger n, but selection
can never regress (feedback_gate_optimization_on_safe_condition).

NOTE ON SCALE (dev-box measurement, GTX 1050 Ti): the two passes row-SUBSAMPLE to ~3000 rows
(``max_rows``) before building the pool, so these corr/residual reductions run at n~3000 where the cupy
H2D + launch overhead dominates the tiny on-device reduction -> the GPU path is SLOWER here and the gate
stays OFF (the same conclusion the resident-FE notes reached for n<50k). The deliverable is the
correct, gated GPU path that wins on a larger subsample / stronger card; the tuner-routed MI kernel
(_pair_combo_mi_cupy) is the genuinely n-scaling piece and already ships GPU-capable.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import cupy as _cp
    _CUPY_AVAIL = True
except Exception:
    _cp = None
    _CUPY_AVAIL = False


def fe_gpu_usability_enabled() -> bool:
    """Whether the gated cupy usability-scoring path is active. OFF unless ``MLFRAME_FE_GPU_USABILITY``
    is truthy AND cupy is importable AND the global GPU off-switch is not set. Default OFF -- the CPU
    numpy/sklearn path is the proven, selection-exact default."""
    if not _CUPY_AVAIL:
        return False
    if os.environ.get("MLFRAME_FE_GPU_USABILITY", "").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    try:
        from ._gpu_policy import gpu_globally_disabled
        if gpu_globally_disabled():
            return False
    except Exception:
        pass
    return True


def gpu_abscorr(u: np.ndarray, v: np.ndarray) -> float:
    """cupy float64 twin of ``_usability_aware_selection._abscorr``: ``|corr(u, v)|`` with the SAME
    std<1e-12 zero-guard and finite-check. Raises on any cupy/device error (caller falls back to the
    exact numpy path). Inputs are upcast to float64 on device for the heavy-tail precision."""
    cp = _cp
    du = cp.asarray(u, dtype=cp.float64).ravel()
    dv = cp.asarray(v, dtype=cp.float64).ravel()
    if du.size == 0:
        return 0.0
    # population std (ddof=0), matching numpy np.std default used by the CPU guard.
    su = float(du.std())
    sv = float(dv.std())
    if su < 1e-12 or sv < 1e-12:
        return 0.0
    um = du - du.mean()
    vm = dv - dv.mean()
    ssu = float((um * um).sum())
    ssv = float((vm * vm).sum())
    if ssu <= 0.0 or ssv <= 0.0:
        return 0.0
    r = float((um * vm).sum()) / float(np.sqrt(ssu * ssv))
    return abs(r) if np.isfinite(r) else 0.0


def gpu_abscorr_batch(cols: "np.ndarray", v: np.ndarray) -> np.ndarray:
    """Batched ``|corr(cols[:, j], v)|`` for every column ``j`` of ``cols`` (shape ``(n, K)``) against
    the 1-D target ``v`` -- the shortlist-scoring inner loop, where ``v`` is the held-out residual (or
    the mean residual) and each column is a candidate's continuous values. cupy float64; mirrors the
    per-candidate ``_abscorr`` (centered dot / sqrt(ss_col*ss_v), std<1e-12 -> 0). Returns a host (K,)
    float64 array of |corr| values. Raises on any cupy/device error so the caller falls back to CPU."""
    cp = _cp
    M = cp.asarray(cols, dtype=cp.float64)
    if M.ndim == 1:
        M = M[:, None]
    dv = cp.asarray(v, dtype=cp.float64).ravel()
    n, K = M.shape
    out = np.zeros(K, dtype=np.float64)
    if n == 0 or K == 0:
        return out
    # std<1e-12 guard per column + for v (population std, ddof=0).
    col_std = M.std(axis=0)  # (K,)
    v_std = float(dv.std())
    if v_std < 1e-12:
        return out
    vm = dv - dv.mean()  # (n,)
    ssv = float((vm * vm).sum())
    if ssv <= 0.0:
        return out
    Mc = M - M.mean(axis=0, keepdims=True)  # (n, K)
    num = Mc.T @ vm  # (K,) centered dot
    ssc = (Mc * Mc).sum(axis=0)  # (K,)
    denom = cp.sqrt(ssc * ssv)
    valid = (col_std >= 1e-12) & (ssc > 0.0) & (denom > 0.0)
    r = cp.where(valid, num / cp.where(denom > 0.0, denom, 1.0), 0.0)
    r = cp.where(cp.isfinite(r), cp.abs(r), 0.0)
    return cp.asnumpy(r).astype(np.float64)


def gpu_additive_basis_residual(fv: np.ndarray, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """cupy twin of the ``_fe_pure_form_retention._adds_nonlinear_value`` residual: regress the form
    ``fv`` on the additive single-operand basis of ``xa`` and ``xb`` (the SAME 6-function basis the CPU
    path builds: standardized x, x^2, x^3, sign*sqrt|x|, sign*log1p|x|, 1/(|x|+1)), with the CPU's
    StandardScaler+LinearRegression replaced by the EQUIVALENT mean-centered OLS (StandardScaler is an
    affine per-column rescale that LinearRegression-with-intercept is invariant to). Returns the host
    residual ``fv - fit`` (float64). Raises on any cupy/device error so the caller falls back to CPU.

    Mean-centered-OLS equivalence: ``make_pipeline(StandardScaler(), LinearRegression())`` predicts the
    same values as a centered-design OLS-with-intercept (the scaler's affine map is absorbed into the
    coefficients + intercept), so the residual is identical up to fp round-off."""
    cp = _cp

    def _basis(x):
        xs = (x - x.mean()) / (x.std() + 1e-12)
        return [xs, xs * xs, xs * xs * xs, cp.sign(xs) * cp.sqrt(cp.abs(xs)), cp.sign(xs) * cp.log1p(cp.abs(xs)), 1.0 / (cp.abs(xs) + 1.0)]

    dfv = cp.asarray(fv, dtype=cp.float64).ravel()
    dxa = cp.asarray(xa, dtype=cp.float64).ravel()
    dxb = cp.asarray(xb, dtype=cp.float64).ravel()
    cols = _basis(dxa) + _basis(dxb)
    Xr = cp.stack(cols, axis=1)  # (n, 12)
    # mean-centered OLS with intercept: center design + target, solve normal equations via lstsq.
    Xc = Xr - Xr.mean(axis=0, keepdims=True)
    ybar = dfv.mean()
    yc = dfv - ybar
    beta, *_ = cp.linalg.lstsq(Xc, yc, rcond=None)
    pred = ybar + Xc @ beta
    resid = dfv - pred
    return cp.asnumpy(resid).astype(np.float64)


__all__ = [
    "fe_gpu_usability_enabled",
    "gpu_abscorr",
    "gpu_abscorr_batch",
    "gpu_additive_basis_residual",
]

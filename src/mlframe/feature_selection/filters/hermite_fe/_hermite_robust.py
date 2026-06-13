"""Outlier-robust axis normalisation AND warp-coefficient fitting for the
orthogonal-polynomial preprocessors.

Two independent robustness layers, each its own env gate, both heavy-tail-gated
so the clean common case is byte-identical to legacy:

1. AXIS NORMALISATION -- a cheap per-column spike-contamination gate
   (``_detect_heavy_tail``) plus the MAD-anchored robust bounds
   (``_robust_lo_hi`` / ``_robust_scale``) the basis preprocessors use when the
   raw per-column scale is corrupted by injected spikes. GATED on
   ``_robust_axis_enabled`` (default ON; ``MLFRAME_ROBUST_AXIS=0`` replays the
   legacy raw-scale path).

2. WARP-COEFFICIENT FITTING (2026-06-10, backlog idea #17) -- the prewarp / ALS
   fits that BUILD a warp feature solve an ORDINARY least-squares problem
   (``np.linalg.lstsq``), whose squared-error loss is dominated by a few extreme
   rows: under a heavy-tailed / outlier marginal the basis matrix entry ``x**k``
   explodes on an outlier row, so the OLS warp chases the outliers instead of
   recovering the true relationship (measured OOS R2 0.9999 -> 0.92 on a monotone
   cubic with 1.5% injected outliers). ``_huber_irls_lstsq`` is a drop-in robust
   replacement (iteratively-reweighted least squares with a Huber influence
   function) used ONLY when ``_detect_heavy_tail`` fires on the operand AND
   ``_robust_warp_fit_enabled`` is on (default ON; ``MLFRAME_ROBUST_WARP_FIT=0``
   replays the OLS path). On a clean (low-kurtosis) column the gate does not fire
   and the fit is byte-identical to the legacy OLS solve. The OLS solver is kept
   under its own name (``_ols_lstsq``) per the keep-all-kernels rule so the
   dispatcher can route by the heavy-tail predicate and we can A-B / roll back.

The two gates are deliberately ORTHOGONAL knobs (a user can robustify the axis
scale without changing the loss, or vice-versa) -- not folded onto one flag.
"""
from __future__ import annotations

import os

import numba
import numpy as np

# Robust bounds = median +/- _ROBUST_AXIS_K * (1.4826*MAD). MAD is contamination-proof up to ~50% of the column, so the
# derived span ~ 6*sigma stays anchored to the CLEAN core regardless of how many 1000x spikes are injected -- unlike an
# inner-quantile trim, which only excludes the tail when the trim fraction exceeds the contamination fraction. k=3 covers
# ~99.7% of a Gaussian core, matching the legacy intent that the working axis span the bulk of the data.
_ROBUST_AXIS_K = 3.0
# Spike-contamination detector parameters. The gate must trip on INJECTED SPIKE contamination (a thin fraction of points
# orders of magnitude beyond a dense bulk) WITHOUT tripping on a genuinely heavy-tailed-but-clean column (lognormal,
# Student-t, exponential) -- robustifying those would change engineered byte values on legitimate data. A single
# half-range/scale ratio cannot separate the two (a heavy lognormal and a 1%-spike column have similar max/MAD ratios), so
# we test for the SEPARATION SIGNATURE instead: contamination leaves a clear multiplicative GAP between the bulk and the
# spikes, while a smooth heavy tail is continuous. ``_ROBUST_AXIS_OUTER_K`` is the robust-scale multiple defining the
# candidate-outlier band; ``_ROBUST_AXIS_GAP`` is the bulk->outer jump that marks a true gap; ``_ROBUST_AXIS_MAX_FRAC``
# caps the outlier fraction so a column that is >20% beyond 10 sigma is treated as genuinely heavy, not spike-contaminated.
_ROBUST_AXIS_OUTER_K = 10.0
_ROBUST_AXIS_GAP = 3.0
_ROBUST_AXIS_MAX_FRAC = 0.20


def _robust_axis_enabled() -> bool:
    """Default ON. ``MLFRAME_ROBUST_AXIS=0`` forces the legacy raw-scale path for replay / A-B compare."""
    import os as _os
    flag = _os.environ.get("MLFRAME_ROBUST_AXIS", "").strip().lower()
    return flag not in ("0", "false", "off", "no")


@numba.njit(cache=True)
def _detect_heavy_tail_core_njit(xf: np.ndarray, outer_k: float, gap: float, max_frac: float) -> int:
    """Compiled core of ``_detect_heavy_tail`` for the common MAD>0 case. Returns 1 (trip) / 0 (no trip) / -1 (MAD
    collapsed -> caller must run the exact ``np.quantile`` IQR fallback). ``xf`` is the pre-filtered finite subset
    (size>=8 guaranteed by the caller). Numerics mirror the numpy body verbatim: ``np.median`` for the centre and the
    MAD, the same ``1.4826*MAD`` robust scale, the same threshold / count / masked-max / masked-min reductions -- so the
    boolean verdict is bit-identical to the numpy path on every column whose MAD does not collapse to zero."""
    med = np.median(xf)
    mad = np.median(np.abs(xf - med))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return -1  # MAD collapsed; exact IQR fallback handled in Python (np.quantile not njit-supported)
    thr = outer_k * scale
    n_outer = 0
    bulk_edge = -1.0
    outer_min = np.inf
    n = xf.size
    for i in range(n):
        dev = abs(xf[i] - med)
        if dev > thr:
            n_outer += 1
            if dev < outer_min:
                outer_min = dev
        elif dev > bulk_edge:
            bulk_edge = dev
    if n_outer == 0 or n_outer > max_frac * n:
        return 0
    be = bulk_edge if bulk_edge > 1e-12 else 1e-12
    return 1 if (outer_min / be) >= gap else 0


def _detect_heavy_tail_njit(x: np.ndarray) -> bool:
    """njit-accelerated ``_detect_heavy_tail``: identical verdict, fewer per-call passes / temporaries. Falls back to the
    exact numpy IQR path only when the MAD collapses (discrete/tied core), which ``np.quantile`` (not njit-supported) must
    handle to stay bit-identical."""
    xf = x[np.isfinite(x)]
    if xf.size < 8:
        return False
    if xf.dtype != np.float64:
        xf = xf.astype(np.float64)
    verdict = _detect_heavy_tail_core_njit(xf, _ROBUST_AXIS_OUTER_K, _ROBUST_AXIS_GAP, _ROBUST_AXIS_MAX_FRAC)
    if verdict >= 0:
        return verdict == 1
    # MAD collapsed: exact IQR-fallback scale, then the same gap test as the numpy body.
    med = float(np.median(xf))
    robust_scale = _robust_scale(xf, med)
    if robust_scale <= 1e-12:
        return False
    dev = np.abs(xf - med)
    thr = _ROBUST_AXIS_OUTER_K * robust_scale
    outer_mask = dev > thr
    n_outer = int(np.count_nonzero(outer_mask))
    if n_outer == 0 or n_outer > _ROBUST_AXIS_MAX_FRAC * xf.size:
        return False
    bulk_edge = float(dev[~outer_mask].max())
    outer_min = float(dev[outer_mask].min())
    return (outer_min / max(bulk_edge, 1e-12)) >= _ROBUST_AXIS_GAP


# Finite-subset size below which the fused njit core (one loop, no boolean-mask / abs-array temporaries) beats the numpy
# body; above it numpy's introselect ``np.median`` outpaces numba's sort-based median. Crossover ~3000 measured on the dev
# box (bench_detect_heavy_tail_njit.py: 1.31x@2407 / 1.20x@1000 win; 0.86-0.90x loss at n>=4000). Env-overridable per HW.
_DETECT_HEAVY_TAIL_NJIT_MAX_N = int(os.environ.get("MLFRAME_DETECT_HEAVY_TAIL_NJIT_MAX_N", "3000"))


def _detect_heavy_tail(x: np.ndarray) -> bool:
    """Cheap per-column SPIKE-contamination gate (size-gated dispatcher). True iff a thin fraction of points sit beyond
    ``_ROBUST_AXIS_OUTER_K`` robust scales AND are separated from the bulk by a multiplicative GAP of >= ``_ROBUST_AXIS_GAP``.

    The gap test is what distinguishes injected spike contamination (a dense bulk + a handful of order-of-magnitude
    outliers with empty space between them) from a genuinely heavy-tailed-but-clean column (lognormal / Student-t /
    exponential), whose tail is CONTINUOUS with the bulk (gap ~ 1.0-1.3, measured). Only spike contamination corrupts the
    raw scale; a smooth heavy tail is the legitimate home of the skewed bases and must stay on the byte-identical legacy
    path. Degenerate columns (<8 finite values, near-constant, all-non-finite) never trip -- there is no scale to corrupt.

    Below ``_DETECT_HEAVY_TAIL_NJIT_MAX_N`` finite values the fused njit core (``_detect_heavy_tail_njit``) is the faster
    BIT-IDENTICAL path (the per-column FE search calls this thousands of times on small columns); above it the numpy body
    wins on its introselect median, so we route there."""
    finite = np.isfinite(x)
    n_finite = int(np.count_nonzero(finite))
    if n_finite < 8:
        return False
    if n_finite < _DETECT_HEAVY_TAIL_NJIT_MAX_N:
        return _detect_heavy_tail_njit(x)
    return _detect_heavy_tail_numpy(x)


def _detect_heavy_tail_numpy(x: np.ndarray) -> bool:
    """Reference numpy body of ``_detect_heavy_tail`` (the large-n path of the dispatcher; also the bit-identity oracle)."""
    xf = x[np.isfinite(x)]
    if xf.size < 8:
        return False
    med = float(np.median(xf))
    robust_scale = _robust_scale(xf, med)
    if robust_scale <= 1e-12:
        return False
    dev = np.abs(xf - med)
    thr = _ROBUST_AXIS_OUTER_K * robust_scale
    outer_mask = dev > thr
    n_outer = int(np.count_nonzero(outer_mask))
    if n_outer == 0 or n_outer > _ROBUST_AXIS_MAX_FRAC * xf.size:
        # No extreme points, or so many that the tail is genuinely heavy rather than a thin contaminating spike.
        return False
    # Gap test without a full sort: the bulk edge is the largest deviation still inside the bulk, the outer edge the
    # smallest deviation in the candidate-outlier band -- two masked reductions (O(n)) instead of an O(n log n) sort.
    bulk_edge = float(dev[~outer_mask].max())
    outer_min = float(dev[outer_mask].min())
    return (outer_min / max(bulk_edge, 1e-12)) >= _ROBUST_AXIS_GAP


def _robust_scale(xf: np.ndarray, med: float) -> float:
    """Contamination-proof scale: 1.4826*MAD, with an IQR fallback when MAD collapses on a discrete / tied core. Returns
    0.0 only on a genuinely degenerate (near-constant) column, which the caller treats as 'no robust path'."""
    mad = float(np.median(np.abs(xf - med)))
    scale = 1.4826 * mad
    if scale > 1e-12:
        return scale
    q25, q75 = np.quantile(xf, [0.25, 0.75])
    iqr_scale = float(q75 - q25) / 1.349  # IQR/1.349 ~ sigma for a Gaussian; recovers a scale when MAD ties to 0.
    return iqr_scale if iqr_scale > 1e-12 else 0.0


def _robust_lo_hi(x: np.ndarray) -> tuple[float, float]:
    """MAD-anchored [lo, hi] bounds = median +/- k*(1.4826*MAD) on the finite subset. The span tracks the CLEAN core even
    under heavy contamination (MAD ignores up to ~50% outliers), unlike a fixed inner-quantile trim which only excludes
    the tail once the trim fraction exceeds the contamination fraction."""
    xf = x[np.isfinite(x)]
    med = float(np.median(xf))
    scale = _robust_scale(xf, med)
    if scale <= 1e-12:
        # Degenerate core: fall back to the actual finite min/max so the caller still gets a usable (non-zero) span.
        return float(np.min(xf)), float(np.max(xf))
    return med - _ROBUST_AXIS_K * scale, med + _ROBUST_AXIS_K * scale


# ---------------------------------------------------------------------------
# Robust warp-coefficient FITTING (backlog idea #17, 2026-06-10)
# ---------------------------------------------------------------------------
# Huber tuning constant in units of the robust residual scale. 1.345*sigma is the
# textbook value giving ~95% asymptotic efficiency at the Gaussian model: residuals
# inside +/-1.345 robust-sigma keep unit weight (ordinary least squares on the clean
# core), residuals beyond it are down-weighted ~1/|r| so a handful of order-of-
# magnitude outlier rows can no longer dominate the squared-error loss and drag the
# fitted warp toward themselves.
_HUBER_C = 1.345
# IRLS controls: few iterations suffice (the OLS solution is already a good start and
# Huber-IRLS converges geometrically). Measured convergence on the clipped Chebyshev
# warp basis is 4-5 iterations (relative-coef tol 1e-4); cap at 6 with margin so the
# common contaminated case stops at convergence rather than burning the old 12-iter
# ceiling -- profiled 11.3ms -> ~6ms per fit at n=4000 with NO change to the fitted
# coefficients (the extra iters past ~5 only re-confirm the converged solution). This
# fit runs inside the per-pair FE search loop, so the halved iteration budget matters.
_IRLS_MAX_ITER = 6
_IRLS_TOL = 1e-4


def _robust_warp_fit_enabled() -> bool:
    """Default ON. ``MLFRAME_ROBUST_WARP_FIT=0`` forces the legacy OLS warp-fit path
    for replay / A-B compare. Independent of ``MLFRAME_ROBUST_AXIS`` (orthogonal knob:
    axis-scale robustness vs loss robustness)."""
    import os as _os
    flag = _os.environ.get("MLFRAME_ROBUST_WARP_FIT", "").strip().lower()
    return flag not in ("0", "false", "off", "no")


def _ols_lstsq(B: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Legacy ordinary-least-squares solve (kept under its own name per the
    keep-all-kernels rule). Returns the coefficient vector or ``None`` on failure /
    non-finite result. This is the EXACT call the prewarp / ALS fits used before the
    robust path; the heavy-tail dispatcher routes here on clean columns so the fit
    stays byte-identical to legacy."""
    try:
        coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(coef)):
        return None
    return np.ascontiguousarray(coef, dtype=np.float64)


def _robust_residual_scale(resid: np.ndarray) -> float:
    """MAD-based residual scale (1.4826*median|r - median(r)|). Contamination-proof up
    to ~50% outlier rows -- the right scale for the Huber threshold because the very
    rows we want to down-weight must NOT inflate the scale that decides who is an
    outlier (an ordinary std would be dragged up by the outliers and weaken the
    down-weighting). Falls back to std then a tiny floor on a degenerate residual."""
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    scale = 1.4826 * mad
    if scale > 1e-12:
        return scale
    s = float(np.std(resid))
    return s if s > 1e-12 else 1e-12


def _huber_irls_lstsq(B: np.ndarray, y: np.ndarray,
                      *, row_weight: np.ndarray | None = None) -> np.ndarray | None:
    """Outlier-robust drop-in for :func:`_ols_lstsq`: solve ``min_c sum rho_Huber((y -
    B c)/s)`` via iteratively-reweighted least squares.

    Each iteration computes residuals ``r = y - B c``, a robust residual scale ``s``
    (MAD, :func:`_robust_residual_scale`), then Huber weights ``w = 1`` where
    ``|r| <= _HUBER_C*s`` and ``w = _HUBER_C*s/|r|`` beyond -- a weighted least-squares
    re-solve with ``sqrt(w)`` row scaling. Rows in the clean core keep OLS weight; a
    handful of extreme outlier rows are down-weighted ~``1/|r|`` so they can no longer
    dominate the loss and the fitted warp tracks the bulk (the true relationship).

    ``row_weight`` (optional) is an EXTRA per-row weight multiplied into the Huber
    weight before each solve -- used by the rank-1 ALS sweep, which already scales rows
    by the partner factor ``g``/``f``. Passing it through keeps the ALS geometry while
    adding outlier robustness.

    Returns the coefficient vector, or ``None`` on a degenerate / non-finite solve
    (caller then falls back to the OLS result, which itself may be ``None``)."""
    B = np.ascontiguousarray(B, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if B.ndim != 2 or B.shape[0] != y.shape[0] or B.shape[0] < B.shape[1]:
        return None
    rw = None
    if row_weight is not None:
        rw = np.ascontiguousarray(row_weight, dtype=np.float64).reshape(-1)
        if rw.shape[0] != y.shape[0]:
            rw = None
    # Warm start from the OLS solution -- IRLS then only has to re-balance the few
    # outlier rows, converging in a handful of iterations.
    coef = _ols_lstsq(B if rw is None else B * rw[:, None],
                      y if rw is None else y * rw)
    if coef is None:
        return None
    prev = coef
    for _ in range(_IRLS_MAX_ITER):
        resid = y - B @ coef
        s = _robust_residual_scale(resid)
        thr = _HUBER_C * s
        aresid = np.abs(resid)
        w = np.ones_like(resid)
        beyond = aresid > thr
        if np.any(beyond):
            w[beyond] = thr / aresid[beyond]
        if rw is not None:
            w = w * rw
        sw = np.sqrt(np.maximum(w, 0.0))
        new = _ols_lstsq(B * sw[:, None], y * sw)
        if new is None:
            return prev  # degenerate re-solve -> keep last good coef
        if float(np.max(np.abs(new - coef))) <= _IRLS_TOL * (1.0 + float(np.max(np.abs(coef)))):
            coef = new
            break
        prev = coef
        coef = new
    if not np.all(np.isfinite(coef)):
        return None
    return np.ascontiguousarray(coef, dtype=np.float64)


def fit_basis_coef_robust(B: np.ndarray, y: np.ndarray, x_operand: np.ndarray,
                          *, row_weight: np.ndarray | None = None) -> tuple:
    """Dispatcher: fit warp coefficients ``c`` for ``y ~ B c`` robustly IFF the
    operand column ``x_operand`` is heavy-tailed / outlier-contaminated and the robust
    warp-fit gate is on; otherwise the byte-identical OLS solve.

    Returns ``(coef, robust_used: bool, winsor_bounds)`` where ``winsor_bounds`` is the
    MAD-anchored ``(lo, hi)`` of the operand when the robust path fired (stored in the
    recipe for leak-safe replay / provenance) or ``None`` on the OLS path. ``coef`` is
    ``None`` only if BOTH solvers fail.

    Gate rationale: ``_detect_heavy_tail`` is the SAME spike-contamination predicate
    the axis-normalisation path uses, so robust fitting fires exactly where the axis
    scale was already deemed corrupted -- a clean (lognormal-but-smooth, Gaussian,
    uniform) column does NOT trip it, so its warp is fit by the identical OLS call and
    is byte-identical to legacy."""
    use_robust = _robust_warp_fit_enabled() and _detect_heavy_tail(np.asarray(x_operand))
    if use_robust:
        coef = _huber_irls_lstsq(B, y, row_weight=row_weight)
        if coef is not None:
            return coef, True, _robust_lo_hi(np.asarray(x_operand))
    # Clean column, gate off, or robust solve failed -> legacy OLS (byte-identical).
    Bw = B if row_weight is None else B * np.asarray(row_weight, dtype=np.float64).reshape(-1)[:, None]
    yw = y if row_weight is None else y * np.asarray(row_weight, dtype=np.float64).reshape(-1)
    return _ols_lstsq(Bw, yw), False, None

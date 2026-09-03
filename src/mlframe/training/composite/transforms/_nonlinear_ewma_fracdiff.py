"""EWMA / fractional-differencing residual transforms carved out of ``nonlinear.py`` (itself
already carved out of ``mlframe.training.composite_transforms``) to keep that sibling under the
project's ~1000 LOC guideline.

Bound back into ``nonlinear.py``'s namespace via re-export at that module's bottom so historical
``from mlframe.training.composite.transforms.nonlinear import _ewma_kernel`` (and the
``composite_transforms`` public re-exports built on top of it) resolve transparently.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Optional,
)

import numpy as np

from ._domain_shared import residual_domain_reshaped

try:
    import numba as _numba
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _numba = None
    _HAS_NUMBA = False

logger = logging.getLogger("mlframe.training.composite_transforms")

# Parent-resident constants referenced as default-arg values in signatures below. Signature
# defaults evaluate at module load, so these MUST be top-level (a lazy in-body import wouldn't
# see them). The parent defines all three BEFORE its bottom-of-module sibling import, so this
# static cycle resolves at runtime. Whitelisted in tests/test_meta/test_no_import_cycles.py.
from . import (
    _EWMA_RESIDUAL_DEFAULT_K,
    _FRAC_DIFF_DEFAULT_D,
    _FRAC_DIFF_DEFAULT_LAGS,
)

# Module-level numba kernels (JIT compile on first call); pure-Python fallback is the in-line
# recursion below when numba is absent. Backend ladder: EWMA + frac-diff-inverse are
# LEFT-RECURRENT in row order (out[i] = f(out[i-1], ...)) so prange over rows is impossible; the
# win comes from a BATCHED kernel (K, N) parallelising across K specs while each row recurrence
# stays serial. CUDA RawKernel (one block per spec) tried and rejected: sequential per-thread
# recurrence is bandwidth-bound + host-device transfer kills it (5-100x SLOWER than njit at every
# size, see _benchmarks/_results/bench_ewma_frac_diff_backends_*.json). Two backends retained:
# single-spec njit (production) + parallel-batched njit.
if _HAS_NUMBA:

    @_numba.njit(cache=True)
    def _ewma_kernel(base_f: np.ndarray, alpha: float, anchor: float) -> np.ndarray:
        """v1 single-spec EWMA recurrence kernel; production default for K=1 path."""
        n = base_f.size
        out = np.empty(n, dtype=np.float64)
        state = anchor
        for i in range(n):
            x = base_f[i]
            if np.isfinite(x):
                state = (1.0 - alpha) * state + alpha * x
            out[i] = state
        return out

    @_numba.njit(cache=True, parallel=True)
    def _ewma_kernel_njit_par_batched(
        base_batch: np.ndarray, alphas: np.ndarray, anchors: np.ndarray,
    ) -> np.ndarray:
        """v2 batched EWMA: K specs in parallel via prange over the spec axis. Each row-recurrence inside one spec stays serial (left-recurrence). Bench: 2.7-3.8x over per-spec v1 at K>=10, N>=100k."""
        K, N = base_batch.shape
        out = np.empty((K, N), dtype=np.float64)
        for s in _numba.prange(K):
            state = anchors[s]
            a = alphas[s]
            for i in range(N):
                x = base_batch[s, i]
                if np.isfinite(x):
                    state = (1.0 - a) * state + a * x
                out[s, i] = state
        return out

    @_numba.njit(cache=True)
    def _frac_diff_inverse_kernel(
        t_f: np.ndarray, lags: int, weights: np.ndarray, anchor: float,
    ) -> np.ndarray:
        """v1 single-spec frac-diff-inverse recurrence kernel; production default for K=1 path."""
        n = t_f.size
        out = np.empty(n, dtype=np.float64)
        inv_w0 = 1.0 / weights[0]
        for i in range(n):
            lag_sum = 0.0
            upper = min(i + 1, lags + 1)
            for k_idx in range(1, upper):
                lag_sum += weights[k_idx] * out[i - k_idx]
            for k_idx in range(upper, lags + 1):
                lag_sum += weights[k_idx] * anchor
            out[i] = (t_f[i] - lag_sum) * inv_w0
        return out

    @_numba.njit(cache=True, parallel=True)
    def _frac_diff_inverse_kernel_njit_par_batched(
        t_batch: np.ndarray, lags: int, weights_batch: np.ndarray, anchors: np.ndarray,
    ) -> np.ndarray:
        """v2 batched frac-diff-inverse: K specs in parallel via prange. Each spec carries its own (weights, anchor) row; row-recurrence inside one spec stays serial. Bench: 3.8-5.4x over per-spec v1 at K>=10."""
        K, N = t_batch.shape
        out = np.empty((K, N), dtype=np.float64)
        for s in _numba.prange(K):
            inv_w0 = 1.0 / weights_batch[s, 0]
            anchor = anchors[s]
            for i in range(N):
                lag_sum = 0.0
                upper = min(i + 1, lags + 1)
                for k_idx in range(1, upper):
                    lag_sum += weights_batch[s, k_idx] * out[s, i - k_idx]
                for k_idx in range(upper, lags + 1):
                    lag_sum += weights_batch[s, k_idx] * anchor
                out[s, i] = (t_batch[s, i] - lag_sum) * inv_w0
        return out
else:
    _ewma_kernel = None
    _ewma_kernel_njit_par_batched = None
    _frac_diff_inverse_kernel = None
    _frac_diff_inverse_kernel_njit_par_batched = None


def _ewma_residual_fit(
    y: np.ndarray, base: np.ndarray, k: int = _EWMA_RESIDUAL_DEFAULT_K,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit stores only the EWMA half-life span ``k``; the EWMA itself is re-computed at forward / inverse time, keeping the fitted params JSON-serialisable and stateless (storing the full N-row EWMA trace would bloat metadata and break predict-on-new-data). The first-row anchor is the train-base mean: ``ewma[0] = mean(base_train)``."""
    # Lazy import: ``.predict`` re-imports this sibling at its bottom, so a top-level ``from .predict import ...`` would create a hard cycle the meta-test flags.
    k = max(1, int(k))
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    finite = _finite_mask if _finite_mask is not None else np.isfinite(base_f)
    anchor = float(np.mean(base_f[finite])) if finite.any() else 0.0
    # tail_anchor is the EWMA state at the LAST train row -- the recency-correct
    # seed for a predict batch that CONTINUES the training series. Opt-in via the
    # estimator's recurrence_continuation flag; the default mean ``anchor`` keeps
    # predict stateless (a fresh batch is not assumed to follow train).
    tail_anchor = anchor
    if finite.any():
        _trace = _ewma_compute(base_f, k, anchor)
        _tf = _trace[np.isfinite(_trace)]
        if _tf.size:
            tail_anchor = float(_tf[-1])
    return {"k": k, "anchor": anchor, "tail_anchor": tail_anchor}


def _ewma_anchor(params: dict[str, Any]) -> float:
    """Mean anchor by default; the train-tail state when the estimator opted into
    recurrence-continuation seeding (streaming a continuation of the train series)."""
    if params.get("recurrence_continuation") and "tail_anchor" in params:
        return float(params["tail_anchor"])
    return float(params["anchor"])
def _ewma_compute(base: np.ndarray, k: int, anchor: float) -> np.ndarray:
    """Exponentially-weighted moving average using ``alpha = 2 / (k + 1)``. Non-finite base values inherit the previous EWMA state (carry-forward), keeping the recursion well-defined on rows the upstream domain check did not yet flag. Single-spec public API; routes through :func:`_ewma_dispatch` so a future force-override or HW-tuned threshold can replace the default njit path without touching every caller. Numba kernel ~300x over pure Python on n=1M; pure-Python fallback otherwise."""
    base_f = np.ascontiguousarray(np.asarray(base, dtype=np.float64).reshape(-1))
    return _ewma_dispatch(base_f, float(k), float(anchor))


# EWMA / frac-diff-inverse backend dispatch. Crossover constants are measurement-derived (bench_ewma_frac_diff_backends.py on GTX 1050 Ti + i7-7700k): batched-parallel is a net win once K>=2 AND N>=50k; below that the prange spawn cost (~50us) overshoots the per-spec serial work. Module-level so kernel_tuning_cache can override via :func:`_lookup_ewma_backend` / :func:`_lookup_frac_diff_inv_backend`.
_EWMA_PAR_MIN_K: int = 2
_EWMA_PAR_MIN_N: int = 50_000
_FRAC_DIFF_INV_PAR_MIN_K: int = 2
_FRAC_DIFF_INV_PAR_MIN_N: int = 10_000


def _ewma_force_backend() -> str:
    """Read env-var override (``MLFRAME_EWMA_BACKEND=njit|njit_par``). Returns empty string when unset / unknown -- dispatcher then uses the size-based default."""
    import os
    v = os.environ.get("MLFRAME_EWMA_BACKEND", "").strip().lower()
    return v if v in ("njit", "njit_par") else ""


def _frac_diff_inv_force_backend() -> str:
    """Read env-var override (``MLFRAME_FRAC_DIFF_INV_BACKEND=njit|njit_par``). Returns empty string when unset / unknown -- dispatcher then uses the size-based default."""
    import os
    v = os.environ.get("MLFRAME_FRAC_DIFF_INV_BACKEND", "").strip().lower()
    return v if v in ("njit", "njit_par") else ""


def _lookup_ewma_backend(K: int, N: int) -> str:
    """Return ``"njit"`` or ``"njit_par"`` via the kernel_tuning_cache when available, else the measurement-backed size-based fallback (K=1 -> njit; K>=2 AND N>=50k -> njit_par). Cache key axes are (K, N); HW-tuned crossovers persist via the same pyutilz KernelTuningCache that powers joint_hist_batched."""
    forced = _ewma_force_backend()
    if forced:
        return forced
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache
        cache = get_kernel_tuning_cache()
        if cache is not None:
            choice = cache.lookup("ewma_dispatch", K=K, N=N)
            if choice is not None:
                bc = str(choice.get("backend_choice", "")).strip().lower()
                if bc in ("njit", "njit_par"):
                    return bc
    except Exception as e:
        logger.debug("swallowed exception in _nonlinear_ewma_fracdiff.py: %s", e)
        pass
    if K >= _EWMA_PAR_MIN_K and N >= _EWMA_PAR_MIN_N:
        return "njit_par"
    return "njit"


def _lookup_frac_diff_inv_backend(K: int, N: int) -> str:
    """Same contract as :func:`_lookup_ewma_backend` for the frac-diff-inverse kernel. Cache key: ``frac_diff_inverse_dispatch``."""
    forced = _frac_diff_inv_force_backend()
    if forced:
        return forced
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache
        cache = get_kernel_tuning_cache()
        if cache is not None:
            choice = cache.lookup("frac_diff_inverse_dispatch", K=K, N=N)
            if choice is not None:
                bc = str(choice.get("backend_choice", "")).strip().lower()
                if bc in ("njit", "njit_par"):
                    return bc
    except Exception as e:
        logger.debug("swallowed exception in _nonlinear_ewma_fracdiff.py: %s", e)
        pass
    if K >= _FRAC_DIFF_INV_PAR_MIN_K and N >= _FRAC_DIFF_INV_PAR_MIN_N:
        return "njit_par"
    return "njit"


def _ewma_dispatch(base_f: np.ndarray, k_param: float, anchor: float) -> np.ndarray:
    """Single-spec dispatcher: 1-D ``base_f`` shape (N,) -> EWMA(N,). Routes to the scalar njit kernel unless the env-var force-override picks ``njit_par`` (in which case the batched kernel runs with K=1 -- useful for A/B testing the par-batched path on a single spec; size-based default never picks njit_par for K=1)."""
    alpha = 2.0 / (k_param + 1.0)
    if not _HAS_NUMBA:
        out = np.empty(base_f.size, dtype=np.float64)
        state = anchor
        for i in range(base_f.size):
            x = base_f[i]
            if np.isfinite(x):
                state = (1.0 - alpha) * state + alpha * x
            out[i] = state
        return out
    backend = _lookup_ewma_backend(1, int(base_f.size))
    if backend == "njit_par":
        base_batch = base_f.reshape(1, -1)
        alphas = np.array([alpha], dtype=np.float64)
        anchors = np.array([anchor], dtype=np.float64)
        return np.asarray(_ewma_kernel_njit_par_batched(base_batch, alphas, anchors)[0])
    return np.asarray(_ewma_kernel(base_f, alpha, float(anchor)))


def _ewma_compute_batched(
    base_batch: np.ndarray, ks: np.ndarray, anchors: np.ndarray,
) -> np.ndarray:
    """Batched public API: run K independent EWMA specs on a (K, N) base matrix and return the (K, N) EWMA result. Each row carries its own ``k`` (half-life) and ``anchor`` (state-zero value). When K>=2 AND N is sufficiently large the parallel-batched njit kernel kicks in -- routed through :func:`_lookup_ewma_backend` so HW-tuned thresholds persist via kernel_tuning_cache. Bench: 2.7-3.8x over per-spec dispatch at K>=10, N>=100k; callers evaluating many EWMA specs on the same series (e.g. cross-target discovery scanning k in [3, 5, 7, 14, 21]) should batch via this entry point."""
    base_batch = np.ascontiguousarray(np.asarray(base_batch, dtype=np.float64))
    if base_batch.ndim == 1:
        base_batch = base_batch.reshape(1, -1)
    K, N = base_batch.shape
    ks_a = np.ascontiguousarray(np.asarray(ks, dtype=np.float64).reshape(-1))
    anchors_a = np.ascontiguousarray(np.asarray(anchors, dtype=np.float64).reshape(-1))
    if ks_a.size != K or anchors_a.size != K:
        raise ValueError(f"_ewma_compute_batched: ks shape {ks_a.shape} and anchors shape {anchors_a.shape} must each equal (K={K},)")
    alphas = 2.0 / (ks_a + 1.0)
    if not _HAS_NUMBA:
        out = np.empty((K, N), dtype=np.float64)
        for s in range(K):
            state = float(anchors_a[s])
            a = float(alphas[s])
            for i in range(N):
                x = float(base_batch[s, i])
                if np.isfinite(x):
                    state = (1.0 - a) * state + a * x
                out[s, i] = state
        return out
    backend = _lookup_ewma_backend(K, N)
    if backend == "njit_par":
        return np.asarray(_ewma_kernel_njit_par_batched(base_batch, alphas, anchors_a))
    out = np.empty((K, N), dtype=np.float64)
    for s in range(K):
        out[s] = _ewma_kernel(
            np.ascontiguousarray(base_batch[s]), float(alphas[s]), float(anchors_a[s]),
        )
    return out


def _frac_diff_inverse_compute(
    t_f: np.ndarray, lags: int, weights: np.ndarray, anchor: float,
) -> np.ndarray:
    """Single-spec frac-diff-inverse public API; routes through :func:`_frac_diff_inverse_dispatch`."""
    t_f = np.ascontiguousarray(np.asarray(t_f, dtype=np.float64).reshape(-1))
    weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float64).reshape(-1))
    return _frac_diff_inverse_dispatch(t_f, int(lags), weights, float(anchor))


def _frac_diff_inverse_dispatch(
    t_f: np.ndarray, lags: int, weights: np.ndarray, anchor: float,
) -> np.ndarray:
    """Single-spec dispatcher (1-D in, 1-D out). Default routes to scalar njit kernel; env-var force-override or KTC entry can pick the par-batched path with K=1."""
    if not _HAS_NUMBA:
        n = t_f.size
        out = np.empty(n, dtype=np.float64)
        inv_w0 = 1.0 / weights[0]
        for i in range(n):
            lag_sum = 0.0
            upper = min(i + 1, lags + 1)
            for k_idx in range(1, upper):
                lag_sum += weights[k_idx] * out[i - k_idx]
            for k_idx in range(upper, lags + 1):
                lag_sum += weights[k_idx] * anchor
            out[i] = (t_f[i] - lag_sum) * inv_w0
        return out
    backend = _lookup_frac_diff_inv_backend(1, int(t_f.size))
    if backend == "njit_par":
        t_batch = t_f.reshape(1, -1)
        weights_batch = weights.reshape(1, -1)
        anchors = np.array([anchor], dtype=np.float64)
        return np.asarray(_frac_diff_inverse_kernel_njit_par_batched(t_batch, lags, weights_batch, anchors)[0])
    return np.asarray(_frac_diff_inverse_kernel(t_f, lags, weights, anchor))


def _frac_diff_inverse_compute_batched(
    t_batch: np.ndarray, lags: int, weights_batch: np.ndarray, anchors: np.ndarray,
) -> np.ndarray:
    """Batched public API: K independent frac-diff-inverse specs on a (K, N) t_hat matrix. ``weights_batch`` is (K, lags+1), ``anchors`` is (K,). Bench: 3.8-5.4x over per-spec dispatch at K>=10."""
    t_batch = np.ascontiguousarray(np.asarray(t_batch, dtype=np.float64))
    if t_batch.ndim == 1:
        t_batch = t_batch.reshape(1, -1)
    K, N = t_batch.shape
    weights_batch = np.ascontiguousarray(np.asarray(weights_batch, dtype=np.float64))
    if weights_batch.ndim == 1:
        weights_batch = np.tile(weights_batch, (K, 1))
    anchors_a = np.ascontiguousarray(np.asarray(anchors, dtype=np.float64).reshape(-1))
    if anchors_a.size != K or weights_batch.shape[0] != K:
        raise ValueError(
            f"_frac_diff_inverse_compute_batched: anchors shape {anchors_a.shape} and weights_batch shape {weights_batch.shape} must each have K={K} rows"
        )
    if not _HAS_NUMBA:
        out = np.empty((K, N), dtype=np.float64)
        for s in range(K):
            anchor = float(anchors_a[s])
            inv_w0 = 1.0 / float(weights_batch[s, 0])
            for i in range(N):
                lag_sum = 0.0
                upper = min(i + 1, lags + 1)
                for k_idx in range(1, upper):
                    lag_sum += float(weights_batch[s, k_idx]) * float(out[s, i - k_idx])
                for k_idx in range(upper, lags + 1):
                    lag_sum += float(weights_batch[s, k_idx]) * anchor
                out[s, i] = (float(t_batch[s, i]) - lag_sum) * inv_w0
        return out
    backend = _lookup_frac_diff_inv_backend(K, N)
    if backend == "njit_par":
        return np.asarray(_frac_diff_inverse_kernel_njit_par_batched(t_batch, lags, weights_batch, anchors_a))
    out = np.empty((K, N), dtype=np.float64)
    for s in range(K):
        out[s] = _frac_diff_inverse_kernel(
            np.ascontiguousarray(t_batch[s]), lags,
            np.ascontiguousarray(weights_batch[s]),
            float(anchors_a[s]),
        )
    return out
def _ewma_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Apply ``T = y - EWMA_k(base)``, recomputing the EWMA trace from the fitted ``k``/mean-anchor at call time."""
    return np.asarray(np.asarray(y, dtype=np.float64) - _ewma_compute(
        base, int(params["k"]), float(params["anchor"]),
    ))
def _ewma_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Undo the transform: ``y = T_hat + EWMA_k(base)``, using :func:`_ewma_anchor` to pick the mean- vs tail-anchor seed."""
    return np.asarray(np.asarray(t_hat, dtype=np.float64) + _ewma_compute(
        base, int(params["k"]), _ewma_anchor(params),
    ))
_ewma_residual_domain: Callable[[Optional[np.ndarray], np.ndarray], np.ndarray] = residual_domain_reshaped
def _rolling_median_pandas(arr_f: np.ndarray, k: int) -> np.ndarray:
    """Reference centred rolling median: pandas ``rolling(window=k, center=True, min_periods=1).median()``. This is the CONTRACT both backends reproduce. ``arr_f`` must already be float64 / 1-D / non-empty; ``k`` already clamped to ``>= 1``."""
    import pandas as pd  # lazy
    return np.asarray(pd.Series(arr_f).rolling(window=k, center=True, min_periods=1).median().to_numpy())


def _rolling_median(arr: np.ndarray, k: int) -> np.ndarray:
    """Centred rolling median with truncation at boundaries.

    Reference semantics (the cross-environment CONTRACT) = pandas ``rolling(window=k, center=True, min_periods=1).median()``: position ``i`` is the median of ``arr[i - k//2 .. i + (k-1)//2]`` clipped to ``[0, n-1]`` (so head/tail windows truncate, and NaN cells inside a window are SKIPPED, never poisoning the window).

    Fast path: ``bottleneck.move_median`` (forward-window O(n log k) quickselect; ~8-10x faster than pandas at k in [7, 21] on n=100k) re-centred to that contract. The forward window ending at index ``j`` is the centred window for ``i = j - (k-1)//2``, so the correct LEFT shift is ``(k-1)//2`` (NOT ``k//2`` -- the historic ``k//2`` shift was off-by-one for every EVEN ``k``). Head positions and tail positions whose centred window would run past the array end carry directly-computed truncated medians (the historic code constant-filled the tail with the last full-window median -- wrong for both even and odd ``k``). ``move_median`` also REQUIRES ``window <= n``, so ``k`` is clamped to ``min(k, n)`` for the kernel call (a centred window wider than the array is identical to ``k = n``; the historic code passed ``k > n`` straight through and ``move_median`` raised, silently dropping the whole result to the non-finite fallback).

    NaN parity: ``bottleneck.move_median`` does NOT skip NaN inside a window (one NaN poisons the window to NaN), whereas pandas' ``min_periods=1`` median skips them. So the fast path is bit-identical to the pandas contract ONLY when the input is all-finite; non-finite input routes to the pandas reference to preserve identical results regardless of whether bottleneck is installed. (The downstream callers domain-check ``base`` finite, so the all-finite fast path is the common case.) After either path, any residual NaN (an entirely-non-finite window under the pandas route) is replaced with the row's own value (or 0.0 if also non-finite) to match the legacy fallback.
    """
    arr_f = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr_f.size == 0:
        return np.asarray(arr_f.copy())
    n = arr_f.size
    k = max(1, int(k))
    out: np.ndarray | None = None
    # Fast path requires all-finite input (move_median can't NaN-skip within a
    # window the way pandas does); otherwise fall through to the pandas reference
    # so results are identical across environments.
    if np.isfinite(arr_f).all():
        try:
            import bottleneck as _bn  # lazy; optional dep but present in mlframe[all]
            k_eff = min(k, n)  # move_median requires 1 <= window <= n
            _fwd = _bn.move_median(arr_f, window=k_eff, min_count=1)
            _shift = (k - 1) // 2  # forward index j = i + (k-1)//2 (NOT k//2)
            _left = k // 2
            out = np.empty(n, dtype=np.float64)
            # Interior positions ``i`` carry the full ``k_eff`` window AND are
            # forward-readable: i >= _left, i + _shift >= k_eff - 1 (full kernel
            # window), i + _shift <= n - 1 (in range). These are a single
            # vectorised slice of the kernel output -- no per-row Python work.
            lo_i = max(_left, k_eff - 1 - _shift)
            hi_i = n - 1 - _shift
            if hi_i >= lo_i:
                out[lo_i : hi_i + 1] = _fwd[lo_i + _shift : hi_i + _shift + 1]
            # Boundary positions (head + tail, O(k) of them): centred window
            # truncates to the array; compute its median directly to match
            # pandas exactly (the historic constant tail-fill was wrong).
            for i in range(0, min(max(lo_i, 0), n)):
                lo = i - _left if i - _left > 0 else 0
                hi = i + _shift if i + _shift < n - 1 else n - 1
                out[i] = np.median(arr_f[lo : hi + 1])
            for i in range(max(hi_i + 1, 0), n):
                lo = i - _left if i - _left > 0 else 0
                hi = i + _shift if i + _shift < n - 1 else n - 1
                out[i] = np.median(arr_f[lo : hi + 1])
        except ImportError:
            out = None
    if out is None:
        out = _rolling_median_pandas(arr_f, k)
    bad = ~np.isfinite(out)
    if bad.any():
        fallback = np.where(np.isfinite(arr_f), arr_f, 0.0)
        out = np.where(bad, fallback, out)
    return out
def _frac_diff_weights(d: float, lags: int) -> np.ndarray:
    """Truncated weight series for (1 - L)^d expansion."""
    lags = max(1, int(lags))
    w = np.empty(lags + 1, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, lags + 1):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w
def _frac_diff_fit(
    y: np.ndarray, base: np.ndarray,
    d: float = _FRAC_DIFF_DEFAULT_D, lags: int = _FRAC_DIFF_DEFAULT_LAGS,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Store fractional order ``d``, lag truncation ``lags``, and the train-y mean used as a pre-window anchor (rows whose lag history is shorter than ``lags`` need a fallback value for the missing past terms)."""
    # Lazy import: ``.predict`` re-imports this sibling at its bottom, so a top-level ``from .predict import ...`` would create a hard cycle the meta-test flags.
    d = float(d)
    lags = max(1, int(lags))
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = _finite_mask if _finite_mask is not None else np.isfinite(y_f)
    anchor = float(np.mean(y_f[finite])) if finite.any() else 0.0
    # tail_anchor pads the pre-window history from the LAST ``lags`` train rows
    # (recency-correct seed for a continuation predict batch) instead of the
    # whole-train mean. Opt-in via recurrence_continuation; default stays mean.
    tail_anchor = anchor
    _yt = y_f[finite]
    if _yt.size:
        tail_anchor = float(np.mean(_yt[-lags:]))
    return {
        "d": d, "lags": lags, "anchor": anchor, "tail_anchor": tail_anchor,
        "weights": _frac_diff_weights(d, lags).tolist(),
    }
def _frac_diff_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """T_i = sum_{k=0}^{lags} w_k * y_{i-k}, padding y_{i-k} with the train anchor for k > i. Vectorised via ``np.convolve(y_padded, weights, 'valid')`` after left-padding ``y`` with ``lags`` copies of the train anchor (~340x over the nested Python loop on n=1M, lags=30)."""
    lags = int(params["lags"])
    weights = np.asarray(params["weights"], dtype=np.float64)
    anchor = float(params["anchor"])
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_f.size == 0:
        return np.asarray(y_f.copy())
    y_padded = np.concatenate([np.full(lags, anchor, dtype=np.float64), y_f])
    return np.convolve(y_padded, weights, mode="valid")
def _frac_diff_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Invert: T_i = w_0 * y_i + sum_{k=1}^{lags} w_k * y_{i-k}, so y_i = (T_i - sum_{k=1}^{lags} w_k * y_{i-k}) / w_0. w_0 == 1 by construction. Past y values are unknown at predict, so we ITERATIVELY reconstruct them: y_0 from T_0 + lag-anchors, y_1 from T_1 + y_0 + lag-anchors, etc. Routes through :func:`_frac_diff_inverse_compute` -> :func:`_frac_diff_inverse_dispatch` so kernel_tuning_cache + env-var force-override choose the backend; default keeps the scalar njit kernel (~260x over pure Python on n=1M, lags=30)."""
    lags = int(params["lags"])
    weights = np.ascontiguousarray(np.asarray(params["weights"], dtype=np.float64))
    anchor = _ewma_anchor(params)  # mean by default, train-tail when opted in
    t_f = np.ascontiguousarray(np.asarray(t_hat, dtype=np.float64).reshape(-1))
    return _frac_diff_inverse_compute(t_f, lags, weights, anchor)
def _frac_diff_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Frac-diff is y-only; base is accepted for signature uniformity but never read. Domain when y is present is finite-y only (a non-finite UNUSED base must not drop rows and compact the y sequence). The base-finite mask is kept solely for the ``y is None`` predict-side call."""
    if y is None:
        return np.asarray(np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1)))
    return np.asarray(np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1)))

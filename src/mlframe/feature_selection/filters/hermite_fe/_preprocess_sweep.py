"""Preprocess (z-score / min-max / shift) replays and their per-host kernel-tuning sweep, split out of ``hermite_fe``."""

from __future__ import annotations

import logging

from typing import cast
import numpy as np
from .._fe_usability_signal import _crit_np_dtype
from mlframe.feature_selection.filters._safe_scale import _REL_TOL, guarded_scale
from pyutilz.performance.kernel_tuning.registry import kernel_tuner
from ._hermite_robust import _detect_heavy_tail, _robust_axis_enabled, _robust_lo_hi

try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    # No-op decorators so the file imports without numba.
    from mlframe._numba_fallback import njit

    def prange(n):
        """No-numba fallback: plain ``range``."""
        return range(n)


logger = logging.getLogger("mlframe.feature_selection.filters.hermite_fe")


def _preprocess_zscore(x):
    """Standardize ``x`` to z-scores; heavy-tailed columns use robust median/inner-quantile-range center/scale with a +/-6-sigma clip instead of the raw mean/std."""
    if _robust_axis_enabled() and _detect_heavy_tail(x):
        xf = x[np.isfinite(x)]
        # Robust centre/scale from the inner-quantile core; map outliers but CLAMP to the Hermite working domain so a
        # +/-1000 spike lands at the edge rather than producing a huge He_n value that inflates MI and breaks shift-stability.
        center = float(np.median(xf))
        lo, hi = _robust_lo_hi(x)
        std = float((hi - lo) / 6.0)  # inner-quantile range ~ 6 sigma for a Gaussian core; matches z-score scale.
        std = std if std > 1e-12 else float(guarded_scale(np.std(xf), np.abs(xf).max()))
        clip = 6.0  # +/-6 robust sigma covers the trimmed core; clipped extremes pin to the working-domain edge.
        z = np.clip((x - center) / std, -clip, clip)
        return z, dict(mean=center, std=std, clip=clip)
    mean = float(np.mean(x))
    std = float(guarded_scale(np.std(x), np.abs(x).max()))
    return (x - mean) / std, dict(mean=mean, std=std)


@njit(cache=True)
def _minmax_neg1_1_njit(x: np.ndarray):
    """Fused min/max + [-1, 1] rescale for the clean (non-heavy-tail) min-max preprocessor. One reduction pass + one
    output pass instead of numpy's np.min + np.max + elementwise (three full passes) on the 1M-row operand. Bit-identical
    to the numpy body: np.min/np.max are order-independent (loop reproduces them exactly, INCLUDING NaN propagation via
    the ``nan_seen`` poison so a NaN column yields the same all-NaN z numpy would), and the per-element expression keeps
    numpy's exact operator order ``2*(x-lo)/span - 1``."""
    n = x.size
    lo = np.inf
    hi = -np.inf
    nan_seen = False
    for i in range(n):
        v = x[i]
        if v != v:  # NaN
            nan_seen = True
        else:
            if v < lo:
                lo = v
            if v > hi:
                hi = v
    if nan_seen:
        lo = np.nan
        hi = np.nan
    span = hi - lo
    mag = abs(lo) if abs(lo) > abs(hi) else abs(hi)
    if not (span > _REL_TOL * mag):  # a NaN span falls here too and still propagates through lo below
        span = 1.0
    # Intentional float64: coefficient / output contract kept f64
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = 2.0 * (x[i] - lo) / span - 1.0
    return out, lo, hi


def _preprocess_minmax_neg1_1(x):
    """Rescale ``x`` onto ``[-1, 1]``; heavy-tailed columns use robust inner-quantile bounds with clamping to +/-1 instead of the raw min/max."""
    if _robust_axis_enabled() and _detect_heavy_tail(x):
        # Min-max onto [-1, 1] from the inner-quantile bounds; clamp so clipped outliers pin to +/-1 (the basis domain edge)
        # instead of compressing the core toward 0. clip is implied (the [-1, 1] clamp), recorded so replay matches.
        lo, hi = _robust_lo_hi(x)
        span = float(guarded_scale(hi - lo, max(abs(lo), abs(hi))))
        z = np.clip(2 * (x - lo) / span - 1, -1.0, 1.0)
        return z, dict(lo=lo, hi=hi, clip=1.0)
    z, lo, hi = _minmax_neg1_1_njit(np.ascontiguousarray(x, dtype=_crit_np_dtype()))
    return z, dict(lo=float(lo), hi=float(hi))


def _preprocess_shift_nonneg(x):
    """Shift ``x`` to be non-negative (Laguerre domain); heavy-tailed columns clamp the upper tail to the robust inner-quantile range instead of the raw max, avoiding an exploding ``L_n`` argument."""
    if _robust_axis_enabled() and _detect_heavy_tail(x):
        # Shift the inner-quantile lower bound to ~0 and clamp the upper tail to the inner-quantile range so a positive
        # spike does not push the Laguerre argument far out where L_n explodes. Upper clamp recorded for replay.
        lo, hi = _robust_lo_hi(x)
        upper = float(hi - lo)
        z = np.clip(x - lo + 1e-9, 0.0, upper + 1e-9)
        return z, dict(lo=lo, clip=upper)
    lo = float(np.min(x))
    return x - lo + 1e-9, dict(lo=lo)


def _apply_zscore(x, params):
    """Replay a fitted ``_preprocess_zscore`` transform onto new data from its stored ``mean``/``std``/(optional)``clip`` params."""
    z = (x - params["mean"]) / params["std"]
    clip = params.get("clip")
    if clip is not None:
        z = np.clip(z, -float(clip), float(clip))
    return z


@njit(cache=True, parallel=True)
def _apply_minmax_njit(x: np.ndarray, lo: float, span: float, has_clip: bool, clip_val: float) -> np.ndarray:
    """Fused single-pass replay of ``2*(x-lo)/span - 1`` (+ optional symmetric clip), one prange pass instead
    of the numpy form's separate subtract/multiply/divide/subtract (+ clip) temp-array passes. Op order
    matches the numpy expression exactly (``2*(x-lo)/span - 1``, not a reciprocal-multiply rewrite) for
    bit-identical FP rounding.

    Only wins over the serial twin (:func:`_apply_minmax_njit_serial`) at extreme n (>~10-20M) -- see
    :func:`_apply_minmax`'s dispatch threshold for the measured crossover; this variant stays for that
    regime, but is NOT the default entry point any more."""
    n = x.shape[0]
    # Intentional float64: coefficient / output contract kept f64
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        z = 2 * (x[i] - lo) / span - 1
        if has_clip:
            if z > clip_val:
                z = clip_val
            elif z < -clip_val:
                z = -clip_val
        out[i] = z
    return out


@njit(cache=True, parallel=False)
def _apply_minmax_njit_serial(x: np.ndarray, lo: float, span: float, has_clip: bool, clip_val: float) -> np.ndarray:
    """Serial twin of :func:`_apply_minmax_njit` -- identical body, ``range`` instead of ``prange``.

    ``parallel=True`` pays a fixed per-call thread-pool dispatch cost that a 2-flop-plus-optional-clip
    elementwise op cannot amortize at any n this codebase's FE search realistically reaches. Measured
    (same-process A/B, warm, best-of-50, ``bench_apply_minmax_parallel_vs_serial.py``): serial is
    2.4x-28x FASTER at n<=5,000,000; the crossover only appears between n=5M (parallel 2.41x SLOWER)
    and n=20M (parallel 1.56x faster) -- see :func:`_apply_minmax`'s threshold for where this is used."""
    n = x.shape[0]
    # Intentional float64: coefficient / output contract kept f64
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        z = 2 * (x[i] - lo) / span - 1
        if has_clip:
            if z > clip_val:
                z = clip_val
            elif z < -clip_val:
                z = -clip_val
        out[i] = z
    return out


_MINMAX_SWEEP_N = [1_000, 1_000_000, 4_000_000]


_MINMAX_SALT = 1


def _make_minmax_inputs(dims: dict) -> tuple:
    """(x, lo, span, has_clip, clip_val) at the sweep's n cell."""
    n = int(dims["n"])
    rng = np.random.default_rng(0)
    # Intentional float64: benchmark / warmup input kept f64
    x = np.ascontiguousarray(rng.random(n) * 10.0, dtype=np.float64)
    return (x, 0.0, 10.0, True, 1.0)


def _run_minmax_sweep() -> list:
    """Serial-vs-parallel wall-clock sweep over the n grid -> kernel_tuning_cache regions."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {
        "serial": lambda *a: _apply_minmax_njit_serial(*a),
        "parallel": lambda *a: _apply_minmax_njit(*a),
    }
    return cast(
        list,
        sweep_backend_grid(
            variants,
            {"n": _MINMAX_SWEEP_N},
            _make_minmax_inputs,
            reference="serial",
            repeats=5,
            equiv_atol=0.0,
            equiv_rtol=0.0,
        ),
    )


def _minmax_fallback_choice(n: int) -> str:
    """Pre-sweep / tuner-failure fallback: parallel above the dev-box-measured n crossover (see the
    module comment above for the confirmed-loss/confirmed-win bracket)."""
    return "parallel" if int(n) >= 10_000_000 else "serial"


_MINMAX_PARALLELISM_SPEC = kernel_tuner(
    kernel_name="hermite_minmax_kernel_parallelism",
    variant_fns=(_apply_minmax_njit_serial, _apply_minmax_njit),
    tuner=_run_minmax_sweep,
    axes={"n": _MINMAX_SWEEP_N},
    fallback=_minmax_fallback_choice,
    gpu_capable=False,
    salt=_MINMAX_SALT,
    cli_label="hermite_minmax_kernel_parallelism",
)


def _apply_minmax(x, params):
    """Replay a fitted ``_preprocess_minmax_neg1_1`` transform onto new data from its stored ``lo``/``hi``/(optional)``clip`` params."""
    span = float(guarded_scale(params["hi"] - params["lo"], max(abs(params["lo"]), abs(params["hi"]))))
    clip = params.get("clip")
    if _NUMBA_AVAILABLE:
        xf = np.ascontiguousarray(x, dtype=_crit_np_dtype())
        try:
            choice = _MINMAX_PARALLELISM_SPEC.choose(n=int(xf.shape[0]))
        except Exception as e:
            logger.warning("hermite_minmax_kernel_parallelism choose() failed, using the size-based fallback: %s", e)
            choice = _minmax_fallback_choice(int(xf.shape[0]))
        fn = _apply_minmax_njit if choice == "parallel" else _apply_minmax_njit_serial
        return fn(xf, float(params["lo"]), float(span), clip is not None, float(clip) if clip is not None else 0.0)
    z = 2 * (x - params["lo"]) / span - 1
    if clip is not None:
        z = np.clip(z, -float(clip), float(clip))
    return z


def _apply_shift(x, params):
    """Replay a fitted ``_preprocess_shift_nonneg`` transform onto new data from its stored ``lo``/(optional)``clip`` params."""
    z = x - params["lo"] + 1e-9
    clip = params.get("clip")
    if clip is not None:
        z = np.clip(z, 0.0, float(clip) + 1e-9)
    return z

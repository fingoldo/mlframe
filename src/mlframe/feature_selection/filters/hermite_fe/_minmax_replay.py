"""Replay of a fitted min-max (-1, 1) preprocessing on new data: serial and parallel njit kernels and the per-host choice between them (kernel_tuning_cache sweep)."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from mlframe.feature_selection.filters._safe_scale import guarded_scale
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    from mlframe._numba_fallback import njit

    def prange(n):
        """No-numba fallback: plain ``range``."""
        return range(n)

logger = logging.getLogger(__name__)


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


# Per-host serial/parallel crossover via the canonical kernel_tuning_cache (NO hardcoded threshold --
# feedback_use_kernel_tuning_cache_for_gpu / feedback_fastest_default_with_dispatch). Dev-box measurement
# found the crossover between n=5M (parallel 2.41x SLOWER) and n=20M (parallel 1.56x faster) -- the
# fallback threshold below (used only pre-sweep / on tuner failure) is deliberately close to the loss
# side for that reason, but the REAL per-host decision comes from the tuner's measured sweep.
# Capped at 4M rows, as the other row sweeps: a 15M-row cell repeats per variant for seconds of work the crossover does not need.
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

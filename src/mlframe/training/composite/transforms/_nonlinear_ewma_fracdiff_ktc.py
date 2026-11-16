"""Kernel-tuning entries of the EWMA and fractional-difference-inverse backends (``ewma_dispatch``, ``frac_diff_inverse_dispatch``).

Both kernels run K independent specs over N rows either as a loop of scalar njit calls (``njit``) or as one parallel-over-specs njit call (``njit_par``). The dispatchers read the cache keys
``ewma_dispatch`` and ``frac_diff_inverse_dispatch`` for years while nothing ever wrote them, so the size rule (K >= 2 and N >= 50000 / 10000) always decided. Here each choice is a registered
``kernel_tuner`` entry whose sweep times the two backends over a (K, N) grid at identical outputs; the size rule stays as the fallback until the sweep has run on the host.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

_SWEEP_K = [1, 4, 16]
_SWEEP_N = [5_000, 50_000, 500_000]
_LAGS = 12
_SALT = 1


def _ewma_inputs(dims: dict) -> tuple:
    """``(base (K, N), alphas, anchors)`` for the EWMA backends."""
    rng = np.random.default_rng(0)
    K, N = int(dims["K"]), int(dims["N"])
    base = rng.standard_normal((K, N))
    return (base, 2.0 / (rng.uniform(2.0, 30.0, K) + 1.0), np.zeros(K))


def _ewma_njit(base: np.ndarray, alphas: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """The scalar kernel looped over the specs."""
    from ._nonlinear_ewma_fracdiff import _ewma_kernel

    out = np.empty_like(base)
    for s in range(base.shape[0]):
        out[s] = _ewma_kernel(np.ascontiguousarray(base[s]), float(alphas[s]), float(anchors[s]))
    return out


def _ewma_njit_par(base: np.ndarray, alphas: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """The parallel-over-specs kernel."""
    from ._nonlinear_ewma_fracdiff import _ewma_kernel_njit_par_batched

    return np.asarray(_ewma_kernel_njit_par_batched(base, alphas, anchors))


def _frac_inputs(dims: dict) -> tuple:
    """``(t (K, N), lags, weights (K, lags+1), anchors)`` for the fractional-difference-inverse backends."""
    rng = np.random.default_rng(0)
    K, N = int(dims["K"]), int(dims["N"])
    w = np.tile(np.concatenate(([1.0], -rng.uniform(0.01, 0.2, _LAGS))), (K, 1))
    return (rng.standard_normal((K, N)), _LAGS, w, np.zeros(K))


def _frac_njit(t: np.ndarray, lags: int, weights: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """The scalar kernel looped over the specs."""
    from ._nonlinear_ewma_fracdiff import _frac_diff_inverse_kernel

    out = np.empty_like(t)
    for s in range(t.shape[0]):
        out[s] = _frac_diff_inverse_kernel(np.ascontiguousarray(t[s]), lags, np.ascontiguousarray(weights[s]), float(anchors[s]))
    return out


def _frac_njit_par(t: np.ndarray, lags: int, weights: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """The parallel-over-specs kernel."""
    from ._nonlinear_ewma_fracdiff import _frac_diff_inverse_kernel_njit_par_batched

    return np.asarray(_frac_diff_inverse_kernel_njit_par_batched(t, lags, weights, anchors))


def _ewma_sweep() -> list:
    """Time the two EWMA backends over the (K, N) grid at identical outputs."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return cast(list, sweep_backend_grid({"njit": _ewma_njit, "njit_par": _ewma_njit_par}, {"K": _SWEEP_K, "N": _SWEEP_N}, _ewma_inputs, reference="njit", repeats=3, equiv_rtol=1e-12, equiv_atol=1e-12))


def _frac_sweep() -> list:
    """Time the two fractional-difference-inverse backends over the (K, N) grid at identical outputs."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return cast(list, sweep_backend_grid({"njit": _frac_njit, "njit_par": _frac_njit_par}, {"K": _SWEEP_K, "N": _SWEEP_N}, _frac_inputs, reference="njit", repeats=3, equiv_rtol=1e-12, equiv_atol=1e-12))


def _ewma_fallback(K: int, N: int) -> str:
    """Size rule before any sweep (the measurement-backed rule the dispatcher had)."""
    return "njit_par" if int(K) >= 2 and int(N) >= 50_000 else "njit"


def _frac_fallback(K: int, N: int) -> str:
    """Size rule before any sweep."""
    return "njit_par" if int(K) >= 2 and int(N) >= 10_000 else "njit"


def _register(name: str, tuner: Any, fallback: Any) -> Any:
    """Register one ``kernel_tuner`` entry; ``None`` when the registry is unavailable."""
    try:
        from pyutilz.performance.kernel_tuning.registry import kernel_tuner

        return kernel_tuner(kernel_name=name, variant_fns=(), tuner=tuner, axes={"K": list(_SWEEP_K), "N": list(_SWEEP_N)}, fallback=fallback, gpu_capable=False, salt=_SALT, cli_label=name)
    except Exception as e:  # best-effort: without a tuner entry the size rule decides
        logger.warning("kernel_tuner registration of %s failed: %s", name, e)
        return None


_EWMA_SPEC: Any = _register("ewma_dispatch", _ewma_sweep, _ewma_fallback)
_FRAC_SPEC: Any = _register("frac_diff_inverse_dispatch", _frac_sweep, _frac_fallback)


def tuned_backend(spec: Any, fallback: Any, K: int, N: int) -> str:
    """``"njit"`` or ``"njit_par"`` from the per-host tuning cache, else the size rule."""
    if spec is None:
        return cast(str, fallback(K, N))
    try:
        choice = str(spec.choose(K=int(K), N=int(N)))
        if choice in ("njit", "njit_par"):
            return choice
    except Exception as exc:
        logger.debug("tuned_backend: choose() failed (%s); size rule", exc)
    return cast(str, fallback(K, N))

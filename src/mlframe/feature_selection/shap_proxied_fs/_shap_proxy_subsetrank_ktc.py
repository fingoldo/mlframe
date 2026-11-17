"""Kernel-tuning entry of the SHAP-proxy subset-rank device choice (``shap_proxy_subsetrank``).

``brute_force_top_n_dispatch`` takes the GPU kernel only past a subset-count crossover. The dispatcher read the cache key ``shap_proxy_subsetrank`` (field ``gpu_min_subsets``) while nothing wrote
it, so the constant ``_DEFAULT_GPU_MIN_SUBSETS`` always decided. Here the choice is a registered ``kernel_tuner`` entry whose sweep times the CPU incremental kernel and the GPU kernel over a
subset-count grid and keeps the GPU only where it wins at identical top-N losses (the two are bit-identical by construction). The constant stays the fallback until the sweep has run on the host.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_SWEEP_SUBSETS = [16_383, 65_535, 262_143, 1_048_575]  # 2**f - 1 subsets of f = 14, 16, 18, 20 features
_ROWS = 2_000
_TOP_N = 30
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(phi, base, y)`` with the feature count whose full subset lattice has ``dims["n_subsets"]`` members."""
    rng = np.random.default_rng(0)
    f = round(np.log2(int(dims["n_subsets"]) + 1))
    phi = rng.standard_normal((_ROWS, f)).astype(np.float64)  # Intentional float64: the kernels take float64 SHAP values
    base = np.zeros(_ROWS, dtype=np.float64)  # Intentional float64: margin base
    y = (rng.random(_ROWS) < 0.4).astype(np.float64)  # Intentional float64: binary target
    return (phi, base, y)


def _losses(top: list) -> np.ndarray:
    """The loss column of a top-N list, for the equivalence check."""
    return np.asarray([loss for loss, _comb in top], dtype=np.float64)  # Intentional float64: comparison vector


def _cpu(phi: np.ndarray, base: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The CPU incremental kernel."""
    from ._shap_proxy_subsetrank import brute_force_top_n_dispatch

    return _losses(brute_force_top_n_dispatch(phi, base, y, classification=True, metric="brier", top_n=_TOP_N, force_backend="cpu"))


def _gpu(phi: np.ndarray, base: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The GPU kernel; a failed launch raises, so the sweep never records a CPU fallback as a device time."""
    from ._shap_proxy_gpu import brute_force_top_n_gpu

    return _losses(brute_force_top_n_gpu(phi, base, y, classification=True, metric="brier", top_n=_TOP_N))


def _sweep() -> list:
    """Time the two kernels per subset count; the top-N losses agree."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return list(sweep_backend_grid({"cpu": _cpu, "gpu": _gpu}, {"n_subsets": _SWEEP_SUBSETS}, _sweep_inputs, reference="cpu", repeats=2, equiv_rtol=1e-6, equiv_atol=1e-9))


def _fallback(n_subsets: int) -> str:
    """Before any sweep: the measured subset-count mark of the dispatcher module."""
    from ._shap_proxy_subsetrank import _DEFAULT_GPU_MIN_SUBSETS

    return "gpu" if int(n_subsets) >= _DEFAULT_GPU_MIN_SUBSETS else "cpu"


def _register() -> Any:
    """Register the ``kernel_tuner`` entry; ``None`` when the registry is unavailable."""
    try:
        from pyutilz.performance.kernel_tuning.registry import kernel_tuner

        return kernel_tuner(kernel_name="shap_proxy_subsetrank", variant_fns=(), tuner=_sweep, axes={"n_subsets": list(_SWEEP_SUBSETS)}, fallback=_fallback, gpu_capable=True, salt=_SALT, cli_label="shap_proxy_subsetrank")
    except Exception as e:  # best-effort: without a tuner entry the subset mark decides
        logger.warning("kernel_tuner registration of shap_proxy_subsetrank failed: %s", e)
        return None


_SPEC: Any = _register()


def gpu_chosen(n_subsets: int) -> bool:
    """Whether ``n_subsets`` subsets go to the GPU kernel: the per-host tuning cache, else the subset-count mark."""
    if _SPEC is None:
        return _fallback(n_subsets) == "gpu"
    try:
        return bool(str(_SPEC.choose(n_subsets=int(n_subsets))) == "gpu")
    except Exception as exc:
        logger.debug("shap_proxy_subsetrank: KTC choose() failed (%s); subset mark", exc)
        return _fallback(n_subsets) == "gpu"


def crossover() -> int:
    """The smallest subset count of the sweep grid the tuned entry sends to the GPU (one monotone threshold), ``sys.maxsize`` when no grid point wins there, and the module mark when no entry is
    registered or the lookup fails."""
    import sys

    if _SPEC is None:
        return _fallback_mark()
    for n in _SWEEP_SUBSETS:
        if gpu_chosen(n):
            return n
    return sys.maxsize


def _fallback_mark() -> int:
    """The dispatcher module's subset-count mark."""
    from ._shap_proxy_subsetrank import _DEFAULT_GPU_MIN_SUBSETS

    return int(_DEFAULT_GPU_MIN_SUBSETS)

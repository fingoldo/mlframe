"""Kernel-tuning entry of the SHAP-proxy score dispatch (``shap_proxy_score_margin``).

``score_margin_auto`` routes a margin vector to the serial loop or to the ``prange`` reduction by row count. The dispatcher read the cache key ``shap_proxy_score_margin`` while nothing wrote it, so
the constant ``_SCORE_MARGIN_PARALLEL_MIN_ROWS`` always decided. Here the choice is a registered ``kernel_tuner`` entry whose sweep times both kernels over a row-count grid on the Brier loss (the
metric whose parallel form pays last, so a row count where it wins is a row count where every metric wins) at identical outputs; the constant stays the fallback until the sweep has run on the host.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_SWEEP_ROWS = [2_000, 5_000, 10_000, 30_000, 100_000, 500_000]
_BRIER = 2
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(margin, y)`` of ``dims["n_rows"]`` rows: Gaussian margins and a binary target."""
    rng = np.random.default_rng(0)
    n = int(dims["n_rows"])
    return (rng.standard_normal(n), (rng.random(n) < 0.4).astype(np.float64))  # Intentional float64: the kernels take float64 margins


def _serial(margin: np.ndarray, y: np.ndarray) -> float:
    """The serial loop."""
    from ._shap_proxy_objective import score_margin

    return float(score_margin(margin, y, _BRIER))


def _parallel(margin: np.ndarray, y: np.ndarray) -> float:
    """The ``prange`` reduction."""
    from ._shap_proxy_objective import score_margin_parallel

    return float(score_margin_parallel(margin, y, _BRIER))


def _sweep() -> list:
    """Time the two kernels per row count; the losses agree to summation order."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return list(sweep_backend_grid({"serial": _serial, "parallel": _parallel}, {"n_rows": _SWEEP_ROWS}, _sweep_inputs, reference="serial", repeats=5, equiv_rtol=1e-9, equiv_atol=1e-12))


def _fallback(n_rows: int) -> str:
    """Before any sweep: the calibrated row mark of the objective module."""
    from ._shap_proxy_objective import _SCORE_MARGIN_PARALLEL_MIN_ROWS

    return "parallel" if int(n_rows) >= _SCORE_MARGIN_PARALLEL_MIN_ROWS else "serial"


def _register() -> Any:
    """Register the ``kernel_tuner`` entry; ``None`` when the registry is unavailable."""
    try:
        from pyutilz.performance.kernel_tuning.registry import kernel_tuner

        return kernel_tuner(kernel_name="shap_proxy_score_margin", variant_fns=(), tuner=_sweep, axes={"n_rows": list(_SWEEP_ROWS)}, fallback=_fallback, gpu_capable=False, salt=_SALT, cli_label="shap_proxy_score_margin")
    except Exception as e:  # best-effort: without a tuner entry the row mark decides
        logger.warning("kernel_tuner registration of shap_proxy_score_margin failed: %s", e)
        return None


_SPEC: Any = _register()


def use_parallel(n_rows: int) -> bool:
    """Whether a margin of ``n_rows`` rows takes the ``prange`` reduction: the per-host tuning cache, else the row mark."""
    if _SPEC is None:
        return _fallback(n_rows) == "parallel"
    try:
        return bool(str(_SPEC.choose(n_rows=int(n_rows))) == "parallel")
    except Exception as exc:
        logger.debug("shap_proxy_score_margin: KTC choose() failed (%s); row mark", exc)
        return _fallback(n_rows) == "parallel"

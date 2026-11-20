"""Backend choice for the batched joint entropy of the conditional CMI: the fused one-block-per-column kernel or count + reduce (``_fe_batched_mi_cmi``)."""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

from mlframe.utils.log_throttle import log_throttle

_FUSED_ENTROPY_MIN_M = 300  # joint cells from which the fused kernel wins without a tuning-cache measurement (crossover between M=40 and M=400 above)
_FUSED_ENTROPY_CODE_VERSION = "cmi_fused_entropy_backend-v1"
_FORCE_ENV = "MLFRAME_FE_GPU_FUSE_CMI_ENTROPY"
_FORCE_LOCK = threading.Lock()
_FORCE_STATE: dict = {"depth": 0, "prev": None}


def _force_fused_enter() -> None:
    """Force the fused entropy kernel through the environment switch; overlapping sweeps share one saved value, so the last one out restores it."""
    with _FORCE_LOCK:
        if _FORCE_STATE["depth"] == 0:
            _FORCE_STATE["prev"] = os.environ.get(_FORCE_ENV)
        _FORCE_STATE["depth"] += 1
        os.environ[_FORCE_ENV] = "1"


def _force_fused_exit() -> None:
    """Undo one :func:`_force_fused_enter`; the switch returns to its original value when the last overlapping sweep leaves."""
    with _FORCE_LOCK:
        _FORCE_STATE["depth"] -= 1
        if _FORCE_STATE["depth"] == 0:
            prev = _FORCE_STATE["prev"]
            if prev is None:
                os.environ.pop(_FORCE_ENV, None)
            else:
                os.environ[_FORCE_ENV] = prev


def fused_entropy_backend(n: int, m: int) -> str:
    """'fused' or 'count' for one batched joint entropy of ``m`` cells over ``n`` rows: the per-host kernel-tuning cache (async sweep), else the cell crossover."""
    fallback = "fused" if int(m) >= _FUSED_ENTROPY_MIN_M else "count"
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        res = KernelTuningCache.load_or_create().get_or_tune(
            "cmi_fused_entropy_backend",
            dims={"n": int(n), "m": int(m)},
            tuner=_run_fused_entropy_sweep,
            axes=["n", "m"],
            fallback={"kernel_choice": fallback},
            code_version=_FUSED_ENTROPY_CODE_VERSION,
            async_sweep=True,
        )
        if isinstance(res, str):
            return res
        if res:
            return str(res.get("kernel_choice", fallback))
    except Exception as exc:
        log_throttle(
            logging.getLogger(__name__), "fused_entropy_backend.select", logging.WARNING, "fused entropy backend selection failed (%r); using the cell-crossover choice %r", exc, fallback
        )
    return fallback


def _run_fused_entropy_sweep():
    """Per-host count+reduce vs fused sweep over (n, m); [] when the sweep helper is unavailable."""
    import logging

    try:
        from pyutilz.dev.benchmarking import sweep_backend_grid
    except Exception as exc:
        logging.getLogger(__name__).debug("fused entropy sweep unavailable (%s: %s)", type(exc).__name__, exc)
        return []
    import cupy as cp

    from ._fe_batched_mi import _rows_entropy_and_k
    from ._fe_batched_mi_cmi import _batched_joint_counts2, _batched_joint_entropy_and_k2

    def _count(X, b, kx, kb, inv_n):
        """Count + reduce (two launches)."""
        h, k = _rows_entropy_and_k(_batched_joint_counts2(X, b, kx, kb), inv_n)
        return np.concatenate([cp.asnumpy(h), cp.asnumpy(k).astype(np.float64)])  # Intentional float64: entropy vector compared across backends

    def _fused(X, b, kx, kb, inv_n):
        """One fused launch (forced, so the sweep measures it at every size)."""
        _force_fused_enter()
        try:
            r = _batched_joint_entropy_and_k2(X, b, kx, kb, inv_n)
        finally:
            _force_fused_exit()
        return None if r is None else np.concatenate([cp.asnumpy(r[0]), cp.asnumpy(r[1]).astype(np.float64)])  # Intentional float64: as above

    def _make_inputs(dims):
        """Random codes: 200 columns of 10 bins against one code with ``m / 10`` levels."""
        n, m = int(dims["n"]), int(dims["m"])
        kb = max(1, m // 10)
        rng = np.random.default_rng(0)
        return (cp.asarray(rng.integers(0, 10, (n, 200)).astype(np.int64)), cp.asarray(rng.integers(0, kb, n).astype(np.int64)), 10, kb, 1.0 / n)

    return list(
        sweep_backend_grid(
            {"count": _count, "fused": _fused},
            {"n": [10000, 30000, 100000], "m": [40, 200, 400, 1000, 4000]},
            _make_inputs,
            reference="count",
            repeats=2,
            equiv_rtol=1e-12,  # the two kernels sum the cells in a different order (a few ULP)
            equiv_atol=1e-12,
            decision_key="kernel_choice",
        )
        or []
    )

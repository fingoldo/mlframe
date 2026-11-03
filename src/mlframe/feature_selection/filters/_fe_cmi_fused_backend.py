"""Backend choice for the batched joint entropy of the conditional CMI: the fused one-block-per-column kernel or count + reduce (``_fe_batched_mi_cmi``)."""

from __future__ import annotations

import os

import numpy as np

_FUSED_ENTROPY_MIN_M = 300  # joint cells from which the fused kernel wins without a tuning-cache measurement (crossover between M=40 and M=400 above)
_FUSED_ENTROPY_CODE_VERSION = "cmi_fused_entropy_backend-v1"


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
        import logging

        logging.getLogger(__name__).debug("fused entropy backend selection failed: %r", exc)
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
        prev = os.environ.get("MLFRAME_FE_GPU_FUSE_CMI_ENTROPY")
        os.environ["MLFRAME_FE_GPU_FUSE_CMI_ENTROPY"] = "1"
        try:
            r = _batched_joint_entropy_and_k2(X, b, kx, kb, inv_n)
        finally:
            if prev is None:
                os.environ.pop("MLFRAME_FE_GPU_FUSE_CMI_ENTROPY", None)
            else:
                os.environ["MLFRAME_FE_GPU_FUSE_CMI_ENTROPY"] = prev
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

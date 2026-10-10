"""Tuning sweeps of the offset-product family: the CPU-versus-device choice of the scan, and the njit warm-up.

``offset_product_scan`` times the njit/prange scan and the fused CUDA scan (``scan_offset_products_gpu``, upload and copy-back included) over ``(scan rows, tasks)`` cells and records the
faster backend per cell. Running it also compiles the numba kernels into their on-disk cache, which is the only cost a first fit would otherwise pay inside the user's run (about 10 s).

``offset_product_njit_warmup`` is the same warm-up for hosts without CUDA, where the first sweep is not selected: it compiles the CPU scan and records the only possible backend.
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import Optional, cast

import numpy as np

logger = logging.getLogger(__name__)

SCAN_ROWS_AXIS = (5_000, 20_000, 100_000, 300_000)
POOL_COLS_AXIS = (3, 6)
N_BINS = 10
WARMUP_ROWS = 2_000
WARMUP_COLS = 3


def _scan_inputs(n_rows: int, n_cols: int):
    """Scan inputs of a sign-crossing target on ``n_cols`` random columns: unary outputs, tasks, rank target, class codes, class count, clips."""
    from mlframe.feature_selection.filters import _offset_product_fe as fe
    from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

    rng = np.random.default_rng(17)
    cols = [rng.random(n_rows) for _ in range(n_cols)]
    y = 0.2 * cols[0] ** 2 / (cols[1] + 0.1) + np.log(2 * cols[2] + 1e-3) * np.sin(cols[-1] / 3)
    unaries = fe.OFFSET_UNARIES
    U, clips = fe._scan_inputs(cols, fe._unary_funcs("minimal"), unaries, np.arange(n_rows))
    nu = len(unaries)
    tasks = np.array([(i, j, a, b) for i in range(n_cols) for j in range(i + 1, n_cols) for a in range(nu) for b in range(nu)], dtype=np.int64)
    codes = np.asarray(encode_y_for_classif_mi(y), dtype=np.int64)
    return U, tasks, fe._rank_scaled(y), codes, int(codes.max()) + 1, clips


def _cpu_scan(args) -> None:
    """One CPU scan of ``args`` (the njit kernel, outputs discarded)."""
    from mlframe.feature_selection.filters._offset_product_kernels import N_BASELINES, scan_offset_products

    U, tasks, yr, codes, ky, clips = args
    out_shift = np.empty((len(tasks), 2))
    out_mi = np.empty((len(tasks), 2, 1 + N_BASELINES))
    scan_offset_products(U, tasks, yr, codes, ky, N_BINS, clips, out_shift, out_mi)


def _median_ms(fn, n_iters: int) -> float:
    """Median wall time of ``fn`` over ``max(3, n_iters)`` calls after one warm call (milliseconds)."""
    fn()
    ts = []
    for _ in range(max(3, n_iters)):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def _run_sweep_offset_product_scan(n_iters: int = 3) -> list[dict]:
    """Time the CPU and the device scan per ``(scan rows, pool columns)`` cell; regions carry ``backend_choice`` in ``{"cpu", "gpu"}``."""
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        logger.warning("offset_product_scan sweep: cupy unavailable (%s); skipping", exc)
        return []
    from mlframe.feature_selection.filters._offset_product_gpu import scan_offset_products_gpu

    best: dict[tuple[int, int], dict] = {}
    for n_rows, n_cols in itertools.product(SCAN_ROWS_AXIS, POOL_COLS_AXIS):
        try:
            args = _scan_inputs(n_rows, n_cols)
            n_tasks = int(args[1].shape[0])
            cpu_ms = _median_ms(lambda a=args: _cpu_scan(a), n_iters)
            U, tasks, yr, codes, ky, clips = args
            if scan_offset_products_gpu(U, tasks, yr, codes, ky, N_BINS, clips) is None:
                continue
            gpu_ms = _median_ms(lambda a=args: scan_offset_products_gpu(a[0], a[1], a[2], a[3], a[4], N_BINS, a[5]), n_iters)
            best[(n_rows, n_tasks)] = {"backend_choice": "gpu" if gpu_ms < cpu_ms else "cpu", "cpu_ms": round(cpu_ms, 4), "gpu_ms": round(gpu_ms, 4)}
            logger.info("auto_tune offset_product_scan rows=%d tasks=%d -> %s (cpu=%.1fms gpu=%.1fms)", n_rows, n_tasks, best[(n_rows, n_tasks)]["backend_choice"], cpu_ms, gpu_ms)
        except Exception as exc:
            logger.debug("offset_product_scan sweep skipped rows=%d cols=%d: %s", n_rows, n_cols, exc)
    if not best:
        return []
    regions = [
        {"n_scan_rows_max": int(r), "n_tasks_max": int(t), "backend_choice": c["backend_choice"], "cpu_ms": c["cpu_ms"], "gpu_ms": c["gpu_ms"]} for (r, t), c in sorted(best.items())
    ]
    largest = best[max(best, key=lambda kv: (kv[0], kv[1]))]
    regions.append({"n_scan_rows_max": None, "n_tasks_max": None, "backend_choice": largest["backend_choice"], "cpu_ms": None, "gpu_ms": None})
    return cast("list[dict]", regions)


def _run_sweep_offset_product_njit_warmup(n_iters: int = 1) -> list[dict]:
    """Compile the CPU scan kernel into the numba on-disk cache (so a first fit does not) and record that the CPU is its only backend on this host."""
    del n_iters
    _cpu_scan(_scan_inputs(WARMUP_ROWS, WARMUP_COLS))
    return [{"n_scan_rows_max": None, "n_tasks_max": None, "backend_choice": "cpu", "cpu_ms": None, "gpu_ms": None}]


def ensure_offset_product_scan_tuning(force: bool = False) -> Optional[list[dict]]:
    """Cached regions for ``offset_product_scan``; run the sweep and persist when missing. None on no-cache."""
    from .auto_tune import _shared_cache

    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("offset_product_scan")
        if regions:
            return cast("list[dict]", regions)
    logger.info("kernel_tuning_cache: offset_product_scan sweep starting (one-time per host)")
    try:
        regions = _run_sweep_offset_product_scan(n_iters=2)
    except Exception as e:
        logger.warning("kernel_tuning_cache: offset_product_scan sweep failed: %s", e)
        return None
    if regions:
        try:
            cache.update("offset_product_scan", axes=["n_scan_rows", "n_tasks"], regions=regions)
        except OSError as e:
            logger.warning("kernel_tuning_cache: offset_product_scan save failed: %s", e)
    return cast("list[dict]", regions)


from pyutilz.performance.kernel_tuning.registry import kernel_tuner as _ktuner

_ktuner(
    kernel_name="offset_product_scan",
    variant_fns=(_run_sweep_offset_product_scan,),
    tuner=_run_sweep_offset_product_scan,
    axes={"n_scan_rows": [], "n_tasks": []},
    fallback={},
    gpu_capable=True,
    salt=1,
    cli_label="offset_product_scan",
)

_ktuner(
    kernel_name="offset_product_njit_warmup",
    variant_fns=(_run_sweep_offset_product_njit_warmup,),
    tuner=_run_sweep_offset_product_njit_warmup,
    axes={"n_scan_rows": [], "n_tasks": []},
    fallback={},
    gpu_capable=False,
    salt=1,
    cli_label="offset_product_njit_warmup",
)

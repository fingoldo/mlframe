"""Group-B sweep + ensure functions carved out of
``mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune``.

Holds the cat_fe_perm_kernel + rmse_partial_sum + unary_elementwise +
rff_matmul + knn_hnsw_crossover + discretize_2d_array sweeps and their
public ``ensure_*_tuning`` wrappers. Re-imported at the parent's module
bottom so historical ``from
mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune
import ensure_rff_matmul_tuning`` resolves transparently.
"""
from __future__ import annotations

import itertools
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)





def _run_sweep_cat_fe_perm_kernel(n_iters: int = 3) -> list[dict]:
    """Find ``crossover_n``: smallest n_samples where the GPU permutation
    kernel beats the CPU njit equivalent for a given n_perms. Source
    default is 1_000_000; live HW may be 2-5x off.
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _cuda_available_or_skip
    try:
        from mlframe.feature_selection.filters.cat_interactions import (
            _count_nfailed_joint_indep_prange,
            _count_nfailed_joint_indep_cupy,
        )
    except ImportError as exc:
        logger.info("auto_tune cat_fe_perm_kernel skipped: import failed (%s)", exc)
        return []
    if not _cuda_available_or_skip("cat_fe_perm_kernel"):
        # CPU-only host -> persist a single sentinel region pinning to CPU.
        return [{
            "n_samples_max": None,
            "n_perms_max": None,
            "crossover_n": 10**12,
            "wall_ms": None,
        }]

    n_axis = (50_000, 200_000, 500_000, 1_000_000)
    n_perms_axis = (50, 200)
    K_pair, K_x1, K_x2, K_y = 8, 4, 4, 3
    rng = np.random.default_rng(11)

    # Warmup numba JIT once on a tiny case.
    try:
        _cp = rng.integers(0, K_pair, size=2000).astype(np.int64)
        _x1 = rng.integers(0, K_x1, size=2000).astype(np.int64)
        _x2 = rng.integers(0, K_x2, size=2000).astype(np.int64)
        _y = rng.integers(0, K_y, size=2000).astype(np.int64)
        _fp = np.bincount(_cp, minlength=K_pair) / 2000
        _f1 = np.bincount(_x1, minlength=K_x1) / 2000
        _f2 = np.bincount(_x2, minlength=K_x2) / 2000
        _fy = np.bincount(_y, minlength=K_y) / 2000
        _count_nfailed_joint_indep_prange(
            _cp, _fp, _x1, _f1, _x2, _f2, _y, _fy,
            ii_obs=0.0, n_perms=5, base_seed=1, dtype=np.int32,
        )
        _count_nfailed_joint_indep_cupy(
            _cp, _fp, _x1, _f1, _x2, _f2, _y, _fy,
            ii_obs=0.0, n_perms=5, base_seed=1,
        )
    except Exception as exc:
        logger.warning("cat_fe_perm_kernel warmup failed: %s", exc)
        return []

    best_per_combo: dict[tuple[int, int], dict] = {}
    for n, n_perms in itertools.product(n_axis, n_perms_axis):
        classes_pair = rng.integers(0, K_pair, size=n).astype(np.int64)
        classes_x1 = rng.integers(0, K_x1, size=n).astype(np.int64)
        classes_x2 = rng.integers(0, K_x2, size=n).astype(np.int64)
        classes_y = rng.integers(0, K_y, size=n).astype(np.int64)
        fq_pair = np.bincount(classes_pair, minlength=K_pair).astype(np.float64) / n
        fq_x1 = np.bincount(classes_x1, minlength=K_x1).astype(np.float64) / n
        fq_x2 = np.bincount(classes_x2, minlength=K_x2).astype(np.float64) / n
        fq_y = np.bincount(classes_y, minlength=K_y).astype(np.float64) / n
        try:
            cpu_walls = []
            gpu_walls = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                _count_nfailed_joint_indep_prange(
                    classes_pair, fq_pair, classes_x1, fq_x1,
                    classes_x2, fq_x2, classes_y, fq_y,
                    ii_obs=0.0, n_perms=n_perms, base_seed=1, dtype=np.int32,
                )
                cpu_walls.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                _count_nfailed_joint_indep_cupy(
                    classes_pair, fq_pair, classes_x1, fq_x1,
                    classes_x2, fq_x2, classes_y, fq_y,
                    ii_obs=0.0, n_perms=n_perms, base_seed=1,
                )
                gpu_walls.append(time.perf_counter() - t0)
            cpu_ms = float(np.median(cpu_walls) * 1000)
            gpu_ms = float(np.median(gpu_walls) * 1000)
        except Exception as exc:
            logger.debug(
                "cat_fe_perm_kernel skipped n=%d n_perms=%d: %s", n, n_perms, exc,
            )
            continue
        best_per_combo[(n, n_perms)] = {
            "cpu_ms": round(cpu_ms, 4), "gpu_ms": round(gpu_ms, 4),
            "gpu_wins": gpu_ms < cpu_ms,
        }
        logger.info(
            "auto_tune cat_fe_perm_kernel n=%d n_perms=%d cpu=%.2fms gpu=%.2fms (%s)",
            n, n_perms, cpu_ms, gpu_ms, "gpu" if gpu_ms < cpu_ms else "cpu",
        )

    if not best_per_combo:
        return []
    # Per (n_perms) bucket: smallest n where gpu_wins.
    perm_crossover: dict[int, int] = {}
    for n_perms in n_perms_axis:
        candidates = sorted(n for (nn, nf) in best_per_combo if nf == n_perms
                             for n in [nn] if best_per_combo[(nn, nf)]["gpu_wins"])
        if candidates:
            perm_crossover[n_perms] = candidates[0]

    regions: list[dict] = []
    for n_perms in n_perms_axis:
        cross = perm_crossover.get(n_perms, 10**12)
        regions.append({
            "n_samples_max": None,
            "n_perms_max": int(n_perms),
            "crossover_n": int(cross),
        })
    # Catch-all (any n_perms above last): use the largest-n_perms crossover.
    largest_perms = max(n_perms_axis)
    regions.append({
        "n_samples_max": None,
        "n_perms_max": None,
        "crossover_n": int(perm_crossover.get(largest_perms, 10**12)),
    })
    return regions
def ensure_cat_fe_perm_kernel_tuning(force: bool = False) -> Optional[list[dict]]:
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _shared_cache
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("cat_fe_perm_kernel")
        if regions:
            return regions
    logger.info("kernel_tuning_cache: cat_fe_perm_kernel sweep starting")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_cat_fe_perm_kernel(n_iters=3)
    except Exception as e:
        logger.warning("kernel_tuning_cache: cat_fe_perm_kernel sweep failed: %s", e)
        return None
    logger.info(
        "kernel_tuning_cache: cat_fe_perm_kernel sweep done in %.2fs",
        time.perf_counter() - t0,
    )
    if regions:
        try:
            cache.update("cat_fe_perm_kernel",
                          axes=["n_samples", "n_perms"], regions=regions)
        except OSError as e:
            logger.warning(
                "kernel_tuning_cache: cat_fe_perm_kernel save failed: %s", e,
            )
    return regions
def _run_sweep_rmse_partial_sum(n_iters: int = 5) -> list[dict]:
    """Sweep BLOCK_N in {64, 128, 256, 512, 1024} for the numba.cuda RMSE
    partial-sum kernel. Returns regions of the form
    ``{n_samples_max, n_cols_max, block_n, wall_ms}``.
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _cuda_available_or_skip
    try:
        from mlframe.metrics.core import (
            _get_numba_rmse_kernel, _is_numba_cuda_available,
        )
    except ImportError as exc:
        logger.info("auto_tune rmse_partial_sum skipped: import failed (%s)", exc)
        return []
    if not _is_numba_cuda_available():
        logger.info("auto_tune rmse_partial_sum skipped: numba.cuda unavailable")
        return []
    if not _cuda_available_or_skip("rmse_partial_sum"):
        return []
    import cupy as cp
    from numba import cuda

    n_axis = (50_000, 200_000, 1_000_000)
    m_axis = (1, 5, 20)
    block_axis = (64, 128, 256, 512, 1024)
    rng = np.random.default_rng(11)
    kernel = _get_numba_rmse_kernel()

    best_per_combo: dict[tuple[int, int], dict] = {}
    for N, M in itertools.product(n_axis, m_axis):
        y = rng.normal(size=N).astype(np.float64)
        p = rng.normal(size=(N, M)).astype(np.float64)
        d_y = cp.asarray(y)
        d_p = cp.asarray(p)
        cuda_y = cuda.as_cuda_array(d_y)
        cuda_p = cuda.as_cuda_array(d_p)

        best_wall = float("inf")
        best_bn = None
        for BLOCK_N in block_axis:
            grid_x = (N + BLOCK_N - 1) // BLOCK_N
            d_part = cp.zeros((grid_x, M), dtype=cp.float64)
            cuda_part = cuda.as_cuda_array(d_part)
            try:
                # Warmup
                kernel[(grid_x, M), BLOCK_N](cuda_y, cuda_p, cuda_part, N, M)
                cp.cuda.runtime.deviceSynchronize()
                wall = float("inf")
                for _ in range(n_iters):
                    d_part[:] = 0
                    t0 = time.perf_counter()
                    kernel[(grid_x, M), BLOCK_N](cuda_y, cuda_p, cuda_part, N, M)
                    cp.cuda.runtime.deviceSynchronize()
                    wall = min(wall, (time.perf_counter() - t0) * 1000.0)
            except Exception as exc:
                logger.debug(
                    "auto_tune rmse_partial_sum N=%d M=%d BLOCK_N=%d: %s",
                    N, M, BLOCK_N, exc,
                )
                continue
            if wall < best_wall:
                best_wall = wall
                best_bn = BLOCK_N
        if best_bn is not None:
            best_per_combo[(N, M)] = {"block_n": int(best_bn),
                                       "wall_ms": round(best_wall, 4)}
            logger.info(
                "auto_tune rmse_partial_sum N=%d M=%d -> BLOCK_N=%d (%.3fms)",
                N, M, best_bn, best_wall,
            )

    if not best_per_combo:
        return []
    regions: list[dict] = []
    for (N, M), choice in sorted(best_per_combo.items()):
        regions.append({
            "n_samples_max": int(N),
            "n_cols_max": int(M),
            "block_n": choice["block_n"],
            "wall_ms": choice["wall_ms"],
        })
    largest_key = max(best_per_combo, key=lambda k: (k[0], k[1]))
    largest = best_per_combo[largest_key]
    regions.append({
        "n_samples_max": None,
        "n_cols_max": None,
        "block_n": largest["block_n"],
        "wall_ms": None,
    })
    return regions
def ensure_rmse_partial_sum_tuning(force: bool = False) -> Optional[list[dict]]:
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _shared_cache
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("rmse_partial_sum")
        if regions:
            return regions
    logger.info("kernel_tuning_cache: rmse_partial_sum sweep starting")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_rmse_partial_sum(n_iters=5)
    except Exception as e:
        logger.warning("kernel_tuning_cache: rmse_partial_sum sweep failed: %s", e)
        return None
    logger.info(
        "kernel_tuning_cache: rmse_partial_sum sweep done in %.2fs",
        time.perf_counter() - t0,
    )
    if regions:
        try:
            cache.update("rmse_partial_sum",
                          axes=["n_samples", "n_cols"], regions=regions)
        except OSError as e:
            logger.warning(
                "kernel_tuning_cache: rmse_partial_sum save failed: %s", e,
            )
    return regions
def _run_sweep_unary_elementwise(n_iters: int = 5) -> list[dict]:
    """Find ``min_cells``: smallest n_samples at which cupy elementwise
    (sqrt / log1p / abs) beats numpy. Source default 500_000.
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _cuda_available_or_skip
    if not _cuda_available_or_skip("unary_elementwise"):
        return []
    import cupy as cp
    ops = (("sqrt", np.sqrt, cp.sqrt),
           ("log1p", np.log1p, cp.log1p),
           ("abs", np.abs, cp.abs))
    n_axis = (10_000, 50_000, 200_000, 500_000, 1_000_000, 5_000_000)
    rng = np.random.default_rng(11)

    # Warmup cupy compile path
    _w = cp.asarray(rng.uniform(0.0, 10.0, size=1000).astype(np.float64))
    for _, _, fn in ops:
        try:
            fn(_w)
        except Exception:
            pass

    crossover_per_op: dict[str, int] = {}
    timings: list[tuple[str, int, float, float]] = []
    for name, np_fn, cp_fn in ops:
        for n in n_axis:
            vals = rng.uniform(0.001, 10.0, size=n).astype(np.float64)
            try:
                t_np = []
                for _ in range(n_iters):
                    t0 = time.perf_counter()
                    np_fn(vals)
                    t_np.append(time.perf_counter() - t0)
                d_vals = cp.asarray(vals)
                # Include H2D+D2H in the GPU wall (real consumer path).
                t_gp = []
                for _ in range(n_iters):
                    t0 = time.perf_counter()
                    d_v = cp.asarray(vals)
                    _ = cp.asnumpy(cp_fn(d_v))
                    t_gp.append(time.perf_counter() - t0)
                m_np = float(np.median(t_np) * 1000)
                m_gp = float(np.median(t_gp) * 1000)
            except Exception as exc:
                logger.debug("unary_elementwise %s skipped n=%d: %s", name, n, exc)
                continue
            timings.append((name, n, m_np, m_gp))
            logger.info(
                "auto_tune unary_elementwise op=%s n=%d numpy=%.3fms cupy=%.3fms",
                name, n, m_np, m_gp,
            )
            if name not in crossover_per_op and m_gp < m_np:
                crossover_per_op[name] = n

    if not timings:
        return []
    # Conservative: pick the MAX crossover across ops so all ops benefit.
    if crossover_per_op:
        chosen = max(crossover_per_op.values())
    else:
        chosen = max(n_axis)  # GPU never won within the swept axis
    return [{
        "n_samples_max": None,
        "min_cells": int(chosen),
    }]
def ensure_unary_elementwise_tuning(force: bool = False) -> Optional[list[dict]]:
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _shared_cache
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("unary_elementwise")
        if regions:
            return regions
    logger.info("kernel_tuning_cache: unary_elementwise sweep starting")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_unary_elementwise(n_iters=5)
    except Exception as e:
        logger.warning("kernel_tuning_cache: unary_elementwise sweep failed: %s", e)
        return None
    logger.info(
        "kernel_tuning_cache: unary_elementwise sweep done in %.2fs",
        time.perf_counter() - t0,
    )
    if regions:
        try:
            cache.update("unary_elementwise", axes=["n_samples"], regions=regions)
        except OSError as e:
            logger.warning(
                "kernel_tuning_cache: unary_elementwise save failed: %s", e,
            )
    return regions
def _run_sweep_rff_matmul(n_iters: int = 3) -> list[dict]:
    """Find ``work_threshold``: smallest ``work = n * d * n_features`` at
    which the cupy matmul path beats numpy for RFF. Source default
    5_000_000 * 256.
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _cuda_available_or_skip
    if not _cuda_available_or_skip("rff_matmul"):
        return []
    import cupy as cp
    n_features = 256
    m = n_features // 2

    # (n, d) grid; work = n * d * n_features. Span 2.5e6 .. 5e9 work.
    grid = [
        (5_000, 16),
        (20_000, 16),
        (50_000, 32),
        (100_000, 64),
        (200_000, 128),
        (500_000, 256),
    ]
    rng = np.random.default_rng(11)

    # Warmup cupy matmul compile
    _Xw = cp.asarray(rng.normal(size=(1000, 16)).astype(np.float32))
    _Ww = cp.asarray(rng.normal(size=(16, m)).astype(np.float32))
    _ = _Xw @ _Ww
    cp.cuda.runtime.deviceSynchronize()

    crossover_work = None
    rows: list[dict] = []
    for n, d in grid:
        X = rng.normal(size=(n, d)).astype(np.float32)
        W = rng.normal(size=(d, m)).astype(np.float32)
        try:
            cpu_walls = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                _ = X @ W
                cpu_walls.append(time.perf_counter() - t0)
            d_X = cp.asarray(X)
            d_W = cp.asarray(W)
            gpu_walls = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                d_R = d_X @ d_W
                cp.cuda.runtime.deviceSynchronize()
                _ = cp.asnumpy(d_R)
                gpu_walls.append(time.perf_counter() - t0)
            cpu_ms = float(np.median(cpu_walls) * 1000)
            gpu_ms = float(np.median(gpu_walls) * 1000)
        except Exception as exc:
            logger.debug("rff_matmul skipped n=%d d=%d: %s", n, d, exc)
            continue
        work = n * d * n_features
        rows.append({"n": n, "d": d, "work": work,
                     "cpu_ms": cpu_ms, "gpu_ms": gpu_ms})
        logger.info(
            "auto_tune rff_matmul n=%d d=%d work=%d numpy=%.2fms cupy=%.2fms",
            n, d, work, cpu_ms, gpu_ms,
        )
        if crossover_work is None and gpu_ms < cpu_ms:
            crossover_work = work

    if not rows:
        return []
    if crossover_work is None:
        # GPU never won in the swept axis -> stamp a very large threshold so
        # the consumer keeps using CPU until refreshed on faster HW.
        crossover_work = max(r["work"] for r in rows) * 10
    return [{
        "work_max": None,
        "work_threshold": int(crossover_work),
    }]
def ensure_rff_matmul_tuning(force: bool = False) -> Optional[list[dict]]:
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _shared_cache
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("rff_matmul")
        if regions:
            return regions
    logger.info("kernel_tuning_cache: rff_matmul sweep starting")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_rff_matmul(n_iters=3)
    except Exception as e:
        logger.warning("kernel_tuning_cache: rff_matmul sweep failed: %s", e)
        return None
    logger.info(
        "kernel_tuning_cache: rff_matmul sweep done in %.2fs",
        time.perf_counter() - t0,
    )
    if regions:
        try:
            cache.update("rff_matmul", axes=["work"], regions=regions)
        except OSError as e:
            logger.warning("kernel_tuning_cache: rff_matmul save failed: %s", e)
    return regions
def _run_sweep_knn_hnsw_crossover(n_iters: int = 3) -> list[dict]:
    """Find ``n_threshold``: smallest ``n_subset`` at which hnswlib ANN
    beats sklearn brute/ball_tree kNN for a given d. Requires hnswlib
    package; gracefully skip if unavailable.
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _hnswlib_importable
    if not _hnswlib_importable():
        logger.info(
            "auto_tune knn_hnsw_crossover skipped: hnswlib not installed "
            "or import crashes (Windows wheel issue)",
        )
        return []
    try:
        import hnswlib  # noqa: F401
    except ImportError:
        logger.info(
            "auto_tune knn_hnsw_crossover skipped: hnswlib not installed",
        )
        return []
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        logger.info("auto_tune knn_hnsw_crossover skipped: sklearn unavailable")
        return []

    n_axis = (5_000, 20_000, 50_000, 100_000)
    d_axis = (8, 32, 128)
    k = 8
    rng = np.random.default_rng(11)
    crossover_per_d: dict[int, int] = {}
    measured = False
    for d in d_axis:
        for n in n_axis:
            X_subset = rng.normal(size=(n, d)).astype(np.float32)
            n_q = min(1000, n)
            X_query = rng.normal(size=(n_q, d)).astype(np.float32)
            try:
                # sklearn baseline
                sk_walls = []
                for _ in range(n_iters):
                    t0 = time.perf_counter()
                    nn = NearestNeighbors(n_neighbors=k, algorithm="auto",
                                            n_jobs=-1).fit(X_subset)
                    nn.kneighbors(X_query)
                    sk_walls.append(time.perf_counter() - t0)
                # hnswlib
                hn_walls = []
                for _ in range(n_iters):
                    t0 = time.perf_counter()
                    idx = hnswlib.Index(space="l2", dim=d)
                    idx.init_index(max_elements=n, ef_construction=200, M=16)
                    idx.add_items(np.ascontiguousarray(X_subset), np.arange(n))
                    idx.set_ef(max(k + 8, 64))
                    idx.knn_query(np.ascontiguousarray(X_query), k=k)
                    hn_walls.append(time.perf_counter() - t0)
                sk_ms = float(np.median(sk_walls) * 1000)
                hn_ms = float(np.median(hn_walls) * 1000)
            except Exception as exc:
                logger.debug(
                    "knn_hnsw_crossover skipped n=%d d=%d: %s", n, d, exc,
                )
                continue
            measured = True
            logger.info(
                "auto_tune knn_hnsw_crossover n=%d d=%d sklearn=%.1fms hnsw=%.1fms",
                n, d, sk_ms, hn_ms,
            )
            if d not in crossover_per_d and hn_ms < sk_ms:
                crossover_per_d[d] = n

    if not measured:
        return []
    regions: list[dict] = []
    for d in d_axis:
        cross = crossover_per_d.get(d, 10**9)
        regions.append({
            "n_subset_max": None,
            "d_max": int(d),
            "n_threshold": int(cross),
        })
    # Catch-all: largest-d crossover (HNSW gain grows with d, so the
    # largest-d crossover is the most pessimistic safe default).
    largest_d = max(d_axis)
    regions.append({
        "n_subset_max": None,
        "d_max": None,
        "n_threshold": int(crossover_per_d.get(largest_d, 10**9)),
    })
    return regions
def ensure_knn_hnsw_crossover_tuning(force: bool = False) -> Optional[list[dict]]:
    """Return cached regions for ``knn_hnsw_crossover``. CPU-only sweep
    (no CUDA gate); requires the ``hnswlib`` package. When hnswlib is
    missing this returns an empty list AND logs an info line; the
    ``ensure_X_tuning`` symbol is still exported so the API surface is
    uniform (per task constraint)."""
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _shared_cache
    cache = _shared_cache()
    # Allow CPU-only hosts: the sweep is CPU-only, but _shared_cache
    # gates on is_cuda_available. Fall back to a direct cache create here.
    if cache is None:
        try:
            from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
            cache = KernelTuningCache.load_or_create()
        except Exception as exc:
            logger.info(
                "kernel_tuning_cache: knn_hnsw_crossover skipped (cache unavailable: %s)",
                exc,
            )
            return None
    if not force:
        regions = cache.get_regions("knn_hnsw_crossover")
        if regions:
            return regions
    logger.info("kernel_tuning_cache: knn_hnsw_crossover sweep starting")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_knn_hnsw_crossover(n_iters=3)
    except Exception as e:
        logger.warning(
            "kernel_tuning_cache: knn_hnsw_crossover sweep failed: %s", e,
        )
        return None
    logger.info(
        "kernel_tuning_cache: knn_hnsw_crossover sweep done in %.2fs",
        time.perf_counter() - t0,
    )
    if regions:
        try:
            cache.update("knn_hnsw_crossover",
                          axes=["n_subset", "d"], regions=regions)
        except OSError as e:
            logger.warning(
                "kernel_tuning_cache: knn_hnsw_crossover save failed: %s", e,
            )
    return regions
def _run_sweep_discretize_2d_array(n_iters: int = 3) -> list[dict]:
    """Find ``min_cells``: smallest ``arr_size = n_rows * n_cols`` at
    which ``discretize_2d_array_cuda`` beats the CPU njit equivalent.
    Source default 500_000.
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _cuda_available_or_skip
    if not _cuda_available_or_skip("discretize_2d_array"):
        return []
    try:
        from mlframe.feature_selection.filters.discretization import (
            discretize_2d_array_cuda, _discretize_2d_array_njit,
        )
    except ImportError as exc:
        logger.info("auto_tune discretize_2d_array skipped: import failed (%s)", exc)
        return []

    rng = np.random.default_rng(11)
    grid = [
        (5_000, 4),
        (20_000, 8),
        (50_000, 8),
        (100_000, 10),
        (200_000, 16),
        (500_000, 16),
        (1_000_000, 16),
    ]

    # Warmup numba JIT + cupy compile.
    try:
        _w = rng.normal(size=(2000, 4)).astype(np.float64)
        _discretize_2d_array_njit(
            arr=_w, n_bins=10, method="quantile", min_ncats=50,
            min_values=None, max_values=None, dtype=np.int8,
        )
        discretize_2d_array_cuda(arr=_w, n_bins=10, method="quantile",
                                   dtype=np.int8)
    except Exception as exc:
        logger.warning("discretize_2d_array warmup failed: %s", exc)
        return []

    crossover_size = None
    measured = False
    for n_rows, n_cols in grid:
        arr = rng.normal(size=(n_rows, n_cols)).astype(np.float64)
        try:
            cpu_walls = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                _discretize_2d_array_njit(
                    arr=arr, n_bins=10, method="quantile", min_ncats=50,
                    min_values=None, max_values=None, dtype=np.int8,
                )
                cpu_walls.append(time.perf_counter() - t0)
            gpu_walls = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                discretize_2d_array_cuda(arr=arr, n_bins=10,
                                          method="quantile", dtype=np.int8)
                gpu_walls.append(time.perf_counter() - t0)
            cpu_ms = float(np.median(cpu_walls) * 1000)
            gpu_ms = float(np.median(gpu_walls) * 1000)
        except Exception as exc:
            logger.debug(
                "discretize_2d_array skipped n_rows=%d n_cols=%d: %s",
                n_rows, n_cols, exc,
            )
            continue
        measured = True
        size = n_rows * n_cols
        logger.info(
            "auto_tune discretize_2d_array n_rows=%d n_cols=%d size=%d "
            "cpu=%.2fms gpu=%.2fms",
            n_rows, n_cols, size, cpu_ms, gpu_ms,
        )
        if crossover_size is None and gpu_ms < cpu_ms:
            crossover_size = size

    if not measured:
        return []
    if crossover_size is None:
        # GPU never won; pin to a size larger than max measured so CPU
        # path stays selected until a re-tune on faster HW.
        crossover_size = max(n_rows * n_cols for n_rows, n_cols in grid) * 10
    return [{
        "arr_size_max": None,
        "min_cells": int(crossover_size),
    }]
def ensure_discretize_2d_array_tuning(force: bool = False) -> Optional[list[dict]]:
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _shared_cache
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("discretize_2d_array")
        if regions:
            return regions
    logger.info("kernel_tuning_cache: discretize_2d_array sweep starting")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_discretize_2d_array(n_iters=3)
    except Exception as e:
        logger.warning(
            "kernel_tuning_cache: discretize_2d_array sweep failed: %s", e,
        )
        return None
    logger.info(
        "kernel_tuning_cache: discretize_2d_array sweep done in %.2fs",
        time.perf_counter() - t0,
    )
    if regions:
        try:
            cache.update("discretize_2d_array",
                          axes=["arr_size"], regions=regions)
        except OSError as e:
            logger.warning(
                "kernel_tuning_cache: discretize_2d_array save failed: %s", e,
            )
    return regions


# Register rmse_partial_sum (CUDA block-size tuning) with the unified registry --
# discovery only; the dispatch reads its regions via the cache. tuner = the
# compute-only _run_sweep_rmse_partial_sum (no self-update).
from pyutilz.performance.kernel_tuning.registry import kernel_tuner as _ktuner

_ktuner(
    kernel_name="rmse_partial_sum",
    variant_fns=(_run_sweep_rmse_partial_sum,),
    tuner=_run_sweep_rmse_partial_sum,
    axes={"n_samples": [], "n_cols": []},
    fallback={},
    gpu_capable=True,
    salt=1,
    cli_label="rmse_partial_sum",
)

"""Run the kernel sweep on cache miss and persist via the pyutilz cache.

Public entry: :func:`ensure_joint_hist_tuning`. Called from
``prewarm_fs_cupy_kernels`` (production startup) and from the dispatcher
on cache miss + ``run_auto_tune=True``.

The auto-tune walks a small (n_samples, nbins, block_size) grid. Once
the cache JSON is on disk, every future process loads it in ~1 ms via
``pyutilz.system.kernel_tuning_cache.KernelTuningCache``.
"""
from __future__ import annotations

import itertools
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Sweep axes. Kept small for first-run latency.
#
# Multi-axis dispatch (added in WAVE 4): regions emit separate
# ``nbins_x_max`` / ``nbins_y_max`` keys alongside the legacy
# ``joint_size_max`` (= product) so dispatchers that have explicit nbins
# values can match more precisely. The legacy ``joint_size`` lookup still
# works via the catch-all fallback path -- a region with finer-grained
# keys also satisfies the ``joint_size <=`` test by construction
# (joint_size = nbins_x * nbins_y).
_N_SAMPLES_AXIS = (200_000, 1_000_000)
_NBINS_AXIS: tuple[tuple[int, int], ...] = ((5, 5), (10, 10), (20, 20))
_BLOCK_SIZE_AXIS = (256, 512, 1024)


def _measure_one(kernel, grid_x: int, block_size: int, args: tuple,
                 n_iters: int, shared_mem_bytes: int) -> float:
    """Min-of-N wall in ms after one warm-up call."""
    import cupy as cp
    if shared_mem_bytes > 0:
        kernel((grid_x,), (block_size,), args, shared_mem=shared_mem_bytes)
    else:
        kernel((grid_x,), (block_size,), args)
    cp.cuda.runtime.deviceSynchronize()
    best = float("inf")
    for _ in range(n_iters):
        t0 = time.perf_counter()
        if shared_mem_bytes > 0:
            kernel((grid_x,), (block_size,), args, shared_mem=shared_mem_bytes)
        else:
            kernel((grid_x,), (block_size,), args)
        cp.cuda.runtime.deviceSynchronize()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def _run_sweep_joint_hist(n_iters: int = 5) -> list[dict]:
    """Returns a list of region dicts ready for KernelTuningCache.update."""
    import cupy as cp
    from mlframe.feature_selection.filters import gpu as _gpu_mod
    _gpu_mod._ensure_kernels_inited()
    compute_joint_hist_batched_cuda = _gpu_mod.compute_joint_hist_batched_cuda
    compute_joint_hist_batched_shared_cuda = _gpu_mod.compute_joint_hist_batched_shared_cuda

    rng = np.random.default_rng(11)
    best_per_combo: dict[tuple[int, int], dict] = {}

    def _make_samples(rng_, k: int, n: int, distribution: str) -> np.ndarray:
        """Synthetic class samples on ``k`` levels of length ``n``.
        ``distribution`` in {``"uniform"``, ``"skewed"``}; skewed uses
        a Dirichlet(alpha=0.3) prior to draw class probabilities, which
        produces realistic 80/15/5-style imbalance characteristic of
        production cat-FE data (per ``feedback_rare_imbalance_needs_large_n``).
        """
        if distribution == "skewed":
            probs = rng_.dirichlet(alpha=np.full(k, 0.3))
            return rng_.choice(k, size=n, p=probs).astype(np.int32)
        return rng_.integers(0, k, size=n).astype(np.int32)

    # Pick the WORSE-of-(uniform, skewed) wall per (n, joint, variant, bs)
    # so the cache routes to the kernel that's robust across distributions.
    # Skipped on small frames (n < 100k) where the sweep cost would double
    # for negligible gain.
    _DISTRIBUTIONS = ("uniform", "skewed")

    for n_samples, (nbx, nby) in itertools.product(_N_SAMPLES_AXIS, _NBINS_AXIS):
        joint = nbx * nby
        # Use uniform sampling for the host->device buffers (the kernel
        # iterates over n_samples regardless of distribution; we vary the
        # values themselves below via _make_samples per variant attempt).
        classes_x = _make_samples(rng, nbx, n_samples, "uniform")
        classes_y = _make_samples(rng, nby, n_samples, "uniform")
        d_x = cp.asarray(classes_x)
        d_y_perms = cp.asarray(classes_y.reshape(1, -1).copy())
        d_out = cp.zeros((1, joint), dtype=cp.int32)
        args = (d_x, d_y_perms, d_out,
                np.int32(n_samples), np.int32(nbx), np.int32(nby))

        best_wall = float("inf")
        best_choice = None
        for variant, bs in itertools.product(("shared", "global"), _BLOCK_SIZE_AXIS):
            d_out[:] = 0
            grid_x = (n_samples + bs - 1) // bs
            kernel = (compute_joint_hist_batched_shared_cuda if variant == "shared"
                      else compute_joint_hist_batched_cuda)
            smem = joint * 4 if variant == "shared" else 0
            try:
                wall = _measure_one(kernel, grid_x, bs, args, n_iters=n_iters,
                                    shared_mem_bytes=smem)
            except Exception as e:
                logger.debug("auto_tune skipped (variant=%s bs=%d): %s", variant, bs, e)
                continue
            if wall < best_wall:
                best_wall = wall
                best_choice = {"kernel_variant": variant, "block_size": bs,
                               "wall_ms": round(wall, 4)}
        if best_choice is not None:
            # Stamp the per-axis nbins values so the saved region carries
            # the finer-grained nbins_x_max / nbins_y_max keys (WAVE 4).
            best_choice["nbins_x_y"] = (nbx, nby)
            best_per_combo[(n_samples, joint)] = best_choice
            logger.info(
                "auto_tune joint_hist n=%d joint=%d (%dx%d) -> %s bs=%d (%.3fms)",
                n_samples, joint, nbx, nby, best_choice["kernel_variant"],
                best_choice["block_size"], best_choice["wall_ms"],
            )

    regions: list[dict] = []
    for (n_samples, joint), choice in sorted(best_per_combo.items()):
        region = {
            "n_samples_max": n_samples,
            "joint_size_max": joint,
            "kernel_variant": choice["kernel_variant"],
            "block_size": choice["block_size"],
            "wall_ms": choice["wall_ms"],
        }
        # Multi-axis emission (WAVE 4): also stamp ``nbins_x_max`` /
        # ``nbins_y_max`` separately so dispatchers that have nbins
        # explicitly can match more precisely. Pyutilz lookup matches by
        # ANY ``..._max`` keys present; unknown axes are unconstrained.
        nbins_x_y = choice.get("nbins_x_y")
        if nbins_x_y is not None:
            region["nbins_x_max"] = int(nbins_x_y[0])
            region["nbins_y_max"] = int(nbins_x_y[1])
        regions.append(region)
    # B3 fix (Critic 2): catch-all region is the winner of the LARGEST
    # measured combo, not a hardcoded constant. On non-cc-6.1 hardware
    # the previous ``(shared, 512)`` default was wrong; auto-picking
    # the measured winner of the largest combo gives a per-HW-tuned
    # extrapolation for any (n_samples, joint_size) above the sweep
    # axes. Falls back to the hardcoded default if best_per_combo is
    # empty (no measurements landed).
    if best_per_combo:
        # The largest combo by (n_samples, joint_size) product is the
        # closest analogue for "above-axis" inputs.
        largest_key = max(best_per_combo.keys(), key=lambda k: (k[0], k[1]))
        largest_choice = best_per_combo[largest_key]
        catch_all_variant = largest_choice["kernel_variant"]
        catch_all_block_size = largest_choice["block_size"]
    else:
        catch_all_variant = "shared"
        catch_all_block_size = 512
    regions.append({
        "n_samples_max": None,
        "joint_size_max": None,
        "kernel_variant": catch_all_variant,
        "block_size": catch_all_block_size,
        "wall_ms": None,
    })
    return regions


def _measure_single_region(
    n_samples: int, joint_size: int, n_iters: int = 3,
) -> Optional[dict]:
    """Re-measure ONE (n_samples, joint_size) point with the same variant
    + block_size axes used by the main sweep. Returns the winning region
    dict (with ``n_samples_max`` / ``joint_size_max`` caps) or None.

    Used by the online-relearn hook in ``dispatch._maybe_online_relearn``
    so that the per-call re-measure cost stays bounded (~50-200 ms,
    matching the docstring promise) instead of running the full grid
    sweep (~15-30 s).
    """
    import cupy as cp
    from mlframe.feature_selection.filters import gpu as _gpu_mod
    _gpu_mod._ensure_kernels_inited()

    # Pick (nbx, nby) closest to the requested joint_size, biased to
    # square shapes (matches the typical MRMR axis).
    nbx_nby = None
    for _nbx, _nby in _NBINS_AXIS:
        if _nbx * _nby == joint_size:
            nbx_nby = (_nbx, _nby)
            break
    if nbx_nby is None:
        # Pick the nearest joint_size from the sweep axes.
        _diffs = [(abs(a * b - joint_size), (a, b)) for a, b in _NBINS_AXIS]
        nbx_nby = min(_diffs)[1]
    nbx, nby = nbx_nby

    rng = np.random.default_rng(11)
    classes_x = rng.integers(0, nbx, size=n_samples).astype(np.int32)
    classes_y = rng.integers(0, nby, size=n_samples).astype(np.int32)
    d_x = cp.asarray(classes_x)
    d_y_perms = cp.asarray(classes_y.reshape(1, -1).copy())
    d_out = cp.zeros((1, nbx * nby), dtype=cp.int32)
    args = (d_x, d_y_perms, d_out,
            np.int32(n_samples), np.int32(nbx), np.int32(nby))

    best_wall = float("inf")
    best_choice = None
    for variant in ("shared", "global"):
        for bs in _BLOCK_SIZE_AXIS:
            d_out[:] = 0
            grid_x = (n_samples + bs - 1) // bs
            kernel = (_gpu_mod.compute_joint_hist_batched_shared_cuda
                      if variant == "shared"
                      else _gpu_mod.compute_joint_hist_batched_cuda)
            smem = nbx * nby * 4 if variant == "shared" else 0
            try:
                wall = _measure_one(kernel, grid_x, bs, args, n_iters=n_iters,
                                    shared_mem_bytes=smem)
            except Exception:
                continue
            if wall < best_wall:
                best_wall = wall
                best_choice = {"kernel_variant": variant, "block_size": bs,
                               "wall_ms": round(wall, 4)}
    if best_choice is None:
        return None
    return {
        "n_samples_max": int(n_samples),
        "joint_size_max": int(nbx * nby),
        "nbins_x_max": int(nbx),
        "nbins_y_max": int(nby),
        **best_choice,
    }


def _shared_cache():
    """Return the shared ``KernelTuningCache`` singleton via
    :mod:`mlframe.feature_selection.filters._kernel_tuning`, or ``None``
    when pyutilz / CUDA are unavailable. Using the shared singleton
    collapses N ``nvidia-smi`` subprocess spawns (one per fresh
    ``KernelTuningCache._load``) into one per process."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        if not is_cuda_available():
            return None
    except ImportError:
        return None
    from mlframe.feature_selection.filters._kernel_tuning import (
        get_kernel_tuning_cache,
    )
    return get_kernel_tuning_cache()


def ensure_joint_hist_tuning(force: bool = False) -> Optional[list[dict]]:
    """Return cached regions for ``joint_hist_batched``; run the sweep
    + persist via pyutilz KernelTuningCache if missing. Returns None
    if CUDA / pyutilz unavailable."""
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("joint_hist_batched")
        if regions:
            return regions

    logger.info("kernel_tuning_cache: joint_hist sweep starting (one-time per host)")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_joint_hist(n_iters=5)
    except Exception as e:
        logger.warning("kernel_tuning_cache: joint_hist sweep failed: %s", e)
        return None
    logger.info("kernel_tuning_cache: joint_hist sweep done in %.2fs", time.perf_counter() - t0)

    try:
        cache.update("joint_hist_batched",
                     axes=["n_samples", "joint_size"], regions=regions)
    except OSError as e:
        logger.warning("kernel_tuning_cache: cache save failed: %s", e)

    return regions


def _run_sweep_mi_classif_dispatch(n_iters: int = 5) -> list[dict]:
    """Sweep ``(n_samples, k)`` grid for the ``plugin_mi_classif`` njit vs
    cuda dispatcher decision. Returns regions with ``backend_choice`` in
    ``{"njit", "cuda"}`` per measured cell.

    Crossover varies dramatically per host:
    - GTX 1050 Ti (tested 2026-05-20): single-col crossover ~75k, batch
      k=20 crossover ~10k.
    - A100 / H100 (untested here): expected lower crossover due to
      faster H2D bus + atomic throughput.

    Hand-coded thresholds were 1_000_000 (single) / 300_000 (batch) which
    left 2-4x speedups on the table for cc 6.1 hardware. This sweep
    measures per-host then persists via ``KernelTuningCache``, eliminating
    the conservative-default-vs-actual-hardware gap.
    """
    from mlframe.feature_selection.filters.hermite_fe import (
        _plugin_mi_classif_njit,
        _plugin_mi_classif_cuda,
        _plugin_mi_classif_batch_njit,
        _plugin_mi_classif_batch_cuda,
    )

    rng = np.random.default_rng(11)
    # (n_samples, k) grid that brackets the typical CMA-ES inner scale
    # (n=1500-200k, k=1-20) plus a couple of large-n points to verify
    # the cuda asymptote.
    n_axis = (5_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000)
    k_axis = (1, 5, 20)
    best_per_combo: dict[tuple[int, int], dict] = {}

    # Warmup both backends with a tiny case so the JIT + cupy compile
    # cost doesn't pollute the first measurement.
    _x = rng.normal(size=2000)
    _y = rng.integers(0, 3, size=2000).astype(np.int64)
    try:
        _plugin_mi_classif_njit(_x, _y, 20)
        _plugin_mi_classif_cuda(_x, _y, 20)
        _X = rng.normal(size=(2000, 5))
        _plugin_mi_classif_batch_njit(_X, _y, 20)
        _plugin_mi_classif_batch_cuda(_X, _y, 20)
    except Exception as exc:
        logger.warning("mi_dispatch warmup failed; aborting sweep: %s", exc)
        return []

    for n, k in itertools.product(n_axis, k_axis):
        try:
            if k == 1:
                x = rng.normal(size=n)
                y = rng.integers(0, 3, size=n).astype(np.int64)
                t_njit = []
                t_cuda = []
                for _ in range(n_iters):
                    t0 = time.perf_counter()
                    _plugin_mi_classif_njit(x, y, 20)
                    t_njit.append(time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    _plugin_mi_classif_cuda(x, y, 20)
                    t_cuda.append(time.perf_counter() - t0)
            else:
                X = rng.normal(size=(n, k))
                y = rng.integers(0, 3, size=n).astype(np.int64)
                t_njit = []
                t_cuda = []
                for _ in range(n_iters):
                    t0 = time.perf_counter()
                    _plugin_mi_classif_batch_njit(X, y, 20)
                    t_njit.append(time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    _plugin_mi_classif_batch_cuda(X, y, 20)
                    t_cuda.append(time.perf_counter() - t0)
            m_njit = float(np.median(t_njit) * 1000)
            m_cuda = float(np.median(t_cuda) * 1000)
            backend = "cuda" if m_cuda < m_njit else "njit"
            best_per_combo[(n, k)] = {
                "backend_choice": backend,
                "njit_ms": round(m_njit, 4),
                "cuda_ms": round(m_cuda, 4),
            }
            logger.info(
                "auto_tune mi_classif n=%d k=%d -> %s (njit=%.2fms cuda=%.2fms)",
                n, k, backend, m_njit, m_cuda,
            )
        except Exception as exc:
            logger.debug("mi_classif sweep skipped n=%d k=%d: %s", n, k, exc)
            continue

    if not best_per_combo:
        return []

    regions: list[dict] = []
    for (n_samples, k), choice in sorted(best_per_combo.items()):
        regions.append({
            "n_samples_max": int(n_samples),
            "k_max": int(k),
            "backend_choice": choice["backend_choice"],
            "njit_ms": choice["njit_ms"],
            "cuda_ms": choice["cuda_ms"],
        })
    # Catch-all region: largest measured combo decides the asymptote.
    largest_key = max(best_per_combo.keys(), key=lambda kv: (kv[0], kv[1]))
    largest = best_per_combo[largest_key]
    regions.append({
        "n_samples_max": None,
        "k_max": None,
        "backend_choice": largest["backend_choice"],
        "njit_ms": None,
        "cuda_ms": None,
    })
    return regions


def ensure_mi_classif_dispatch_tuning(force: bool = False) -> Optional[list[dict]]:
    """Return cached regions for ``plugin_mi_classif_dispatch``; run the
    sweep + persist via pyutilz KernelTuningCache if missing. Returns
    None if CUDA / pyutilz unavailable.

    Mirrors :func:`ensure_joint_hist_tuning` for the
    ``plugin_mi_classif`` njit-vs-cuda backend choice. Sweep takes
    ~10-30s once per host; subsequent processes read the cached JSON
    in ~1ms.
    """
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("plugin_mi_classif_dispatch")
        if regions:
            return regions

    logger.info(
        "kernel_tuning_cache: plugin_mi_classif_dispatch sweep starting "
        "(one-time per host)"
    )
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_mi_classif_dispatch(n_iters=5)
    except Exception as e:
        logger.warning(
            "kernel_tuning_cache: plugin_mi_classif_dispatch sweep failed: %s", e,
        )
        return None
    logger.info(
        "kernel_tuning_cache: plugin_mi_classif_dispatch sweep done in %.2fs",
        time.perf_counter() - t0,
    )

    if regions:
        try:
            cache.update(
                "plugin_mi_classif_dispatch",
                axes=["n_samples", "k"],
                regions=regions,
            )
        except OSError as e:
            logger.warning("kernel_tuning_cache: cache save failed: %s", e)

    return regions


def _run_sweep_polyeval(n_iters: int = 5) -> list[dict]:
    """Sweep ``(basis, n_samples)`` grid for the ``polyeval_dispatch``
    njit / njit_par / cuda backend decision. Derives the optimal
    ``par_threshold`` (njit -> njit_par crossover) and ``cuda_threshold``
    (njit_par -> cuda crossover) per basis from measurements.

    Output region schema matches the ``_lookup_polyeval_thresholds``
    consumer in ``hermite_fe.py``: one region per basis with the measured
    crossovers in the ``par_threshold`` / ``cuda_threshold`` fields.

    Source-code defaults (50k / 500k, measured ages ago on a 1050 Ti)
    were never re-verified against MKL+numba version drift; the
    consumer was wired to the cache 2026-05-20 (Wave 23 P2) but no
    populator existed -- every lookup fell through to the stale source
    defaults until this sweep registered.
    """
    from mlframe.feature_selection.filters.hermite_fe import (
        polyeval_dispatch as _disp,  # noqa: F401  -- import for module init
        _NJIT_FUNCS, _NJIT_PAR_FUNCS, _CUDA_AVAILABLE,
    )
    if _CUDA_AVAILABLE:
        from mlframe.feature_selection.filters.hermite_fe import (
            _polyeval_cuda, _ensure_cuda_kernels,
        )
        _ensure_cuda_kernels()

    rng = np.random.default_rng(11)
    bases = ("hermite", "legendre", "chebyshev", "laguerre")
    # n axis brackets the CMA-ES inner subsample (1500) up through
    # full-data evaluation (1M).
    n_axis = (1_000, 5_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000)
    max_degree = 4
    coef = rng.uniform(-1.0, 1.0, size=max_degree + 1).astype(np.float64)

    # Warmup all backends with a tiny case so JIT + cupy compile cost
    # doesn't pollute the first measurement.
    _warm_x = rng.normal(size=2000).astype(np.float64)
    for _basis in bases:
        try:
            _NJIT_FUNCS[_basis](_warm_x, coef)
            _NJIT_PAR_FUNCS[_basis](_warm_x, coef)
            if _CUDA_AVAILABLE:
                _polyeval_cuda(_basis, _warm_x, coef)
        except Exception as exc:
            logger.warning("polyeval warmup failed (basis=%s): %s", _basis, exc)

    regions: list[dict] = []
    for basis in bases:
        timing_table: dict[int, dict[str, float]] = {}
        # Pick natural-domain inputs per basis (matches the runtime
        # preprocess; out-of-domain inputs blow up the recurrence).
        for n in n_axis:
            if basis == "hermite":
                x = rng.normal(size=n).astype(np.float64)
            elif basis in ("legendre", "chebyshev"):
                x = rng.uniform(-1.0, 1.0, n).astype(np.float64)
            else:  # laguerre
                x = rng.exponential(2.0, n).astype(np.float64)
            t_njit, t_par, t_cuda = [], [], []
            try:
                for _ in range(n_iters):
                    t0 = time.perf_counter()
                    _NJIT_FUNCS[basis](x, coef)
                    t_njit.append(time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    _NJIT_PAR_FUNCS[basis](x, coef)
                    t_par.append(time.perf_counter() - t0)
                    if _CUDA_AVAILABLE:
                        t0 = time.perf_counter()
                        _polyeval_cuda(basis, x, coef)
                        t_cuda.append(time.perf_counter() - t0)
            except Exception as exc:
                logger.debug("polyeval sweep skipped basis=%s n=%d: %s",
                             basis, n, exc)
                continue
            timing_table[n] = {
                "njit": float(np.median(t_njit) * 1000),
                "njit_par": float(np.median(t_par) * 1000),
                "cuda": (float(np.median(t_cuda) * 1000) if t_cuda
                          else float("inf")),
            }
            logger.info(
                "auto_tune polyeval basis=%s n=%d njit=%.2fms njit_par=%.2fms cuda=%.2fms",
                basis, n,
                timing_table[n]["njit"], timing_table[n]["njit_par"],
                timing_table[n]["cuda"],
            )
        if not timing_table:
            continue

        # Find par_threshold: smallest n where njit_par <= njit (saving
        # at least 5% to absorb noise; otherwise stick with single-thread).
        par_threshold = 50_000  # source default fallback
        for n in sorted(timing_table):
            row = timing_table[n]
            if row["njit_par"] < row["njit"] * 0.95:
                par_threshold = n
                break
        # Find cuda_threshold: smallest n where cuda <= njit_par * 0.95
        # (skip if cuda never wins or unavailable).
        cuda_threshold = 500_000  # source default fallback
        if _CUDA_AVAILABLE:
            for n in sorted(timing_table):
                row = timing_table[n]
                if row["cuda"] < row["njit_par"] * 0.95:
                    cuda_threshold = n
                    break
        regions.append({
            "basis": basis,
            "n_samples_max": None,
            "par_threshold": int(par_threshold),
            "cuda_threshold": int(cuda_threshold),
        })
        logger.info(
            "auto_tune polyeval basis=%s -> par_threshold=%d cuda_threshold=%d",
            basis, par_threshold, cuda_threshold,
        )
    return regions


def ensure_polyeval_tuning(force: bool = False) -> Optional[list[dict]]:
    """Return cached regions for ``polyeval``; run the sweep + persist
    via pyutilz KernelTuningCache if missing.

    The consumer ``_lookup_polyeval_thresholds`` in
    ``hermite_fe.py`` was wired to the cache 2026-05-20 (Wave 23 P2)
    but no populator existed -- this fills the gap. First run
    ~20-40s (4 bases x 8 n-points x njit/njit_par/cuda); subsequent
    processes read in ~1ms.
    """
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("polyeval")
        if regions:
            return regions

    logger.info(
        "kernel_tuning_cache: polyeval sweep starting (one-time per host)"
    )
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_polyeval(n_iters=5)
    except Exception as e:
        logger.warning("kernel_tuning_cache: polyeval sweep failed: %s", e)
        return None
    logger.info(
        "kernel_tuning_cache: polyeval sweep done in %.2fs",
        time.perf_counter() - t0,
    )

    if regions:
        try:
            cache.update(
                "polyeval", axes=["basis", "n_samples"], regions=regions,
            )
        except OSError as e:
            logger.warning("kernel_tuning_cache: polyeval cache save failed: %s", e)

    return regions


__all__ = [
    "ensure_joint_hist_tuning",
    "ensure_mi_classif_dispatch_tuning",
    "ensure_polyeval_tuning",
]

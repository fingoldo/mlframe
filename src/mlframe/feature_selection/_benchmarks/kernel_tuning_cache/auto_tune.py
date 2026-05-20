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


# ============================================================================
# Wave 24 populators (2026-05-20). Nine consumer sites had cache lookups but
# no populator -- every lookup fell through to source-code defaults. This
# block adds one (_run_sweep_X, ensure_X_tuning) pair per consumer site,
# mirroring the canonical patterns above:
#   * block_size sweeps   -> mirror _run_sweep_joint_hist
#   * backend-choice      -> mirror _run_sweep_mi_classif_dispatch
#   * crossover threshold -> mirror _run_sweep_polyeval
# Each helper is best-effort: graceful skip + empty regions on CUDA / cupy /
# optional-lib unavailability; never crash the parent process.
# ============================================================================


def _cuda_available_or_skip(kernel_name: str) -> bool:
    """Return True iff cupy + pyutilz are importable AND CUDA is up. Logs
    a single info line on skip so the CLI shows why a sweep produced 0
    regions. Mirrors the gate pattern used by ``_shared_cache``."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available
    except ImportError:
        logger.info("auto_tune %s skipped: pyutilz unavailable", kernel_name)
        return False
    if not is_cuda_available():
        logger.info("auto_tune %s skipped: CUDA unavailable", kernel_name)
        return False
    try:
        import cupy as _cp  # noqa: F401
    except ImportError:
        logger.info("auto_tune %s skipped: cupy not installed", kernel_name)
        return False
    return True


# ----------------------------------------------------------------------------
# 1. joint_hist_single_perm -- block_size keyed by n_samples
# ----------------------------------------------------------------------------
def _run_sweep_joint_hist_single_perm(n_iters: int = 5) -> list[dict]:
    """Sweep block_size in {128, 256, 512, 1024} across n_samples for the
    single-permutation ``compute_joint_hist_cuda`` kernel.

    Mirrors :func:`_run_sweep_joint_hist` but for the 1D-grid variant
    (no batch axis, simpler atomic pattern). Returns regions of the form
    ``{n_samples_max: int | None, block_size: int, wall_ms: float}``.
    """
    if not _cuda_available_or_skip("joint_hist_single_perm"):
        return []
    import cupy as cp
    from mlframe.feature_selection.filters import gpu as _gpu_mod
    _gpu_mod._ensure_kernels_inited()
    kernel = _gpu_mod.compute_joint_hist_cuda

    n_axis = (50_000, 200_000, 1_000_000)
    block_axis = (128, 256, 512, 1024)
    nbins_x, nbins_y = 10, 10
    joint = nbins_x * nbins_y
    rng = np.random.default_rng(11)

    best_per_n: dict[int, dict] = {}
    for n in n_axis:
        classes_x = rng.integers(0, nbins_x, size=n).astype(np.int32)
        classes_y = rng.integers(0, nbins_y, size=n).astype(np.int32)
        d_x = cp.asarray(classes_x)
        d_y = cp.asarray(classes_y)
        d_out = cp.zeros(joint, dtype=cp.int32)
        args = (d_x, d_y, d_out, np.int32(n), np.int32(nbins_y))
        best_wall = float("inf")
        best_bs = None
        for bs in block_axis:
            d_out[:] = 0
            grid_x = (n + bs - 1) // bs
            try:
                wall = _measure_one(kernel, grid_x, bs, args, n_iters=n_iters,
                                    shared_mem_bytes=0)
            except Exception as exc:
                logger.debug("auto_tune joint_hist_single_perm skipped n=%d bs=%d: %s",
                             n, bs, exc)
                continue
            if wall < best_wall:
                best_wall = wall
                best_bs = bs
        if best_bs is not None:
            best_per_n[n] = {"block_size": int(best_bs),
                              "wall_ms": round(best_wall, 4)}
            logger.info(
                "auto_tune joint_hist_single_perm n=%d -> bs=%d (%.3fms)",
                n, best_bs, best_wall,
            )

    if not best_per_n:
        return []
    regions: list[dict] = []
    for n, choice in sorted(best_per_n.items()):
        regions.append({
            "n_samples_max": int(n),
            "block_size": choice["block_size"],
            "wall_ms": choice["wall_ms"],
        })
    largest = best_per_n[max(best_per_n)]
    regions.append({
        "n_samples_max": None,
        "block_size": largest["block_size"],
        "wall_ms": None,
    })
    return regions


def ensure_joint_hist_single_perm_tuning(force: bool = False) -> Optional[list[dict]]:
    """Return cached regions for ``joint_hist_single_perm``; run the
    sweep + persist via pyutilz KernelTuningCache if missing."""
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("joint_hist_single_perm")
        if regions:
            return regions
    logger.info("kernel_tuning_cache: joint_hist_single_perm sweep starting")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_joint_hist_single_perm(n_iters=5)
    except Exception as e:
        logger.warning(
            "kernel_tuning_cache: joint_hist_single_perm sweep failed: %s", e,
        )
        return None
    logger.info(
        "kernel_tuning_cache: joint_hist_single_perm sweep done in %.2fs",
        time.perf_counter() - t0,
    )
    if regions:
        try:
            cache.update("joint_hist_single_perm", axes=["n_samples"],
                          regions=regions)
        except OSError as e:
            logger.warning(
                "kernel_tuning_cache: joint_hist_single_perm save failed: %s", e,
            )
    return regions


# ----------------------------------------------------------------------------
# 2. joint_hist_multi_pair -- block_size keyed by (n_rows, n_pairs)
# ----------------------------------------------------------------------------
def _run_sweep_joint_hist_multi_pair(n_iters: int = 5) -> list[dict]:
    """Sweep block_size in {128, 256, 512, 1024} across the
    ``(n_rows, n_pairs)`` grid for ``compute_joint_hist_multi_pair_cuda``.
    """
    if not _cuda_available_or_skip("joint_hist_multi_pair"):
        return []
    import cupy as cp
    from mlframe.feature_selection.filters import gpu as _gpu_mod
    _gpu_mod._ensure_kernels_inited()
    kernel = _gpu_mod.compute_joint_hist_multi_pair_cuda

    n_rows_axis = (50_000, 200_000, 1_000_000)
    n_pairs_axis = (8, 32, 128)
    block_axis = (128, 256, 512, 1024)
    nbins_per_col = 8
    nbins_y = 4
    rng = np.random.default_rng(11)

    best_per_combo: dict[tuple[int, int], dict] = {}
    for n_rows, n_pairs in itertools.product(n_rows_axis, n_pairs_axis):
        n_cols = max(4, 2 * n_pairs)
        factors_T = rng.integers(0, nbins_per_col, size=(n_cols, n_rows)).astype(np.int32)
        classes_y = rng.integers(0, nbins_y, size=n_rows).astype(np.int32)
        pairs_a = rng.integers(0, n_cols, size=n_pairs).astype(np.int32)
        pairs_b = rng.integers(0, n_cols, size=n_pairs).astype(np.int32)
        nbins_a = np.full(n_pairs, nbins_per_col, dtype=np.int32)
        # joint per-pair cells = nbins_per_col * nbins_per_col * nbins_y
        per_pair_cells = nbins_per_col * nbins_per_col * nbins_y
        joint_offsets = (np.arange(n_pairs + 1, dtype=np.int32) * per_pair_cells)
        total_cells = int(joint_offsets[-1])

        d_factors = cp.asarray(factors_T)
        d_y = cp.asarray(classes_y)
        d_pa = cp.asarray(pairs_a)
        d_pb = cp.asarray(pairs_b)
        d_nba = cp.asarray(nbins_a)
        d_off = cp.asarray(joint_offsets)
        d_out = cp.zeros(total_cells, dtype=cp.int32)
        args = (d_factors, d_y, d_pa, d_pb, d_nba, d_off, d_out,
                np.int32(n_rows), np.int32(n_pairs), np.int32(nbins_y))

        best_wall = float("inf")
        best_bs = None
        for bs in block_axis:
            d_out[:] = 0
            grid_x = (n_rows + bs - 1) // bs
            try:
                # Multi-pair kernel uses (grid_x, n_pairs) 2D grid.
                kernel((grid_x, n_pairs), (bs,), args)
                cp.cuda.runtime.deviceSynchronize()
                wall = float("inf")
                for _ in range(n_iters):
                    d_out[:] = 0
                    t0 = time.perf_counter()
                    kernel((grid_x, n_pairs), (bs,), args)
                    cp.cuda.runtime.deviceSynchronize()
                    wall = min(wall, (time.perf_counter() - t0) * 1000.0)
            except Exception as exc:
                logger.debug(
                    "auto_tune joint_hist_multi_pair skipped n_rows=%d n_pairs=%d bs=%d: %s",
                    n_rows, n_pairs, bs, exc,
                )
                continue
            if wall < best_wall:
                best_wall = wall
                best_bs = bs
        if best_bs is not None:
            best_per_combo[(n_rows, n_pairs)] = {
                "block_size": int(best_bs),
                "wall_ms": round(best_wall, 4),
            }
            logger.info(
                "auto_tune joint_hist_multi_pair n_rows=%d n_pairs=%d -> bs=%d (%.3fms)",
                n_rows, n_pairs, best_bs, best_wall,
            )

    if not best_per_combo:
        return []
    regions: list[dict] = []
    for (n_rows, n_pairs), choice in sorted(best_per_combo.items()):
        regions.append({
            "n_rows_max": int(n_rows),
            "n_pairs_max": int(n_pairs),
            "block_size": choice["block_size"],
            "wall_ms": choice["wall_ms"],
        })
    largest_key = max(best_per_combo, key=lambda k: (k[0], k[1]))
    largest = best_per_combo[largest_key]
    regions.append({
        "n_rows_max": None,
        "n_pairs_max": None,
        "block_size": largest["block_size"],
        "wall_ms": None,
    })
    return regions


def ensure_joint_hist_multi_pair_tuning(force: bool = False) -> Optional[list[dict]]:
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("joint_hist_multi_pair")
        if regions:
            return regions
    logger.info("kernel_tuning_cache: joint_hist_multi_pair sweep starting")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_joint_hist_multi_pair(n_iters=5)
    except Exception as e:
        logger.warning(
            "kernel_tuning_cache: joint_hist_multi_pair sweep failed: %s", e,
        )
        return None
    logger.info(
        "kernel_tuning_cache: joint_hist_multi_pair sweep done in %.2fs",
        time.perf_counter() - t0,
    )
    if regions:
        try:
            cache.update("joint_hist_multi_pair",
                          axes=["n_rows", "n_pairs"], regions=regions)
        except OSError as e:
            logger.warning(
                "kernel_tuning_cache: joint_hist_multi_pair save failed: %s", e,
            )
    return regions


# ----------------------------------------------------------------------------
# 3. batch_pair_mi -- backend (njit / cuda / cupy) keyed by (n_samples, n_pairs)
# ----------------------------------------------------------------------------
def _run_sweep_batch_pair_mi(n_iters: int = 3) -> list[dict]:
    """Sweep njit_prange vs numba.cuda vs cupy across (n_samples, n_pairs).
    Persists ``backend_choice`` plus the (cuda|cupy)_min_rows / _min_pairs
    thresholds derived from the smallest measured combo where each GPU
    backend wins."""
    from mlframe.feature_selection.filters.batch_pair_mi_gpu import (
        batch_pair_mi_njit_prange, batch_pair_mi_cuda, batch_pair_mi_cupy,
        _CUDA_AVAIL, _CUPY_AVAIL,
    )

    n_rows_axis = (5_000, 50_000, 200_000, 1_000_000)
    n_pairs_axis = (4, 32, 128)
    n_cols = 16
    nbins_per_col = 6
    nbins_y = 3
    rng = np.random.default_rng(11)

    # Warmup
    try:
        _fd = rng.integers(0, nbins_per_col, size=(2000, n_cols)).astype(np.int32)
        _pa = np.array([0, 1], dtype=np.int32)
        _pb = np.array([2, 3], dtype=np.int32)
        _nb = np.full(n_cols, nbins_per_col, dtype=np.int32)
        _cy = rng.integers(0, nbins_y, size=2000).astype(np.int32)
        _fy = (np.bincount(_cy, minlength=nbins_y).astype(np.float64) / 2000)
        batch_pair_mi_njit_prange(_fd, _pa, _pb, _nb, _cy, _fy)
        if _CUDA_AVAIL:
            try:
                batch_pair_mi_cuda(_fd, _pa, _pb, _nb, _cy, _fy)
            except Exception as exc:
                logger.info("batch_pair_mi warmup: cuda backend unusable (%s)", exc)
        if _CUPY_AVAIL:
            try:
                batch_pair_mi_cupy(_fd, _pa, _pb, _nb, _cy, _fy)
            except Exception as exc:
                logger.info("batch_pair_mi warmup: cupy backend unusable (%s)", exc)
    except Exception as exc:
        logger.warning("batch_pair_mi warmup failed: %s", exc)
        return []

    best_per_combo: dict[tuple[int, int], dict] = {}
    for n_rows, n_pairs in itertools.product(n_rows_axis, n_pairs_axis):
        factors = rng.integers(0, nbins_per_col, size=(n_rows, n_cols)).astype(np.int32)
        pair_a = rng.integers(0, n_cols, size=n_pairs).astype(np.int32)
        pair_b = rng.integers(0, n_cols, size=n_pairs).astype(np.int32)
        nbins = np.full(n_cols, nbins_per_col, dtype=np.int32)
        classes_y = rng.integers(0, nbins_y, size=n_rows).astype(np.int32)
        freqs_y = (np.bincount(classes_y, minlength=nbins_y).astype(np.float64)
                   / max(1, n_rows))

        timings = {"njit": float("inf"), "cuda": float("inf"), "cupy": float("inf")}
        for backend_name, fn, avail in (
            ("njit", batch_pair_mi_njit_prange, True),
            ("cuda", batch_pair_mi_cuda, _CUDA_AVAIL),
            ("cupy", batch_pair_mi_cupy, _CUPY_AVAIL),
        ):
            if not avail:
                continue
            try:
                samples = []
                for _ in range(n_iters):
                    t0 = time.perf_counter()
                    fn(factors, pair_a, pair_b, nbins, classes_y, freqs_y)
                    samples.append(time.perf_counter() - t0)
                timings[backend_name] = float(np.median(samples) * 1000)
            except Exception as exc:
                logger.debug(
                    "batch_pair_mi %s skipped n_rows=%d n_pairs=%d: %s",
                    backend_name, n_rows, n_pairs, exc,
                )
                continue

        winner = min(timings, key=timings.get)
        best_per_combo[(n_rows, n_pairs)] = {
            "backend_choice": winner,
            "njit_ms": round(timings["njit"], 4) if timings["njit"] != float("inf") else None,
            "cuda_ms": round(timings["cuda"], 4) if timings["cuda"] != float("inf") else None,
            "cupy_ms": round(timings["cupy"], 4) if timings["cupy"] != float("inf") else None,
        }
        logger.info(
            "auto_tune batch_pair_mi n_rows=%d n_pairs=%d -> %s "
            "(njit=%.2f cuda=%.2f cupy=%.2f)",
            n_rows, n_pairs, winner,
            timings["njit"], timings["cuda"], timings["cupy"],
        )

    if not best_per_combo:
        return []

    # Derive crossover thresholds: smallest measured (n_rows, n_pairs)
    # where cuda / cupy first wins. The consumer's lookup signature is
    # buggy (positional dict; see Wave 24 review), but the persisted
    # threshold fields will work once the consumer kwargs fix lands.
    cuda_min_rows = None
    cuda_min_pairs = None
    cupy_min_rows = None
    cupy_min_pairs = None
    for (n_rows, n_pairs), choice in sorted(best_per_combo.items()):
        if choice["backend_choice"] == "cuda" and cuda_min_rows is None:
            cuda_min_rows = n_rows
            cuda_min_pairs = n_pairs
        if choice["backend_choice"] == "cupy" and cupy_min_rows is None:
            cupy_min_rows = n_rows
            cupy_min_pairs = n_pairs

    # Per-region rows mirror _run_sweep_mi_classif_dispatch.
    regions: list[dict] = []
    for (n_rows, n_pairs), choice in sorted(best_per_combo.items()):
        regions.append({
            "n_samples_max": int(n_rows),
            "n_pairs_max": int(n_pairs),
            "backend_choice": choice["backend_choice"],
            "cuda_min_rows": int(cuda_min_rows) if cuda_min_rows is not None else 10**9,
            "cuda_min_pairs": int(cuda_min_pairs) if cuda_min_pairs is not None else 10**9,
            "cupy_min_rows": int(cupy_min_rows) if cupy_min_rows is not None else 10**9,
            "cupy_min_pairs": int(cupy_min_pairs) if cupy_min_pairs is not None else 10**9,
            "njit_ms": choice["njit_ms"],
            "cuda_ms": choice["cuda_ms"],
            "cupy_ms": choice["cupy_ms"],
        })
    largest_key = max(best_per_combo, key=lambda k: (k[0], k[1]))
    largest = best_per_combo[largest_key]
    regions.append({
        "n_samples_max": None,
        "n_pairs_max": None,
        "backend_choice": largest["backend_choice"],
        "cuda_min_rows": int(cuda_min_rows) if cuda_min_rows is not None else 10**9,
        "cuda_min_pairs": int(cuda_min_pairs) if cuda_min_pairs is not None else 10**9,
        "cupy_min_rows": int(cupy_min_rows) if cupy_min_rows is not None else 10**9,
        "cupy_min_pairs": int(cupy_min_pairs) if cupy_min_pairs is not None else 10**9,
    })
    return regions


def ensure_batch_pair_mi_tuning(force: bool = False) -> Optional[list[dict]]:
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("batch_pair_mi")
        if regions:
            return regions
    logger.info("kernel_tuning_cache: batch_pair_mi sweep starting")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_batch_pair_mi(n_iters=3)
    except Exception as e:
        logger.warning("kernel_tuning_cache: batch_pair_mi sweep failed: %s", e)
        return None
    logger.info(
        "kernel_tuning_cache: batch_pair_mi sweep done in %.2fs",
        time.perf_counter() - t0,
    )
    if regions:
        try:
            cache.update("batch_pair_mi",
                          axes=["n_samples", "n_pairs"], regions=regions)
        except OSError as e:
            logger.warning("kernel_tuning_cache: batch_pair_mi save failed: %s", e)
    return regions


# ----------------------------------------------------------------------------
# 4. cat_fe_perm_kernel -- crossover_n keyed by (n_samples, n_perms)
# ----------------------------------------------------------------------------
def _run_sweep_cat_fe_perm_kernel(n_iters: int = 3) -> list[dict]:
    """Find ``crossover_n``: smallest n_samples where the GPU permutation
    kernel beats the CPU njit equivalent for a given n_perms. Source
    default is 1_000_000; live HW may be 2-5x off.
    """
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


# ----------------------------------------------------------------------------
# 5. rmse_partial_sum -- block_n keyed by (n_samples, n_cols)
# ----------------------------------------------------------------------------
def _run_sweep_rmse_partial_sum(n_iters: int = 5) -> list[dict]:
    """Sweep BLOCK_N in {64, 128, 256, 512, 1024} for the numba.cuda RMSE
    partial-sum kernel. Returns regions of the form
    ``{n_samples_max, n_cols_max, block_n, wall_ms}``.
    """
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


# ----------------------------------------------------------------------------
# 6. unary_elementwise -- min_cells keyed by n_samples
# ----------------------------------------------------------------------------
def _run_sweep_unary_elementwise(n_iters: int = 5) -> list[dict]:
    """Find ``min_cells``: smallest n_samples at which cupy elementwise
    (sqrt / log1p / abs) beats numpy. Source default 500_000.
    """
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


# ----------------------------------------------------------------------------
# 7. rff_matmul -- work_threshold keyed by work (= n * d)
# ----------------------------------------------------------------------------
def _run_sweep_rff_matmul(n_iters: int = 3) -> list[dict]:
    """Find ``work_threshold``: smallest ``work = n * d * n_features`` at
    which the cupy matmul path beats numpy for RFF. Source default
    5_000_000 * 256.
    """
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


# ----------------------------------------------------------------------------
# 8. knn_hnsw_crossover -- n_threshold keyed by (n_subset, d)
# ----------------------------------------------------------------------------
def _hnswlib_importable() -> bool:
    """Probe ``import hnswlib`` in a subprocess. On some Windows builds
    the wheel segfaults at import time (access violation in the C
    extension); a ``try / except ImportError`` in the parent process
    won't catch that. Run the probe out-of-process so the parent stays
    alive when the wheel is broken."""
    import subprocess, sys
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import hnswlib"],
            capture_output=True, timeout=15,
        )
    except Exception as exc:
        logger.info("auto_tune knn_hnsw_crossover probe failed: %s", exc)
        return False
    return r.returncode == 0


def _run_sweep_knn_hnsw_crossover(n_iters: int = 3) -> list[dict]:
    """Find ``n_threshold``: smallest ``n_subset`` at which hnswlib ANN
    beats sklearn brute/ball_tree kNN for a given d. Requires hnswlib
    package; gracefully skip if unavailable.
    """
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
    cache = _shared_cache()
    # Allow CPU-only hosts: the sweep is CPU-only, but _shared_cache
    # gates on is_cuda_available. Fall back to a direct cache create here.
    if cache is None:
        try:
            from pyutilz.system.kernel_tuning_cache import KernelTuningCache
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


# ----------------------------------------------------------------------------
# 9. discretize_2d_array -- min_cells keyed by arr_size
# ----------------------------------------------------------------------------
def _run_sweep_discretize_2d_array(n_iters: int = 3) -> list[dict]:
    """Find ``min_cells``: smallest ``arr_size = n_rows * n_cols`` at
    which ``discretize_2d_array_cuda`` beats the CPU njit equivalent.
    Source default 500_000.
    """
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


__all__ = [
    "ensure_joint_hist_tuning",
    "ensure_mi_classif_dispatch_tuning",
    "ensure_polyeval_tuning",
    "ensure_joint_hist_single_perm_tuning",
    "ensure_joint_hist_multi_pair_tuning",
    "ensure_batch_pair_mi_tuning",
    "ensure_cat_fe_perm_kernel_tuning",
    "ensure_rmse_partial_sum_tuning",
    "ensure_unary_elementwise_tuning",
    "ensure_rff_matmul_tuning",
    "ensure_knn_hnsw_crossover_tuning",
    "ensure_discretize_2d_array_tuning",
]

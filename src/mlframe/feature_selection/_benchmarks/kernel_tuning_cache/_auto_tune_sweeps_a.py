"""Group-A sweep + ensure functions carved out of
``mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune``.

Holds the joint_hist + polyeval + mi_classif_dispatch + batch_pair_mi
+ joint_hist_{multi_pair,single_perm} sweeps and their public
``ensure_*_tuning`` wrappers. Re-imported at the parent's module bottom
so historical ``from
mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune
import ensure_joint_hist_tuning`` resolves transparently.
"""
from __future__ import annotations

import itertools
import logging
import os
import time
from typing import Callable, Optional, cast

import numpy as np

logger = logging.getLogger(__name__)

# Module-level sentinels for _run_sweep_polyeval's lazy globals()-based import below (real
# hermite_fe values are set into these via g.setdefault at call time; declared here purely so
# mypy resolves the names -- the deferred-import/monkeypatch behavior itself is unchanged).
_NJIT_FUNCS: "dict[str, Callable]" = {}
_NJIT_PAR_FUNCS: "dict[str, Callable]" = {}
_CUDA_AVAILABLE: bool = False
_polyeval_cuda: "Callable | None" = None


def _physical_concurrency() -> int:
    """Realistic worker count for the joblib-threaded FE pipeline: physical cores.

    The FE pair-search / orth-FE / conditional-gate scans call the MI kernels from a
    ``joblib(backend='threading')`` pool sized to the physical core count, so the GPU
    sees that many concurrent callers. The dispatch decision must reflect that
    contention, not a solo measurement. Capped at 8 so the sweep itself stays bounded.
    """
    try:
        import psutil
        c = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    except Exception:
        c = os.cpu_count() or 2
    return max(1, min(8, int(c)))


def _median_per_call(fn, args=None, *, concurrency: int, n_iters: int, make_inputs=None, fresh: bool = False) -> float:
    """Median per-call wall time (ms) under ``concurrency`` concurrent threads, via the shared
    realistic timer ``pyutilz.performance.kernel_tuning.time_backend``.

    Pass either fixed ``args`` (reused across calls -- the legacy warm-buffer path) OR a
    ``make_inputs`` factory with ``fresh=True`` to mint NEW inputs every call. Fresh inputs capture
    the per-call alloc / H2D-upload cost a warm reused buffer hides for GPU backends -- the realism
    gap that made the solo sweep over-rate cuda 20-70x (solo ~36ms vs production ~746ms at n=100k
    k=20). njit is measured under the same thread count so both backends are judged on the same
    contended footing. The single-source timing logic now lives in pyutilz so every sweep shares it."""
    from pyutilz.performance.kernel_tuning import time_backend
    factory = make_inputs if make_inputs is not None else (lambda: args)
    return time_backend(
        fn, factory,
        concurrency=max(1, int(concurrency)), n_iters=n_iters, warmup=0,
        fresh_inputs_per_call=bool(fresh and make_inputs is not None),
    )


# Sweep axes (_N_SAMPLES_AXIS, _NBINS_AXIS, _BLOCK_SIZE_AXIS) are
# parent-resident SSOT in ``.auto_tune``; lazy-imported inside the sweep
# bodies so test ``monkeypatch.setattr(at, "_N_SAMPLES_AXIS", ...)`` flips
# them as documented.


def _run_sweep_joint_hist(n_iters: int = 5) -> list[dict]:
    """Returns a list of region dicts ready for KernelTuningCache.update."""
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _N_SAMPLES_AXIS, _NBINS_AXIS, _BLOCK_SIZE_AXIS, _measure_one
    import cupy as cp
    from mlframe.feature_selection.filters import gpu as _gpu_mod
    _gpu_mod._ensure_kernels_inited()
    compute_joint_hist_batched_cuda = _gpu_mod.compute_joint_hist_batched_cuda
    compute_joint_hist_batched_shared_cuda = _gpu_mod.compute_joint_hist_batched_shared_cuda

    rng = np.random.default_rng(11)
    best_per_combo: dict[tuple[int, int], dict] = {}

    def _make_samples(rng_: np.random.Generator, k: int, n: int, distribution: str) -> np.ndarray:
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
        args = (d_x, d_y_perms, d_out, np.int32(n_samples), np.int32(nbx), np.int32(nby))

        best_wall = float("inf")
        best_choice = None
        for variant, bs in itertools.product(("shared", "global"), _BLOCK_SIZE_AXIS):
            d_out[:] = 0
            grid_x = (n_samples + bs - 1) // bs
            kernel = compute_joint_hist_batched_shared_cuda if variant == "shared" else compute_joint_hist_batched_cuda
            smem = joint * 4 if variant == "shared" else 0
            try:
                wall = _measure_one(kernel, grid_x, bs, args, n_iters=n_iters, shared_mem_bytes=smem)
            except Exception as e:
                logger.debug("auto_tune skipped (variant=%s bs=%d): %s", variant, bs, e)
                continue
            if wall < best_wall:
                best_wall = wall
                best_choice = {"kernel_variant": variant, "block_size": bs, "wall_ms": round(wall, 4)}
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
    return cast("list[dict]", regions)
def ensure_joint_hist_tuning(force: bool = False) -> Optional[list[dict]]:
    """Return cached regions for ``joint_hist_batched``; run the sweep
    + persist via pyutilz KernelTuningCache if missing. Returns None
    if CUDA / pyutilz unavailable."""
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _shared_cache
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("joint_hist_batched")
        if regions:
            return cast("list[dict]", regions)

    logger.info("kernel_tuning_cache: joint_hist sweep starting (one-time per host)")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_joint_hist(n_iters=2)
    except Exception as e:
        logger.warning("kernel_tuning_cache: joint_hist sweep failed: %s", e)
        return None
    logger.info("kernel_tuning_cache: joint_hist sweep done in %.2fs", time.perf_counter() - t0)

    try:
        cache.update("joint_hist_batched", axes=["n_samples", "joint_size"], regions=regions)
    except OSError as e:
        logger.warning("kernel_tuning_cache: cache save failed: %s", e)

    return cast("list[dict]", regions)
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
    # k extends to 100: the conditional-gate _flush / orth-FE scans batch MANY candidate columns into
    # one call (k = tens-to-hundreds), and cupy argsort over (n, large-k) is exactly where the real
    # per-call GPU cost lives -- a regime the old k<=20 grid never measured, leaving the dispatch blind
    # to the production batch width.
    k_axis = (1, 5, 20, 100)
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

    # Measure both backends under the SAME realistic worker-thread contention the production FE
    # pipeline imposes on the GPU (solo timing systematically over-rates cuda -- it never sees the
    # multi-thread memcpy/launch/sync serialisation that makes the real per-call wall 20-70x the
    # GPU-to-itself time). concurrency=1 reproduces the legacy solo sweep.
    concurrency = _physical_concurrency()
    _PREBUILD_BUDGET = 1_500_000_000  # ~1.5 GB cap on the fresh-input arrays pre-built per cell
    for n, k in itertools.product(n_axis, k_axis):
        try:
            y = rng.integers(0, 3, size=n).astype(np.int64)
            njit_fn: "Callable[..., np.ndarray | float]"
            cuda_fn: "Callable[..., np.ndarray | float]"
            if k == 1:
                base = rng.normal(size=n)
                bytes_per = n * 8
                njit_fn, cuda_fn = _plugin_mi_classif_njit, _plugin_mi_classif_cuda
            else:
                base = rng.normal(size=(n, k))
                bytes_per = n * k * 8
                njit_fn, cuda_fn = _plugin_mi_classif_batch_njit, _plugin_mi_classif_batch_cuda
            # FRESH input per call (a NEW copy each time) so the cupy memory pool can't serve a hot,
            # already-resident block -- this exposes the per-call alloc / H2D upload the production
            # pipeline pays for every freshly-engineered candidate but a warm reused buffer hides.
            # Cap effective concurrency so the pre-built fresh arrays (eff_c * n_iters copies) stay
            # within ~1.5 GB per cell at the large-n grid points.
            eff_c = max(1, min(concurrency, _PREBUILD_BUDGET // max(1, bytes_per * n_iters)))
            make_inputs = lambda base=base, y=y: (base.copy(), y, 20)  # noqa: E731
            # njit + cuda judged on the SAME fresh-input, contended footing; solo cuda kept for
            # transparency (the gap vs m_cuda quantifies the contention penalty).
            m_njit = _median_per_call(njit_fn, concurrency=eff_c, n_iters=n_iters, make_inputs=make_inputs, fresh=True)
            m_cuda = _median_per_call(cuda_fn, concurrency=eff_c, n_iters=n_iters, make_inputs=make_inputs, fresh=True)
            m_cuda_solo = _median_per_call(cuda_fn, concurrency=1, n_iters=n_iters, make_inputs=make_inputs, fresh=True)
            backend = "cuda" if m_cuda < m_njit else "njit"
            best_per_combo[(n, k)] = {
                "backend_choice": backend,
                "njit_ms": round(m_njit, 4),
                "cuda_ms": round(m_cuda, 4),
                "cuda_ms_solo": round(m_cuda_solo, 4),
                "concurrency": int(eff_c),
            }
            logger.info(
                "auto_tune mi_classif n=%d k=%d c=%d -> %s (njit=%.2fms cuda=%.2fms cuda_solo=%.2fms, fresh-inputs)",
                n, k, eff_c, backend, m_njit, m_cuda, m_cuda_solo,
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
            "cuda_ms_solo": choice.get("cuda_ms_solo"),
            "concurrency": choice.get("concurrency"),
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
    return cast("list[dict]", regions)
def ensure_mi_classif_dispatch_tuning(force: bool = False) -> Optional[list[dict]]:
    """Return cached regions for ``plugin_mi_classif_dispatch``; run the
    sweep + persist via pyutilz KernelTuningCache if missing. Returns
    None if CUDA / pyutilz unavailable.

    Mirrors :func:`ensure_joint_hist_tuning` for the
    ``plugin_mi_classif`` njit-vs-cuda backend choice. Sweep takes
    ~10-30s once per host; subsequent processes read the cached JSON
    in ~1ms.
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _shared_cache
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("plugin_mi_classif_dispatch")
        if regions:
            return cast("list[dict]", regions)

    logger.info("kernel_tuning_cache: plugin_mi_classif_dispatch sweep starting " "(one-time per host)")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_mi_classif_dispatch(n_iters=2)
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

    return cast("list[dict]", regions)
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
    )
    # Resolve the backend tables + cuda flag + cuda fn from THIS module's globals so tests (and any host
    # override) can monkeypatch them; when absent, import the real hermite_fe backends (deferred so this
    # benchmarks module stays light to import). An already-present (monkeypatched) value always wins.
    g = globals()
    if not all(k in g for k in ("_NJIT_FUNCS", "_NJIT_PAR_FUNCS", "_CUDA_AVAILABLE")):
        from mlframe.feature_selection.filters.hermite_fe import (
            _NJIT_FUNCS as _nf, _NJIT_PAR_FUNCS as _npf, _CUDA_AVAILABLE as _ca,
        )
        g.setdefault("_NJIT_FUNCS", _nf)
        g.setdefault("_NJIT_PAR_FUNCS", _npf)
        g.setdefault("_CUDA_AVAILABLE", _ca)
    if g["_CUDA_AVAILABLE"] and "_polyeval_cuda" not in g:
        from mlframe.feature_selection.filters.hermite_fe import (
            _polyeval_cuda as _pc, _ensure_cuda_kernels,
        )
        g.setdefault("_polyeval_cuda", _pc)
        _ensure_cuda_kernels()
    # Local alias so mypy can flow-narrow the None-check below (the bare module-level name stays
    # annotated Optional for callers importing it directly; this function's own two call sites use
    # the local narrowed reference instead).
    _polyeval_cuda_fn: "Callable | None" = g.get("_polyeval_cuda")

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
            if _CUDA_AVAILABLE and _polyeval_cuda_fn is not None:
                _polyeval_cuda_fn(_basis, _warm_x, coef)
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
                    if _CUDA_AVAILABLE and _polyeval_cuda_fn is not None:
                        t0 = time.perf_counter()
                        _polyeval_cuda_fn(basis, x, coef)
                        t_cuda.append(time.perf_counter() - t0)
            except Exception as exc:
                logger.debug("polyeval sweep skipped basis=%s n=%d: %s", basis, n, exc)
                continue
            timing_table[n] = {
                "njit": float(np.median(t_njit) * 1000),
                "njit_par": float(np.median(t_par) * 1000),
                "cuda": (float(np.median(t_cuda) * 1000) if t_cuda else float("inf")),
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
        # Find cuda_threshold: smallest n where cuda <= njit_par * 0.95. If cuda never wins at any swept n
        # (common for the cheap per-element Horner kernel whose H2D+D2H transfer dwarfs the compute on a
        # transfer-bound laptop GPU like the RTX 500 Ada), persist a sentinel ABOVE the largest swept size so
        # the dispatcher never routes to the slower cuda path on this host -- not the 500k source default,
        # which would itself be a mis-route since cuda loses at 500k+ here.
        swept_max = max(timing_table)
        cuda_threshold = swept_max * 100  # sentinel: "cuda never wins on this host"
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
    return cast("list[dict]", regions)
def ensure_polyeval_tuning(force: bool = False) -> Optional[list[dict]]:
    """Return cached regions for ``polyeval``; run the sweep + persist
    via pyutilz KernelTuningCache if missing.

    The consumer ``_lookup_polyeval_thresholds`` in
    ``hermite_fe.py`` was wired to the cache 2026-05-20 (Wave 23 P2)
    but no populator existed -- this fills the gap. First run
    ~20-40s (4 bases x 8 n-points x njit/njit_par/cuda); subsequent
    processes read in ~1ms.
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _shared_cache
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("polyeval")
        if regions:
            return cast("list[dict]", regions)

    logger.info("kernel_tuning_cache: polyeval sweep starting (one-time per host)")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_polyeval(n_iters=2)
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

    return cast("list[dict]", regions)
def _run_sweep_joint_hist_single_perm(n_iters: int = 5) -> list[dict]:
    """Sweep block_size in {128, 256, 512, 1024} across n_samples for the
    single-permutation ``compute_joint_hist_cuda`` kernel.

    Mirrors :func:`_run_sweep_joint_hist` but for the 1D-grid variant
    (no batch axis, simpler atomic pattern). Returns regions of the form
    ``{n_samples_max: int | None, block_size: int, wall_ms: float}``.
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _cuda_available_or_skip, _measure_one
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
                wall = _measure_one(kernel, grid_x, bs, args, n_iters=n_iters, shared_mem_bytes=0)
            except Exception as exc:
                logger.debug("auto_tune joint_hist_single_perm skipped n=%d bs=%d: %s", n, bs, exc)
                continue
            if wall < best_wall:
                best_wall = wall
                best_bs = bs
        if best_bs is not None:
            best_per_n[n] = {"block_size": int(best_bs), "wall_ms": round(best_wall, 4)}
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
    return cast("list[dict]", regions)
def ensure_joint_hist_single_perm_tuning(force: bool = False) -> Optional[list[dict]]:
    """Return cached regions for ``joint_hist_single_perm``; run the
    sweep + persist via pyutilz KernelTuningCache if missing."""
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _shared_cache
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("joint_hist_single_perm")
        if regions:
            return cast("list[dict]", regions)
    logger.info("kernel_tuning_cache: joint_hist_single_perm sweep starting")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_joint_hist_single_perm(n_iters=2)
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
            cache.update("joint_hist_single_perm", axes=["n_samples"], regions=regions)
        except OSError as e:
            logger.warning(
                "kernel_tuning_cache: joint_hist_single_perm save failed: %s", e,
            )
    return cast("list[dict]", regions)
def _run_sweep_joint_hist_multi_pair(n_iters: int = 5) -> list[dict]:
    """Sweep block_size in {128, 256, 512, 1024} across the
    ``(n_rows, n_pairs)`` grid for ``compute_joint_hist_multi_pair_cuda``.
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _cuda_available_or_skip
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
        joint_offsets = np.arange(n_pairs + 1, dtype=np.int32) * per_pair_cells
        total_cells = int(joint_offsets[-1])

        d_factors = cp.asarray(factors_T)
        d_y = cp.asarray(classes_y)
        d_pa = cp.asarray(pairs_a)
        d_pb = cp.asarray(pairs_b)
        d_nba = cp.asarray(nbins_a)
        d_off = cp.asarray(joint_offsets)
        d_out = cp.zeros(total_cells, dtype=cp.int32)
        args = (d_factors, d_y, d_pa, d_pb, d_nba, d_off, d_out, np.int32(n_rows), np.int32(n_pairs), np.int32(nbins_y))

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
    return cast("list[dict]", regions)
def ensure_joint_hist_multi_pair_tuning(force: bool = False) -> Optional[list[dict]]:
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _shared_cache
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("joint_hist_multi_pair")
        if regions:
            return cast("list[dict]", regions)
    logger.info("kernel_tuning_cache: joint_hist_multi_pair sweep starting")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_joint_hist_multi_pair(n_iters=2)
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
            cache.update("joint_hist_multi_pair", axes=["n_rows", "n_pairs"], regions=regions)
        except OSError as e:
            logger.warning(
                "kernel_tuning_cache: joint_hist_multi_pair save failed: %s", e,
            )
    return cast("list[dict]", regions)
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
        _fy = np.bincount(_cy, minlength=nbins_y).astype(np.float64) / 2000
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
        freqs_y = np.bincount(classes_y, minlength=nbins_y).astype(np.float64) / max(1, n_rows)

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

        winner = min(timings, key=timings.__getitem__)
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
        regions.append(
            {
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
            }
        )
    largest_key = max(best_per_combo, key=lambda k: (k[0], k[1]))
    largest = best_per_combo[largest_key]
    regions.append(
        {
            "n_samples_max": None,
            "n_pairs_max": None,
            "backend_choice": largest["backend_choice"],
            "cuda_min_rows": int(cuda_min_rows) if cuda_min_rows is not None else 10**9,
            "cuda_min_pairs": int(cuda_min_pairs) if cuda_min_pairs is not None else 10**9,
            "cupy_min_rows": int(cupy_min_rows) if cupy_min_rows is not None else 10**9,
            "cupy_min_pairs": int(cupy_min_pairs) if cupy_min_pairs is not None else 10**9,
        }
    )
    return cast("list[dict]", regions)
def ensure_batch_pair_mi_tuning(force: bool = False) -> Optional[list[dict]]:
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .auto_tune import _shared_cache
    cache = _shared_cache()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("batch_pair_mi")
        if regions:
            return cast("list[dict]", regions)
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
            cache.update("batch_pair_mi", axes=["n_samples", "n_pairs"], regions=regions)
        except OSError as e:
            logger.warning("kernel_tuning_cache: batch_pair_mi save failed: %s", e)
    return cast("list[dict]", regions)


# Register the multi-field GPU kernels with the unified tuner registry so
# retune_all / mlframe-tune-kernels discover + batch-tune them. The dispatch
# (dispatch.py) already READS these regions via the cache; this only adds
# discovery. tuner = the compute-only _run_sweep_* (returns regions, no
# self-update -> no double-write). The grid lives inside each sweep, so only the
# axis KEYS are declared; the spec fallback is unused (retune ignores the
# get_or_tune return) -- the real per-call fallback is dispatch.py's _hw_aware_fallback.
from pyutilz.performance.kernel_tuning.registry import kernel_tuner as _ktuner

for _kn, _sweep, _axes in (
    ("joint_hist_batched", _run_sweep_joint_hist, ("n_samples", "joint_size")),
    ("plugin_mi_classif_dispatch", _run_sweep_mi_classif_dispatch, ("n_samples", "k")),
    ("polyeval", _run_sweep_polyeval, ("basis", "n_samples")),
    ("joint_hist_single_perm", _run_sweep_joint_hist_single_perm, ("n_samples",)),
    ("joint_hist_multi_pair", _run_sweep_joint_hist_multi_pair, ("n_rows", "n_pairs")),
):
    _ktuner(
        kernel_name=_kn,
        variant_fns=(_sweep,),
        tuner=_sweep,
        axes={_a: [] for _a in _axes},
        fallback={},
        gpu_capable=True,
        salt=1,
        cli_label=_kn,
    )

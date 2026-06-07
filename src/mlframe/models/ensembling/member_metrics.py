"""Per-member MAE/STD metric dispatcher carved out of ``_ensembling_base``.

Holds the HW-calibrated numpy-vs-numba backend selector (``_per_member_use_numba``) and the public per-member reduction (``_per_member_mae_std``) plus its kernel-tuning-cache constants. The numba kernel (``_per_member_mae_std_njit``) and the ``_HAS_NUMBA_PER_MEMBER`` probe stay in ``_ensembling_base`` (they share a try/except with the RRF kernel) and are imported here. The parent re-exports the moved symbols so external importers keep resolving them.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np

logger = logging.getLogger("mlframe.models.ensembling")

_PER_MEMBER_KERNEL_NAME = "per_member_mae_std"
# Below this element count (= N for the 2-D path) numpy and numba are a sub-ms
# wash, so skip the one-time numba cache-load; at/above it numba wins decisively.
_PER_MEMBER_NUMBA_FLOOR_ELEMENTS = 10_000


@lru_cache(maxsize=256)
def _per_member_use_numba(elements_per_member: int, n_groups: int, ndim: int = 2) -> bool:
    """Pick the numba-vs-numpy backend for the ``_per_member_mae_std`` path (2-D or 3-D).

    The crossover is HW-dependent, so this follows the project dispatch
    convention (mirrors ``feature_engineering._recursion_dispatch``): env
    override -> per-host ``kernel_tuning_cache`` (populated by
    ``training._benchmarks.per_member_tuning.ensure_per_member_tuning`` once per
    machine) -> measurement-backed fallback.

    The fallback defaults to numba whenever it is available and the input clears
    a small element floor: the njit prange kernel beats the numpy broadcast
    5-18x across the ENTIRE 2-D regime on multi-core HW (bench 2026-06-04,
    6-core: K=2-20 / N=20k-500k -> numpy 2-128 ms vs numba 0.4-7 ms; a sub-ms
    tie only below ~N=5k). 2-D numba == numpy to ~1e-14 (both two-pass), so this
    is a pure-speed swap. The prior ``K>20 or elements>500k`` gate was a
    single-machine artifact that hid the win for the common ensembling sizes
    (K~3-10, N~10-100k).

    3-D (K, N, C) dispatches here too -- ``ndim`` is carried in the cache key so
    2-D and 3-D CAN tune separately, though the current sweep varies only
    ``elements_per_member`` and both share that crossover. The 3-D per-column njit
    is bit-identical to numpy (max abs diff 0.0) and 12-22x faster (measured
    2026-06-05), so numba is the default above the same floor.

    GPU: cupy was measured under BOTH residencies (2026-06-05, GTX 1050 Ti, with
    per-call synchronize). DRAM-resident loses badly (H2D transfer-bound: 31 ms
    vs numba 6 ms at K=8/N=1M). VRAM-resident (data already on device) skips the
    transfer and is ~3x faster than DRAM cupy -- but STILL loses to numba on this
    GPU (11 vs 6 ms): a 768-core 2014 card can't out-compute multi-core prange on
    an axis-1 reduction. So no GPU variant is dispatched here. This is
    residency-AND-HW-dependent, though: on a modern GPU (10-100x this compute)
    VRAM-resident cupy would likely win, which is exactly what the cache's
    ``location`` (host/device) residency axis is built to capture per host.
    (Caution: an earlier async measurement timed kernel *launch* not completion
    and falsely showed cupy_VRAM winning 16x -- always synchronize GPU timings.)
    """
    from .base import _HAS_NUMBA_PER_MEMBER
    if not _HAS_NUMBA_PER_MEMBER:
        return False
    env = os.environ.get("MLFRAME_PER_MEMBER_BACKEND", "").strip().lower()
    if env in ("numpy", "numba"):
        return env == "numba"
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
        from .per_member_tuning import run_per_member_sweep, per_member_code_version
        autotune = os.environ.get("MLFRAME_PER_MEMBER_AUTOTUNE", "1").strip() != "0"
        # Shared orchestrator: env override -> per-host cache (code-version
        # checked) -> on-miss sweep (once/process, cross-process locked, logs
        # winners + persists to ~/.pyutilz/kernel_tuning/<fp>.json) ->
        # measurement-backed fallback. Replaces the hand-rolled
        # lookup/miss/sweep/re-lookup dance + the module-level _AUTOTUNE guard
        # (get_or_tune keys its once-per-process guard on (kernel, cache-path)).
        result = KernelTuningCache.load_or_create().get_or_tune(
            _PER_MEMBER_KERNEL_NAME,
            dims={"elements_per_member": elements_per_member, "n_groups": n_groups, "ndim": ndim},
            tuner=(lambda: run_per_member_sweep(observed_elements=elements_per_member)) if autotune else (lambda: None),
            axes=["elements_per_member", "n_groups", "ndim"],
            fallback={"backend_choice": "numba" if elements_per_member >= _PER_MEMBER_NUMBA_FLOOR_ELEMENTS else "numpy"},
            env_key="MLFRAME_PER_MEMBER_BACKEND",
            code_version=per_member_code_version(),
        )
        # env_key short-circuits to the raw string "numpy"/"numba"; the sweep or
        # fallback return a region dict with "backend_choice".
        if isinstance(result, str):
            return result.strip().lower() == "numba"
        return str((result or {}).get("backend_choice", "")) == "numba"
    except Exception as e:  # pyutilz missing / cache error -> measurement-backed fallback
        logger.debug("per_member backend get_or_tune failed: %s", e)
    return elements_per_member >= _PER_MEMBER_NUMBA_FLOOR_ELEMENTS


def _per_member_mae_std(arr: np.ndarray, median_preds: np.ndarray) -> tuple:
    """Vectorised per-member MAE / STD of |arr - median_preds| reduced to one scalar per member.

    Semantics match the prior Python loop: per-column MAE first (mean over the N axis), then mean
    across remaining columns; per-column std uses the per-column mean as anchor and is then averaged
    across columns. For 2-D (K, N) inputs columns degenerate to one value, so the result is the
    same as a flat mean / std. Both 2-D and 3-D are HW-calibrated by
    ``_per_member_use_numba`` (env -> per-host kernel_tuning_cache -> measured
    fallback); on multi-core HW numba wins 5-18x (2-D) / 12-22x (3-D) so it is
    the default above a small element floor (the old ``K>20 or elements>500k``
    gate was a single-machine artifact). The 3-D numba branch was fixed to the
    per-column std so it is now bit-identical to numpy; the numpy path here is
    the fallback / sub-floor tier.
    """
    from .base import _HAS_NUMBA_PER_MEMBER, _per_member_mae_std_njit
    K = arr.shape[0]
    elements_per_member = int(arr.size // max(K, 1))
    # Both 2-D and 3-D dispatch to numba now. The 3-D njit branch was fixed to
    # the per-COLUMN std (matching numpy's per-class-then-mean-across-C path)
    # instead of the old pooled N*C std -- so it is bit-identical (max abs diff
    # 0.0, verified 2026-06-05) and 12-22x faster, even larger than the 2-D win.
    # ``ndim`` is threaded into the cache key so 2-D and 3-D CAN tune separately,
    # but the current sweep varies only elements_per_member, so both share that
    # crossover. HW-calibrated via _per_member_use_numba (env ->
    # kernel_tuning_cache -> measured fallback).
    use_numba = (
        _HAS_NUMBA_PER_MEMBER
        and arr.dtype == np.float64
        and arr.ndim in (2, 3)
        and _per_member_use_numba(elements_per_member, K, arr.ndim)
    )
    if use_numba:
        return _per_member_mae_std_njit(arr, median_preds)
    # bench-attempt-rejected (2026-05-21, c0123 K=3 / N=200k): single-pass
    # variance via ``E[X^2] - E[X]^2`` (avoids the (K, N) squared-diff
    # intermediate) saved 15-26 % wall on numpy paths (2-D 7.5 -> 6.4 ms;
    # 3-D 25.6 -> 20.4 ms) at <= 9e-15 max abs std diff. Rejected to match
    # the numba branch's explicit "no catastrophic cancellation" guarantee:
    # ``E[X^2] - E[X]^2`` loses precision when std/mean is tiny (members
    # nearly identical), and we never want the perf-tier numpy path to
    # silently shift the gate decision under that regime.
    diffs = np.abs(arr - median_preds)
    if arr.ndim == 2:
        # (K, N): one column, mae/std collapse to a single scalar per member.
        per_member_mae = diffs.mean(axis=1)
        # ``np.var`` computes ``mean((x - mean(x))**2)`` via the SAME centered
        # two-pass numpy always uses -- NOT the rejected single-pass
        # ``E[X^2] - E[X]^2`` form -- so it is bit-identical (max abs diff 0.0,
        # verified) to the prior explicit expression while skipping the
        # Python-level (K, N) squared-diff intermediate and the second mean
        # recompute. Bench (iter, /loop microopt): K=3/N=100k 2.16->1.60 ms
        # (1.35x), K=20/N=500k 78.1->69.3 ms (1.13x). The 3-D branch keeps the
        # explicit form below -- np.var over a non-contiguous axis-1 reduction
        # there is ~6 % SLOWER (measured 0.92-0.94x), so it is deliberately not
        # converted.
        per_member_std = np.sqrt(np.var(diffs, axis=1))
    else:
        # (K, N, C) -- mae_per_col across N, then mean across C; same for std.
        mae_per_col = diffs.mean(axis=1)  # (K, C)
        std_per_col = np.sqrt(((diffs - mae_per_col[:, None, :]) ** 2).mean(axis=1))  # (K, C)
        per_member_mae = mae_per_col.mean(axis=1)
        per_member_std = std_per_col.mean(axis=1)
    return per_member_mae, per_member_std

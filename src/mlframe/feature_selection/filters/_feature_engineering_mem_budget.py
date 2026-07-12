"""RAM-budget / hoist-vs-recompute decision logic for the FE pair-search shared buffer.

Carved out of ``feature_engineering.py`` (which re-exports these names) to keep that
module under the repo's 1k-LOC gate. See ``_can_hoist_shared_buffer`` for the entry
point consumed by ``_feature_engineering_pairs``.
"""
from __future__ import annotations

import logging
import sys
import threading

# Same logger name as the parent ``feature_engineering`` module (not ``__name__``): callers
# and tests (``caplog.at_level(..., logger=fe_mod.logger.name)``) filter on the PARENT's
# logger name, which is the pre-split single-file module's historical logger identity.
logger = logging.getLogger("mlframe.feature_selection.filters.feature_engineering")


def _pmod():
    """The parent ``feature_engineering`` module, if already imported, else ``None``.

    Callers (and their tests) monkeypatch the mutable RAM-budget knobs on the PARENT
    module's re-exported names (its public surface), not on this sibling. Every read/write
    of a monkeypatch-able name below goes through ``_get_live``/``_set_live`` so a
    parent-side patch is observed here, and a cache write here is visible back on the
    parent's re-exported attribute too. Deferred (call-time, via ``sys.modules``) rather
    than a top-level ``from . import feature_engineering`` -- the parent imports THIS module
    at its own load time, so a top-level back-import would be a hard cycle; by call time both
    modules are fully initialised."""
    return sys.modules.get(f"{__package__}.feature_engineering")


def _get_live(name):
    """Current value of module-global ``name``: the parent's re-exported attribute if the
    parent module is loaded and monkeypatched (or just holds the live value), else this
    module's own global."""
    mod = _pmod()
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return globals()[name]


def _set_live(name, value) -> None:
    """Write ``name`` both locally and onto the parent's re-exported attribute (if loaded),
    so a value set here (e.g. the OPT5 vmem cache) is visible through ``fe_mod.<name>`` too."""
    globals()[name] = value
    mod = _pmod()
    if mod is not None:
        try:
            setattr(mod, name, value)
        except Exception:  # nosec B110 - best-effort sync, non-fatal by design
            pass

# Wave 27 P1 (2026-05-20): ``check_prospective_fe_pairs`` is dispatched via
# ``parallel_run`` from mrmr.py with backend='threading'. The function
# accumulates per-binary-transform timings into a shared ``times_spent``
# defaultdict via ``+=``. Python's ``+=`` on a float is load-add-store and
# NOT atomic even under the GIL between threads; concurrent workers can
# drop updates silently, under-reporting the diagnostic at mrmr.py:1691.
# This module-level lock serialises the increment; threading workers
# synchronise correctly. Under loky/spawn each worker gets its own
# defaultdict copy (no shared state); the lock has no effect there but
# also doesn't break.
_TIMES_SPENT_LOCK = threading.Lock()

# CRITICAL: the hoisted shared buffer at
# ``check_prospective_fe_pairs`` allocates ``(n, max_n_combs * len(binary))``
# float32. With n=4M and the medium preset that's ~17.6 GiB -- production
# MRMR crashed with numpy.core._exceptions._ArrayMemoryError on a real run.
# The hoist landed in Wave Pack G (commit 068acdd) under small-n benchmarks
# and never measured peak RAM on million-row data.
#
# Two-strategy dispatch:
#   Fast path (current): if buffer < ``_FE_BUFFER_RAM_BUDGET_RATIO`` * available
#     RAM, allocate the shared buffer and use the hoist (cheapest if it fits).
#   Recompute fallback: drop the multi-column buffer, scratch into a fresh 1D
#     ``np.empty(n, float32)`` per inner iteration, and rebuild the ~10
#     survivor columns from their (transformations_pair, bin_func_name) metadata
#     after the inner loop. Extra recompute cost: ~K bin_func calls per pair
#     (K = num survivors, typically <= fe_max_pair_features + |leading|);
#     <= 1% of the ~max_combs*|binary| calls already done in the inner loop.
#
# Subsample path remains a separate opt-in (``subsample_n`` parameter); this
# memory dispatcher is the deterministic, accuracy-preserving fallback that
# auto-engages when the shared buffer would OOM.
_FE_BUFFER_RAM_BUDGET_RATIO: float = 0.3

# LARGE-N PEAK-MEMORY FIX (2026-06-08). The ``_FE_BUFFER_RAM_BUDGET_RATIO`` budget was
# applied PER buffer in isolation -- it sized the cross-pair ``_chunk_buffer`` (n x W
# float32) to fill ~0.4*available, but did NOT account for the buffers that are ALIVE AT
# THE SAME TIME while that chunk is scored:
#   (1) the discretised int8/int16 code matrix produced by ``discretize_2d_quantile_batch``
#       over the chunk slice (n x W x {1,2} bytes);
#   (2) the ``batch_mi_with_noise_gate`` per-column working set + the MI output
#       (the ``_pairs_dispatch`` allocation -- ~n x W float32-class transient);
#   (3) the per-pair ``final_transformed_vals_shared`` buffer (n x max_n_combs*n_binary
#       float32) which is held ALIVE for the whole pair loop even when the chunk path is on.
# A tracemalloc n=100000 / 5-col / F1 snapshot at the native RSS peak measured:
#   _chunk_buffer 2225 MiB + final_transformed_vals_shared 742 MiB + disc 556 MiB +
#   batch-mi dispatch 556 MiB  =>  ~4.2 GiB peak from a chunk buffer the 0.4 budget
# thought was the ONLY large allocation. On an 8 GiB box (or once the joblib threading
# path multiplies the per-call buffers across workers) this OOMs in a worker with numba's
# "Allocation failed (probably too large)". The fix below caps the chunk/shared width so
# the SUM of the coexisting buffers stays inside the SAME 0.4*available envelope: divide the
# raw budget by ``_FE_PEAK_OVERHEAD_FACTOR`` (the measured coexisting-buffer multiple,
# rounded up for safety) AND by the number of CONCURRENTLY-RUNNING workers on the joblib
# ``backend="threading"`` path (each thread allocates its OWN buffers in the shared address
# space, so N threads each taking 0.4*available collectively need 0.4*N*available -- a
# guaranteed OOM for N>=3). Selection stays BYTE-IDENTICAL: this only changes the chunk
# WIDTH (how many candidate columns are materialised+scored per batched pass), never which
# candidates are produced, their order, codes, MI, or the per-pair best/tie-break replay --
# the chunk batching is already documented bit-identical to the per-pair path, and a
# narrower chunk simply means more (smaller) batched passes over the SAME candidates.
_FE_PEAK_OVERHEAD_FACTOR: float = 3.0

# HOIST-GATE RE-CALIBRATION (2026-06-24). The conservative ``_fe_effective_buffer_budget_bytes``
# envelope (raw 0.3*usable / overhead / n_workers) is the RIGHT cap for the chunk WIDTH and for
# the large-n OOM ceiling, but it OVER-DECLINES a buffer that is SMALL RELATIVE TO AVAILABLE RAM:
# on the F2 100k fit a 222.5 MiB single-pair buffer was declined with 5.7 GiB available, because
# 0.3*(5.7-3.0)GiB / (3.0 overhead * 2 workers) = ~138 MiB < 222.5 MiB. That decline forced 15370
# pairs onto the serial ``discretize_array(quantile)`` recompute-fallback path instead of the
# already-bit-parity-validated ``discretize_2d_quantile_batch`` batch path the hoisted buffer
# routes to. The decline was spurious: the buffer's REALISTIC peak footprint (buffer * overhead *
# n_workers = 1.3 GiB) leaves 4.4 GiB free -- far above the 3 GiB reserve -- so there was never any
# OOM risk. The fix (in ``_can_hoist_shared_buffer``) adds an ABSOLUTE-HEADROOM acceptance path:
# hoist when the buffer's peak footprint still leaves at least ``_fe_min_free_ram_bytes()`` of host
# RAM free, EVEN IF it exceeds the conservative relative budget. This moves the 15370 serial calls
# onto the validated batch path WITHOUT weakening the large-n protection: at large n the buffer
# footprint approaches ``available`` and the headroom path declines too (free-after < reserve), so
# the OOM ceiling is intact. Selection is unaffected -- the batch path the buffer routes to is
# documented byte-identical to the recompute path (the hoist/recompute decision only chooses HOW
# the survivors are produced, never WHICH). ``_FE_HOIST_HEADROOM_OVERHEAD`` is the peak-footprint
# multiplier for THIS headroom check (defaults to the same conservative overhead so the worst-case
# coexisting-buffer sum is accounted for); routed through the per-host kernel_tuning_cache so a box
# with measured-tighter or measured-looser real peaks can record its own value (never hardcode a
# per-HW memory threshold -- mlframe is shared infra). Override via env ``MLFRAME_FE_HOIST_HEADROOM_OVERHEAD``.
_FE_HOIST_HEADROOM_OVERHEAD: float = 3.0


def _fe_hoist_headroom_overhead() -> float:
    """Peak-footprint multiplier for the absolute-headroom hoist acceptance path.

    Resolution order: env ``MLFRAME_FE_HOIST_HEADROOM_OVERHEAD`` (if set + parseable and > 0)
    overrides the per-host ``kernel_tuning_cache`` entry under key ``fe_hoist_headroom_overhead``
    (field ``overhead``), which overrides the module constant ``_FE_HOIST_HEADROOM_OVERHEAD``
    (default 3.0, matching the conservative coexisting-buffer multiple). NEVER hardcode a per-HW
    memory threshold; the cache lets a quiet/large-RAM box record a measured value. Read live each
    call so tests / callers can monkeypatch the constant or env. Cold/missing cache + missing env
    falls through to the safe constant default -- byte-identical to the pre-KTC behaviour."""
    import os

    val = float(_get_live("_FE_HOIST_HEADROOM_OVERHEAD"))
    try:
        from ._kernel_tuning import get_kernel_tuning_cache

        _cache = get_kernel_tuning_cache()
        if _cache is not None:
            tuned = _cache.lookup("fe_hoist_headroom_overhead")
            if tuned:
                _v = float(tuned.get("overhead", 0.0) or 0.0)
                if _v > 0.0:
                    val = _v
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed in _feature_engineering_mem_budget.py:130: %s", e)
        pass
    env = os.environ.get("MLFRAME_FE_HOIST_HEADROOM_OVERHEAD")
    if env is not None:
        try:
            _e = float(env)
            if _e > 0.0:
                val = _e
        except (TypeError, ValueError):
            pass
    return val


# ABSOLUTE FREE-RAM FLOOR (2026-06-13). The ``_FE_BUFFER_RAM_BUDGET_RATIO`` (0.3) cap is RELATIVE: on
# a small-RAM host with lots of free RAM it still sizes a multi-GB buffer (observed ~9 GiB peak for a
# full-n diagnostic), leaving NO guaranteed free headroom for the rest of the process / other procs.
# This adds an ABSOLUTE floor: the allocator must always leave at least ``_fe_min_free_ram_bytes()``
# of host RAM free. We carve the reserve off ``available`` FIRST (``usable = max(0, available -
# reserve)``), then apply the existing ratio/overhead/worker divide to ``usable`` -- so the reserve is
# host-global (subtracted ONCE, before the per-worker divide), independent of n_workers. The reserve is
# overridable via env ``MLFRAME_FE_MIN_FREE_RAM_GB`` (mirrors the existing ``MLFRAME_*`` FE knobs --
# mlframe is shared infra, no project-specific prefix) AND the module constant ``_FE_MIN_FREE_RAM_GB``.
# Set the reserve to 0 (env or constant) to get BYTE-IDENTICAL legacy behaviour (usable == available).
# When the reserve cannot be met (available <= reserve) the budget collapses toward 0; callers floor the
# chunk width to the hard minimum (>= one pair's worth) so the loop ALWAYS makes progress, never deadlocks.
_FE_MIN_FREE_RAM_GB: float = 3.0


def _fe_min_free_ram_bytes() -> int:
    """Absolute host-RAM reserve (bytes) the FE buffer allocator must always leave free.

    Resolution order: env ``MLFRAME_FE_MIN_FREE_RAM_GB`` (if set + parseable) overrides the module
    constant ``_FE_MIN_FREE_RAM_GB`` (default 3.0 GiB). A value <= 0 disables the floor (byte-identical
    legacy behaviour). Read live each call so tests / callers can monkeypatch the constant or env."""
    import os

    gb = _get_live("_FE_MIN_FREE_RAM_GB")
    env = os.environ.get("MLFRAME_FE_MIN_FREE_RAM_GB")
    if env is not None:
        try:
            gb = float(env)
        except (TypeError, ValueError):
            gb = _get_live("_FE_MIN_FREE_RAM_GB")
    if gb <= 0.0:
        return 0
    return int(gb * 2**30)


# ABSOLUTE BUFFER CEILING (2026-07-09). The RELATIVE budget above (``0.3*(available-reserve)/(3.0*nw)``)
# re-measures ``available`` via the 0.25s-TTL psutil cache on every call, so on a large-RAM host (or one
# where a sibling process transiently frees memory) the computed budget -- and the chunk width the caller
# derives from it -- can re-inflate to tens of GB between calls within the SAME fit, then shrink again once
# RAM pressure returns: a RAM sawtooth, not a bug in any single call's math. An absolute cap bounds a
# SINGLE hoisted FE buffer regardless of how much RAM is momentarily free. 8 GiB is chosen as a defensible
# ceiling: it already exceeds every measured peak buffer in this file's own history (the 4.2 GiB tracemalloc
# snapshot cited above, the 17.6 GiB pre-fix crash this dispatcher exists to prevent), so it never binds on
# a host that needed the relative sizing anyway, while still capping the pathological large-RAM case. The
# recompute fallback (bit-identical survivors, see the block comment above) absorbs any decline this causes.
_FE_BUFFER_ABSOLUTE_MAX_GB: float = 8.0


def _fe_buffer_absolute_max_bytes() -> int:
    """Absolute ceiling (bytes) on a single hoisted FE buffer, independent of how much RAM is free.

    Resolution order: env ``MLFRAME_FE_BUFFER_MAX_GB`` (if set + parseable and > 0) overrides the module
    constant ``_FE_BUFFER_ABSOLUTE_MAX_GB`` (default 8 GiB), mirroring ``_fe_min_free_ram_bytes`` /
    ``_fe_hoist_headroom_overhead``'s env+constant resolution pattern. Read live each call so tests /
    callers can monkeypatch the constant or env."""
    import os

    gb = _get_live("_FE_BUFFER_ABSOLUTE_MAX_GB")
    env = os.environ.get("MLFRAME_FE_BUFFER_MAX_GB")
    if env is not None:
        try:
            _e = float(env)
            if _e > 0.0:
                gb = _e
        except (TypeError, ValueError):
            pass
    return int(gb * 2**30)


def _fe_effective_buffer_budget_bytes(available_bytes: int, n_workers: int = 1) -> int:
    """Per-call byte budget for the FE candidate buffer that keeps the SUM of the
    coexisting buffers (chunk float32 + disc codes + batch-MI working set + the
    held-alive single-pair buffer) AND the concurrent-worker multiplication inside
    ``_FE_BUFFER_RAM_BUDGET_RATIO * available`` (see the block comment above), AND keeps
    an ABSOLUTE ``_fe_min_free_ram_bytes()`` of host RAM free (the 2026-06-13 floor: the
    relative 0.4 ratio alone left no guaranteed headroom on small-RAM hosts), AND is capped at
    ``_fe_buffer_absolute_max_bytes()`` regardless of how much RAM is free (the 2026-07-09 anti-sawtooth
    ceiling -- see the block comment above).

    ``available_bytes < 0`` (no psutil) preserves the legacy permissive behaviour by
    returning ``-1`` (callers treat that as "no cap"). ``n_workers`` is the number of
    threads that may run ``check_prospective_fe_pairs`` CONCURRENTLY (1 on the
    serial-main-thread path; ``n_jobs`` on the joblib ``backend="threading"`` path).

    The reserve is HOST-GLOBAL: subtracted from ``available`` ONCE before the per-worker
    divide (``usable = max(0, available - reserve)``), so concurrency cannot multiply the
    reserve away. With reserve == 0 and the absolute ceiling disabled this is byte-identical to the
    legacy formula."""
    if available_bytes < 0:
        return -1
    nw = max(1, int(n_workers))
    usable = available_bytes - _fe_min_free_ram_bytes()
    if usable < 0:
        usable = 0
    raw = float(usable) * _get_live("_FE_BUFFER_RAM_BUDGET_RATIO")
    budget = int(raw / (_get_live("_FE_PEAK_OVERHEAD_FACTOR") * nw))
    return min(budget, _fe_buffer_absolute_max_bytes())


# UNIFIED FE/screen subsample (2026-06-25). ONE knob for the whole FE block -- the relevance SCREEN, the FE
# PAIR-SEARCH, and the "fast" preset all read this single value. Two facts justify 30k with NO accuracy loss:
#   * The FE-screen accuracy bench (bench_fe_pair_subsample_accuracy.py) measured survivor-jaccard=1.0 and
#     winner-match 5/5 vs the FULL-n screen for every n_eff >= 25_000; 30_000 sits above that validated
#     floor with headroom for the gate-detection MI band (the screen's >25k requirement). 10_000 is the
#     marginal floor (jaccard slips) -- DO NOT go below 25_000.
#   * Since 2026-06-20 the default ``MRMR()`` fit shrinks ``fe_check_pairs_subsample_n`` (and
#     ``fe_smart_polynom_subsample_n``) to this screen size at large n via ``_apply_default_screen_subsample``.
#     Verified BIT-IDENTICAL selection (chosen features, recipe names, get_feature_names_out order) at the
#     unified 30k vs the legacy mixed 200k/30k/25k across the F2 goal (uniform/scaled_1_5/heavy_tailed/mixed)
#     + the canonical biz_value suite + engineered-replay. The 2026-05-18 "200k kept a marginal hermite
#     feature vs 100k" note is superseded: that path (fe_smart_polynom_subsample_n) is likewise screen-shrunk
#     to 30k by the default profile and the canonical suite (which exercises it) passes at 30k.
# This is the SINGLE source of truth (no back-compat aliases -- every FE site reads THIS name); the per-host
# KTC tune (cache key ``mrmr_default_screen_n``, resolved by ``MRMR._default_screen_subsample_n``) is the ONE
# place a host re-tunes the value -- never re-introduce a second subsample constant.
UNIFIED_FE_SUBSAMPLE_N: int = 30_000


def _estimate_fe_shared_buffer_bytes(n_rows: int, max_n_combs: int, n_binary: int) -> int:
    """Bytes the hoisted shared buffer would consume at full precision."""
    return int(n_rows) * int(max_n_combs) * int(n_binary) * 4  # float32 = 4 bytes


# OPT5 (2026-06-07): short-TTL cache for ``psutil.virtual_memory().available``.
# ``_can_hoist_shared_buffer`` is called once per FE chunk AND once per ext-val survivor
# (scene-1500x299 cProfile: 2926 calls -> 11.8s / ~4% of fit in ``psutil_windows.virtual_mem``
# alone; the Windows psutil syscall is ~4ms). The available-RAM reading is stable across the
# rapid successive calls within one FE sweep, so caching it for a brief window collapses those
# 2926 syscalls to a handful WITHOUT changing the hoist/recompute decision in practice -- and
# even a borderline flip is selection-safe because the recompute-fallback path produces
# BIT-IDENTICAL survivors (the documented accuracy-preserving fallback), so MRMR's selected
# features + recipe are unaffected by which path the buffer-fit check picks. TTL kept short
# (0.25s) so a genuine RAM drop between distinct allocation phases is still caught by a fresh
# reading. ``_FE_VMEM_CACHE`` is a 2-tuple ``(monotonic_timestamp, available_bytes)``.
import time as _time

_FE_VMEM_TTL_S: float = 0.25
_FE_VMEM_CACHE: tuple[float, int] | None = None
_FE_VMEM_LOCK = threading.Lock()


def _available_ram_bytes_cached() -> int:
    """``psutil.virtual_memory().available`` with a ~0.25s TTL cache (OPT5).

    Returns ``-1`` when psutil is unavailable (callers treat that as "permissive yes",
    preserving the legacy no-psutil behaviour). Thread-safe: the FE sweep can run under
    joblib ``backend="threading"``, so the read+store is guarded by ``_FE_VMEM_LOCK``.
    """
    now = _time.monotonic()
    cached = _get_live("_FE_VMEM_CACHE")
    if cached is not None and (now - cached[0]) < _FE_VMEM_TTL_S:
        return int(cached[1])
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except Exception:
        return -1
    with _FE_VMEM_LOCK:
        _set_live("_FE_VMEM_CACHE", (now, available))
    return available


def _can_hoist_shared_buffer(
    buffer_bytes: int,
    budget_ratio: float = _FE_BUFFER_RAM_BUDGET_RATIO,
    n_workers: int = 1,
    overhead_aware: bool = True,
) -> tuple[bool, int, int]:
    """Decide whether the shared scratch buffer fits in available RAM.

    Uses ``psutil.virtual_memory().available`` (via the OPT5 short-TTL cache
    ``_available_ram_bytes_cached``) for a conservative "RAM I can take right now" reading --
    ``total`` would include pages owned by other processes which we cannot evict cleanly.
    Falls back to a permissive yes when psutil is unavailable so single-test environments
    without it still take the historical fast path (and OOM loudly on truly large n if so).

    LARGE-N FIX (2026-06-08): with ``overhead_aware=True`` (default) the fit check uses the
    ``_fe_effective_buffer_budget_bytes`` envelope -- the raw ``budget_ratio*available``
    divided by ``_FE_PEAK_OVERHEAD_FACTOR`` (the coexisting disc/MI/single-pair buffers) and
    by ``n_workers`` (concurrent joblib-threading callers, each allocating its own buffers in
    the shared address space). This is what stops the n=100000 chunk buffer from sizing itself
    to fill 0.4*available while 1.85 GiB of sibling buffers push the real footprint past it
    (the measured 4.2 GiB peak on the 5-col F1 repro). Pass ``overhead_aware=False`` for the
    raw single-buffer check (legacy semantics) when no coexisting buffers are in play.

    Returns (can_hoist, buffer_bytes, available_bytes).
    """
    available = _available_ram_bytes_cached()
    if available < 0:
        # No psutil: preserve legacy behaviour (always hoist, OOM is the signal).
        logger.debug("_can_hoist_shared_buffer: no psutil -- legacy permissive hoist, buffer_bytes=%d", buffer_bytes)
        return True, buffer_bytes, -1
    _ceiling = _fe_buffer_absolute_max_bytes()
    if buffer_bytes > _ceiling:
        # ABSOLUTE CEILING (2026-07-09): refuse regardless of how much RAM is free -- the anti-sawtooth
        # cap (see ``_fe_buffer_absolute_max_bytes`` docstring). Applies to BOTH acceptance paths below.
        logger.debug(
            "_can_hoist_shared_buffer: buffer_bytes=%d exceeds absolute ceiling=%d (available=%d) -- declining, recompute fallback.",
            buffer_bytes, _ceiling, available,
        )
        return False, buffer_bytes, available
    if overhead_aware:
        budget = _fe_effective_buffer_budget_bytes(available, n_workers=n_workers)
    else:
        budget = int(available * budget_ratio)
    if buffer_bytes < budget:
        logger.debug(
            "_can_hoist_shared_buffer: buffer_bytes=%d < budget=%d (available=%d) -- hoisting.",
            buffer_bytes, budget, available,
        )
        return True, buffer_bytes, available
    # ABSOLUTE-HEADROOM ACCEPTANCE (2026-06-24): the conservative relative budget over-declines a
    # buffer that is small relative to available RAM (the F2 100k 222 MiB-at-5.7 GiB case). Accept
    # the buffer anyway when its REALISTIC peak footprint (buffer * overhead * n_workers, covering
    # the coexisting disc/MI/single-pair buffers AND concurrent threading workers) still leaves at
    # least the host-global ``_fe_min_free_ram_bytes()`` reserve free. This is strictly safer than
    # the relative budget at the OOM boundary: at large n the footprint approaches ``available`` and
    # free-after drops below the reserve, so this path declines too -- the large-n protection holds.
    # Only the relative (over-conservative) gate is relaxed; selection is unchanged (the batch path
    # the buffer routes to is byte-identical to the recompute fallback). overhead_aware==False keeps
    # the raw single-buffer semantics (no coexisting siblings) and skips this footprint path.
    # A zeroed budget is the callers'/tests' explicit "disable hoisting entirely" signal (the
    # fallback-forcing knob, e.g. monkeypatching ``_FE_BUFFER_RAM_BUDGET_RATIO`` to 0.0 -> budget==0
    # via ``_fe_effective_buffer_budget_bytes``, OR passing ``budget_ratio<=0`` on the raw path). The
    # headroom path honours either so the disable knob always declines. ``budget > 0`` covers both:
    # the overhead-aware budget is 0 iff the live ratio (or usable RAM) is 0; the raw budget is 0 iff
    # ``budget_ratio<=0``.
    if overhead_aware and budget > 0 and budget_ratio > 0.0:
        reserve = _fe_min_free_ram_bytes()
        nw = max(1, int(n_workers))
        peak_footprint = int(buffer_bytes) * _fe_hoist_headroom_overhead() * nw
        if available - peak_footprint >= reserve:
            logger.debug(
                "_can_hoist_shared_buffer: buffer_bytes=%d over budget=%d but absolute-headroom accepted "
                "(available=%d, peak_footprint=%d, reserve=%d) -- hoisting.",
                buffer_bytes, budget, available, peak_footprint, reserve,
            )
            return True, buffer_bytes, available
    logger.debug(
        "_can_hoist_shared_buffer: buffer_bytes=%d over budget=%d, no headroom -- declining, recompute fallback (available=%d).",
        buffer_bytes, budget, available,
    )
    return False, buffer_bytes, available

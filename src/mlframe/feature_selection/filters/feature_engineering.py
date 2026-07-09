"""Pair-discovery feature engineering used by ``MRMR.fit``.

Public functions:

* ``check_prospective_fe_pairs`` -- given high-MI raw pairs, enumerate unary x binary transformation candidates, evaluate each via ``mi_direct``, recommend top-N replacements.
* ``compute_pairs_mis`` -- precompute MI for every pair candidate so the screening loop has a fast lookup.
* ``create_unary_transformations`` / ``create_binary_transformations`` -- preset registries of numpy / scipy.special / hermval functions.
* ``get_existing_feature_name`` / ``get_new_feature_name`` -- string formatters for engineered-feature names.
"""
from __future__ import annotations

import logging
import threading
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermval
from scipy import special as sp

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

    val = float(_FE_HOIST_HEADROOM_OVERHEAD)
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
        logger.debug("suppressed in feature_engineering.py:130: %s", e)
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

    gb = _FE_MIN_FREE_RAM_GB
    env = os.environ.get("MLFRAME_FE_MIN_FREE_RAM_GB")
    if env is not None:
        try:
            gb = float(env)
        except (TypeError, ValueError):
            gb = _FE_MIN_FREE_RAM_GB
    if gb <= 0.0:
        return 0
    return int(gb * 2**30)


def _fe_effective_buffer_budget_bytes(available_bytes: int, n_workers: int = 1) -> int:
    """Per-call byte budget for the FE candidate buffer that keeps the SUM of the
    coexisting buffers (chunk float32 + disc codes + batch-MI working set + the
    held-alive single-pair buffer) AND the concurrent-worker multiplication inside
    ``_FE_BUFFER_RAM_BUDGET_RATIO * available`` (see the block comment above), AND keeps
    an ABSOLUTE ``_fe_min_free_ram_bytes()`` of host RAM free (the 2026-06-13 floor: the
    relative 0.4 ratio alone left no guaranteed headroom on small-RAM hosts).

    ``available_bytes < 0`` (no psutil) preserves the legacy permissive behaviour by
    returning ``-1`` (callers treat that as "no cap"). ``n_workers`` is the number of
    threads that may run ``check_prospective_fe_pairs`` CONCURRENTLY (1 on the
    serial-main-thread path; ``n_jobs`` on the joblib ``backend="threading"`` path).

    The reserve is HOST-GLOBAL: subtracted from ``available`` ONCE before the per-worker
    divide (``usable = max(0, available - reserve)``), so concurrency cannot multiply the
    reserve away. With reserve == 0 this is byte-identical to the legacy formula."""
    if available_bytes < 0:
        return -1
    nw = max(1, int(n_workers))
    usable = available_bytes - _fe_min_free_ram_bytes()
    if usable < 0:
        usable = 0
    raw = float(usable) * _FE_BUFFER_RAM_BUDGET_RATIO
    return int(raw / (_FE_PEAK_OVERHEAD_FACTOR * nw))


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


def _rebuild_full_survivor_col(
    config: tuple,
    X_full,
    original_cols: dict,
    unary_transformations: dict,
    binary_transformations: dict,
    prewarp_spec_by_var: dict | None = None,
    gate_med_median_by_var: dict | None = None,
    cols: list | None = None,
    engineered_operand_values: dict | None = None,
) -> np.ndarray:
    """Rebuild a survivor column at full n from raw X by re-applying its unary + binary transforms.

    Used by every survivor-packing path (fast / fallback / subsample) so a single
    code path produces the full-n output the caller (mrmr.py) expects regardless
    of how the MI sweep was scratched. Cost: 2 unary + 1 binary ufunc per
    survivor; with len(this_pair_features) <= fe_max_pair_features this is
    bounded to ~K calls per pair, trivial vs the MI sweep work.

    ENGINEERED-OPERAND SUPPORT (subsample path): at FE step k>1 an operand index can
    point at a column appended by a prior step -- NOT a raw position in
    ``original_cols``. Such operands are resolved by name from the full-n continuous
    store ``engineered_operand_values[cols[var_idx]]`` (preferred, lossless) or, as a
    fallback, by name from ``X_full``. Mirrors ``_extval_raw_col`` so the subsample
    survivor-rebuild matches the MI-sweep operand semantics; pre-fix this raised
    ``KeyError`` on engineered operands when ``fe_check_pairs_subsample_n`` was active.
    """
    transformations_pair, bin_func_name, _i = config
    var_a_idx, unary_a_name = transformations_pair[0]
    var_b_idx, unary_b_name = transformations_pair[1]
    _eng_vals = engineered_operand_values or {}

    def _operand_full_vals(var_idx):
        """Resolve the full-``n`` continuous values for operand ``var_idx``: raw column position when known, else the engineered-operand store/frame by name; raises the original ``KeyError`` rather than fabricating data."""
        if var_idx in original_cols:
            if isinstance(X_full, pd.DataFrame):
                return X_full.iloc[:, original_cols[var_idx]].values
            return X_full[:, original_cols[var_idx]].to_numpy()
        # Engineered operand (not a raw position): resolve by name, preferring the
        # full-n continuous store over the (discretised, lossy) augmented-frame column.
        _name = cols[var_idx] if (cols is not None and 0 <= var_idx < len(cols)) else None
        if _name is not None:
            _cv = _eng_vals.get(_name)
            if _cv is not None:
                return np.asarray(_cv)
            if isinstance(X_full, pd.DataFrame) and _name in X_full.columns:
                return X_full[_name].to_numpy()
            if hasattr(X_full, "columns") and _name in getattr(X_full, "columns", []):
                return X_full[_name].to_numpy()
        # No raw position and no resolvable engineered column: surface the original
        # KeyError so the failure is explicit rather than silently fabricating data.
        return X_full.iloc[:, original_cols[var_idx]].values if isinstance(X_full, pd.DataFrame) else X_full[:, original_cols[var_idx]].to_numpy()

    vals_a = _operand_full_vals(var_a_idx)
    vals_b = _operand_full_vals(var_b_idx)
    # ``poly_*`` unary keys hold hermval coefficient arrays, not callables;
    # check_prospective_fe_pairs handles them via the same hermval(c=tr_func) path.
    # ``prewarp`` is the per-operand learned pseudo-unary (2026-06-02): its fitted
    # spec lives in ``prewarp_spec_by_var`` and replays closed-form from x alone.
    _pw = prewarp_spec_by_var or {}
    # ``gate_med`` is the per-operand median-gate pseudo-unary (2026-06-04): its
    # only fitted state is one TRAIN-median float in ``gate_med_median_by_var``;
    # replays closed-form as ``(x > stored_median).astype(float)`` from x alone.
    _gm = gate_med_median_by_var or {}

    def _apply_unary(name, var_idx, vals):
        """Apply a named unary transform to ``vals``, dispatching the pseudo-unaries (prewarp/gate_med, whose fitted state lives in the enclosing per-var stores) and the ``poly_*`` hermval-coefficient keys separately from plain callables."""
        if name == "prewarp":
            from .hermite_fe import apply_operand_prewarp
            return apply_operand_prewarp(vals, _pw[var_idx])
        if name == "gate_med":
            from ._feature_engineering_pairs import _gate_med_apply
            return _gate_med_apply(vals, _gm[var_idx])
        if "poly_" in name:
            return hermval(vals, c=unary_transformations[name])
        return unary_transformations[name](vals)

    param_a = _apply_unary(unary_a_name, var_a_idx, vals_a)
    param_b = _apply_unary(unary_b_name, var_b_idx, vals_b)
    col = binary_transformations[bin_func_name](param_a, param_b)
    np.nan_to_num(col, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(col.astype(np.float32, copy=False))


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
    global _FE_VMEM_CACHE
    now = _time.monotonic()
    cached = _FE_VMEM_CACHE
    if cached is not None and (now - cached[0]) < _FE_VMEM_TTL_S:
        return cached[1]
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except Exception:
        return -1
    with _FE_VMEM_LOCK:
        _FE_VMEM_CACHE = (now, available)
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
        return True, buffer_bytes, -1
    if overhead_aware:
        budget = _fe_effective_buffer_budget_bytes(available, n_workers=n_workers)
    else:
        budget = int(available * budget_ratio)
    if buffer_bytes < budget:
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
            return True, buffer_bytes, available
    return False, buffer_bytes, available


# Domain-validity tags for unary transforms produced by ``create_unary_transformations(preset="maximal")``. Consumers that need to clip / reject inputs before
# applying these transforms can look them up here. Tag vocabulary: ``-1to1``, ``-pi/2topi/2``, ``1toinf``, ``-0.(9)to0.(9)``, ``pos``, ``nonzero``. Transforms not
# listed are unconstrained on real inputs (e.g. ``sin``, ``exp``, ``sqr``).
UNARY_INPUT_CONSTRAINTS: dict[str, str] = {
    # reverse trigonometric
    "arccos": "-1to1",
    "arcsin": "-1to1",
    "arctan": "-pi/2topi/2",
    # reverse hyperbolic
    "arccosh": "1toinf",
    "arctanh": "-0.(9)to0.(9)",
    # powers
    "sqrt": "pos",
    "log": "pos",
    "reciproc": "nonzero",
    "invsquared": "nonzero",
    "invqubed": "nonzero",
    "invcbrt": "nonzero",
    "invsqrt": "nonzero",
}

from ._internals import njit_functions_dict, smart_log
from .discretization import discretize_array, discretize_2d_quantile_batch  # noqa: F401 -- re-exported (see _feature_engineering_pairs/_pairs_core.py)
from .permutation import mi_direct

logger = logging.getLogger(__name__)


def compute_pairs_mis(
    all_pairs: Sequence,
    data,
    target_indices,
    nbins,
    classes_y,
    classes_y_safe,
    freqs_y,
    fe_min_nonzero_confidence,
    fe_npermutations,
    cached_confident_MIs: dict,
    cached_MIs: dict,
    fe_min_pair_mi: float,
    fe_min_pair_mi_prevalence: float,
):
    """Compute (or reuse from cache) confidence-permutation-tested MI for every candidate pair, dropping pairs below the prevalence/relevance floor."""
    # Live progress: on the single-thread path ``all_pairs`` is the "getting pairs MIs"
    # tqdm bar itself, so we surface the running top single feature by its (already-
    # computed) marginal MI with y, plus the strongest pair MI, in the bar postfix.
    # ``set_postfix`` exists only on the tqdm object, not on the plain chunk lists handed
    # to the parallel workers -- guard via ``hasattr``. Pure display: the MIs shown are
    # exactly the ones we cache, no extra compute.
    _bar = all_pairs if hasattr(all_pairs, "set_postfix") else None
    _top_var, _top_var_mi = None, -1.0
    _top_pair, _top_pair_mi = None, -1.0
    for raw_vars_pair in all_pairs:
        # check that every element of a pair has computed its MI with target
        for var in raw_vars_pair:
            if (var,) not in cached_confident_MIs and (var,) not in cached_MIs:
                mi, _conf = mi_direct(
                    data,
                    x=(var,),
                    y=target_indices,
                    factors_nbins=nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    min_nonzero_confidence=fe_min_nonzero_confidence,
                    npermutations=fe_npermutations,
                )
                cached_MIs[(var,)] = mi
                if _bar is not None:
                    try:
                        if mi is not None and float(mi) > _top_var_mi:
                            _top_var_mi, _top_var = float(mi), var
                    except (TypeError, ValueError):
                        pass

        # ensure that pair as a whole has computed its MI with target
        if raw_vars_pair not in cached_confident_MIs and raw_vars_pair not in cached_MIs:
            ind_elems_mi_sum = cached_MIs[(raw_vars_pair[0],)] + cached_MIs[(raw_vars_pair[1],)]
            if ind_elems_mi_sum > fe_min_pair_mi:
                mi, _conf = mi_direct(
                    data,
                    x=raw_vars_pair,
                    y=target_indices,
                    factors_nbins=nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    min_nonzero_confidence=fe_min_nonzero_confidence,
                    npermutations=fe_npermutations,
                )

                # T3#24 2026-05-18 Pack #5 pair-MI cache: ALWAYS cache the computed pair MI, not only when it passes ``fe_min_pair_mi_prevalence``.
                # Pre-fix this branch dropped sub-threshold MIs on the floor; adaptive retry with relaxed prevalence (Pack #5) then re-computed them.
                # The downstream consumer in MRMR.fit applies the prevalence gate on the cached value, so caching unconditionally preserves filtering behaviour.
                cached_MIs[raw_vars_pair] = mi
                if _bar is not None:
                    try:
                        if mi is not None and float(mi) > _top_pair_mi:
                            _top_pair_mi, _top_pair = float(mi), raw_vars_pair
                    except (TypeError, ValueError):
                        pass

        if _bar is not None and (_top_var is not None or _top_pair is not None):
            _pf = {}
            if _top_var is not None:
                _pf["top_feat"] = f"col{_top_var}={_top_var_mi:.4f}"
            if _top_pair is not None:
                _pf["top_pair"] = f"({_top_pair[0]},{_top_pair[1]})={_top_pair_mi:.4f}"
            try:
                _bar.set_postfix(_pf, refresh=False)
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in feature_engineering.py:520: %s", e)
                pass

    return cached_MIs


def get_existing_feature_name(fe_tuple: tuple, cols_names: Sequence) -> str:
    """Display name for a unary-transformed feature: the bare column name for ``identity``, else ``transform(name)``."""
    fname = cols_names[fe_tuple[0]]
    if fe_tuple[1] == "identity":
        return str(fname)
    else:
        return f"{fe_tuple[1]}({fname})"


def get_new_feature_name(fe_tuple: tuple, cols_names: Sequence) -> str:
    """Display name for a binary-engineered feature: ``binary_op(unary(a), unary(b))``."""
    return f"{fe_tuple[1]}({get_existing_feature_name(fe_tuple=fe_tuple[0][0],cols_names=cols_names)},{get_existing_feature_name(fe_tuple=fe_tuple[0][1],cols_names=cols_names)})"  # (((2, 'log'), (3, 'sin')), 'mul', 1016)


# GPU broadcast: per-function gpu_compatible flag. Many transformations have direct CuPy equivalents (``np.log`` -> ``cp.log`` etc.); some require ``scipy.special``
# (erfinv, gammaln, ...) which CuPy lacks. The FE driver groups columns into a CuPy tensor and applies the GPU-compatible transformation in one launch, falling back
# to per-column CPU dispatch for the rest.


def gpu_compatible_unary_names() -> set:
    """Names of unary transformations with a direct CuPy equivalent. Anything outside this set falls back to per-column CPU dispatch."""
    return {
        "identity", "sign", "neg", "abs", "rint",
        "sqr", "qubed", "reciproc", "invsquared", "invqubed",
        "cbrt", "sqrt", "invcbrt", "invsqrt",
        "log", "exp",
        "sin", "cos", "tan",
        "sinh", "cosh", "tanh",
    }


def gpu_compatible_binary_names() -> set:
    """Same idea for binary transformations."""
    return {
        "add", "sub", "mul", "div", "pow", "min", "max",
        "hypot", "atan2",
    }


def apply_gpu_unary_batched(
    cols_data,
    column_indices: Sequence[int],
    transformation_name: str,
):
    """Apply a unary transformation to a batch of columns on GPU. Returns a CuPy 2-D array with shape ``(n_samples, len(column_indices))``. Only safe for names in
    ``gpu_compatible_unary_names()``.

    Raises:
        ValueError: if the transformation is not GPU-compatible.
        ImportError: if CuPy is not installed.
    """
    import cupy as cp

    name = transformation_name
    if name not in gpu_compatible_unary_names():
        raise ValueError(f"transformation {name!r} is not GPU-compatible")

    if hasattr(cols_data, "to_numpy"):
        cols_np = cols_data.to_numpy() if hasattr(cols_data, "to_numpy") else np.asarray(cols_data)
    else:
        cols_np = np.asarray(cols_data)

    # Slice the requested columns.
    sub = cols_np[:, list(column_indices)] if cols_np.ndim == 2 else cols_np.reshape(-1, 1)
    sub_gpu = cp.asarray(sub.astype(np.float32))

    if name == "identity":
        return sub_gpu
    if name == "sign":
        return cp.sign(sub_gpu)
    if name == "neg":
        return -sub_gpu
    if name == "abs":
        return cp.abs(sub_gpu)
    if name == "rint":
        return cp.rint(sub_gpu)
    if name == "sqr":
        return cp.power(sub_gpu, 2)
    if name == "qubed":
        return cp.power(sub_gpu, 3)
    if name == "reciproc":
        return cp.power(sub_gpu, -1)
    if name == "invsquared":
        return cp.power(sub_gpu, -2)
    if name == "invqubed":
        return cp.power(sub_gpu, -3)
    if name == "cbrt":
        return cp.cbrt(sub_gpu)
    if name == "sqrt":
        return cp.sqrt(cp.abs(sub_gpu))
    if name == "invcbrt":
        return cp.power(sub_gpu, -1.0 / 3.0)
    if name == "invsqrt":
        return cp.power(sub_gpu, -0.5)
    if name == "log":
        # Smart log: avoid log of non-positive.
        x_min = cp.nanmin(sub_gpu)
        if x_min > 0:
            return cp.log(sub_gpu)
        return cp.log(sub_gpu + (1e-5 - x_min))
    if name == "exp":
        return cp.exp(sub_gpu)
    if name == "sin":
        return cp.sin(sub_gpu)
    if name == "cos":
        return cp.cos(sub_gpu)
    if name == "tan":
        return cp.tan(sub_gpu)
    if name == "sinh":
        return cp.sinh(sub_gpu)
    if name == "cosh":
        return cp.cosh(sub_gpu)
    if name == "tanh":
        return cp.tanh(sub_gpu)
    raise ValueError(f"GPU dispatch missing for {name!r} despite being in compatible set")


def apply_gpu_binary_batched(
    a_gpu,
    b_gpu,
    transformation_name: str,
):
    """Apply a binary transformation element-wise to two CuPy arrays already on the GPU. Names must be in ``gpu_compatible_binary_names()``."""
    import cupy as cp

    name = transformation_name
    if name not in gpu_compatible_binary_names():
        raise ValueError(f"transformation {name!r} is not GPU-compatible")

    if name == "add":
        return a_gpu + b_gpu
    if name == "sub":
        return a_gpu - b_gpu
    if name == "mul":
        return a_gpu * b_gpu
    if name == "div":
        return a_gpu / b_gpu
    if name == "pow":
        return cp.power(a_gpu, b_gpu)
    if name == "min":
        return cp.minimum(a_gpu, b_gpu)
    if name == "max":
        return cp.maximum(a_gpu, b_gpu)
    if name == "hypot":
        return cp.hypot(a_gpu, b_gpu)
    if name == "atan2":
        return cp.arctan2(a_gpu, b_gpu)
    raise ValueError(f"GPU dispatch missing for binary {name!r}")


_KNOWN_UNARY_PRESETS = ("minimal", "medium", "maximal")
_KNOWN_BINARY_PRESETS = ("minimal", "medium", "maximal")


def _resolve_preset(preset: str | None) -> str:
    """Canonicalise a preset name to one of {minimal, medium, maximal}.

    ``rich`` / ``full`` are treated as aliases for ``maximal`` (the richest
    tier) so callers using the historical loose vocabulary still get a
    well-defined registry. ``None`` (and the empty string) mean "use the
    default tier" and resolve to ``medium`` -- callers pass ``None`` to opt
    into the standard preset without naming it. Any other value raises
    ValueError so a typo (``minimal``) surfaces loudly rather than silently
    aliasing to ``medium``.
    """
    if preset is None:
        return "medium"
    p = preset.strip().lower()
    if not p:
        return "medium"
    if p in _KNOWN_UNARY_PRESETS:
        return p
    if p in ("rich", "full"):
        return "maximal"
    if p in ("basic", "default"):
        return "minimal"
    raise ValueError(f"unknown FE preset {preset!r}; expected one of " f"{_KNOWN_UNARY_PRESETS} (or aliases 'rich'/'full' -> 'maximal')")


def create_unary_transformations(preset: str = "minimal"):
    """Build the ``{name: callable}`` unary-transform registry for the given preset (``minimal``/``medium``/``maximal``, monotone supersets)."""
    # Domain-validity tags for each transform live in the module-level ``UNARY_INPUT_CONSTRAINTS`` dict so callers that need to clip / reject inputs can look them up
    # by transform name (e.g. ``arccos`` requires ``-1to1``).
    #
    # Preset ladder (monotone: minimal subset of medium subset of maximal):
    #   minimal -- non-degenerate workhorse set able to express the common
    #     algebraic targets (a**2/b, log(c)*sin(d), ...): identity + the
    #     elementary powers, sqrt, log, sin and their sign/neg/abs companions.
    #     Every tier MUST have >1 member; an identity-only "minimal" silently
    #     crippled MRMR's pair FE (it could only form mul(a,b)/add(a,b), never
    #     sqr(a) or log(c)), so the default fit found ZERO engineered cols.
    #   medium -- minimal + exp, reciprocal/inverse powers, cbrt, rint.
    #   maximal -- medium + trig/hyperbolic/special families (below).
    preset = _resolve_preset(preset)
    unary_transformations = {
        # simplest
        "identity": lambda x: x,
        # sign / magnitude companions
        "neg": np.negative,
        "abs": np.abs,
        # powers
        "sqr": lambda x: np.power(x, 2),
        "reciproc": lambda x: np.power(x, -1),
        "sqrt": lambda x: np.sqrt(np.abs(x)),
        # logarithms
        "log": smart_log,
        # trigonometric
        "sin": np.sin,
    }
    if preset != "minimal":
        unary_transformations.update(
            {
                "sign": np.sign,
                # outliers removal
                # Rounding
                "rint": np.rint,
                # np.modf Return the fractional and integral parts of an array, element-wise.
                # clip
                # powers
                "qubed": lambda x: np.power(x, 3),
                "invsquared": lambda x: np.power(x, -2),
                "invqubed": lambda x: np.power(x, -3),
                "cbrt": np.cbrt,
                "invcbrt": lambda x: np.power(x, -1 / 3),
                "invsqrt": lambda x: np.power(x, -1 / 2),
                # logarithms
                "exp": np.exp,
            }
        )

    if preset == "maximal":
        unary_transformations.update(
            {
                # math an
                # GPU-DISABLED(restore): cross-row np.gradient has no row-blocked GPU form (2026-06-21,
                # full-GPU residency build). Uncomment with the matching _gpu_resident_fe._FULL_UNARY entries.
                # "grad1": np.gradient,
                # "grad2": lambda x: np.gradient(x, edge_order=2),
                # trigonometric
                "sinc": np.sinc,
                "cos": np.cos,
                "tan": np.tan,
                # reverse trigonometric
                "arcsin": np.arcsin,
                "arccos": np.arccos,
                "arctan": np.arctan,
                # hyperbolic
                "sinh": np.sinh,
                "cosh": np.cosh,
                "tanh": np.tanh,
                # reverse hyperbolic
                "arcsinh": np.arcsinh,
                "arccosh": np.arccosh,
                "arctanh": np.arctanh,
                # special
                #'psi':sp.psi, polygamma(0,x) is same as psi
                "erf": sp.erf,
                # GPU-DISABLED(restore): no cupyx.scipy.special.dawsn (2026-06-21, full-GPU residency build).
                # "dawsn": sp.dawsn,
                "gammaln": sp.gammaln,
                #'spherical_jn':sp.spherical_jn
            }
        )

    njit_functions_dict(unary_transformations)

    if preset == "maximal":
        for order in range(3):
            unary_transformations["polygamma_" + str(order)] = lambda x, _order=order: sp.polygamma(_order, x)
            unary_transformations["struve" + str(order)] = lambda x, _order=order: sp.struve(_order, x)
            unary_transformations["jv" + str(order)] = lambda x, _order=order: sp.jv(_order, x)

        """j0
        faster version of this function for order 0.

        j1
        faster version of this function for order 1.
        """

    return unary_transformations


def _safe_div(x, y):
    """Element-wise division that is EXACT for every nonzero denominator and
    finite (never +-inf) when the denominator is exactly zero. Required so the
    binary registry can express genuine ratio targets (e.g. a**2/b) directly
    rather than only via reciproc-then-multiply, which loses a representative on
    tight unary presets.

    HEAVY-TAIL FIX (2026-06-13): the prior form ``x / (y + sign(y)*eps + eps)``
    perturbed EVERY positive denominator by ``2*eps`` (and, asymmetrically, left
    negative ones exact). On a heavy-tailed ratio target the perturbation is
    negligible for ordinary ``y`` but reaches ~``2*eps/y`` relative error as
    ``y -> 0`` -- e.g. ``b=1e-6`` gives a 0.2% error on ``a**2/b``, and on a
    target whose magnitude is dominated by that single small-``b`` point a linear
    downstream's MAE is inflated by ~0.05 (measured on ``y=0.2*a**2/b``: replayed
    feature -> 0.10 test MAE vs 0.05 with the exact ratio). Dividing exactly
    wherever ``y != 0`` removes the perturbation entirely; the ``eps`` floor only
    ever substitutes for an exact-zero denominator (which the downstream
    ``nan_to_num`` scrub would otherwise map through +-inf)."""
    # NOTE: this is njit-compiled via the binary-op registry, so it must stay numba-nopython
    # compatible. A ``np.array(y, dtype=...)`` + ``np.putmask`` rewrite (bench-attempt-rejected
    # 2026-06-17) was ~22% faster in PURE PYTHON but (a) numba rejects ``np.array`` on a READONLY
    # input array and does not support ``np.putmask``, breaking compilation -> ``div`` silently
    # unavailable in the FE sweep, and (b) the Python-level allocation saving is irrelevant once
    # njit-compiled. ``np.asarray`` + ``np.where`` compile cleanly; keep them.
    eps = 1e-9
    y = np.asarray(y, dtype=np.float64)
    safe_y = np.where(y == 0.0, eps, y)
    return np.asarray(x, dtype=np.float64) / safe_y


def create_binary_transformations(preset: str = "minimal"):
    """Build the ``{name: callable}`` binary-transform registry for the given preset (``minimal``/``medium``/``maximal``, monotone supersets)."""
    # Preset ladder (monotone: minimal subset of medium subset of maximal):
    #   minimal -- the elementary closed binary algebra: mul, add, sub,
    #     div, max, min. ``sub`` and ``div`` were absent from EVERY
    #     prior tier; division was only reachable as reciproc o mul, so a plain
    #     a/b ratio target could not be formed when the unary preset lacked
    #     reciproc. Every tier MUST have >1 member.
    #   medium -- minimal + abs_diff, hypot (richer magnitude combinations).
    #   maximal -- medium + the full numpy / scipy.special family below.
    preset = _resolve_preset(preset)

    binary_transformations = {
        # Basic
        "mul": np.multiply,
        "add": np.add,
        "sub": np.subtract,
        # Safe division (sign-stable eps; never +-inf on divide-by-zero).
        "div": _safe_div,
        # Extrema
        "max": np.maximum,
        "min": np.minimum,
    }

    if preset != "minimal":
        binary_transformations.update(
            {
                # Richer magnitude combinations available from medium up.
                "abs_diff": lambda x, y: np.abs(x - y),
                "hypot": np.hypot,
                # Interaction shapes a bilinear product cannot linearize (measured: a linear downstream gets logit
                # 0.49 on y=-|a-b| with the product vs 0.88 with abs_diff; 0.79 vs 0.88 on y=sign(a)*|b|; a capacity
                # control confirms the gain is the OPERATOR CLASS, not column count). Both are leak-free pure functions
                # (no fit) and NON-SYMMETRICAL, so the pair search must try both operand orders.
                "signed": lambda x, y: np.sign(x) * np.abs(y),  # signed magnitude: sign(a)*|b|  (non-symmetrical!)
                "ratio_abs": lambda x, y: x / (np.abs(y) + 1.0),  # standardized ratio a/(|b|+1), inf-safe (non-symmetrical!)
            }
        )

    if preset == "maximal":
        binary_transformations.update(
            {
                # All kinds of averages
                "hypot": np.hypot,
                "logaddexp": np.logaddexp,
                # GPU-DISABLED(restore): no cupyx agm (2026-06-21, full-GPU residency build).
                # "agm": sp.agm,  # Compute the arithmetic-geometric mean of a and b.
                # Rational routines
                #'lcm':np.lcm, # requires int arguments #  ufunc 'lcm' did not contain a loop with signature matching types (<class 'numpy.dtype[float32]'>, <class 'numpy.dtype[float32]'>) -> None
                #'gcd':np.gcd, # requires int arguments
                #'mod':np.remainder, # requires int arguments
                # Powers
                "pow": np.power,  # non-symmetrical! may required dtype=complex for arbitrary numbers
                # Logarithms
                # GPU-DISABLED(restore): np.emath.logn (complex base) -- real GPU form is
                # log(y-ymin+0.1)/log(x-xmin+0.1) (2026-06-21, full-GPU residency build).
                # "logn": lambda x, y: np.emath.logn(x - np.min(x) + 0.1, y - np.min(y) + 0.1),  # non-symmetrical!
                # DSP
                # 'convolve':np.convolve, # symmetrical wrt args. scipy.signal.fftconvolve should be faster? SLOW?
                # Linalg
                "heaviside": np.heaviside,  # non-symmetrical!
                #'cross':np.cross, # symmetrical # incompatible dimensions for cross product (dimension must be 2 or 3)
                # Comparison
                "greater": lambda x, y: np.greater(x, y).astype(int),
                "less": lambda x, y: np.less(x, y).astype(int),
                "equal": lambda x, y: np.equal(x, y).astype(int),
                # special
                "beta": sp.beta,  # symmetrical
                # GPU-DISABLED(restore): deferred with the full-GPU residency batch (2026-06-21).
                # "binom": sp.binom,  # non-symmetrical! binomial coefficient considered as a function of two real variables.
            }
        )

    njit_functions_dict(binary_transformations)

    return binary_transformations


# ----------------------------------------------------------------------
# Sibling-module re-export. The 505-LOC ``check_prospective_fe_pairs``
# body lives in ``_feature_engineering_pairs.py`` so this file stays
# below the 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._feature_engineering_pairs import check_prospective_fe_pairs  # noqa: F401

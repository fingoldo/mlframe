"""Resident-GPU twin of the ORDER-2 pooled-max joint-MI permutation-null floor.

``_permutation_null.pooled_pair_permutation_null_joint_mi_floor`` is a PURE-CPU njit loop: for each of the K
target shuffles it rebuilds the WHOLE candidate-pair pool's plug-in joint MI ``I((x_a, x_b); y_perm)`` via
``batch_pair_mi_prange`` and records the per-shuffle MAXIMUM over the pool; the host then takes the
``quantile``-th quantile of the K maxes (the Westfall-Young maxT floor on the selection statistic). The pool
operands - ``factors_data``, ``pair_a/pair_b``, ``nbins``, ``freqs_y`` and (if mm-debias) the per-pair
Miller-Madow bias - are FIT-CONSTANT across shuffles; only the target labelling ``y_perm`` changes per shuffle
(a relabelling of rows that leaves every X-column marginal AND the class-frequency vector invariant).

This module ports the SAME pooled-MAX construction to the device:
  * The pool operands are uploaded ONCE and held resident. In particular the per-pair joint X-code
    ``cls_x = x_a * nbins_b + x_b`` is PERMUTATION-INVARIANT, so it is built once on the device as a
    ``(n_pairs, n)`` int64 matrix and reused for every shuffle - the only per-shuffle device input is the
    target permutation.
  * The K target permutations are BORN on the device (argsort of i.i.d. random keys, ``cupy.random.RandomState``
    seeded from ``random_seed``), the same device-RNG shuffle pattern the order-1 resident floor and the CMI
    perm-null use. So the ``(K, n)`` shuffle matrix never leaves the device and only the small ``(n,)`` class
    codes go up once.
  * Per shuffle the all-pair joint MI is the plug-in ``H(x) + H(y) - H(x, y_perm)`` computed from a batched
    ``cp.bincount`` over the flat joint index ``cls_x * n_classes_y + y_perm`` - the EXACT same plug-in joint-MI
    estimator (and the same joint contingency table) the existing pair-MI GPU backend (``batch_pair_mi_cupy``)
    and the CPU ``batch_pair_mi_prange`` use, so the per-shuffle max is on the same scale as the gated
    ``pair_mi``. The per-pair MM bias (if mm-debias) is subtracted on device (permutation-invariant, uploaded
    once). The per-shuffle MAX over the pool reduces on device; only the ``(K,)`` maxes return to the host.
  * The ``(tile_perm x n_pairs)`` joint-histogram intermediate is TILED over perms (mirroring the order-1
    resident ``_PERMNULL_TILE_CELLS`` budget) so the ``(pb * n_pairs * n)`` working set stays inside a small
    (4GB) card.

The host owns the final ``np.quantile`` so the floor value is computed identically to the CPU path.

CORRECTNESS BAR: SELECTION-EQUIVALENCE, not byte-identity. The floor is the 0.95-quantile of a RANDOM
permutation null. A device RNG seeded from ``random_seed`` makes the draw reproducible across runs, but its
stream is NOT bit-identical to numpy's ``default_rng`` (same contract as the order-1 resident gen and the CMI
perm-null). The plug-in joint MI of the SAME integer contingency table differs from the njit kernel only in FP
reduction order (~1e-15). Over K=25 draws the 0.95-quantile floor agrees with the CPU floor to within the
gate's razor - so the gate decision ``pair_mi >= floor`` is IDENTICAL on F2 and selection is unchanged.

Returns ``None`` on ANY cupy error (OOM / device fault) so the caller falls back to the exact CPU njit floor
(correctness first). NEVER call ``cp.get_default_memory_pool().free_all_blocks()`` here - it nukes the resident
pool (a +47s regression measured previously on the CMI perm-null port).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from mlframe.feature_selection.filters._lazy_host_codes import length_of
from mlframe.feature_selection.filters._lazy_host_codes import shape_of
from mlframe.feature_selection.filters._device_sort_safe import device_argsort, device_sort  # noqa: F401  (thrust aborts the process on a short pool)

# Per-shuffle working-set budget: the largest intermediate is the (pb * n_pairs * n) int64 flat-joint-index
# array. We tile over perms so ``pb * n_pairs * n * 8`` bytes stays well inside a 4GB card. Below ~24M cells
# -> ~192MB per int64 buffer, comfortably under 4GB with headroom for the resident cls_x matrix and uploads.
# Mirrors ``_permutation_null_resident._PERMNULL_TILE_CELLS``.
_PERMNULL_PAIR_TILE_CELLS = 24_000_000  # pb * n_pairs * n upper bound (int64 cells)


#: Device bytes the entropy step of one tile holds per histogram cell: the int64 counts, their float64 cast, the probabilities, the mask and the log / where / product temporaries.
_TILE_BYTES_PER_CELL = 64


def _tile_cell_budget(cp: Any) -> int:
    """Histogram cells one permutation tile may hold: ``_PERMNULL_PAIR_TILE_CELLS``, or less when the device (or the fit's pool cap, ``fe_gpu_pool``) has less than ``_TILE_BYTES_PER_CELL`` bytes per cell free.

    A fixed 24M cells needed about 1.5 GB at once; with 2.1 GB of a 2.6 GB pool already in use the 463 MB temporary of the entropy step failed, the OOM tripped the circuit breaker and every later floor of the process ran on the CPU."""
    try:
        pool = cp.get_default_memory_pool()
        avail = int(cp.cuda.Device().mem_info[0]) + int(pool.free_bytes())
        limit = int(pool.get_limit())
        if limit > 0:
            avail = min(avail, max(0, limit - int(pool.used_bytes())))
    except Exception:  # an unreadable device keeps the fixed budget; the OOM handler below still guards the call
        return _PERMNULL_PAIR_TILE_CELLS
    return max(1, min(_PERMNULL_PAIR_TILE_CELLS, int(avail * 0.5) // _TILE_BYTES_PER_CELL))


def _joint_code_entropies(cp: Any, cls_x: Any, max_joint: int, inv_n: float) -> Any:
    """``-sum p log p`` of every row of the (n_pairs, n) joint X-code matrix ``cls_x`` (codes in ``[0, max_joint)``): one bincount over offset codes for a block of pairs and one row reduction, instead of a
    bincount, a mask and a reduction per pair. Cells beyond a pair's own cardinality are empty and add exactly zero."""
    n_pairs = int(cls_x.shape[0])
    # Intentional float64: entropy accumulator of a permutation floor, exact ties decide the selection
    h_x = cp.empty(n_pairs, dtype=cp.float64)
    block = max(1, min(n_pairs, (1 << 24) // max(1, int(max_joint))))  # a block's (pairs, max_joint) count table stays within 2^24 cells
    for p0 in range(0, n_pairs, block):
        p1 = min(n_pairs, p0 + block)
        flat = cls_x[p0:p1] + (cp.arange(p1 - p0, dtype=cls_x.dtype) * max_joint)[:, None]
        px = cp.bincount(flat.ravel(), minlength=(p1 - p0) * max_joint).reshape(p1 - p0, max_joint).astype(cp.float64) * inv_n  # Intentional float64: see above
        nz = px > 0
        h_x[p0:p1] = -cp.sum(cp.where(nz, px * cp.log(cp.where(nz, px, 1.0)), 0.0), axis=1)
    return h_x


def pooled_pair_permutation_null_joint_mi_floor_cupy(
    factors_data: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    *,
    n_permutations: int = 25,
    quantile: float = 0.95,
    mm_debias: bool = False,
    mm_bias: np.ndarray | None = None,
    random_seed: int | None = None,
) -> float | None:
    """Resident-GPU twin of :func:`_permutation_null.pooled_pair_permutation_null_joint_mi_floor`.

    Returns the order-2 maxT permutation-null floor (host double precision scalar): the ``quantile``-th quantile of the
    K per-shuffle MAX plug-in joint MIs over the candidate-pair pool. Same pooled-MAX construction as the CPU
    njit floor, just on the device with the pool operands resident and the K target shuffles device-born.

    ``mm_bias`` (if ``mm_debias``) is the per-pair Miller-Madow joint-MI bias vector (length ``n_pairs``),
    permutation-invariant; it is subtracted from EACH pair's joint MI before the per-shuffle max, exactly like
    the CPU floor. When ``mm_debias`` and ``mm_bias is None`` the bias is recomputed on host (occupied joint-K)
    so the two paths debias identically.

    Returns ``None`` on ANY cupy error (OOM / device fault) -> caller falls back to the exact CPU njit floor.
    Returns ``0.0`` for the same degenerate pools the CPU floor no-ops on (n<8, <2 pairs, single-class y,
    K<1)."""
    try:
        import cupy as cp

        n = int(factors_data.shape[0])
        n_pairs = length_of(pair_a)
        nperm = int(n_permutations)
        if n < 8 or nperm < 1 or n_pairs < 2:
            return 0.0
        n_classes_y = int(shape_of(freqs_y)[0])
        if n_classes_y < 2:
            return 0.0

        # The same row cap as the CPU / one-launch floor (``_permutation_null._pair_maxt_max_rows``, skipped when mm_debias): the floor is a coarse noise threshold whose dominant term is the plug-in
        # bias ~ (k_joint - 1)(k_y - 1) / 2n, so the twin must read the same rows or its floor sits on another scale (7 times lower at 100000 rows) and the host-or-device route changes which pairs pass.
        from ._permutation_null import _pair_maxt_max_rows

        _max_rows = 0 if mm_debias else _pair_maxt_max_rows()
        if 0 < _max_rows < n:
            factors_data = factors_data[:_max_rows]
            classes_y = np.asarray(classes_y)[:_max_rows]
            n = _max_rows
            freqs_y = np.bincount(np.asarray(classes_y).astype(np.int64), minlength=n_classes_y).astype(np.float64) / n  # Intentional float64: y marginal of the capped rows

        pa = np.ascontiguousarray(pair_a, dtype=np.int64)
        pb_idx = np.ascontiguousarray(pair_b, dtype=np.int64)
        nb = np.ascontiguousarray(nbins, dtype=np.int64)

        # Per-pair MM bias (occupied joint-K), permutation-invariant. Recompute on host if not supplied so the
        # device floor debiases IDENTICALLY to the CPU floor (which uses the same occupied joint-K bias).
        d_mm = None
        if mm_debias:
            if mm_bias is None:
                from ._permutation_null import _pairwise_occupied_joint_k

                k_joint = _pairwise_occupied_joint_k(factors_data, pa, pb_idx, nb)
                k_y = n_classes_y
                # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
                _bias = (k_joint - 1).astype(np.float64) * float(k_y - 1) / (2.0 * n)
                _bias[k_joint <= 1] = 0.0
            else:
                # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
                _bias = np.ascontiguousarray(mm_bias, dtype=np.float64)
            d_mm = cp.asarray(_bias)[:, None]  # (n_pairs, 1)

        inv_n = 1.0 / n

        # ---- Resident pool operands (uploaded ONCE) -----------------------------------------------------
        # factors_data stays in its native (typically int32) dtype -> no int64 RAM blow-up of the screening
        # matrix; the per-pair joint X-code is built as int64 below (bounded by max joint card * n_classes_y).
        d_data = cp.asarray(np.ascontiguousarray(factors_data))  # (n, n_features), native dtype
        d_pa = cp.asarray(pa)  # (n_pairs,)
        d_pb = cp.asarray(pb_idx)  # (n_pairs,)
        d_nb = cp.asarray(nb)  # (n_features,)

        # Per-pair joint X-code cls_x = x_a * nbins_b + x_b, PERMUTATION-INVARIANT -> built ONCE, (n_pairs, n).
        nb_b = d_nb[d_pb][:, None]  # (n_pairs, 1)
        x_a = d_data[:, d_pa].T.astype(cp.int64)  # (n_pairs, n)
        x_b = d_data[:, d_pb].T.astype(cp.int64)  # (n_pairs, n)
        cls_x = x_a * nb_b + x_b  # (n_pairs, n) joint X-code
        del x_a, x_b
        # Per-pair joint cardinality (host) drives the flat-index stride / bincount extent per tile.
        joint_card = (nb[pa] * nb[pb_idx]).astype(np.int64)  # (n_pairs,)
        max_joint = int(joint_card.max())
        per_pair_extent = max_joint * n_classes_y  # uniform stride per pair (pad-safe)

        # Marginal H(x) per pair (permutation-invariant): -sum p log p over the joint X-code counts.
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        h_x = _joint_code_entropies(cp, cls_x, max_joint, inv_n)
        h_x = h_x[:, None]  # (n_pairs, 1)

        # H(y) is permutation-invariant (freqs_y unchanged under relabelling) -> one host scalar.
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        fy = np.ascontiguousarray(freqs_y, dtype=np.float64)
        fy_nz = fy[fy > 0]
        h_y = float(-(fy_nz * np.log(fy_nz)).sum())

        # ---- Device-born K target shuffles (argsort of i.i.d. keys) --------------------------------------
        seed = 0x9E3779B9 if random_seed is None else (int(random_seed) & 0x7FFFFFFF)
        rs = cp.random.RandomState(seed)
        d_y = cp.asarray(np.ascontiguousarray(classes_y).astype(np.int64))  # (n,) one tiny upload
        keys = rs.random_sample(size=(nperm, n))  # (nperm, n) device draw, no H2D
        order = device_argsort(keys, axis=1)  # per-row uniform permutation
        del keys
        y_perms = d_y[order]  # (nperm, n) device shuffles
        del order

        # ---- Per-shuffle pooled MAX joint MI, tiled over perms -------------------------------------------
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        d_maxes = cp.empty(nperm, dtype=cp.float64)
        # Tile perms so the (pb * n_pairs * n) flat-joint intermediate fits the card.
        # the histogram of a tile is (pb * n_pairs * per_pair_extent) cells and the entropy step holds about five float64 copies of it: bound that product as well as the (pb * n_pairs * n) joint codes (a many-bin
        # strategy gave a 1.95 GB bincount on a 4 GB card, the OOM tripped the circuit breaker and every later floor of the process ran on the CPU)
        _cells = _tile_cell_budget(cp)
        perm_chunk = max(1, min(_cells // max(1, n_pairs * n), _cells // max(1, n_pairs * per_pair_extent)))
        from mlframe.feature_selection.filters._permutation_null_pair_tiles import _pooled_pair_null_perm_tiles
        _pair_chunk = max(1, min(n_pairs, _cells // max(1, per_pair_extent)))  # a permutation's histogram of every pair above the budget: pairs in blocks
        _pooled_pair_null_perm_tiles(nperm, perm_chunk, y_perms, cp, n_pairs, per_pair_extent, cls_x, n_classes_y, inv_n, h_x, h_y, d_mm, d_maxes, pair_chunk=_pair_chunk)

        maxes = cp.asnumpy(d_maxes)  # ONLY (nperm,) crosses back to host
        return float(np.quantile(maxes, float(quantile)))
    except Exception as exc:
        if _is_cupy_out_of_memory(exc):
            # Recoverable: an out-of-memory error leaves the context healthy. Empty the pool and send THIS call to the exact CPU floor; the next call may fit (tripping the breaker sent every later floor of the
            # process to the CPU: measured in a profiled 100000 x 100 fit, 693 MB refused under a 2.58 GB pool limit).
            import logging

            from mlframe.utils.log_throttle import log_throttle

            _free_device_pool()
            log_throttle(
                logging.getLogger("mlframe.feature_selection.filters.mrmr"), "pair_maxt_resident.oom", logging.WARNING,
                "resident order-2 pair-maxT floor ran out of device memory (%s); this call uses the CPU njit floor, the GPU path stays armed", exc,
            )
            return None
        # A WDDM-TDR cudaErrorLaunchFailure POISONS the CUDA context; returning None WITHOUT tripping the
        # breaker made every later FE step re-attempt the poisoned GPU and re-fault (a ~1h CPU floor made
        # invisible). Trip on the first fault so subsequent calls skip straight to the exact CPU njit floor.
        import logging

        logging.getLogger("mlframe.feature_selection.filters.mrmr").warning(
            "resident order-2 pair-maxT floor faulted; tripping the GPU circuit breaker and using the CPU njit floor for the rest of this process", exc_info=True,
        )
        trip_pair_maxt_gpu_circuit_breaker()
        return None


def _is_cupy_out_of_memory(exc: BaseException) -> bool:
    """Whether ``exc`` is cupy's device or pool out-of-memory error (recoverable, unlike a launch fault that poisons the context)."""
    try:
        import cupy as cp
    except ImportError:  # no cupy, no cupy error
        return False
    return isinstance(exc, cp.cuda.memory.OutOfMemoryError)


def _free_device_pool() -> None:
    """Return the unused blocks of the default device pool so the next allocation can succeed; never raises."""
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
    except Exception as exc:  # best-effort cleanup, the CPU floor runs either way
        import logging

        logging.getLogger(__name__).debug("freeing the device pool failed: %s", exc)


# Process-level circuit breaker (mirrors info_theory._cmi_cuda._CMI_GPU_FAILED). A WDDM-TDR
# cudaErrorLaunchFailure on a small/weak card (GTX 1050 Ti = 4 GB) POISONS the CUDA context, so every
# subsequent resident-floor attempt this process re-faults and silently falls back to the ~1h CPU floor.
# Trip the breaker on the FIRST fault so later calls skip the GPU immediately (no futile re-attempt spam)
# and go straight to the CPU njit floor. Reset only via reset_pair_maxt_gpu_circuit_breaker() (tests).
_PAIR_MAXT_GPU_FAILED = False


def trip_pair_maxt_gpu_circuit_breaker() -> None:
    """Mark the resident pair-maxT GPU path dead for the rest of the process (called on a launch fault)."""
    global _PAIR_MAXT_GPU_FAILED
    _PAIR_MAXT_GPU_FAILED = True


def reset_pair_maxt_gpu_circuit_breaker() -> None:
    """Re-arm the resident pair-maxT GPU path (tests / after a fresh CUDA context)."""
    global _PAIR_MAXT_GPU_FAILED
    _PAIR_MAXT_GPU_FAILED = False


def pair_maxt_perm_null_gpu_enabled(n: int, n_pairs: int) -> bool:
    """Whether the resident-GPU order-2 maxT permutation-null floor engages for this (n, n_pairs).

    DEVICE-BORN UNCONDITIONALLY under STRICT GPU mode (``MLFRAME_FE_GPU_STRICT``), no KTC crossover: STRICT means
    "every data-touching kernel runs on the device", so the resident floor engages whenever STRICT is on (``n`` /
    ``n_pairs`` are accepted for signature stability but no longer gate). ``MLFRAME_FE_PAIR_MAXT_PERM_NULL_GPU=0``
    is the explicit opt-out that ALWAYS wins (forces the CPU njit floor even under STRICT). The non-STRICT default
    keeps the exact CPU njit floor (byte-identical).

    The device floor is selection-equivalent to the CPU floor (same pooled-MAX construction, same plug-in joint
    MI of the same integer contingency table to FP round-off, host-owned quantile) so engaging it never flips the
    gate decision ``pair_mi >= floor`` - verified in ``test_pair_maxt_perm_null_resident.py``. On a small / weak
    card the resident floor can be a wall LOSS (measured 0.26-0.66x vs the CPU njit floor on a GTX 1050 Ti); that
    is accepted under STRICT (residency is the contract, not the wall on a weak card), and the non-STRICT default
    path is unaffected.

    Returns ``False`` (caller stays on the exact CPU njit floor) on the opt-out, non-STRICT, or any failure."""
    if os.environ.get("MLFRAME_FE_PAIR_MAXT_PERM_NULL_GPU", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    if _PAIR_MAXT_GPU_FAILED:  # context poisoned by a prior launch fault -> never re-attempt the GPU this process.
        return False
    # ABSOLUTE cushion guard: even under STRICT, decline the resident GPU floor on a near-full /
    # SHARED card so its (n, n_pairs) device buffers do not fault the next launch -> the caller keeps the exact
    # CPU njit floor. Deliberately checked BEFORE the STRICT gate: a cushion violation is REAL OOM risk (unlike a
    # conservative size threshold STRICT may legitimately force past). Permissive without cupy. Pure ADD.
    try:
        from ._fe_gpu_vram import fe_gpu_has_vram_cushion
        if not fe_gpu_has_vram_cushion(int(n) * max(int(n_pairs), 1) * 8):
            import logging
            from mlframe.utils.log_throttle import log_throttle
            log_throttle(logging.getLogger(__name__), "pair_maxt_resident.vram_cushion", logging.WARNING, "order-2 pair-maxT resident floor declined: no VRAM cushion for %d bytes, using the CPU njit floor", int(n) * max(int(n_pairs), 1) * 8)
            return False
    except Exception as e:  # nosec B110 - best-effort/optional path
        import logging
        logging.getLogger(__name__).debug("VRAM cushion probe failed, permissive default keeps resident path eligible: %s", e)
    try:
        from ._fe_gpu_strict import fe_gpu_strict_enabled

        return bool(fe_gpu_strict_enabled(n=int(n), p=int(n_pairs)))
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("pair_maxt_perm_null_gpu_enabled: fe_gpu_strict_enabled() check failed, caller keeps the exact CPU njit floor: %s", e)
        return False


__all__ = [
    "pooled_pair_permutation_null_joint_mi_floor_cupy",
    "pair_maxt_perm_null_gpu_enabled",
    "trip_pair_maxt_gpu_circuit_breaker",
    "reset_pair_maxt_gpu_circuit_breaker",
]

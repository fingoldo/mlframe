"""Batched (multi-permutation-per-launch) GPU MI kernels -- ``mi_direct_gpu_batched`` and its CUDA-streams twin.

Split out from ``gpu.py`` to keep that file below the 1k-line monolith threshold (both functions carry
extensive perf-history comments and together ran to ~420 lines). Behaviour preserved bit-for-bit; both
names are re-exported from ``gpu`` so existing imports (``from mlframe.feature_selection.filters.gpu import
mi_direct_gpu_batched``) continue to work.

The CUDA kernels themselves (``compute_joint_hist_batched_cuda`` etc.) stay module attributes of ``gpu.py``,
populated lazily by ``gpu.init_kernels()``; this module resolves them via ``from . import gpu as
_gpu_module`` + attribute access at CALL time (not import time), since importing the bare names directly
would cache the pre-init ``None`` placeholder.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ._internals import GPU_MAX_BLOCK_SIZE
from .info_theory import compute_mi_from_classes, merge_vars


def _gpu_batched_bytes_per_perm(n: int) -> int:
    """True per-row VRAM cost of one batch iteration of ``mi_direct_gpu_batched`` / ``_streamed`` (2026-07-09
    fix; shared by both OOM guards so the formula lives in exactly one place).

    Each iteration holds simultaneously: ``rand`` (b,n) float64 = 8 bytes/row, ``perm_idx`` (b,n) int64 =
    8 bytes/row (``cp.argsort`` output dtype), and ``perms_y`` (b,n) int32 = 4 bytes/row -- 20 bytes/row.
    The prior formula used ``n * 4`` (only ``perms_y``), a ~5x undercount of actual peak VRAM need for a
    given ``batch_size``."""
    return int(n) * 20


def _mi_from_counts_cupy(cp, counts_2d, n, nbins_x, nbins_y, outer_safe):
    """(b, joint_size) int counts -> (b,) MI in nats via the noise-gate reduction (masked-safe ratio; shared by both mi_direct_gpu_batched twins)."""
    jf = (counts_2d.astype(cp.float64) / n).reshape(-1, nbins_x, nbins_y)
    ratio = cp.where(jf > 0, jf / outer_safe, 1.0)
    return (jf * cp.log(ratio)).sum(axis=(1, 2))


def mi_direct_gpu_batched(
    factors_data: Any,
    x: tuple,
    y: tuple,
    factors_nbins: np.ndarray,
    npermutations: int = 100,
    batch_size: int = 64,
    dtype: type = np.int32,
    classes_y: Optional[np.ndarray] = None,
    freqs_y: Optional[np.ndarray] = None,
    min_nonzero_confidence: float = 0.95,
) -> tuple:
    """Batched GPU permutation MI test.

    Generates ``npermutations`` shuffled copies of ``classes_y`` once, chunks them into batches of ``batch_size``, and processes each batch in a SINGLE kernel
    launch (instead of one launch per permutation as in ``mi_direct_gpu``). Cuts kernel-launch overhead from O(npermutations) to O(npermutations / batch_size).

    Auto-fallback: if ``batch_size * n * 4 bytes`` would exceed half of free GPU memory, ``batch_size`` shrinks to fit. For small datasets the overhead of
    permutation matrix generation can outweigh the saved launches; ``mi_direct_gpu`` remains the right call for ``npermutations < 32``.

    Returns ``(original_mi, confidence)`` -- same contract as ``mi_direct_gpu``.
    """
    import cupy as cp

    from . import gpu as _gpu_module

    _gpu_module._ensure_kernels_inited()
    _gpu_module._pin_device_if_needed()  # D2 part 2: per-thread CUDA device pin

    classes_x, freqs_x, _ = merge_vars(
        factors_data=factors_data, vars_indices=x,
        var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
    )
    if classes_y is None:
        classes_y, freqs_y, _ = merge_vars(
            factors_data=factors_data, vars_indices=y,
            var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
        )
    assert freqs_y is not None  # the None branch above always sets classes_y/freqs_y together

    original_mi = compute_mi_from_classes(
        classes_x=classes_x, freqs_x=freqs_x,
        classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
    )

    if original_mi <= 0 or npermutations <= 0:
        return original_mi, 0.0

    n = len(classes_x)
    nbins_x = len(freqs_x)
    nbins_y = len(freqs_y)

    # ABSOLUTE cushion guard (2026-07-05): decline the GPU entirely on a near-full / SHARED card BEFORE the
    # relative half-free batch cap below (which is computed only after the cupy pool may have already eaten the
    # device). On a cushion violation route to the exact CPU MI path (selection-equivalent, prefer_gpu=False so
    # it does not recurse back onto the GPU) instead of launching a kernel that would fault. Pure ADD -- tightens.
    try:
        from mlframe.feature_selection.filters._fe_gpu_vram import fe_gpu_has_vram_cushion
        _cushion_ok = fe_gpu_has_vram_cushion(n * 4)
    except Exception:  # -- cushion module unavailable: leave existing guards in charge
        _cushion_ok = True
    if not _cushion_ok:
        from .permutation import mi_direct
        return mi_direct(
            factors_data, x=x, y=y, factors_nbins=factors_nbins,
            npermutations=npermutations, dtype=dtype,
            classes_y=classes_y, freqs_y=freqs_y,
            min_nonzero_confidence=min_nonzero_confidence, prefer_gpu=False,
        )
    # OOM guard: cap batch_size to half of available free GPU memory.
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    bytes_per_perm = _gpu_batched_bytes_per_perm(n)
    safe_batch = max(1, int(free_bytes // 2 // bytes_per_perm))
    if batch_size > safe_batch:
        batch_size = safe_batch

    classes_x_gpu = cp.asarray(classes_x.astype(np.int32))
    classes_y_gpu = cp.asarray(classes_y.astype(np.int32))
    freqs_x_gpu = cp.asarray(freqs_x).astype(cp.float64)
    freqs_y_gpu = cp.asarray(freqs_y).astype(cp.float64)

    # Joint-hist kernel dispatch via the per-host kernel_tuning_cache. The
    # cache is built on first use by a 30s sweep and persisted to
    # ``~/.mlframe/kernel_tuning/{hw_fingerprint}.json``. Subsequent calls
    # do an O(N_regions) dict lookup (N_regions <= ~10). Falls back to the
    # hand-tuned source-code default (shared, bs=512) on cache miss /
    # CUDA-unavailable / lookup error so production fits never block on
    # the cache.
    #
    # Both kernel variants stay alongside per ``feedback_keep_all_kernel_versions``:
    #   * compute_joint_hist_batched_shared_cuda -- shared-mem atomic;
    #     wins at most (n_samples, joint_size) combos.
    #   * compute_joint_hist_batched_cuda -- global-atomic; wins at large
    #     joint_size on certain (n_samples, block_size) combos that the
    #     auto-tune sweep discovered (e.g. n=1M, joint=400 on cc 6.1).
    joint_size = nbins_x * nbins_y
    try:
        from mlframe.feature_selection._benchmarks.kernel_tuning_cache.dispatch import (
            lookup_joint_hist,
        )
        _choice = lookup_joint_hist(n_samples=n, joint_size=joint_size)
        use_shared_hist = _choice["kernel_variant"] == "shared"
        block_size = int(_choice["block_size"])
    except Exception:
        # Hand-tuned source-code fallback (matches the pre-cache defaults).
        use_shared_hist = joint_size <= 4096
        block_size = 512 if use_shared_hist else GPU_MAX_BLOCK_SIZE

    shared_mem_bytes = joint_size * 4 if use_shared_hist else 0
    grid_x = (n + block_size - 1) // block_size

    # ``.get()`` coalescing v2: stage each batch's failure-count as a
    # CuPy 0-d array in a Python list, then sync ONCE at end-of-loop via
    # ``cp.stack([...]).sum().get()``. The previous per-batch
    # ``int((mi_batch >= original_mi).sum().get())`` pattern triggered a
    # cross-device sync on every iteration -- profile on 1M x 30,
    # fe_npermutations=50 showed ``cupy.ndarray.get`` at 5.2s of 13.2s fit
    # wall (39%) with 42 calls = ~124ms each (Windows WDDM driver fault
    # latency).
    #
    # An earlier attempt (commit 7319f11 noted in its log) using a
    # device-side scalar accumulator ``cp.zeros((), int64)`` with ``+=``
    # measured worse at n=10k because per-batch in-place add on a 0-d
    # array is itself a kernel launch. The list-of-arrays pattern below
    # avoids both pitfalls: no per-batch sync, no per-batch device-side
    # arithmetic; just queue compute, sync once.
    # Loop-invariant marginal product + masked-safe form, hoisted out of the batch loop.
    outer_marginals = freqs_x_gpu[:, None] * freqs_y_gpu[None, :]
    outer_safe = cp.where(outer_marginals[None, :, :] > 0, outer_marginals[None, :, :], 1.0)

    # GATE REFERENCE (FP-consistency, 2026-06-22): the gate counts #{perm_mi >= observed_mi}; mi_batch is
    # GPU-reduced, so recompute the observed MI through the SAME kernel+reduction on the UNPERMUTED y (a
    # ~1e-15 order delta vs the CPU-njit original_mi flips a count). CPU original_mi stays the RETURNED value.
    _id_counts = cp.zeros((1, nbins_x * nbins_y), dtype=cp.int32)
    _id_perm = classes_y_gpu.reshape(1, n).astype(cp.int32)
    if use_shared_hist:
        _gpu_module.compute_joint_hist_batched_shared_cuda(
            (grid_x, 1), (block_size,),
            (classes_x_gpu, _id_perm, _id_counts, np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
            shared_mem=shared_mem_bytes,
        )
    else:
        _gpu_module.compute_joint_hist_batched_cuda(
            (grid_x, 1), (block_size,),
            (classes_x_gpu, _id_perm, _id_counts, np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
        )
    original_mi_gpu = _mi_from_counts_cupy(cp, _id_counts, n, nbins_x, nbins_y, outer_safe)  # (1,) device scalar, compared against mi_batch below

    batch_failures = []  # list of CuPy 0-d arrays
    nchecked = 0
    remaining = npermutations
    # Use the modern cupy Generator (XORWOW) rather than the legacy global cp.random.uniform: the legacy
    # host-API cuRAND generator fails to initialise (CURAND_STATUS_INITIALIZATION_FAILED) on some
    # driver/lib combos where the Generator API works fine -- and the streamed twin already uses default_rng.
    _rng = cp.random.default_rng()
    while remaining > 0:
        b = min(batch_size, remaining)
        # ``cp.argsort(random((b, n)))`` is statistically equivalent to Fisher-Yates for permutation tests (argsort of distinct floats is a bijection).
        rand = _rng.random((b, n), dtype=cp.float64)
        perm_idx = cp.argsort(rand, axis=1)  # (b, n) int64
        # Gather classes_y at these indices -> shuffled copies.
        perms_y = classes_y_gpu[perm_idx].astype(cp.int32)

        joint_counts_batch = cp.zeros((b, nbins_x * nbins_y), dtype=cp.int32)
        if use_shared_hist:
            _gpu_module.compute_joint_hist_batched_shared_cuda(
                (grid_x, b),
                (block_size,),
                (
                    classes_x_gpu, perms_y, joint_counts_batch,
                    np.int32(n), np.int32(nbins_x), np.int32(nbins_y),
                ),
                shared_mem=shared_mem_bytes,
            )
        else:
            _gpu_module.compute_joint_hist_batched_cuda(
                (grid_x, b),
                (block_size,),
                (
                    classes_x_gpu, perms_y, joint_counts_batch,
                    np.int32(n), np.int32(nbins_x), np.int32(nbins_y),
                ),
            )

        # MI per row via the SAME reduction as the gate reference (the masked-safe ratio avoids nan/inf in
        # cp.log; outer_safe hoisted above). joint_freqs = joint_counts / n; mi = sum p_xy log(p_xy/(p_x p_y)).
        mi_batch = _mi_from_counts_cupy(cp, joint_counts_batch, n, nbins_x, nbins_y, outer_safe)  # (b,)

        # Stage on device; defer cross-device sync. Compare against the GPU-reduced observed MI (same FP
        # order as mi_batch) -- NOT the CPU-njit original_mi -- so an equality near-tie cannot spuriously flip.
        batch_failures.append((mi_batch >= original_mi_gpu).sum())
        nchecked += b
        remaining -= b

    # Single end-of-loop sync. cp.stack into 1-D array of length len(batch_failures),
    # sum along axis 0, .get() once. For small batch counts (e.g. n=10k -> 8 batches)
    # the stack is cheap; for large (e.g. 1M -> 8 batches per call, ~9 calls -> 72
    # total) we still pay only 1 sync per call instead of N per call.
    if batch_failures:
        nfailed = int(cp.stack(batch_failures).sum())
    else:
        nfailed = 0

    confidence = (1.0 - nfailed / nchecked) if nchecked > 0 else 0.0
    # 2026-05-30 Wave 9.1 fix (loop iter 4): respect caller's
    # min_nonzero_confidence instead of the previous hardcoded 0.05
    # threshold. The hardcode silently diverged CPU and GPU paths when the
    # caller (default MRMR ctor: min_nonzero_confidence=0.99) wanted a
    # tighter gate. Same mirror as permutation.py:344 CPU path. The
    # ``max(1, ...)`` clamp protects against degenerate ``npermutations``
    # auto-rejecting every candidate when the threshold rounds to 0.
    max_failed = int(npermutations * (1.0 - float(min_nonzero_confidence)))
    if max_failed <= 1:
        max_failed = 1
    if nfailed >= max_failed:
        original_mi = 0.0

    return original_mi, confidence


def mi_direct_gpu_batched_streamed(
    factors_data: Any,
    x: tuple,
    y: tuple,
    factors_nbins: np.ndarray,
    npermutations: int = 100,
    batch_size: int = 64,
    dtype: type = np.int32,
    classes_y: Optional[np.ndarray] = None,
    freqs_y: Optional[np.ndarray] = None,
    n_streams: int = 2,
    min_nonzero_confidence: float = 0.95,
) -> tuple:
    """CUDA-streams variant of :func:`mi_direct_gpu_batched`.

    Same contract (``(original_mi, confidence)``) but the batch loop
    alternates iterations across ``n_streams`` non-blocking CuPy streams
    (default 2). This lets iter (i+1)'s ``cp.random.uniform`` + argsort
    queue concurrently with iter (i)'s histogram kernel + MI math,
    potentially filling SM idle gaps that the single-stream variant leaves.

    On cc 6.1 (GTX 1050 Ti, 6 SMs) the SMs are usually fully saturated by
    one kernel at a time so the overlap is minimal; bench expected
    speedup is 0-15%. On cc 8.x (Ampere, 100+ SMs) the overlap is larger.

    Per ``feedback_keep_all_kernel_versions``: kept ALONGSIDE the original
    ``mi_direct_gpu_batched``. The dispatcher (kernel_tuning_cache) can
    route to either via the ``variant`` field after a sweep finds the
    crossover.

    Caveats:
    * Streams share the GPU memory pool; over-subscribing batches can
      cause OOM at smaller n. ``batch_size`` is still subject to the
      memGetInfo-derived cap (same as the non-streamed variant).
    * The cupy legacy ``cp.random.uniform`` global RNG serialises through
      a Python-side lock; with 2 streams the lock contention is tiny but
      observable. Migrating to ``cp.random.Generator`` per stream would
      remove the lock at the cost of more state.
    """
    import cupy as cp

    from . import gpu as _gpu_module

    _gpu_module._ensure_kernels_inited()
    _gpu_module._pin_device_if_needed()  # D2 part 2: per-thread CUDA device pin

    classes_x, freqs_x, _ = merge_vars(
        factors_data=factors_data, vars_indices=x,
        var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
    )
    if classes_y is None:
        classes_y, freqs_y, _ = merge_vars(
            factors_data=factors_data, vars_indices=y,
            var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
        )
    assert freqs_y is not None  # the None branch above always sets classes_y/freqs_y together

    original_mi = compute_mi_from_classes(
        classes_x=classes_x, freqs_x=freqs_x,
        classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
    )
    if original_mi <= 0 or npermutations <= 0:
        return original_mi, 0.0

    n = len(classes_x)
    nbins_x = len(freqs_x)
    nbins_y = len(freqs_y)

    # ABSOLUTE cushion guard (2026-07-05): as in mi_direct_gpu_batched, decline the GPU on a near-full / shared
    # card BEFORE the relative half-free cap and route to the exact CPU MI path (selection-equivalent). Pure ADD.
    try:
        from mlframe.feature_selection.filters._fe_gpu_vram import fe_gpu_has_vram_cushion
        _cushion_ok = fe_gpu_has_vram_cushion(n * 4)
    except Exception:
        _cushion_ok = True
    if not _cushion_ok:
        from .permutation import mi_direct
        return mi_direct(
            factors_data, x=x, y=y, factors_nbins=factors_nbins,
            npermutations=npermutations, dtype=dtype,
            classes_y=classes_y, freqs_y=freqs_y,
            min_nonzero_confidence=min_nonzero_confidence, prefer_gpu=False,
        )

    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    bytes_per_perm = _gpu_batched_bytes_per_perm(n)
    safe_batch = max(1, int(free_bytes // 2 // bytes_per_perm))
    if batch_size > safe_batch:
        batch_size = safe_batch

    classes_x_gpu = cp.asarray(classes_x.astype(np.int32))
    classes_y_gpu = cp.asarray(classes_y.astype(np.int32))
    freqs_x_gpu = cp.asarray(freqs_x).astype(cp.float64)
    freqs_y_gpu = cp.asarray(freqs_y).astype(cp.float64)

    joint_size = nbins_x * nbins_y
    # Wave 23 #1 fix (2026-05-20): mirror the sibling lookup at L466-475
    # so this streamed variant adapts to live HW. Pre-fix it open-coded
    # the same hand-tuned defaults from before kernel_tuning_cache landed
    # and silently fell behind on non-dev hardware.
    try:
        from mlframe.feature_selection._benchmarks.kernel_tuning_cache.dispatch import (
            lookup_joint_hist,
        )
        _choice = lookup_joint_hist(n_samples=n, joint_size=joint_size)
        use_shared_hist = _choice["kernel_variant"] == "shared"
        block_size = int(_choice["block_size"])
    except Exception:
        # Hand-tuned source-code fallback (matches the pre-cache defaults).
        use_shared_hist = joint_size <= 4096
        block_size = 512 if use_shared_hist else GPU_MAX_BLOCK_SIZE
    shared_mem_bytes = joint_size * 4 if use_shared_hist else 0
    grid_x = (n + block_size - 1) // block_size

    # B2 fix: ``classes_y_gpu``, ``freqs_x_gpu``, ``freqs_y_gpu`` were just
    # allocated on the default stream. Non-blocking child streams do NOT
    # implicitly wait for the default stream's H2D copies, so the first
    # iter could read garbage. Force a sync before launching child streams.
    cp.cuda.Stream.null.synchronize()

    # FP-consistency gate reference (mirrors the non-streamed twin, 2026-06-22): reduce the OBSERVED MI via
    # the same kernel+reduction on the unpermuted y (default stream, before child streams) so the >= gate
    # compares like-with-like. CPU original_mi stays the RETURNED value.
    outer_marginals = freqs_x_gpu[:, None] * freqs_y_gpu[None, :]
    outer_safe = cp.where(outer_marginals[None, :, :] > 0, outer_marginals[None, :, :], 1.0)

    _id_counts = cp.zeros((1, joint_size), dtype=cp.int32)
    _id_perm = classes_y_gpu.reshape(1, n).astype(cp.int32)
    if use_shared_hist:
        _gpu_module.compute_joint_hist_batched_shared_cuda(
            (grid_x, 1), (block_size,),
            (classes_x_gpu, _id_perm, _id_counts, np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
            shared_mem=shared_mem_bytes,
        )
    else:
        _gpu_module.compute_joint_hist_batched_cuda(
            (grid_x, 1), (block_size,),
            (classes_x_gpu, _id_perm, _id_counts, np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
        )
    cp.cuda.Stream.null.synchronize()
    original_mi_gpu = _mi_from_counts_cupy(cp, _id_counts, n, nbins_x, nbins_y, outer_safe)  # (1,) device scalar for the gate comparison below

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(max(1, n_streams))]
    # B1 fix: cp.random.uniform uses a singleton device-side RandomState
    # whose curandState advance is NOT safe under concurrent stream access.
    # One Generator per stream avoids the race; each generator's Philox
    # state advance stays serial within its own stream.
    rngs = [cp.random.default_rng() for _ in range(len(streams))]
    batch_failures = []
    nchecked = 0
    remaining = npermutations
    iter_idx = 0
    while remaining > 0:
        b = min(batch_size, remaining)
        stream = streams[iter_idx % len(streams)]
        rng = rngs[iter_idx % len(rngs)]
        with stream:
            # ``rng.random`` (per-stream) replaces the legacy global
            # ``cp.random.uniform`` to stay race-free across streams.
            rand = rng.random(size=(b, n), dtype=cp.float64)
            perm_idx = cp.argsort(rand, axis=1)
            perms_y = classes_y_gpu[perm_idx].astype(cp.int32)

            joint_counts_batch = cp.zeros((b, joint_size), dtype=cp.int32)
            if use_shared_hist:
                _gpu_module.compute_joint_hist_batched_shared_cuda(
                    (grid_x, b),
                    (block_size,),
                    (classes_x_gpu, perms_y, joint_counts_batch, np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
                    shared_mem=shared_mem_bytes,
                )
            else:
                _gpu_module.compute_joint_hist_batched_cuda(
                    (grid_x, b),
                    (block_size,),
                    (classes_x_gpu, perms_y, joint_counts_batch, np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
                )
            # Same reduction as the gate reference (outer_safe + closure hoisted above); mask the ratio (not
            # the log) to keep nan/inf out of cp.log entirely.
            mi_batch = _mi_from_counts_cupy(cp, joint_counts_batch, n, nbins_x, nbins_y, outer_safe)
            batch_failures.append((mi_batch >= original_mi_gpu).sum())
        iter_idx += 1
        nchecked += b
        remaining -= b

    # Sync all streams before the host-side aggregation.
    for s in streams:
        s.synchronize()

    if batch_failures:
        nfailed = int(cp.stack(batch_failures).sum())
    else:
        nfailed = 0

    confidence = (1.0 - nfailed / nchecked) if nchecked > 0 else 0.0
    # 2026-05-30 Wave 9.1 fix (loop iter 4): same fix as
    # ``mi_direct_gpu_batched`` above - respect caller's
    # min_nonzero_confidence; clamp to avoid degenerate auto-reject.
    max_failed = int(npermutations * (1.0 - float(min_nonzero_confidence)))
    if max_failed <= 1:
        max_failed = 1
    if nfailed >= max_failed:
        original_mi = 0.0
    return original_mi, confidence

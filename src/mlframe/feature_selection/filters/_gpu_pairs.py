"""Cat-FE GPU dispatch -- batched joint MI computation across many pairs.

Wave 99 (2026-05-21): split out from ``gpu.py`` to keep that file
below the 1k-line monolith threshold. Behaviour preserved bit-for-bit;
``mi_direct_gpu_batched_pairs`` is re-exported from ``gpu`` so
existing imports continue to work.
"""
from __future__ import annotations

import logging

import numpy as np

from ._internals import GPU_MAX_BLOCK_SIZE

logger = logging.getLogger(__name__)


# ============================================================================


def mi_direct_gpu_batched_pairs(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    factors_nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype=np.int32,
) -> np.ndarray:
    """Compute ``I(X_i, X_j; Y)`` for every ``(i, j) in zip(pairs_a, pairs_b)`` on GPU. Returns a 1-D ``float64`` array of joint MIs aligned with the input order.

    Kernel-level batching via 3D grid: ONE launch processes all pairs in parallel through ``compute_joint_hist_multi_pair_cuda``. Joint MI per pair is then
    computed on CPU (cheap relative to histogram aggregation). Cuts per-pair kernel-launch overhead (~30us) to zero -- at n_pairs=4950 that saves ~150ms vs the
    naive per-pair loop.

    Preconditions: CuPy installed and a GPU available; ``classes_y`` / ``freqs_y`` precomputed by the caller. Raises a clear error if CuPy is missing -- callers
    must resolve ``backend`` before calling.
    """
    try:
        import cupy as cp
    except ImportError as e:
        raise RuntimeError(
            "mi_direct_gpu_batched_pairs requires CuPy; install via "
            "`pip install cupy-cudaXX` matching your CUDA toolkit, "
            "or set CatFEConfig(backend='cpu')."
        ) from e

    # Wave 99: lazy-import the parent module so kernel symbols can be
    # resolved AFTER _ensure_kernels_inited() populates them. The kernel
    # globals start as None at module-load and get reassigned by
    # init_kernels(); importing the names directly would cache the None.
    from . import gpu as _gpu_module

    _gpu_module._ensure_kernels_inited()
    _gpu_module._pin_device_if_needed()  # per-thread CUDA device pin
    n_pairs = len(pairs_a)
    n_rows = factors_data.shape[0]
    if n_pairs == 0:
        return np.zeros(0, dtype=np.float64)

    # E1 fix: CUDA gridDim.y maxes at 65535 on cc 6.x (still 65535 on cc 7+;
    # only x can go up to 2^31-1). At n_pairs > 65535 the launch fails with
    # an opaque InvalidConfiguration. Validate host-side with a clear
    # error so callers know to chunk.
    if n_pairs > 65535:
        raise ValueError(f"mi_direct_gpu_batched_pairs: n_pairs={n_pairs} exceeds CUDA " f"gridDim.y limit of 65535; chunk the call host-side.")

    # A5 fix (Critic 1): validate factors_data values are within their
    # declared bin range. Out-of-range values produce a ``merged`` index
    # that overflows the per-pair slice and silently corrupts the next
    # pair's histogram (still inside joint_counts_flat -- no segfault,
    # just wrong MI numbers). The CPU path validates via ``merge_vars``;
    # the GPU path didn't. Cheap host-side max-per-column scan.
    _ref_cols = np.unique(np.concatenate([np.asarray(pairs_a), np.asarray(pairs_b)]))
    for _c in _ref_cols:
        _c_int = int(_c)
        _col = factors_data[:, _c_int]
        _hi = int(_col.max()) if _col.size else -1
        _nb = int(factors_nbins[_c_int])
        if _hi >= _nb:
            raise ValueError(
                f"mi_direct_gpu_batched_pairs: factors_data[:, {_c_int}] "
                f"contains {_hi}, but factors_nbins[{_c_int}]={_nb} "
                f"(values must be in [0, nbins)). The kernel's "
                f"``merged = va + vb * nba`` arithmetic would overflow "
                f"the pair's joint slice."
            )
        if _col.size and int(_col.min()) < 0:
            raise ValueError(f"mi_direct_gpu_batched_pairs: factors_data[:, {_c_int}] " f"contains negative values; factor codes must be >= 0.")

    # Per-pair joint cardinalities ((nbins_a * nbins_b) cells for the MERGED dim) plus prefix-sum offsets. Each pair's joint table has shape (merged_size, nbins_y),
    # flattened row-major.
    nbins_a = factors_nbins[pairs_a].astype(np.int32)
    nbins_b = factors_nbins[pairs_b].astype(np.int32)
    nbins_y = int(np.asarray(freqs_y).shape[0])
    pair_merged_sizes = nbins_a.astype(np.int64) * nbins_b.astype(np.int64)
    pair_joint_sizes = pair_merged_sizes * nbins_y
    # A3 fix: int32 cumsum truncates at total_cells >= 2**31 (the 4 GB guard
    # below was only catching this by 1 bit of headroom). Promote
    # joint_offsets to int64 so the cumsum can never wrap.
    joint_offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    joint_offsets[1:] = np.cumsum(pair_joint_sizes)
    total_cells = int(joint_offsets[-1])

    # GPU memory bound: total_cells int32. At total_cells = 1e7 that's 40 MB; at 1e8 that's 400 MB. Caller should constrain via ``max_combined_nbins`` to stay in bounds.
    # Use >= so a request that lands EXACTLY at the 4 GiB ceiling still raises;
    # the kernel needs a few MB of extra scratch for grid/block metadata, so
    # allowing precisely 4 GiB would still OOM in practice.
    if total_cells * 4 >= 4 * 1024 * 1024 * 1024:  # >= 4 GB
        raise MemoryError(
            f"mi_direct_gpu_batched_pairs: total joint cells {total_cells} "
            f"would require {total_cells*4/2**30:.1f} GB on GPU. Tighten "
            f"max_combined_nbins or run on CPU."
        )

    # RESIDENT UPLOAD (2026-07-13): this function is invoked ONCE per fit (the cat-FE step runs it once
    # before the screening loop -- see ``_fit_impl_core.py``'s "Runs once before the screening loop"
    # comment at its call site), so ``resident_operand``'s cross-call dedup has no repeat-call win to claim
    # for the CURRENT callers; applied anyway (cheap + correct via content-hash, never a perf regression)
    # in case a future/other caller invokes this more than once per fit. ``classes_y``/``nbins_a``/
    # ``joint_offsets`` are genuinely fit-constant-shaped operands; ``factors_data_T`` is rebuilt fresh
    # from ``factors_data.T`` every call (a transient transpose, not a stored fit-constant), so its content
    # generally differs call-to-call and the resident cache will practically never hit for it in this
    # single-call-per-fit codepath -- kept for correctness/consistency, not for a measured win.
    from ._fe_resident_operands import resident_operand
    factors_data_T = np.ascontiguousarray(factors_data.T.astype(np.int32))
    factors_data_T_gpu = resident_operand(factors_data_T, "pairs_factors_data_T", dtype=np.int32)
    classes_y_gpu = resident_operand(np.asarray(classes_y), "pairs_classes_y", dtype=np.int32)
    pairs_a_gpu = cp.asarray(pairs_a.astype(np.int32))
    pairs_b_gpu = cp.asarray(pairs_b.astype(np.int32))
    nbins_a_gpu = resident_operand(nbins_a, "pairs_nbins_a", dtype=np.int32)
    # Host-side joint_offsets is int64 (overflow-safe cumsum); the kernel
    # signature ``const int *joint_offsets`` (int32) is preserved by
    # narrowing here -- safe because the earlier 4 GB total_cells guard
    # enforces ``total_cells < 2**30 < 2**31``, which fits int32 exactly.
    joint_offsets_gpu = resident_operand(joint_offsets, "pairs_joint_offsets", dtype=np.int32)
    joint_counts_flat = cp.zeros(total_cells, dtype=cp.int32)

    # block_size via per-host kernel_tuning_cache (cache miss -> hand-tuned
    # 256, the original default for atomic-heavy multi-pair). cc 8+ devices
    # often prefer 512 here; the auto-tune sweep finds the win.
    block_size = min(GPU_MAX_BLOCK_SIZE, 256)
    # Module-singleton cache (see ._kernel_tuning); a fresh KernelTuningCache()
    # here would re-spawn nvidia-smi per call.
    from ._kernel_tuning import get_kernel_tuning_cache
    _ktc = get_kernel_tuning_cache()
    if _ktc is not None:
        try:
            _entry = _ktc.lookup(
                "joint_hist_multi_pair",
                n_rows=int(n_rows), n_pairs=int(n_pairs),
            )
            if _entry is not None and "block_size" in _entry:
                block_size = int(_entry["block_size"])
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _gpu_pairs.py:147: %s", e)
            pass
    grid_x = (n_rows + block_size - 1) // block_size

    # Kernel-variant dispatch: shared-mem multi-pair kernel wins for small
    # per-pair joint_size_y. Pre-fix the 4096-cell cap was a hardcoded
    # cc 6.x 16 KB assumption; cc 7+ opt-in shared memory reaches 96-227 KB.
    # Wave 23 P1 fix (2026-05-20): probe live device's shared-mem budget
    # via pyutilz.system.gpu_dispatch.get_shared_mem_budget_per_block
    # (the helper that batch_pair_mi_gpu._derive_max_joint_bins already
    # uses); fall back to 4096 only when the probe is unavailable.
    max_joint_size_y = int(pair_joint_sizes.max()) if len(pair_joint_sizes) else 0
    try:
        from pyutilz.system.gpu_dispatch import get_shared_mem_budget_per_block as _shared_budget
        # Reserve 1/8th of the shared budget per pair (matches the
        # _derive_max_joint_bins partitioning in batch_pair_mi_gpu).
        # Each cell is int32 (4 bytes), so capacity in CELLS = budget // 4 // 8.
        _budget_bytes = _shared_budget()
        _SHARED_MULTI_PAIR_MAX = max(4096, _budget_bytes // 4 // 8)
    except Exception:
        # No probe available -> fall back to the pre-2026-05-20 default.
        _SHARED_MULTI_PAIR_MAX = 4096
    use_shared_multi_pair = max_joint_size_y > 0 and max_joint_size_y <= _SHARED_MULTI_PAIR_MAX

    # SINGLE kernel launch processes all pairs via 3D grid (rows along X, pairs along Y); per-pair launch overhead amortised to zero.
    if use_shared_multi_pair:
        _gpu_module.compute_joint_hist_multi_pair_shared_cuda(
            (grid_x, n_pairs),
            (block_size,),
            (
                factors_data_T_gpu, classes_y_gpu, pairs_a_gpu, pairs_b_gpu,
                nbins_a_gpu, joint_offsets_gpu,
                joint_counts_flat, n_rows, n_pairs, nbins_y,
                np.int32(max_joint_size_y),
            ),
            shared_mem=max_joint_size_y * 4,
        )
    else:
        _gpu_module.compute_joint_hist_multi_pair_cuda(
            (grid_x, n_pairs),
            (block_size,),
            (
                factors_data_T_gpu, classes_y_gpu, pairs_a_gpu, pairs_b_gpu,
                nbins_a_gpu, joint_offsets_gpu,
                joint_counts_flat, n_rows, n_pairs, nbins_y,
            ),
        )
    joint_counts_host = cp.asnumpy(joint_counts_flat)

    # Per-pair MI from joint counts (numpy). joint shape = (merged_size, nbins_y); marg_m = joint.sum(axis=1); marg_y = joint.sum(axis=0); total = n_rows.
    # MI = sum over non-zero cells of (jc/n) * log(jc * n / (marg_m * marg_y)), in nats.
    n_total = float(n_rows)
    joint_mi_out = np.zeros(n_pairs, dtype=np.float64)
    for k in range(n_pairs):
        off = int(joint_offsets[k])
        merged_size = int(pair_merged_sizes[k])
        joint_2d = joint_counts_host[off : off + merged_size * nbins_y].reshape(merged_size, nbins_y)
        marg_m = joint_2d.sum(axis=1)
        marg_y = joint_2d.sum(axis=0)
        # MI in nats
        mi = 0.0
        for m in range(merged_size):
            mm = marg_m[m]
            if mm == 0:
                continue
            for y in range(nbins_y):
                jc = joint_2d[m, y]
                if jc == 0:
                    continue
                my = marg_y[y]
                if my == 0:
                    continue
                jf = jc / n_total
                mi += jf * np.log(jc * n_total / (mm * my))
        joint_mi_out[k] = mi
    return joint_mi_out

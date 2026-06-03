"""Symmetric-Uncertainty pairwise clustering for ShapProxiedFS.

Sibling module of ``_shap_proxy_cluster.py`` (Pearson-based clustering). Reuses
the pre-binned per-column class arrays produced by MRMR's ``categorize_dataset``
(surfaced via ``MRMR.export_artifacts()['bins']``) so two features are linked
into the same cluster when ``SU(X_i, X_j) >= threshold``. Catches non-linear
redundancy (XOR, saddle, sinusoidal) that the Pearson backend misses because
``|corr|`` is near zero on those relationships even when full mutual
information is high.

The return contract mirrors ``cluster_correlated_features``: an ``np.ndarray``
of cluster labels with shape ``(n_features,)`` indexed 0..K-1 contiguous, so
the downstream ``build_unit_matrix`` / ``cluster_summary`` callers stay
unchanged.

Reuses ``compute_su_from_classes(classes_x, freqs_x, classes_y, freqs_y)`` from
``filters.info_theory`` (numba-cached) — same SU primitive the MRMR screen and
DCD pairwise SU branch run. The marginal entropies (and bincount frequencies)
are cached per column inside one ``cluster_correlated_features_su`` call so
each column's marginal pass runs once even though the column appears in
``n_features - 1`` pairs.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable

import numpy as np
from numba import njit, prange

from mlframe.feature_selection._shap_proxy_cluster import _uf_labels

logger = logging.getLogger(__name__)


def _resolve_parallel_min_features(default: int = 50) -> int:
    """Smallest feature count at which the parallel prange kernel beats the serial path.

    Below this width the per-pair work is small enough that prange thread-spawn dwarfs
    the saved CPU time. Above it, the O(f^2) pair count scales the wall and parallel
    pays off. The default (50) is dispatcher-tunable per HW via
    ``pyutilz.system.kernel_tuning_cache`` (key
    ``mlframe.shap_proxied_fs.cluster_su.parallel_min_features``).
    """
    try:
        from pyutilz.system import kernel_tuning_cache

        value = kernel_tuning_cache.get(
            "mlframe.shap_proxied_fs.cluster_su.parallel_min_features", default=default)
        return int(value)
    except Exception:
        return default


def _resolve_gpu_min_features(default: int = 500) -> int:
    """Smallest feature count at which the GPU pairwise SU path is preferred over the CPU prange kernel.

    Below this width the cupy/CUDA launch overhead + onehot-pack allocation dwarfs even the
    parallel CPU kernel's wall (~0.14s at f=500 / n_bins=10 / n=1500 on iter69's bench).
    Dispatcher-tunable per HW via ``pyutilz.system.kernel_tuning_cache`` (key
    ``mlframe.shap_proxied_fs.cluster_su.gpu_min_features``); default 500.
    """
    try:
        from pyutilz.system import kernel_tuning_cache

        value = kernel_tuning_cache.get(
            "mlframe.shap_proxied_fs.cluster_su.gpu_min_features", default=default)
        return int(value)
    except Exception:
        return default


_GPU_AVAILABLE_CACHE: bool | None = None


def cluster_su_gpu_available() -> bool:
    """Process-cached probe for a cupy CUDA device. Lazy import; never raises.

    Returns True iff ``cupy`` imports cleanly AND ``cp.cuda.runtime.getDeviceCount() > 0``
    AND a tiny float32 allocation round-trips. The result is cached for the process
    lifetime: probing is cheap on the hit path (single dict lookup) but the first call
    pays the cupy import + nvrtc init, which we don't want to repeat across pair-loop
    calls.
    """
    global _GPU_AVAILABLE_CACHE
    if _GPU_AVAILABLE_CACHE is not None:
        return _GPU_AVAILABLE_CACHE
    try:
        import cupy as cp  # noqa: F401

        if cp.cuda.runtime.getDeviceCount() <= 0:
            _GPU_AVAILABLE_CACHE = False
            return False
        # Round-trip a tiny allocation so a half-installed cupy (DLL load OK but device
        # alloc broken) fails the gate here, not deep inside the kernel.
        probe = cp.zeros(4, dtype=cp.float32)
        _ = probe.sum().get()
        _GPU_AVAILABLE_CACHE = True
        return True
    except Exception:
        _GPU_AVAILABLE_CACHE = False
        return False


def _gpu_free_memory_bytes() -> int:
    """Best-effort free GPU memory probe; 0 on failure (caller treats as 'cannot fit')."""
    try:
        import cupy as cp

        free, _total = cp.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        return 0


def _should_route_su_gpu(
    n_features: int,
    n_samples: int,
    max_n_bins: int,
    *,
    gpu_min_features: int | None = None,
    memory_safety_factor: float = 0.5,
) -> bool:
    """Decide whether to route the pairwise SU scan to the GPU kernel.

    Three gates, all must pass:
      1. cupy + CUDA device available (``cluster_su_gpu_available()``).
      2. ``n_features >= gpu_min_features`` (kernel_tuning_cache-tunable; default 500).
      3. One-hot working set ``n_features * max_n_bins * n_samples * 4`` bytes fits in
         ``memory_safety_factor`` of free GPU memory. The 0.5 default leaves headroom for
         the joint-matrix tensor and cuBLAS scratch.
    """
    if not cluster_su_gpu_available():
        return False
    gmin = gpu_min_features if gpu_min_features is not None else _resolve_gpu_min_features()
    if n_features < int(gmin):
        return False
    onehot_bytes = int(n_features) * int(max_n_bins) * int(n_samples) * 4
    # 2026-06-03 (audit shap-proxy-clustering-3): the gate previously sized ONLY
    # the float32 one-hot and ignored the DOMINANT inner working set -- the
    # einsum joint tensor (chunk, f, mb, mb) plus its float64 siblings (joint_p,
    # px_outer, safe_outer, ratio, log_ratio). With the old hardcoded chunk=4096
    # (>= f) that tensor was the FULL f*f*mb^2*8 bytes (~19 GB at f=2000/mb=10)
    # while the gate only checked ~120 MB -> guaranteed OOM. The kernel now
    # auto-shrinks the chunk to fit, but if even a SINGLE i-row's working set
    # (~10 * f * mb^2 * 8 bytes, budgeting the float32 counts + ~5 float64
    # temporaries) will not fit, we must route to the CPU kernel instead.
    joint_row_bytes = 10 * int(n_features) * int(max_n_bins) * int(max_n_bins) * 8
    free_bytes = _gpu_free_memory_bytes()
    if free_bytes <= 0:
        return False
    return (onehot_bytes + joint_row_bytes) < int(free_bytes * float(memory_safety_factor))


def _pairwise_su_edges_gpu(
    bins_packed: np.ndarray,
    nbins: np.ndarray,
    h_marginals: np.ndarray,
    constant_mask: np.ndarray,
    threshold: float,
    pair_chunk_size: int = 4096,
) -> np.ndarray:
    """GPU pairwise SU edge scan via batched one-hot @ one-hot.T joints.

    Input has the same column-major layout as ``_pairwise_su_edges`` so the CPU and GPU
    paths share ``_pack_bins_for_kernel``. The algorithm:

      1. Build a padded one-hot tensor ``X_onehot`` of shape ``(n_features, max_nb, n_samples)``
         float32 on GPU. Padded rows (bin id >= ``nbins[i]``) stay zero so they contribute
         nothing to joints; constant columns are masked out before edge extraction.
      2. For each chunk of ``i`` rows compute the joint matrix against ALL ``j > i`` via
         ``einsum('ias,jbs->ijab', X_onehot[i_chunk], X_onehot)`` -> shape ``(chunk, f, mb, mb)``.
         The on-device working tensor is ``chunk * f * mb^2 * 8 bytes`` (float64 for
         numerically stable log) which we keep under the ``pair_chunk_size`` cap.
      3. Compute SU per pair entry-wise: ``mi = sum_{ab} joint_ab * log(joint_ab /
         (px_a * py_b))`` (skipping zero entries), ``su = 2*mi / (h_i + h_j)``.
      4. Return ``flags`` of shape ``(n_features, n_features)`` uint8 upper-triangle, matching
         the CPU kernel's contract so the caller's ``np.where`` extraction is unchanged.
    """
    import cupy as cp

    n_features, n_samples = bins_packed.shape
    max_nb = int(nbins.max()) if nbins.size else 0
    inv_n = cp.float64(1.0 / n_samples) if n_samples > 0 else cp.float64(0.0)

    bins_dev = cp.asarray(bins_packed, dtype=cp.int32)
    nbins_dev = cp.asarray(nbins, dtype=cp.int64)
    h_dev = cp.asarray(h_marginals, dtype=cp.float64)
    constant_dev = cp.asarray(constant_mask, dtype=cp.bool_)

    # One-hot tensor: positions (f, b, s) = 1 iff bins_packed[f, s] == b. Padded rows
    # (b >= nbins[f]) stay zero - cheaper than tracking nb_i per pair on GPU.
    onehot = cp.zeros((n_features, max_nb, n_samples), dtype=cp.float32)
    bin_axis = cp.arange(max_nb, dtype=cp.int32).reshape(1, max_nb, 1)
    onehot[:] = (bins_dev[:, None, :] == bin_axis).astype(cp.float32)

    # Marginal probability vector per feature from the one-hot sums (axis=samples).
    # Shape: (n_features, max_nb) float64. Padded bins always sum to 0 -> px=0 -> skipped.
    px_dev = onehot.sum(axis=2).astype(cp.float64) * inv_n

    flags = cp.zeros((n_features, n_features), dtype=cp.uint8)

    # 2026-06-03 (audit shap-proxy-clustering-3): auto-shrink the i-chunk so the
    # inner (chunk, f, mb, mb) working set fits in free GPU memory. Budget
    # ~10 * f * mb^2 * 8 bytes per i-row (float32 joint_counts + the float64
    # joint_p/px_outer/safe_outer/ratio/log_ratio siblings). The prior hardcoded
    # 4096 (>= f) allocated the FULL f x f x mb x mb tensor at once -> OOM on any
    # non-trivial width (~19 GB at f=2000/mb=10 on a 4 GB card). onehot is
    # already resident, so budget the remaining free memory for the joints.
    _free = _gpu_free_memory_bytes()
    _onehot_bytes = n_features * max_nb * n_samples * 4
    _bytes_per_row = max(1, 10 * n_features * max_nb * max_nb * 8)
    _budget = max(0, int(_free * 0.4) - _onehot_bytes)
    _auto_chunk = max(1, _budget // _bytes_per_row)
    pair_chunk_size = max(1, min(int(pair_chunk_size), int(_auto_chunk), n_features))

    # Chunked i-loop so the einsum's working tensor stays bounded. Each chunk computes
    # the SU row for all j > i_start.
    for i_start in range(0, n_features, pair_chunk_size):
        i_end = min(i_start + pair_chunk_size, n_features)
        chunk_size = i_end - i_start

        # joint[i_local, j, a, b] = sum_s onehot[i_global, a, s] * onehot[j, b, s]
        # 2026-06-03 (audit shap-proxy-clustering-3): einsum in float32. The joint
        # COUNTS are integer sums of 0/1 products over <= n_samples (<< 2^24),
        # exactly representable in float32, so we avoid the prior full float64
        # copy of the one-hot (onehot.astype(cp.float64) was an unbudgeted
        # f*mb*n*8 allocation). Cast the integer counts up to float64 for the log.
        joint_counts = cp.einsum(
            "ias,jbs->ijab",
            onehot[i_start:i_end],
            onehot,
        )
        joint_p = joint_counts.astype(cp.float64) * inv_n  # (chunk, f, mb, mb)

        # mi = sum_{a,b} joint_p * log(joint_p / (px_i_a * px_j_b)) over a, b
        # We need px_outer[i_local, j, a, b] = px[i_global, a] * px[j, b].
        px_i = px_dev[i_start:i_end][:, None, :, None]  # (chunk, 1, mb, 1)
        px_j = px_dev[None, :, None, :]                  # (1, f, 1, mb)
        px_outer = px_i * px_j                            # (chunk, f, mb, mb)

        # log-ratio only where joint_p > 0 AND px_outer > 0; zero elsewhere
        eps = cp.float64(1e-300)
        safe_outer = cp.where(px_outer > 0.0, px_outer, eps)
        ratio = cp.where(joint_p > 0.0, joint_p / safe_outer, cp.float64(1.0))
        log_ratio = cp.where(joint_p > 0.0, cp.log(ratio), cp.float64(0.0))
        mi = (joint_p * log_ratio).sum(axis=(2, 3))  # (chunk, f)

        h_i_chunk = h_dev[i_start:i_end][:, None]       # (chunk, 1)
        denom = h_i_chunk + h_dev[None, :]               # (chunk, f)
        # denom == 0 happens iff BOTH columns have entropy 0 (both constant). For those
        # pairs ``constant_mask`` already excludes them below.
        safe_denom = cp.where(denom > 1e-12, denom, cp.float64(1.0))
        su = cp.where(denom > 1e-12, 2.0 * mi / safe_denom, cp.float64(0.0))

        # Mask: upper triangle only AND not constant on either side.
        i_idx = cp.arange(i_start, i_end)[:, None]       # (chunk, 1)
        j_idx = cp.arange(n_features)[None, :]            # (1, f)
        upper = j_idx > i_idx                             # (chunk, f)
        not_const = (~constant_dev[i_start:i_end])[:, None] & (~constant_dev[None, :])

        passes = (su >= cp.float64(threshold)) & upper & not_const
        flags[i_start:i_end] = passes.astype(cp.uint8)

        # Free chunk temporaries before next chunk to keep peak GPU memory bounded.
        del joint_counts, joint_p, px_outer, safe_outer, ratio, log_ratio, mi, denom
        del safe_denom, su, upper, not_const, passes

    return cp.asnumpy(flags)


@njit(parallel=True, nogil=True, cache=True, fastmath=False)
def _pairwise_su_edges(
    bins_packed: np.ndarray,
    nbins: np.ndarray,
    freqs_packed: np.ndarray,
    freqs_offsets: np.ndarray,
    h_marginals: np.ndarray,
    constant_mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Pairwise SU matrix above ``threshold`` returned as a dense flag matrix.

    ``bins_packed`` is a ``(n_features, n_samples)`` int32 view (column-major
    layout: each feature's per-sample bin ids occupy a contiguous row, so the
    inner sample-scan reads two contiguous int32 strips and saturates the L1
    cache line instead of jumping ``n_features * 4`` bytes per sample);
    ``nbins[i]`` is the cardinality used to size the joint-counts matrix on
    column ``i``; ``freqs_packed`` is the concatenation of all per-column
    marginal probability vectors with offsets in ``freqs_offsets`` (shape
    ``(n_features + 1,)``); ``h_marginals[i]`` is the pre-computed Shannon
    entropy of column ``i``; ``constant_mask[i]`` is ``True`` when column ``i``
    has <=1 distinct bin (SU=0 vs anyone).

    Returns an upper-triangle flag matrix (``flag[i, j] = 1`` iff ``SU(i, j) >= threshold``
    and ``i < j``). Edge extraction happens outside the njit kernel so the prange
    iterations stay purely numeric.
    """
    n_features, n_samples = bins_packed.shape
    flags = np.zeros((n_features, n_features), dtype=np.uint8)
    # max joint cardinality controls a single thread-local reusable buffer per
    # outer iteration; avoids per-pair np.zeros allocation that bottlenecks at
    # width >= 2000 (numba memory allocator under contention).
    #
    # bench-attempt-rejected (2026-05-31, iter72): j-tile block of B consecutive
    # j-columns sharing the i-row L1 read in the sample sweep. Tested B in
    # {2,3,4,5,7,8,16,32} at width=2000 / n_samples=1500 / n_bins=10 / 8 numba
    # threads: every B regressed (best B=8 was 2.59s vs per-pair 2.13s = 0.82x).
    # Likely the B strided writes to joints[jb, x_i, x_j] saturate L1 stores and
    # the extra zeroing work (B * nb_i * nb_j cells per tile) wipes any savings
    # from sharing the i-row read. Do not re-attempt pure j-tiling without first
    # changing the inner-loop store pattern (e.g. SoA joints[B, k_chunk] +
    # post-reduction, or sample-tile + j-tile two-level blocking).
    max_nb = 0
    for i in range(n_features):
        if nbins[i] > max_nb:
            max_nb = nbins[i]
    for i in prange(n_features):
        if constant_mask[i]:
            continue
        nb_i = nbins[i]
        h_i = h_marginals[i]
        off_i = freqs_offsets[i]
        # one int64 buffer per outer-i (thread-local because prange allocates
        # locals inside the parallel region on the worker thread's stack).
        joint = np.zeros((max_nb, max_nb), dtype=np.int64)
        for j in range(i + 1, n_features):
            if constant_mask[j]:
                continue
            nb_j = nbins[j]
            # reset only the cells we'll touch.
            for a in range(nb_i):
                for b in range(nb_j):
                    joint[a, b] = 0
            for k in range(n_samples):
                joint[bins_packed[i, k], bins_packed[j, k]] += 1
            inv_n = 1.0 / n_samples
            mi = 0.0
            off_j = freqs_offsets[j]
            for a in range(nb_i):
                px = freqs_packed[off_i + a]
                if px <= 0.0:
                    continue
                for b in range(nb_j):
                    jc = joint[a, b]
                    if jc == 0:
                        continue
                    py = freqs_packed[off_j + b]
                    if py <= 0.0:
                        continue
                    jf = jc * inv_n
                    mi += jf * math.log(jf / (px * py))
            denom = h_i + h_marginals[j]
            if denom <= 1e-12:
                continue
            su = 2.0 * mi / denom
            if su >= threshold:
                flags[i, j] = 1
    return flags


@njit(parallel=True, nogil=True, cache=True, fastmath=False)
def _compute_marginals_packed(
    bins_packed: np.ndarray,
    nbins: np.ndarray,
    freqs_offsets: np.ndarray,
    freqs_packed: np.ndarray,
    h_marginals: np.ndarray,
    constant_mask: np.ndarray,
) -> None:
    """Fill freqs_packed/h_marginals/constant_mask from bins_packed in one parallel sweep.

    For each feature i (in prange): bincount its row of ``bins_packed`` into the
    contiguous slice ``freqs_packed[freqs_offsets[i] : freqs_offsets[i + 1]]``, normalize
    in place to a probability vector, compute Shannon entropy in the same pass, and
    set ``constant_mask[i]`` from non-zero-bin cardinality. Replaces the prior
    two-pass Python loop (``_column_marginal`` per column followed by ``_pack_bins_for_kernel``'s
    per-column entropy Python sum) with a single parallel kernel that touches each
    column's data exactly once.
    """
    n_features, n_samples = bins_packed.shape
    inv_n = 1.0 / n_samples if n_samples > 0 else 0.0
    for i in prange(n_features):
        nb_i = nbins[i]
        off_i = freqs_offsets[i]
        # Zero the slice we're about to fill (np.empty allocation upstream).
        for b in range(nb_i):
            freqs_packed[off_i + b] = 0.0
        # Bincount the row.
        for k in range(n_samples):
            freqs_packed[off_i + bins_packed[i, k]] += 1.0
        # Normalize + compute entropy in one pass; count non-zero bins for constancy.
        nonzero_bins = 0
        h = 0.0
        for b in range(nb_i):
            c = freqs_packed[off_i + b]
            if c > 0.0:
                p = c * inv_n
                freqs_packed[off_i + b] = p
                h -= p * math.log(p)
                nonzero_bins += 1
        h_marginals[i] = h
        # constant column: only one bin id ever appears (regardless of how many
        # padded bin slots the cardinality hint allocates).
        constant_mask[i] = nonzero_bins <= 1


def _setup_su_kernel_inputs(
    arrays: list[np.ndarray],
    nbins_hints: list[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Fused replacement for ``_column_marginal`` + ``_pack_bins_for_kernel``.

    Single Python pass over ``arrays``: determines per-feature ``nb_i`` (max of observed
    bin id + 1 and any user hint), writes the int32 column-major ``bins_packed`` buffer,
    and lays out the ``freqs_offsets`` table. Then a single ``@njit(parallel=True)`` sweep
    (``_compute_marginals_packed``) fills the marginal probability vectors + Shannon
    entropies in one parallel pass.

    Returns ``None`` when columns have heterogeneous sample counts (the dict layout that
    MRMR's ``categorize_dataset`` produces always shares ``n_samples`` so this is just a
    defensive guard).
    """
    if not arrays:
        return None
    n_samples = int(arrays[0].shape[0])
    for a in arrays[1:]:
        if int(a.shape[0]) != n_samples:
            return None
    n_features = len(arrays)

    # Pre-determine nb_i per feature (one .max() per column - O(n_samples) total per
    # column; cheap C call, no Python-loop overhead per element). The hint overrides
    # the observed max so the joint-counts matrix dimension stays consistent with
    # MRMR's view of the bin space even when a column never realises its highest bin.
    nbins = np.empty(n_features, dtype=np.int64)
    freqs_offsets = np.empty(n_features + 1, dtype=np.int64)
    offset = 0
    for i, arr in enumerate(arrays):
        if arr.size == 0:
            observed_max = 0
        else:
            observed_max = int(arr.max()) + 1
        hint = nbins_hints[i] if nbins_hints is not None else 0
        nb = max(observed_max, int(hint) if hint else 0)
        nbins[i] = nb
        freqs_offsets[i] = offset
        offset += nb
    freqs_offsets[n_features] = offset

    bins_packed = np.empty((n_features, n_samples), dtype=np.int32, order="C")
    freqs_packed = np.empty(int(offset), dtype=np.float64)
    h_marginals = np.empty(n_features, dtype=np.float64)
    constant_mask = np.empty(n_features, dtype=np.bool_)

    # Column-major write: each feature's bin ids occupy a contiguous int32 row so the
    # SU kernel's inner sample loop reads stride-1 strips. The astype(copy=False) is a
    # no-op view when arr is already int32, otherwise a single C-level copy.
    for i, arr in enumerate(arrays):
        bins_packed[i, :] = arr.astype(np.int32, copy=False)

    # Parallel sweep: bincount + normalize + entropy + constant-mask, all in one
    # pass per column on a worker thread. Replaces the prior pair of Python loops
    # (np.bincount in _column_marginal + Python entropy sum in _pack_bins_for_kernel).
    _compute_marginals_packed(
        bins_packed, nbins, freqs_offsets, freqs_packed, h_marginals, constant_mask,
    )

    return bins_packed, nbins, freqs_packed, freqs_offsets, h_marginals, constant_mask


def _pack_bins_for_kernel(
    arrays: list[np.ndarray],
    marginals: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Pack per-column bin arrays + marginals into the contiguous buffers the kernel reads.

    Returns ``None`` when columns have heterogeneous sample counts (the dict layout
    that MRMR's ``categorize_dataset`` produces always shares ``n_samples`` so this
    is just a defensive guard). Each returned array is C-contiguous and dtyped to
    match the kernel's signature exactly so numba doesn't re-specialize per call.
    """
    if not arrays:
        return None
    n_samples = int(arrays[0].shape[0])
    for a in arrays[1:]:
        if int(a.shape[0]) != n_samples:
            return None
    n_features = len(arrays)
    # Column-major layout: feature i's per-sample bin ids live in row i so the
    # kernel's inner sample loop reads two contiguous int32 strips
    # (bins_packed[i, :] and bins_packed[j, :]) — one cache line load per ~16
    # samples instead of one per sample under the prior (n_samples, n_features)
    # row-major layout where the i and j columns lived on different cache lines.
    bins_packed = np.empty((n_features, n_samples), dtype=np.int32, order="C")
    nbins = np.empty(n_features, dtype=np.int64)
    freqs_offsets = np.empty(n_features + 1, dtype=np.int64)
    h_marginals = np.empty(n_features, dtype=np.float64)
    constant_mask = np.empty(n_features, dtype=np.bool_)

    total_freqs = 0
    for i, (_classes_unused, freqs) in enumerate(marginals):
        total_freqs += int(freqs.shape[0])
    freqs_packed = np.empty(total_freqs, dtype=np.float64)
    offset = 0
    for i, (classes_i, freqs_i) in enumerate(marginals):
        # classes_i is int64 by construction in _column_marginal; downcast to the
        # int32 packed dtype (bin ids are tiny, never overflow int32). Writing
        # the whole row at once stays sequential in the destination buffer.
        bins_packed[i, :] = classes_i.astype(np.int32, copy=False)
        nb = int(freqs_i.shape[0])
        nbins[i] = nb
        freqs_offsets[i] = offset
        freqs_packed[offset:offset + nb] = freqs_i
        offset += nb
        constant_mask[i] = nb <= 1
        h = 0.0
        for p in freqs_i:
            if p > 0.0:
                h -= float(p) * math.log(float(p))
        h_marginals[i] = h
    freqs_offsets[n_features] = offset
    return bins_packed, nbins, freqs_packed, freqs_offsets, h_marginals, constant_mask


def _resolve_columns(
    bins: dict[str, np.ndarray],
    feature_names: Iterable[str] | None,
) -> tuple[list[str], list[np.ndarray]]:
    """Materialise an ordered (names, bin-arrays) view of the bins dict.

    ``feature_names`` pins the iteration order so the returned labels array is
    axis-aligned with the caller's ``X_search.columns``. ``None`` falls back to
    ``bins.keys()`` order (insertion order; only safe when the caller has not
    re-ordered the columns since artifact export).
    """
    if feature_names is None:
        names = list(bins.keys())
    else:
        names = [n for n in feature_names if n in bins]
    arrays = [np.ascontiguousarray(bins[n]) for n in names]
    return names, arrays


def _column_marginal(
    classes: np.ndarray,
    n_bins_hint: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(classes_int64, freqs_float64)`` ready for ``compute_su_from_classes``.

    The frequency array is a probability vector (sums to 1 if ``classes.size > 0``);
    its length matches the maximum observed bin id + 1, or ``n_bins_hint`` if that
    is larger (keeps the joint-counts matrix dimension matching across pairs even
    when one column happens to never realise its highest bin in this sample).
    """
    cls = np.ascontiguousarray(classes, dtype=np.int64)
    if cls.size == 0:
        return cls, np.empty(0, dtype=np.float64)
    observed_max = int(cls.max()) + 1 if cls.size else 0
    nb = max(observed_max, int(n_bins_hint) if n_bins_hint else 0)
    counts = np.bincount(cls, minlength=nb).astype(np.float64)
    total = counts.sum()
    if total <= 0.0:
        return cls, counts
    return cls, counts / total


def _run_cpu_pairwise_su(
    bins_packed: np.ndarray,
    nbins: np.ndarray,
    freqs_packed: np.ndarray,
    freqs_offsets: np.ndarray,
    h_marginals: np.ndarray,
    constant_mask: np.ndarray,
    threshold: float,
    *,
    use_bitmap: bool,
    bitmap_min_features: int | None,
    bitmap_max_n_bins: int | None,
) -> np.ndarray:
    """Pick the CPU pairwise SU kernel: popcount-bitmap when its gates pass, else scalar.

    Both kernels return identical ``flags`` (uint8 upper-triangle), so the caller
    treats them interchangeably. The bitmap kernel wins at moderate width with
    small ``n_bins`` because each per-pair sample sweep collapses to
    ``n_bins^2 * ceil(n_samples/64)`` POPCNT-AND operations; the scalar kernel
    stays the fallback for wide bin spaces or tiny widths where the bitmap pack
    overhead dominates.
    """
    if use_bitmap:
        from mlframe.feature_selection._shap_proxy_cluster_su_bitmap import (
            pairwise_su_edges_bitmap,
            should_route_bitmap,
        )

        n_features, n_samples = bins_packed.shape
        max_nb = int(nbins.max()) if nbins.size else 0
        if should_route_bitmap(
            n_features=n_features,
            n_samples=n_samples,
            max_n_bins=max_nb,
            bitmap_min_features=bitmap_min_features,
            bitmap_max_n_bins=bitmap_max_n_bins,
        ):
            try:
                return pairwise_su_edges_bitmap(
                    bins_packed, nbins, freqs_packed, freqs_offsets,
                    h_marginals, constant_mask, threshold,
                )
            except Exception as exc:
                logger.warning(
                    "Bitmap SU kernel failed (%s); falling back to scalar prange kernel", exc,
                )
    return _pairwise_su_edges(
        bins_packed, nbins, freqs_packed, freqs_offsets,
        h_marginals, constant_mask, threshold,
    )


def cluster_correlated_features_su(
    bins: dict[str, np.ndarray],
    *,
    threshold: float = 0.5,
    feature_names: Iterable[str] | None = None,
    nbins_per_feature: dict[str, int] | None = None,
    edge_cap: int = 20_000_000,
    use_parallel: bool = True,
    parallel_min_features: int | None = None,
    use_gpu: str | bool = "auto",
    gpu_min_features: int | None = None,
    gpu_pair_chunk_size: int = 4096,
    use_bitmap: bool = True,
    bitmap_min_features: int | None = None,
    bitmap_max_n_bins: int | None = None,
) -> np.ndarray:
    """Cluster features by single-linkage on ``SU(X_i, X_j) >= threshold``.

    Mirrors ``_shap_proxy_cluster.cluster_correlated_features``'s return type
    (``np.ndarray`` shape ``(n_features,)`` dtype int64, contiguous 0..K-1
    cluster ids) so the downstream ``build_unit_matrix`` consumer is reused
    verbatim.

    Parameters
    ----------
    bins
        ``feature_name -> per-row int bin labels`` (e.g.
        ``MRMR.export_artifacts()['bins']`` after ``restrict_artifacts``).
    threshold
        SU cutoff in [0, 1]. Pairs with ``SU >= threshold`` are linked. The
        scale differs from Pearson ``|corr|``: SU is bounded by 1 but reaches
        it only for deterministic relationships; mid-strong dependencies
        cluster around SU ~0.4-0.6. The default 0.5 is calibrated to roughly
        match the linking density of Pearson at ``|corr| >= 0.7``.
    feature_names
        Ordering pin. When provided, the returned labels array indexes against
        this ordering (typically ``X_search.columns``). ``None`` falls back to
        ``bins.keys()``.
    nbins_per_feature
        Optional ``feature_name -> bin count`` hint. When supplied, the
        marginal frequency arrays use this length even if the column never
        realises its highest bin in the sample, keeping shapes consistent
        with the MRMR screen's view.
    edge_cap
        Reject the clustering if more than this many above-threshold edges
        are produced. Mirrors the Pearson backend's safeguard against
        runaway pair density. Raises ``MemoryError`` if exceeded.
    use_parallel
        Route the O(f^2) pair scan through the numba prange kernel
        ``_pairwise_su_edges`` when ``n_features >= parallel_min_features``.
        Default ``True``. The serial path is kept as the fallback (and the
        chosen path at small widths where prange thread-spawn dominates).
    parallel_min_features
        Smallest ``n_features`` at which the parallel kernel is selected.
        ``None`` consults ``pyutilz.system.kernel_tuning_cache`` (key
        ``mlframe.shap_proxied_fs.cluster_su.parallel_min_features``);
        default 50.
    use_gpu
        ``"auto"`` (default), ``True``, or ``False``. When ``"auto"`` the GPU
        path runs iff cupy + a CUDA device are present AND ``n_features``
        exceeds ``gpu_min_features`` AND the one-hot working set fits in 50%
        of free GPU memory. ``True`` forces the GPU path (still falls back
        when cupy fails to import). ``False`` keeps the CPU prange kernel.
    gpu_min_features
        Smallest ``n_features`` at which the GPU pairwise-SU kernel beats the
        CPU prange kernel (cupy launch overhead amortises only above this
        width). ``None`` consults ``pyutilz.system.kernel_tuning_cache`` (key
        ``mlframe.shap_proxied_fs.cluster_su.gpu_min_features``); default 500.
    gpu_pair_chunk_size
        Maximum number of i-rows processed per GPU einsum batch; bounds peak
        GPU memory at ``chunk * n_features * max_nb^2 * 8`` bytes. Default 4096.
    use_bitmap
        Enable the popcount-bitmap pairwise kernel
        (``_shap_proxy_cluster_su_bitmap.pairwise_su_edges_bitmap``) when
        ``n_features >= bitmap_min_features`` AND ``max(n_bins) <= bitmap_max_n_bins``
        AND ``n_samples`` exceeds the bitmap-amortization floor AND the bitmap
        memory fits the 256 MB cap. The bitmap path collapses the per-pair
        sample sweep to ``n_bins^2 * ceil(n_samples/64)`` POPCNT-AND ops with
        hardware ``POPCNT``; default ``True``. Disable to force the scalar
        ``_pairwise_su_edges`` kernel (legacy / debug). Routed under the CPU
        path only - GPU path takes priority when both apply.
    bitmap_min_features
        Smallest ``n_features`` at which the bitmap kernel beats the scalar
        prange kernel. ``None`` consults ``pyutilz.system.kernel_tuning_cache``
        (key ``mlframe.shap_proxied_fs.cluster_su.bitmap_min_features``);
        default 200.
    bitmap_max_n_bins
        Upper ``max(n_bins)`` at which the bitmap kernel still wins. Above this
        the ``n_bins^2`` per-pair work overwhelms the 64-wide POPCNT speedup
        and the scalar kernel is faster. ``None`` consults
        ``pyutilz.system.kernel_tuning_cache``
        (key ``mlframe.shap_proxied_fs.cluster_su.bitmap_max_n_bins``);
        default 16.

    Returns
    -------
    labels : np.ndarray
        ``(n_features,)`` int64. Constant columns and features with no
        above-threshold partner become singleton clusters.
    """
    from mlframe.feature_selection.filters.info_theory import compute_su_from_classes

    if not isinstance(bins, dict):
        raise TypeError(
            f"cluster_correlated_features_su: expected bins dict, got {type(bins).__name__}"
        )
    names, arrays = _resolve_columns(bins, feature_names)
    f = len(names)
    if f == 0:
        return np.empty(0, dtype=np.int64)

    threshold = float(threshold)
    ei_parts: list[int] = []
    ej_parts: list[int] = []
    total = 0

    # Parallel path: when we have enough features for prange overhead to pay back,
    # build packed buffers once and let the numba kernel run the O(f^2) loop with
    # thread-local joint-count buffers. Falls back to the serial loop on small f
    # or when packing isn't safe (heterogeneous n_samples per column).
    pmin = parallel_min_features if parallel_min_features is not None else _resolve_parallel_min_features()
    use_kernel = bool(use_parallel) and f >= int(pmin)
    if use_kernel:
        # Fused builder: one Python pass over arrays + one njit(parallel=True) sweep
        # for bincount/normalize/entropy/constant-mask. Replaces the prior two-pass
        # _column_marginal + _pack_bins_for_kernel chain (both Python-level O(f) loops).
        nbins_hints: list[int] | None
        if nbins_per_feature is not None:
            nbins_hints = [
                int(nbins_per_feature[name]) if name in nbins_per_feature else 0
                for name in names
            ]
        else:
            nbins_hints = None
        packed = _setup_su_kernel_inputs(arrays, nbins_hints)
        if packed is not None:
            bins_packed, nbins, freqs_packed, freqs_offsets, h_marginals, constant_mask = packed

            # GPU dispatch: ahead of the prange kernel because the GPU path wins at
            # f >= gpu_min_features. Falls back transparently if cupy/CUDA misbehave.
            gpu_route = False
            if use_gpu in ("auto", True):
                n_samples_kernel = int(bins_packed.shape[1])
                max_nb = int(nbins.max()) if nbins.size else 0
                if use_gpu is True:
                    # forced path: only the availability + memory gates apply (no width gate)
                    gpu_route = _should_route_su_gpu(
                        n_features=f, n_samples=n_samples_kernel, max_n_bins=max_nb,
                        gpu_min_features=0,
                    )
                else:
                    gpu_route = _should_route_su_gpu(
                        n_features=f, n_samples=n_samples_kernel, max_n_bins=max_nb,
                        gpu_min_features=gpu_min_features,
                    )
            if gpu_route:
                try:
                    flags = _pairwise_su_edges_gpu(
                        bins_packed, nbins, h_marginals, constant_mask, threshold,
                        pair_chunk_size=int(gpu_pair_chunk_size),
                    )
                except Exception as exc:
                    logger.warning("GPU SU kernel failed (%s); falling back to CPU prange kernel", exc)
                    flags = _run_cpu_pairwise_su(
                        bins_packed, nbins, freqs_packed, freqs_offsets,
                        h_marginals, constant_mask, threshold,
                        use_bitmap=use_bitmap,
                        bitmap_min_features=bitmap_min_features,
                        bitmap_max_n_bins=bitmap_max_n_bins,
                    )
            else:
                flags = _run_cpu_pairwise_su(
                    bins_packed, nbins, freqs_packed, freqs_offsets,
                    h_marginals, constant_mask, threshold,
                    use_bitmap=use_bitmap,
                    bitmap_min_features=bitmap_min_features,
                    bitmap_max_n_bins=bitmap_max_n_bins,
                )
            ei_arr, ej_arr = np.where(flags == 1)
            if ei_arr.size > edge_cap:
                raise MemoryError(
                    f"ShapProxiedFS SU clustering: >{edge_cap} edges at "
                    f"threshold={threshold}. Raise cluster_su_threshold to merge fewer features."
                )
            ei = ei_arr.astype(np.int64, copy=False)
            ej = ej_arr.astype(np.int64, copy=False)
            return _uf_labels(f, ei, ej)

    # Serial fallback only reached when use_kernel is False (small width) or packing
    # was unsafe. Build marginals lazily here so the parallel path (fused builder
    # above) doesn't pay the per-column Python-loop cost.
    marginals: list[tuple[np.ndarray, np.ndarray]] = []
    for name, col in zip(names, arrays):
        hint = None
        if nbins_per_feature is not None and name in nbins_per_feature:
            hint = int(nbins_per_feature[name])
        marginals.append(_column_marginal(col, n_bins_hint=hint))

    for i in range(f - 1):
        classes_i, freqs_i = marginals[i]
        if freqs_i.size == 0 or freqs_i.size == 1:
            # constant column -> SU=0 with every partner
            continue
        for j in range(i + 1, f):
            classes_j, freqs_j = marginals[j]
            if freqs_j.size == 0 or freqs_j.size == 1:
                continue
            # ``compute_su_from_classes`` is numba-jitted and reads the
            # int64 class arrays + float64 freq arrays we computed above.
            su = float(compute_su_from_classes(classes_i, freqs_i, classes_j, freqs_j))
            if su >= threshold:
                ei_parts.append(i)
                ej_parts.append(j)
                total += 1
                if total > edge_cap:
                    raise MemoryError(
                        f"ShapProxiedFS SU clustering: >{edge_cap} edges at "
                        f"threshold={threshold}. Raise cluster_su_threshold to merge fewer features."
                    )
    if not ei_parts:
        ei = np.empty(0, dtype=np.int64)
        ej = np.empty(0, dtype=np.int64)
    else:
        ei = np.asarray(ei_parts, dtype=np.int64)
        ej = np.asarray(ej_parts, dtype=np.int64)
    return _uf_labels(f, ei, ej)

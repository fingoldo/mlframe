"""GPU backend for the friend-graph BUILD (node stats + O(k^2) pairwise-MI edge pass).

Twin of the CPU compute in :mod:`friend_graph` (``build_friend_graph``), modeled on
``batch_pair_mi_gpu.py`` / ``batch_mi_noise_gate_gpu.py``: two GPU joint-histogram
backends (cupy + numba.cuda), a per-host backend chooser via ``get_or_tune`` keyed by
``(k, n)``, a measurement-backed fallback heuristic, a CPU-vs-GPU sweep tuner, an
availability-guarded dispatcher, and a ``@kernel_tuner`` registration. Default fallback
is CPU (safe).

What is accelerated
-------------------
For a selected set of ``k`` features over the discretized matrix the build computes:

  * per-node entropy ``H(X_i)``  (k marginals);
  * per-node relevance ``I(X_i; Y)`` (k feature-target MIs);
  * the O(k^2) pairwise feature-feature edge MI ``I(X_a; X_b)`` over all C(k, 2) pairs.

The last term dominates for large k and is the point of this module. The conditional
``neighbor_unique_target`` pass (``sum_j I(Y; X_j | X_i)``) is NOT batched here: it runs
only for the (small) subset of suspect nodes whose degree clears ``garbage_min_degree``,
so it is not an O(k^2) cost -- the CPU path keeps it.

BIT-IDENTITY (non-negotiable -- the diagnostic + the prune/cluster path are pinned)
-----------------------------------------------------------------------------------
The edge significance floor (``pairwise_mi_edge``) is a float comparison; a GPU MI off by
one ULP would flip a borderline edge and change the graph topology / which features the
prune path drops. We keep the result bit-for-bit identical to the CPU build by splitting
the work the same way the noise-gate GPU kernel does:

  1. The GPU computes only the INTEGER joint / marginal histograms (deterministic
     counting -> bit-exact). The joint code for a pair ``(a, b)`` is ``va + vb*nbins[a]``
     -- EXACTLY ``merge_vars``' dense encoding for ``vars_indices=[a, b]`` (var ``a``
     melted first, ``current_nclasses=nbins[a]``). The marginal code for a node is just
     ``va`` (a 1-var ``merge_vars``).
  2. The histogram is pruned of empty bins in ascending-code order (``counts[counts>0]``)
     and divided by ``n`` (``counts / n`` -- an int/int float division, identical to
     ``merge_vars``' ``freqs / len(factors_data)``), yielding the SAME ``freqs`` array
     ``merge_vars`` produces.
  3. The Shannon entropy is computed by calling the project's OWN ``entropy()`` njit
     (numpy ``-(log(freqs)*freqs).sum()``) on those freqs -- so ``H`` is bit-identical.
  4. ``H(X_a) + H(X_b) - H(X_a, X_b)`` (clamped at 0) reproduces ``pairwise_mi_edge``'s
     cached-marginal path EXACTLY (which is itself bit-identical to ``mi()``); relevance
     reproduces ``mi(X, Y)`` the same way. The significance floor + ADC direction +
     classification are then applied UNCHANGED by ``build_friend_graph`` on the CPU.

So the only GPU work is the O(n*k^2) (edges) + O(n*k) (nodes) integer counting; the
entropy + every keep/drop decision stay on the bit-exact CPU path. Verified by
``test_friend_graph_gpu.py`` (array_equal of node stats, edge MIs, and full graph).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .info_theory import entropy as _entropy

logger = logging.getLogger(__name__)

# Optional GPU deps -- mirror batch_pair_mi_gpu.py's probe order exactly.
try:
    from numba import cuda as _nb_cuda
except Exception:
    _nb_cuda = None

try:
    from pyutilz.core.pythonlib import is_cuda_available as _pyutilz_is_cuda_available
    _CUDA_AVAIL = _pyutilz_is_cuda_available()
except Exception:
    try:
        _CUDA_AVAIL = bool(getattr(_nb_cuda, "is_available", lambda: False)()) if _nb_cuda is not None else False
    except Exception:
        _CUDA_AVAIL = False

# Require numba.cuda kernel compilability (not just device presence) so a cudatoolkit/NVVM
# mismatch falls back to cupy/CPU rather than raising NvvmSupportError mid-dispatch.
from ._internals import numba_cuda_can_compile as _numba_cuda_can_compile
_CUDA_AVAIL = _CUDA_AVAIL and _numba_cuda_can_compile()

try:
    import cupy as _cp
    _CUPY_AVAIL = True
except Exception:
    _cp = None
    _CUPY_AVAIL = False


# ---------------------------------------------------------------------------
# Bit-exact CPU entropy-from-counts (shared by both GPU backends)
# ---------------------------------------------------------------------------


def _entropy_from_counts(counts: np.ndarray, n: int) -> float:
    """Shannon entropy (nats) of an integer histogram, reproducing the CPU build to the bit.

    ``counts`` is the FULL (possibly empty-bin-padded) integer histogram for one variable
    or joint. We prune empty bins in ascending-code order and divide by ``n`` -- exactly
    what ``merge_vars`` does (``freqs = counts / len(factors_data)`` after the empty-bin
    remap) -- then call the project's ``entropy()`` njit. ``counts / n`` is an int/int
    float division (NOT ``counts * (1/n)``), matching ``merge_vars`` array division, so the
    resulting ``freqs`` array is bit-identical and ``entropy()`` returns the identical float.
    """
    nz = counts[counts > 0]
    if nz.size == 0:
        return 0.0
    freqs = nz / n
    return float(_entropy(freqs=freqs))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class FriendGraphGPUStats:
    """Bit-exact GPU-computed node + edge statistics for ``build_friend_graph``.

    ``H[i]`` / ``rel[i]`` -- per selected-column entropy + target relevance (dicts keyed
    by the ORIGINAL column index, mirroring the CPU build's ``H`` / ``rel`` dicts).
    ``edge_mi`` -- a dict ``(a, b) -> raw I(X_a; X_b)`` (clamped >=0, BEFORE the
    significance floor) for every upper-triangular pair ``a < b`` in ``sel``; the build
    applies the floor + ADC direction itself, unchanged.
    """

    __slots__ = ("H", "backend", "edge_mi", "rel")

    def __init__(self, H: dict, rel: dict | None, edge_mi: dict, backend: str):
        self.H = H
        self.rel = rel
        self.edge_mi = edge_mi
        self.backend = backend


# ---------------------------------------------------------------------------
# Pair / target index plumbing (shared by both backends)
# ---------------------------------------------------------------------------


def _pair_indices(sel: list) -> tuple:
    """Upper-triangular (a, b) pairs (a < b, in ``sel`` order) as two int64 arrays."""
    k = len(sel)
    n_pairs = k * (k - 1) // 2
    pa = np.empty(n_pairs, dtype=np.int64)
    pb = np.empty(n_pairs, dtype=np.int64)
    p = 0
    for ia in range(k):
        a = sel[ia]
        for ib in range(ia + 1, k):
            pa[p] = a
            pb[p] = sel[ib]
            p += 1
    return pa, pb


def _target_col(target_indices: np.ndarray) -> Optional[int]:
    """Single target column for the GPU relevance path, else None.

    The GPU relevance kernel handles the dominant single-target case bit-exactly (joint
    code ``v_feat + v_y*nbins`` in ``merge_vars`` sorted-var order). For multi-column
    targets the caller falls back to CPU ``node_relevance`` (rare; not an O(k^2) cost)."""
    t = np.asarray(target_indices, dtype=np.int64).ravel()
    return int(t[0]) if t.shape[0] == 1 else None


# ---------------------------------------------------------------------------
# cupy backend
# ---------------------------------------------------------------------------


def friend_graph_stats_cupy(
    sel: list,
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    target_indices: np.ndarray,
    dtype: type = np.int32,
) -> FriendGraphGPUStats:
    """CuPy GPU twin of the friend-graph node + edge compute. BIT-IDENTICAL.

    GPU work (one H2D of the selected sub-frame + target up front, O(1) transfers
    otherwise):

      * node marginals: a single batched ``cupy.bincount`` over the flattened
        ``(col_block_offset + va)`` index for all k selected columns at once.
      * feature-target joints: a batched ``cupy.bincount`` over
        ``(col_block_offset + code(va, y))`` for all k columns -> relevance.
      * pairwise joints: tiled batched ``cupy.bincount`` over
        ``(pair_block_offset + va + vb*nbins[a])`` for all C(k, 2) pairs, tiled over the
        pair axis against a queried free-memory budget so big k never OOMs.

    The MI / entropy is reduced from those INTEGER counts on the bit-exact CPU path
    (``_entropy_from_counts`` -> the project ``entropy()`` njit). Raises ``RuntimeError``
    if cupy is unavailable.
    """
    if not _CUPY_AVAIL:
        raise RuntimeError("cupy is not available on this host")
    cp = _cp

    n = int(factors_data.shape[0])
    k = len(sel)
    nbins = np.asarray(factors_nbins, dtype=np.int64)
    sel_arr = np.asarray(sel, dtype=np.int64)

    # ---- ONE H2D: the selected sub-frame (n, k) as int32.
    sub = np.ascontiguousarray(factors_data[:, sel_arr], dtype=np.int32)
    d_sub = cp.asarray(sub)  # (n, k)
    nb_sel = nbins[sel_arr]  # (k,) per selected-column cardinality

    # ----- Node marginals: one batched bincount over all k columns.
    off_node = np.zeros(k + 1, dtype=np.int64)
    off_node[1:] = np.cumsum(nb_sel)
    total_node = int(off_node[k])
    d_off_node = cp.asarray(off_node[:k].reshape(1, k))  # (1, k)
    d_node_idx = (d_off_node + d_sub.astype(cp.int64)).reshape(-1)  # (n*k,)
    node_counts = cp.asnumpy(cp.bincount(d_node_idx, minlength=total_node)[:total_node])

    H: dict = {}
    for i in range(k):
        block = node_counts[off_node[i] : off_node[i + 1]]
        H[int(sel_arr[i])] = _entropy_from_counts(block, n)

    # ----- Feature-target relevance (single-target fast path on GPU; else CPU).
    rel: dict | None = {}
    tcol = _target_col(target_indices)
    if tcol is not None:
        nb_y = int(nbins[tcol])
        d_y = cp.asarray(np.ascontiguousarray(factors_data[:, tcol], dtype=np.int64)).reshape(n, 1)
        h_y = _entropy_from_counts(np.bincount(np.asarray(factors_data[:, tcol], dtype=np.int64), minlength=nb_y).astype(np.int64), n)
        # merge_vars sorts vars -> for feature f and target t the joint code is
        # ``val_first + val_second*nbins_first`` over sorted (f, t). Build per-column codes.
        per_col_joint = nb_sel * nb_y  # (k,)
        off_rel = np.zeros(k + 1, dtype=np.int64)
        off_rel[1:] = np.cumsum(per_col_joint)
        total_rel = int(off_rel[k])
        # joint code with bin layout matching merge_vars sorted-var order.
        d_codes = cp.empty((n, k), dtype=cp.int64)
        d_y_flat = d_y.reshape(n)
        for i in range(k):
            f = int(sel_arr[i])
            va = d_sub[:, i].astype(cp.int64)
            if f < tcol:
                code = va + d_y_flat * np.int64(int(nb_sel[i]))  # nbins_first = nbins[f]
            else:
                code = d_y_flat + va * np.int64(nb_y)  # nbins_first = nbins[t]
            d_codes[:, i] = code + np.int64(off_rel[i])
        rel_counts = cp.asnumpy(cp.bincount(d_codes.reshape(-1), minlength=total_rel)[:total_rel])
        assert rel is not None  # this branch starts with rel = {} above
        for i in range(k):
            f = int(sel_arr[i])
            block = rel_counts[off_rel[i] : off_rel[i + 1]]
            h_xy = _entropy_from_counts(block, n)
            rel[f] = H[f] + h_y - h_xy  # raw mi() value (mi() does not clamp relevance)
    else:
        rel = None  # caller computes relevance on CPU

    # ----- Pairwise edges: tiled batched bincount over all C(k, 2) pairs.
    edge_mi: dict = {}
    pa, pb = _pair_indices(sel)
    n_pairs = pa.shape[0]
    if n_pairs > 0:
        # column position within ``sel`` for each pair endpoint (to index d_sub).
        pos_of = {int(sel_arr[i]): i for i in range(k)}
        pos_a = np.array([pos_of[int(a)] for a in pa], dtype=np.int64)
        pos_b = np.array([pos_of[int(b)] for b in pb], dtype=np.int64)
        nb_a = nbins[pa]
        nb_b = nbins[pb]
        joint_card = nb_a * nb_b  # (n_pairs,) joint size per pair
        off_pair = np.zeros(n_pairs + 1, dtype=np.int64)
        off_pair[1:] = np.cumsum(joint_card)

        d_pos_a = cp.asarray(pos_a)
        d_pos_b = cp.asarray(pos_b)
        d_nb_a = cp.asarray(nb_a)

        # Budget tiling over the pair axis: each tile materialises a (rows, n) index
        # array + its bincount output. Index array (rows*n int64) dominates.
        try:
            free_b, _tot_b = cp.cuda.runtime.memGetInfo()
        except Exception:
            free_b = 512 * 1024 * 1024
        budget = int(free_b * 0.35)
        # per-pair tile cost ~ n int64 (index col) + joint_card int64 (count slot).
        max_jc = int(joint_card.max()) if n_pairs else 1
        bytes_per_pair = 8 * (n + max_jc)
        pairs_per_tile = max(1, budget // max(1, bytes_per_pair))
        if pairs_per_tile > n_pairs:
            pairs_per_tile = n_pairs
        if pairs_per_tile < n_pairs:
            logger.info(
                "friend_graph cupy: tiling %d pairs into tiles of %d (free=%dMB, n=%d)",
                n_pairs, pairs_per_tile, free_b // (1024 * 1024), n,
            )

        start = 0
        while start < n_pairs:
            stop = min(start + pairs_per_tile, n_pairs)
            rows = stop - start
            # (rows, n) joint codes: va + vb*nbins[a], offset into per-pair count slots.
            t_nb_a = d_nb_a[start:stop].reshape(rows, 1)
            # gather columns: d_sub is (n, k); we want (rows, n) where row r uses col pos_a[r].
            va = d_sub[:, d_pos_a[start:stop]].T.astype(cp.int64)  # (rows, n)
            vb = d_sub[:, d_pos_b[start:stop]].T.astype(cp.int64)  # (rows, n)
            code = va + vb * t_nb_a  # (rows, n)
            # per-tile slot offsets so one bincount fills all rows of the tile.
            tile_off = off_pair[start:stop] - off_pair[start]
            d_tile_off = cp.asarray(tile_off.reshape(rows, 1))
            tile_total = int(off_pair[stop] - off_pair[start])
            flat = (code + d_tile_off).reshape(-1)
            tile_counts = cp.asnumpy(cp.bincount(flat, minlength=tile_total)[:tile_total])
            for r in range(rows):
                p = start + r
                lo = int(off_pair[p] - off_pair[start])
                hi = int(off_pair[p + 1] - off_pair[start])
                h_ab = _entropy_from_counts(tile_counts[lo:hi], n)
                m = H[int(pa[p])] + H[int(pb[p])] - h_ab
                edge_mi[(int(pa[p]), int(pb[p]))] = m if m > 0.0 else 0.0
            del va, vb, code, flat, tile_counts
            start = stop

    return FriendGraphGPUStats(H=H, rel=rel, edge_mi=edge_mi, backend="cupy")


# ---------------------------------------------------------------------------
# numba.cuda backend
# ---------------------------------------------------------------------------


def _cuda_pair_hist_kernel_factory():
    """Build the numba.cuda joint-histogram kernel lazily. One block per pair; threads
    stride over rows and atomically populate a per-pair joint histogram in GLOBAL memory
    (no shared-mem cap on joint cardinality -> works for any column cardinality)."""
    if not _CUDA_AVAIL:
        return None

    @_nb_cuda.jit
    def _kernel(
        sub,  # (n, k) int32 -- selected sub-frame
        pair_posa,  # (n_pairs,) int64 -- column position in sub for endpoint a
        pair_posb,  # (n_pairs,) int64 -- column position in sub for endpoint b
        pair_nba,  # (n_pairs,) int64 -- nbins of endpoint a (joint stride)
        pair_off,  # (n_pairs,) int64 -- start offset of pair p in counts_flat
        counts_flat,  # (total_size,) int64 -- output, zeroed by host
        n,
    ):
        """Device kernel: one CUDA block per column pair accumulates that pair's joint histogram into ``counts_flat`` via atomic adds, striding threads over rows."""
        p = _nb_cuda.blockIdx.x
        if p >= pair_posa.shape[0]:
            return
        ca = pair_posa[p]
        cb = pair_posb[p]
        nba = pair_nba[p]
        off = pair_off[p]
        tid = _nb_cuda.threadIdx.x
        nthreads = _nb_cuda.blockDim.x
        for r in range(tid, n, nthreads):
            va = sub[r, ca]
            vb = sub[r, cb]
            _nb_cuda.atomic.add(counts_flat, off + va + vb * nba, 1)

    return _kernel


def _cuda_node_hist_kernel_factory():
    """One block per (selected) column; atomic marginal histogram in global memory.
    Also used for feature-target joints by treating the joint code as the cell."""
    if not _CUDA_AVAIL:
        return None

    @_nb_cuda.jit
    def _kernel(
        codes,  # (n, k) int64 -- per-column code (marginal: va; joint: code(va, y))
        col_off,  # (k,) int64 -- start offset of column i in counts_flat
        counts_flat,  # (total_size,) int64 -- output, zeroed by host
        n,
        k,
    ):
        """Device kernel: one CUDA block per column accumulates its marginal (or feature-target joint) histogram into ``counts_flat`` via atomic adds, striding threads over rows."""
        i = _nb_cuda.blockIdx.x
        if i >= k:
            return
        off = col_off[i]
        tid = _nb_cuda.threadIdx.x
        nthreads = _nb_cuda.blockDim.x
        for r in range(tid, n, nthreads):
            _nb_cuda.atomic.add(counts_flat, off + codes[r, i], 1)

    return _kernel


_CUDA_PAIR_KERNEL: Any = None
_CUDA_NODE_KERNEL: Any = None


def friend_graph_stats_cuda(
    sel: list,
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    target_indices: np.ndarray,
    dtype: type = np.int32,
    threads_per_block: int = 128,
) -> FriendGraphGPUStats:
    """numba.cuda GPU twin of the friend-graph node + edge compute. BIT-IDENTICAL.

    GPU work: global-atomic histogram kernels (one block per node/pair) for the node
    marginals, the feature-target joints (single-target fast path), and all C(k, 2)
    pairwise joints. MI/entropy reduced from the integer counts on the bit-exact CPU path
    (``_entropy_from_counts``). Raises ``RuntimeError`` if numba.cuda is unavailable.
    """
    global _CUDA_PAIR_KERNEL, _CUDA_NODE_KERNEL
    if not _CUDA_AVAIL:
        raise RuntimeError("numba.cuda is not available on this host")
    if _CUDA_PAIR_KERNEL is None:
        _CUDA_PAIR_KERNEL = _cuda_pair_hist_kernel_factory()
    if _CUDA_NODE_KERNEL is None:
        _CUDA_NODE_KERNEL = _cuda_node_hist_kernel_factory()
    if _CUDA_PAIR_KERNEL is None or _CUDA_NODE_KERNEL is None:
        raise RuntimeError("numba.cuda kernel factory failed to build")

    import warnings as _warnings
    try:
        from numba.core.errors import NumbaPerformanceWarning as _NbPerfWarn
    except Exception:
        _NbPerfWarn = None

    n = int(factors_data.shape[0])
    k = len(sel)
    nbins = np.asarray(factors_nbins, dtype=np.int64)
    sel_arr = np.asarray(sel, dtype=np.int64)

    sub = np.ascontiguousarray(factors_data[:, sel_arr], dtype=np.int32)
    d_sub = _nb_cuda.to_device(sub)
    nb_sel = nbins[sel_arr]

    def _launch_node(d_codes, col_off, total):
        """Upload column offsets, allocate + zero the flat device counts buffer, launch ``_CUDA_NODE_KERNEL`` over ``k`` blocks, and pull the resulting per-column histograms back to host."""
        d_off = _nb_cuda.to_device(np.ascontiguousarray(col_off, dtype=np.int64))
        d_counts = _nb_cuda.device_array(int(total), dtype=np.int64)
        d_counts[:] = 0
        with _warnings.catch_warnings():
            if _NbPerfWarn is not None:
                _warnings.simplefilter("ignore", _NbPerfWarn)
            _CUDA_NODE_KERNEL[k, threads_per_block](d_codes, d_off, d_counts, n, k)
        return d_counts.copy_to_host()

    # ---- Node marginals: codes == sub (va), per-column offsets by nbins.
    off_node = np.zeros(k + 1, dtype=np.int64)
    off_node[1:] = np.cumsum(nb_sel)
    # ``d_node_codes`` is just ``d_sub`` widened to int64 -- cast ON DEVICE (mirrors
    # ``friend_graph_stats_cupy``'s ``d_sub.astype(cp.int64)``) instead of re-uploading ``sub`` from host a
    # second time with the same content. ``numba.cuda.DeviceNDArray`` implements the CUDA Array Interface,
    # so wrapping ``d_sub`` with ``cp.asarray`` is a zero-copy view (no H2D); the resulting cupy array is a
    # verified drop-in argument for the ``numba.cuda.jit`` kernel launch below (see module wave notes).
    if _CUPY_AVAIL:
        d_node_codes = _cp.asarray(d_sub).astype(_cp.int64)
    else:
        d_node_codes = _nb_cuda.to_device(np.ascontiguousarray(sub, dtype=np.int64))
    node_counts = _launch_node(d_node_codes, off_node[:k], int(off_node[k]))
    H: dict = {}
    for i in range(k):
        H[int(sel_arr[i])] = _entropy_from_counts(node_counts[off_node[i] : off_node[i + 1]], n)

    # ---- Relevance (single-target GPU fast path; else CPU on the caller).
    tcol = _target_col(target_indices)
    if tcol is not None:
        nb_y = int(nbins[tcol])
        y = np.ascontiguousarray(factors_data[:, tcol], dtype=np.int64)
        h_y = _entropy_from_counts(np.bincount(y, minlength=nb_y).astype(np.int64), n)
        per_col_joint = nb_sel * nb_y
        off_rel = np.zeros(k + 1, dtype=np.int64)
        off_rel[1:] = np.cumsum(per_col_joint)
        codes = np.empty((n, k), dtype=np.int64)
        sub64 = sub.astype(np.int64)
        for i in range(k):
            f = int(sel_arr[i])
            if f < tcol:
                codes[:, i] = sub64[:, i] + y * int(nb_sel[i])
            else:
                codes[:, i] = y + sub64[:, i] * nb_y
        d_codes = _nb_cuda.to_device(np.ascontiguousarray(codes))
        rel_counts = _launch_node(d_codes, off_rel[:k], int(off_rel[k]))
        rel: dict | None = {}
        assert rel is not None  # just assigned {} above
        for i in range(k):
            f = int(sel_arr[i])
            h_xy = _entropy_from_counts(rel_counts[off_rel[i] : off_rel[i + 1]], n)
            rel[f] = H[f] + h_y - h_xy
    else:
        rel = None

    # ---- Pairwise edges.
    edge_mi: dict = {}
    pa, pb = _pair_indices(sel)
    n_pairs = pa.shape[0]
    if n_pairs > 0:
        pos_of = {int(sel_arr[i]): i for i in range(k)}
        pos_a = np.array([pos_of[int(a)] for a in pa], dtype=np.int64)
        pos_b = np.array([pos_of[int(b)] for b in pb], dtype=np.int64)
        nb_a = nbins[pa]
        nb_b = nbins[pb]
        joint_card = nb_a * nb_b
        off_pair = np.zeros(n_pairs + 1, dtype=np.int64)
        off_pair[1:] = np.cumsum(joint_card)
        total_pair = int(off_pair[n_pairs])

        d_posa = _nb_cuda.to_device(np.ascontiguousarray(pos_a, dtype=np.int64))
        d_posb = _nb_cuda.to_device(np.ascontiguousarray(pos_b, dtype=np.int64))
        d_nba = _nb_cuda.to_device(np.ascontiguousarray(nb_a, dtype=np.int64))
        d_poff = _nb_cuda.to_device(np.ascontiguousarray(off_pair[:n_pairs], dtype=np.int64))
        d_counts = _nb_cuda.device_array(total_pair, dtype=np.int64)
        d_counts[:] = 0
        with _warnings.catch_warnings():
            if _NbPerfWarn is not None:
                _warnings.simplefilter("ignore", _NbPerfWarn)
            _CUDA_PAIR_KERNEL[n_pairs, threads_per_block](
                d_sub, d_posa, d_posb, d_nba, d_poff, d_counts, n,
            )
        pair_counts = d_counts.copy_to_host()
        for p in range(n_pairs):
            block = pair_counts[off_pair[p] : off_pair[p + 1]]
            h_ab = _entropy_from_counts(block, n)
            m = H[int(pa[p])] + H[int(pb[p])] - h_ab
            edge_mi[(int(pa[p]), int(pb[p]))] = m if m > 0.0 else 0.0

    return FriendGraphGPUStats(H=H, rel=rel, edge_mi=edge_mi, backend="cuda")


# ---------------------------------------------------------------------------
# Backend chooser + tuner (mirrors batch_pair_mi_gpu.py / batch_mi_noise_gate_gpu.py)
# ---------------------------------------------------------------------------

# Measurement-backed fallback for the CPU-vs-GPU crossover, keyed on n_rows and k
# (number of selected features; the edge pass is O(k^2)). The CPU njit build wins for
# small/medium graphs (tiny per-pair joint-hist pass; the H2D copy + bincount launch
# overhead dominate on GPU). GPU pays off for large k at moderate-to-large n where the
# O(n*k^2) counting amortises. Default fallback = CPU (safe). The live dispatch consults
# the per-host kernel_tuning_cache first (per feedback_use_kernel_tuning_cache_for_gpu)
# so capable cards learn their own crossover instead of inheriting the dev box's.
#
# Dev-box measurement (GTX 1050 Ti cc6.1 4GB; cupy 13.6.0; nbins=8, n_classes_y=4),
# speedup = cpu/gpu, biteq=True at every cell:
#   n=5000  k=64  (2016 pairs): cuda 1.35x  cupy 1.04x
#   n=5000  k=128 (8128 pairs): cuda 1.38x  cupy 1.30x
#   n=5000  k=256 (32640 pairs): cuda 1.64x  cupy 0.11x
#   n=20000 k=128 (8128 pairs): cuda 3.61x  cupy 0.34x
# => the numba.cuda atomic-histogram backend (one block per pair, O(total_joint) global
# memory) SCALES with the O(n*k^2) edge work and is the GPU win. The cupy backend is
# competitive only at small k: its pairwise path tiles a dense (rows, n) int64 index
# array + a per-pair take_along_axis gather, which at large k materialises a huge
# index buffer and blows up (0.11x at k=256). So the fallback prefers CUDA for the GPU
# region. (bench-attempt: a resident cupy 3-D pair index like the noise-gate kernel was
# NOT pursued -- the cuda atomic kernel already wins and needs no n*k^2 index array.)
GPU_MIN_ROWS = 3_000
GPU_MIN_K = 128

_FG_SWEEP_N_ROWS = [2_000, 5_000, 20_000]
_FG_SWEEP_K = [32, 64, 128, 256]
_FG_SWEEP_NBINS = 8
_FG_SWEEP_N_CLASSES_Y = 4
_FG_SALT = 1  # bump on any numerics change to invalidate stale per-host cache


def _make_friend_graph_inputs(dims: dict):
    """Synthetic (sel, factors_data, factors_nbins, target_indices, dtype) tuple at
    ``dims['n_rows']`` rows with ``dims['k']`` selected feature columns + 1 target,
    matching the GPU/CPU stats call signature."""
    rng = np.random.default_rng(0)
    n = int(dims["n_rows"])
    k = int(dims["k"])
    nbins_v = _FG_SWEEP_NBINS
    data = rng.integers(0, nbins_v, size=(n, k + 1)).astype(np.int32)
    data[:, k] = rng.integers(0, _FG_SWEEP_N_CLASSES_Y, size=n)  # target last col
    factors_nbins = np.array([nbins_v] * k + [_FG_SWEEP_N_CLASSES_Y], dtype=np.int64)
    sel = list(range(k))
    target_indices = np.array([k], dtype=np.int64)
    return (sel, data, factors_nbins, target_indices, np.int32)


def _friend_graph_cpu_stats(sel, factors_data, factors_nbins, target_indices, dtype):
    """CPU reference: the exact node + edge math ``build_friend_graph`` runs, packaged
    as a ``FriendGraphGPUStats`` so the sweep can compare it to the GPU backends. Reuses
    the friend_graph primitives (``_node_entropy`` / ``node_relevance`` /
    ``_joint_entropy_2vars``) so it is bit-identical to the production CPU build."""
    from .friend_graph import _node_entropy, node_relevance, _joint_entropy_2vars

    H = {int(i): _node_entropy(factors_data, int(i), factors_nbins, None, dtype) for i in sel}
    tcol = _target_col(target_indices)
    if tcol is not None:
        rel = {int(i): node_relevance(factors_data, int(i), np.asarray(target_indices, dtype=np.int64), factors_nbins, dtype=dtype) for i in sel}
    else:
        rel = None
    edge_mi = {}
    pa, pb = _pair_indices([int(i) for i in sel])
    for p in range(pa.shape[0]):
        a, b = int(pa[p]), int(pb[p])
        h_ab = _joint_entropy_2vars(factors_data, a, b, factors_nbins, dtype=dtype)
        m = H[a] + H[b] - h_ab
        edge_mi[(a, b)] = m if m > 0.0 else 0.0
    return FriendGraphGPUStats(H=H, rel=rel, edge_mi=edge_mi, backend="cpu")


def _stats_to_vector(stats: "FriendGraphGPUStats") -> np.ndarray:
    """Flatten node H, relevance, and (sorted) edge MIs into one float64 vector so the
    sweep's array-equivalence harness can compare backends."""
    H_keys = sorted(stats.H)
    vec = [stats.H[k_] for k_ in H_keys]
    if stats.rel is not None:
        vec += [stats.rel[k_] for k_ in sorted(stats.rel)]
    vec += [stats.edge_mi[e] for e in sorted(stats.edge_mi)]
    return np.asarray(vec, dtype=np.float64)


def _run_friend_graph_sweep() -> list:
    """Full (n_rows x k) grid sweep -> backend_choice regions: cpu / cuda / cupy, fastest
    BIT-IDENTICAL variant per cell. The GPU variants are exact (integer counting on GPU,
    entropy on the bit-exact CPU path), so equivalence holds tightly."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    def _vec(fn):
        """Wrap a ``friend_graph_stats_*`` backend to return a flat numeric vector instead of a ``FriendGraphGPUStats`` struct, so ``sweep_backend_grid`` can compare backends for bit-identity by array equality."""
        return lambda *a: _stats_to_vector(fn(*a))

    variants = {
        "cpu": _vec(_friend_graph_cpu_stats),
    }
    if _CUDA_AVAIL:
        variants["cuda"] = _vec(friend_graph_stats_cuda)
    if _CUPY_AVAIL:
        variants["cupy"] = _vec(friend_graph_stats_cupy)
    return sweep_backend_grid(
        variants,
        {"n_rows": _FG_SWEEP_N_ROWS, "k": _FG_SWEEP_K},
        _make_friend_graph_inputs,
        reference="cpu",
        repeats=3, equiv_rtol=1e-9, equiv_atol=1e-12,
    )


def _friend_graph_code_version():
    """code_version over the CPU reference + GPU bodies + the shared reducer; re-tunes on
    any kernel edit."""
    try:
        from pyutilz.performance.kernel_tuning.code_versioning import compute_code_version

        fns = [_friend_graph_cpu_stats, _entropy_from_counts]
        if _CUDA_AVAIL:
            fns.append(friend_graph_stats_cuda)
        if _CUPY_AVAIL:
            fns.append(friend_graph_stats_cupy)
        return compute_code_version(*fns, salt=_FG_SALT)
    except Exception:
        return None


def _friend_graph_fallback_choice(n_rows: int, k: int) -> str:
    """Pre-sweep heuristic: GPU only for large n AND large k (where the O(n*k^2) counting
    amortises the H2D copy + launch overhead); CPU otherwise (safe default). Prefers CUDA
    (atomic-histogram, one block per pair -- scales with the edge work) over cupy (whose
    dense (rows, n) pair-index buffer blows up at large k; see the bench table above)."""
    if n_rows >= GPU_MIN_ROWS and k >= GPU_MIN_K:
        if _CUDA_AVAIL:
            return "cuda"
        if _CUPY_AVAIL:
            return "cupy"
    return "cpu"


def _friend_graph_backend_choice(n_rows: int, k: int) -> str:
    """Per-host backend (cpu/cuda/cupy) for this (n_rows, k) via the shared get_or_tune
    orchestrator; measurement-backed threshold fallback (default CPU)."""
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        result = KernelTuningCache.load_or_create().get_or_tune(
            "friend_graph_build",
            dims={"n_rows": int(n_rows), "k": int(k)},
            tuner=_run_friend_graph_sweep,
            axes=["n_rows", "k"],
            fallback={"backend_choice": _friend_graph_fallback_choice(n_rows, k)},
            code_version=_friend_graph_code_version(),
        )
        bc = result if isinstance(result, str) else str((result or {}).get("backend_choice", ""))
        if bc == "gpu":
            bc = "cupy" if _CUPY_AVAIL else ("cuda" if _CUDA_AVAIL else "cpu")
        if bc in ("cpu", "cuda", "cupy"):
            return bc
    except Exception as e:
        logger.debug("friend_graph_build get_or_tune failed: %s", e)
    return _friend_graph_fallback_choice(n_rows, k)


def dispatch_friend_graph_stats(
    sel: list,
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    target_indices: np.ndarray,
    dtype: type = np.int32,
    force_backend: str | None = None,
) -> Optional[FriendGraphGPUStats]:
    """Run the chosen GPU backend, returning a ``FriendGraphGPUStats`` or ``None`` when
    GPU is unavailable / not chosen (so ``build_friend_graph`` falls back to its CPU
    edge pass). Mirrors the other dispatchers' force_backend + availability guards. The
    CPU build is NOT run here -- the caller owns the CPU path.

    Returns ``None`` (CPU fallback) when ``len(sel) < 2`` (no edges to batch) too.
    """
    k = len(sel)
    n = int(factors_data.shape[0])
    if k < 2 or n == 0:
        return None

    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        # MLFRAME_DISABLE_GPU=1 / CUDA_VISIBLE_DEVICES="" must win even over an explicit
        # force_backend="cuda"/"cupy" caller request -- same absolute override as every other
        # GPU dispatch site in this package (see _gpu_policy.py's module docstring).
        return None

    # ABSOLUTE cushion guard (2026-07-05): on a near-full / SHARED card return None so the caller runs its CPU
    # edge pass, BEFORE the per-tile relative ``free_b * 0.35`` budget inside the cupy backend. The dominant
    # device buffer is the (rows, n) pair-index array; estimate one pair-row as the cushion's bytes_needed.
    # Pure ADD -- tightens, never loosens; permissive without cupy.
    try:
        from ._fe_gpu_vram import fe_gpu_has_vram_cushion
        if not fe_gpu_has_vram_cushion(n * 8):
            return None
    except Exception as e:
        logger.debug("swallowed exception in friend_graph_gpu.py: %s", e)
        pass

    if force_backend is not None:
        fb = force_backend.lower()
        if fb == "cupy" and _CUPY_AVAIL:
            return friend_graph_stats_cupy(sel, factors_data, factors_nbins, target_indices, dtype)
        if fb == "cuda" and _CUDA_AVAIL:
            return friend_graph_stats_cuda(sel, factors_data, factors_nbins, target_indices, dtype)
        return None  # forced CPU (or unavailable forced backend) -> caller uses CPU

    choice = _friend_graph_backend_choice(n, k)

    if choice == "cupy" and _CUPY_AVAIL:
        try:
            return friend_graph_stats_cupy(sel, factors_data, factors_nbins, target_indices, dtype)
        except Exception as e:
            logger.debug("friend_graph cupy backend failed (%s); falling back to CPU", e)
    if choice == "cuda" and _CUDA_AVAIL:
        try:
            return friend_graph_stats_cuda(sel, factors_data, factors_nbins, target_indices, dtype)
        except (ValueError, RuntimeError) as e:
            logger.debug("friend_graph cuda backend failed (%s); falling back to CPU", e)

    return None  # CPU region (or GPU failed) -> caller runs the CPU edge pass


# Register with the @kernel_tuner registry so retune_all / mlframe-tune-kernels discover
# + batch-tune friend_graph_build. GPU-capable (cuda/cupy backends).
try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    kernel_tuner(
        kernel_name="friend_graph_build",
        variant_fns=(),  # entropy reduction is the project entropy() njit; GPU covered by salt
        tuner=_run_friend_graph_sweep,
        axes={"n_rows": list(_FG_SWEEP_N_ROWS), "k": list(_FG_SWEEP_K)},
        fallback={"backend_choice": "cpu"},
        gpu_capable=True,
        salt=_FG_SALT,
        cli_label="friend_graph_build",
    )
except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
    logger.debug("suppressed in friend_graph_gpu.py:747: %s", e)
    pass


__all__ = [
    "FriendGraphGPUStats",
    "friend_graph_stats_cupy",
    "friend_graph_stats_cuda",
    "dispatch_friend_graph_stats",
    "_friend_graph_cpu_stats",
    "_friend_graph_backend_choice",
    "_friend_graph_code_version",
    "_run_friend_graph_sweep",
    "_CUDA_AVAIL",
    "_CUPY_AVAIL",
]

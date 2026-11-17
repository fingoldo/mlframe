"""Device form of the table TreeSHAP (``_shap_proxy_treeshap_table``): one thread per sample walks every tree keeping the bit mask of the path features the sample satisfies and, at a leaf, adds ``ud``
entries of the per-leaf contribution table to its phi row. The ``O(depth^2)`` unwinding of the path kernel (``_shap_proxy_treeshap_gpu``) is gone from the per-sample work: the table is built once per
ensemble on the host (numba, with the path kernel's own arithmetic) and uploaded.

Element type ``{REAL}``: float32 by default (the table is rounded once, the sums are over a few entries per leaf), double for the exact reference (``MLFRAME_SHAP_TREESHAP_REAL=double``).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import TreeEnsemble

logger = logging.getLogger(__name__)

_DEFAULT_BLOCK_SIZE = 128

_KERNEL_SRC = r"""
extern "C" __global__
void treeshap_table(const {REAL}* X, const int n, const int f,
                    const int* cl, const int* cr, const int* cd, const int* feat, const {REAL}* thr,
                    const int* node_slot, const int* leaf_of_node,
                    const int* leaf_ud, const int* leaf_feat_off, const int* leaf_feat, const int* leaf_table_off,
                    const {REAL}* table, const int* roots, const int n_trees, {REAL}* phi){
    // X is feature-major (f, n): the lanes of a warp (consecutive samples) read consecutive values of one feature row, one coalesced transaction per node
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
#if STACK_SHARED
    // the DFS stack of every thread in shared memory, laid out [level][thread] so the lanes of a warp hit distinct banks (spill-free, costs shared memory per block)
    extern __shared__ int smem[];
    int* st_node = smem;
    int* st_mask = smem + MAXSP * blockDim.x;
    #define SLOT(sp) ((sp) * blockDim.x + threadIdx.x)
#else
    // the DFS stack in the thread's stack frame (local memory, L1-resident): dynamically indexed, so ptxas keeps it out of registers
    int st_node[MAXSP];
    int st_mask[MAXSP];
    #define SLOT(sp) (sp)
#endif
    for (int t = 0; t < n_trees; t++){
        int sp = 0;
        st_node[SLOT(sp)] = roots[t]; st_mask[SLOT(sp)] = -1; sp++;
        while (sp > 0){
            sp--;
            const int node = st_node[SLOT(sp)];
            const int mask = st_mask[SLOT(sp)];
            const int left = cl[node];
            if (left < 0){   // leaf: add the table entries of the satisfied-feature mask
                const int leaf = leaf_of_node[node];
                const int ud = leaf_ud[leaf];
                if (ud > 0){
                    const int base = leaf_table_off[leaf] + (mask & ((1 << ud) - 1)) * ud;  // width-ok: a row of the leaf table, below 2^ud * ud for a path of ud <= max_depth features; the offsets are int32 by construction
                    const int fo = leaf_feat_off[leaf];
                    // phi is stored (feature, sample): the lanes of a warp stand on the same leaf, so they add to the same feature and consecutive samples, one coalesced transaction
                    for (int j = 0; j < ud; j++) phi[(long long)leaf_feat[fo + j] * n + i] += table[base + j];
                }
                continue;
            }
            const {REAL} xv = X[(long long)feat[node] * n + i];
            int hot;
            if (isnan(xv)) hot = cd[node];
            else if (xv < thr[node]) hot = left;
            else hot = cr[node];
            // the table form visits every node, so the order is the same for every sample (right pushed first, left popped first): the lanes of a warp stand on the same node at each step and the node reads
            // are broadcasts; only the mask differs (the cold child clears the bit of this node's feature slot)
            const int right = cr[node];
            const int cold_mask = mask & ~(1 << node_slot[node]);
            st_node[SLOT(sp)] = right; st_mask[SLOT(sp)] = (hot == right) ? mask : cold_mask; sp++;
            st_node[SLOT(sp)] = left;  st_mask[SLOT(sp)] = (hot == left) ? mask : cold_mask; sp++;
        }
    }
}
"""

_PREORDER_SRC = r"""
extern "C" __global__
void treeshap_table_preorder(const {REAL}* __restrict__ X, const int n, const int f, const int4* __restrict__ rec, const int n_rec,
                             const {REAL}* __restrict__ table, const int* __restrict__ leaf_feat, {REAL}* phi){
    // X is feature-major (f, n); the records are the static preorder of every tree (_shap_proxy_treeshap_preorder), the same for all samples, so the loop has no node stack: only the masks of the
    // children of the node at each depth are kept (mL: the left child's mask, mR: the right child's), and the record loads do not depend on the previous step
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int mL[MAXD];
    int mR[MAXD];
    for (int p = 0; p < n_rec; p++){
        const int4 r = rec[p];
        const int depth = r.x & 255;
        const int mask = (depth == 0) ? -1 : (((r.x >> 8) & 1) ? mL[depth - 1] : mR[depth - 1]);
        if ((r.x >> 9) & 1){   // leaf: add the table entries of the satisfied-feature mask
            const int ud = r.y;
            if (ud > 0){
                const int base = r.w + (mask & ((1 << ud) - 1)) * ud;  // width-ok: a row of the leaf table, below 2^ud * ud for a path of ud <= max_depth features; the offsets are int32 by construction
                // phi is stored (feature, sample): the lanes of a warp stand on the same leaf, so they add to the same feature and consecutive samples, one coalesced transaction
                for (int j = 0; j < ud; j++) phi[(long long)leaf_feat[r.z + j] * n + i] += table[base + j];
            }
            continue;
        }
        const {REAL} xv = X[(long long)r.y * n + i];
        const bool go_left = isnan(xv) ? (((r.x >> 10) & 1) != 0) : (xv < __int_as_float(r.z));
        const int cold = mask & ~(1 << r.w);   // the cold child clears the bit of this node's feature slot
        mL[depth] = go_left ? mask : cold;
        mR[depth] = go_left ? cold : mask;
    }
}
"""

_KERNELS: dict = {}
_LOCK = threading.Lock()


def _real():
    """Element type of the device table form: float32 by default, ``MLFRAME_SHAP_TREESHAP_REAL=double precision`` for the exact reference."""
    return np.float64 if os.environ.get("MLFRAME_SHAP_TREESHAP_REAL", "").strip().lower() == "float64" else np.float32  # Intentional float64: dtype check: float64 is a supported column type here, nothing is computed in it


_SMEM_BUDGET = 32 * 1024  # bytes of shared memory the DFS stacks of one block may take (the device limit is 48 KB)


def _maxsp(max_depth: int) -> int:
    """Entries of the explicit DFS stack: two frames per level."""
    return 2 * (int(max_depth) + 2) + 4


def _stack_shared() -> bool:
    """Whether the DFS stacks live in shared memory: ``MLFRAME_SHAP_TREESHAP_STACK=shared`` (spill-free) or ``local`` (the stack frame, the default; see the exemption in the register-pressure registry)."""
    return os.environ.get("MLFRAME_SHAP_TREESHAP_STACK", "local").strip().lower() == "shared"


def kernel_source(max_depth: int, real=np.float32, shared_stack: bool = False) -> str:
    """The kernel source for trees up to ``max_depth``, the element type ``real`` and the stack placement."""
    from mlframe.feature_selection.filters.shared import render

    return render(f"#define MAXSP {_maxsp(max_depth)}\n#define STACK_SHARED {int(shared_stack)}\n" + _KERNEL_SRC, real, acc=real)


def kernel(cp, real=np.float32):
    """``treeshap_table`` (the stack walk, kept as the reference) at depth 6 for the element type ``real``; the register-pressure registry and the offline per-arch test compile it through this."""
    return cp.RawKernel(kernel_source(6, real), "treeshap_table")


def preorder_source(max_depth: int, real=np.float32) -> str:
    """The preorder-walk kernel source for trees up to ``max_depth`` and the element type ``real``."""
    from mlframe.feature_selection.filters.shared import render

    return render(f"#define MAXD {int(max_depth) + 2}\n" + _PREORDER_SRC, real, acc=real)


def kernel_preorder(cp, real=np.float32, max_depth: int = 6):
    """``treeshap_table_preorder`` for trees up to ``max_depth`` and the element type ``real``: the compile point of the production walk and (at the default depth) the registry factory."""
    return cp.RawKernel(preorder_source(max_depth, real), "treeshap_table_preorder")


def _use_stack_walk() -> bool:
    """``MLFRAME_SHAP_TREESHAP_KERNEL=stack`` selects the older per-thread node-stack walk (``treeshap_table``) instead of the default static-preorder walk."""
    return os.environ.get("MLFRAME_SHAP_TREESHAP_KERNEL", "preorder").strip().lower() == "stack"


def _ensure_preorder_kernel(max_depth: int, real):
    """Compile the preorder kernel once per (depth, type) under a module lock."""
    key = ("preorder", int(max_depth), np.dtype(real).name)
    with _LOCK:
        k = _KERNELS.get(key)
        if k is None:
            import cupy as cp

            k = _KERNELS[key] = kernel_preorder(cp, real, int(max_depth))
    return k


def _ensure_kernel(max_depth: int, real, shared_stack: bool = False):
    """Compile once per (depth, type, stack placement) under a module lock."""
    key = (int(max_depth), np.dtype(real).name, bool(shared_stack))
    with _LOCK:
        k = _KERNELS.get(key)
        if k is None:
            import cupy as cp

            k = _KERNELS[key] = cp.RawKernel(kernel_source(max_depth, real, shared_stack), "treeshap_table")
    return k


def _block_size(max_depth: int = 6) -> int:
    """CUDA block size: ``MLFRAME_SHAP_TREESHAP_BLOCK`` when set, else the default, cut to a multiple of 32 that keeps the two DFS stacks of the block (``2 * MAXSP * block * 4`` bytes) within ``_SMEM_BUDGET``."""
    forced = os.environ.get("MLFRAME_SHAP_TREESHAP_BLOCK", "").strip()
    block = int(forced) if forced.isdigit() and int(forced) > 0 else _DEFAULT_BLOCK_SIZE
    fit = (_SMEM_BUDGET // (8 * _maxsp(max_depth))) // 32 * 32
    return max(32, min(block, fit))


_ROW_BLOCK_MEMORY_SHARE = 0.4  # share of the free device memory one row block of X, its transpose and its phi may take (the tables and records stay resident next to them)
_ROW_BLOCK_MIN = 4096


def _row_block(n: int, f: int, real) -> int:
    """Rows per device block: the (f, rows) float32 transpose of X, the staged float32 block of X and the (f, rows) ``phi`` of the element type ``real`` have to fit in a share of the free memory,
    so a frame of tens of millions of rows is streamed in row blocks instead of exhausting the card (the per-row cost is ``f * (4 + 4 + itemsize(real))`` bytes)."""
    import cupy as cp

    free, _total = cp.cuda.runtime.memGetInfo()
    per_row = f * (8 + np.dtype(real).itemsize)
    return int(max(_ROW_BLOCK_MIN, min(n, (free * _ROW_BLOCK_MEMORY_SHARE) // max(1, per_row))))


def treeshap_phi_base_table_gpu(ensemble: "TreeEnsemble", X: np.ndarray) -> Optional[tuple]:
    """``(phi (n, f) double precision, base)`` from the device table kernel, or ``None`` when the ensemble does not fit the table form or the rows do not repay the table build (the caller keeps the path kernel).
    The rows go through the device in row blocks sized from the free memory (:func:`_row_block`); the node arrays, the table and the preorder records are uploaded once and stay resident for the blocks."""
    import cupy as cp

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table import build_tables

    Xf = ensemble.as_float32(X)
    n, f = Xf.shape
    tab = build_tables(ensemble, n)
    if tab is None:
        return None
    real = _real()
    i32 = np.int32
    table_d = cp.asarray(tab["table"].astype(real))
    leaf_feat_d = cp.asarray(tab["leaf_feat"].astype(i32))
    preorder = not _use_stack_walk()
    shared = False
    if preorder:
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_preorder import preorder_records

        rec = preorder_records(ensemble, tab)
        rec_d = cp.asarray(rec)
        kern = _ensure_preorder_kernel(ensemble.max_depth, real)
        block = min(_block_size(), 1024)
    else:
        shared = _stack_shared()
        kern = _ensure_kernel(ensemble.max_depth, real, shared)
        block = _block_size(ensemble.max_depth) if shared else min(_block_size(), 1024)
        consts: tuple[Any, ...] = (
            cp.asarray(ensemble.children_left), cp.asarray(ensemble.children_right), cp.asarray(ensemble.children_default), cp.asarray(ensemble.features), cp.asarray(ensemble.thresholds.astype(real, copy=False)),
            cp.asarray(tab["node_slot"].astype(i32)), cp.asarray(tab["leaf_of_node"].astype(i32)),
            cp.asarray(tab["leaf_ud"].astype(i32)), cp.asarray(tab["leaf_feat_off"].astype(i32)), leaf_feat_d, cp.asarray(tab["leaf_table_off"].astype(i32)),
            table_d, cp.asarray(ensemble.tree_roots), i32(ensemble.tree_roots.shape[0]),
        )
    rows = _row_block(n, f, real)
    phi = np.empty((n, f), dtype=np.float64)  # Intentional float64: the caller's phi contract
    for start in range(0, n, rows):
        stop = min(n, start + rows)
        m = stop - start
        xt_d = cp.ascontiguousarray(cp.asarray(Xf[start:stop].astype(real, copy=False)).T)  # feature-major (f, m): the node reads of a warp coalesce
        phi_d = cp.zeros((f, m), dtype=real)  # (feature, sample): coalesced adds, transposed once below
        grid = ((m + block - 1) // block,)
        if preorder:
            kern(grid, (block,), (xt_d, i32(m), i32(f), rec_d, i32(rec.shape[0]), table_d, leaf_feat_d, phi_d))
        else:
            kern(grid, (block,), (xt_d, i32(m), i32(f), *consts, phi_d), shared_mem=8 * _maxsp(ensemble.max_depth) * block if shared else 0)
        phi[start:stop] = cp.asnumpy(phi_d).T
    return phi, float(ensemble.base_offset)

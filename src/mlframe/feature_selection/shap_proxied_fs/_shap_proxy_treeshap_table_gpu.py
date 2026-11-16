"""Device form of the table TreeSHAP (``_shap_proxy_treeshap_table``): one thread per sample walks every tree keeping the bit mask of the path features the sample satisfies and, at a leaf, adds ``ud``
entries of the per-leaf contribution table to its phi row. The ``O(depth^2)`` unwinding of the path kernel (``_shap_proxy_treeshap_gpu``) is gone from the per-sample work: the table is built once per
ensemble on the host (numba, with the path kernel's own arithmetic) and uploaded.

Element type ``{REAL}``: float32 by default (the table is rounded once, the sums are over a few entries per leaf), float64 for the exact reference (``MLFRAME_SHAP_TREESHAP_REAL=float64``).
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
void treeshap_table(const float* X, const int n, const int f,
                    const int* cl, const int* cr, const int* cd, const int* feat, const float* thr,
                    const int* node_slot, const int* leaf_of_node,
                    const int* leaf_ud, const int* leaf_feat_off, const int* leaf_feat, const int* leaf_table_off,
                    const {REAL}* table, const int* roots, const int n_trees, {REAL}* phi){
    // coalescing-ok: one thread walks the trees for one sample; the node and table reads are data-dependent gathers and the walks of different samples diverge, so no layout makes the lanes adjacent
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* xi = X + (long long)i * f;
    // the DFS stack of every thread lives in shared memory, laid out [level][thread] so the lanes of a warp hit distinct banks: dynamically indexed per-thread arrays would be a local-memory spill
    extern __shared__ int smem[];
    int* st_node = smem;
    int* st_mask = smem + MAXSP * blockDim.x;
    const int tid = threadIdx.x;
    for (int t = 0; t < n_trees; t++){
        int sp = 0;
        st_node[sp * blockDim.x + tid] = roots[t]; st_mask[sp * blockDim.x + tid] = -1; sp++;
        while (sp > 0){
            sp--;
            const int node = st_node[sp * blockDim.x + tid];
            const int mask = st_mask[sp * blockDim.x + tid];
            const int left = cl[node];
            if (left < 0){   // leaf: add the table entries of the satisfied-feature mask
                const int leaf = leaf_of_node[node];
                const int ud = leaf_ud[leaf];
                if (ud > 0){
                    const int base = leaf_table_off[leaf] + (mask & ((1 << ud) - 1)) * ud;
                    const int fo = leaf_feat_off[leaf];
                    // phi is stored (feature, sample): the lanes of a warp stand on the same leaf, so they add to the same feature and consecutive samples, one coalesced transaction
                    for (int j = 0; j < ud; j++) phi[(long long)leaf_feat[fo + j] * n + i] += table[base + j];
                }
                continue;
            }
            const float xv = xi[feat[node]];
            int hot;
            if (isnan(xv)) hot = cd[node];
            else if (xv < thr[node]) hot = left;
            else hot = cr[node];
            // the table form visits every node, so the order is the same for every sample (right pushed first, left popped first): the lanes of a warp stand on the same node at each step and the node reads
            // are broadcasts; only the mask differs (the cold child clears the bit of this node's feature slot)
            const int right = cr[node];
            const int cold_mask = mask & ~(1 << node_slot[node]);
            st_node[sp * blockDim.x + tid] = right; st_mask[sp * blockDim.x + tid] = (hot == right) ? mask : cold_mask; sp++;
            st_node[sp * blockDim.x + tid] = left;  st_mask[sp * blockDim.x + tid] = (hot == left) ? mask : cold_mask; sp++;
        }
    }
}
"""

_KERNELS: dict = {}
_LOCK = threading.Lock()


def _real():
    """Element type of the device table form: float32 by default, ``MLFRAME_SHAP_TREESHAP_REAL=float64`` for the exact reference."""
    return np.float64 if os.environ.get("MLFRAME_SHAP_TREESHAP_REAL", "").strip().lower() == "float64" else np.float32


_SMEM_BUDGET = 32 * 1024  # bytes of shared memory the DFS stacks of one block may take (the device limit is 48 KB)


def _maxsp(max_depth: int) -> int:
    """Entries of the explicit DFS stack: two frames per level."""
    return 2 * (int(max_depth) + 2) + 4


def kernel_source(max_depth: int, real=np.float32) -> str:
    """The kernel source for trees up to ``max_depth`` and the element type ``real``."""
    from mlframe.feature_selection.filters._kernel_types import render

    return render(f"#define MAXSP {_maxsp(max_depth)}\n" + _KERNEL_SRC, real, acc=real)


def kernel(cp, real=np.float32):
    """``treeshap_table`` at depth 6 for the element type ``real``; the register-pressure registry and the offline per-arch test compile it through this."""
    return cp.RawKernel(kernel_source(6, real), "treeshap_table")


def _ensure_kernel(max_depth: int, real):
    """Compile once per (depth, type) under a module lock."""
    key = (int(max_depth), np.dtype(real).name)
    with _LOCK:
        k = _KERNELS.get(key)
        if k is None:
            import cupy as cp

            k = _KERNELS[key] = cp.RawKernel(kernel_source(max_depth, real), "treeshap_table")
    return k


def _block_size(max_depth: int = 6) -> int:
    """CUDA block size: ``MLFRAME_SHAP_TREESHAP_BLOCK`` when set, else the default, cut to a multiple of 32 that keeps the two DFS stacks of the block (``2 * MAXSP * block * 4`` bytes) within ``_SMEM_BUDGET``."""
    forced = os.environ.get("MLFRAME_SHAP_TREESHAP_BLOCK", "").strip()
    block = int(forced) if forced.isdigit() and int(forced) > 0 else _DEFAULT_BLOCK_SIZE
    fit = (_SMEM_BUDGET // (8 * _maxsp(max_depth))) // 32 * 32
    return max(32, min(block, fit))


def treeshap_phi_base_table_gpu(ensemble: "TreeEnsemble", X: np.ndarray) -> Optional[tuple]:
    """``(phi (n, f) float64, base)`` from the device table kernel, or ``None`` when the ensemble does not fit the table form or the rows do not repay the table build (the caller keeps the path kernel)."""
    import cupy as cp

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table import build_tables

    Xf = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n, f = Xf.shape
    tab = build_tables(ensemble, n)
    if tab is None:
        return None
    real = _real()
    kern = _ensure_kernel(ensemble.max_depth, real)
    i32 = np.int32
    args: tuple[Any, ...] = (
        cp.asarray(Xf), i32(n), i32(f),
        cp.asarray(ensemble.children_left), cp.asarray(ensemble.children_right), cp.asarray(ensemble.children_default), cp.asarray(ensemble.features), cp.asarray(ensemble.thresholds),
        cp.asarray(tab["node_slot"].astype(i32)), cp.asarray(tab["leaf_of_node"].astype(i32)),
        cp.asarray(tab["leaf_ud"].astype(i32)), cp.asarray(tab["leaf_feat_off"].astype(i32)), cp.asarray(tab["leaf_feat"].astype(i32)), cp.asarray(tab["leaf_table_off"].astype(i32)),
        cp.asarray(tab["table"].astype(real)), cp.asarray(ensemble.tree_roots), i32(ensemble.tree_roots.shape[0]),
    )
    phi_d = cp.zeros((f, n), dtype=real)  # (feature, sample): coalesced adds, transposed once below
    block = _block_size(ensemble.max_depth)
    kern(((n + block - 1) // block,), (block,), (*args, phi_d), shared_mem=8 * _maxsp(ensemble.max_depth) * block)
    phi = np.ascontiguousarray(cp.asnumpy(phi_d).T).astype(np.float64, copy=False)  # Intentional float64: the caller's phi contract
    return phi, float(ensemble.base_offset)

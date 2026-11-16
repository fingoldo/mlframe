"""Table form of the TreeSHAP *interaction* tensor on the device.

For a leaf with ``ud`` unique path features (elements) the conditioned scans of the interaction kernel collapse to a table, exactly as the values kernel (``_shap_proxy_treeshap_table_gpu``) does for ``phi``:
conditioning on element ``a`` removes it from the path, the on-pass keeps the leaf only when the sample satisfies ``a`` (one fraction 1) and the off-pass scales it by the zero fraction ``z_a`` of ``a``, so
``Phi[e_i, e_a] += (bit_a - z_a) * T[a][mask without a][i]`` with ``T`` the (halved) unwound-sum contribution of element ``i`` on the path of the OTHER ``ud - 1`` elements. The per-sample, per-feature-pair
O(depth^2) conditioned scans of ``_shap_proxy_treeshap_interactions_gpu`` are gone; a sample only walks the trees keeping the bit mask and adds ``ud * (ud - 1)`` table entries per leaf.

The kernel also adds the unconditioned table entries to ``phi`` (the main effect), so the row-sum diagonal needs no second launch. Outputs are stored (pair, sample) so the adds of a warp coalesce, and the
host side (device ops) symmetrises, fills the diagonal and transposes to the ``(n, P, P)`` contract. The device path kernel stays as the fallback for ensembles whose pair table does not fit.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numba import njit, prange

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import _extend_path, _unwound_sum

if TYPE_CHECKING:
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import TreeEnsemble

logger = logging.getLogger(__name__)

MAX_PAIR_TABLE_ENTRIES = 24_000_000  # entries of the pair table (float32: 96 MB)
_PHI_CHUNK_BYTES = 256 * 1024 * 1024  # bytes of the (P*P, rows) device accumulator of one row chunk

_KERNEL_SRC = r"""
extern "C" __global__
void treeshap_interactions_table(const float* X, const int n, const int f, const int P,
                    const int* cl, const int* cr, const int* cd, const int* feat, const float* thr,
                    const int* node_slot, const int* leaf_of_node,
                    const int* leaf_ud, const int* leaf_feat_off, const int* leaf_feat, const int* leaf_table_off,
                    const {REAL}* table, const {REAL}* leaf_zero, const long long* leaf_pair_off, const {REAL}* pair_table,
                    const int* roots, const int n_trees, {REAL}* phi, {REAL}* pairs){
    // coalescing-ok: one thread walks the trees for one sample; the node and table reads are data-dependent gathers and the walks of different samples diverge, so no layout makes the lanes adjacent
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* xi = X + (long long)i * f;
#if STACK_SHARED
    extern __shared__ int smem[];
    int* st_node = smem;
    int* st_mask = smem + MAXSP * blockDim.x;
    #define SLOT(sp) ((sp) * blockDim.x + threadIdx.x)
#else
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
            if (left < 0){
                const int leaf = leaf_of_node[node];
                const int ud = leaf_ud[leaf];
                if (ud > 0){
                    const int fo = leaf_feat_off[leaf];
                    const int base = leaf_table_off[leaf] + (mask & ((1 << ud) - 1)) * ud;
                    for (int j = 0; j < ud; j++) phi[(long long)leaf_feat[fo + j] * n + i] += table[base + j];
                    if (ud > 1){
                        const long long pbase = leaf_pair_off[leaf];
                        for (int a = 0; a < ud; a++){
                            const {REAL} fa = ((mask >> a) & 1) ? ({REAL})1 - leaf_zero[fo + a] : -leaf_zero[fo + a];
                            const int mp = (mask & ((1 << a) - 1)) | (((mask >> (a + 1)) & ((1 << (ud - 1 - a)) - 1)) << a);
                            const long long row = pbase + (((long long)a << (ud - 1)) + mp) * (ud - 1);
                            const int fa_feat = leaf_feat[fo + a];
                            for (int ip = 0; ip < ud - 1; ip++){
                                const int e = ip < a ? ip : ip + 1;
                                pairs[((long long)leaf_feat[fo + e] * P + fa_feat) * n + i] += fa * pair_table[row + ip];
                            }
                        }
                    }
                }
                continue;
            }
            const float xv = xi[feat[node]];
            int hot;
            if (isnan(xv)) hot = cd[node];
            else if (xv < thr[node]) hot = left;
            else hot = cr[node];
            const int right = cr[node];
            const int cold_mask = mask & ~(1 << node_slot[node]);
            st_node[SLOT(sp)] = right; st_mask[SLOT(sp)] = (hot == right) ? mask : cold_mask; sp++;
            st_node[SLOT(sp)] = left;  st_mask[SLOT(sp)] = (hot == left) ? mask : cold_mask; sp++;
        }
    }
}
"""

_KERNELS: dict = {}
_LOCK = threading.Lock()


@njit(cache=True, parallel=True)
def _build_pair_tables(leaf_ud, leaf_feat_off, leaf_zero, leaf_pair_off, leaf_value, pair_table):
    """Pair table: for leaf ``L``, conditioned element ``a`` and mask ``m`` over the other ``ud - 1`` elements (bit order kept, ``a`` removed), entry ``(a, m, i')`` is
    ``0.5 * unwound_sum * (one - zero) * leaf`` of the ``i'``-th other element on the path of the others (the path kernel's own EXTEND / unwound-sum routines)."""
    for leaf in prange(leaf_ud.shape[0]):
        ud = leaf_ud[leaf]
        if ud < 2:
            continue
        off = leaf_feat_off[leaf]
        pf_feat = np.zeros(ud + 1, dtype=np.int64)
        pf_zero = np.empty(ud + 1, dtype=np.float64)  # Intentional float64: the path kernel's weights are double
        pf_one = np.empty(ud + 1, dtype=np.float64)  # Intentional float64: the path kernel's weights are double
        pweight = np.empty(ud + 1, dtype=np.float64)  # Intentional float64: the path kernel's weights are double
        m_ud = ud - 1
        for a in range(ud):
            for m in range(1 << m_ud):
                _extend_path(pf_feat, pf_zero, pf_one, pweight, 0, 0, 1.0, 1.0, -1)
                for ip in range(m_ud):
                    e = ip if ip < a else ip + 1
                    _extend_path(pf_feat, pf_zero, pf_one, pweight, 0, ip + 1, leaf_zero[off + e], 1.0 if (m >> ip) & 1 else 0.0, 0)
                row = leaf_pair_off[leaf] + ((a << m_ud) + m) * m_ud
                for ip in range(1, m_ud + 1):
                    w = _unwound_sum(pf_zero, pf_one, pweight, 0, m_ud, ip)
                    pair_table[row + ip - 1] = 0.5 * w * (pf_one[ip] - pf_zero[ip]) * leaf_value[leaf]


def build_pair_tables(tab: dict) -> Optional[dict]:
    """``tab`` (a ``build_tables`` result) extended with ``leaf_pair_off`` and ``pair_table``, or ``None`` when the pair table exceeds ``MAX_PAIR_TABLE_ENTRIES``."""
    ud = tab["leaf_ud"].astype(np.int64)
    sizes = np.where(ud >= 2, ud * (np.int64(1) << np.maximum(ud - 1, 0)) * (ud - 1), 0).astype(np.int64)
    if int(sizes.sum()) > MAX_PAIR_TABLE_ENTRIES:
        return None
    off = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
    pair = np.zeros(max(int(sizes.sum()), 1), dtype=np.float64)  # Intentional float64: the exact reference table, rounded once on upload
    _build_pair_tables(tab["leaf_ud"], tab["leaf_feat_off"], tab["leaf_zero"], off, tab["leaf_value"], pair)
    return {**tab, "leaf_pair_off": off, "pair_table": pair}


def kernel_source(max_depth: int, real=np.float32, shared_stack: bool = False) -> str:
    """The kernel source for trees up to ``max_depth`` and the element type ``real``."""
    from mlframe.feature_selection.filters._kernel_types import render
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table_gpu import _maxsp

    return render(f"#define MAXSP {_maxsp(max_depth)}\n" + _KERNEL_SRC, real, acc=real)


def kernel(cp, real=np.float32):
    """``treeshap_interactions_table`` at depth 6 for the element type ``real``; registry factory."""
    return cp.RawKernel(kernel_source(6, real), "treeshap_interactions_table")


def _ensure_kernel(max_depth: int, real, shared_stack: bool = False):
    """Compile once per (depth, type, stack placement) under a module lock."""
    key = (int(max_depth), np.dtype(real).name, bool(shared_stack))
    with _LOCK:
        k = _KERNELS.get(key)
        if k is None:
            import cupy as cp

            k = _KERNELS[key] = cp.RawKernel(kernel_source(max_depth, real, shared_stack), "treeshap_interactions_table")
    return k


def interaction_tensor_table_gpu(ensemble: "TreeEnsemble", X: np.ndarray) -> Optional[tuple]:
    """``(Phi (n, P, P) float64, phi (n, P) float64, base)`` from the device pair-table kernel, or ``None`` when the ensemble does not fit the table form (the caller keeps the path kernel)."""
    import cupy as cp

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table import build_tables
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table_gpu import _block_size, _maxsp, _real, _stack_shared

    Xf = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n, f = Xf.shape
    P = int(ensemble.n_features)
    if f != P:
        raise ValueError(f"X has {f} features but ensemble expects {P}.")
    tab = build_tables(ensemble, None)
    if tab is None:
        return None
    tab = build_pair_tables(tab)
    if tab is None:
        return None
    real = _real()
    shared = _stack_shared()
    kern = _ensure_kernel(ensemble.max_depth, real, shared)
    i32 = np.int32
    consts: tuple[Any, ...] = (
        cp.asarray(ensemble.children_left), cp.asarray(ensemble.children_right), cp.asarray(ensemble.children_default), cp.asarray(ensemble.features), cp.asarray(ensemble.thresholds),
        cp.asarray(tab["node_slot"].astype(i32)), cp.asarray(tab["leaf_of_node"].astype(i32)),
        cp.asarray(tab["leaf_ud"].astype(i32)), cp.asarray(tab["leaf_feat_off"].astype(i32)), cp.asarray(tab["leaf_feat"].astype(i32)), cp.asarray(tab["leaf_table_off"].astype(i32)),
        cp.asarray(tab["table"].astype(real)), cp.asarray(tab["leaf_zero"].astype(real)), cp.asarray(tab["leaf_pair_off"]), cp.asarray(tab["pair_table"].astype(real)),
        cp.asarray(ensemble.tree_roots), i32(ensemble.tree_roots.shape[0]),
    )
    block = _block_size(ensemble.max_depth) if shared else min(_block_size(), 1024)
    item = np.dtype(real).itemsize
    chunk = max(block, min(n, _PHI_CHUNK_BYTES // max(P * P * item, 1)))
    Phi = np.empty((n, P, P), dtype=np.float64)  # Intentional float64: the caller's tensor contract
    phi_out = np.empty((n, P), dtype=np.float64)  # Intentional float64: the caller's phi contract
    idx = cp.arange(P)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        m = e - s
        X_d = cp.asarray(Xf[s:e])
        phi_d = cp.zeros((P, m), dtype=real)  # (feature, sample): coalesced adds
        pairs_d = cp.zeros((P, P, m), dtype=real)  # (row feature, conditioned feature, sample)
        kern(((m + block - 1) // block,), (block,), (X_d, i32(m), i32(f), i32(P), *consts, phi_d, pairs_d), shared_mem=8 * _maxsp(ensemble.max_depth) * block if shared else 0)
        sym = pairs_d + pairs_d.transpose(1, 0, 2)
        sym *= 0.5
        sym[idx, idx, :] = phi_d - sym.sum(axis=1)  # the row-sum identity sum_k Phi_jk == phi_j; the diagonal of `sym` is zero before this
        Phi[s:e] = cp.asnumpy(sym.transpose(2, 0, 1))
        phi_out[s:e] = cp.asnumpy(phi_d.T)
    return Phi, phi_out, float(ensemble.base_offset)

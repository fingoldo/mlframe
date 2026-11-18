"""Optional cupy/CUDA variant of the path-dependent TreeSHAP scan (sample-parallel).

Mirrors the numba kernel in ``_shap_proxy_treeshap`` exactly (same EXTEND/UNWIND polynomial, same
float32 routing, same cover-ratio weights) but with one CUDA thread per SAMPLE iterating all trees.
The flat ensemble tensors are uploaded once; the kernel writes the (n, f) phi matrix on-device. This
is the always-kept GPU version (the numba kernel is the fallback) - the dispatcher in
``_shap_proxy_explain`` selects it only on a CUDA box for large ``n * f``.

Per-thread path scratch lives in local memory sized ``(max_depth+2) * (max_depth+2)``; this caps the
supported depth (asserted) so the kernel needs no dynamic allocation. Lazy compile is guarded by a
``multiprocessing.Lock`` + double-checked locking and cached as a module global (Windows-spawn safe),
mirroring ``_shap_proxy_gpu`` and ``filters/gpu.py``.
"""

from __future__ import annotations

import logging
import multiprocessing
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import TreeEnsemble

logger = logging.getLogger(__name__)

# Narrow exceptions that legitimately mean "no usable CUDA, demote to CPU": cupy not installed, or a
# cupy/CUDA runtime/driver error (no device, OOM, driver mismatch). Anything else (a bug in our own
# dispatch / lookup code) must propagate, not be silently swallowed into a CPU demotion (T4).
def _cuda_demote_errors():
    """Build the exception tuple that legitimately means "no usable CUDA, demote to CPU" (cupy missing, or a cupy runtime/compile error); anything outside this tuple must propagate as a real bug rather than being silently swallowed."""
    excs: tuple = (ImportError,)
    try:
        import cupy as cp

        excs = (*excs, cp.cuda.runtime.CUDARuntimeError)
        compile_exc = getattr(cp.cuda, "compiler", None)
        if compile_exc is not None and hasattr(compile_exc, "CompileException"):
            excs = (*excs, compile_exc.CompileException)
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed: %s", e)
        pass
    return excs


_TREESHAP_KERNELS: dict[tuple[int, str], Any] = {}
_KERNEL_INIT_LOCK = multiprocessing.Lock()

# Max tree depth the local-memory scratch supports without spilling to slow global memory.
_MAX_SUPPORTED_DEPTH = 24
_DEFAULT_BLOCK_SIZE = 32  # measured: 32 beat 64 / 128 / 256 by 14 % (n = 20000, 100 trees, depth 6, float32); the walks diverge, so small blocks balance the tail

_KERNEL_SRC = r"""
// Node record: the child cover shares of a node (cover of the child over the node's cover, ``0`` for a leaf or a zero-cover node) and the leaf value, read once per visited node.
#define NODE_LOAD(nd) const int ncl = cl[nd], ncr = cr[nd], ncd = cd[nd], nfeat = feat[nd]; const {REAL} nthr = thr[nd]; const {ACC} nval = val[nd]; const {ACC} wn_ = cover[nd];     const {ACC} rl = (wn_ > 0 && ncl >= 0) ? cover[ncl] / wn_ : ({ACC})0; const {ACC} rr = (wn_ > 0 && ncl >= 0) ? cover[ncr] / wn_ : ({ACC})0;
extern "C" __global__
void treeshap(const {REAL}* X, const int n, const int f,
              const int* cl, const int* cr, const int* cd,
              const int* feat, const {REAL}* thr, const {REAL}* val, const {REAL}* cover,
              const int* roots, const int n_trees, const int max_path,
              {REAL}* phi){
    // coalescing-ok: one thread walks the trees for one sample; xi[feat[node]] is a data-dependent gather and the walks diverge, so no layout makes lanes adjacent; selector off the MRMR fit path
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int width = max_path + 2;        // entries per recursion level
    // Per-thread path scratch (local memory). MAXW / MAXSP are injected #defines derived from the Python
    // ``_MAX_SUPPORTED_DEPTH`` cap (T1) so the buffer sizes can never drift out of sync with the host cap.
    int pf_feat[MAXW * MAXW];
    {ACC} pf_zero[MAXW * MAXW];
    {ACC} pf_one[MAXW * MAXW];
    {ACC} pweight[MAXW * MAXW];

    const {REAL}* xi = X + (long long)i * f;
    {REAL}* phi_i = phi + (long long)i * f;

    // Explicit stack replacing recursion: each frame is (node, level, unique_depth, zero, one, feat).
    int   st_node[MAXSP];
    int   st_level[MAXSP];
    int   st_ud[MAXSP];
    {ACC} st_zero[MAXSP];
    {ACC} st_one[MAXSP];
    int   st_feat[MAXSP];

    for (int t = 0; t < n_trees; t++){
        int sp = 0;
        st_node[sp]=roots[t]; st_level[sp]=0; st_ud[sp]=0; st_zero[sp]=1; st_one[sp]=1; st_feat[sp]=-1;
        sp++;
        while (sp > 0){
            sp--;
            int node=st_node[sp]; int level=st_level[sp]; int ud=st_ud[sp];
            {ACC} zfrac=st_zero[sp]; {ACC} ofrac=st_one[sp]; int fin=st_feat[sp];
            long long off = (long long)level * width;

            // Copy parent's path window into this level, then EXTEND.
            if (level > 0){
                long long poff = (long long)(level-1) * width;
                for (int k=0;k<ud;k++){
                    pf_feat[off+k]=pf_feat[poff+k]; pf_zero[off+k]=pf_zero[poff+k];
                    pf_one[off+k]=pf_one[poff+k];   pweight[off+k]=pweight[poff+k];
                }
            }
            pf_feat[off+ud]=fin; pf_zero[off+ud]=zfrac; pf_one[off+ud]=ofrac;
            pweight[off+ud] = (ud==0)?({ACC})1:({ACC})0;
            for (int ii=ud-1; ii>=0; ii--){
                pweight[off+ii+1] += ofrac*pweight[off+ii]*(ii+1)/(ud+1);
                pweight[off+ii]    = zfrac*pweight[off+ii]*(ud-ii)/(ud+1);
            }

            NODE_LOAD(node)
            if (ncl < 0){   // leaf
                {ACC} leaf = nval;
                for (int ii=1; ii<=ud; ii++){
                    {ACC} one_f=pf_one[off+ii]; {ACC} zero_f=pf_zero[off+ii];
                    {ACC} nxt=pweight[off+ud]; {ACC} total=0;
                    if (one_f != 0){
                        for (int jj=ud-1; jj>=0; jj--){
                            {ACC} tmp = nxt/((jj+1)*one_f);
                            total += tmp;
                            nxt = pweight[off+jj] - tmp*zero_f*(ud-jj);
                        }
                        total *= (ud+1);
                    } else {
                        for (int jj=ud-1; jj>=0; jj--)
                            total += pweight[off+jj]/(zero_f*(ud-jj));
                        total *= (ud+1);
                    }
                    phi_i[pf_feat[off+ii]] += ({REAL})(total*(one_f-zero_f)*leaf);  // the thread owns its phi row: a plain add, no atomic
                }
                continue;
            }

            int ff = nfeat;
            {REAL} xv = xi[ff];
            int hot, cold;
            if (isnan(xv)) hot = ncd;
            else if (xv < nthr) hot = ncl;
            else hot = ncr;
            cold = (hot == ncl) ? ncr : ncl;

            {ACC} hw = (hot == ncl) ? rl : rr;
            {ACC} cw = (hot == ncl) ? rr : rl;

            {ACC} inc_zero=1, inc_one=1; int pidx=0;
            for (int k=1;k<=ud;k++){ if (pf_feat[off+k]==ff){ pidx=k; break; } }
            int next_ud = ud + 1;
            if (pidx != 0){
                inc_zero = pf_zero[off+pidx]; inc_one = pf_one[off+pidx];
                // UNWIND in place
                {ACC} one_f=pf_one[off+pidx]; {ACC} zero_f=pf_zero[off+pidx];
                {ACC} nxt=pweight[off+ud];
                for (int ii=ud-1; ii>=0; ii--){
                    if (one_f != 0){
                        {ACC} tmp=pweight[off+ii];
                        pweight[off+ii]=nxt*(ud+1)/((ii+1)*one_f);
                        nxt = tmp - pweight[off+ii]*zero_f*(ud-ii)/(ud+1);
                    } else {
                        pweight[off+ii]=pweight[off+ii]*(ud+1)/(zero_f*(ud-ii));
                    }
                }
                for (int ii=pidx; ii<ud; ii++){
                    pf_feat[off+ii]=pf_feat[off+ii+1]; pf_zero[off+ii]=pf_zero[off+ii+1];
                    pf_one[off+ii]=pf_one[off+ii+1];
                }
                next_ud = ud;
            }

            // push cold then hot (process hot first by popping last)
            st_node[sp]=cold; st_level[sp]=level+1; st_ud[sp]=next_ud;
            st_zero[sp]=inc_zero*cw; st_one[sp]=0; st_feat[sp]=ff; sp++;
            st_node[sp]=hot;  st_level[sp]=level+1; st_ud[sp]=next_ud;
            st_zero[sp]=inc_zero*hw; st_one[sp]=inc_one; st_feat[sp]=ff; sp++;
        }
    }
}
"""


def gpu_treeshap_available() -> bool:
    """Return whether a CUDA device usable for the GPU TreeSHAP kernel is present; any cupy-missing or CUDA-runtime error demotes to ``False`` (CPU fallback) rather than raising."""
    try:
        import cupy as cp

        return bool(cp.cuda.runtime.getDeviceCount() > 0)
    except _cuda_demote_errors() as exc:
        logger.debug("GPU TreeSHAP unavailable, demoting to CPU: %s", exc)
        return False


def _block_size() -> int:
    """Hardware-tuned CUDA block size for the TreeSHAP kernel, or ``_DEFAULT_BLOCK_SIZE`` on a tuning-cache miss."""
    from ._gpu_block_size import block_size_from_tuning_cache

    import os

    forced = os.environ.get("MLFRAME_SHAP_TREESHAP_BLOCK", "").strip()
    if forced.isdigit() and int(forced) > 0:
        return int(forced)
    return block_size_from_tuning_cache("shap_proxy_treeshap", "gpu_block_size", _DEFAULT_BLOCK_SIZE)


def kernel_for_depth(cp: Any, max_depth: int, real: Any = np.float64) -> Any:  # Intentional float64: double-precision variant: the statistic needs the full mantissa
    """``treeshap`` with the per-thread path scratch sized for one ensemble depth and the tree data, path weights and output in ``real`` (float32 or float64): compile point of ``_ensure_kernel``."""
    from mlframe.feature_selection.filters.shared import render

    import os

    fast = np.dtype(real) == np.float32 and os.environ.get("MLFRAME_SHAP_TREESHAP_FASTMATH", "1").strip() != "0"
    return cp.RawKernel(render(_kernel_source(int(max_depth)), real, acc=real), "treeshap", options=("--use_fast_math",) if fast else ())


def kernel(cp: Any, real: Any = np.float64) -> Any:  # Intentional float64: double-precision variant: the statistic needs the full mantissa
    """``treeshap`` at depth 6 (the reference depth of the offline per-arch recipe) for tree data of dtype ``real``; the register-pressure registry compiles it through this."""
    return kernel_for_depth(cp, 6, real)


def _ensure_kernel(max_depth: int, real=np.float64):  # Intentional float64: double-precision variant: the statistic needs the full mantissa
    """Compile (once per depth and type, under a module-global lock with double-checked locking) and cache the ``cp.RawKernel``
    for the TreeSHAP scan (the lookup and the compile share one lock block, so a hit costs one lock acquire). The per-thread scratch is sized by the ensemble's real depth, not the cap: at the cap's
    depth 24 the kernel reserves gigabytes of device local memory that stay allocated and starve later GPU paths."""
    key = (int(max_depth), np.dtype(real).name)
    with _KERNEL_INIT_LOCK:
        kernel = _TREESHAP_KERNELS.get(key)
        if kernel is None:
            import cupy as cp

            kernel = kernel_for_depth(cp, int(max_depth), real)
            _TREESHAP_KERNELS[key] = kernel
    return kernel


def _kernel_source(max_depth: int) -> str:
    """Prepend ``#define``s sized for one ensemble depth: MAXW = depth + 2 (entries per level); MAXSP sizes the explicit
    DFS stack at 2*MAXW + 12. The host asserts ``max_depth <= _MAX_SUPPORTED_DEPTH`` before launch, so these sizes
    always bound the realised path and stack depth."""
    maxw = int(max_depth) + 2
    maxsp = 2 * maxw + 12
    return f"#define MAXW {maxw}\n#define MAXSP {maxsp}\n" + _KERNEL_SRC


def _treeshap_real():
    """Element type of the device TreeSHAP: float32 by default (about twice as fast on a 1/32-rate FP64 card, SHAP values within ~2e-5 of the double run and the same SHAP rank), ``MLFRAME_SHAP_TREESHAP_REAL=double precision`` for the exact reference."""
    import os

    return np.float64 if os.environ.get("MLFRAME_SHAP_TREESHAP_REAL", "").strip().lower() == "float64" else np.float32  # Intentional float64: dtype check: float64 is a supported column type here, nothing is computed in it


def treeshap_phi_base_gpu(ensemble: TreeEnsemble, X: np.ndarray, use_table: bool = True) -> tuple[np.ndarray, float]:
    """cupy path-dependent TreeSHAP. Identical contract to ``treeshap_phi_base_numba``: ``(phi, base)``.

    Raises ``ValueError`` if the ensemble depth exceeds the local-memory scratch cap (caller falls back
    to the numba kernel, which has no such cap). ``use_table=False`` runs the path kernel (the reference of the table form and its fallback)."""
    import cupy as cp

    if ensemble.max_depth > _MAX_SUPPORTED_DEPTH:
        raise ValueError(f"GPU TreeSHAP supports depth <= {_MAX_SUPPORTED_DEPTH}; ensemble depth is " f"{ensemble.max_depth}. Use the numba backend.")

    Xf = ensemble.as_float32(X)
    n, f = Xf.shape
    if f != ensemble.n_features:
        raise ValueError(f"X has {f} features but ensemble expects {ensemble.n_features}.")

    # Table form first (``_shap_proxy_treeshap_table_gpu``: no per-leaf unwinding in the per-sample work); ``None`` when the ensemble does not fit the table or the rows do not repay its build.
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table_gpu import treeshap_phi_base_table_gpu

    tabled = treeshap_phi_base_table_gpu(ensemble, Xf) if use_table else None
    if tabled is not None:
        return tabled

    real = _treeshap_real()
    kernel = _ensure_kernel(ensemble.max_depth, real)
    block = _block_size()

    roots_d = cp.asarray(ensemble.tree_roots)
    node_args = (
        cp.asarray(ensemble.children_left), cp.asarray(ensemble.children_right), cp.asarray(ensemble.children_default),
        cp.asarray(ensemble.features), cp.asarray(ensemble.thresholds.astype(real, copy=False)),
        cp.asarray(ensemble.values, dtype=real), cp.asarray(ensemble.node_sample_weight, dtype=real),
    )
    # The rows stream through the device in row blocks sized from the free device memory: the (rows, f) X block of ``real`` and the (rows, f) phi block must fit in a share of it (the node arrays stay resident for all blocks).
    free, _total = cp.cuda.runtime.memGetInfo()
    rows = int(max(4096, min(n, (free * 0.4) // max(1, 2 * f * np.dtype(real).itemsize))))
    phi = np.empty((n, f), dtype=np.float64)  # Intentional float64: the caller's phi contract is float64
    for start in range(0, n, rows):
        stop = min(n, start + rows)
        m = stop - start
        X_d = cp.asarray(Xf[start:stop].astype(real, copy=False))
        phi_d = cp.zeros((m, f), dtype=real)
        kernel(((m + block - 1) // block,), (block,), (
            X_d, np.int32(m), np.int32(f),
            *node_args,
            roots_d, np.int32(ensemble.tree_roots.shape[0]), np.int32(ensemble.max_depth),
            phi_d,
        ))
        phi[start:stop] = cp.asnumpy(phi_d)
    return phi, float(ensemble.base_offset)

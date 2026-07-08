"""Optional cupy/CUDA variant of the path-dependent TreeSHAP scan (sample-parallel).

Mirrors the numba kernel in ``_shap_proxy_treeshap`` exactly (same EXTEND/UNWIND polynomial, same
float32 routing, same cover-ratio weights) but with one CUDA thread per SAMPLE iterating all trees.
The flat ensemble tensors are uploaded once; the kernel writes the (n, f) phi matrix on-device. This
is the always-kept GPU version (the numba kernel is the fallback) -- the dispatcher in
``_shap_proxy_explain`` selects it only on a CUDA box for large ``n * f``.

Per-thread path scratch lives in local memory sized ``(max_depth+2) * (max_depth+2)``; this caps the
supported depth (asserted) so the kernel needs no dynamic allocation. Lazy compile is guarded by a
``multiprocessing.Lock`` + double-checked locking and cached as a module global (Windows-spawn safe),
mirroring ``_shap_proxy_gpu`` and ``filters/gpu.py``.
"""

from __future__ import annotations

import logging
import multiprocessing
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# Narrow exceptions that legitimately mean "no usable CUDA, demote to CPU": cupy not installed, or a
# cupy/CUDA runtime/driver error (no device, OOM, driver mismatch). Anything else (a bug in our own
# dispatch / lookup code) must propagate, not be silently swallowed into a CPU demotion (T4).
def _cuda_demote_errors():
    """Build the exception tuple that legitimately means "no usable CUDA, demote to CPU" (cupy missing, or a cupy runtime/compile error); anything outside this tuple must propagate as a real bug rather than being silently swallowed."""
    excs: tuple = (ImportError,)
    try:
        import cupy as cp

        excs = excs + (cp.cuda.runtime.CUDARuntimeError,)
        compile_exc = getattr(cp.cuda, "compiler", None)
        if compile_exc is not None and hasattr(compile_exc, "CompileException"):
            excs = excs + (compile_exc.CompileException,)
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed in _shap_proxy_treeshap_gpu.py:37: %s", e)
        pass
    return excs


_TREESHAP_KERNEL: Any = None
_KERNEL_INIT_LOCK = multiprocessing.Lock()

# Max tree depth the local-memory scratch supports without spilling to slow global memory.
_MAX_SUPPORTED_DEPTH = 24
_DEFAULT_BLOCK_SIZE = 128

_KERNEL_SRC = r"""
extern "C" __global__
void treeshap(const float* X, const int n, const int f,
              const int* cl, const int* cr, const int* cd,
              const int* feat, const float* thr, const double* val, const double* cover,
              const int* roots, const int n_trees, const int max_path,
              double* phi){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int width = max_path + 2;        // entries per recursion level
    const int n_levels = max_path + 2;
    // Per-thread path scratch (local memory). MAXW / MAXSP are injected #defines derived from the Python
    // ``_MAX_SUPPORTED_DEPTH`` cap (T1) so the buffer sizes can never drift out of sync with the host cap.
    long pf_feat[MAXW * MAXW];
    double pf_zero[MAXW * MAXW];
    double pf_one[MAXW * MAXW];
    double pweight[MAXW * MAXW];

    const float* xi = X + (long)i * f;
    double* phi_i = phi + (long)i * f;

    // Explicit stack replacing recursion: each frame is (node, level, unique_depth, zero, one, feat).
    int   st_node[MAXSP];
    int   st_level[MAXSP];
    int   st_ud[MAXSP];
    double st_zero[MAXSP];
    double st_one[MAXSP];
    int   st_feat[MAXSP];

    for (int t = 0; t < n_trees; t++){
        int sp = 0;
        st_node[sp]=roots[t]; st_level[sp]=0; st_ud[sp]=0; st_zero[sp]=1.0; st_one[sp]=1.0; st_feat[sp]=-1;
        sp++;
        while (sp > 0){
            sp--;
            int node=st_node[sp]; int level=st_level[sp]; int ud=st_ud[sp];
            double zfrac=st_zero[sp]; double ofrac=st_one[sp]; int fin=st_feat[sp];
            long off = (long)level * width;

            // Copy parent's path window into this level, then EXTEND.
            if (level > 0){
                long poff = (long)(level-1) * width;
                for (int k=0;k<ud;k++){
                    pf_feat[off+k]=pf_feat[poff+k]; pf_zero[off+k]=pf_zero[poff+k];
                    pf_one[off+k]=pf_one[poff+k];   pweight[off+k]=pweight[poff+k];
                }
            }
            pf_feat[off+ud]=fin; pf_zero[off+ud]=zfrac; pf_one[off+ud]=ofrac;
            pweight[off+ud] = (ud==0)?1.0:0.0;
            for (int ii=ud-1; ii>=0; ii--){
                pweight[off+ii+1] += ofrac*pweight[off+ii]*(ii+1.0)/(ud+1.0);
                pweight[off+ii]    = zfrac*pweight[off+ii]*(ud-ii)/(ud+1.0);
            }

            if (cl[node] < 0){   // leaf
                double leaf = val[node];
                for (int ii=1; ii<=ud; ii++){
                    double one_f=pf_one[off+ii]; double zero_f=pf_zero[off+ii];
                    double nxt=pweight[off+ud]; double total=0.0;
                    if (one_f != 0.0){
                        for (int jj=ud-1; jj>=0; jj--){
                            double tmp = nxt/((jj+1.0)*one_f);
                            total += tmp;
                            nxt = pweight[off+jj] - tmp*zero_f*(ud-jj);
                        }
                        total *= (ud+1.0);
                    } else {
                        for (int jj=ud-1; jj>=0; jj--)
                            total += pweight[off+jj]/(zero_f*(ud-jj));
                        total *= (ud+1.0);
                    }
                    atomicAdd(&phi_i[pf_feat[off+ii]], total*(one_f-zero_f)*leaf);
                }
                continue;
            }

            int ff = feat[node];
            float xv = xi[ff];
            int hot, cold;
            if (isnan(xv)) hot = cd[node];
            else if (xv < thr[node]) hot = cl[node];
            else hot = cr[node];
            cold = (hot == cl[node]) ? cr[node] : cl[node];

            double wn = cover[node];
            double hw = (wn>0.0)? cover[hot]/wn : 0.0;
            double cw = (wn>0.0)? cover[cold]/wn : 0.0;

            double inc_zero=1.0, inc_one=1.0; int pidx=0;
            for (int k=1;k<=ud;k++){ if (pf_feat[off+k]==ff){ pidx=k; break; } }
            int next_ud = ud + 1;
            if (pidx != 0){
                inc_zero = pf_zero[off+pidx]; inc_one = pf_one[off+pidx];
                // UNWIND in place
                double one_f=pf_one[off+pidx]; double zero_f=pf_zero[off+pidx];
                double nxt=pweight[off+ud];
                for (int ii=ud-1; ii>=0; ii--){
                    if (one_f != 0.0){
                        double tmp=pweight[off+ii];
                        pweight[off+ii]=nxt*(ud+1.0)/((ii+1.0)*one_f);
                        nxt = tmp - pweight[off+ii]*zero_f*(ud-ii)/(ud+1.0);
                    } else {
                        pweight[off+ii]=pweight[off+ii]*(ud+1.0)/(zero_f*(ud-ii));
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
            st_zero[sp]=inc_zero*cw; st_one[sp]=0.0; st_feat[sp]=ff; sp++;
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
    """Look up a hardware-tuned CUDA block size for this kernel from the shared ``kernel_tuning_cache``, falling back to ``_DEFAULT_BLOCK_SIZE`` when the cache is unavailable or has no entry for this kernel."""
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache

        ktc = get_kernel_tuning_cache()
        if ktc is not None:
            entry = cast(Any, ktc).lookup("shap_proxy_treeshap")
            if isinstance(entry, dict) and entry.get("gpu_block_size"):
                return int(entry["gpu_block_size"])
    except (ImportError, KeyError, ValueError, TypeError) as exc:
        logger.debug("GPU block-size tuning-cache lookup failed, using default: %s", exc)
    return _DEFAULT_BLOCK_SIZE


def _ensure_kernel():
    """Lazily compile (or return the already-compiled) ``cp.RawKernel`` for the TreeSHAP scan, guarded by a module-global lock + double-checked locking so concurrent callers (including Windows-spawned worker processes) never race the NVCC compile."""
    global _TREESHAP_KERNEL
    if _TREESHAP_KERNEL is not None:
        return _TREESHAP_KERNEL
    with _KERNEL_INIT_LOCK:
        if _TREESHAP_KERNEL is None:  # double-checked
            import cupy as cp

            _TREESHAP_KERNEL = cp.RawKernel(_kernel_source(), "treeshap")
    return _TREESHAP_KERNEL


def _kernel_source() -> str:
    """Prepend ``#define``s derived from the Python ``_MAX_SUPPORTED_DEPTH`` cap so the CUDA scratch sizes
    cannot drift from the host cap (T1). MAXW = depth + 2 (entries per level); MAXSP sizes the explicit DFS
    stack at 2*MAXW + 12 (the original literal was 64 at depth 24, i.e. 2*26 + 12; this formula reproduces
    that headroom and scales with the cap). The host asserts ``ensemble.max_depth <= _MAX_SUPPORTED_DEPTH``
    before launch, so these sizes always bound the realised path/stack depth."""
    maxw = _MAX_SUPPORTED_DEPTH + 2
    maxsp = 2 * maxw + 12
    return f"#define MAXW {maxw}\n#define MAXSP {maxsp}\n" + _KERNEL_SRC


def treeshap_phi_base_gpu(ensemble, X: np.ndarray):
    """cupy path-dependent TreeSHAP. Identical contract to ``treeshap_phi_base_numba``: ``(phi, base)``.

    Raises ``ValueError`` if the ensemble depth exceeds the local-memory scratch cap (caller falls back
    to the numba kernel, which has no such cap)."""
    import cupy as cp

    if ensemble.max_depth > _MAX_SUPPORTED_DEPTH:
        raise ValueError(f"GPU TreeSHAP supports depth <= {_MAX_SUPPORTED_DEPTH}; ensemble depth is " f"{ensemble.max_depth}. Use the numba backend.")

    Xf = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n, f = Xf.shape
    if f != ensemble.n_features:
        raise ValueError(f"X has {f} features but ensemble expects {ensemble.n_features}.")

    kernel = _ensure_kernel()
    block = _block_size()

    X_d = cp.asarray(Xf)
    cl_d = cp.asarray(ensemble.children_left)
    cr_d = cp.asarray(ensemble.children_right)
    cd_d = cp.asarray(ensemble.children_default)
    feat_d = cp.asarray(ensemble.features)
    thr_d = cp.asarray(ensemble.thresholds)
    val_d = cp.asarray(ensemble.values)
    cover_d = cp.asarray(ensemble.node_sample_weight)
    roots_d = cp.asarray(ensemble.tree_roots)
    phi_d = cp.zeros((n, f), dtype=cp.float64)

    grid = ((n + block - 1) // block,)
    kernel(grid, (block,), (
        X_d, np.int32(n), np.int32(f),
        cl_d, cr_d, cd_d, feat_d, thr_d, val_d, cover_d,
        roots_d, np.int32(ensemble.tree_roots.shape[0]), np.int32(ensemble.max_depth),
        phi_d,
    ))
    phi = cp.asnumpy(phi_d)
    return phi, float(ensemble.base_offset)

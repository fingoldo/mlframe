"""Optional cupy/CUDA variant of the path-dependent TreeSHAP *interaction* scan (sample-parallel).

Companion GPU kernel to the numba interaction kernel in ``_shap_proxy_treeshap_interactions`` (which
stays the always-available fallback -- ALL kernel versions are kept so we can re-bench per HW). One
CUDA thread per SAMPLE runs the same conditioned-scan recurrence the numba kernel does (one
unconditioned pass for the main effect ``phi`` plus, for each distinct split feature ``j``, an on-pass
and an off-pass), writes the off-diagonal ``Phi[:, p, j] = (phi_on[p] - phi_off[p]) / 2`` on-device,
symmetrises in place, then fills the diagonal from the row-sum identity ``sum_k Phi_ik == phi_i``.

It mirrors the EXTEND/UNWIND polynomial, float32 feature routing, NaN->default-child routing and
cover-ratio weights of both the numba interaction kernel and the main-effect GPU kernel in
``_shap_proxy_treeshap_gpu`` byte for byte, so the GPU tensor matches the numba tensor to ~1e-6 and
preserves exact symmetry + the row-sum identity + additivity.

Per-thread path scratch lives in local memory sized for ``_MAX_SUPPORTED_DEPTH``; conditioned passes
can reach one extra recursion level (the conditioned feature occupies no path slot at its child), so
the scratch and the explicit DFS stack are sized with that headroom, mirroring the numba kernel's
``n_levels = max_path + 3`` / ``stack_size = 2 * (max_path + 3)``. The local ``phi_*`` accumulators are
sized for ``_MAX_SUPPORTED_FEATURES`` (asserted at the host); ensembles wider than that fall back to
numba. Lazy compile is guarded by a ``multiprocessing.Lock`` + double-checked locking and cached as a
module global (Windows-spawn safe), mirroring ``_shap_proxy_treeshap_gpu`` and ``filters/gpu.py``.
"""

from __future__ import annotations

import logging
import multiprocessing
from typing import Any

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu import _cuda_demote_errors

logger = logging.getLogger(__name__)

_INTERACTION_KERNEL: Any = None
_KERNEL_INIT_LOCK = multiprocessing.Lock()

# Max tree depth the local-memory path scratch supports without spilling to slow global memory. Matches
# the main-effect GPU kernel's cap; conditioned passes add one extra level so MAXW = depth + 3.
_MAX_SUPPORTED_DEPTH = 24
# Max proxy width the local ``phi_*`` accumulators support. The interaction path is bounded to a small
# post-clustering proxy column count (see ``_shap_proxy_interactions``), so 256 is generous headroom.
_MAX_SUPPORTED_FEATURES = 256
_DEFAULT_BLOCK_SIZE = 64

_KERNEL_SRC = r"""
extern "C" __global__
void treeshap_interactions(const float* X, const int n, const int P,
                           const int* cl, const int* cr, const int* cd,
                           const int* feat, const float* thr, const double* val, const double* cover,
                           const int* roots, const int n_trees, const int max_path,
                           const int* cond_feats, const int n_cond,
                           double* Phi, double* phi_main){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int width = max_path + 2;        // entries per recursion level

    // Per-thread path scratch (local memory). MAXW / MAXLVL / MAXP are injected #defines derived from the
    // Python ``_MAX_SUPPORTED_DEPTH`` / ``_MAX_SUPPORTED_FEATURES`` caps (T1) so they cannot drift from the
    // host caps; the +3 levels mirror the numba kernel's n_levels = max_path + 3 (conditioned child deeper).
    long   pf_feat[MAXW * MAXLVL];
    double pf_zero[MAXW * MAXLVL];
    double pf_one [MAXW * MAXLVL];
    double pweight[MAXW * MAXLVL];

    // Explicit DFS stack (mirrors the numba kernel's stack_size = 2 * (max_path + 3)).
    const int MAXSP = 2 * MAXLVL;
    int    st_node [MAXSP];
    int    st_level[MAXSP];
    int    st_ud   [MAXSP];
    double st_zero [MAXSP];
    double st_one  [MAXSP];
    int    st_feat [MAXSP];
    double st_cfrac[MAXSP];

    double phi_unc[MAXP];
    double phi_on [MAXP];
    double phi_off[MAXP];

    const float* xi = X + (long)i * P;
    double* Phi_i = Phi + (long)i * P * P;
    double* phi_i = phi_main + (long)i * P;

    // ----- pass 0: unconditioned main effect -> phi_unc -----
    for (int p = 0; p < P; p++) phi_unc[p] = 0.0;
    treeshap_scan(xi, phi_unc, cl, cr, cd, feat, thr, val, cover, roots, n_trees, width,
                  0, -1, pf_feat, pf_zero, pf_one, pweight,
                  st_node, st_level, st_ud, st_zero, st_one, st_feat, st_cfrac);
    for (int p = 0; p < P; p++) phi_i[p] = phi_unc[p];

    // ----- conditioning passes: one (on, off) pair per distinct split feature -----
    for (int c = 0; c < n_cond; c++){
        int j = cond_feats[c];
        for (int p = 0; p < P; p++){ phi_on[p] = 0.0; phi_off[p] = 0.0; }
        treeshap_scan(xi, phi_on, cl, cr, cd, feat, thr, val, cover, roots, n_trees, width,
                      1, j, pf_feat, pf_zero, pf_one, pweight,
                      st_node, st_level, st_ud, st_zero, st_one, st_feat, st_cfrac);
        treeshap_scan(xi, phi_off, cl, cr, cd, feat, thr, val, cover, roots, n_trees, width,
                      -1, j, pf_feat, pf_zero, pf_one, pweight,
                      st_node, st_level, st_ud, st_zero, st_one, st_feat, st_cfrac);
        for (int p = 0; p < P; p++)
            if (p != j) Phi_i[(long)p * P + j] = 0.5 * (phi_on[p] - phi_off[p]);
    }

    // Symmetrise the off-diagonal in place, THEN fill the diagonal from the row-sum identity.
    for (int a = 0; a < P; a++){
        for (int b = a + 1; b < P; b++){
            double avg = 0.5 * (Phi_i[(long)a * P + b] + Phi_i[(long)b * P + a]);
            Phi_i[(long)a * P + b] = avg;
            Phi_i[(long)b * P + a] = avg;
        }
    }
    for (int j = 0; j < P; j++){
        double row_sum = 0.0;
        for (int p = 0; p < P; p++) if (p != j) row_sum += Phi_i[(long)j * P + p];
        Phi_i[(long)j * P + j] = phi_i[j] - row_sum;
    }
}
"""

# The conditioned per-tree scan, emitted as a __device__ function ABOVE the kernel so the kernel can
# call it. Mirrors numba ``_treeshap_one_tree_conditioned`` frame for frame (EXTEND-skip on the
# conditioned feature, condition_fraction split, leaf scaled by the surviving fraction).
_DEVICE_SRC = r"""
__device__ void treeshap_scan(
        const float* xi, double* phi_row,
        const int* cl, const int* cr, const int* cd,
        const int* feat, const float* thr, const double* val, const double* cover,
        const int* roots, const int n_trees, const int width,
        const int condition, const int condition_feature,
        long* pf_feat, double* pf_zero, double* pf_one, double* pweight,
        int* st_node, int* st_level, int* st_ud, double* st_zero, double* st_one,
        int* st_feat, double* st_cfrac){
    for (int t = 0; t < n_trees; t++){
        int sp = 0;
        st_node[sp]=roots[t]; st_level[sp]=0; st_ud[sp]=0;
        st_zero[sp]=1.0; st_one[sp]=1.0; st_feat[sp]=-1; st_cfrac[sp]=1.0;
        sp++;
        while (sp > 0){
            sp--;
            int node=st_node[sp]; int level=st_level[sp]; int ud=st_ud[sp];
            double parent_zero=st_zero[sp]; double parent_one=st_one[sp];
            int parent_feat=st_feat[sp]; double cond_frac=st_cfrac[sp];
            long off = (long)level * width;

            if (cond_frac == 0.0) continue;   // pruned conditioned branch

            // Copy parent's path prefix (ud+1 entries, mirroring the numba copy), then maybe EXTEND.
            if (level > 0){
                long poff = (long)(level-1) * width;
                for (int k=0;k<=ud;k++){
                    pf_feat[off+k]=pf_feat[poff+k]; pf_zero[off+k]=pf_zero[poff+k];
                    pf_one[off+k]=pf_one[poff+k];   pweight[off+k]=pweight[poff+k];
                }
            }
            if (condition == 0 || condition_feature != parent_feat){
                pf_feat[off+ud]=parent_feat; pf_zero[off+ud]=parent_zero; pf_one[off+ud]=parent_one;
                pweight[off+ud] = (ud==0)?1.0:0.0;
                for (int ii=ud-1; ii>=0; ii--){
                    pweight[off+ii+1] += parent_one*pweight[off+ii]*(ii+1.0)/(ud+1.0);
                    pweight[off+ii]    = parent_zero*pweight[off+ii]*(ud-ii)/(ud+1.0);
                }
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
                    phi_row[pf_feat[off+ii]] += total*(one_f-zero_f)*leaf*cond_frac;
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

            double incoming_zero=1.0, incoming_one=1.0; int pidx=0;
            for (int k=1;k<=ud;k++){ if (pf_feat[off+k]==ff){ pidx=k; break; } }
            int child_depth = ud + 1;
            if (pidx != 0){
                incoming_zero = pf_zero[off+pidx]; incoming_one = pf_one[off+pidx];
                // UNWIND the prior occurrence in place.
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
                child_depth = ud;
            }

            double hot_cfrac = cond_frac;
            double cold_cfrac = cond_frac;
            if (condition > 0 && ff == condition_feature){
                cold_cfrac = 0.0;
                child_depth -= 1;
            } else if (condition < 0 && ff == condition_feature){
                hot_cfrac = cond_frac * hw;
                cold_cfrac = cond_frac * cw;
                child_depth -= 1;
            }

            // push cold then hot (hot popped first)
            st_node[sp]=cold; st_level[sp]=level+1; st_ud[sp]=child_depth;
            st_zero[sp]=incoming_zero*cw; st_one[sp]=0.0; st_feat[sp]=ff; st_cfrac[sp]=cold_cfrac; sp++;
            st_node[sp]=hot;  st_level[sp]=level+1; st_ud[sp]=child_depth;
            st_zero[sp]=incoming_zero*hw; st_one[sp]=incoming_one; st_feat[sp]=ff; st_cfrac[sp]=hot_cfrac; sp++;
        }
    }
}
"""


def gpu_interactions_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except _cuda_demote_errors() as exc:
        logger.debug("GPU interaction kernel unavailable, demoting to CPU: %s", exc)
        return False


def _block_size() -> int:
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache

        ktc = get_kernel_tuning_cache()
        if ktc is not None:
            entry = ktc.lookup("shap_proxy_treeshap")
            if isinstance(entry, dict) and entry.get("interaction_gpu_block_size"):
                return int(entry["interaction_gpu_block_size"])
    except (ImportError, KeyError, ValueError, TypeError) as exc:
        logger.debug("GPU interaction block-size tuning-cache lookup failed, using default: %s", exc)
    return _DEFAULT_BLOCK_SIZE


def _ensure_kernel():
    global _INTERACTION_KERNEL
    if _INTERACTION_KERNEL is not None:
        return _INTERACTION_KERNEL
    with _KERNEL_INIT_LOCK:
        if _INTERACTION_KERNEL is None:  # double-checked
            import cupy as cp

            # The __device__ scan must precede the kernel that calls it; compile them as one module
            # and fetch the kernel entry point. Scratch-size #defines (T1) are injected from the Python
            # caps at the very front so neither the device scan nor the kernel can use a drifted literal.
            module = cp.RawModule(code=_defines_header() + _DEVICE_SRC + _KERNEL_SRC)
            _INTERACTION_KERNEL = module.get_function("treeshap_interactions")
    return _INTERACTION_KERNEL


def _defines_header() -> str:
    """CUDA ``#define``s for the scratch caps, derived from the Python constants (T1): MAXW = depth + 2,
    MAXLVL = depth + 3 (conditioned child reaches one level deeper), MAXP = max supported features. The
    host asserts ``ensemble.max_depth <= _MAX_SUPPORTED_DEPTH`` and width ``<= _MAX_SUPPORTED_FEATURES``
    before launch, so these bound the realised path/stack depth and the per-thread phi accumulators."""
    maxw = _MAX_SUPPORTED_DEPTH + 2
    maxlvl = _MAX_SUPPORTED_DEPTH + 3
    return f"#define MAXW {maxw}\n#define MAXLVL {maxlvl}\n#define MAXP {_MAX_SUPPORTED_FEATURES}\n"


def interaction_tensor_gpu(ensemble, X: np.ndarray):
    """cupy path-dependent TreeSHAP interaction tensor. Identical contract to
    ``interaction_tensor_numba``: returns ``(Phi (n,P,P) float64, phi (n,P) float64, base float)``.

    Raises ``ValueError`` if the ensemble depth or width exceeds the local-memory scratch caps (caller
    falls back to the numba kernel, which has no such caps)."""
    import cupy as cp

    if ensemble.max_depth > _MAX_SUPPORTED_DEPTH:
        raise ValueError(
            f"GPU TreeSHAP interactions support depth <= {_MAX_SUPPORTED_DEPTH}; ensemble depth is " f"{ensemble.max_depth}. Use the numba backend."
        )
    P = ensemble.n_features
    if P > _MAX_SUPPORTED_FEATURES:
        raise ValueError(f"GPU TreeSHAP interactions support width <= {_MAX_SUPPORTED_FEATURES}; ensemble width is " f"{P}. Use the numba backend.")

    Xf = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n, f = Xf.shape
    if f != P:
        raise ValueError(f"X has {f} features but ensemble expects {P}.")

    split_feats = ensemble.features[ensemble.features >= 0]
    cond_feats = np.unique(split_feats).astype(np.int32) if split_feats.size else np.empty(0, dtype=np.int32)

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
    cond_d = cp.asarray(cond_feats)
    Phi_d = cp.zeros((n, P, P), dtype=cp.float64)
    phi_d = cp.zeros((n, P), dtype=cp.float64)

    grid = ((n + block - 1) // block,)
    kernel(grid, (block,), (
        X_d, np.int32(n), np.int32(P),
        cl_d, cr_d, cd_d, feat_d, thr_d, val_d, cover_d,
        roots_d, np.int32(ensemble.tree_roots.shape[0]), np.int32(ensemble.max_depth),
        cond_d, np.int32(cond_feats.shape[0]),
        Phi_d, phi_d,
    ))
    Phi = cp.asnumpy(Phi_d)
    phi = cp.asnumpy(phi_d)
    return Phi, phi, float(ensemble.base_offset)

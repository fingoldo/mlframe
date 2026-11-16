"""Interleaved kernel-only A/B of variants of the device table TreeSHAP kernel (block size, register cap, X layout, node packing), every variant checked against the baseline output.

``python -m mlframe.feature_selection._benchmarks.profiling.treeshap_table_gpu_variants ROWS [FEATURES TREES DEPTH] [--rounds N] [--only name,name]``. The tables and inputs are uploaded once; each variant
is timed with CUDA events around the launch only, round-robin over the rounds in one process, with the hardware state before and after.
"""

from __future__ import annotations

import argparse

import numpy as np

from mlframe.feature_selection._benchmarks.profiling.hardware_state import line, snapshot
from mlframe.feature_selection._benchmarks.profiling.treeshap_gpu_profile import build_ensemble

_X_FEATURE_MAJOR = ("const float xv = X[(long long)feat[node] * n + i];", "const float xv = xi[feat[node]];")

_AOS_SRC = r"""
extern "C" __global__
void treeshap_table(const float* X, const int n, const int f, const int4* nodes, const {REAL}* table, const int* leaf_feat, const int* roots, const int n_trees, {REAL}* phi){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int st_node[MAXSP];
    int st_mask[MAXSP];
    for (int t = 0; t < n_trees; t++){
        int sp = 0;
        st_node[sp] = roots[t]; st_mask[sp] = -1; sp++;
        while (sp > 0){
            sp--;
            const int node = st_node[sp];
            const int mask = st_mask[sp];
            const int4 a = nodes[2 * node];
            const int4 b = nodes[2 * node + 1];
            if (a.x < 0){
                const int ud = b.x;
                if (ud > 0){
                    const int base = b.z + (mask & ((1 << ud) - 1)) * ud;
                    for (int j = 0; j < ud; j++) phi[(long long)leaf_feat[b.y + j] * n + i] += table[base + j];
                }
                continue;
            }
            const float xv = X[(long long)a.z * n + i];
            int hot;
            if (isnan(xv)) hot = b.x;
            else if (xv < __int_as_float(a.w)) hot = a.x;
            else hot = a.y;
            const int cold_mask = mask & ~(1 << b.y);
            st_node[sp] = a.y; st_mask[sp] = (hot == a.y) ? mask : cold_mask; sp++;
            st_node[sp] = a.x; st_mask[sp] = (hot == a.x) ? mask : cold_mask; sp++;
        }
    }
}
"""


def _pack_nodes(ens, tab):
    """Node records of 32 bytes: ``[left, right, feature, threshold bits | default, slot, 0, 0]`` for a split, ``[-1, 0, 0, 0 | ud, leaf_feat_off, leaf_table_off, 0]`` for a leaf."""
    nn = ens.children_left.shape[0]
    rec = np.zeros((nn, 8), dtype=np.int32)
    rec[:, 0] = ens.children_left
    rec[:, 1] = ens.children_right
    rec[:, 2] = np.maximum(ens.features, 0)
    rec[:, 3] = np.ascontiguousarray(ens.thresholds, dtype=np.float32).view(np.int32)
    rec[:, 4] = ens.children_default
    rec[:, 5] = tab["node_slot"]
    leaf = ens.children_left < 0
    lid = tab["leaf_of_node"][leaf]
    rec[leaf, 4] = tab["leaf_ud"][lid]
    rec[leaf, 5] = tab["leaf_feat_off"][lid]
    rec[leaf, 6] = tab["leaf_table_off"][lid]
    return rec


_PRE_SRC = r"""
extern "C" __global__
void treeshap_table(const float* X, const int n, const int f, const int4* rec, const int n_rec, const {REAL}* table, const int* leaf_feat, {REAL}* phi){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int mL[MAXD];
    int mR[MAXD];
    for (int p = 0; p < n_rec; p++){
        const int4 r = rec[p];
        const int depth = r.x & 255;
        const int mask = (depth == 0) ? -1 : (((r.x >> 8) & 1) ? mL[depth - 1] : mR[depth - 1]);
        if ((r.x >> 9) & 1){
            const int ud = r.y;
            if (ud > 0){
                const int base = r.w + (mask & ((1 << ud) - 1)) * ud;
                for (int j = 0; j < ud; j++) phi[(long long)leaf_feat[r.z + j] * n + i] += table[base + j];
            }
            continue;
        }
        const float xv = X[(long long)r.y * n + i];
        const bool go_left = isnan(xv) ? (((r.x >> 10) & 1) != 0) : (xv < __int_as_float(r.z));
        const int cold = mask & ~(1 << r.w);
        mL[depth] = go_left ? mask : cold;
        mR[depth] = go_left ? cold : mask;
    }
}
"""


def _pack_preorder(ens, tab):
    """Static preorder (left first) records of all trees, 16 bytes each: ``x = depth | is_left << 8 | is_leaf << 9 | default_left << 10``; split: ``y`` feature, ``z`` threshold bits, ``w`` slot; leaf: ``y`` ud, ``z`` feature offset, ``w`` table offset."""
    cl, cr, cd, ft = ens.children_left, ens.children_right, ens.children_default, ens.features
    recs = []
    for root in ens.tree_roots:
        stack = [(int(root), 0, 0)]
        while stack:
            node, depth, is_left = stack.pop()
            if cl[node] < 0:
                lid = tab["leaf_of_node"][node]
                recs.append((depth | (is_left << 8) | (1 << 9), int(tab["leaf_ud"][lid]), int(tab["leaf_feat_off"][lid]), int(tab["leaf_table_off"][lid])))
                continue
            thr = int(np.float32(ens.thresholds[node]).view(np.int32))
            recs.append((depth | (is_left << 8) | ((1 if cd[node] == cl[node] else 0) << 10), int(ft[node]), thr, int(tab["node_slot"][node])))
            stack.append((int(cr[node]), depth + 1, 0))
            stack.append((int(cl[node]), depth + 1, 1))
    return np.asarray(recs, dtype=np.int32)



def _variants(base_src: str) -> dict:
    """``name -> (source, compile options, block, feature_major_x)``: the production kernel (feature-major X) with block / register-cap variants, and the row-major X form it replaced."""
    v = {"base_b128": (base_src, (), 128, True)}
    for b in (64, 256, 512):
        v[f"base_b{b}"] = (base_src, (), b, True)
    for r in (24, 40, 48, 64):
        v[f"base_rreg{r}"] = (base_src, (f"--maxrregcount={r}",), 128, True)
    decl = "    int i = blockIdx.x * blockDim.x + threadIdx.x;"
    row = base_src.replace(_X_FEATURE_MAJOR[0], _X_FEATURE_MAJOR[1]).replace(decl, decl + " const float* xi = X + (long long)i * f;", 1)
    assert row != base_src
    v["xrow_b128"] = (row, (), 128, False)
    v["aos_b128"] = (None, (), 128, True)
    v["pre_b128"] = (None, (), 128, True)
    v["pre_b256"] = (None, (), 256, True)
    v["pre_unr4"] = (None, (), 128, True)
    v["pre_rstr"] = (None, (), 128, True)
    v["pre_rstr_unr4"] = (None, (), 128, True)
    v["pre_rstr_shared"] = (None, (), 128, True)
    v["pre_rstr_shared64"] = (None, (), 64, True)
    v["pre_pair"] = (None, (), 128, True)
    v["pre_rreg40"] = (None, ("--maxrregcount=40",), 128, True)
    v["aos_b256"] = (None, (), 256, True)
    return v


def main(argv=None) -> int:
    """Time the variants round-robin and print best / median per variant with the difference to the baseline phi."""
    import cupy as cp

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table import build_tables
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table_gpu import _real, kernel_source

    ap = argparse.ArgumentParser()
    ap.add_argument("rows", type=int)
    ap.add_argument("features", type=int, nargs="?", default=100)
    ap.add_argument("trees", type=int, nargs="?", default=100)
    ap.add_argument("depth", type=int, nargs="?", default=6)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    ens, X = build_ensemble(a.rows, a.features, a.trees, a.depth)
    tab = build_tables(ens, a.rows)
    real = _real()
    n, f = X.shape
    i32 = np.int32
    X_d, XT_d = cp.asarray(X), cp.asarray(np.ascontiguousarray(X.T))
    consts = (
        cp.asarray(ens.children_left), cp.asarray(ens.children_right), cp.asarray(ens.children_default), cp.asarray(ens.features), cp.asarray(ens.thresholds),
        cp.asarray(tab["node_slot"].astype(i32)), cp.asarray(tab["leaf_of_node"].astype(i32)), cp.asarray(tab["leaf_ud"].astype(i32)), cp.asarray(tab["leaf_feat_off"].astype(i32)),
        cp.asarray(tab["leaf_feat"].astype(i32)), cp.asarray(tab["leaf_table_off"].astype(i32)), cp.asarray(tab["table"].astype(real)), cp.asarray(ens.tree_roots), i32(ens.tree_roots.shape[0]),
    )
    variants = _variants(kernel_source(ens.max_depth, real))
    if a.only:
        keep = set(a.only.split(",")) | {"base_b128"}
        variants = {k: v for k, v in variants.items() if k in keep}
    aos_src = _AOS_SRC.replace("{REAL}", "float" if real == np.float32 else "double").replace("MAXSP", str(kernel_source(ens.max_depth, real).split("#define MAXSP ")[1].split()[0]))
    nodes_d = cp.asarray(_pack_nodes(ens, tab))
    leaf_feat_d, table_d, roots_d = cp.asarray(tab["leaf_feat"].astype(i32)), cp.asarray(tab["table"].astype(real)), cp.asarray(ens.tree_roots)
    pre_src = _PRE_SRC.replace("{REAL}", "float" if real == np.float32 else "double").replace("MAXD", str(ens.max_depth + 2))
    pre_rec = _pack_preorder(ens, tab)
    pre_d = cp.asarray(pre_rec)
    ud_a, fo_a, to_a = tab["leaf_ud"], tab["leaf_feat_off"], tab["leaf_table_off"]
    pair_h = np.zeros(int(to_a[-1] + (1 << ud_a[-1]) * ud_a[-1]), dtype=[("v", real), ("f", np.int32)])
    pair_h["v"] = tab["table"].astype(real)
    for L in range(ud_a.shape[0]):
        if ud_a[L]:
            pair_h["f"][to_a[L] : to_a[L] + (1 << ud_a[L]) * ud_a[L]] = np.tile(tab["leaf_feat"][fo_a[L] : fo_a[L] + ud_a[L]].astype(i32), 1 << ud_a[L])
    pair_d = cp.asarray(pair_h.view(np.uint8))
    rt = "float" if real == np.float32 else "double"

    def pre_variant(k):
        src = pre_src
        if "unr4" in k:
            src = src.replace("    for (int p = 0; p < n_rec; p++){", "    #pragma unroll 4\n    for (int p = 0; p < n_rec; p++){")
        if "rstr" in k or "pair" in k:
            src = src.replace("const float* X,", "const float* __restrict__ X,").replace("const int4* rec", "const int4* __restrict__ rec").replace("const {REAL}* table".replace("{REAL}", "float" if real == np.float32 else "double"), "const " + ("float" if real == np.float32 else "double") + "* __restrict__ table").replace("const int* leaf_feat", "const int* __restrict__ leaf_feat")
        if "shared" in k:
            md = str(ens.max_depth + 2)
            src = src.replace("    int mL[" + md + "];", "    extern __shared__ int smem[]; int* mL = smem;").replace("    int mR[" + md + "];", "    int* mR = smem + " + md + " * blockDim.x;")
            src = src.replace("mL[depth - 1]", "mL[(depth - 1) * blockDim.x + threadIdx.x]").replace("mR[depth - 1]", "mR[(depth - 1) * blockDim.x + threadIdx.x]")
            src = src.replace("mL[depth] =", "mL[depth * blockDim.x + threadIdx.x] =").replace("mR[depth] =", "mR[depth * blockDim.x + threadIdx.x] =")
        if "pair" in k:
            src = src.replace("const " + rt + "* __restrict__ table", "const PairT* __restrict__ table").replace("const int* __restrict__ leaf_feat, ", "").replace(
                "phi[(long long)leaf_feat[r.z + j] * n + i] += table[base + j];", "{ const PairT e = table[base + j]; phi[(long long)e.f * n + i] += e.v; }"
            )
            src = "struct PairT { " + rt + " v; int f; };\n" + src
        return src

    kernels = {k: cp.RawKernel(pre_variant(k) if k.startswith("pre") else (aos_src if src is None else src), "treeshap_table", options=opts) for k, (src, opts, _b, _x) in variants.items()}
    for k, kern in kernels.items():
        kern.compile()
        print(f"{k:14s} regs {kern.num_regs:3d} local {kern.local_size_bytes:4d} B")

    def launch(name):
        _src, _opts, block, fmajor = variants[name]
        phi = cp.zeros((f, n), dtype=real)
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        if name == "pre_pair":
            args = (XT_d, i32(n), i32(f), pre_d, i32(pre_rec.shape[0]), pair_d, phi)
        elif name.startswith("pre"):
            args = (XT_d, i32(n), i32(f), pre_d, i32(pre_rec.shape[0]), table_d, leaf_feat_d, phi)
        elif name.startswith("aos"):
            args = (XT_d, i32(n), i32(f), nodes_d, table_d, leaf_feat_d, roots_d, i32(ens.tree_roots.shape[0]), phi)
        else:
            args = (XT_d if fmajor else X_d, i32(n), i32(f), *consts, phi)
        kernels[name](((n + block - 1) // block,), (block,), args, shared_mem=8 * (ens.max_depth + 2) * block if "shared" in name else 0)
        e1.record()
        e1.synchronize()
        return cp.cuda.get_elapsed_time(e0, e1), phi

    ref = launch("base_b128")[1]
    print("hardware before:", line(snapshot()))
    times = {k: [] for k in variants}
    diffs = {}
    for _ in range(a.rounds):
        for k in variants:
            t, phi = launch(k)
            times[k].append(t)
            diffs[k] = float(cp.abs(phi - ref).max())
    print("hardware after: ", line(snapshot()))
    base = min(times["base_b128"])
    for k, v in times.items():
        print(f"{k:14s} best {min(v):8.2f} ms  median {sorted(v)[len(v) // 2]:8.2f} ms  x{base / min(v):5.3f}  max|dphi| {diffs[k]:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    kernels = {k: cp.RawKernel(src, "treeshap_table", options=opts) for k, (src, opts, _b, _x) in variants.items()}
    for k, kern in kernels.items():
        kern.compile()
        print(f"{k:14s} regs {kern.num_regs:3d} local {kern.local_size_bytes:4d} B")

    def launch(name):
        _src, _opts, block, fmajor = variants[name]
        phi = cp.zeros((f, n), dtype=real)
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        kernels[name](((n + block - 1) // block,), (block,), (XT_d if fmajor else X_d, i32(n), i32(f), *consts, phi))
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

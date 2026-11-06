"""Same-process A/B of pair_mi_shared_fused: threads per block, and a variant with four independent element loads in flight per thread.

100000 rows x 120 columns, 3000 pairs of 12-bin columns against a 2-class target (joint cells 144 x 2), the kind of block the pair gate sends. The variant keeps the histogram
updates in the same per-thread order within each chunk of four, so the counts (integers) and therefore the MI are equal.
"""
import time
import warnings

import cupy as cp
import numpy as np

warnings.simplefilter("ignore")
from mlframe.feature_selection.filters import _batch_pair_mi_cuda_shared_fused as M

rng = np.random.default_rng(0)
n, nf, n_pairs, nb = 100_000, 120, 3000, 12
nbins = np.full(nf, nb, dtype=np.int32)
factors = rng.integers(0, nb, (n, nf)).astype(np.int32)
y = rng.integers(0, 2, n).astype(np.int32)
freqs = (np.bincount(y, minlength=2) / n).astype(np.float64)
pa = rng.integers(0, nf, n_pairs).astype(np.int64)
pb = ((pa + rng.integers(1, nf, n_pairs)) % nf).astype(np.int64)

OLD = """    for (long long i = tid; i < n_samples; i += nt) {
        int va = factors_data[i * (long long)n_features + a];
        int vb = factors_data[i * (long long)n_features + b];
        int cls_x = va * nb_b + vb;
        int cls_y = classes_y[i];
        atomicAdd(&sh[cls_x * n_classes_y + cls_y], 1);
        atomicAdd(&sh[n_joint_cells + cls_x], 1);
    }"""
NEW = """    for (long long i0 = tid; i0 < n_samples; i0 += 4LL * nt) {
        int va4[4], vb4[4], cy4[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const long long i = i0 + (long long)u * nt;
            const bool ok = i < n_samples;
            va4[u] = ok ? factors_data[i * (long long)n_features + a] : -1;
            vb4[u] = ok ? factors_data[i * (long long)n_features + b] : -1;
            cy4[u] = ok ? classes_y[i] : 0;
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (va4[u] >= 0) {
                int cls_x = va4[u] * nb_b + vb4[u];
                atomicAdd(&sh[cls_x * n_classes_y + cy4[u]], 1);
                atomicAdd(&sh[n_joint_cells + cls_x], 1);
            }
        }
    }"""
assert M._PAIR_MI_SHARED_FUSED_SRC.count(OLD) == 1
unrolled = cp.RawKernel(M._PAIR_MI_SHARED_FUSED_SRC.replace(OLD, NEW), "pair_mi_shared_fused")
plain = M._get_shared_fused_kernel()


def run(kernel, threads):
    """One timed call (it includes the host-to-device upload of the factor matrix, as in the fit) with the given kernel and block width."""
    M._SHARED_FUSED_KERNEL = kernel
    t = time.perf_counter()
    out = M.batch_pair_mi_cuda_shared_fused(factors, pa, pb, nbins, y, freqs, threads_per_block=threads)
    return time.perf_counter() - t, out


configs = [("plain", plain, 128), ("plain", plain, 256), ("plain", plain, 512), ("unroll4", unrolled, 128), ("unroll4", unrolled, 256)]
for _, k, th in configs:
    run(k, th)
times = {i: [] for i in range(len(configs))}
ref = None
for _ in range(5):
    for i, (_, k, th) in enumerate(configs):
        d, out = run(k, th)
        times[i].append(d)
        ref = out if ref is None else ref
        assert np.allclose(out, ref, rtol=0, atol=1e-12), "MI differs"
base = np.median(times[0])
for i, (name, _, th) in enumerate(configs):
    print(f"{name:8s} {th:4d} threads: median {np.median(times[i]) * 1e3:7.1f} ms   {base / np.median(times[i]):.2f}x of plain/128")

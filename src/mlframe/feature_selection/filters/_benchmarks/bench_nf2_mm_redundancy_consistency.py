"""Bench/demonstration for critique N-F2 part 1: the Fleuret redundancy CMI now carries the same Miller-Madow bias
correction as the MM relevance. Shows the plug-in redundancy over-estimation scaling with cardinality (which biased
the relevance-minus-redundancy objective against high-card candidates), and how the MM correction removes it.
Run: python -m mlframe.feature_selection.filters._benchmarks.bench_nf2_mm_redundancy_consistency
"""
import numpy as np
from mlframe.feature_selection.filters.info_theory._entropy_kernels import conditional_mi, _cmi_miller_madow_bias

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 5000
    y = rng.integers(0, 2, n); z = rng.integers(0, 6, n)
    print("Kx  plugin-CMI  MM-CMI   MM-bias  (plug-in redundancy over-estimation grows with cardinality)")
    for kx in (4, 8, 16, 32, 64):
        x = rng.integers(0, kx, n)
        fd = np.column_stack([x, y, z]).astype(np.int32); nb = np.array([kx, 2, 6], dtype=np.int64)
        xi, yi, zi = np.array([0], np.int64), np.array([1], np.int64), np.array([2], np.int64)
        plug = conditional_mi(fd, xi, yi, zi, None, nb, dtype=np.int32, use_mm=False)
        mm = conditional_mi(fd, xi, yi, zi, None, nb, dtype=np.int32, use_mm=True)
        bias = _cmi_miller_madow_bias(fd, xi, yi, zi, nb, np.int32)
        print(f"{kx:>3} {plug:>10.5f} {mm:>8.5f} {bias:>8.5f}")

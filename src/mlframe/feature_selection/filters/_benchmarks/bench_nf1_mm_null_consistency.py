"""Bench/demonstration for critique N-F1: with Miller-Madow active, the permutation null must use MM too.

Shows the debiasing inconsistency the fix removes: with an MM observed relevance, a PLUG-IN null mean (the pre-fix
null) yields a different max(0, observed - null_mean) than the consistent MM null -- the plug-in null over/under-
subtracts because the two legs carry different bias magnitudes. Run:
  python -m mlframe.feature_selection.filters._benchmarks.bench_nf1_mm_null_consistency
"""
import numpy as np
from mlframe.feature_selection.filters.permutation import parallel_mi_prange_with_null
from mlframe.feature_selection.filters.info_theory._class_mi_kernels import compute_relevance_score

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for kx, strength in [(6, 0.15), (10, 0.10)]:
        n = 3000
        cx = rng.integers(0, kx, n).astype(np.int32)
        y = np.where(rng.random(n) < strength, cx % 2, rng.integers(0, 2, n)).astype(np.int32)
        fx = np.bincount(cx, minlength=kx).astype(np.float64) / n
        fy = np.bincount(y, minlength=2).astype(np.float64) / n
        obs_mm = compute_relevance_score(False, cx, fx, y, fy, use_mm=True)
        _, nc, s_plugin = parallel_mi_prange_with_null(cx, fx, y, fy, 64, obs_mm, np.uint64(1), np.int32, False, False)
        _, _, s_mm = parallel_mi_prange_with_null(cx, fx, y, fy, 64, obs_mm, np.uint64(1), np.int32, False, True)
        null_plugin, null_mm = s_plugin / nc, s_mm / nc
        print(f"kx={kx} strength={strength}: observed_mm={obs_mm:.5f}  "
              f"plugin-null-mean={null_plugin:.5f} (debiased={max(0,obs_mm-null_plugin):.5f})  "
              f"MM-null-mean={null_mm:.5f} (debiased={max(0,obs_mm-null_mm):.5f})  "
              f"-- pre-fix used the plugin null against an MM observed")

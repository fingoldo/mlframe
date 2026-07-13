"""iter113 bench: float-categorical canonical-token mapping in _categorical_to_string_array @10M.

OLD: np.unique(arr, return_inverse=True) argsorts the full 10M float array (O(n log n)) only to derive inverse codes.
NEW: pd.factorize(arr, sort=False) is hash-based O(n); tokens are computed per UNIQUE value then gathered, so the
unique ORDER is irrelevant and the output is bit-identical.

Measured (py3.14): K=500 8.61x (2.032->0.236s); K=50000 4.35x (3.259->0.749s), identical=True.
Separate-process e2e (fit+transform woe, cv=3, @10M): 5.602s -> 3.427s = 1.63x, checksum byte-identical (0.711767).
"""
import numpy as np, pandas as pd, time
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 10_000_000
    for K in (500, 50000):
        arr = rng.integers(0, K, size=N).astype(np.float64)
        # OLD: np.unique
        best_old = 1e9
        for _ in range(3):
            t = time.perf_counter()
            uniq, inv = np.unique(arr, return_inverse=True)
            toks = np.array([str(int(u)) if (np.isfinite(u) and u == int(u)) else repr(float(u)) for u in uniq], dtype=object)
            out_old = toks[np.asarray(inv).reshape(-1)]
            best_old = min(best_old, time.perf_counter() - t)
        # NEW: pd.factorize sort=False
        best_new = 1e9
        for _ in range(3):
            t = time.perf_counter()
            codes, uniqf = pd.factorize(arr, sort=False)
            toks2 = np.array([str(int(u)) if (np.isfinite(u) and u == int(u)) else repr(float(u)) for u in uniqf], dtype=object)
            out_new = toks2[codes]
            best_new = min(best_new, time.perf_counter() - t)
        print(f"K={K}: old(np.unique)={best_old:.3f}s new(factorize)={best_new:.3f}s speedup={best_old/best_new:.2f}x identical={np.array_equal(out_old,out_new)}")

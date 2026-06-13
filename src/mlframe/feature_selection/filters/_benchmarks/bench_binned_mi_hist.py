"""Bench + bit-identity check for the joint-histogram rewrite of _wavelet_basis_fe._binned_mi.

The prior implementation built the contingency table with an O(|fa|*|yb|*n) double loop
(one O(n) boolean mask per cell); the rewrite uses a single bincount over the dense joint
code. Bit-identical by construction (same plug-in counts, same row-major nonzero sum order).

Run: PYTHONPATH=<worktree>/src python bench_binned_mi_hist.py
"""
import time
import numpy as np


def _binned_mi_legacy(feat, y, nbins=10):
    feat = np.asarray(feat, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    n = feat.size
    if n == 0 or n != y.size:
        return 0.0
    uniq_f = np.unique(feat)
    if uniq_f.size <= nbins:
        fb = np.searchsorted(uniq_f, feat)
    else:
        edges = np.quantile(feat, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
        fb = np.digitize(feat, edges)
    if np.issubdtype(y.dtype, np.integer) and np.unique(y).size <= 20:
        yb = y.astype(np.int64)
    elif np.unique(y).size <= 20:
        uy = np.unique(y)
        yb = np.searchsorted(uy, y)
    else:
        edges_y = np.quantile(y, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
        yb = np.digitize(y, edges_y)
    mi = 0.0
    fa_vals = np.unique(fb)
    yb_vals = np.unique(yb)
    for a in fa_vals:
        pa = np.mean(fb == a)
        if pa <= 0:
            continue
        mask_a = fb == a
        for b in yb_vals:
            pab = np.mean(mask_a & (yb == b))
            if pab > 0:
                pb = np.mean(yb == b)
                mi += pab * np.log(pab / (pa * pb))
    return float(max(mi, 0.0))


from mlframe.feature_selection.filters._wavelet_basis_fe import _binned_mi as _binned_mi_new

rng = np.random.default_rng(0)
max_abs = 0.0
configs = []
for _ in range(400):
    n = int(rng.integers(60, 2500))
    # Haar-leg-like ternary feature OR continuous; both code paths
    if rng.random() < 0.5:
        feat = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.3, 0.4, 0.3])
    else:
        feat = rng.normal(size=n)
    if rng.random() < 0.6:
        y = rng.integers(0, int(rng.integers(2, 8)), size=n)  # discrete classes
    else:
        y = rng.normal(size=n)  # continuous -> quantile binned
    a = _binned_mi_legacy(feat, y)
    b = _binned_mi_new(feat, y)
    max_abs = max(max_abs, abs(a - b))
    configs.append((feat, y))

print(f"bit-identity max abs diff over {len(configs)} configs: {max_abs:.3e}")

# warm
for feat, y in configs[:5]:
    _binned_mi_legacy(feat, y); _binned_mi_new(feat, y)

def timeit(fn):
    best = 1e9
    for _ in range(5):
        t = time.perf_counter()
        for feat, y in configs:
            fn(feat, y)
        best = min(best, time.perf_counter() - t)
    return best

tl = timeit(_binned_mi_legacy)
tn = timeit(_binned_mi_new)
print(f"legacy: {tl*1e3:.2f} ms / {len(configs)} calls")
print(f"new   : {tn*1e3:.2f} ms / {len(configs)} calls")
print(f"speedup: {tl/tn:.2f}x")

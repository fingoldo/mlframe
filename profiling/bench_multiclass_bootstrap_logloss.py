"""Bench + correctness check: multiclass extension of _vectorized_bootstrap_logloss_samples (iter120).

iter118 vectorised the 1-D binary and 2-D multilabel paths but the multiclass
case (y shape (n,) int + p shape (n, K) softmax) fell through to the sklearn
log_loss loop -- 1500 ms at n=1500, R=1000 on the c0074 fuzz combo.
iter120 added a third branch keyed off y.ndim==1 and p.ndim==2 that uses
fancy-indexing on the true-class column.

Bench at n=1500, K=4, R=1000:

    new vectorised : 12 ms
    sklearn loop   : 1500 ms     (~120x)

Output matches sklearn at fp64 epsilon (max abs diff 2.22e-16).
Run: ``python profiling/bench_multiclass_bootstrap_logloss.py``
"""

import time
import sys
import numpy as np
sys.path.insert(0, 'src')

from sklearn.metrics import log_loss
from mlframe.training.dummy_baselines import _vectorized_bootstrap_logloss_samples

rng = np.random.default_rng(0)
n, K = 1500, 4
y = rng.integers(0, K, size=n).astype(np.int64)
# Random softmax-like probs (rows sum to 1).
logits = rng.standard_normal((n, K))
p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

# Warmup
_ = _vectorized_bootstrap_logloss_samples(y, p, 100, 42)

print("== new vectorised multiclass path ==")
for _ in range(3):
    t = time.perf_counter()
    samples_new = _vectorized_bootstrap_logloss_samples(y, p, 1000, 42)
    print(f'  {(time.perf_counter()-t)*1000:.1f}ms  shape={samples_new.shape}')

# sklearn reference: per-resample log_loss loop
def sklearn_loop(y, p, n_resamples, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(n_resamples, dtype=np.float64)
    n = len(y)
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        out[r] = log_loss(y[idx], p[idx], labels=np.arange(K))
    return out

print()
print("== sklearn reference loop ==")
for _ in range(3):
    t = time.perf_counter()
    samples_ref = sklearn_loop(y, p, 1000, 42)
    print(f'  {(time.perf_counter()-t)*1000:.1f}ms')

# Identity: both should match per-resample (same seed = same idx)
print()
print(f'max abs diff (new vs sklearn): {np.abs(samples_new - samples_ref).max():.4e}')
print(f'mean abs diff:                 {np.abs(samples_new - samples_ref).mean():.4e}')

# Also test the existing paths still work
print()
print("== regression: 1-D binary path still works ==")
y_bin = rng.integers(0, 2, size=n).astype(np.float64)
p_bin = np.clip(rng.random(n), 1e-3, 1 - 1e-3)
out_bin = _vectorized_bootstrap_logloss_samples(y_bin, p_bin, 200, 0)
print(f'  binary shape={out_bin.shape}, mean={out_bin.mean():.4f}')

print()
print("== regression: 2-D multilabel path still works ==")
y_mlb = rng.integers(0, 2, size=(n, 3)).astype(np.float64)
p_mlb = np.clip(rng.random((n, 3)), 1e-3, 1 - 1e-3)
out_mlb = _vectorized_bootstrap_logloss_samples(y_mlb, p_mlb, 200, 0)
print(f'  multilabel shape={out_mlb.shape}, mean={out_mlb.mean():.4f}')

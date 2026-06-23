"""CPX34: bench _conditional_permutation_importance per-feature conditioning-set build.

Old path allocated a fresh (n, p-1) array via ``np.delete(X_arr, j, axis=1)`` per feature
=> O(n*p^2) allocation/copy. New path refills one reused (n, p-1) C-contiguous scratch via
two contiguous block-copies (bit-identical to np.delete). Warm, best-of-N wall A/B.

Run: CUDA_VISIBLE_DEVICES="" python src/mlframe/feature_selection/wrappers/_benchmarks/bench_cpx34_helpers_importance.py
"""
import time
import numpy as np
from sklearn.linear_model import Ridge

from mlframe.feature_selection.wrappers._helpers_importance import _conditional_permutation_importance


def _make(n=20000, p=100, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    # correlate a few cols so conditioning trees are non-trivial
    X[:, 1] += 0.7 * X[:, 0]
    X[:, 2] += 0.5 * X[:, 0]
    y = X[:, 0] * 1.5 + X[:, 3] - 0.8 * X[:, 5] + rng.standard_normal(n) * 0.3
    return X, y


def main():
    X, y = _make()
    model = Ridge().fit(X, y)
    # warm
    _conditional_permutation_importance(model, X, y, n_repeats=2, random_state=0)

    N = 5
    best = float("inf")
    for _ in range(N):
        t = time.perf_counter()
        imp = _conditional_permutation_importance(model, X, y, n_repeats=2, random_state=0)
        best = min(best, time.perf_counter() - t)
    print(f"shape=({X.shape[0]},{X.shape[1]}) best-of-{N}: {best*1000:.1f} ms  imp[:3]={imp[:3]}")


if __name__ == "__main__":
    main()

"""A/B bench for the prange-parallel MDLP permutation-null kernel (2026-07-31).

``_permutation_null_gain_njit`` (called from ``_split_significant`` at every MDLP tree node
reaching the permutation fallback) was the #1 mlframe hotspot by tottime in a multiclass 2M-row
cProfile (62.8s tottime / 3176 calls, ``profile_one_combo.py --combo c0021_f0cef153 --rows
2000000``) -- the module's own docstring already names this "20-80x permutation-fallback cost" as
the confirmed driver and flags a batched-parallel rewrite as the unimplemented next step.

Parallelising the ``n_permutations`` loop via ``numba.prange`` changes the exact RNG draw sequence
(each permutation now seeds independently as ``seed + p`` instead of one continuous stream) -- NOT
bit-identical to the old sequential kernel, but still a valid uniformly-random permutation per
draw, so the null distribution -- and every accept/reject decision downstream -- is
SELECTION-EQUIVALENT. This bench confirms (a) the accept-rate on pure-noise and real-signal
synthetic scenarios matches within Monte-Carlo noise across many seeds, and (b) the real wall-time
win, at a representative node shape.

Usage:
    python profiling/bench_mdlp_permutation_null_parallel.py
"""

from __future__ import annotations

import time

import numba
import numpy as np

from mlframe.feature_selection.filters.supervised_binning import _mdlp_best_split_njit
from mlframe.feature_selection.filters._mdlp_validated_split import _permutation_null_gain_njit


@numba.njit(nogil=True, cache=False)
def _permutation_null_gain_seq_baseline(x_sorted, y_compact, n_classes, min_split_size, n_permutations, seed):
    """Pre-optimization sequential baseline (one continuous np.random stream), kept here only as the A/B reference."""
    np.random.seed(seed)
    n = y_compact.shape[0]
    null_gains = np.empty(n_permutations, dtype=np.float64)
    y_perm = y_compact.copy()
    for p in range(n_permutations):
        for i in range(n - 1, 0, -1):
            j = int(np.random.randint(0, i + 1))
            tmp = y_perm[i]
            y_perm[i] = y_perm[j]
            y_perm[j] = tmp
        _, gain, _, _, _ = _mdlp_best_split_njit(x_sorted, y_perm, n_classes, min_split_size)
        null_gains[p] = gain if gain > 0.0 else 0.0
    return null_gains


def _accept_rate(x_sorted, y_compact, n_classes, min_split_size, n_permutations, kernel, n_trials, alpha=0.05):
    """Fraction of ``n_trials`` independent significance tests (fresh seed each trial) that accept the
    OBSERVED best split as significant, using ``kernel`` to build the null distribution."""
    _, observed_gain, _, _, _ = _mdlp_best_split_njit(x_sorted, y_compact, n_classes, min_split_size)
    if observed_gain <= 0.0:
        return 0.0
    accepts = 0
    for trial in range(n_trials):
        null_gains = np.sort(kernel(x_sorted, y_compact, n_classes, min_split_size, n_permutations, trial * 97 + 1))
        q_idx = max(0, min(int(np.ceil((1.0 - alpha) * n_permutations)) - 1, n_permutations - 1))
        threshold = float(null_gains[q_idx])
        if observed_gain > threshold:
            accepts += 1
    return accepts / n_trials


def _check_selection_equivalence(label, n, n_classes, seed, n_permutations=30, min_split_size=5, n_trials=300):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.random(n))
    y = rng.integers(0, n_classes, size=n).astype(np.int64)

    r_seq = _accept_rate(x, y, n_classes, min_split_size, n_permutations, _permutation_null_gain_seq_baseline, n_trials)
    r_par = _accept_rate(x, y, n_classes, min_split_size, n_permutations, _permutation_null_gain_njit, n_trials)
    # Binomial Monte-Carlo tolerance at n_trials=300: SE ~= sqrt(0.5*0.5/300) ~= 0.029; 3*SE ~= 0.087.
    tol = 0.10
    ok = abs(r_seq - r_par) < tol
    print(f"{label}: seq accept-rate={r_seq:.3f} par accept-rate={r_par:.3f} diff={abs(r_seq - r_par):.3f} " f"({'OK' if ok else 'MISMATCH'})")
    return ok


def main():
    print("=== Selection-equivalence (accept-rate parity, pure noise + real signal) ===")
    ok1 = _check_selection_equivalence("pure noise, n=200, k=4", n=200, n_classes=4, seed=1)
    ok2 = _check_selection_equivalence("pure noise, n=60, k=3", n=60, n_classes=3, seed=2)

    rng = np.random.default_rng(3)
    n = 200
    x = np.sort(rng.random(n))
    y = np.where(x < 0.5, 0, 1).astype(np.int64)  # a genuine, strong signal
    r_seq = _accept_rate(x, y, 2, 5, 30, _permutation_null_gain_seq_baseline, 200)
    r_par = _accept_rate(x, y, 2, 5, 30, _permutation_null_gain_njit, 200)
    ok3 = r_seq > 0.9 and r_par > 0.9  # a strong real signal must be accepted almost always by both
    print(f"real signal, n=200: seq accept-rate={r_seq:.3f} par accept-rate={r_par:.3f} " f"({'OK' if ok3 else 'MISMATCH'})")

    print("\n=== Wall-time (representative node shape: n=200, n_permutations=30) ===")
    rng = np.random.default_rng(0)
    n = 200
    x = np.sort(rng.random(n))
    y = rng.integers(0, 4, size=n).astype(np.int64)
    _permutation_null_gain_seq_baseline(x, y, 4, 5, 30, 0)
    _permutation_null_gain_njit(x, y, 4, 5, 30, 0)  # warm both kernels

    n_calls = 2000
    t0 = time.perf_counter()
    for i in range(n_calls):
        _permutation_null_gain_seq_baseline(x, y, 4, 5, 30, i)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(n_calls):
        _permutation_null_gain_njit(x, y, 4, 5, 30, i)
    t_par = time.perf_counter() - t0

    print(f"seq: {t_seq:.4f}s  par: {t_par:.4f}s  speedup: {t_seq / t_par:.2f}x  ({n_calls} calls)")

    assert ok1 and ok2 and ok3, "parallel kernel's accept/reject decisions diverged from the sequential baseline"


if __name__ == "__main__":
    main()

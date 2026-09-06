"""iter78 @200k: fused single-pass merge for wasserstein_1d + ks_distribution_distance.

Replaces numpy ``concatenate+sort`` of the merged support plus two ``searchsorted`` scans with one
O(na+nb) njit pointer-merge over the two pre-sorted arrays (one sort per input only). Bit-identical:
exact on ties/discrete (positional ties = searchsorted-right), ~1e-15 FP-order on continuous.

Measured (this box, py3.14, n=200k, best-of-N):
  wasserstein_1d:        old 1.143s -> new 0.254s  (40 iters)  ~4.5x isolated
  ks_distribution_dist:  old 1.953s -> new 0.325s  (40 iters)  ~6.0x isolated
  e2e (8 W1 + 8 KS @200k, separate-process A/B vs HEAD): 0.454s -> 0.114s = 3.97x, checksum identical to 12 dp.

Run:  python -m mlframe.metrics._benchmarks.bench_drift_fused_merge_iter78
"""
from __future__ import annotations
import time
import numpy as np
from mlframe.metrics._drift import wasserstein_1d, ks_distribution_distance


def _old_w1(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    # Production filters non-finite values and returns nan on an empty side (_drift.py). Without the same
    # two steps here the identity assertion in main() guards nothing on the non-finite path -- and the two
    # sides genuinely disagree there: on a=[1,2,nan,4], b=[1,2,3,4] the unfiltered form gives nan for W1 and
    # 0.25 for KS against production's 0.3333333333333333 and 0.16666666666666663.
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    all_values = np.concatenate((a, b)); all_values.sort(kind="quicksort")
    deltas = np.diff(all_values)
    cdf_a = np.searchsorted(np.sort(a), all_values[:-1], side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), all_values[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * deltas))


def _old_ks(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    # Production filters non-finite values and returns nan on an empty side (_drift.py). Without the same
    # two steps here the identity assertion in main() guards nothing on the non-finite path -- and the two
    # sides genuinely disagree there: on a=[1,2,nan,4], b=[1,2,3,4] the unfiltered form gives nan for W1 and
    # 0.25 for KS against production's 0.3333333333333333 and 0.16666666666666663.
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_s = np.sort(a); b_s = np.sort(b)
    all_values = np.concatenate((a_s, b_s)); all_values.sort()
    cdf_a = np.searchsorted(a_s, all_values, side="right") / a_s.size
    cdf_b = np.searchsorted(b_s, all_values, side="right") / b_s.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def main():
    rng = np.random.default_rng(0)

    # The non-finite path the identity assertion below could not reach while the frozen copies lacked
    # production's filter. Checked once, at a small size, before the timing shapes.
    nan_a = np.array([1.0, 2.0, np.nan, 4.0]); nan_b = np.array([1.0, 2.0, 3.0, 4.0])
    for name, new_f, old_f in (("W1", wasserstein_1d, _old_w1), ("KS", ks_distribution_distance, _old_ks)):
        got, ref = new_f(nan_a, nan_b), old_f(nan_a, nan_b)
        assert abs(got - ref) < 1e-10, f"{name} diverges on non-finite input: {got} vs {ref}"  # nosec B101 - internal invariant check in src/mlframe/metrics/_benchmarks, not reachable with untrusted input
    print(f"non-finite identity OK: W1={wasserstein_1d(nan_a, nan_b):.12g} KS={ks_distribution_distance(nan_a, nan_b):.12g}")

    for n in (50000, 200000, 1000000):
        a = rng.random(n); b = rng.random(n) + 0.1
        wasserstein_1d(a, b); ks_distribution_distance(a, b)
        for name, new_f, old_f in (("W1", wasserstein_1d, _old_w1), ("KS", ks_distribution_distance, _old_ks)):
            assert abs(new_f(a, b) - old_f(a, b)) < 1e-10  # nosec B101 - internal invariant check in src/mlframe/metrics/_benchmarks, not reachable with untrusted input
            t = time.perf_counter()
            for _ in range(30):
                old_f(a, b)
            to = time.perf_counter() - t
            t = time.perf_counter()
            for _ in range(30):
                new_f(a, b)
            tn = time.perf_counter() - t
            print(f"n={n} {name}: old={to:.4f} new={tn:.4f} speedup={to/tn:.2f}x")


if __name__ == "__main__":
    main()

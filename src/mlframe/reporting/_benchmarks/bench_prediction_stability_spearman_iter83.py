"""iter83 bench: prediction-stability uncertainty-calibration Spearman at N=200k.

The OLD ``_spearman`` ran two ``_rankdata`` passes, each a full ``np.argsort`` plus a pure-Python while-loop to average tied
ranks -- 200k Python iterations on the sorted array. Routing the whole-vector rank+Pearson through the existing njit batched
kernel (``spearmanr_batched_numba`` on a 1-row batch) does the argsort + tie-average in machine code, same average-rank
convention, bit-identical.

Run: ``python -m mlframe.reporting._benchmarks.bench_prediction_stability_spearman_iter83``

Measured (Win11 / py3.14 store / CUDA off, cupy blocked to dodge cold-import segfault):
  isolated _spearman @200k: OLD 90.4 ms -> njit 1-row 42.5 ms = ~2.1x (bit-identical, ties included).
  e2e _uncertainty_calibration @200k (nbins=20): OLD 110.7 ms -> NEW 56.8 ms = ~1.95x, Spearman + mid/err bit-identical.
"""

import sys
import time

import numpy as np

sys.modules.setdefault("cupy", None)  # block cupy: cold import native-segfaults under contention on py3.14

import scipy.stats  # noqa: E402,F401  (import before mlframe to avoid the cold-import segfault)
import numba  # noqa: E402,F401

from mlframe.reporting.charts import prediction_stability as ps  # noqa: E402


def _bench(fn, r=7):
    ts = []
    for _ in range(r):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000.0, float(np.median(ts)) * 1000.0


def main():
    rng = np.random.default_rng(1)
    n = 200_000
    spread = rng.random(n)
    abs_err = spread * 0.4 + rng.random(n) * 0.6

    ps._uncertainty_calibration(spread, abs_err, nbins=20)  # warm njit
    new_min, new_med = _bench(lambda: ps._uncertainty_calibration(spread, abs_err, nbins=20))
    new_out = ps._uncertainty_calibration(spread, abs_err, nbins=20)

    orig = ps._SPEARMAN_NJIT_MIN_N
    ps._SPEARMAN_NJIT_MIN_N = 10**12
    old_min, old_med = _bench(lambda: ps._uncertainty_calibration(spread, abs_err, nbins=20))
    old_out = ps._uncertainty_calibration(spread, abs_err, nbins=20)
    ps._SPEARMAN_NJIT_MIN_N = orig

    print(f"e2e @200k: OLD {old_min:.3f} ms (med {old_med:.3f}) -> NEW {new_min:.3f} ms (med {new_med:.3f}) = {old_min/new_min:.2f}x")
    print(f"Spearman  OLD {old_out[2]:.17g}  NEW {new_out[2]:.17g}  diff {abs(old_out[2]-new_out[2]):.3e}")
    print(f"mid array_equal={np.array_equal(old_out[0], new_out[0])}  err array_equal={np.array_equal(old_out[1], new_out[1])}")


if __name__ == "__main__":
    main()

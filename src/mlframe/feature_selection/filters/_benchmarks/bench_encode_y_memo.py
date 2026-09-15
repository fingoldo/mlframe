"""Bench: encode_y_for_classif_mi on a continuous target, uncached vs content-hash memo hit.

The FE cascade discretises the same fit target in about seventeen stages. Reports, per n, the uncached cost (np.unique sort + pd.qcut), the
memo-hit cost (content hash + copy of the cached codes), and the total for 17 calls each way. Warm, best-of-N.

Run: PYTHONPATH=src python -m mlframe.feature_selection.filters._benchmarks.bench_encode_y_memo
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters import _y_encoding

_N_STAGES = 17


def _best(fn, reps):
    """Best wall time of ``reps`` calls to ``fn``."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    """Print the uncached vs memo-hit timing table."""
    print(f"{'n':>10} {'uncached_ms':>12} {'hit_ms':>9} {'x_per_call':>11} {'17-stage_uncached_s':>20} {'17-stage_memo_s':>16}")
    for n in (10_000, 100_000, 1_000_000, 4_000_000):
        y = np.random.default_rng(0).normal(size=n)
        reps = 7 if n <= 1_000_000 else 3
        _y_encoding._encode_y_uncached(y)
        t_unc = _best(lambda: _y_encoding._encode_y_uncached(y), reps)
        _y_encoding._clear_encode_cache()
        _y_encoding.encode_y_for_classif_mi(y)
        t_hit = _best(lambda: _y_encoding.encode_y_for_classif_mi(y), reps)
        np.testing.assert_array_equal(_y_encoding.encode_y_for_classif_mi(y), _y_encoding._encode_y_uncached(y))
        print(
            f"{n:>10} {t_unc * 1e3:>12.2f} {t_hit * 1e3:>9.2f} {t_unc / t_hit:>11.1f} {t_unc * _N_STAGES:>20.3f} {t_unc + t_hit * (_N_STAGES - 1):>16.3f}"
        )


if __name__ == "__main__":
    main()

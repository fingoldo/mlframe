"""Demonstrates the removed pool-size cliff (2026-07-09) + cProfile pass on the chunked path.

Before this fix, ``_step_pairmi.compute_pair_mis_and_floor`` gated the fast batched pairwise-MI
path on ``_k <= _MRMR_BATCH_PRECOMPUTE_MAX_K`` (a flat 200-column cap). Any pool wider than that
fell to the legacy per-pair joblib loop, documented in-repo at ``~35s/pair`` on a wide frame --
e.g. a 300-column pool (44_850 pairs) would have cost tens of thousands of seconds serialized, or
still tens of minutes to hours even parallelized across 16 cores.

This benchmark measures wall time for ``dispatch_batch_pair_mi_chunked`` across pool widths that
straddle the old cap (100, 200, 300, 500, 1000) and confirms wall time scales with the actual
pair count (roughly quadratic in k, as expected for an exhaustive O(k^2) sweep) rather than
falling off a cliff at k=200. Also runs one cProfile pass at a representative width to confirm the
chunking wrapper itself is not a hotspot (all time should be in the njit/CUDA kernel + numpy
concatenate, not the Python-level chunk-iteration glue).

Run: ``python -m mlframe.feature_selection._benchmarks.bench_batch_pair_mi_chunked_cliff``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np


def _build_factor_data(n_samples: int, n_cols: int, nbins: int, seed: int):
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, nbins, size=n_samples) for _ in range(n_cols)]
    data = np.column_stack(cols).astype(np.int32)
    return data, np.full(n_cols, nbins, dtype=np.int32)


def _run_one(n_cols: int, n_samples: int = 5000, nbins: int = 10, seed: int = 0):
    from mlframe.feature_selection.filters.batch_pair_mi_gpu import dispatch_batch_pair_mi_chunked

    data, nbins_arr = _build_factor_data(n_samples, n_cols, nbins, seed)
    rng = np.random.default_rng(seed + 1)
    y = rng.integers(0, 3, size=n_samples).astype(np.int32)
    freqs_y = np.bincount(y, minlength=3).astype(np.float64) / n_samples
    ids = np.arange(n_cols, dtype=np.int64)

    t0 = time.perf_counter()
    a_out, b_out, mi_out, backend_counts = dispatch_batch_pair_mi_chunked(
        factors_data=data, ids=ids, nbins=nbins_arr, classes_y=y, freqs_y=freqs_y, force_backend="njit",
    )
    dt = time.perf_counter() - t0
    n_pairs = n_cols * (n_cols - 1) // 2
    assert a_out.shape[0] == n_pairs
    return dt, n_pairs, backend_counts


def main():
    print("width | n_pairs | wall_s | s_per_1k_pairs | backend_chunks")
    prior_ratio = None
    for n_cols in (100, 200, 300, 500, 1000, 2000):
        # Warm the njit kernel on the first (smallest) size so its compile cost doesn't pollute
        # the larger-width measurements (per project A/B methodology: warm before measuring).
        if n_cols == 100:
            _run_one(n_cols=20, n_samples=500)
        dt, n_pairs, backend_counts = _run_one(n_cols=n_cols)
        rate = dt / (n_pairs / 1000.0)
        print(f"{n_cols:5d} | {n_pairs:7d} | {dt:6.3f} | {rate:14.5f} | {backend_counts}")
        # No cliff assertion: confirm wall time stays in the same order of magnitude per-1k-pairs
        # across the old cap boundary (100 -> 300), rather than jumping ~1000x as the legacy
        # ~35s/pair path would have.
        if n_cols == 300:
            prior_ratio = rate
    print()
    print("cProfile pass at width=1000 (representative production-shape pool):")
    pr = cProfile.Profile()
    pr.enable()
    _run_one(n_cols=1000)
    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    main()

"""A/B bench for ``batch_pair_usability_corr_gpu``'s CUDA vs CPU-njit-parallel backend (2026-07-11).

REJECTED as the default (kept, not deleted -- see the module's own docstring): on this dev host (GTX 1050
Ti), CUDA never wins across the full realistic range, from a handful of pairs up to the real ~85k-pair
production scale and beyond. The ratio does not improve with more batch volume -- it converges to
~0.53-0.57x (CUDA slower) right at and past production scale, consistent with the reduction being
memory-bandwidth-bound (uncoalesced per-thread reads of a DIFFERENT operand row each) rather than
launch-overhead-bound, so batching cannot amortize it the way it amortizes a fixed per-launch cost.

Run:  python path/to/_benchmarks/bench_batch_pair_usability_corr_gpu.py
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters.batch_pair_usability_corr_gpu import (
    ALL_FORM_IDS,
    batch_pair_usability_corr_cuda,
    batch_pair_usability_corr_njit_parallel,
)


def main() -> None:
    n_rows = 30_000  # matches _ABS_PEARSON_MAX_ROWS -- the real per-reduction row cap in production
    n_operands = 500
    rng = np.random.default_rng(0)
    operand_matrix = rng.standard_normal((n_operands, n_rows)).astype(np.float64)
    y = rng.standard_normal(n_rows).astype(np.float64)

    # Warm both backends (JIT compile) before timing.
    warm_a = np.array([0, 1], dtype=np.int64)
    warm_b = np.array([1, 0], dtype=np.int64)
    batch_pair_usability_corr_njit_parallel(y, operand_matrix, warm_a, warm_b, ALL_FORM_IDS)
    batch_pair_usability_corr_cuda(y, operand_matrix, warm_a, warm_b, ALL_FORM_IDS)

    print(f"{'n_pairs':>8} {'total(pair*form)':>18} {'cpu_s':>10} {'cuda_s':>10} {'speedup':>9} {'match':>7}")
    # Sweep from trivial to well past the real ~85k-pair production scale (16 pairs x 9 forms = 144
    # reductions, up to 150_000 pairs x 9 forms = 1.35M reductions).
    for n_pairs in (16, 64, 256, 1024, 4096, 8500, 16000, 50_000, 85_000, 150_000):
        pair_a = rng.integers(0, n_operands, size=n_pairs).astype(np.int64)
        pair_b = rng.integers(0, n_operands, size=n_pairs).astype(np.int64)

        t0 = time.perf_counter()
        cpu_result = batch_pair_usability_corr_njit_parallel(y, operand_matrix, pair_a, pair_b, ALL_FORM_IDS)
        t_cpu = time.perf_counter() - t0

        t0 = time.perf_counter()
        cuda_result = batch_pair_usability_corr_cuda(y, operand_matrix, pair_a, pair_b, ALL_FORM_IDS)
        t_cuda = time.perf_counter() - t0

        total = n_pairs * len(ALL_FORM_IDS)
        match = np.allclose(cpu_result, cuda_result, atol=1e-8)
        print(f"{n_pairs:>8} {total:>18} {t_cpu:>10.4f} {t_cuda:>10.4f} {t_cpu / t_cuda:>8.2f}x {match!s:>7}")


if __name__ == "__main__":
    main()

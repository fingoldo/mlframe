"""A/B: single-launch opt-in-shared-memory pair-MI kernel (``batch_pair_mi_cuda_shared_fused``) vs the
existing row-chunked CUDA kernel (``batch_pair_mi_cuda_row_chunked``), at the wellbore-100k production
shape (n_classes_y=20, max_joint up to 441-528) that forces the static-shared kernel's shape guard to
reject and fall back to row-chunked.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_batch_pair_mi_shared_fused``

Verdict (2026-07-16, isolated separate-process A/B, cupy memory pool freed between backends):
  * n_pairs=20000, clean 3.17GB free VRAM: fused 0.454s vs row-chunked 0.533s (1.17x).
  * n_pairs=85000, clean 3.17GB free VRAM: fused 1.575s vs row-chunked 2.170s (1.38x).
  * Under REALISTIC mid-fit VRAM pressure (simulated via ``_choose_pair_subchunk_rows`` at 200MB free,
    plausible with other resident FE buffers active): row-chunked needs 151 launches for 85000 pairs
    (vs a theoretical 3 at 3.17GB free); the fused kernel needs exactly 1 launch REGARDLESS of free
    VRAM (its per-block shared-memory footprint depends only on max_joint*n_classes_y, never on
    n_pairs), so its advantage GROWS precisely in the fragmentation regime that caused 78-92s of the
    original ~500-585s wellbore-100k fit wall.

An earlier version of the fused kernel reduced the histogram to MI on a single thread per block
(leaving 127/128 threads idle during the log()-heavy reduction phase) and measured ~6x SLOWER than
row-chunked at n_pairs=20000 (35.4s) -- fixed by parallelizing the reduction across the block via a
shared atomicAdd accumulator (see ``_batch_pair_mi_cuda_shared_fused.py``'s kernel source comment).
"""
from __future__ import annotations

import time

import numpy as np


def _build_pair_inputs(n_samples, n_features, n_pairs, n_classes_y, nbins_range, seed):
    """Build a random factors/pair/class fixture for the fused-vs-row-chunked A/B bench."""
    rng = np.random.default_rng(seed)
    nbins = rng.integers(nbins_range[0], nbins_range[1] + 1, n_features).astype(np.int32)
    factors_data = np.column_stack([rng.integers(0, int(nb), n_samples) for nb in nbins]).astype(np.int32)
    classes_y = rng.integers(0, n_classes_y, n_samples).astype(np.int32)
    freqs_y = np.bincount(classes_y, minlength=n_classes_y).astype(np.float64) / n_samples
    pair_a = rng.integers(0, n_features, n_pairs).astype(np.int64)
    pair_b = ((pair_a + rng.integers(1, n_features, n_pairs)) % n_features).astype(np.int64)
    return factors_data, nbins, classes_y, freqs_y, pair_a, pair_b


def _free_pool() -> None:
    """Release cupy's default and pinned memory pools so successive bench runs start from clean VRAM."""
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _bench_one(fn, factors_data, pair_a, pair_b, nbins, classes_y, freqs_y, reps: int = 3) -> float:
    """Warm fn once, then time reps runs (freeing VRAM between each) and return the median wall time."""
    import cupy as cp

    _free_pool()
    fn(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)
    cp.cuda.Stream.null.synchronize()
    _free_pool()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
        _free_pool()
    return float(np.median(times))


def main() -> None:
    """Run the fused-vs-row-chunked A/B at the wellbore-100k production shape and print the speedup."""
    from mlframe.feature_selection.filters._batch_pair_mi_cuda_kernels import batch_pair_mi_cuda_row_chunked
    from mlframe.feature_selection.filters._batch_pair_mi_cuda_shared_fused import batch_pair_mi_cuda_shared_fused

    for n_pairs in (20_000, 85_000):
        factors_data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(
            99_401, 30, n_pairs, 20, (15, 22), 0,
        )
        t_fused = _bench_one(batch_pair_mi_cuda_shared_fused, factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)
        t_chunked = _bench_one(batch_pair_mi_cuda_row_chunked, factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)
        print(f"n_pairs={n_pairs:6d}  fused={t_fused:.3f}s  row_chunked={t_chunked:.3f}s  speedup={t_chunked / t_fused:.2f}x")


if __name__ == "__main__":
    main()

"""Bench: borderline_smote._smote_synthesize_from row-loop vs draw-then-vectorise gather+lerp.

The SMOTE interpolation drew (src, nbr, alpha) per iteration AND wrote one output row per iteration.
The per-row numpy fancy-index + lerp dominated (~81% tottime at n_synthetic=5000, vs ~18% for the
NearestNeighbors fit). The RNG draws must stay in the exact interleaved per-iteration order to keep the
PCG64 stream (hence the synthetic cloud) bit-identical, so only the gather+lerp is hoisted out of the loop.

Measured (py3.14.3, n_min=500, d=30, k_smote=5, best-of-15):
  OLD (row loop)            ~48.7 ms
  NEW (draws + vec lerp)    ~31.7 ms   => ~35% faster, BIT-IDENTICAL.

Run: python -m mlframe.feature_engineering._benchmarks.bench_borderline_smote_synthesize_vectorized
"""
from __future__ import annotations

import functools

from mlframe.feature_engineering._benchmarks._smote_bench_shared import smote_bit_identity_and_speed_main
from mlframe.feature_engineering.transformer._smote_kernels_shared import smote_synthesize_rowloop as _old, smote_synthesize_vectorized as _new

main = functools.partial(smote_bit_identity_and_speed_main, _old, _new)


if __name__ == "__main__":
    main()

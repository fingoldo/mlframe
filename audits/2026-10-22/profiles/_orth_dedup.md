# `_orth_dedup` (pairwise-complete |Pearson r| of the source-side collinear dedup)

Measured 2026-11-16, GTX 1050 Ti (sm_61, 112 GB/s). Shape: Q and R of 256 x 100000 in the criterion dtype (float32 by default), about 3 % missing values, 10 % of the rows near-duplicates.

## Backends and the tuned choice

Wall per call, numpy (BLAS) / njit (parallel) / cupy (explicit GEMMs), ms: 16 x 5000: 1.3 / 0.9 / 1.7; 64 x 20000: 22.4 / 59.7 / 3.5; 256 x 100000: 628 / 4677 / 232. Largest difference of |r| against numpy: njit 6e-7, cupy 9e-7. In float64 the picture flips on this card (cupy 1.9 s against numpy 1.3 s at 256 x 100000: double GEMMs run at 1/32 rate), which is why the sweep is on the criterion dtype the dedup really passes.

The choice is the tuner entry `fe_pairwise_complete_corr` (`_resolve_pc_backend`: env force, then STRICT-GPU, then `_SPEC.choose(p=max(q, r), n=n)`; fallback numpy), a three-way grid sweep over (columns, rows) with every backend held to 1e-5 of the numpy reference. It replaces the earlier read-only override hook in `dispatch.lookup_pairwise_corr_backend`, whose docstring said that no sweep was registered; the numpy default rested on a float64 solo measurement and on the contention argument, neither of which the sweep shares.

## Launches per call and the GPU profile (nvprof, 6 calls of the cupy form, 256 x 100000)

Launches per call: 6 explicit cuBLAS GEMMs (`maxwell_sgemm_128x64_tn`, `sgemm_32x32x32_NT_vec`), 1 fused `pc_corr_final` elementwise pass (covariance, both variances, |r| and the nan mask for fewer than 8 common rows or a near-constant side; about a dozen elementwise launches before), plus the masks (`isfinite`, `where`, the two squares, the casts): about 24 launches per call.

Metrics (nvprof): `maxwell_sgemm_128x64_tn` achieved occupancy 0.25; `sgemm_32x32x32_NT_vec` occupancy 0.06 to 0.15, multiprocessor activity 87 %, single-precision FLOP efficiency 40 % average, 58 % peak, device-memory read throughput 43 GB/s; `pc_corr_final` occupancy 0.44, multiprocessor activity 71 %, read throughput 29 GB/s (up to 49 of 112 GB/s). GPU time: GEMMs 26 %, host-to-device copies 17 %, the mask elementwise kernels 5 to 6 % each.

## Host profile (cProfile, 5 calls of the cupy form)

1.00 s for 5 calls, 17101 function calls; 0.85 s is the blocking `.get()` of the result (it waits for the GEMMs, the GPU time surfacing), 0.11 s the uploads of the tiles, the Python around the launches below 0.01 s.

## Streaming

The six masked products hold about 5 (rows, n) operand arrays per side plus the (q, r) results; when that does not fit a quarter of the free VRAM the pass runs in (row block x row block) tiles sized by the free room, R's tiles uploaded once, and below `_MIN_CUPY_DEDUP_TILE` rows per tile it hands over to numpy. The rows are capped at 100000 by `MLFRAME_FE_DEDUP_MAX_CORR_ROWS`, so the work matrices stay bounded for any n.

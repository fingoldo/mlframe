# _pairs_analytic_gpu (fused observed MI and occupied bins, int8 resident codes)

Kernel `analytic_obs_mi_bins_i8` in `src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_analytic_gpu.py`.

Measured (GTX 1050 Ti, 100000 rows x 2000 int8 columns of 10 levels, binary target, synchronized, min of 3): 15 ms per call, tile 32 columns per block.
The kernel is not the cost of `score_prospective_pairs` in the stage map: the 1.8 s the sampler charges to `_fused_observed_mi_and_bins` is the `.get()` that waits for the candidate-build work queued before it (the main thread blocks at the first
host read of the call). Optimising this kernel would not move the stage; the lever is overlapping that build with host work.

nvprof counters of the run mix the helper launches (the min / max reduction, the cast) with the fused kernel; the fused kernel alone is 15 ms for 200 MB of codes read once (13 GB/s), i.e. far below the card's memory bandwidth, and the 6 SMs are
covered by 63 blocks of 256 threads. A wider tile or a column-block split was not tried because the call is not on the critical path.
Tests: the fused path is covered by the analytic gate tests of the pair search.

## Tile-width sweep, 2026-11-12 (GTX 1050 Ti, 100000 rows x 2000 int8 columns of 10 levels, binary target, synchronized, best of 3)

| tile (columns per block) | ms per call | result equal to tile 32 |
|---|---|---|
| 32 | 14.93 | (reference) |
| 16 | 15.04 | yes |
| 8 | 14.44 | yes |

The tile width does not move the time (within 4 percent, with identical output), so the limiter is not the block width. The remaining candidates are the number of row groups per column (each column's rows are summed by `blockDim / tile` threads, so a wider row split per column is untried) and the memory layout (the codes are row-major, so a column's rows are strided; a column-major copy would coalesce the row reads, but the caller holds the row-major resident matrix). Neither is implemented; point 5 stays open on these two.


## 2026-11-03: no per-group Python lists for candidates

100000 x 100 (4 group columns, 32 numeric columns, 384 candidates): the stage sample put 12 of 21 samples on the main thread waiting for the column pool, whose
tasks converted every group's sorted values and the global sorted column to Python lists (`.tolist()`, GIL-bound, so the threads could not overlap it). The
candidate payloads now keep the arrays; `build_grouped_quantile_recipe` already converts the values of the `top_k` winners to float lists. Family wall 3.96 -> 2.73 s
(one process, old and new module side by side), same eight winners, equal recipes and columns.

## 2026-11-10: p-value path binned in column blocks
`_p_added_by_column` binned every engineered and source column with the host `_bin` (np.quantile + searchsorted, ~3.5 thread-seconds at 100k x 100, 26 columns/pool worker). `_prebinned_codes` now bins all finite columns with more than n_bins distinct values in blocks through `quantile_codes_block_auto` (device kernel for large float32 blocks, njit kernel otherwise); the partition equals the host's (21 columns incl. ties/exponential/cubed/zero-inflated, device and host kernels: identical partitions), which is all the G-test reads. Few-valued (<= n_bins distinct) and non-finite columns stay on the host binner because the host's float32 quantile edges can merge adjacent levels (test_grouped_quantile_prebinned_p_added pins this). Fit wall 100k x 100: 73.0 s (noise-level; `_bin` left the top of the thread census).

## 2026-11-16: batched statistics (numba form, CUDA form, KTC) put the family under the contract

The percentile rank within the group and the group spreads were a per-group host loop on the pool (a radix order of each group's finite values, the average-rank kernel, the linear quantiles). `_grouped_quantile_gpu` (carved child, with `_grouped_quantile_ranks`) does a whole block of numeric columns at once and returns the same four things (`pct`, the sorted segments, `iqr`, `p90p10`) BIT-EQUAL to the loop (test: ties, signed zeros, a group of three rows, both backends, the generated frame and the recipe payloads):
- device form: each column is sorted ONCE by the 64-bit key (group code << 32 | order-preserving 32-bit key of the value; `-0.0` canonicalised to `+0.0`), `group_pct_ranks` (one thread per sorted position, two binary searches for the run of equal values inside the group's segment) and `group_quantiles` (one thread per group and column, numpy's lerp with the float32 difference and the float64 interpolation);
- host form: `_stats_host`, the per-group radix order and the nogil rank kernel (the old loop, batched);
- kernel-tuning entry `grouped_quantile_stats_backend` (sweep over rows, 16 columns x 12 groups, exact equality), `MLFRAME_GROUPED_QUANTILE_DEVICE=0|1` forces a side; swept on this host: host at 5000 rows, device from 20000. Columns with a non-finite value, another criterion dtype or a group order that differs from the host's keep the per-group loop. Column blocks sized from the room under the pool (28 bytes per value).

### cProfile and timings

Block of 32 float32 columns x 100000 rows x 20 groups: host form 0.584-0.627 s (single thread), device form 0.058-0.184 s (sort, kernels and the D2H of the ranks and the sorted values included). The family call (`hybrid_grouped_quantile_fe`, 4 group + 32 numeric columns, 100000 rows): 2.6-2.7 s with the loop, 2.1 s with the device form, the same winners and recipes. cProfile of the earlier loop: `generate_grouped_quantile_features` 0.93 s, `_prebinned_codes` 0.93 s, scoring 2.7 s.

### nvprof (GTX 1050 Ti, sm_61), 2 calls of the 32 x 100000 block

Kernel launches per call: 256 radix-sort launches (a group of them per column; 38 % of the device time), 1 `group_pct_ranks` (2.6 ms, 9 %), 1 `group_quantiles`, 2 gathers, plus 4 copies each way (the D2H of the ranks and the sorted values is 18 %). `group_pct_ranks`: achieved occupancy 0.97, multiprocessor activity 99.8 %, global load efficiency 81.9 %, device read throughput 19 GB/s of the 112 GB/s card (binary searches over a cached segment: latency bound). `group_quantiles`: occupancy 0.08 (one thread per group, 20 threads a block: the kernel is 0.1 ms, a launch of its own is cheaper than fusing it into the rank kernel); SM activity 72 %. Whole-GPU utilization of the call: the device is busy about 25 ms of the 58 ms (the rest is the host staging of the block and the output copies).

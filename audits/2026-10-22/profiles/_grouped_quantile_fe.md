
## 2026-11-03: no per-group Python lists for candidates

100000 x 100 (4 group columns, 32 numeric columns, 384 candidates): the stage sample put 12 of 21 samples on the main thread waiting for the column pool, whose
tasks converted every group's sorted values and the global sorted column to Python lists (`.tolist()`, GIL-bound, so the threads could not overlap it). The
candidate payloads now keep the arrays; `build_grouped_quantile_recipe` already converts the values of the `top_k` winners to float lists. Family wall 3.96 -> 2.73 s
(one process, old and new module side by side), same eight winners, equal recipes and columns.

## 2026-11-10: p-value path binned in column blocks
`_p_added_by_column` binned every engineered and source column with the host `_bin` (np.quantile + searchsorted, ~3.5 thread-seconds at 100k x 100, 26 columns/pool worker). `_prebinned_codes` now bins all finite columns with more than n_bins distinct values in blocks through `quantile_codes_block_auto` (device kernel for large float32 blocks, njit kernel otherwise); the partition equals the host's (21 columns incl. ties/exponential/cubed/zero-inflated, device and host kernels: identical partitions), which is all the G-test reads. Few-valued (<= n_bins distinct) and non-finite columns stay on the host binner because the host's float32 quantile edges can merge adjacent levels (test_grouped_quantile_prebinned_p_added pins this). Fit wall 100k x 100: 73.0 s (noise-level; `_bin` left the top of the thread census).

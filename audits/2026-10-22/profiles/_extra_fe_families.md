# Profile record: conditional residual and dispersion families (`_extra_fe_families`, `_extra_fe_families_dispersion`, `_conditional_bin_moments`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores. Measured 2026-11-14 on an idle machine, medians of 5 after a warm call.

## What was added
`_conditional_bin_moments.py`: the per-bin count, mean and centred sum of squares of many columns x_i grouped by one x_j's bin codes. Before, each (x_i, x_j) pair walked the column twice in numba.
Now one batch per x_j goes through `bin_moments_dispatch`: the numba loops (`_bin_sum_count_masked_njit`, `_bin_css_njit`) or `cond_bin_moments`, one 256-thread block per column with a shared fixed-point
(sum, count, css) histogram, so the result is reproducible (no order-dependent double atomics). Both generators (`generate_conditional_residual_features`, `generate_conditional_dispersion_features`) use it.
Kernel-tuning entry `cond_bin_moments_backend`, swept by the work n * p. Column blocks (`_device_blocks.first_block` / `run_column_blocks`, halved on OOM) bound the device memory.

## cProfile (dispersion generator, 12 columns x 200,000 rows, whole call 1.27 s)
`bin_moments_host` 0.405 s cumulative (12 calls), `_zscore_from_bins_njit` 0.148 s, `_std_njit` 0.147 s, `_emit_kind` 0.124 s, `ascontiguousarray` 0.135 s. The moments are 32 percent of the call.

## nvprof / nsys
`cond_bin_moments`: launches per x_j batch: 1 per column block (one block per column). nvprof counters fail on this machine today (profiling error 4211), so occupancy is theoretical:
256 threads per block, 4 * n_bins * 8 bytes of shared memory (320 bytes at 10 bins), launch bounds 256, so shared memory does not limit residency and 8 blocks of 256 threads (2048 threads, full occupancy on sm_61) fit an SM.

## Wall clock (64 columns x 50,000 rows, 10 bins)
| part | time |
|---|---|
| numba host loops | 16.6 ms |
| kernel on resident data | 5.18 ms (3.2x) |
| stack + upload of the float64 columns | 11.9 ms |
| device call from host columns (kernel + upload) | 19.8 ms |

Derived throughput of the kernel: 64 x 50,000 x 8 B x 2 passes = 51 MB in 5.18 ms = 9.9 GB/s of the card's 112 GB/s peak bandwidth; the loss is the 64-bit shared atomics on 10 addresses (contention), not the memory system.
The kernel-tuning sweep (`tune_spec(force=True)`) picks the host for host-resident columns at every size, because the upload costs more than the kernel saves; the device path is for columns already on the card
(`bin_moments_dispatch` takes a `(p, n)` device matrix without any upload and always runs on the device then). Equality: count exact, mean and css to 1e-12 relative against the numba loops on columns spanning six decades, with NaN, inf and an empty bin; two runs give identical bits (`tests/feature_selection/gpu/test_conditional_bin_moments.py`).

## Open
Per-warp private histograms would remove the atomic contention (the same fix as the modular rank kernel version 6); not measured yet.

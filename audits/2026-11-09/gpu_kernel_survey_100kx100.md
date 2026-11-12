# GPU kernel survey of one default MRMR fit, 100000 x 100 (2026-11-13)

`nvprof --csv` over `MRMR(verbose=0, random_state=0).fit(X, y)` after a warm-up fit on 5000 rows (`D:/Temp/exp/nv_fit.py`; GTX 1050 Ti, CUDA 11.8 CUPTI on `PATH`, elevated session). 31.3 s of GPU activity
over 454 distinct kernels and copies; nvprof serialises launches, so the shares are an upper bound on the wall (CUDA.md section 6). The table is the order in which to look, not a wall budget.

| share | GPU s | calls | avg ms | kernel |
|---|---|---|---|---|
| 17.4% | 5.43 | 671 | 8.10 | `family_null_bits_v2` |
| 14.5% | 4.54 | 1133 | 4.00 | `radix_select_f32_v5` |
| 6.2% | 1.94 | 3520 | 0.55 | cub (device reduce / scan, via cupy) |
| 3.8% | 1.18 | 52 | 22.7 | `cupy_concatenate` |
| 2.5% | 0.79 | 647 | 1.22 | `fe_materialise_cm` |
| 2.3% | 0.72 | 1305 | 0.55 | `cupy_bitwise_or` |
| 2.2% | 0.70 | 513 | 1.37 | `transpose_f32` |
| 2.1% | 0.66 | 787 | 0.84 | cub |
| 2.1% | 0.66 | 2 | 327 | `root_split_accept` |
| 2.1% | 0.64 | 10946 | 0.06 | memcpy HtoD |
| 1.5% | 0.48 | 276 | 1.73 | `radix_select_interp_f64_v3` |
| 1.3% | 0.42 | 840 | 0.50 | `strata_xy_entropy` (GPU11-82) |
| 1.3% | 0.41 | 600 | 0.68 | `batched_joint_hist2_i32_smem_u8_v2` |
| 1.3% | 0.41 | 4 | 101 | `gate_logistic_auc` |
| 1.3% | 0.40 | 2 | 202 | `triple_mi` |
| 1.2% | 0.36 | 9 | 39.9 | `cond_mi_columns` |
| 1.0% | 0.33 | 238 | 1.37 | `clenshaw_basis` |
| 1.0% | 0.30 | 67 | 4.50 | `bin_codes_f32e_int8` |

## Verdicts for the two largest

**`family_null_bits_v2`** (one thread per (permutation, column); ANDs a column's bin bit set with the permutation's class bit sets, `popcll`). Bench `D:/Temp/exp/bench_family_bits.py`, 100000 rows, 1000 columns of 8 bins,
40 permutations: 9.2 ms for two classes (500M word operations, 54 G/s), 15.2 ms for four classes, 2.1 ms for 100 columns. The loads are coalesced (word-major bit sets, consecutive columns per warp) and the class words are a shared
broadcast. The limiter is the population count: Pascal runs `popc` at a quarter of the integer rate, two per 64-bit word, so 1G `popc` is about 3.7 ms of the 9.2 ms on 6 SMs; the rest is the AND, the add and the load. A
carry-save (Harley-Seal) tree trades 16 popcounts for about 75 logic operations plus 5 popcounts, which is no saving at a quarter rate. Blocking several permutations per thread would cut the loads (about a fifth of the issue
slots), bounded by registers and the 48 KB shared class tile. Not changed: at most about 1.5x of 5.4 s, under 2% of the wall, and the kernel is already past what the popcount rate allows.

**`radix_select_f32_v5`** (one block of 768 threads per column, byte-wise radix select with candidate compaction). nvprof counters at 100000 x 100 (`D:/Temp/exp/nv_radix.py`): 2.66 ms per launch, grid (100), 40 registers,
18 KB dynamic shared memory, achieved occupancy 0.73, sm_efficiency 95%, global load efficiency 99.99%, DRAM read 40 GB/s (about a third of the card's peak), IPC 2.51, warp execution 88%, shared efficiency 2.65%
(the shared-memory histogram atomics), stalls: execution dependency 33%, synchronization 24%, memory dependency 8%. It is compute and barrier bound on the histogram atomics, not on memory; v1 to v5 already compacted the
candidates. Not changed.

## Next

`cupy_concatenate` (52 calls, 22.7 ms each: a copy that can be avoided at the call site) is traced by `D:/Temp/exp/concat_sites.py`; `transpose_f32`, `fe_materialise_cm`, `cupy_bitwise_or` and the cub reductions are the
launch-bound and copy rows of the next pass.

# `_pair_mi_perm_cuda_bin` (pair joint MI against K permuted BINARY targets: the inner loop of the order-2 maxT floor, outcome-code form)

Measured 2026-11-26 on the GTX 1050 Ti (4 GB, 112 GB/s, sm_61, 6 SMs, DP at 1/32 of SP). GPU state around the timed run (`nvidia-smi`): 77 C and SM 1708 of 1911 MHz before, 81 to 82 C and 1695 MHz after, throttle-reason mask 0, so the clocks were not throttled.
Shape: the default-binning order-2 floor, 55296 rows (the scaled floor's subsample for 48 bins), 600 pairs of 48 x 48 bins (2304 cells), K = 25 permuted binary targets, 2 permutations a pass (13 passes), one launch of 600 blocks.
Script: `src/mlframe/feature_selection/_benchmarks/profiling/nvprof_pair_mi_perm_bin.py` (`time` mode: interleaved rounds with GPU state; `log` / `lut` mode: one warm launch in the profiler window). Counters taken with `nvprof --metrics` (kernel replay; the kernel writes its output without reading it, so replay is safe).

## What was found

The kernel version `_SRC` (logarithm form) spent its time in the epilogue, not in the row loop. Only `kb` warps (2 of 16) of a block ran the MI loop over the 2304 cells with two double-precision `log` per non-empty cell and permutation, and the other 14 warps waited at the barrier:
`stall_sync` 85 percent, issue-slot utilization 8 percent, IPC 0.35, `flop_dp_efficiency` 40 percent. The cost per pair-pass was the same at 27000, 55000 and 110000 rows (19 us), the signature of cell-side work, not of the rows.

## Versions (kept side by side; `MLFRAME_PAIR_PERM_LUT=0` runs the first)

| | `_SRC` (log) | `_SRC_LUT`, epilogue on `kb` warps | `_SRC_LUT`, epilogue on all warps (shipped) |
|---|---:|---:|---:|
| kernel time, one launch (nvprof trace) | 136.45 ms | 18.77 ms | 13.43 ms |
| block / registers / dynamic + static shared | 512 / 64 / 36 KB + 0 | 576 / 56 / 36 KB + 0 | 576 / 56 / 36 KB + 384 B |
| spill (`local_size_bytes`) | 0 | 0 | 0 |
| achieved occupancy | 0.499 | 0.562 | 0.562 |
| SM efficiency | 97.3 % | 98.6 % | 98.5 % |
| IPC | 0.35 | 1.58 | 2.34 |
| issue-slot utilization | 8.0 % | 35.9 % | 52.9 % |
| `flop_dp_efficiency` | 40.0 % | 7.1 % | 11.6 % |
| `stall_sync` | 85.4 % | 65.3 % | 12.3 % |
| `stall_exec_dependency` | 12.2 % | 16.4 % | 44.1 % |
| `stall_memory_dependency` | 1.0 % | 11.9 % | 24.1 % |
| global load efficiency | 99.9 % | 90.8 % | 90.9 % |
| DRAM read | 7.7 GB/s | 42.3 GB/s | 58.8 GB/s (of about 112) |
| warp execution efficiency | 99.9 % | 99.8 % | 99.7 % |

Changes, in the order taken: (1) the table form: with integer counts the plug-in MI is `n MI = sum over cells [T(c0) + T(c1) - T(cx)] - T(n0) - T(n1) + T(n)`, `T(c) = c ln c`, so a table of T for `c = 0 .. n` (one upload per call, `_xlogx_host`) replaces the logarithms
by three loads per non-empty cell and the class totals come from the cells (7.3x on the kernel); (2) the cells of a permutation are split among `n_warps / kb` warps with partial sums added in a fixed order (deterministic: two launches give the same bits, tested), 18.77 -> 13.43 ms (1.4x). `stall_sync` is gone as the limiter; what is left is the execution dependency of the shared-memory atomics and loads.
Rejected by the counters: nothing tried and discarded here; not yet tried: a second pass-level overlap of the next pass's histogram clear with the current epilogue.

## Whole call and sizes (host-in, `launch_pairs_binary`, interleaved 9 rounds, minimum)

600 pairs x 25 permutations at 55296 rows, 48 bins: 157.5 ms -> 34.2 ms (4.6x; medians 160.5 -> 36.1 ms); what remains outside the kernel is the 21 MB upload, the plane packing and the readback.
Per size (`D:/Temp/exp/bench_lut_kernel.py`, 6000 pairs, synchronized, min of 2): 20000 rows x 16 bins 2.29x; 55296 x 48 6.90x; 55296 x 32 3.77x; 500000 x 16 1.05x; 500000 x 48 1.68x; the values agree to 4e-15 (up to 2.4e-2). The table form never lost, so there is no CPU/GPU-style crossover to tune and no KTC entry for the choice; `MLFRAME_PAIR_PERM_LUT=0` is the escape.
Inside the fit (500000 x 1000 ALL, default binning, n_jobs=-1, first version of the table form): the pooled pair floor 150 -> 45 s, wall 586.1 -> 470.4 s, same selection (`_TRACKER.md`, PLAN-SPEED-27).

## Tests

`tests/feature_selection/gpu/test_pair_mi_perm_cuda_bin.py`: the table form equals the logarithm form to 1e-12 for narrow and wide tables and row counts 3, 4001, 55297; for block widths of 1, 2, 3, 9 and 32 warps (fewer warps than permutations, more, odd) on three table shapes and three seeds; bit-repeatable; the cap rule gives zeros in both forms; the table is `c ln c`.
`test_cuda_kernels_offline_per_arch.py`: `_SRC_LUT` assembles on sm_61 and sm_120 without a spill.

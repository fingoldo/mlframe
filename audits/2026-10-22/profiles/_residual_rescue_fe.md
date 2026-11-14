# _residual_rescue_fe.py (residual rescue) with its carved modules `_residual_rescue_gpu` and `_residual_rescue_ktc`: profile record (2026-11-14)

Family: `_residual_rescue_fe` (the rescue driver, the numba bin-coding `_codes_matrix` and the numba scoring `_score_columns`), `_residual_rescue_gpu` (their device twins `codes_matrix_gpu` and `score_columns_gpu`) and `_residual_rescue_ktc` (the two kernel-tuning entries and the dispatchers). Idle machine, GTX 1050 Ti, float32 criterion dtype, n = 50000 scan rows, 6 design columns, 300 pool columns, one admitted column.

## cProfile

One `residual_rescue` call, host path (`D:/Temp/exp/rescue_prof2.py`): 0.861 s under the profiler, wall 0.65 to 0.70 s without it. `_code_columns` 0.552 s cumulative (64 percent; of it the numba coding `_codes_matrix` 0.284 s, the column assembly `np.stack`/slicing 0.171 s, `astype` 0.029 s), the driver's own time 0.111 s, the two scoring calls `_score_columns` 0.053 s (6 percent), the per-column variance of the pre-cut 0.095 s. The scoring is the small part of a rescue call; the coding is the large one.

## Kernels (nvprof, one device call of each twin, 50000 x 300, after one warm-up)

`D:/Temp/exp/rescue_nv.py`. Kernel launches per call: `rescue_score_columns` 1 launch per column block (here one block of 300 columns), 16.68 ms, 30.6 percent of the GPU time; `rescue_codes` 1 launch per column block, 3.33 ms, 6.1 percent; the quantile edges of the sample come from 12 cub sort launches (5.19 ms plus 3.06 ms) and 5 small cupy elementwise and reduction launches (isfinite 1.05 ms, where 0.47 ms, sum 0.36 ms, copy 0.16 ms, sum 0.16 ms); the transfers are 4 host-to-device copies (18.6 ms, 34.1 percent: the float32 matrix of 60 MB and the int32 codes of 60 MB) and 4 device-to-host copies (4.5 ms). So the kernels are 37 percent of the GPU time and the host-to-device copies another 34 percent.

## Occupancy and throughput

nvprof counters (achieved occupancy, DRAM throughput, issue-slot utilization) could not be collected: every `--metrics` run on this machine ended with `Internal superseded by the Hardware counters section below` today, including a kernel that was measured with counters in the morning, so the figures below come from the resource usage and the timings. `rescue_score_columns`: 32 registers, 272 bytes of static and 448 bytes of dynamic shared memory at 256 threads, 8 resident blocks per SM by the thread limit (the register limit allows 8, the shared memory 136): theoretical occupancy 1.00. `rescue_codes`: 19 registers, no shared memory, 8 blocks per SM by threads (registers allow 13): theoretical occupancy 1.00. Throughput from the timings: the coding kernel handles 15 million elements (4 bytes read, 1 written) in 3.33 ms, 22.5 GB/s of the card's 112 GB/s peak; the scoring kernel reads the 15 million int32 codes in three passes over half the rows each (even rows once, odd rows twice, for the mean and then the variance), 90 MB of codes in 16.68 ms, 5.4 GB/s, with the 8-byte residual of each element read through the cache: it is bound by the dependent shared-memory loads of the bin sums, not by the memory system.

## Host-or-device choice

Two entries of the kernel-tuning cache (`_residual_rescue_ktc`), swept on the idle machine: `residual_rescue_codes_backend` (swept by the matrix cells n x p): host below 400000 cells, device from 400000 up (25000 x 16 columns and larger); measured medians of 5 calls, float32 / float64: 50000 x 300 host 214 / 279 ms, device 109 / 153 ms (1.97x / 1.83x); 25000 x 120 host 56 / 55 ms, device 28 / 33 ms; 900 x 10 host 0.5 ms, device 5.4 ms. The codes are integers and equal to the numba loop's exactly, for both float types, with NaN, infinities, a constant column, a nominal-like column and a mostly missing column. `residual_rescue_score_backend` (swept by n x p): host at every size swept up to 25.6 million column-rows; medians of 9 calls: 25000 x 300 host 13.5 ms, device 48.8 ms; 50000 x 300 host 27.5 ms, device 70.4 ms; the device pays an upload of the int32 codes and a kernel that is latency-bound, so the numba loop (one column per thread, 4 cores) stays the choice on this host and the device path is the tuned alternative for a host with fewer cores or a faster card. The scores agree to 1e-7 relative (the bin sums are 2^-36 fixed-point integers on the device, so a run gives the same bits as the next; equal to the numba loop to 1e-14 relative before that change). Whole `residual_rescue` call, both paths forced: 0.65 to 0.70 s on the host, 0.68 to 0.72 s with the device coding and scoring forced for every call (the column assembly on the host is the same work).

## Streaming

Both twins move the pool to the device in column blocks (`_device_blocks.first_block` sizes the first block from the room under the pool, `run_column_blocks` halves a block that runs out of memory); the residual and the even mask are uploaded once. The tests pin that narrow blocks give exactly the single-block result.

## Hardware counters (nvprof --metrics, elevated shell, 2026-11-14)
Superseding the theoretical occupancy stated above (the counters were blocked by the driver's restriction until the process ran as administrator). Kernel alone, last sweep size, one profiled call after a warm call. Pascal sm_61: Nsight Compute reports the device as unsupported, nvprof is the tool.

| kernel | achieved occupancy | SM efficiency | DRAM read | DRAM write | global load eff. | global store eff. | shared eff. |
|---|---|---|---|---|---|---|---|
| rescue_score_columns | 98.5 % | 97.8 % | 10.0 GB/s | 0.54 GB/s | 53.8 % | 0.16 % | 3.74 % |

Load efficiency 53.8 percent: the pool is read as (column, row) int32 while the residual and mask are read per row by every column block. The `rescue_codes` kernel was not profiled.

## Byte bin codes in `rescue_score_columns` (2026-11-14)
The counters showed global load efficiency 53.8 percent and DP efficiency 27.1 percent with high occupancy (98.5): the kernel is not starved of threads, it reads. Every column read its 32-bit bin codes three times (the even-row histogram, the odd-row sum, the odd-row variance) and the host uploaded them as 32-bit integers although a code is below `n_bins` (16 by default). Change: when `n_bins <= 127` the codes are uploaded and read as `signed char` (kernel compiled per code type, `{QT}`); the first column-block size follows the narrower element. Outputs are bit-identical (`array_equal` on t and gain).
Interleaved A/B, 9 rounds, medians, whole `score_columns_gpu` call with upload (400 columns x 25,000 rows, 16 bins): 34.3 -> 21.9 ms (1.57x). Counters after: achieved occupancy 96.8 percent, DP efficiency 27.4, DRAM read 2.8 GB/s, load efficiency 51.2 (the residual and the mask are read per row by every column and stay the reads that dominate; the byte codes cut the code traffic and the upload to a quarter, not the broadcast reads). The kernel's remaining cost is the double-precision per-row arithmetic (27 percent of the card's double rate); a single-precision rejection is not applicable because every odd row contributes to the sum. The offline per-arch recipe compiles both code types.

# `_wavelet_legs_fused_gpu` (dyadic-Haar leg code matrices in two launches)

Measured 2026-11-14, GTX 1050 Ti (sm_61, 112 GB/s), one column of 400000 rows, `max_scale` 3 (15 legs, all eligible), float64 input (the critical dtype converts it to float32 on the card before the launches).

## Launches per call and the GPU profile (nvprof, 6 calls)

Launches per call: 1 `leg_supports`, 1 `leg_codes`, plus the small host reads (one 30-counter vector back, the eligible-leg tables up). GPU time per call: `leg_codes` 1.33 ms (74 % of the module's GPU time), `leg_supports` 0.31 ms (17 %), the float32 conversion of the column 0.2 ms, uploads 0.5 ms in total.

Metrics (nvprof): `leg_supports` achieved occupancy 0.93, multiprocessor activity 95 to 99 %, device-memory read throughput 3.4 to 7.5 GB/s (the column is read once; the kernel is bound by its shared-memory counters, not by memory). `leg_codes` achieved occupancy 0.69, multiprocessor activity 95 to 100 %, device-memory write throughput 29.5 to 72.4 GB/s of 112 (the write stream of the two int64 code matrices, 48 MB per call), global store efficiency 25 %, load efficiency 23 % (the `eligible_j` / `eligible_k` table reads, broadcast, and the one-row-per-thread store pattern).

## Host profile (cProfile, 20 calls)

0.047 s for 20 calls (2.3 ms per call), 2281 function calls: 0.033 s of it is the one `.get()` of the 30 support counters, which waits for the supports kernel, i.e. the GPU time of the whole call surfaces there; the Python around the launches is 0.004 s (0.2 ms per call). No per-leg or per-column loop.

## This pass

* `leg_supports` counts into a block-private shared-memory histogram (2 counters per leg) and adds each non-zero counter to the global array once per block, where every thread issued one global atomic per row and level before. Kernel in isolation, 400000 rows: 1.17 ms -> 0.48 ms; 100000 rows 0.26 -> 0.25 (same); counts identical (the per-leg loop reference test over scales 0 to 4 and 10 passes). The shared histogram bounds `max_scale` at 10 (16 KB); the production scale is 3, deeper ones decline to the per-leg loop.
* **Rejected, kept out of the module:** a `leg_codes` that gives one thread to each (row, leg) cell with the leg fastest, so a warp writes consecutive words. Same codes (bit equal), but 0.42 against 0.38 ms at 100000 rows, 1.69 against 1.38 ms at 400000 rows, 6.6 against 8.9 ms at 1000000 rows: the kernel is bound by the volume of its int64 stores, not by their pattern (the 64-bit division per cell costs what coalescing gains). The lever left is a byte code matrix, which needs the MI-from-codes kernels to read bytes (they widen to int64 today); not taken here, the whole call is 2.8 ms at 400000 rows against the MI that follows.

## Block structure

The builder makes every leg in one pass, so the caller uses it only when all legs fit one column block of `_DEVICE_LEG_BLOCK_BYTES`; a bigger leg set goes through the per-leg loop in column blocks (`run_column_blocks`, halving on out-of-memory). The code matrices are cushion-checked before allocation.

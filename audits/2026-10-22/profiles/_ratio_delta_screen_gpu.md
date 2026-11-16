# `_ratio_delta_screen_gpu` (device redundancy screen of the ordered-pair ratio / log-ratio pools)

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. Hardware: GPU 80 to 82 C, SM 1683 to 1695/1911 MHz, mem 3504 MHz, throttle none; CPU 2808/2808 MHz, load 17 to 18 %.
Bench: `ratio_pair_screen` over p = 100 float32 columns, n = 100000 rows (9900 ordered pairs), sources resident for the device form, min of 3.

## Whole pass, by path

| path | wall per screen |
|---|---|
| host screen at p = 100 (parallel numba `_screen_njit`) | 3.17 to 3.27 s |
| device, double elementwise math | 1.39 s |
| device, float elementwise math, double accumulators | 1.03 s |
| device, float per-thread partial sums of the centred values | 0.85 s |

The Python loop it replaced (build the column, two numpy std, two corr per pair, accepted columns kept) ran `pairwise_ratio_features` on 100000 x 40 columns in 5.09 s against 2.68 s now (same 1560 accepted pairs, identical output frames); at p = 100 the loop could not finish its 9900 accepted float64 columns (a 7.4 GiB allocation). The cProfile of the end-to-end pass is in `_ratio_delta_fe.md`. The accept matrices of the device and the host are identical (0 differing pairs, ratio and log-ratio, 20000 and 100000 rows, sweep inputs with NaNs, a collinear pair, a near-constant denominator and a constant column). The `ratio_pair_screen_backend` sweep picks the device at 5000, 20000, 100000 and 400000 rows (24 columns). End to end, `pairwise_ratio_with_recipes` with the MI gate on 100000 x 40 columns: 6.7 s (host) -> 4.4 s (STRICT).

## Launches per call and the GPU profile (nvprof)

Launches per call: one `ratio_pair_screen` launch for the whole pool (10000 blocks of 256 threads, 52 registers, 2 KB shared) plus one upload of the (p, n) matrix and one 10000-byte read-back; the screen is one launch where the Python loop made about 9900 x 6 numpy calls. Achieved occupancy 0.50; device-memory read throughput 9.6 GB/s of the card's 112 GB/s (the two source columns of a block are re-read from L2 across the 99 blocks that share them), so the kernel is arithmetic bound (the first version ran at 58 % of the double-precision peak, which is why the elementwise math moved to the column dtype). Whole-GPU utilization during the launch: one kernel occupying the card for 0.85 s.

## Backend choice and memory

Choice: `ratio_pair_screen_backend` (kernel_tuner, sweep over the row count at identical accept matrices, `MLFRAME_RATIO_SCREEN_DEVICE=0|1` forces a side), reached when the GPU strategies are engaged. Memory: the (p, n) source matrix is the only device operand (p x n x 4 bytes, 40 MB at 100 x 100000); rows are strided over the block's threads, nothing sized by the pair count x n is stored.

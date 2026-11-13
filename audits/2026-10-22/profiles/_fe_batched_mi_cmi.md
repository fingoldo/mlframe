# _fe_batched_mi_cmi.py: shared-memory joint histogram (2026-11-09)

`batched_joint_hist2_i32` (the per-column joint counts of an (n, K) int64 code matrix with one target code, 1984 launches / 6.1 s of GPU time in the nvprof summary of a 30000 x 300 fit) issues one global `atomicAdd` per (row, column). `batched_joint_hist2_i32_smem` gives a block a tile of up to 32 columns and a chunk of up to 16384 rows, counts into a shared (tile, Kx*Kb) int32 histogram with shared atomics and adds its occupied cells to the global buffer once. Dispatch in `_batched_joint_counts2` through `_hist_smem_tile` (`MLFRAME_HIST_SMEM=0` turns it off): used only when Kx*Kb is at most 128 cells, where it measured faster.

Measured (GTX Pascal, 5 launches averaged, equal counts in every case; `tests/feature_selection/gpu/test_batched_joint_counts_smem.py`): 200000 x 300 x 40 cells 7.62 -> 5.83 ms (x1.3); 30000 x 1944 x 20 cells 8.29 -> 5.59 ms (x1.5); 100000 x 100 x 100 cells 1.39 -> 1.28 ms (x1.1); 100000 x 60 x 288 cells 0.79 -> 1.08 ms (x0.7, kept on the global kernel); 100000 x 20 x 2400 cells 0.33 -> 0.45 ms (x0.7, kept on the global kernel).

Not done: counter-level profile (nvprof achieved occupancy, shared efficiency, stalls) of the shared kernel, and the int64 operand width (8 bytes per code read; an int8/int16 code matrix would cut the read traffic 4-8x but the matrix is built by callers as int64).

## nvprof counters, 200000 x 300 codes, 10 x 4 cells (2026-11-09)

| counter | `batched_joint_hist2_i32` (global atomics) | `batched_joint_hist2_i32_smem` |
|---|---|---|
| achieved occupancy | 0.81 | 0.89 |
| global load efficiency | 91.8 % | 45.9 % (the broadcast 8-byte load of `b[row]` per warp) |
| issue slot utilisation | 85 % | 69 % |
| stall: memory dependency | 37.7 % | 58.0 % |
| stall: synchronisation | 0 % | 0.6 % |
| device-memory read throughput | 61 GB/s | 81 GB/s (about 73 % of this card's ~112 GB/s) |

Both kernels move the same bytes: the int64 code matrix is 200000 x 300 x 8 = 480 MB per launch, and the shared kernel (5.83 ms) reads it at 82 GB/s, close to the card's bandwidth. So the lever left in this kernel is the operand width, not the atomics: an int8 code matrix would read 60 MB (an 8x cut, 5.8 ms -> about 1 ms if bandwidth stays the limit). That needs the whole CMI path (`batched_cmi_gpu`, its marginal / entropy kernels, the callers that build the codes as int64) to carry narrow codes; not started.

## Fused joint entropy `batched_joint_entropy2`: tiled coalesced reads (2026-11-12)

The fused kernel gave one block per column and read X[i*K + col] with threads striding rows, so a warp's loads were K elements apart. It is the fused path, which the dispatch takes only for large joints (or when `MLFRAME_FE_GPU_FUSE_CMI_ENTROPY=1`).

The tiled kernel keeps the same histogram and reduction per column, now with G columns per block: a flat loop over rows x G reads consecutive columns of one row, and each column keeps its own histogram in shared memory. The per-column reduction keeps the original tree order, so the sums match exactly.

Measured on the public fused call (GTX 1050 Ti, interleaved, min of 10):
- n = 100000, K = 384, M = 40: original 36.2 ms, tiled G = 8 4.1 ms (G = 4 8.9 ms, G = 2 11.5 ms, G = 16 3.9 ms; G = 8 chosen as the best that also fits the M = 1000 case).
- n = 30000, K = 930, M = 1000: original 19.9-21.3 ms, tiled G = 8 6.4 ms.
- n = 30000, K = 930, M = 40: original 22.0 ms, tiled G = 4 5.7 ms.

Equivalence: the entropies equal the original kernel's (maxdiff 0) and the unfused count+reduce reference (maxdiff 0 on the production shape). `test_joint_entropy_tiled.py` (8 tests) covers K in {1, 9, 13, 16} with Kb in {2, 7}; a mutant that drops the column offset fails 6 cases.

Still open: `batched_joint_entropy2_g` (global-scratch variant for joints over the shared limit) has the same row-major read and stays in the uncoalesced baseline.

## Global-scratch joint entropy `batched_joint_entropy2_g`: tiled route and an untiled route for large joints (2026-11-12)

The global-scratch kernel serves joints over the shared-memory limit. Its read was row-major with one column per block (the same K-element stride as the fused kernel). A tiled route now handles joints up to 6144 cells: a block walks groups of columns (8 for joints up to 1024 cells, otherwise one), its scratch holds one histogram per column, and the blocks per SM are doubled when the doubled scratch fits in a quarter of L2. Joints above 6144 cells keep the untiled kernel `batched_joint_entropy2_g_wide`, which measured faster there and stays in the uncoalesced baseline.

Measured (GTX 1050 Ti, 6 SMs, interleaved, min of 8-10, committed tree vs new):
- n = 30000, K = 930, M = 1000: 11.3 -> 6.1 ms.
- n = 100000, K = 384, M = 400: 13.8 -> 6.9 ms.
- n = 30000, K = 930, M = 5000: 14.7 -> 11.9 ms (one column per block; eight columns per block were 44 ms, so the column group is limited to joints up to 1024 cells).
- n = 30000, K = 930, M = 15000: 23.3 -> 23.4 ms (untiled route, unchanged). Tiled at this size measured 26.6 ms, so it stays on the untiled route.
- Fused path (the one-launch kernel with the shared-memory tile): old and new tiled versions agree within noise at M = 1000, 5000, 10000 (6.4, 11.1-11.4, 15.4 ms).

Two measured dead ends: more blocks per SM at M = 1000 (4 per SM 14.8 ms, 2 per SM 6.8 ms), and dividing the flat index by the column count per element (the per-element division cost a few ms, replaced by a per-thread row index).

Correctness: `test_joint_entropy_global_tiled.py` checks both routes against the unfused count+reduce reference with K in {1, 9, 13} (partial last group) and Kb in {2, 7, 40}. The test found a real bug first: with a block size not divisible by the group width, the row stride let threads past the grid count rows twice. Threads outside the g x rows grid now sit out. A mutant that drops the column offset fails 2 cases.

## Shared-memory joint histogram: row blocks that fill the device (2026-11-13)

`batched_joint_hist2_i32_smem_u8_v2` (600 launches, 0.41 s of the 100000 x 100 fit survey) at 100000 rows x 25 uint8 columns, 10 x 2 cells, isolated harness `D:/Temp/exp/nv_hist2.py`: grid (7, 2) = 14 blocks of 256 threads
on a 6-SM card (rows per block fixed at 16384), kernel 338 us, achieved occupancy 0.254, IPC 1.01, issue stalls 75.5% on memory dependency, DRAM read 8.9 GB/s, global load efficiency 33% (16 columns per tile read as 16
consecutive bytes of a row, half a 32-byte sector), warp execution 78%. The limiter was the grid, not the access pattern: too few blocks to hide the load latency.

Change: `_hist_rows_per_block(n, col_tiles)` sizes the row block for 8 blocks per SM (`_sm_count()`), a multiple of 256 rows, at least 1024 and at most the old 16384. After: grid (23, 2), kernel 122 us, occupancy 0.93, IPC 2.97,
stalls on memory dependency 66.5%, DRAM read 25.9 GB/s; the load efficiency is unchanged (33%: the tile width, a separate lever). Interleaved A/B of the whole launcher, best of 9 (old / new ms): 100000 x 8 0.300 / 0.121,
x 25 0.428 / 0.183, x 100 0.798 / 0.532, x 400 1.887 / 1.859; 30000 x 25 0.305 / 0.132, x 100 0.728 / 0.211; 1000000 x 25 and x 100 unchanged (1.56, 6.17: the old grid already filled the card); 100000 x 25 with 8 x 8 cells
0.442 / 0.185; 5000 x 50 0.186 / 0.094. Counts are exact (`test_hist_rows_per_block.py`: bincount equality over six shapes, byte and int64 paths, two seeds; the mutant that drops the last partial block fails 20 of 26).

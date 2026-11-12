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

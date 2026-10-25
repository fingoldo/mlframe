# Profile record: _fe_sufficient_summary_pairs (pair residual test)

Host: GTX 1050 Ti (sm_61, 4 GiB), nvprof 10.x with elevated counters, nsys 2022.4.2 (off PATH), ncu refuses Pascal. Workload: p=400 columns, n=5000 rows,
k=5 coarse bins, kr=10 residual codes, 79,800 pairs. Warm, synchronized, resident codes.

## cProfile (host side, one call of `best_pair_streamed`, warm)

| backend | wall | where the time is |
|---|---|---|
| njit | 245 ms | 3 calls of `_pair_mi_block_cpu`, 245 ms tottime, nothing else above 0.1 ms |
| gpu_fused (v2) | 16.4 ms | 3 calls of `_block_best_gpu_fused`, 15 ms tottime (launch + sync), cupy array upload 1 ms, device argmax <1 ms |

cProfile is blind to the asynchronous kernel, so the GPU figure is the synchronized wall clock; the device-side numbers come from nvprof below.

## Kernel launches per call

3 launches per call at p=400 with `BLOCK_COLS=256`: one diagonal block pair and two rectangles. The pair scoring, the histogram, the MI reduction and the argmax over
partners run in ONE kernel per block pair (fused); the only extra device work is a final argmax over the per-first-column results. The generic dispatcher path that
this replaced issued several kernels per pair chunk and was about 149x slower on this shape.

## Whole-GPU counters (nvprof, kernel `pair_mi_block`, v1 source, 12 invocations)

| metric | value |
|---|---|
| achieved occupancy | 0.984 (0.978 to 0.991) |
| sm_efficiency | 99.89% |
| executed IPC | 2.84 |
| issue slot utilization | 67.0% |
| dram read throughput | 4.6 GB/s (dram utilization Low) |
| dram write throughput | 97 MB/s |
| warp execution efficiency | 96.5% |
| shared memory efficiency | 4.5% |
| stalls | memory dependency 30.8%, synchronization 29.2%, execution dependency 20.8%, other 9.7% |

Reading: compute and shared-memory-atomic bound, not DRAM bound (the codes are int8 and re-read from L2). Occupancy is full, so the remaining lever is the
shared-memory access pattern (efficiency 4.5%, atomics on k*k*kr cells) and the 29% synchronization stall.

### v2 kernel (`pair_mi_block_v2`, the default), same workload, 12 invocations, elevated nvprof

| metric | v1 | v2 |
|---|---|---|
| achieved occupancy | 0.984 | 0.963 |
| sm_efficiency | 99.89% | 96.9% |
| executed IPC | 2.84 | 3.08 |
| issue slot utilization | 67.0% | 72.0% |
| dram read throughput | 4.6 GB/s | 5.8 GB/s (utilization Low) |
| dram write throughput | 97 MB/s | 0.99 GB/s |
| warp execution efficiency | 96.5% | 98.3% |
| shared memory efficiency | 4.5% | 2.1% |
| stall: synchronization | 29.2% | 5.9% |
| stall: memory dependency | 30.8% | 31.7% |
| stall: execution dependency | 20.8% | 23.5% |
| stall: other | 9.7% | 27.1% |

Reading: the warp-shuffle reduction removed most of the barrier stall (29% to 6%) and raised IPC and issue-slot use, which is the 13 to 29% wall gain. The memory-dependency stall (shared-memory atomics and global loads) is now the largest and unchanged, so the next lever is the shared-memory access pattern (efficiency 2.1%), for example warp-private histograms. Occupancy is still full; the kernel is not DRAM bound. The "other" stall grew to 27%, not decomposed (nvprof Pascal does not split it).

## Measured speedups

* fused v2 versus njit: about 13x at p=400, n=5000; versus the generic GPU dispatcher: about 149x.
* `{REAL}`=float versus double: 2.2x faster on this Pascal part (1/32 double rate).
* KTC sweep (`pair_ktc_sweep3.log`, 20 cells, 119 s): CPU only at 100 pairs for n<=5000; GPU everywhere else; max abs diff to the CPU reference <= 1.8e-8.

## Candidate optimizations not yet done

* 4-byte vector loads (`char4`) with a padded row stride.
* Upper-triangle-only launch for diagonal blocks (about 5 to 10 percent of blocks are redundant).
* Warp-private histograms to lift the 4.5% shared efficiency (atomics contend on few cells).

## Scale: p = 1000 to 10000 (n=5000 rows, k=5 coarse bins, kr=10 residual codes, planted pair (1, 2), `D:/Temp/pair_scale.py`, warm, synchronized)

| p | pairs | gpu_fused | njit | speedup |
|---|---|---|---|---|
| 1,000 | 499,500 | 0.11 s | 1.33 s | 12x |
| 2,000 | 1,999,000 | 0.45 s | 4.85 s | 11x |
| 5,000 | 12,497,500 | 3.43 s | 30.91 s | 9x |
| 10,000 | 49,995,000 | 16.40 s | not run (about 120 s extrapolated) | |

The planted pair (1, 2) is the best pair at every p on both backends and the MI values agree (0.6500, 0.6375, 0.6344, 0.6447 on the GPU). Cost grows with the pair count: about 0.2 to 0.3 microseconds per pair on the GPU and 2.4 microseconds on njit. No cap loses the pair in this range; the work budget (`PAIR_WORK_BUDGET`, `PAIR_WORK_BUDGET_GPU`) strides rows, not columns.

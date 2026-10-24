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

These counters were collected for v1. The v2 kernel (cell-parallel, warp-shuffle reduction, 1 barrier fewer per stage) was timed but its counters are NOT yet
collected: the counter run needs elevation (ERR_NVGPUCTRPERM) and has not been repeated. v2 versus v1 measured by wall clock: 13 to 29 percent faster.

## Measured speedups

* fused v2 versus njit: about 13x at p=400, n=5000; versus the generic GPU dispatcher: about 149x.
* `{REAL}`=float versus double: 2.2x faster on this Pascal part (1/32 double rate).
* KTC sweep (`pair_ktc_sweep3.log`, 20 cells, 119 s): CPU only at 100 pairs for n<=5000; GPU everywhere else; max abs diff to the CPU reference <= 1.8e-8.

## Candidate optimizations not yet done

* 4-byte vector loads (`char4`) with a padded row stride.
* Upper-triangle-only launch for diagonal blocks (about 5 to 10 percent of blocks are redundant).
* Warp-private histograms to lift the 4.5% shared efficiency (atomics contend on few cells).

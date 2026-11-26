# `_mdlp_gpu_presort` (device sort, runs and duplicate pairs of a block of 8 columns for the MDLP edge search)

Measured 2026-11-26 on the GTX 1050 Ti (4 GB, 112 GB/s, sm_61), GPU 72 to 80 C, SM 1708 of 1911 MHz, throttle mask 0 (`nvidia-smi` around the runs). One block: 8 float32 columns x 500000 rows, binary int8 labels.
Scripts: `D:/Temp/exp/prof_presort.py` (version 1) and `prof_presort_v2.py` (version 2) under `nvprof --profile-from-start off --print-gpu-summary`, `prof_presort_cpu.py` (cProfile), `bench_mdlp_stage.py ROWS COLS ROUNDS` (the whole edge search of a frame, device against host, interleaved).

## What the first version did, from the counters

The first version was a chain of cupy operations and was judged on wall time alone (3.82 s against 3.04 s for 160 columns of 500000 rows, so it was switched off); it had not been profiled. The profile of one block:
GPU activity 80 ms of a 95 ms call (218 ms under nvprof). The `cub` radix sort is 14 ms in all (two sorts of 4M 64-bit keys), the device-to-host copies 28.4 ms (35 percent: 15 copies, one of 32 MB of double precision values in 11.9 ms, a pageable
copy at 2.7 GB/s), `bincount` 4.9 ms, 5 gathers 3.9 ms, and about 35 elementwise cupy kernels of 0.3 to 1 ms (key build, bit tricks, `where`, casts) 25 ms. On the host side (cProfile): sorting 29 ms, downloads 29 ms,
numpy `astype` copies in `_split` 18 ms (made while the device lock was held), launch overhead and `bincount` the rest. So the sort was a sixth of the cost.

## Version 2

`mdlp_keys` (one kernel instead of 10 cupy launches) and `mdlp_unpack` (value, label and run-start flag in one pass instead of 25) fuse the elementwise chains; downloads of 1 MB and more go through page-locked memory; the runs per column come from `count_nonzero`;
`_split` runs after the device lock is released. Values, labels, distinct values, counts and deduplicated pairs are bit-equal to the host passes (tests: 5 label types x 8 row counts around the key-width boundaries, the zero signs, ties, constants).

| one block of 8 columns x 500000 rows | version 1 | version 2 |
|---|---:|---:|
| call wall (min of 5, synchronized) | 95.3 ms | 75.9 ms |
| GPU activity (nvprof) | about 80 ms | about 58 ms |
| device-to-host copies | 28.4 ms | 19.0 ms (pinned) |
| `cub` sort passes | 13.6 ms | 14.3 ms |
| fused unpack kernel | - | `mdlp_unpack` 0.8 ms |

Offline per-arch build: `_KEYS_SRC` and `_UNPACK_SRC` (8 label types) assemble on sm_61 and sm_120 without a spill.

## The whole edge search of a frame (160 columns, `per_feature_edges(method="mdlp", n_jobs=-1)`, 8 workers, edges equal in every round)

| rows | host passes: wall / process CPU | device presort: wall / process CPU |
|---:|---:|---:|
| 50000 | 0.33 s / 1.08 s | 0.47 s / 0.81 s |
| 100000 | 0.56 s / 2.41 s | 0.65 s / 1.62 s |
| 200000 | 1.19 s / 5.69 s | 1.27 s / 3.61 s |
| 300000 | 1.88 s / 9.30 s | 1.75 s / 6.33 s |
| 500000 | 3.16 s / 15.80 s | 2.87 s / 11.00 s (version 1: 3.82 s / 11.4 s) |
| 1000000 (80 columns) | 3.29 s / 17.06 s | 2.98 s / 11.38 s |

The device saves 26 to 36 percent of the CPU at every size and wins in wall from about 250000 rows with 8 workers; below that its fixed cost per block is more than the host does in parallel. The host passes run on every worker at once and the blocks go through the device one at
a time, so the row floor is `MIN_ROWS_PER_WORKER` (31250) x the workers of the call (the crossover scales with the workers; a sweep of one call cannot see that, which is why this is a size rule and not a KTC entry). `MLFRAME_MDLP_GPU_PRESORT=0` switches it off, `1` drops the floor.

## Memory

The columns of a block share the arrays downloaded for it: 8 columns x (values 8 + labels 1 + distinct values 4 + counts 8 bytes) a row, charged to the edge workers' memory budget (`_PRESORT_BYTES_PER_ROW`; 84 MB a worker at 500000 rows, and the budget lowers the worker count at big frames as for the host passes).

## Ceiling

The three passes are 35.6 CPU-ms of the 107 CPU-ms of a column's edge search at 500000 rows, so the CPU a perfect device path can free is about 36 thread-seconds for 1000 columns: about 4.5 s of wall on the 8 logical cores, in the first 35 s of a fit where the cascades share them. The stage numbers above are what this version delivers of it.

## Inside a whole fit (why it is off by default)

Paired, one after the other on the same machine state: 500000 x 1000 ALL, `BENCH_NBINS=mdlp`, n_jobs=-1, free RAM 9.6 GB, GPU 71 and 76 C at 1708 MHz. With the presort (`auto`): wall 628.5 s, peak RSS growth 5144 MB, lowest available RAM 2726 MB. Without (`MLFRAME_MDLP_GPU_PRESORT=0`): 482.7 s, 4620 MB, 3432 MB. The same selection. The stage A/B above runs with the card and the host to itself; in the fit the cascades hold the card at the time the warm-up runs (the first blocks failed with `cudaErrorMemoryAllocation` and went to the host), the host memory the blocks take (about 0.5 GB) brings the machine to the paging edge, and the stage walls rose in every stage. The cause is not separated yet (candidates in `_TRACKER.md`, PLAN-SPEED-29: the `free_all_blocks()` of the shared device pool after each block, host memory pressure, queueing on the device lock).

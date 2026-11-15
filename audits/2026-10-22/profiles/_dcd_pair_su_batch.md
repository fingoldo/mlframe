# Profile record: pair joint entropies of the dynamic cluster discovery (`_dcd_pair_su_batch`, device twin `_dcd_pair_su_gpu`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores, process elevated. Measured 2026-11-14 on an idle machine, medians of 5 after a warm call. Workload: a 20-column, 12-level code matrix over 20,000 rows and pairs drawn over it.

## What was added
`joint_entropies_gpu`: one 256-thread block per pair, the rows fill a shared `nb_a * nb_b` integer histogram (integer atomics), the threads add `p ln p` over the occupied cells and a warp-shuffle block reduction
gives the entropy. The host sums in ascending cell order, the device in another order, so a value agrees to about one ULP (the documented selection-equivalence of the multi-pair numba kernel). Pairs over the
shared-memory limit (8192 cells) keep the host value. Pairs stream in blocks, each uploading only the columns it references as contiguous rows (a column block of the code matrix; `run_column_blocks` halves the block
on OOM). `joint_entropies_of_pairs` dispatches by the kernel-tuning entry `dcd_pair_entropy_backend` (rows x pairs); the numba strided and contiguous kernels are `_joint_entropies_host`.

## cProfile (device call, 640 pairs)
5 ms in total: the block function 3 ms own time (the transposed contiguous copy of the referenced columns), `cupy.array` uploads 1 ms (5 calls), `ndarray.get` 1 ms.

## nvprof (hardware counters, elevated process)
`pair_joint_entropy` (launches per call: 1 per pair block): achieved occupancy 94.8 percent, SM efficiency 99.2 percent, DRAM read 39.6 GB/s, DRAM write 0.70 GB/s, global load efficiency 99.4 percent,
global store efficiency 25 percent (one 8-byte value per block, negligible), shared-memory efficiency 0.22 percent: the shared atomics land on a few histogram cells (12 x 12 = 144) and serialise;
per-warp private histograms are the lever, not measured here.
Whole-GPU utilization: the measured DRAM read throughput of 39.6 GB/s is 35 percent of the card's 112 GB/s peak bandwidth with SMs busy 99 percent of the time, so the kernel is bound by the atomics, not by memory.

## Wall clock (call with uploads)
| pairs | numba | device |
|---|---|---|
| 10 | 0.46 ms | 2.54 ms |
| 40 | 0.65 ms | 3.89 ms |
| 160 | 3.13 ms | 3.56 ms |
| 640 | 7.53 ms | 5.22 ms (1.4x) |

The swept choice on this host: host at 10 and 40 pairs, device at 160 and 640. The gain is small because the numba kernel is already parallel over pairs and the device call pays the column gather and transfer.
Equality to 1e-12 (`tests/feature_selection/gpu/test_dcd_pair_entropy_device.py`).

## Tried and rejected: private histogram copies (2026-11-14)
The 0.22 percent shared-memory efficiency suggested serialised atomics. A variant with C private copies of the histogram per block (copy chosen by lane mod C, C = 2, 4, 8 and one copy per lane = 32) was timed against the kernel, interleaved, 15 pairs of runs, 640 pairs x 20,000 rows, resident data, medians:
base 0.56 to 0.58 ms in every run; C=2 0.568, C=4 0.568, C=8 0.557 (equal within noise), one copy per lane 0.887 ms (slower: 18 KB of shared memory per block halves the resident blocks and the 32-way reduction costs more than it saves). Results identical to the bit. The effective read rate of the base kernel is 183 GB/s of code traffic (L2 re-reads), so the kernel is not atomic-bound; the shared-efficiency counter measures requested versus transferred bytes of the atomics and is not a bottleneck indicator here. Reverted.

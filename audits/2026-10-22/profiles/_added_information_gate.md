# Profile record: added-information gate (`_added_information_gate`, device twin `_added_information_gpu`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores. Measured 2026-11-14 on an idle machine, medians of 5 after a warm call. Workload: candidates of 10 levels over 30,000 rows, a 4-level source, a binary target.

## What was added
`_added_information_gpu.g_statistics_gpu`: one 256-thread block per candidate, the rows fill a shared integer (source, candidate, target) count table plus its marginals (integer atomics, order independent),
the threads add `n * (ln n_scy + ln n_s - ln n_sc - ln n_sy)` over the occupied cells, one warp-shuffle block reduction. `added_information_p_many` scores the candidates of one source in one call and keeps the host
p-value code (degrees of freedom, sparsity rule, chi-square tail), so both paths decide alike; candidates whose table exceeds 4096 cells use the host path. Candidates stream in blocks of their code matrix
(`run_column_blocks`, halved on OOM); source and target upload once. `_grouped_quantile_fe._p_added_by_column` now bins the columns on the pool, then calls `added_information_p_many` once per source.
Kernel-tuning entry `added_information_backend`, swept by rows x candidates.

## cProfile
Before the change `added_information_p` ran once per candidate inside the binning pool (the `np.bincount` of the joint key and the log terms of a (bs, bc, by) array per call); the host loop below is that cost.

## nvprof / nsys
`added_information_g`: launches per source group: 1 per candidate block. nvprof counters fail on this machine today (superseded by the Hardware counters section below), so occupancy is theoretical: 256 threads, launch bounds 256,
dynamic shared memory (cells + marginals) x 4 bytes, 4 x 10 x 2 + 40 + 8 + 4 = 132 ints = 528 bytes in the sweep, so 8 resident blocks per SM. Kernel alone, 256 candidates x 30,000 rows: 0.73 ms, effective
read throughput 126.6 GB/s of the code traffic (above the card's 112 GB/s: the source and target vectors are re-read from L2 by every block).

## Wall clock (p-values for all candidates, with uploads)
| candidates | host loop | device G | host G (numpy) |
|---|---|---|---|
| 4 | 1.1 ms | 1.1 ms | 0.8 ms |
| 16 | 4.3 ms | 3.0 ms | 3.1 ms |
| 64 | 17.5 ms | 10.2 ms | 13.1 ms |
| 256 | 70.1 ms | 36.2 ms | 50.3 ms |

The kernel is 0.73 ms of the 36.2 ms at 256 candidates; the rest is the conversion and upload of the 256 x 30,000 int64 code matrix (a 1.9x gain over the host loop, limited by the transfer, not the kernel).
The swept choice on this host: host at 4 and 16 candidates, device at 64 and 256. Equality: p-values equal the host to 1e-9 relative, oversize tables take the host path (`tests/feature_selection/gpu/test_added_information_device.py`).

## Hardware counters (nvprof --metrics, elevated shell, 2026-11-14)
Superseding the theoretical occupancy stated above (the counters were blocked by the driver's restriction until the process ran as administrator). Kernel alone, last sweep size, one profiled call after a warm call. Pascal sm_61: Nsight Compute reports the device as unsupported, nvprof is the tool.

| kernel | achieved occupancy | SM efficiency | DRAM read | DRAM write | global load eff. | global store eff. | shared eff. |
|---|---|---|---|---|---|---|---|
| added_information_g | 95.0 % | 97.4 % | 43.4 GB/s | 0.61 GB/s | 99.9 % | 25.0 % | 0.02 % |

Shared efficiency 0.02 percent: same cause, the table has 80 cells.

# Profile record: rounding and digit columns (`_numeric_decompose_fe`, device twin `_numeric_decompose_gpu`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores. Measured 2026-11-14 on an idle machine, medians of 5 after a warm call.

## What was added
`_numeric_decompose_gpu.decompose_columns_gpu`: one launch makes every (source column, output) pair of a column block, kind 0 `rint(x / p) * p` in the pipeline's criterion value type (same as `np.round`, half to even),
kind 1 `floor(x * 10^k) mod 10` in double with non-finite values mapped to 0, exactly as `_digit_extract_njit`. Equal to the host columns bit for bit (`tests/feature_selection/gpu/test_numeric_decompose_device.py`,
ties at half-integers, negatives, NaN and inf included). Both generators pick the backend per call from the kernel-tuning entry `numeric_decompose_backend`; source columns stream in column blocks (`_device_blocks`,
halved on OOM); the existing scoring loop already works in blocks of source columns (`block_cols`).

## cProfile (20 columns x 500,000 rows, rounding + digit generators, 0.33 s)
The wall is thread-pool waiting (`acquire` 0.19 s over 268 calls, the host columns are made side by side on the edge pool) and the pandas frame stack (0.086 s); the elementwise numpy/numba passes release the GIL.

## nvprof / nsys
`decompose_columns`: launches per call: 1 per column block (grid: rows x outputs x columns). nvprof counters fail on this machine today (profiling error 4211): occupancy is theoretical, 256 threads per block,
launch bounds 256, no shared memory, so residency is limited only by registers (full occupancy expected). Measured throughput of the kernel alone (16 columns x 100,000 rows, 7 outputs): 3.37 ms = 28.5 GB/s
of the card's 112 GB/s bandwidth; the digit outputs divide and floor in double, which the sm_61 card runs at 1/32 rate, so the kernel is arithmetic bound, not memory bound.

## Wall clock (7 outputs, 100,000-row columns, with upload and download)
| columns | host | device |
|---|---|---|
| 4 | 15.2 ms | 15.3 ms |
| 16 | 63.7 ms | 68.5 ms |

The swept choice on this host: host at the smallest and largest size, device in between (a tie within noise); the host path overlaps columns on the edge pool, the device pays the transfer of the columns and of the 7x larger outputs.
Not a speedup for host columns; the twin is for the resident path where the columns are already on the card.

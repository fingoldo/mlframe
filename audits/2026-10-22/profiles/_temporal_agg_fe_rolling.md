# Profile record: rolling time-window aggregation (`_temporal_agg_fe_rolling`, device twin `_rolling_window_gpu`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores, elevated process (hardware counters). Measured 2026-11-15, medians of 5 after a warm call. Workload: 500 entities in (entity, time) order, a numeric time axis, a noisy value column with 2 percent NaN, a window of 30 time units, the standard deviation statistic.

## What was added
The host kernel `_rolling_stat_past_only_njit` walks, for every row, the entity's history newest to oldest through a `prev` linked list while the time stays inside the window, and folds the strictly past finite values into count, sum, Welford mean and m2, minimum and maximum. `_rolling_stat_past_only` now asks `rolling_dispatch` first: one thread takes one row and walks the same links in the same order, so count, minimum and maximum are identical and the mean and the standard deviation follow the same Welford recurrence. The links are built on the host in one stable argsort. Int64 nanosecond times (datetime columns) and double times (numeric columns) are two compiled variants. The host/device choice is the kernel-tuning entry `rolling_window_backend` (rows axis), a device fault falls back to the host with a throttled warning. The data is one column block per call: the arrays of one value column and its entity codes, which are the whole input of one rolling feature.

## cProfile (device call, 500,000 rows)
The host side of the call is the stable argsort of the entity codes plus four uploads and one download; the walk itself is one kernel launch, so there is no host loop to remove.

## nvprof (hardware counters)
`rolling_stat_past_only` (launches per call: 1 for all rows): 19.3 ms, 85.8 percent of the traced GPU time; the uploads are 9.2 percent (3 copies, 2.1 ms) and the download 5.1 percent (1.1 ms). Achieved occupancy 0.63, global load efficiency 90.1 percent, DRAM read throughput 0.6 GB/s (the walk re-reads the same few hundred cache lines of one entity, so it is served from cache, not memory), double-precision FLOP efficiency 52.5 percent of a card with a 1/32 rate: the kernel is bound by the double divisions of the Welford step, not by memory. Throughput: 500,000 rows in 19.3 ms is 26 M rows per second of GPU time; utilization of the SMs is high (occupancy 0.63) with no idle launches between the 1 launch.

## Wall clock (numba walk against the device call with uploads)
| rows | host | device |
|---|---|---|
| 20,000 | 0.68 ms | 0.63 ms (1.1x) |
| 100,000 | 11.4 ms | 2.9 ms (3.9x) |
| 500,000 | 270.1 ms | 30.7 ms (8.8x) |

The tuner sweep picks the device from 20,000 rows up on this card. The device equals the host for all five statistics, for datetime and numeric times, for sorted and interleaved entities, with ties in time and NaN values (count, minimum and maximum exactly, mean and standard deviation to 1e-12); the wrapper test counts the device launcher calls and a fault test checks the host fallback.

# `_mrmr_fe_step/_conditional_maxt_gpu` (conditional MI of every candidate column for the observed and the permuted targets, one block per (target, column))

Measured 2026-11-26 on the GTX 1050 Ti (sm_61, 112 GB/s). Scripts: `D:/Temp/exp/prof_condmi.py N P B KZ KW KY [prof]` (the production call with the profiler window), `D:/Temp/exp/proto_condmi_u8.py` (both kernels on the same inputs).

## Why it was looked at

In a 300000 x 300 fit (py-spy, main thread) the waits for this call were 4.6 s, the second largest device wait on the critical path after the order-2 floor kernel.

## Version 1 (`_SRC`, `cond_mi_columns`), counters at 300000 rows x 300 columns x 41 targets, 16 strata, 10 bins, binary target

| | |
|---|---:|
| kernel (nvprof summary) | 209.7 ms, 80 percent of the call's GPU time (host-to-device copies 30 ms) |
| achieved occupancy | 0.62 |
| IPC / issue slots | 1.79 / 41.7 percent |
| stall: memory dependency | 72.3 percent |
| stall: execution dependency / sync | 18.9 / 1.5 percent |
| DRAM read | 47.3 GB/s |
| global load efficiency | 99.99 percent |
| `flop_dp_efficiency` | 1.8 percent |

Memory-bound: each row of each block reads a 2-byte code, a 4-byte stratum and a 4-byte target; the logarithms of the epilogue are not the cost.

## Version 2 (`_SRC_U8`, `cond_mi_columns_u8`)

Code, stratum (at most 64) and class (at most 16) each fit a byte: 3 bytes a row instead of 10, and a thread reads 4 rows with one 4-byte load of each array (rows padded to a multiple of 4,
the pad never counted; technique 1 of CUDA.md section 19). The counts are integers and the epilogue is the same code, so the values are bit-identical.

| shape | version 1 | version 2 |
|---|---:|---:|
| 300000 x 300 columns x 41 targets, kz 16, kw 10, ky 2 | 213.6 ms | 78.7 ms (2.71 x) |
| 100003 x 120 x 33, kz 64, kw 12, ky 3 | 30.4 ms | 15.8 ms (1.92 x) |
| 5 x 7 x 3 | 0.1 ms | 0.1 ms |

The codes also go up as bytes (half the upload of the int16 matrix). Version 1 stays for codes wider than a byte. Offline per-arch build (sm_61, sm_120) without a spill; bit-equality
test against version 1 at three shapes and two seeds (`tests/feature_selection/test_conditional_maxt_device.py`).

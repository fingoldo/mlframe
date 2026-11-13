# `_hermite_fe_mi` (batched plug-in MI dispatch and the polynomial-evaluation CUDA kernels)

Measured 2026-11-13, GTX 1050 Ti (sm_61, 112 GB/s, double rate 1:32), one `float32` column of 1 000 000 rows, 6 coefficients, `{ACC}` = double (the recurrences accumulate in double for every x type, FC11-38).

## What the module has on the device

Four evaluation kernels, one launch per evaluated column (`hermeval_kernel`, `legval_kernel`, `chebval_kernel`, `lagval_kernel`: a thread per row walks the three-term recurrence and the Clenshaw-free sum `s += c[k] * p_k`), plus the resident batched plug-in MI helpers (`_plugin_mi_classif_batch_cuda_resident`) that the dispatcher reaches on request. The default of the batched MI dispatch is the njit backend by an end-to-end measurement (114 s against 368 s for the whole 100000-row fit, selection identical: the dispatcher's own docstring; `MLFRAME_MI_BACKEND=cuda` forces the device), so the kernels below are the device part that other FE modules call with resident data.

## This pass: the per-step double division

`legval_kernel` and `lagval_kernel` divided `1.0 / k` inside the loop, once per row and degree: a double division costs tens of instructions on a card with a 1:32 double rate. The reciprocals now come from a 64-entry table in shared memory filled once per block (degrees beyond 63 keep dividing); the correctly rounded value is the same, so the outputs are bit-identical (`test_hermite_polyeval_reciprocal_table.py`, which also checks degrees 2, 3, 6, 40 and 70 against a host recurrence). Kernel time per launch, 1 000 000 rows x 6 coefficients, isolated: Legendre 1.326 -> 0.853 ms, Laguerre 1.367 -> 0.900 ms; 10 coefficients 2.59 -> 1.55 ms and 2.55 -> 1.59 ms; 100 000 rows 0.167 -> 0.108 ms. Under nvprof (6 launches): `legval_kernel` 1.28 -> 0.83 ms, `lagval_kernel` 1.30 -> 0.85 ms.

## nvprof summary (6 launches of each, 1 000 000 x 6)

| kernel | per launch | share |
|---|---:|---:|
| `lagval_kernel` | 0.846 ms | 32.6 % |
| `legval_kernel` | 0.827 ms | 31.8 % |
| `hermeval_kernel` | 0.468 ms | 18.0 % |
| `chebval_kernel` | 0.336 ms | 12.9 % |

Launches per call: exactly one kernel launch per evaluated column and basis (the host launch costs about 0.8 ms of Python time per 20 launches in cProfile, i.e. the launcher is not the cost; the call is device time).

## nvprof metrics (`lagval_kernel`, replayed)

Achieved occupancy 0.946, multiprocessor activity 99.6 %, global load efficiency 55 % (the per-thread coefficient reads are a warp-wide broadcast of one 8-byte word, which counts as partly useful sectors; they are served from L1), global store efficiency 100 %, device-memory read throughput 4.5 GB/s and write throughput 8.9 GB/s of 112: the kernel is bound by the FP64 pipeline (double recurrence), not by bandwidth, which is why replacing the divisions mattered.

## Not changed, with the reason

The recurrences stay in double: float accumulation of high-degree terms loses digits, and the FP64 cost is the price of the exactness the host kernel has. A float-accumulating variant was measured in the same session at 0.20 ms against 0.30 ms for four coefficients (1 000 000 rows): below 1.5x for the cheap bases, and not adopted for that gain.

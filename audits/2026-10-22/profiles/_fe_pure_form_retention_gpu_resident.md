# Profile record: batched non-separability gate of the pure-form retention (`_fe_pure_form_retention_gpu_resident`, numba twin `_pure_form_gate_njit`)

Host: GTX 1050 Ti (sm_61, 4 GiB, 112 GB/s, double rate 1:32), 8 logical cores, elevated process (hardware counters). Measured 2026-11-15, medians of 3 after a warm call. Workload: the regression pool of the STRICT route: 8 raw operands, 40 candidate forms over random operand pairs (a ratio, a sum and a noise form in turn), a target carrying one ratio; 100000 rows unless stated.

## What the module is
For every candidate pair the gate regresses the candidate form on the 12-column additive basis of its two raw operands (centered), and keeps the form when the residual is not negligible and correlates with the target. The default path is a `LinearRegression` per candidate on the host and is untouched. Under STRICT residency the pool is scored as a batch: the four scalars of every pair (form std, residual std, residual sum of squares, residual . y) come from the normal equations with a trace-scaled ridge, on the device (`pair_gram_blocks` and one batched solve) or on the host (`pair_gate_scalars_njit`, a `prange` over the pairs with its own 12 x 12 elimination, no BLAS threads). The kernel-tuning entry `pure_form_gate_backend` chooses the side by row count; both give the same verdicts.

## Column blocks (streaming)
The form-value columns are uploaded a column block of pairs at a time (a block is a quarter of the room under this thread's pool limit over four copies of its columns), so the working set does not grow with the number of candidates; the centered operand blocks (one (6, n) block per distinct operand) and their per-operand Gram sums stay resident for every block. A test with a room of about four form columns forces several blocks and compares the verdicts with the one-block run.

## cProfile (one device batch, 100000 rows, 40 pairs)
0.122 s for 2776 function calls: 0.048 s in `_upload_pure_form_inputs` (scrub and stack of the host columns: the host side of the upload, 0.067 s cumulative), 0.025 s in the one blocking copy of the scalars, 0.011 s scrubbing 49 columns, 0.005 s building the operand bases. The device work itself is a small part of the wall (see the counters).

## nvprof (hardware counters)
Launches per batch: 2 `pair_gram_blocks` (the per-operand diagonal blocks once, the per-pair cross blocks once per column block), 18 reductions (`cupy_mean`), 9 variance cores, 79 small copies and about 45 elementwise launches for the centering, the basis stack and the solve inputs. GPU time 35 ms: `pair_gram_blocks` 21 percent (7.4 ms for two launches), host-to-device copies 17 percent (6.1 ms), `cupy_mean` 13 percent, copies 12 percent, `cupy_var_core` 10 percent.

| kernel | achieved occupancy | other counters |
|---|---|---|
| `pair_gram_blocks` | 0.19 | double-precision FLOP efficiency 87 percent of the 1:32 peak (the kernel is compute bound on the card's double pipe, as a 12 x 12 Gram is), global load efficiency 90.6 percent after the operand blocks became feature-major (was 25.8 percent: a 48-byte row stride) |
| `cupy_mean` | 0.25 | global load efficiency 33 percent (the reductions are the elementwise tail of the centering) |

Whole-GPU utilization: 35 ms of GPU time in a 114 ms wall (31 percent busy); the rest is the host side of the upload (scrub, stack) and the single blocking copy, so at 40 pairs the batch is host bound, not kernel bound.

## Wall clock (host numba batch against the device batch, same verdicts)
| rows | host | device |
|---|---|---|
| 5,000 | 10.7 ms | 25.9 ms (host wins) |
| 30,000 | 74.9 ms | 39.9 ms (1.9x) |
| 100,000 | 282 ms | 114 ms (2.5x) |

Before the batch: 200 pairs x 40 operands at 100000 rows took 957 ms in the per-pair cupy loop and 123 ms with `pair_gram_blocks` (7.8x, see the tracker entry TL11-04). The numba twin and the device batch agree on the four scalars to 1e-8 of the form's own sum of squares (a form that is exactly linear in its operands leaves a residual at rounding level, where the two solvers differ in the last digits and the verdict is the same).

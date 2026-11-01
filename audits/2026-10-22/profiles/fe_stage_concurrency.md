# Running the independent FE stage cascades concurrently (ceiling and why it needs a device scheduler)

Contract: `_fe_stage_cascade_run._run_fe_stage_cascades` runs `early_a` first and then `early_b`, `mid_a`, `mid_b`; each of the last three reads the SAME step-input frame and returns it plus its own new columns, folded in
the fixed cascade order, so no stage consumes another's output (inputs are independent; shared mutable state is the estimator's list attributes, appended in stage order, e.g. `hybrid_orth_features_`).

## Stage walls at 100000 x 100, every family, sequential (GTX 1050 Ti, one process)

| stage | seconds (run 1 / run 2) |
|---|---|
| `early_a` | 50.1 / 44.2 |
| `early_b` | 11.7 / 14.4 |
| `mid_a` | 34.3 / 34.2 |
| `mid_b` | 19.3 / 18.4 |
| cascade total | 115.5 / 111.2 |

The three independent stages sum to 65 s (run 1) and the longest is 34 s, so a perfect overlap would save up to about 31 s of the cascade (the host-bound majority: GPU utilisation of the whole fit is 18%, process CPU
178% of 800%).

## Attempt (bench `D:/Temp/bench_stage_threads_all.py`, adapted from `_benchmarks/bench_fe_stage_threads.py`, copies of the estimator per stage, one process, fit cache cleared between modes)

The threaded half did not finish. With three stages on one 4 GB card each stage allocates from the shared cupy pool (cap 2.58 GB): the log fills with `Out of memory allocating 457,600,000 bytes (allocated so far
2,245,231,616 ...)` and `_radix_select_interior_edges failed, falling back to the exact percentile path`, and the process ended before the wall line. (A first attempt returned in 3.1 s because the second fit of
identical data came from `MRMR._FIT_CACHE`; the cache must be cleared between modes.) The earlier note in `_fe_stage_cascade_run.py` (no gain at 300k rows in a 9 s fit, "they queue on the same device") measured a
different regime; at this size the limit is device MEMORY, not device time.

## What a working design needs (not built)

* A device scheduler that hands each stage a VRAM budget (a share of the card's free memory, from `memGetInfo` of that device) and a device id; the pool cap and the cushion guard of `_fe_gpu_vram.py` become per
  stage, not process-wide. On several GPUs of different size the shares follow free memory and the number of SMs.
* Per-device resident-operand caches (`_fe_resident_operands` keys are content-only today) and per-device kernel modules (`RawModuleCache` is keyed by element type only).
* Estimator copies per stage with the list attributes merged back in the fixed stage order (deterministic selection), as the bench does for the frame.
* A measured gate: concurrency only when the free VRAM covers the stages' budgets, else the sequential cascade (the CPU fallback of every device path already exists).

## Implemented and measured (flag `MLFRAME_FE_STAGE_CONCURRENCY=1`, off by default)

`_fe_device_scheduler.py` (device plan + private pool per stage), `_mrmr_fit_impl/_fe_stage_concurrency.py` (private estimator copies, fold in stage order) and the switch in `_fe_stage_cascade_run.py`.
The state the stages share, found by diffing the estimator and `recipes` after each stage of a sequential run: the lists `hybrid_orth_features_` and `_fe_rejection_records_` (every stage appends), the per-family
`*_features_` lists (created empty by early_b, filled by mid_a / mid_b), a few scalars; the recipes of the three stages are disjoint families. The merge rules reproduce the sequential state exactly
(`test_fe_stage_concurrency_merge.py`), and fits at 20000 x 30 and 100000 x 100 give the same support, the same `hybrid_orth_features_` and the same rejection count in both modes.

Measured at 100000 x 100, every family except the surrogate, one process, fit cache cleared between the modes:

| run | sequential s | concurrent s |
|---|---|---|
| numba parallel-kernel lock on | 154.3 | 151.2 |
| lock off under the TBB layer | 154.3 | 155.2 |

No gain either way: the three stages did run together (plan: budgets 0.52 / 1.26 / 0.67 GB on the one card), so the time is not waiting for a lock or for memory. What the sampler showed holds: the stage
time is host Python and numpy that holds the GIL, and a GPU that is idle 82% of the time because only one thread feeds it. Threads cannot overlap GIL-bound work; the remaining ways are to move the host work
into nogil kernels or the device (the per-family kernels of the other notes), or to run the stages in separate processes with their own copies of the data (large memory, and the resident caches are per process).
The code stays, off, for hosts where the stages are not GIL-bound (a faster host, free-threaded Python) and as the base of a multi-device plan.

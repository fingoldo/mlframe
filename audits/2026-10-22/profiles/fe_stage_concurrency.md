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

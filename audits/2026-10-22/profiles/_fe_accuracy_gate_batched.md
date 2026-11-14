# _fe_accuracy_gate_batched.py (batched held-out uplift of engineered columns) with `_fe_accuracy_gate_kernel` and `_fe_accuracy_gate_newton`: profile record (2026-11-14)

Family: `_fe_accuracy_gate_batched` (the batched driver, the tuner entry `fe_accuracy_gate_batch`, the VRAM-sized blocks), `_fe_accuracy_gate_kernel` (the fused Newton + held-out AUC kernel, one block per design and fold) and `_fe_accuracy_gate_newton` (the numpy reference fit the driver mirrors). The per-kernel counters are in `_fe_accuracy_gate_kernel.md`; this record adds the fit-level view.

## cProfile of the whole fit (100000 x 100 default fit, one warm pass)

| function | calls | own s | cumulative s |
|---|---|---|---|
| `keep_engineered_over_source` | 3 | 0.00 | 0.13 |
| `measure_feature_uplift` | 4 | 0.00 | 0.17 |
| `_held_out_probe_score` (the per-candidate host probe) | 80 | 0.01 | 0.17 |

The gate is 0.17 s of the default fit: `batched_uplifts_for_pairs` was entered 3 times and `_use_device` chose the host each time (0.0 s, no `batched_uplift` or `_fold_auc_launch` call), and the 80 designs were probed on the host (`_held_out_probe_score`); below the tuner's candidate count the host pool wins (fallback 24 candidates; the entry `fe_accuracy_gate_batch` decides per host). The fused kernel therefore made no launch in this fit: kernel launches per fit 0, launches per batched call 1 (one launch covers every design and fold). The gate matters at hundreds of columns, where the host probe costs about 26 ms per candidate (about 65 thread-seconds at 1000 columns).

## Device counters (nvprof, 96 designs x 10 folds x 5000 rows, k = 2, from `_fe_accuracy_gate_kernel.md`)

achieved occupancy 0.43, SM efficiency 97.7 % (the card's six SMs are almost always running a block), IPC 0.43, warp execution efficiency 59.7 % (the Newton step and the Cholesky factor run on thread 0 by design), 99.68 % of the GPU activity of the call is this one kernel, 100 ms per launch of 960 blocks. The kernel is compute-bound on the small dense solves, not bandwidth-bound: throughput is limited by the issue rate (IPC 0.43), the memory system is not the limit; the utilisation figure of the card during a call is the SM efficiency above.

## Streaming

The engineered candidates go through in column blocks: `_device_block_designs` sizes a block from the free VRAM (a third of it over the bytes of one design, at least one design), the host keeps the rest, and a device fault hands the candidates back to the host probe (`nan`).

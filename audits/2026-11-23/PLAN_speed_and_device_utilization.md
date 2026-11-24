# Plan: speed and device utilization of the MRMR fit (queued behind the algorithm work, owner 2026-11-24)

Order fixed by the owner: first finish every task that improves the MRMR ALGORITHMS (the unary and interaction credit, the target-side port, the multiclass criterion, snapped closed forms with a fitted frequency or exponent, the red tests), then return to speed. The new mechanisms (unary credit, interaction credit, credit parsimony, the pair sufficiency gate, the omnibus test) add stages that have never been profiled, so new hotspots, new GPU twins and new KTC entries are expected.

## 1. Re-run and profile

1. Run the 100k x 1000 ALL benchmark again (warm, with the hardware state recorded: GPU and CPU temperature, mean clocks, free RAM, free VRAM, stale nvprof check), profile it (cProfile of the host side first, the sampler timeline second) and optimize the hotspots, each with a profiled GPU twin where the stage has one and a KTC entry measured only after the kernel is profiled and optimized (CUDA.md).
2. Make 1M x 1000 ALL pass on this machine (it fails with an out-of-memory error today). The cause named by the owner: engineered columns are stored for all rows in float32. Replace that with storing their bins, and the values on a subsample only, streamed by feature blocks (no row cap on the selection itself). Verify with `rss_scale_fit.py` at 200k -> 500k -> 1M.

## 2. Idle devices: a per-stage ledger, then a scheduler

The owner has seen stretches where the CPU, the GPU or both are idle during a fit, with a thousand columns and about eighty FE families that could be evaluated in parallel. A GPU idle for tens of seconds is a defect to be explained, not a measurement to be accepted.

1. **Per-stage record.** Every MRMR step writes, into the existing sampler timeline (`RSS_TICK_S`, `tick_utilization.py`, `timeline_idle_windows.py`, `hardware_state.py`): the timestamp of its start, the timestamp of its end, the size of its input data in GB, and the device it ran on (cpu or gpu, and which gpu). The sampler already reports CPU percent, GPU percent, free RAM and free VRAM in GB; the stage record is joined to it on time.
2. **The question asked before every stage.** For each stage, is there, in the window before it, a device with low load and enough free memory to hold the stage's current data? If there is, why was the stage not started earlier? The answer goes into a table (stage, window, idle device, free memory, blocking dependency or missing implementation), and every row of the table is either closed by a change or explained by a real data dependency.
3. **Use ALL executors.** The owner rents servers with 128 and more cores and sometimes several GPUs. One saturated GPU is not enough if 127 of 128 cores idle for the whole fit. For every stage the work available at that point (columns, FE families, candidate pairs, permutation batches) is split over all executors (every CPU core group, every GPU) in proportion to what each executor can do: a cost model of seconds per work item per device, measured by the existing KTC machinery and refreshed from the stage records, and a dispatcher that assigns the available items by that model. The goal metric is the one already adopted (utilization as the efficiency measure): CPU percent and GPU percent timelines with the sync counts, next to the wall time, on a 128-core configuration as well as on this machine.
4. Independent FE families run concurrently where their inputs are ready (the stage graph of the fit gives the dependencies); a stage that waits for a device must have a recorded reason.

## 3. Acceptance

- The per-stage ledger exists for every stage of a fit and is part of the profiling output; `timeline_idle_windows.py` reports no window with both devices under-used that the ledger cannot explain by a dependency.
- 100k x 1000 ALL: wall time, CPU and GPU timelines before and after; 1M x 1000 ALL completes without an out-of-memory error.
- The dispatcher is exercised with a simulated many-core configuration (the cost model with 128 workers) and a multi-GPU configuration as unit tests; it must reproduce the single-device behaviour when one device is available.

Related: `project_loop_work_plan_2026_10_25`, `feedback_utilization_is_the_efficiency_metric`, `feedback_check_idle_cpu_gpu_at_scale`, `project_mrmr_host_to_device_priority`, `project_last_task_matrix_storage_gpu_vs_cpu`, `project_mrmr_1m_x_1000_completed_2026_11_23`, `feedback_every_fe_module_needs_profiled_gpu_path`.

## Status 2026-11-24 (first speed round)

100000 x 100 all families, quiet machine: 146 s (the unary credit at its slowest) -> 46.8 s; the per-stage ledger (`BENCH_LEDGER`, `stage_idle_report`, `ledger_timeline.py`) shows both devices idle for 1.1% of the fit. What the ledger said and what was done:

- The stage that dominated, `stage:unary_credit` (44 s per call), was the bootstrap null of the binary-target credit: now a fused kernel (record `_unary_credit_logit_cuda.md`), 1.2 s per call in the fit.
- `pair_maxt_null_floor` 13.6 s: 7 s of it was counting the occupied cells of the pair pool twice, serially, through a stride walk; now 1.7 s.
- The pair family gate (5 calls, about 1.3 s each): scored every column; now the distinct partitions only, through one-launch gather kernels (`_family_maxt_gate_device.md`).
- Foreign load is a measurement hazard of this machine (VS Code processes 30 to 40% CPU): single walls of 59 to 74 s were taken under it; compare quiet runs only.

Not yet done from section 2: the dispatcher over all cores and GPUs by measured capability (the three FE lanes are hand-balanced; a dynamic scheduler over their 7 hybrid units would be bound by the largest unit, about 6.6 s), the 100k x 1000 run, the 1M x 1000 storage change.

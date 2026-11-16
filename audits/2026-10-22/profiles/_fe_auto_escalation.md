# `_fe_auto_escalation` (+ carved `_fe_auto_escalation_proposers`): escalation of under-delivering pairs to the richer bases

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. Bench: `MRMR(random_state=42).fit` on 100000 rows x 6 normal columns, `y = sin(3.7 x0) x1 + 0.1 noise` (the fixture of `test_biz_val_escalation_recovers_inner_frequency_pair_and_replays`), cProfile with `MLFRAME_FE_STAGE_CONCURRENCY=0` so the worker threads are in the profile. The escalation recovers `esc_fourier_mul(x0,x1)`.

## Host profile (cProfile of the whole fit, 183.3 s with the profiler)

`run_fe_auto_escalation` 0.620 s cumulative: `_propose_pair_candidates` 0.560 s over 5 candidate pairs (`_propose_poly` 0.332 s, of which `_best_pair_basis` 0.122 s and `_basis_block` 0.086 s over 40 calls; `_propose_fourier_both_warps` 0.212 s, `_detect_jobs` 0.103 s), `_subsample_decision_inputs` 0.032 s, `_cmi_gate_accepted` 0.024 s, `_materialise_escalation_admitted` 0.014 s. The stage is 0.3 % of the fit: it works on a deterministic stride subsample, so its cost does not grow with the row count, and no Python loop dominates.

## Launches per call and the GPU profile

The escalation launches no kernel of its own. Its device work is the admission gate (`_cmi_gate_accepted`, 24 ms for the pair of survivors here) on the batched CMI and the CMI redundancy gate; the nvprof table, launches per call, occupancy and whole-GPU throughput of those kernels are in `_fe_cmi_redundancy_gate.md` and `_fe_batched_mi.md`. A GPU path for the proposers would not pay: they fit 5 pairs on a row subsample for 0.56 s in total.

## Backend choice and memory

Choice: the gate's own (`_cmi_gpu_enabled`, STRICT / AUTO by fit size, tuned null backend). Memory: one row block (the stride subsample) of the operand columns per eligible pair, at most `max_pairs` pairs.

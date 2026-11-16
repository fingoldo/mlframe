# `_fe_retention_subsumption` (engineered-versus-engineered subsumption guard of the retention passes)

Measured 2026-11-16 on the same host as the pure-form retention record (GTX 1050 Ti, sm_61, 112 GB/s, 8 logical cores).

## What the profile shows

The guard conditions a re-attached form on the engineered survivors admitted before retention, by the CMI redundancy gate. It has no kernel of its own: its cost is the gate's observed CMI and conditional permutation null. In the 100000-row retention pass of `_fe_pure_form_retention.md` (stub MRMR without engineered incumbents) `retention_form_is_subsumed` is not reached (0 calls in the cProfile of the pass: 1174644 function calls, 2.35 s), so the guard adds no time when there is nothing to condition on; with incumbents each call is one gate scoring of one candidate against the support, whose figures per pass are in `_fe_cmi_redundancy_gate.md` (a 30-candidate pass: 3.5 to 3.99 s on the device, about 10.5 s on the host at 100000 rows) and, for the null, `_fe_cmi_perm_null_gpu.md`.

## Launches per call and the GPU profile

Launches per call: those of one `_conditional_perm_null` and one batched CMI scoring in the gate record (about 30 launches per null call: `flat_strata_key_pack` at occupancy 0.89 and 65 to 70 GB/s read of 112 GB/s, `strata_xy_entropy`, the cuRAND draw, the batched joint histogram). The nvprof table of the gate pass is in `_fe_cmi_redundancy_gate.md`; the whole-GPU throughput of a STRICT gate pass there is 3.5 to 3.8 s of device-bound work (the batched cupy sort 1.77 s of 3.86 s cProfile), with `flat_strata_key_pack` reading 65 to 70 GB/s of the card's 112 GB/s.

## Backend choice and memory

Choice: `_cmi_gpu_enabled` (STRICT / AUTO by fit size) and the `cmi_perm_null_backend` entry for the null, as in the gate. Memory: one re-attached form and the admitted incumbents' binned columns, a column block of at most the survivors, plus the gate's own bound (the VRAM cushion admission of the null's permutation matrix).

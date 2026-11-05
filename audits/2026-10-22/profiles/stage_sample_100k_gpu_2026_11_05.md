# Stage sample, 100k x 100, every family, GPU, 2026-11-05

Run: `stage_sample.py 100000 100 gpu` (one process, warm-up fit on 20000 x 30 first, MLFRAME_FE_GPU_STRICT / CMI_GPU / FE_VRAM_F32 / FE_GPU_DISCRETIZE / FE_GPU_BINNING on), tree at the commit that carries the zero-baseline launch geometry (45af52fd0). No profiler; the main thread's innermost stage frame is sampled every 0.2 s.

* Fit wall: 76.1 s (target ~100 s). One run, not repeated, so it is a single sample; an earlier warm 100k fit was recorded at 85-86 s.
* The main thread spends 35.4 s of it waiting in `run_stages_concurrently` for the worker threads (`_wait_for_tstate_lock`): the work itself is in the stage threads, so the per-stage table below under-counts it.
* Innermost stage frames (seconds on the main thread): `_fit_body_run_fit_impl` 33.8 (the wait above), `_fe_stage_cascade_early_a_part1` 11.4, `honest_null_columns_device` 3.0, `generate_extra_basis_features` 2.4, `score_prospective_pairs` 1.6, `propose_gradient_interaction_pairs` 1.4, everything else 1.2 s or less.
* Not measured here: CPU and GPU utilization timelines and nsys occupancy for this tree; the next profile should use `util_timeline.py` and the nsys window.

# Stage sample, 100k x 100, every family, GPU, 2026-11-05

Run: `stage_sample.py 100000 100 gpu` (one process, warm-up fit on 20000 x 30 first, MLFRAME_FE_GPU_STRICT / CMI_GPU / FE_VRAM_F32 / FE_GPU_DISCRETIZE / FE_GPU_BINNING on), tree at the commit that carries the zero-baseline launch geometry (45af52fd0). No profiler; the main thread's innermost stage frame is sampled every 0.2 s.

* Fit wall: 76.1 s (target ~100 s). One run, not repeated, so it is a single sample; an earlier warm 100k fit was recorded at 85-86 s.
* The main thread spends 35.4 s of it waiting in `run_stages_concurrently` for the worker threads (`_wait_for_tstate_lock`): the work itself is in the stage threads, so the per-stage table below under-counts it.
* Innermost stage frames (seconds on the main thread): `_fit_body_run_fit_impl` 33.8 (the wait above), `_fe_stage_cascade_early_a_part1` 11.4, `honest_null_columns_device` 3.0, `generate_extra_basis_features` 2.4, `score_prospective_pairs` 1.6, `propose_gradient_interaction_pairs` 1.4, everything else 1.2 s or less.
* Not measured here: CPU and GPU utilization timelines and nsys occupancy for this tree; the next profile should use `util_timeline.py` and the nsys window.

## Utilization timeline (util_timeline.py, same fit, warm, 8 logical cores)

Wall 73.3 s. Per 10 s bucket: mean CPU 12.5 to 43.6 percent (overall about 33), cores above 50 percent 0.6 to 3.7 of 8, GPU 10.7 to 35.1 percent (overall about 21). Seconds 1 to 15 sit in `_fe_stage_cascade_early_a_part1` (CPU 37 to 44, GPU 21 to 26), 22 to 52 in `_fit_body_run_fit_impl` (GPU peaks 35 at 38 to 47 s), the last 20 s fall to CPU 12 to 28 and GPU 16 to 20 with the serial tail stages `retain_usable_pure_forms` (2.1 s), `check_prospective_fe_pairs` (1.7 s) and `_run_fe_step_impl_part6` (1.5 s).

Neither side is near 100 percent anywhere in the fit: the GPU is idle about four fifths of the time and no bucket keeps more than four of the eight cores busy. The fit was run with `n_jobs=1`, so the CPU figure is what the stage threads and the njit parallel kernels reach by themselves. Next levers: overlap the stage threads' GPU work (more concurrent FE stages in flight), and give the serial tail stages parallel kernels; the existing note in `retain_usable_pure_forms` already records that its pool build is dominated by njit-parallel kernels, so its tail is not a Python-loop hotspot.

## Concurrent-section walls (stage_walls.py, debug log of `_fe_stage_concurrency`, one process, wall 72.7 s)

* Hybrid-orth groups together: stage walls 11.52 / 5.20 / 9.64 s, so the section costs 11.5 s (the slowest group).
* The three later cascades (early_b, mid_a, mid_b) together: 10.68 / 24.20 / 20.45 s, so the section costs 24.2 s and mid_a is the critical path; early_b finishes after 10.7 s and its slot idles for the remaining 13.5 s.
* Together the two sections set 35.7 s of the 72.7 s; early_a (11.4 s on the main thread) runs before the second section. Running early_a beside the other three (`MLFRAME_FE_STAGE_CONCURRENCY_EARLY_A=1`) was measured at 100k x 100 earlier and lost (see the comment in `_fe_stage_cascade_run.py`), so it stays off.
* Lever: shorten mid_a (cat_pair, composite_group_agg, conditional_gate, grouped_agg, grouped_quantile, integer_lattice, modular, numeric_decompose, pairwise_modular, row_argmax) or move a family from mid_a to the idle early_b slot; `STAGE_WEIGHTS = (14, 34, 18)` already says mid_a is the heaviest.

## Wall seconds per family function (family_walls.py, wrapper timing, one process, fit wall 74.5 s)

The wrapped `*_part1` entries contain the family functions below them, so the families are the rows to compare, not the sums. Largest family functions: early_b `binned_numeric_agg` 9.37 s; mid_a `grouped_quantile` 5.85, `modular` 4.61, `numeric_decompose` 4.22, `composite_group_agg` 4.01, `conditional_gate` 1.87, `row_argmax` 1.28, `layer87_grouped_agg` 1.13, `pairwise_modular` 1.10; mid_b `group_distance` 5.64, `conditional_quantile_rank` 3.06, `wavelet` 2.22, `ordinal_pattern` 1.85; early_a `extra_orthogonal_bases` 2.64, `greedy_cmi` 1.27. Everything else is under 1.2 s.

Reading against the section walls above: mid_a (24.2 s wall) is the critical path of the three-cascade section, mid_b is next (20.5 s), early_b ends at 10.7 s and its slot idles. Moving the two or three heaviest independent families of mid_a (`grouped_quantile`, `numeric_decompose`) to early_b would put the section at about mid_b's 20.5 s, roughly 3.7 s off the wall, if the families do not read each other's output; that needs a selection-parity check (merge order of the new columns changes) and was not done here. Speeding up `binned_numeric_agg` does not shorten the wall while early_b is off the critical path.

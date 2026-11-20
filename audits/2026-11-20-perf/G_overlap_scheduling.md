# Direction G: overlap and scheduling of the MRMR 100000 x 100 benchmark (HEAD 871110e1f)

Read-only audit. Sources: the 2026-11-19 findings memory, tracker rows MRMR-WAVES-BALANCED, MRMR-STAGE-THREADS, MRMR-UTILIZATION-TIMELINE, MR11-08, MR11-11, the MR11-12 / stage-thread notes of 2026-11-09 and 2026-11-17, `audits/2026-11-09/stage_thread_restructuring_options.md`, `D:/Temp/exp/stage_sampler_1120.txt` (WALL 41.1 s), and the code path (`_fit_impl_core_parts.py`, `_fe_stage_cascade_run.py`, `_fe_stage_concurrency.py`, `_fe_stage_cascade_early_a.py`, `_screen_predictors_prescreen.py`, `_honest_mdlp_null.py`, `_fit_impl_stages/_engineered_dedup.py`, `_eng_dedup_scan.py`, `_assign_support_tail.py`).

Overall verdict: the critical path is a chain of main-thread steps whose inputs are the full set of columns (screen, pair step, re-screen, support tail). Most of the overlap that is still possible is worth 1 to 3 s in total, which is inside the 3 to 5 s run-to-run noise of the benchmark, so it must be judged by isolated A/B. The 30 s goal is not reachable through scheduling alone. Several candidates that look like overlap were checked and are either real data dependencies or were already measured and rejected (listed at the end).

Stage timeline used for the estimates (stage_sampler_1120, WALL 41.1 s): early_a part1 8.0 s alone; the three concurrent cascades ~10 s (walls 9.8 / 10.5 / 10.0 s in wave 1 per MRMR-WAVES-BALANCED); main-thread chain after them: temporal agg, accuracy gate 0.6 s, cross-stage dedup ~1.1 s, discretise 1.6 s, first screen with honest FDR floor 2.6 s (honest null 1.2 s), pair step ~6 to 7 s, re-screen ~1 to 2.5 s, support tail ~2.7 to 3.3 s.

---

### F01 - Honest raw-column null computed on the idle stage slot during the cascades (MR11-08 open candidate, now with the slot that is free)

- **Expected saving**: 1.0 to 1.4 s on the 100000 x 100 benchmark. Evidence: `honest_null_columns_device` 1.2 s and `honest_fdr_floor` 0.8 s inclusive in `stage_sampler_1120.txt`; MR11-08 estimates about 1.4 s of the tail for the honest null of 184 columns x 999 permutations. Not measured as a wall change; by reading plus the sampler.
- **Location**: `src/mlframe/feature_selection/filters/_screen_predictors_prescreen.py:155` (call of `honest_fdr_floor`), `src/mlframe/feature_selection/filters/_honest_mdlp_null.py:142` (`honest_null_columns_device` call) and `:377` (`honest_fdr_floor`), `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core_parts.py:737-739` (existing `start_discretization_warmup`, the pattern to copy), `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_stage_cascade_run.py:159` (wave 2, where one stage thread finishes 1.5 to 2.3 s before the others).
- **Problem**: the honest maxT null of the screen is computed on the main thread after the cascades, although its per-column part (the permuted-target binning of a column) depends only on the column's values, the target and the fit's permutation seed. The fit already keeps a per-column cache across screens (MR11-08: the second screen computes only its 6 new columns), so the raw-column part can be filled earlier. The screen's evaluation rows and target codes are created before the first screen, so a worker started at the cascade call can build the same per-column entries. Meanwhile the wave-2 slot that finishes first sits idle while the other two stages run.
- **Proposed change**: in `_fit_impl_core_parts.py` next to `start_discretization_warmup` (line 739), start a worker that calls the honest null builder for the raw columns (the set known before the cascades, `_raw_input_cols_pre_fe`) with the same seed, rows and target codes the screen will use, and writes into the same per-column cache; `honest_fdr_floor` then reads from the cache, and columns not in the cache are computed as now. Only the cache fill moves; the gain floor, the permutation draws and the selection are unchanged. Keep the worker on the device slot of the finished stage (`stage_device_slot`) or on the main thread after `run_stages_concurrently` returns if the slot is not free.
- **Risk**: the permutation RNG must be drawn in the same order as the serial path, or the floor changes; the rows used by the screen must be known before the screen (check `honest_eval_rows`). A second worker can contend for the GIL with the three stages (the MR11-11 lesson: an extra pool on the same cores gave no net gain), so the saving is only as big as the idle slot. Check the honest null count (999 permutations at 184 columns) on 4 GB VRAM against the stage slots.
- **Disposition**: OPEN

### F02 - Per-column inputs of the cross-stage dedup computed inside each stage thread at its end

- **Expected saving**: 0.5 to 0.8 s. Evidence: the cross-stage dedup takes 1.1 s on the main thread (MR11-11 timeline "dedup 1.1"); its pieces are `_engineered_mi_scores` (plug-in MI of every appended engineered column about y) and `_precompute_full_ranks` (full-column ranks in scan order, sampler 0.4 s). By reading: neither depends on another stage's columns.
- **Location**: `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_stages/_engineered_dedup.py:107` (`_engineered_mi_scores`), `:128` (`_dedup_engineered_across_stages`), `:145` (call); `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_eng_dedup_scan.py:112` (`_precompute_full_ranks`); call site `_fit_impl_core_parts.py:789`.
- **Problem**: the MI and rank of a column are per-column work, but they run on the main thread after the whole wave, so the wave's stage threads are idle during them. Only the pairwise |rank correlation| comparison truly needs every column.
- **Proposed change**: when a stage returns, compute its appended columns' MI and full ranks on the same thread (stage-local memo keyed by column name, the same bin edges that `_engineered_mi_scores` would use), and let `_dedup_engineered_across_stages` read the memo, computing only missing names. The pairwise loop and the drop order (stage order, first appended kept) stay as they are.
- **Risk**: the MI must use the same binning as the main-thread call (the stage-local binning must not depend on what other stages produced); the memo must be keyed by name AND content (a stage-merged column of the same name must not hit). MR11-11 found that moving an edge search into a background pool slowed the dedup scan on the same cores, so the net gain is uncertain.
- **Disposition**: OPEN

### F03 - Engineered-accuracy gate run per stage instead of once after the wave

- **Expected saving**: 0.3 to 0.6 s. Evidence: `_gate_engineered_accuracy` 0.6 s inclusive in `stage_sampler_1120.txt`. By reading: the gate scores each engineered column against y through its source columns (`_gate_base_columns`, `_gate_fourier_columns`), so the verdict of a column does not depend on the other stages' columns; only the drop application (`_apply_gate_drops`) touches the shared frame.
- **Location**: `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_stages/_engineered_gates.py:270` (`_gate_engineered_accuracy`), `:247` (`_apply_gate_drops`), `:197` (`_gate_fourier_columns`); call at `_fit_impl_core_parts.py:765`.
- **Problem**: the gate runs on the main thread between the wave and the cross-stage dedup, so the stage threads are idle for its 0.6 s; the verdict is computed for every column at once even though each stage produces its own.
- **Proposed change**: run the per-family scoring at the end of each stage (inside `run_stages_concurrently`'s `_run`, after the stage's own columns are final), store the per-column verdicts on the stage copy, and merge them with the stage state (the existing `merge_stage_state` already carries new dict entries); `_apply_gate_drops` stays on the main thread in stage order.
- **Risk**: the gate's `surviving_by_src` input and its CV seed must match the serial order; `_gate_fourier_columns` uses the `groups` argument, which the stage copies must receive. Modest code change for under a second.
- **Disposition**: OPEN

---

## Dropped after the read (checked, not findings)

- **Run early_a together with the three cascades (four stages at once)**: the real dependency is only the dominance gate (`dominant_candidate` after early_a, `_fe_stage_cascade_run.py:113-150`), and the stages read `_X_step_input`, not early_a's output. But it is already measured and rejected: the MR11-01 block of `audits/2026-11-09/_TRACKER.md` (line ~820) records 60.0 s against 51.7 to 55.6 s for the opt-in four-stage section, with early_a's own stage wall stretched to 37.9 s. Its code is opt-in (`MLFRAME_FE_STAGE_CONCURRENCY_EARLY_A`). Not re-proposed.
- **Early_a part2 overlapping part1**: part1 receives `X_acc` from part2 and the CMI-greedy seed excludes names from both (`_fe_stage_cascade_early_a.py:205-223`). A real dependency.
- **Second discretisation warm-up after the cascades (overlap with dedup)**: measured in MR11-11, discretise 1.6 to 1.1 s but dedup 1.1 to 2.1 s on the same edge pool; wall 48.4 / 49.2 s, reverted.
- **Friend-graph analysis as a background diagnostic**: it is not diagnostic; `_friend_graph_and_redundancy_passes` mutates `selected_vars`, `cols` and `data` (`_fit_impl_stages/_post_selection.py:289-330`). Dependent.
- **Support tail passes in parallel** (`_assign_support_tail.py:44-160`): each pass mutates `selected_vars` and `self.n_features_` that the next reads. Only the read-only parts (the float64 column cache for the linear-dependence guard) could be prefetched, and that is under 0.2 s.
- **Per-stage `_synchronize_device` before merge**: needed; the main thread reads the stage outputs right after, so the sync only moves a wait.
- **Launch-cost / second CUDA stream per stage**: the measured floor is host-side launch time (25 to 82 us per call, MRMR-LAUNCH-COST-FLOOR), and streams do not remove the GIL-bound launch path.
- **Pair step GPU/CPU double buffering (F-type item 4)**: not verified in this pass; the pair step is a serial dependency of the re-screen, and I did not read its loop bodies, so no claim is made. Worth a separate read if the owner wants it.

## Summary

Surviving findings: F01 (1.0 to 1.4 s), F02 (0.5 to 0.8 s), F03 (0.3 to 0.6 s). Total about 2 to 2.8 s, roughly 5 to 7 percent of the 41 to 44 s wall, and within the run-to-run spread (3 to 5 s); measure as isolated alternating A/B with the stage sampler. Reaching 30 s needs work removed from the stages themselves (already the subject of the batched-family-pass item), not overlap.

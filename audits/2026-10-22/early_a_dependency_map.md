# early_a dependency map (2026-11-06)

Question: which blocks of `_fe_stage_cascade_early_a` need the output of another block, and which could run beside it?
Method: reading of `_fe_stage_cascade_early_a.py` (part2, part1, the three stage functions) and `_hybrid_orth_family_variants/`, plus a run-time check (`D:/Temp/early_a_deps.py`: at the entry of each block, how many of the names it excludes occur among the columns of the frame it reads; 20000 x 30, every family, one FE step).

## The blocks, in the order they run, 100000 x 100 timings from the per-family walls

| # | Block | Reads (data) | Reads (estimator state) | Writes | Wall |
|---|---|---|---|---|---|
| B1 | Layer 23 hybrid orthogonal polynomial (+ pair cross), part2 | `X` (the step input, never rebound), `y` | `fe_hybrid_orth_*` settings, `_fe_shared_subsample_idx` | `X_acc` (append), `hybrid_orth_features_` (assigned), `_hybrid_orth_pre_recipes` (overwrite) | with B2: 3.1 s |
| B2 | extra bases (Fourier, ...) `_stage_extra_orthogonal_bases` | `X` | `hybrid_orth_features_` (exclusion list) | `X_acc`, `hybrid_orth_features_` (append), `_adaptive_fourier_features_` (append), pre-recipes | in B1 + B2 |
| B3 | hinge change-point (part1, first block) | `X`, `y` | `hybrid_orth_features_` (exclusion list) | deferred value / recipe dicts only (no `X_acc`) | about 1 s |
| B4 | `_hybrid_orth_family_variants` (three groups, already concurrent) | `X`, which each variant rebinds to the hybrid-augmented frame inside the chain | `hybrid_orth_features_` (re-read before each variant: a real exclusion), `_seeded_triplets_names_` (set before early_a) | `X_acc`, `hybrid_orth_features_` (append), pre-recipes | 11.5 s (groups 11.5 / 5.2 / 9.6) |
| B5 | Layer 26 MI-greedy `_stage_mi_greedy` | `X` | `hybrid_orth_features_` (exclusion list) | `X_acc`, `mi_greedy_features_` / `_candidates_` (reset before, then assigned), pre-recipes, rejection records | 0.2 s |
| B6 | Layer 60 CMI-greedy `_stage_greedy_cmi` | `X` | `hybrid_orth_features_` and `mi_greedy_features_` (exclusion lists) | `X_acc`, `mi_greedy_features_` / `_candidates_` (append with dedupe), pre-recipes (first writer wins) | 1.3 s |

## Findings

1. No block reads another block's output COLUMNS. Every block reads `X`, the step input, which `_run_fe_stage_cascades` documents as never rebound; new columns go to `X_acc`, which no block reads. The only chain of columns is inside B4, where each variant of a group sees the frame augmented by the earlier variants, and that chain already runs as three concurrent groups with private estimator copies and a fold-in (`_hybrid_orth_family_variants/_concurrent.py`).
2. The "already appended" exclusions of B2, B3, B5 and B6 are vacuous by construction: they exclude names that are NOT columns of the frame the block reads (an engineered column is only appended to `X_acc`; a name already in `X` is never re-appended). Run-time check, one real fit: at the entry of B2 the exclusion set held 1 name, 0 of them among the columns of `X`; at the entry of B3 (part1) 2 names, 0 among the columns; at B5 and B6 51 names, 0 among the columns. The exclusion matters only inside B4.
3. What does couple the blocks is shared estimator state, not data: the ORDER of `hybrid_orth_features_` and `mi_greedy_features_` (B1 assigns, B2 and B4 append; B5 assigns, B6 appends with a dedupe against it), the pre-recipe dictionaries (B1 overwrites a name, B6 keeps the first), the merge order of the new columns into `X_acc` (duplicates are resolved "keep the first"), `_fe_rejection_records_`, the wall-clock gate `_fe_budget_ok()` (a time-based decision: it sees a different elapsed time when blocks overlap) and `_fe_steps_executed_`.
4. Independent units that would give the same outputs if folded back in the order B1 B2 B3 B4 B5 B6: A = {B1, B2} (3.1 s), B = {B3} (about 1 s), C = {B4} (11.5 s, already three-way concurrent), D = {B5, B6} (1.5 s).

## What overlapping them could give, and what it costs

* Upper bound: A, B and D (about 6 s) beside C (11.5 s), so early_a goes from about 17.5 s to about 12 s, about 8 percent of the 70 to 75 s fit. Not measured.
* Memory: C already puts three device workloads side by side; A and D would add two more. The four-stage run of the cascades hit 336 out-of-memory fallbacks of the tiled transposes (193 s); the guard `_stages_fit_device` now counts the pool cap and refuses plans that exceed it. Measured device pool at 100000 x 100: in use p50 67 MB, p90 317 MB, maximum 815 MB, but the pool held up to 1587 MB in total (cached blocks) of the 2.58 GB cap. The 95th-percentile peaks are in `_pairs_analytic_gpu._fused_observed_mi_and_bins` (the resident int8 code matrix of the pair screen), `_family_maxt_gate_device.__init__` and `null_max` (bit sets).
* Merge risk to check before building it: `merge_stage_state` folds list attributes of the stage copies back; B1/B2 (unit A) and B4 (unit C) both append to `hybrid_orth_features_`, so the fold must append the later unit's names after the earlier one's and not rebuild the list from the last stage ("the last stage wins" branch of `merge_stage_state`). Not verified here.
* Selection check: every variant needs the old-versus-new comparison of selected features and recipes at 100000 x 100 (about 70 s a fit).

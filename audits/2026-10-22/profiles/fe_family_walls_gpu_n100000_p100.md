# FE family walls, GPU-resident default, n=100000, p=100 (one isolated fit)

Measured 2026-11-12 on the development host (GTX 1050 Ti, Pascal sm_61), nothing else running, `MRMR(verbose=0, random_state=0)` with the default FE configuration, float32 synthetic frame of 100000 x 100 (threshold + product + noise target).
Script: `fe_family_wall_` of the fitted estimator, printed longest first. Fit wall (s): 120.3.

`stage:` rows are nested (a stage contains the families below it) so they do not add up; the `gate:` and named-family rows are the leaf regions.

| region | wall (s) | calls |
|---|---|---|
| stage:decide_family_call | 56.51 | 36 |
| stage:fe_step | 50.28 | 1 |
| stage:fe_step_part6 | 27.00 | 1 |
| stage:materialise_and_finalise | 27.00 | 1 |
| stage:cascade_early_a | 23.66 | 1 |
| stage:early_a_part1_variants_and_greedy | 16.77 | 1 |
| stage:hybrid_variants | 15.87 | 1 |
| stage:fe_step_part5 | 14.88 | 1 |
| stage:check_prospective_pairs | 14.87 | 1 |
| gate:fe_hinge_enable | 14.73 | 1 |
| stage:screen_predictors | 13.88 | 2 |
| stage:score_one_pair | 13.55 | 43 |
| stage:cascade_mid_b | 12.84 | 1 |
| stage:cascade_mid_a | 10.83 | 1 |
| stage:fe_step_part7 | 8.39 | 1 |
| stage:cascade_early_b | 8.21 | 1 |
| stage:fe_step_part3 | 8.06 | 1 |
| gate:fe_binned_numeric_agg_enable | 7.95 | 1 |
| binned_numeric_agg | 7.88 | 1 |
| gate:fe_hybrid_orth_adaptive_degree_enable | 7.45 | 1 |
| stage:score_prospective_pairs | 6.20 | 1 |
| triplet | 5.78 | 2 |
| gate:fe_hybrid_orth_triplet_enable | 5.62 | 1 |
| stage:early_a_part2_univariate_and_extra | 5.52 | 1 |
| stage:assign_support_tail | 5.24 | 1 |
| stage:decide_replay | 4.96 | 16 |
| gate:fe_hybrid_orth_bootstrap_enable | 4.74 | 1 |
| orth_mi_bootstrap | 4.72 | 1 |
| stage:pair_family_gate | 4.44 | 35 |
| stage:fdr_gain_floor | 3.75 | 2 |
| gate:fe_hybrid_orth_auto_scorer_enable | 3.63 | 1 |
| gate:fe_modular_enable | 3.63 | 1 |
| orth_mi_cmim | 3.63 | 2 |
| orth_mi_auto_scorer | 3.60 | 1 |
| gate:fe_univariate_fourier_enable | 3.58 | 1 |
| extra_basis | 3.43 | 1 |
| gate:fe_conditional_gate_enable | 2.97 | 1 |
| conditional_gate | 2.97 | 1 |
| stage:support_fe_max_steps_pass | 2.87 | 1 |
| stage:categorize_dataset | 2.77 | 2 |

## Reading

- `gate:fe_hinge_enable` is the largest single named wrapper wall (14.7 s, 12% of the fit), but the wrapper wall includes waiting while stages run concurrently; the isolated call is 2.11 s (below). `_hinge_basis_fe` carries a measured exemption in `_fe_gpu_contract.json` made at n=20000 (0.11 s of 141 s) with the stated revocation condition 're-profile and move it back to pending if the region crosses 0.5 s at another n or column count'. At n=100000, p=100 the isolated call is 2.11 s (1.56 s of it the host numpy QR in `_hinge_slope_change_plausible`, 100 columns), which still exceeds 0.5 s, so the condition is met, by a factor of about 4 and not about 29.
- `orth_mi_bootstrap` (4.7 s) and `orth_mi_auto_scorer` / `orth_mi_cmim` (3.6 s) belong to `_orthogonal_bootstrap_mi_fe` and `_orthogonal_scorer_auto_fe`, which have no GPU reference; a static heuristic had filed them as 'no compute'.

## Isolated hinge call (cProfile, after a numba warm-up)

`hybrid_hinge_fe_with_recipes(X, y)` alone on the same 100000 x 100 frame: 2.11 s, 4 engineered columns, 4 recipes. `_detect_hinge_breakpoints_for_columns` 1.86 s (100 columns, 18 ms each), of which `_hinge_slope_change_plausible` 1.56 s (its numpy `qr` 0.46 s, `quantile`/`partition` 0.24 s). The remaining time is dedup (0.13 s) and std/var. The wrapper walls above overstate a family's own cost under stage concurrency; rank by isolated calls.

## Isolated family calls, 100 columns (wall seconds, after a small-frame warm-up)

The entry points called directly on the same synthetic frame, once with the default switch (outside a fit the `auto` strict gate has no fit shape, so this is the host path) and once with `MLFRAME_FE_GPU_STRICT=1` (device paths forced where they exist).

| family | n=30000 host | n=30000 strict | n=100000 host | n=100000 strict | device path |
|---|---|---|---|---|---|
| `binned_numeric_agg` | 3.22 | 1.99 | 8.13 | 3.94 | partial (about 2x) |
| `orth_mi_auto_scorer` | 1.98 | 1.99 | 8.06 | 8.20 | none (no change) |
| `orth_mi_cmim` | 1.08 | 1.09 | 4.37 | 3.44 | small |
| `orth_mi_bootstrap` | 0.93 | 0.99 | 2.74 | 3.02 | none (no change) |
| hinge | | | 2.11 | | none |

Inside the fit these stages run on the 30000-row decision subsample, so each costs 1 to 2 s, about 1 to 2% of the 120 s fit. The large costs of the fit are structural (the `stage:` rows: `decide_family_call` 56.5 s over 36 calls, `materialise_and_finalise` 27 s, `check_prospective_pairs` 14.9 s, `screen_predictors` 13.9 s), not single modules of the contract list.

`binned_numeric_agg` at n=100000 (cProfile, host path): 7.9 s, of which `_quantile_bin` 3.7 s (504 columns, `np.quantile` 1.8 s and `searchsorted` 1.5 s), `_keep_linearly_usable` 2.5 s; it accepted 0 columns on this frame, so the work produced nothing here.

## `orth_mi_auto_scorer`: what one bootstrap resample costs (100 columns, seconds, one call each)

The family scores every column with seven batched scorers on each of about 200 bootstrap resamples (pool of threads; cProfile sees only the main thread waiting). One resample, 100 columns:

| scorer | n=30000 | n=100000 | device path |
|---|---|---|---|
| ksg (discrete target) | 0.127 | 0.550 | `_ksg_cd_gpu` through `_mi_cd_columns_auto`; the rest is host preparation (float64 copy, scaling, noise) |
| tail_dep | 0.033 | 0.355 | none |
| xi | 0.038 | 0.133 | device block when available |
| copula | 0.041 | 0.092 | device block when available |
| plug_in | 0.011 | 0.058 | partial |
| hsic | 0.012 | 0.012 | n/a (sampled) |
| dcor | 0.011 | 0.010 | n/a (sampled) |

`tail_dependence_columns` (`_orthogonal_tail_dependence_fe`) is the one large scorer with no device path at all: about 30% of the per-resample cost at n=100000 and 12% at n=30000.

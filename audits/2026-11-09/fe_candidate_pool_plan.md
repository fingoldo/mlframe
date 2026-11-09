# FE source-column pools: which families run on which columns (plan, 2026-11-09)

Status: plan only. Nothing is implemented; every number below marked "estimate" is unmeasured. Reviewed by three independent read-only reviews (statistics, compute budget, linear-model value).

## Current state (from the code)

- Pair FE (unary/binary pair search, polynomial pair search) runs on `selected_vars` (screening-relevant columns) plus zero-marginal columns flagged by the synergy bootstrap (`_mrmr_fe_step_helpers.apply_synergy_bootstrap`; first FE step only, behind a column-count cap, a minimum-rows guard and an `n*p^2` cost gate).
- The cascade families (modular, hinge, wavelet, grouped_*, ordinal, conditional_*, MI-greedy, row_stat, ...) take every raw column of X except hybrid-appended ones (`_fe_stage_cascade_early_a.py:53-55`, `mid_a.py:270,760,826,896`, `mid_b.py:672-677`). `factors_names_to_use` is a user whitelist, not relevance. The ordinal / conditional families are already capped at 5-6 columns.

## Which families may use a smaller pool

| Family | Pool | Why |
|---|---|---|
| XOR duplets / triplets / quadruplets, synergy pairs, hybrid triplet / quadruplet | all | zero marginal MI |
| modular, Fourier, sin/cos | all | fast oscillation is invisible to binned MI |
| ratio / normaliser operands | all | the partner or normaliser can have zero marginal |
| grouped_* (group keys and value columns) | all | an irrelevant key can be the whole signal |
| row_stat, mi_greedy | all | aggregate / screening steps themselves |
| ordinal_pattern, conditional_* | unchanged (already capped) | pick the capped subset by a non-binned score |
| pure unary shape families (rankgauss, quantile, power, hinge, wavelet, spline) | candidates for a smaller pool | the only ones where the pool is plausibly safe |

Default for every family stays `all` until the benchmark shows no out-of-sample loss. Per-family option: enum `all` / `relevant` / `relevant+k`.

## Pool rule for the unary shape families

Not "top 2-3x relevant by marginal MI". A column enters the pool if it passes either test:
1. the permutation p-value of a cheap non-binned dependence score (distance correlation / HSIC / shallow-tree gain), or the quick linear CV-loss fit of the family on that column (`_linear_cv_loss.py`, `_fe_usability_signal.py`), which keeps symmetric U / window shapes that a marginal cut drops;
2. its incremental gain over the relevant set (conditional MI or residual score), which keeps conditional effects.

Cutoff: permutation FDR q <= 0.3, capped at `max(2.5 * k, 15)`; always keep columns that appear in an accepted engineered feature. Relevance comes from the full-n screening result (not the 30000-row subsample), and the pool is frozen before any family runs (no order dependence). The pool must not be chosen on the data that later scores its survivors without a split or a permutation-null calibration (selection bias).

## Architecture (one place, minimal code)

`_mrmr_fit_impl/_candidate_pool.py`: frozen `CandidatePool(relevant, near_miss, all_cols, p_values, fingerprint)` built once after screening, stored on the estimator; accessor `columns(kind)` with kinds `all` / `marginal` / `selected`; registry `FAMILY_POOL_KIND`. Applied at the single seam `fe_decide_on_subsample` / `fe_subsample_kwargs`, not inside each family. Throttled warning when columns are dropped. To check first: that screening finishes before `cascade_early_a` (`_fit_impl_core.py`).

## Expected payoff (estimate, from the 100000 x 100 timings)

The shape families are a few seconds each; restricting them to ~25 columns saves about 6-10 s of the ~80 s fit. The larger levers are elsewhere (hybrid_variants ~20 s, the decide_on_subsample calls, the serial screening confirmation), so this is a later, smaller step.

## Benchmark (decides each family)

- p = 100, n in {20k, 100k}, 10-20 relevant columns, 70-80 noise columns (some correlated, some heavy-tailed); 5 seeds.
- Planted cases: weak marginal shape (hinge, U, window; marginal p swept 0.05-0.5), hidden periodic (frequency swept), XOR of 2 / 3 / 4 columns, ratio with an irrelevant normaliser, grouped effect keyed by an irrelevant categorical (cardinality swept), conditional hinge (active only when another column is on), pure-noise null.
- Existing assets: `tests/feature_selection/_suite_fe_helpers.py:85-141` (ratio, bilinear, poly, trig-product, cluster-linear; no noise columns), `tests/feature_selection/_synth/biz_val_synth.py:35-226` (`make_3way_xor`, signal-plus-noise, ...), `src/mlframe/feature_selection/_benchmarks/bench_pid_xor_synergy.py:35`. Missing: hidden periodic, grouped-by-irrelevant-key, weak-marginal shape generators. Test template: `tests/feature_selection/fe/test_composite_group_key_sets_by_relevance.py`.
- Arms: all columns; relevant only; relevant + top-3x by MI; the proposed rule.
- Metrics per family and arm: recall of the planted engineered feature via `fe_provenance_`; Ridge / logistic out-of-sample gain versus raw; false-positive features on the null; wall time of the family and its share of the fit.
- Decision: a family gets a filter only if its out-of-sample gain is not statistically lower than the all-columns arm at every signal-to-noise ratio and n, and it is more than about 10 % of the fit's wall time.

## Order of work

1. hybrid_variants by sub-gate and the repeated work behind the decide_on_subsample calls.
2. Generators for the missing planted signals; the recall / out-of-sample benchmark with the pool applied only to the linear-scaling unary families.
3. Only then the pool for anything else.

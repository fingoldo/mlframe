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

## Baseline, all columns (2026-11-09)

`bench_pool_planted_recall.py 20000 100 1`, all families, one process, the generators of `tests/feature_selection/_synth/pool_planted.py`. Recall is crude (a selected column whose name mentions every planted source), so it says the fit built something from the planted columns, not that it found the planted transform.

| generator | wall | selected | columns that mention a planted source |
|---|---|---|---|
| weak_marginal_shape | 58.1 s | 2 | sig_x, sig_x__pade2 |
| hidden_periodic | 43.2 s | 2 | sig_x, sig_x__digit_2 (no sine column: the periodic structure is picked up through the raw column and a digit feature) |
| grouped_by_irrelevant_key | 46.1 s | 5 | sig_x, sig_x*sig_key__He1_He1, sig_x__round_1p0, esc_poly_hermite_mul(sig_x,sig_key), sub(log(...),prewarp(...)) |
| ratio_irrelevant_normaliser | 43.9 s | 2 | sig_a, mul(cbrt(sig_a),invcbrt(sig_b)) |
| conditional_hinge | 48.5 s | 2 | div(cbrt(sig_x),sign(sig_z)), sig_x__pade2 |

Recall 1/1 on every generator at one seed. Next: more seeds, the metric through `fe_provenance_` and a linear out-of-sample gain, then the filtered arms.

### 3 seeds, held-out linear AUC (all columns, 14000 train / 6000 test rows, p = 100)

| generator | recall (name mention) | linear AUC, selected set | linear AUC, raw columns | fit wall per seed |
|---|---|---|---|---|
| weak_marginal_shape | 3/3 | 0.752 / 0.746 / 0.741 | 0.508 / 0.498 / 0.490 | 108 / 98 / 95 s |
| hidden_periodic | 3/3 | 0.704 / 0.704 / 0.697 | 0.500 / 0.503 / 0.499 | 69 / 72 / 80 s |
| grouped_by_irrelevant_key | 3/3 | 0.738 / 0.714 / 0.673 | 0.593 / 0.654 / 0.642 | 88 / 117 / 85 s |
| ratio_irrelevant_normaliser | 3/3 | 0.904 / 0.912 / 0.905 | 0.887 / 0.891 / 0.888 | 76 / 75 / 73 s |
| conditional_hinge | 3/3 | 0.862 / 0.864 / 0.857 | 0.493 / 0.501 / 0.501 | 71 / 70 / 69 s |

Reading: the all-columns arm already gives a large linear gain everywhere the raw columns carry nothing linearly (weak shape +0.25, periodic +0.20, conditional hinge +0.36). Open points for the filtered arms: (1) hidden_periodic reaches only 0.70 and the selected set holds `sig_x__digit_2` but no sine column, so the periodic family is not recovering a 20-period signal at 14000 rows (a better ceiling is worth measuring: the planted score alone gives a much higher AUC); (2) grouped_by_irrelevant_key varies most between seeds (0.67-0.74); (3) fit walls are 69-117 s here against 43-58 s in the single-seed run on the same generators, the machine's free memory differs between the runs, so walls are only comparable inside one run.

## Correction: the cascades run BEFORE screening (found 2026-11-09 while wiring the pool)

`_fit_impl_core_parts.py:737` calls `_run_fe_stage_cascades` on the raw frame right after the fit-cache check and the input preparation; the screening (`screen_predictors`, `selected_vars`, `cached_MIs`) runs afterwards on the frame the cascades extended, and `self.cached_MIs` is only assigned near the end of the fit. So no relevance set exists when hinge, wavelet and the other cascade families pick their columns; "run FE only on the relevant columns" is not available for them as the code stands. The plan's open check ("screening finishes before cascade_early_a") is answered: it does not.

Consequence for the design: a pool for the cascade families needs its own cheap pre-screen of the raw columns before the cascades: quantile codes of every raw column on the shared 30000-row subsample (`quantile_codes_block_auto`), the joint count with the target codes, the Miller-Madow-corrected MI and a permutation p-value from a few shuffles of the target codes (all integer counting, tens of milliseconds for 100 columns). The pool is then the columns with p below a loose bound plus the top few by MI. Only `selected_vars` from the later screening exists for the pair FE, which already uses it.

## First filtered arm: `fe_shape_pool="relevant"` for hinge and wavelet (2026-11-09)

Implemented as an opt-in (`_mrmr_fit_impl/_candidate_pool.py`, parameter `fe_shape_pool`, default `"all"`); the pre-screen is the corrected-plan design (quantile codes on the shared subsample, G-test p below 0.2, top 5 by MI, non-finite columns unjudged). Benchmark: `bench_pool_planted_recall.py 20000 100 3`, all five generators, 3 seeds, one process per arm (arm `all` is the 3-seed table above plus the hidden-periodic and four-generator reruns after the Fourier grid fix).

| generator | linear AUC, arm `all` | linear AUC, arm `relevant` | fit wall `all` / `relevant` |
|---|---|---|---|
| weak_marginal_shape | 0.752 / 0.746 / 0.741 | 0.752 / 0.746 / 0.741 | 96 / 91 / 80 s vs 110 / 85 / 80 s |
| hidden_periodic | 0.969 / 0.969 / 0.970 | 0.969 / 0.969 / 0.970 | 82 / 65 / 66 s vs 58 / 58 / 58 s |
| grouped_by_irrelevant_key | 0.738 / 0.714 / 0.673 | 0.738 / 0.714 / 0.673 | 70 / 75 / 73 s vs 73 / 78 / 74 s |
| ratio_irrelevant_normaliser | 0.904 / 0.912 / 0.905 | 0.904 / 0.912 / 0.905 | 62 / 66 / 62 s vs 63 / 65 / 61 s |
| conditional_hinge | 0.862 / 0.864 / 0.857 | 0.862 / 0.864 / 0.857 | 61 / 59 / 62 s vs 59 / 58 / 60 s |

Reading: no accuracy loss on any of the 15 fits (identical held-out AUC to three decimals, recall 3/3 everywhere), and no measurable wall gain: hinge and wavelet are 1-3 s of a 60-110 s fit at p = 100, below the run-to-run noise. By the plan's own decision rule (a family gets a filter only when it is more than about 10 % of the fit's wall time) the default stays `"all"`. The option stays for wider frames, where the cost of these families grows with the column count (wavelet is capped at `fe_wavelet_max_cols=100`); re-run this table at p = 500-1000 before changing any default. Not tested here: the cases the statistics review warned about for a marginal cut (a hinge that matters only after another engineered feature exists), which the generators do not plant.

## The pool on the 12 univariate scorer variants, p = 300 (2026-11-09)

`fe_shape_pool="relevant"` now also narrows the columns of the twelve single-column basis-selection variants of `hybrid_variants` (three_gate, bootstrap, ksg, copula, dcor, hsic, jmim, tc, cmim, auto, ensemble, meta; `_hybrid_orth_family_variants/_group2-4.py`). The interaction variants (triplet, quadruplet, adaptive arity / degree, routing, diff basis, cluster basis) keep all columns.

Where the cost grows with the column count (`scale_stages_p.py`, 30000 rows, sequential mode, `F._make` data): fit wall 56.4 s at p = 100, 126.3 s at p = 300; `hybrid_variants` 16.2 -> 46.2 s (x2.9), `orth_mi_cmim` x4.2, `orth_mi_meta` x4.0, `orth_mi_jmim` x5.2, `orth_mi_three_gate` x3.0, `score_prospective_pairs` x5.5 (pair FE, not touched by the pool). With the pool on the same data: wall 126.3 -> 100.3 s (-21 %), `hybrid_variants` 46.2 -> 23.9 s, `orth_mi_cmim` 8.0 -> 0.8 s.

Accuracy at p = 300 (`bench_pool_planted_recall.py 20000 300 2`, five generators, 2 seeds, one process per arm): the held-out linear AUC of the `relevant` arm equals the `all` arm on all 10 fits (weak shape 0.752 / 0.746, hidden periodic 0.969 / 0.969, grouped 0.742 / 0.683, ratio 0.904 / 0.912, conditional hinge 0.862 / 0.864), recall 2/2 everywhere. Fit walls on these generators: `all` 196.5 / 190.5, 133.7 / 135.3, 169.1 / 159.7, 129.3 / 140.1, 134.7 / 141.1 s; `relevant` 192.0 / 195.0, 130.3 / 130.5, 161.5 / 154.4, 122.7 / 131.4, 126.2 / 130.6 s: 2-8 s (2-5 %), much less than the 26 s of the `F._make` data. The saving depends on how many columns pass the pre-screen: `F._make` has few informative columns, the planted-signal frames have 300 noise columns of which p_loose = 0.2 keeps about a fifth (about 65 columns), and the pair FE and screening, which the pool does not touch, dominate there.

Status: opt-in, default `"all"` (no accuracy loss measured in 25 fits at p = 100 and 300, a wall gain of 2-21 % at p = 300). Open: a tighter pre-screen (permutation FDR instead of p < 0.2) would shrink the pool further on noisy frames; the pair FE (`score_prospective_pairs`, x5.5 from p = 100 to 300) is the next column-count cost and is not a shape family.

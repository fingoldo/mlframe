# Cross-cutting audit: stale frozen reference / duplicated logic drift

Date: 2026-09-05. Scope: `src/mlframe` (584k LOC) + `tests/`. Read-only.

Method: enumerated every frozen-reference helper in `tests/` (98 files matching `_old_*` / `_reference_*` / `_naive_*` / `_ref_*` / `_baseline_*` / `_slow_*` / `_expected_*`) and every `_old_*`/`_reference_*` in `src/**/_benchmarks/`; read each against the CURRENT production twin and diffed line by line. Separately ran an AST body-hash duplicate detector over `src/mlframe` to find production modules that inline a copy of another production module's logic. Every finding below was verified by reading both sides, and numerically where a number was obtainable. No test suite was run.

Note: both seed instances given in the brief are already fixed in this tree (`compare.py:162` carries `(n_tail+1)/(n_boot+1)`; `importance_shootout.py:40` now imports `compute_auc_mean`). Everything below is new.

---

### SRD-01 [P1] residual-band transformer cluster: three siblings never got the OOF leakage fix

**File:** `src/mlframe/feature_engineering/transformer/hard_row_attention.py:43`, `src/mlframe/feature_engineering/transformer/multi_temp_residual_band.py:38`, `src/mlframe/feature_engineering/transformer/signed_residual_band.py:42` (twin sites that DID get the fix: `bidir_residual_band.py:41`, `class_balanced_hard_row.py:49`, `multi_temp_cbhr.py:37`, `baseline_surprise.py:43`, `fisher_weighted_residual.py:39`)

**Summary:** `_fit_baseline_predict(Xt, y_t, task, seed, n_estimators=50, max_depth=3)` is copy-pasted into eight transformer modules with an identical name and signature. Four of them were later fixed to return inner-`KFold(3)` OUT-OF-FOLD predictions; the fix comment in `bidir_residual_band.py:42-48` states the rationale: "An in-sample prediction is close to `y_t` almost by construction (the model was just fit on these exact rows), which systematically understates the true baseline residual and distorts which rows look easy/hard for band assignment." Three copies still fit and predict on the same `Xt` and are documented as returning "in-sample predictions". `class_balanced_hard_row.py:54`, `disagreement_band.py:48`, `multi_baseline_hard_row.py:50` and `fisher_weighted_residual.py:36` each carry a comment calling this "the leakage class already fixed for the sibling `bidir_residual_band.py::_fit_baseline_predict`" -- so the fix was propagated by hand and three siblings were missed. The asymmetry is sharpest between the two multi-temp modules added as a pair: `multi_temp_cbhr.py` is OOF, `multi_temp_residual_band.py` is not.

**Failure scenario:** `multi_temp_residual_band.py:104-115` bands train rows on `residuals = |y_t - preds_tr|` quantiles; `signed_residual_band.py:108-117` on signed-residual quantiles; `hard_row_attention.py:109-120` picks the top-K hardest train rows by `|residual|` and every query row then attends to those rows (emitting `hrattn_w_h*`, `hrattn_y_agg`, `hrattn_best_hard`). Measured on n=400, p=6, `y = x0 + 0.5*x1 + 0.3*noise`, LGBM(50, depth 3), against the same inner KFold(3) the fixed siblings use: mean `|residual|` is **0.2092** in-sample vs **0.2939** OOF (40% of the residual mass is memorisation), and **209 of 400 rows land in a different quintile band**. Every downstream feature column these three modules emit is computed from that band assignment.

**Evidence:** AST comparison of the eight same-named `_fit_baseline_predict` bodies (`KFold` present/absent, printed per file); read `bidir_residual_band.py:41-48` (fix + rationale) against `hard_row_attention.py:43-63`, `multi_temp_residual_band.py:38-58`, `signed_residual_band.py:42-63`; ran the in-sample-vs-OOF band comparison above with lightgbm + `sklearn.model_selection.KFold`.

**Suggested fix:** replace the three in-sample bodies with the fixed sibling's inner-KFold(3) form (seed offsets `seed+11` for the splitter and `seed+7+fold` per fold, matching `bidir_residual_band.py`). Better, since eight hand-maintained copies is what produced this drift: hoist one `_fit_baseline_predict` into a shared module in the `transformer/` package and have all eight import it.

*Alternative reading:* a case can be made for P0 -- these are user-facing feature columns, not an internal decision. Rated P1 because the leak biases which rows the bands/thresholds select rather than writing the target itself into a query-row feature.

**RESOLVED.** An AST sweep of every ``_fit_baseline_predict`` in the package found the leak in **five**
modules, not three: the finding missed ``prediction_band_attention.py`` (same signature, bands on the
predictions themselves) and ``y_quintile_baseline_knn.py`` (different signature -- it took a separate
``Xall`` to predict on, but its one call site passed the train matrix, so the parameter was vestigial and
the fit was in-sample all the same). The five copies that shared a signature now call one
``_baseline_oof.fit_baseline_predict_oof``, which is the audit's own preferred fix and removes the drift
class rather than the instance; ``y_quintile_baseline_knn`` calls it too, with its own deeper baseline
(100 iterations, depth 5). Copies with a genuinely different shape (``baseline_surprise`` also predicts on
a held-out Xq, ``class_balanced_hard_row`` refits class-balanced) keep their own bodies -- they are
already out-of-fold, and folding them in would change what they compute. Reproduced the leak before
fixing: mean |residual| 0.2092 in-sample against 0.2968 out-of-fold, 244 of 400 rows changing quintile
band. Guard: tests/feature_engineering/transformer/test_residual_band_baseline_is_out_of_fold.py, 8 of
whose assertions were verified failing against the pre-fix modules.

---

### SRD-02 [P1] ERR reference pins `y_true.max()` after production froze `max_grade = 4.0`

**File:** `tests/metrics/test_ranking_batch_kernel_dispatch.py:37` (twin: `src/mlframe/metrics/_ranking_extras.py:276`, constant at `:35`)

**Summary:** `_ref_per_group` computes expected reciprocal rank with `max_grade = float(yt.max())` -- the old per-call default. Production was deliberately changed to the fixed ceiling `_DEFAULT_ERR_MAX_GRADE = 4.0`; the docstring at `_ranking_extras.py:270-273` explicitly rejects the old default ("A per-call `y_true.max()` default would re-scale the gain map per split and make train/test ERR incomparable"). The test passes only because its fixture is `rng.integers(0, 5, ...)`, whose max is coincidentally exactly 4.0.

**Failure scenario:** any relevance scale other than 0..4. With `y_true = rng.integers(0, 3, 700)`, 100 groups of 7, k=5: production `expected_reciprocal_rank` returns **0.18827772712707522**, the frozen reference returns **0.5725546875** -- 3.04x apart. Conversely, reverting production to `max_grade = y_true.max()` leaves the test green, so it guards the wrong thing.

**Evidence:** read `_err_batch_kernel`'s `mg = _DEFAULT_ERR_MAX_GRADE` feed at `_ranking_extras.py:276` and `:284` against the test's hardcoded `float(yt.max())`; ran both and printed the values above.

**Suggested fix:** have `_ref_per_group` import and use `_DEFAULT_ERR_MAX_GRADE`, and add a fixture whose relevance max is not 4 so the constant is actually pinned.

---

### SRD-03 [P1] "pre-CPX16" reference loads the live class; the identity test cannot fail

**File:** `tests/models/test_cpx16_optimization_membership_identity.py:58-100` (twin: `src/mlframe/models/optimization.py:265` -> `src/mlframe/models/_optimization_search.py`)

**Summary:** `_old_sequence` claims to exec "the HEAD version of `optimization.py`" and run the pre-CPX16 `MBHOptimizer` against the current one. It does neither. The worktree is clean, so `git show HEAD:...optimization.py` is byte-identical to the live file (md5 `c18269f79da3942a5d275a0c1127d259` on both sides). Permanently worse: Wave 100 moved `MBHOptimizer` out of `optimization.py` entirely -- `optimization.py:265` is now only `from ._optimization_search import _ETRWithStd, MBHOptimizer  # noqa: F401`. Exec'ing the "old" file therefore just re-imports the live class object.

**Failure scenario:** any regression in `MBHOptimizer.suggest_candidate` -- reverting `known_candidates_set` to an `x not in self.known_candidates` ndarray scan, or reintroducing the MODELS-1 int-cast that corrupts fractional search spaces -- is applied to both sides, so `new_seq == old_seq` stays trivially true for every seed. The test written specifically to guard the CPX16 optimization can never fail on any change to the code it claims to pin.

**Evidence:** replayed the test's own loader with `PYTHONPATH=src`: `mod.MBHOptimizer is mlframe.models.optimization.MBHOptimizer` -> `True`. Both `_new_sequence` and `_old_sequence` call `_run_sequence` on the same class object. `test_membership_set_matches_ndarray_for_np_scalar_keys` (`:103`) is also self-contained, so the only test in the file that touches production is `test_known_candidates_preserve_float_dtype_on_continuous_search_space` (`:114`).

**Suggested fix:** vendor the pre-CPX16 `suggest_candidate` body inline (as every other `_old_*` helper in this tree does), or `git show <pre-CPX16-sha>:src/mlframe/models/_optimization_search.py` with an explicit pinned SHA, never `HEAD`. Add `assert mod.MBHOptimizer is not MBHOptimizer` so this failure mode is self-detecting. Secondary: `_old_sequence` writes `tests/models/_cpx16_optimization_OLD_tmp.py` into the source tree instead of `tmp_path`.

---

### SRD-04 [P2] cpx36 frozen baseline missed the empty-band fallback fix

**File:** `src/mlframe/feature_engineering/_benchmarks/_cpx36_baseline/fisher_weighted_residual_old.py:117` (twin: `src/mlframe/feature_engineering/transformer/fisher_weighted_residual.py:180-182`; asserted by `tests/feature_engineering/test_cpx36_batched_predict_identity.py:64`)

**Summary:** the frozen baseline exists to pin ONE change (batched vs per-perturbation predict), and `test_batched_predict_bit_identical` asserts `np.array_equal` over ALL output columns. Production later gained an unrelated fix: `band_y_mean` is seeded with `np.full(n_bands, float(y_t.mean()))` instead of `np.zeros(n_bands)`, with the comment "Global fallback for an empty band (tied weighted-residual values collapsing a quantile boundary) -- 0.0 misleadingly reads as a genuinely low-residual band rather than 'no data'." The frozen copy still has `np.zeros`. This is the exact seed shape: an identity test guarding batching would fail on a statistic it never guarded.

**Failure scenario:** any fold whose `weighted_train` has enough ties to collapse quantile boundaries leaves a band empty. Replaying the band loop with `w = [0,0,0,0,0,0,0,0,5]`, `y = [1]*8+[9]`, `n_bands=5` (quantiles `[0,0,0,0,0,5]`): OLD -> `[1, 0, 0, 0, 9]`, NEW -> `[1, 1.8889, 1.8889, 1.8889, 9]`. The `fishres_band_y_mean` column then differs for every query row assigned to bands 1-3 and `np.array_equal(a, b)` fails, blaming the batching. Today the fixture happens to produce no empty band, so the column silently contributes nothing to the batching guard.

**Evidence:** full `diff -u` of all three `_cpx36_baseline/*_old.py` against their production twins; ran the band loop under both initialisers and printed the arrays above. (The other two baselines, `adversarial_flip_old.py` and `counterfactual_substitution_old.py`, are clean -- their diffs are the declared batching change plus docstrings and `np.asarray` wrappers.)

**Suggested fix:** change `band_y_mean = np.zeros(n_bands, ...)` to `np.full(n_bands, float(y_t.mean()), ...)` in the frozen baseline, matching what the snapshot is supposed to hold constant.

---

### SRD-05 [P2] pair-enumeration identity test imports nothing from mlframe

**File:** `tests/feature_selection/filters/test_cat_interactions_pair_enum_vectorized.py:29-42` (twin: `src/mlframe/feature_selection/filters/_cat_interactions_step.py:296-308`)

**Summary:** the module docstring says it "Pins the vectorized enumeration against a frozen copy of the pre-fix nested loop", but `_new_enum` is itself a hand-copy of production lines 296-308 (character equivalent modulo `_`-prefixed local names) and the file imports only `numpy`. Two local copies are compared to each other; production is never executed.

**Failure scenario:** drop `& (nb_prod < 2**31)` from `_cat_interactions_step.py:303`, or remove the `dtype=np.int64` cast on `:301` that the second test calls "load-bearing, not cosmetic" -- production then admits a 46341x46341 pair that overflows downstream int32 combined-code arithmetic, and both tests in this file still pass green, because the 46341/46341 case only ever reaches the test's own copy.

**Evidence:** `grep` for `mlframe` in the test file returns nothing; side-by-side read of test lines 29-40 against `_cat_interactions_step.py:296-308`.

**Suggested fix:** extract lines 296-308 into a module-level `_enumerate_candidate_pairs(candidate_idxs_arr, nbins, max_combined)` and have the test import and call it as the "new" side.

---

### SRD-06 [P2] `MLFRAME_MRMR_ADDONE_PVALUE=0` is honoured at one of four MRMR p-value sites

**File:** `src/mlframe/feature_selection/filters/permutation.py:155` (canonical `_perm_pvalue`) (twins that inline a frozen add-one form: `src/mlframe/feature_selection/filters/estimators.py:170`, `src/mlframe/feature_selection/filters/_cmi_perm_stop.py:155`, `src/mlframe/feature_selection/filters/_conditional_permutation.py:123`, `src/mlframe/feature_selection/structure_discovery.py:161`)

**Summary:** `_perm_pvalue` is the canonical helper and carries two later additions: the `full_budget` denominator correction, and an env-var opt-out (`MLFRAME_MRMR_ADDONE_PVALUE=0` -> "the legacy plain-rate estimator"), documented globally in `docs/ENVIRONMENT_VARIABLES.md:182`. Four other sites hardcode `(1 + count) / (1 + n)` inline and consult neither. Three of them are on the MRMR path the knob names: `evaluation.py:830` calls `cmi_permutation_stop` and `:868` calls `conditional_permutation_test`. `structure_discovery.py:144` additionally shadows the canonical name with a completely different signature `(feat_col, yb, nbins, n_perm, seed)`.

**Failure scenario:** an operator sets `MLFRAME_MRMR_ADDONE_PVALUE=0` to reproduce legacy selection. Verified with `PYTHONPATH=src MLFRAME_MRMR_ADDONE_PVALUE=0`: `_perm_pvalue(0, 50)` returns **0.0** while the inline form at `estimators.py:170` still returns **0.0196078431372549** (1/51). The run then mixes two p-value conventions across the same pipeline, and the significance gates at `permutation.py:786` and `_cmi_perm_stop.py:156` disagree about the same feature at `alpha` between those values. With the var unset (the default) all four agree, which is why this is latent.

**Evidence:** read all five sites; ran the two forms under the env var as above; confirmed the var is documented as a general knob, not a `fleuret`-local one.

**Suggested fix:** replace the four inline forms with calls to the canonical `filters.permutation._perm_pvalue`, and rename `structure_discovery._perm_pvalue` so it stops shadowing.

---

### SRD-07 [P2] group-aware MRMR greedy: three local copies asserted against each other

**File:** `tests/feature_selection/wrappers/test_ranker_fs_group_relevance_identity.py:158-248` (twin: `src/mlframe/training/ranking/_ranker_fs.py:413-436`, `group_aware_mrmr_select`)

**Summary:** `test_group_aware_mrmr_incremental_redundancy_matches_mean_reference` defines `_ref_greedy`, `_inc_greedy` and `_vec_greedy` locally and asserts they agree with each other. The shipped greedy loop is never imported or called.

**Failure scenario:** change `_ranker_fs.py:422-436` -- `scores[best_i] <= 0.0` to `< 0.0`, drop the `below_floor` mask, or change `red_sum / _ns` to `/ (ns + 1)` -- and the test stays green, because `_vec_greedy` is asserted against `_ref_greedy` and neither moves. The test's stated subject ("the shipped vectorised greedy loop") is unverified.

**Evidence:** no import of `group_aware_mrmr_select` anywhere in the file; the only production symbols imported (lines 19-23, 136) are `_binned_mi`, `_mi_from_edges`, `group_aware_relevance`, `_group_features_mi_njit`, none of which this test uses.

**Suggested fix:** drive `group_aware_mrmr_select` with a synthetic `rel`/`red` (or extract the greedy loop into a callable production helper) and keep only `_ref_greedy` as the reference.

---

### SRD-08 [P2] y-clip-bounds "bit-identity" test never calls the function

**File:** `tests/training/composite/kernels_identity/test_y_clip_bounds_quantile_bit_identity.py:15-85` (twin: `src/mlframe/training/composite/estimator/__init__.py:40-61`, `_y_train_clip_bounds`)

**Summary:** the declared reference `_reference_bounds` (`:15`) is never called from anywhere. `test_quantiles_bit_identical_to_two_calls` (`:29`) compares `np.quantile(y, 0.001)` against `np.quantile(y, (0.001, 0.999))[0]` -- a numpy-vs-numpy identity; it imports `_y_train_clip_bounds` and never calls it. `test_perf_sentinel_one_call_not_slower` (`:54`) likewise times two local lambdas.

**Failure scenario:** change `estimator/__init__.py:52` from `np.quantile(finite, (0.001, 0.999))` to any other probability pair, revert it to two separate calls, or drop the `finite = y_train[np.isfinite(y_train)]` filter at `:48` -- the headline "bit-identity" test still passes. Only `test_clip_bounds_stable_on_fixture` (`:40`) reaches production, and it uses all-finite input, so the NaN filter at `:48-50` is untested by the whole module.

**Evidence:** `grep -rn "_reference_bounds" tests/` returns only the definition line. The constants do currently agree (`_Y_CLIP_LOW_FRAC = 0.1` / `_Y_CLIP_HIGH_FRAC = 10.0` at `:34-35` give the `0.9`/`9.0` the test hardcodes at `:49-50`), which is why this is a coverage hole rather than a live divergence.

**Suggested fix:** delete `_reference_bounds` or wire it in; rewrite the identity test to compare `_y_train_clip_bounds(y)` against a two-call reference that reconstructs both bounds including the `_Y_CLIP_*_FRAC` extension; add NaN/inf and `span <= 0` cases.

---

### SRD-09 [P2] GPU joint-MI reduction re-typed inside the test body

**File:** `tests/feature_selection/regression/test_regression_mrmr_audit_2026_07_22.py:1064` (reference at `:1069`, inline copy of the optimised path at `:1107-1120`; twin: `src/mlframe/feature_selection/filters/_gpu_pairs.py:252-265`)

**Summary:** the test's named contract is that the vectorised reduction "must produce IDENTICAL results to the original triple-nested Python loop". It defines `_reference_loop` and then re-types production's vectorised reduction inline rather than calling it. `mlframe.feature_selection.filters._gpu_pairs` is not imported in this test function (the file's only `_gpu_pairs` import, at `:1011`, belongs to the unrelated `test_regression_gpu_pairs_shared_mem_set_has_lock`).

**Failure scenario:** change `_gpu_pairs.py:264` from `joint_mi_out[k] = float(np.sum(jf * np.log(ratio)))` to `np.sum(jf) * np.log(...)`, or flip the `valid` mask to drop the `marg_y > 0` term -- every GPU-path pair MI becomes wrong and this test still passes, because it re-executes its own private copy of the correct code.

**Evidence:** `grep -n "_gpu_pairs"` in the file hits only lines 1000 (comment), 1006, 1011 (other test), 1058 (comment), 1064 (this test's name). Test lines 1109-1120 are a character-for-character duplicate of `_gpu_pairs.py:252-264`; the two currently agree, so the test is not wrong today, it simply guards nothing.

**Suggested fix:** extract the reduction into a module-level `_joint_mi_from_counts(joint_counts_host, joint_offsets, pair_merged_sizes, nbins_y, n_pairs, n_total)`, call it from the batched-CUDA path, and have the test import that symbol. This also drops the cupy dependency from the test, since the helper is pure numpy.

---

### SRD-10 [P2] cat-confirm permutation identity test is degenerate (0.0 == 0.0)

**File:** `tests/feature_selection/fe/categorical/test_cat_confirm_permutation_single_merge_cache.py:170-216` (twin: `src/mlframe/feature_selection/filters/_cat_confirm_permutation.py:683-960`)

**Summary:** `test_single_merge_cache_bit_identical_to_uncached_reference` compares confidences that are exactly 0.0 on both sides for every pair, so the "bit-identity" loop is `0.0 == 0.0` and would pass for any numerics change in the cached path.

**Failure scenario:** the fixture `_make_data(n=3000, n_cols=5)` with `ii_arr = 0.0005` (`:128`/`:183`) makes `ii_obs` so small that `ii_perm >= ii_obs` fires on every permutation, so `n_failed == n_perms` -> `p = (n_perms + 1) / (n_perms + 1) = 1.0` -> `conf = 0.0` for all four star pairs, at both `n_perms=30` and `100`. Any cache bug that changes a confidence from, say, 0.94 to 0.69 is invisible here. Secondary drift at the same site: the reference hardcodes `_count_nfailed_joint_indep_serial` while production selects serial/prange/cupy via the per-host tuned `_perm_kernel_backend_choice` (`:900-935`); on a host whose tuning cache maps small `n` to `cpu_parallel` the two paths use different per-thread seeds. Currently masked by the degeneracy.

**Evidence:** ran production `_confirm_pairs_via_permutation` with the test's exact inputs -- printed `30 {(1,2):0.0, (1,3):0.0, (1,4):0.0, (1,5):0.0}` and all-0.0 again at 100. Direct kernel check: `_count_nfailed_joint_indep_serial(...) == 100` and `_count_nfailed_joint_indep_prange(...) == 100`, so even a backend swap is invisible.

**Suggested fix:** raise `ii_arr` into the discriminating band (the sibling conditional test at `test_cat_confirm_conditional_hoist_x1_mi.py` achieves 0.94/0.69/0.0), assert `any(v not in (0.0, 1.0) for v in conf.values())` before the equality loop, and pin `cfg.backend` so the reference's serial kernel is the one production uses.

---

### SRD-11 [P2] engineered-dedup: local "new" copy has already drifted from production

**File:** `tests/feature_selection/filters/test_eng_dedup_batch_corr_masked_kernel.py:131-212` (twin: `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core.py:758-868`)

**Summary:** `_new_dedup` is described as mirroring "the current `_fit_impl_core.py` masked-buffer integration" but is a local copy; the test imports only the leaf kernel `one_vs_many_abs_corr_masked` and never invokes production's dedup block. The copy is already stale.

**Failure scenario:** production has two branches the copy lacks. (a) `_fit_impl_core.py:762-771` force-keeps columns in `_adaptive_fourier_keep` -- appended to `_eng_keep`/`_eng_arrs` with no rank-buffer row and never dedup-tested as a candidate. (b) `:774-778` collapses a duplicate-label `X[_c]` DataFrame to its first column. Given an engineered set with two rank-identical adaptive-Fourier columns, production keeps both while the test's `_new_dedup` drops one. Since the test asserts nothing about production, neither this divergence nor any future change to the real block is caught.

**Evidence:** side-by-side of test lines 139-148 against `_fit_impl_core.py:758-782`; the `if _c in _adaptive_fourier_keep:` and `isinstance(_col_view, pd.DataFrame)` branches have no counterpart in the copy. The remainder (rank cache, fast-path gate `fully_finite and n >= 8 and next_row > 0`, the `>= 0.99` threshold, the buffer append) matches line for line.

**Suggested fix:** extract the dedup block into a callable helper and drive that from the test (old loop vs real production function) instead of maintaining a second copy of the "new" side.

---

### SRD-12 [P2] DFA reference keeps the pre-fix `n < 20` guard; production moved to `n < 50`

**File:** `tests/feature_engineering/test_hurst.py:118` (guard at `:121`) (twin: `src/mlframe/feature_engineering/hurst.py:226`)

**Summary:** `_ref_dfa_alpha` freezes `if n < 20: return np.nan`; production `dfa_alpha` now uses `if n < 50: return np.nan`. Every other line matches (the only other difference is the `t`/`tm` hoist, which is genuinely bit-identical).

**Failure scenario:** `x = np.cumsum(default_rng(0).standard_normal(45))` gives production `dfa_alpha(x) = nan` and `_ref_dfa_alpha(x) = 2.049126989836409`; same at n=46..49; both give `1.4523072576095135` at n=50. The test only ever calls with n=2000, so the divergence is invisible and anyone reverting the 50 -> 20 guard sees a green suite.

**Evidence:** read both bodies; ran both kernels side by side at n=45, 48, 49, 50 and printed the values.

**Suggested fix:** change the reference's guard to `n < 50` and parametrise over `n in (45, 49, 50, 2000)`.

---

### SRD-13 [P2] numerical-stability kernels: four print-only tests over a module with no production consumer

**File:** `tests/feature_engineering/test_numerical_stability_bench.py:113, 138, 157, 180` (twin: `src/mlframe/feature_engineering/_numerical_stable.py`, whole module)

**Summary:** four of the five tests contain zero asserts -- they only `print`. Every kernel they exercise (`welford_mean_var_seq`, `welford_moments_seq`, `kahan_sum_seq`, `kahan_two_pass_var_seq`, `naive_*_two_pass_seq`) has no production consumer: `grep -rn "_numerical_stable" src/` returns nothing outside the module itself. The module docstring's claim that `naive_mean_var_two_pass_seq` is "the current pattern in `compute_simple_stats_numba`" is stale -- `compute_simple_stats_numba` (`numerical.py:278`) dispatches to `_compute_simple_stats_fast` / `_compute_simple_stats_compensated`.

**Failure scenario:** a regression making `welford_moments_seq` return `nan` passes 4/5 tests silently; only the skew sign on one fixture is checked. And even a correct fix there cannot affect any shipped feature, because nothing in `src/` calls these kernels.

**Evidence:** read the whole test file; grepped `src/` for consumers (none outside the module); confirmed `sp_skew(fixture) = 6.805`, so the `or abs(ref_skew) < 1e-3` escape in the single real assert is not what makes it weak.

**Suggested fix:** either wire `compute_simple_stats_numba`'s compensated path to these kernels, or move the file under `benchmarks/` and replace the prints with per-distribution relative-error assertions.

---

### SRD-14 [P2] local-curvature quad-term identity compares two local copies

**File:** `tests/feature_engineering/transformer/test_local_curvature_quadterm_broadcast_identity.py:38, 64` (twin: `src/mlframe/feature_engineering/transformer/local_curvature.py:77-104`)

**Summary:** `test_quadterm_and_hessian_construction_bit_identical` asserts `_old_construction == _new_construction`, both defined in the test file. Production `compute_local_curvature_features` is never asked for its `A_quad`/`H`; the only test that calls production asserts that two identical calls are deterministic and finite.

**Failure scenario:** change `local_curvature.py:104` to `H[iu[diag_mask], ju[diag_mask]] = quad_coefs[diag_mask]` (dropping the factor 2 on the diagonal, so `trace_H` and `frob_H` are wrong) -- both tests still pass: the identity test never touches production, and the determinism test still gets `a == b`.

**Evidence:** read the test against `local_curvature.py:72-105`; they currently agree, but no assertion links them.

**Suggested fix:** extract the construction into a module-level helper in `local_curvature.py` and have the test import it, so `_old_construction` is diffed against the real code path.

---

### SRD-15 [P3] `prediction_band_attention` in-sample predictions (same cluster as SRD-01)

**File:** `src/mlframe/feature_engineering/transformer/prediction_band_attention.py:44` (twin: `bidir_residual_band.py:41`)

**Summary:** a fourth `_fit_baseline_predict` with the same name and signature as the SRD-01 cluster, also documented as returning "in-sample predictions". Its module docstring (`:3`) explicitly positions it as "Orthogonal to residual-band family (iters 60-63)": it bands on prediction quantiles (`:107-115`), not residual quantiles, so the "residual is understated by construction" argument does not transfer directly.

**Failure scenario:** in-sample predictions are still shrunk toward `y_t` relative to honest ones, so the prediction-quantile band boundaries at `:107` sit on an overfit prediction distribution. Not quantified here, and the sign of the effect on band membership is not the same clean argument as SRD-01.

**Evidence:** read `prediction_band_attention.py:44-63` and `:103-117` against `bidir_residual_band.py`.

**Suggested fix:** decide explicitly whether this module opts out of the cluster's OOF convention, and record the decision in the docstring next to the "Orthogonal to residual-band family" note -- the current state is indistinguishable from having been missed alongside SRD-01's three.

*Alternative reading:* this may be deliberate, which is why it is P3 and separate from SRD-01.

---

### SRD-16 [P3] backward-elimination reference freezes the pre-fix coupled bar

**File:** `tests/feature_selection/test_biz_val_greedy_backward_elimination.py:94-110` (twin: `src/mlframe/feature_selection/greedy_backward_elimination.py:151-164`)

**Summary:** `_reference_greedy_backward_elimination` is a verbatim copy of the pre-fix loop, where the running maximum and the acceptance bar are the same variable (`best_score` seeded from `current_score`, candidates accepted on `score > best_score + tol`). Production was explicitly fixed (comment at `:145-150`) to separate them: a `best_score = -inf` argmax scan, then a single `best_score <= current_score + tol` acceptance check. The two forms coincide only at `tol == 0.0`, which is the only value the test exercises.

**Failure scenario:** with `tol > 0` the frozen reference is order-dependent -- an earlier accepted candidate raises the bar for later ones, so it can drop a non-argmax column while production always drops the argmax. The test never passes `tol`, so the fix it claims to pin is untested and a revert to the coupled-variable form keeps it green.

**Evidence:** test line 104 `if score > best_score + tol: best_score, best_candidate = score, col` (bar == running max) vs production `:156-160` (`if score > best_score` argmax, then a separate `best_score <= current_score + tol` gate).

**Suggested fix:** update the reference to the two-variable form and add a `tol > 0` parametrisation with a column-order permutation that distinguishes them.

---

### SRD-17 [P3] CPI reference's single-feature branch predates the nanmean/try-except fix

**File:** `tests/feature_selection/wrappers/test_helpers_importance_scratch_identity.py:188-247` (twin: `src/mlframe/feature_selection/wrappers/_helpers_importance.py:284-302`)

**Summary:** the frozen `_reference_cpi` single-feature branch uses `np.mean(score_losses)` with no try/except around `model.score`; production uses `np.nanmean(...)` inside the E11 try/except-and-record-NaN wrapper.

**Failure scenario:** not live -- the identity test uses `p=30`, so the `Xnotj.shape[1] == 0` branch is never reached by the comparison (`test_single_feature_path_runs` calls production alone with no reference). It is a frozen branch free to drift further unobserved.

**Evidence:** read both branches.

**Suggested fix:** mirror the nanmean/try-except in the reference and add a `p=1` identity case, or delete the dead branch from the reference.

---

### SRD-18 [P3] fuzz text-row test duplicates the builder and its vocab

**File:** `tests/training/fuzz/test_fuzz_text_row_build.py:33-83` (twin: `tests/training/_fuzz_combo/frame_builder.py:239-280`)

**Summary:** `test_vectorised_text_rows_match_old_per_row_build` asserts `_new_rows(n, seed) == _old_rows(n, seed)`, both local. `_new_rows` (`:71`) duplicates `frame_builder.py:276-280` and `_TEXT_VOCAB` (`:33-58`) duplicates the 24-word `text_vocab` list at `frame_builder.py:239-264`. The real builder is never called.

**Failure scenario:** change the vocab list or the draw shape in `frame_builder.py` (e.g. `size=(n,3)` -> 4 tokens, or a word added/reordered) and the test keeps comparing its own two frozen copies and passes. The only other test in the file asserts `len(s.split()) == 3`, which a vocab change survives. Rated P3 because the guarded code is the fuzz harness, not shipped `src/`.

**Evidence:** no `mlframe` import in the file; diffed the test's `_new_rows`/`_TEXT_VOCAB` against `frame_builder.py:239-280` -- they match token-for-token today.

**Suggested fix:** import `text_vocab` (or a shared `_build_text_rows` helper extracted from `frame_builder.py`) rather than redeclaring it.

---

### SRD-19 [P3] pre-screen reference freezes a pre-sparse-support branch

**File:** `tests/feature_selection/screening/test_pre_screen_unsupervised.py:126-155` (twin: `src/mlframe/feature_selection/pre_screen.py:137-228`)

**Summary:** `_reference_drops_via_isna` keeps `if isinstance(s.dtype, pd.SparseDtype): continue` (sparse columns never dropped) from before production grew a sparse-aware null count (`:152-161`) and a closed-form sparse population-variance branch (`:190-228`) that can drop them.

**Failure scenario:** add `df["sp"] = pd.arrays.SparseArray(np.full(2000, 7.0), fill_value=7.0)` to the fixture: production returns `['sp', ...]` (constant sparse -> variance 0 -> dropped), the reference returns the set without `'sp'`, so `assert fast == reference` fails on a case where production is correct. Verified live: `compute_unsupervised_drops(pd.DataFrame({"sp": pd.arrays.SparseArray(np.full(100, 7.0), fill_value=7.0)}))` -> `['sp']`, while `_reference_drops_via_isna` skips it and returns `[]`. The fixture has no sparse column, so the sparse path added specifically to stop TF-IDF passthrough columns being screened out has zero reference coverage.

**Evidence:** read both; ran the sparse case above.

**Suggested fix:** mirror production's sparse branch in the reference and add constant-sparse, NaN-filled-sparse and rare-nonzero-sparse columns to the fixture.

---

### SRD-20 [P3] decorrelator reference returns labels; production canonicalised to positions

**File:** `tests/estimators/test_cpx39_decorrelator_identity.py:12` (loop at `:20`) (twin: `src/mlframe/estimators/custom.py:37`)

**Summary:** the reference returns column labels (`corr_matrix.columns[i]`); production now canonicalises `correlated_features_` to integer positions (`{c for c in range(len(cols)) ...}`), a deliberate fix for fit-on-DataFrame / transform-on-ndarray. The sets are equal only when labels happen to equal positions, which is exactly what `_make()` produces (a default RangeIndex).

**Failure scenario:** `X = pd.DataFrame(b, columns=['a','b','c','d'])` with `b[:,3] = b[:,1]`, threshold 0.95 -> production `correlated_features_ == {3}`, reference `== {'d'}`, and the identity assertion fails. So the test cannot be extended to named columns as written, and today it does not guard the label -> position canonicalisation at all.

**Evidence:** ran both on a named-column frame (`prod {3}` / `ref {'d'}`).

**Suggested fix:** make the reference emit positions (`correlated_features.add(i)`) and add a named-column parametrisation.

---

### SRD-21 [P3] bootstrap-bundle reference's jackknife wiring is stale

**File:** `tests/evaluation/test_bootstrap_fused_binary_bundle.py:38` (`jackknife_fns` at `:68`) (twin: `src/mlframe/training/honest_diagnostics.py:176-178`)

**Summary:** the reference docstring says it "Replicates `honest_diagnostics._bootstrap_block`'s exact `metric_fns`/`jackknife_fns` wiring", but `_bootstrap_block` also wires `jackknife_fns["ece"] = _jackknife_ece` (added 2026-07-31); the reference passes only `{"roc_auc": ...}`, so ECE falls back to the generic gather jackknife. The fused production path does use `_jackknife_ece` (`_bootstrap_fused_binary_bundle.py:326`).

**Failure scenario:** n=5000, n_bootstrap=100, seed 0: `ref["ece"]["lo"] = 0.34180582016483996` vs fused `0.34180582016484`, delta 5.55e-17. With the production wiring the delta is exactly 0.0. The test's 1e-9 tolerance absorbs it, so a real BCa-acceleration divergence in the ECE closed form is only caught above 1e-9 -- and the stated contract ("mirrors prod wiring") is already false.

**Evidence:** read both wirings; ran `bootstrap_metrics` with and without `jackknife_fns["ece"]` against the fused bundle and printed the deltas above.

**Suggested fix:** add `"ece": lambda yy, pp: _jackknife_ece(yy, pp)` to the reference's `jackknife_fns`.

---

### SRD-22 [P3] ADASYN reference lacks production's `n_neighbors` cap

**File:** `tests/feature_engineering/transformer/test_adasyn_smote_synthesize_vectorized_identity.py:26` (loop at `:33`; twin: `src/mlframe/feature_engineering/transformer/adasyn_smote.py:53`)

**Summary:** the reference uses `NearestNeighbors(n_neighbors=k_global + 1)`; production caps it as `min(k_global + 1, X_full.shape[0])` (the FE_TRANSFORMER_A-4 fix). Results are identical for every current parametrisation because `_make` always builds `n_full = nrows * 3 >= k_global + 1`.

**Failure scenario:** add a parameter set with `X_full` smaller than `k_global + 1` (e.g. `n_full=5`, `k_global=10`) and the reference raises sklearn's `ValueError: Expected n_neighbors <= n_samples` -- a failure unrelated to the vectorisation the test pins. (The cap itself is separately covered by `test_adasyn_synthesize_k_global_exceeds_small_full_dataset`.)

**Evidence:** read both; re-ran all 12 seed x parameter identity combos -- all bit-identical, `array_equal True`, 0 differing elements.

**Suggested fix:** mirror the cap in `_old_reference`.

---

### SRD-23 [P3] dead frozen reference in the numerical fast-path test

**File:** `tests/feature_engineering/test_numerical.py:771` (`_reference_via_unique_path`)

**Summary:** the helper is defined inside `TestFusedNuniqueModesQuantilesFastPath` and never called; the tests inline `np.unique` instead. It is a frozen reference that guards nothing and will drift unnoticed.

**Failure scenario:** none today -- it is unreachable. Listed because it is the seed condition for this bug class: a reference that nothing calls diverges silently and is later trusted.

**Evidence:** the only occurrence of the name in the file is its definition.

**Suggested fix:** delete it, or use it in `test_fast_path_nunique_modes_ncrossings_bit_identical` in place of the inline `np.unique`.

---

### SRD-24 [P3] MDL-binning combo tests compare two local copies

**File:** `tests/feature_engineering/transformer/test_mdl_binning_combo_count_vectorized.py:26-56, 76-117` (twin: `src/mlframe/feature_engineering/transformer/mdl_binning_pairwise.py:222-230`)

**Summary:** both the identity test and the FE_TRANSFORMER_B-4 collision test compare local copies (`_old_combo`/`_new_combo`, plus an inline re-implementation of the dynamic-base encoding at `:99-106`). Production `compute_mdl_binning_pairwise_features` is only exercised by `test_full_feature_function_runs_and_deterministic`, which asserts shape/finiteness/determinism and never the combo counts or the base.

**Failure scenario:** revert `mdl_binning_pairwise.py:222` to `combo_base = 100`; with feature 1 having more than 100 bin edges, `(bin0=0, bin1=105)` and `(bin0=1, bin1=5)` collide and `mdlbin_combo_count` returns the summed count. All four tests in the file still pass -- the collision test only compares its own inline copy against a local `Counter`.

**Evidence:** production lines 222-230 are the source that test lines 99-106 duplicate verbatim; `combo_base` is never read from the module in any assertion.

**Suggested fix:** drive the collision case through `compute_mdl_binning_pairwise_features` (or export the combo helper) so the dynamic base is asserted at the production site.

---

### SRD-25 [P3] cluster-aggregate compact-stack test never calls the step it pins

**File:** `tests/feature_selection/filters/test_cluster_aggregate_mi_compact_stack_identity.py:22-41` (twin: `src/mlframe/feature_selection/filters/_cluster_aggregate.py:582-583`)

**Summary:** `_old_form` and `_new_form` are both local; only the leaf `mi` is imported. `run_cluster_aggregate_step` -- the function whose optimisation is being pinned -- is never called.

**Failure scenario:** change `_cluster_aggregate.py:583` from `np.arange(_n_t)` to `np.array([0])` while `target` has 2 columns: MI is then computed against only the first target column, and `test_compact_stack_handles_multi_target` still passes because it exercises the test's own `_new_form`.

**Evidence:** production lines 582-583 currently match `_new_form` (`_target_block == data[:, tcols]`, `_compact_nbins == nbins[tcols] + [qnb]`); the test's only mlframe import is `mi`.

**Suggested fix:** assert the equality through `run_cluster_aggregate_step`, or extract the compact-stack scoring into a helper the test can call.

---

### SRD-26 [P3] binned-numeric-agg reference lacks production's two skip guards

**File:** `tests/feature_selection/fe/test_binned_numeric_agg_fe.py:226-274` (twin: `src/mlframe/feature_selection/filters/_binned_numeric_agg_fe.py:356-372`)

**Summary:** the frozen `_old_reference` lacks production's `if not np.isfinite(gvals).all(): continue` (`:363`) and `if edges.size == 0: continue` (`:367`), and uses raw `np.quantile(...)` where production uses `quantile_edges`.

**Failure scenario:** put one NaN in `g0` -- production returns zero feature columns, the frozen reference returns 4 (`np.quantile` on NaN data yields NaN edges), so `assert set(ref.columns) == set(feat_df.columns)` fails for a reason unrelated to the fold-gate / global-stats fusion the test claims to pin. Verified: `fit_binned_numeric_agg` on `X` with `X.loc[0, 'g0'] = np.nan` returns `feat_df.columns == []`; the reference's loop body has no such guard. The current fixture is all-finite, so the divergence is latent.

**Evidence:** read both loop bodies; ran the NaN case above.

**Suggested fix:** mirror the two guards in `_old_reference`, or seed the fixture with a NaN-bearing group column and assert both sides skip it.

---

### SRD-27 [P3] `bench_compare_bootstrap` identity check is now always False

**File:** `src/mlframe/training/composite/_benchmarks/bench_compare_bootstrap.py:26` (twin: `src/mlframe/training/composite/compare.py:162`)

**Summary:** `_old_monolithic` computes the bootstrap p-value as the raw fraction `tail = np.mean(boot_means <= 0)`. Production `_paired_bootstrap_ci` adopted the Davison-Hinkley add-one correction `tail = (n_tail + 1) / (n_boot + 1)` (documented at `compare.py:128-138`). The bench's final "CI identity ... bit-identical" check compares full 3-tuples, so it now prints `False` for every input, since `count/n_boot != (count+1)/(n_boot+1)` unless `count == n_boot`. This is the src-side twin of the seed finding, still unfixed.

**Failure scenario:** `diff = default_rng(7).standard_normal(2000)*0.1 + 0.01`, `n_boot=1000`, `alpha=0.05`, `rng=default_rng(123)` -> OLD `(0.0016798894911289484, 0.009998141981685615, 0.008)` vs NEW `(..., ..., 0.00999000999000999)`; printed `bit-identical=False`. The CI bounds `lo`/`hi` do match bit-for-bit (the block-chunking RNG-order argument holds), so only the p-value slot diverges.

**Evidence:** ran both.

**Suggested fix:** add the `+1/+1` smoothing to `_old_monolithic`, or restrict the identity print to `a[:2] == b[:2]` and note the p-value convention change separately.

---

### SRD-28 [P3] minimax bench freezes both sides and misses production's empty-opponents guard

**File:** `src/mlframe/votenrank/_benchmarks/bench_minimax_winning_votes.py:26` (twin: `src/mlframe/votenrank/leaderboard/_rules.py:167-173`)

**Summary:** the bench freezes both `_old_minimax` and `_new_minimax` as local reimplementations and never calls production `minimax_ranking`. Production later gained an empty-opponents guard (`score = opponents.max() if not opponents.empty else 0.0`, with a comment explaining that a NaN silently breaks `minimax_election`'s `ranking == ranking.max()` winner selection). Neither bench copy has it, so `assert a.equals(b)` passes while validating pre-fix behaviour, and the reported speedup is measured against a function that is no longer what ships.

**Failure scenario:** single-model leaderboard `tbl = DataFrame([[1.0, 2.0]], index=['m0'])` -> bench `_new_minimax` returns `{'m0': nan}`; production `Leaderboard(tbl).minimax_ranking()` returns `{'m0': -0.0}`.

**Evidence:** ran both.

**Suggested fix:** have `_new_minimax` call the shipped `Leaderboard.minimax_ranking` (or add the same `opponents.empty -> 0.0` branch), and add a 1-model case to the bench shapes.

---

### SRD-29 [P3] drift bench's frozen kernels lack production's finite filter

**File:** `src/mlframe/metrics/_benchmarks/bench_drift_fused_merge_iter78.py:20, 29` (twin: `src/mlframe/metrics/_drift.py:345-348`, `:397-399`)

**Summary:** `_old_w1` / `_old_ks` are frozen pre-fix copies lacking production's `a = a[np.isfinite(a)]` filter and the `size == 0 -> nan` guards. The bench's `assert abs(new - old) < 1e-10` (`:44`) therefore guards nothing on the non-finite path, which the bench never generates.

**Failure scenario:** `a = [1., 2., nan, 4.]`, `b = [1., 2., 3., 4.]` -> `_old_w1 = nan` vs `wasserstein_1d = 0.3333333333333333`; `_old_ks = 0.25` vs `ks_distribution_distance = 0.16666666666666663`.

**Evidence:** ran both.

**Suggested fix:** add the finite-filter and empty guards to the two `_old_*` copies and extend `main()` with one NaN-bearing shape.

---

### SRD-30 [P3] `_subsample_indices` inlined into a sibling module with nothing pinning them equal

**File:** `src/mlframe/feature_selection/filters/_orthogonal_hsic_fe.py:124` (twin: `src/mlframe/feature_selection/filters/_orthogonal_dcor_fe.py:102`)

**Summary:** the HSIC copy's own docstring says "Identical to the Layer 67 helper but inlined here to keep the sibling module dependency surface tight (no cross-layer import)". The two bodies are currently byte-identical after AST normalisation (only the docstrings differ), so this is shape (2) of the bug class with the drift not yet realised.

**Failure scenario:** no divergence today. Nothing imports across the two, and no test asserts they agree -- `grep -rn "_subsample_indices"` finds only the two definitions, their five call sites, and an unrelated third definition at `training/composite/discovery/_stability.py:130` with a different signature. A seeding or `n <= n_sample` change applied to one is invisible in the other; the two modules' subsample index sets would silently stop matching for the same `random_state`.

**Evidence:** AST body-hash duplicate scan over `src/mlframe` flagged the pair; `diff` of the two bodies confirms only docstring differences.

**Suggested fix:** if the no-cross-layer-import constraint is real, add a test asserting the two produce identical indices for a shared `(n, n_sample, random_state)` grid so a one-sided edit fails loudly. Otherwise share one helper.

---

## Sub-areas checked and found clean

Reported explicitly rather than omitted.

**Frozen references verified equivalent to their current production twin** (read line by line, most re-run numerically): `test_cusum_walk_njit` - `test_date_features_batched_pandas` - `test_ensemble_features_histogram_njit` - `test_ewma_residual_recurrence_njit` - `test_spectral_centroid_matvec_identity` - `test_borderline_smote_` / `test_density_weighted_smote_` / `test_pseudo_smote_synthesize_vectorized_identity` - `test_local_intrinsic_dim_batched_spectrum_identity` - `test_local_linear_batched_ridge_identity` - `test_per_column_rff_fused_njit` - `test_monotonic_stability_batched_spearman_equivalence` - `test_categorize_dataset_adaptive_searchsorted_batch` - `test_cat_confirm_conditional_hoist_x1_mi` - `test_fe_encoding_vectorized` - `test_corr_sq_centered_noalloc` - `test_dispersion_zscore_njit_identity` - `test_fastmi_mise_lse_hoist_identity` (32/32 exact) - `test_ksg_count_within_eps_identity` (9/9) - `test_rankgauss_avg_tie_rank_single_sweep` - `test_rankgauss_fit_self_rank_identity` - `test_ratio_delta_fe_identity` - `test_target_encoding_apply_identity` - `test_temporal_agg_fe_identity` - `test_joint_entropy_2var` / `test_joint_freqs_2var` - `test_biz_value_mrmr_interaction_info_prefilter_speedup` - `test_mrmr_degenerate_frames` - the four `shap_proxied` tests - `test_biz_val_unanimous_permutation_prune` - `test_mi_greedy_fused_joint_entropy` - `test_perm_null_marginal_hoist` - `test_noise_floor_plateau_vectorize_identity` - `test_calibration_binning_strategy` - `test_fused_regression_metrics_block` - `test_accuracy_ratio_tie_invariance` (max diff 0 over 40 cases) - `test_brier_fused_validation` - `test_optimal_threshold_bootstrap_ci` - `test_prob_separation_seq_fused_identity` - `test_ranking_public_batch_dispatch_cpx24` - `test_ranking_sort_count_hoist_cpx23` - `test_cpx15_selection_split_identity` - `test_favorize_unexplored_cat_loop_identity` - `test_pairwise_corr_scatter_identity` - `test_ltr_charts_perf` - `test_matplotlib_heatmap_batch_text_color` - `test_viz_perf` - `test_dummy_baseline_stratified_accumulator` (bit-identical over 1000 draws) - `test_grouped_fit_segment_identity` - `test_qrf_leaf_bucketed_weights_kernel` - `test_quantile_assign_bins_njit_identity` - `test_screening_prebin_dtype` - `test_biz_val_feature_subset_bagging` - `test_composite_auto_detect_biz_value` - `test_composite_target_name_parse` - `test_mi_y_prebin_speedup` - `test_pu_learning_synthetic` - `test_remove_constant_columns_parity` - `test_biz_val_shapley_blend` - `test_regression_minimax_winning_votes`.

**The `+1` / `ceil((n+1)(1-alpha))` correction sites the brief flagged as classic drift candidates are consistent.** Conformal radius: `test_conformal_online_incremental_identity` and `test_conformal_mondrian_identity` match `conformal_online.py:54`/`:78` and `conformal.py:62` including both saturations and the `rank > m -> inf` guard; the four other `ceil((n+1)(1-alpha))` sites (`conformal.py:62`, `:304`, `conformal_classification.py:64`, `conformal_glm.py:84`) agree. Quantile-edge derivation: `test_quantile_assign_bins_njit_identity` and `test_screening_prebin_dtype` match their production kernels statement for statement. Bootstrap/permutation p-values: all `(count+1)/(n+1)` sites (`_cat_confirm_fwer.py:141`, `_cat_confirm_permutation.py:957`, `_dcd_swap.py:637`, `_dcd_swap_null.py:163`/`:259`, `_mdlp_validated_split.py:279`, `_eval_stats.py:184`, `compare.py:162`) are consistent; `_dummy_bootstrap.py:259` uses `(n+1)/(n+2)` deliberately and documents why. SRD-06 is the one exception, and it is about the opt-out path only.

**src-side benchmark frozen copies verified clean:** `bench_cpx27_conformal_online` - `bench_macro_avg_present_classes` - `bench_top_k_accuracy_kernel` - `bench_tune_decision_threshold` - `bench_noise_floor_plateau` - `bench_ranking_public_batch_dispatch_cpx24` (its `_old_*` wrappers call the live per-query primitives, so they cannot drift).

**Production-inlines-production candidates cleared:** `_hermite_fe_optimise.py:801 _baseline_mi_pair` delegates to the real estimators rather than copying them - `boruta_shap/_fit_explain.py:69 _naive_accepted_set_stable` is a documented rejected-contrast rule with no call site in the fit loop - `_numerical_stable.py`'s `naive_*_two_pass_seq` are precision baselines with no identity assert (see SRD-13 for the separate problem there) - `discovery/_collinear_numba.py:250 _ref_pair_corr` IS an inlined copy of `_eval_stats._near_collinear_keep_mask_numpy`'s pair arithmetic but both sides carry the same `1e-24` `_VAR_FLOOR`, the same `n_pair < 3` skip and the same strict `corr > thr` - `fs_hybrid/` has no duplicated helper besides the already-fixed `compute_auc_mean` - all three Mann-Whitney null-variance sites (`split_comparison.py:167`, `_drift_shared.py:57`, `_drift_adversarial.py:198`) carry the same `n <= 0` guard - the exact-duplicate pairs `class_balanced_hard_row._fit_predict` / `residual_band_attention._fit_predict`, `distributional.__init__` / `quantile.__init__`, and `row_wise_summary_polars._quantile_expr` / `bench_row_wise_polars._interp_quantile` are byte-identical today (latent only, and less exposed than SRD-30, which is the one with an explicit "inlined here" comment).

**Helpers checked and confirmed NOT frozen copies** (independent baselines or fixtures, so out of scope): `tests/core/test_arrays.py` `_baseline_argsort*` (plain `np.argsort`) - `test_hard_row_attention.py` `_expected_cols` (column-name contract, asserted against real production) - `test_biz_val_multicollinear_pollution.py` `_baseline_auc` - `test_mechanism_dataset_showdown.py` `_baseline_score` - `test_mrmr_concurrency_fixes.py` `_slow_check` (a thread-timing shim wrapping the real `_check_groups_contract`) - `test_composite_gate_and_edges.py` `_slow_ar1_dominant_lag` (data generator) - `test_jit_prewarm_scope.py` `_slow_body` - `test_learning_curve.py` `_slow_factory`.

**Not covered by this audit:** dynamically constructed references (a copy assembled via `exec`/`getattr` rather than a named `def`) would not be caught by the name-pattern enumeration; SRD-03 was found only because its loader happened to be named `_old_sequence`. The AST duplicate scan matched normalised whole-function bodies, so a copied *fragment* inside a larger differing function is found only where a reviewer read both sides (SRD-05, SRD-09, SRD-11, SRD-24 and SRD-25 were all found that way, so the fragment class is represented but not exhaustively swept).

---

## Summary

| Severity | Count | IDs |
|---|---|---|
| P0 | 0 | -- |
| P1 | 3 | SRD-01, SRD-02, SRD-03 |
| P2 | 11 | SRD-04 .. SRD-14 |
| P3 | 16 | SRD-15 .. SRD-30 |
| **Total** | **30** | |

By shape: 12 findings are a test comparing two local copies with production never invoked (SRD-03, 05, 07, 08, 09, 10, 11, 14, 18, 23, 24, 25); 15 are a frozen copy that missed a later production fix (SRD-01, 02, 04, 06, 12, 16, 17, 19, 20, 21, 22, 26, 27, 28, 29 -- SRD-01 and SRD-06 being production-to-production rather than test-to-production); 2 are latent duplication with no drift yet (SRD-15, SRD-30); 1 is a test module with no asserts over kernels with no production consumer (SRD-13).

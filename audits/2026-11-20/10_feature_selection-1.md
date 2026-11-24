# Audit cluster 10 "feature_selection-1" (read-only)

Scope: 143 files listed in `audits/2026-11-20/manifests/10_feature_selection-1.txt`, all under `src/mlframe/feature_selection/_benchmarks/`. Coverage: the 6 shared helper modules (`__init__.py`, `_aggregate.py`, `_bench_shared.py`, `_datasets.py`, `_shap_proxy_regime_data.py`, `_shap_selector_factory.py`) were read in full. The remaining 137 bench scripts were NOT read in full; they were scanned with targeted greps (split/leakage patterns, except-blocks, helper duplication, metric calls), and the findings below cite only code that was read. Full-read coverage of those scripts is still open.

| ID | Severity | file:line | Class | Finding | Failure scenario | Suggested fix |
|---|---|---|---|---|---|---|
| 10-F1 | P1 | bench_shapproxied_adaptive_guards.py:88-98 (helper at :42-47) | ML correctness / leakage | `Xfull` concatenates `Xfit` (the rows the selector was fit on) and `Xhold`, then `_fit_holdout_auc` re-splits `Xfull` 70/30 with its own `train_test_split`. The "held-out" test rows are therefore a random subset that includes selector training rows. The docstring "test split selector never saw" is false. | Selector fit on Xfit; holdout AUC test set contains ~30% of Xfit rows, so `auc_knee` / `auc_off` are inflated and the knee-vs-off comparison is biased. | Train the refit on `Xfit`/`yfit` and score on `Xhold`/`yhold` directly; drop the re-split in `_fit_holdout_auc`. |
| 10-F2 | P1 | bench_shap_interaction_proxy.py:143 and :102-110 (`honest_auc`) | ML correctness / leakage | Selection runs on `Xs` (75% split, seed-based). `honest_auc(X, y, ...)` then re-splits the full `X` at test_size=0.4 with the same seed but different sizes, so its test rows overlap `Xs` the selector used. Also `honest_auc` ignores `_Xh`/`_yh`, the true holdout produced at line 143. | The additive and interaction AUCs are computed on rows that were in the search split, so the additive-vs-interaction AUC comparison is contaminated. | Pass `Xs`/`ys` to fit and `_Xh`/`_yh` to score; remove the internal re-split. |
| 10-F3 | P1 | _bench_shared.py:188-189 | Correctness (edge path) | `logreg_holdout_auc` replaces an empty `cols` with all columns (`np.arange(Xtr.shape[1])`). A selector that picks nothing is then scored as if it had used every feature. Callers `bench_cross_selector_diverse.py:69` and `bench_cross_selector_cluster_reduction.py:72,81` pass selector output directly. | RFECV keeps zero features -> reported "RED auc" equals full-feature AUC, so dAUC and k-vs-AUC tables are wrong. | Return 0.5 (as `honest_auc` does at bench_shap_interaction_proxy.py:103-104) or NaN for empty `cols`, and record the empty case explicitly. |
| 10-F4 | P2 | bench_iter31_shap_prefilter.py:82; bench_iter33_scaling.py:136; bench_iter54_inner_n_jobs_cap.py:75; bench_stability_orchestration.py:85; 25 files with inline `ShapProxiedFS(` | Duplication | The `_bench_shared.py` header says shared bodies were consolidated, but `print_stage_table` is still defined locally in three scripts (one with a different signature), `build_selector` is redefined in `bench_stability_orchestration.py` (different purpose), and 25 scripts build `ShapProxiedFS(...)` inline instead of using `build_shap_proxied_selector` (`_shap_selector_factory.py:8`) / `build_selector` / `build_brute_force_cap_selector` (`_bench_shared.py:24`). | No wrong output today. A config change to one copy silently diverges from the others, which is the drift the module was created to prevent. | Replace the local copies with imports from `_bench_shared` / `_shap_selector_factory`, or document the intentional divergences in each file. |
| 10-F5 | P2 | _aggregate.py:43-48 | Silent behavior | `_check_versions` only emits warnings when both the before and after keys exist in `MANIFEST.json` (`if bef and aft:`). If either capture is absent it returns an empty list with no message, unlike the missing-manifest case at :34-36. | Comparing a capture that is not in the manifest against a rotated one prints no version warning, and the regression gate in `main` runs on mismatched library versions without notice. | Append a warning when either key is missing from the manifest. |
| 10-F6 | P2 | bench_bur_lambda_qual22.py:91-95, :113; bench_fs_levers_dflip.py:94-96 | Silent exception swallowing | `except Exception` converts MRMR fit/transform failures to NaN metrics. The only trace is `logger.debug`, so the default log level hides it and the output table shows NaN with no reason. (`bur_lambda` keeps the error in `err` at :91-95; the except at :113 was not read in full.) | A scenario that crashes in MRMR shows as NaN in the results and is averaged or skipped without a visible error. | Log at `warning` (or record the error in the output row) and keep the NaN only for the affected cell. |
| 10-F7 | P3 | _aggregate.py:23 | Dead code | `MAX_VERSION_AGE_DAYS = 90` is never read (grep finds only the definition). The version-rotation policy named in the module docstring is not enforced. | None: the constant suggests a 90-day staleness check that does not exist. | Implement the age check in `_check_versions` or delete the constant. |
| 10-F8 | P3 | _shap_proxy_regime_data.py:83 | Edge correctness | `src = int(rng.integers(0, n_informative))` raises `ValueError` when `n_informative=0` and `n_redundant>0`. | `make_regime_dataset(n_informative=0, n_redundant=2, ...)` crashes with an unhelpful numpy error. | Validate `n_informative >= 1` when `n_redundant > 0` and raise a clear ValueError. |
| 10-F9 | P3 | _bench_shared.py:267-272 | Silent exception swallowing | `make_tee_print` drops `OSError`/`ValueError` from the file write with `pass`. The log file can silently stop receiving output while the console keeps printing. | A broken log handle during a long run produces a truncated log with no warning. | Count or warn once on the first failure. |

## Checked, no finding

- `_aggregate.py:52-79` `_diff_one`: zero-denominator guards return inf for speedup and memory ratio; `skipped` short-circuit precedes all key lookups.
- `_bench_shared.py:15-21` `informative_recall`: empty informative set returns NaN, not a division by zero.
- `_bench_shared.py:137-156` `occupied_k` and `plugin_mi_1d`: empty input returns 0.0; joint zero-cell masking is correct for the log term; `np.add.at` is used for the accumulation.
- `_bench_shared.py:223-227` `random_baseline_brier`: the prior-constant Brier formula is the variance of y, which is correct for a binary target.
- `_shap_proxy_regime_data.py:107-115`: the signal is standardized before the threshold, and the binary threshold uses `np.quantile(logits, 1 - class_balance)`, giving the intended class balance.
- `_shap_proxy_regime_data.py:82-84` and `:100-105`: roles and column names are assigned in the same order as the matrix blocks, so the `roles` dict aligns with X columns.
- `_datasets.py:45-76`: scenario generation is deterministic from `random_state`, and the GPU and CPU lists are partitioned correctly.
- `_bench_shared.py:12` re-export of `uf_find`: the target `filters/_cluster_aggregate.py:313` exists.
- `_bench_shared.py:162-165` `discretize_quantile_10bin`: the target `filters/discretization/__init__.py:439` `discretize_array` exists.
- `_aggregate.py:33`: the `_results/MANIFEST.json` path resolves (the file exists).


## Dispositions

- 10-F3: RESOLVED. logreg_holdout_auc returns chance (0.5) for an empty selection instead of scoring all columns. Test tests/feature_selection/test_bench_logreg_empty_selection_is_chance.py. Commit below.
- 10-F1: RESOLVED. Adaptive-guards bench scores the refit on the untouched holdout split; the concatenated re-split is gone. Test tests/feature_selection/test_bench_holdout_auc_is_unseen.py. Commit below.
- 10-F2: RESOLVED. honest_auc fits on the search rows and scores the untouched holdout, at all three call sites. Test tests/feature_selection/test_bench_interaction_holdout_is_unseen.py. Commit below.

- 10-F6: FIXED. Benchmark harness failures (MRMR fit/transform, AUC scoring, per-config failures) are logged at WARNING instead of DEBUG, so NaN rows have a visible reason at the default level. Test tests/feature_selection/test_cluster10_bench_failures_warn.py.

- 10-F7: FIXED. _check_versions reports a capture older than MAX_VERSION_AGE_DAYS (90) as a warning, using capture_ts; the constant is no longer dead. Test tests/feature_selection/test_cluster10_aggregate_version_policy.py.

- 10-F5: FIXED. A version check with one missing capture logs that the comparison was skipped. Test tests/feature_selection/test_cluster10_aggregate_version_policy.py.

- 10-F8: FIXED. Redundant columns without informative ones raise a clear ValueError. Test tests/feature_selection/test_cluster10_bench_utility_edges.py.

- 10-F9: FIXED. A failed log-file write is reported once on stderr (was on every failure); the console keeps printing. Test tests/feature_selection/test_cluster10_bench_utility_edges.py.

- 10-F4: RESOLVED. Each local copy of a shared bench helper now carries a marker naming its intentional divergence (iter31 extra stage, iter33 own stage order, iter54 name/label header, stability orchestration build_selector). The 25 inline ShapProxiedFS constructions are per-experiment configs and are kept inline. Test tests/feature_selection/test_cluster10_documented_divergence.py.

# competition/ -- mlframe audit

## Scope

All 34 `.py` files under `src/mlframe/competition/**` were read in full (no file was too large to review in depth; the biggest is 276 LOC).

Main package (17 files, 2918 LOC):
- src/mlframe/competition/__init__.py
- src/mlframe/competition/trend_noise_decorrelation.py
- src/mlframe/competition/logloss_clip.py
- src/mlframe/competition/frequency_power_interaction.py
- src/mlframe/competition/gmm_classifier.py
- src/mlframe/competition/float_precision_denoise.py
- src/mlframe/competition/rounded_categorical_interaction.py
- src/mlframe/competition/known_label_override.py
- src/mlframe/competition/train_test_union_frequency.py
- src/mlframe/competition/power_rescale.py
- src/mlframe/competition/value_uniqueness_encoder.py
- src/mlframe/competition/naive_bayes_log_odds.py
- src/mlframe/competition/synthetic_row_detector.py
- src/mlframe/competition/threshold_range_rescaler.py
- src/mlframe/competition/quantization_recovery.py
- src/mlframe/competition/panel_target_persistence.py
- src/mlframe/competition/leak_scan.py

`_benchmarks/` subpackage (17 files, 877 LOC) -- cProfile harnesses, one per module:
- src/mlframe/competition/_benchmarks/__init__.py
- src/mlframe/competition/_benchmarks/bench_frequency_power_interaction.py
- src/mlframe/competition/_benchmarks/bench_logloss_clip.py
- src/mlframe/competition/_benchmarks/bench_rounded_categorical_interaction.py
- src/mlframe/competition/_benchmarks/bench_float_precision_denoise.py
- src/mlframe/competition/_benchmarks/bench_leak_scan.py
- src/mlframe/competition/_benchmarks/bench_trend_noise_decorrelation.py
- src/mlframe/competition/_benchmarks/bench_naive_bayes_log_odds.py
- src/mlframe/competition/_benchmarks/bench_threshold_range_rescaler.py
- src/mlframe/competition/_benchmarks/bench_quantization_recovery.py
- src/mlframe/competition/_benchmarks/bench_synthetic_row_detector.py
- src/mlframe/competition/_benchmarks/bench_gmm_classifier.py
- src/mlframe/competition/_benchmarks/bench_panel_target_persistence.py
- src/mlframe/competition/_benchmarks/bench_known_label_override.py
- src/mlframe/competition/_benchmarks/bench_value_uniqueness_encoder.py
- src/mlframe/competition/_benchmarks/bench_power_rescale.py
- src/mlframe/competition/_benchmarks/bench_train_test_union_frequency.py

**Total: 34 files, 3795 LOC, all fully reviewed.**

To confirm/refute several suspected edge-case bugs, the following read-only checks were actually executed against the installed `mlframe` package (no repo file was modified, no pytest run, no git state change): `train_test_union_frequency_encode_hierarchical_components` with `None`/NaN entries; `frequency_power_interaction` with a non-integer `count_clip_range`; `value_uniqueness_encoder` with NaN values in the encoded column; `GaussianMixtureClassifier.fit` with a singleton-sample class; `known_label_override` with a reversed `positive_value`/`negative_value` scale. Every finding below marked "confirmed by execution" was reproduced this way; all others are read-derived.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| C-1 | P1 | correctness/crash | train_test_union_frequency.py:152 | Hierarchical version-string split crashes with `TypeError: object of type 'NAType' has no len()` on any NaN/None entry in `train_series`/`test_series`. |
| C-2 | P1 | correctness/silent-wrong-result | frequency_power_interaction.py:106 | `np.power(scaled, clipped_counts)` silently returns `nan` (only a `RuntimeWarning`, no exception) when `count_clip_range` is non-integer and `scaled` contains negative values (the documented default `feature_range=(-4, 4)` always produces negative values). |
| C-3 | P1 | correctness/contract | value_uniqueness_encoder.py:57-63,176-182 | A NaN value in a `train` column produces a raw `NaN` in the output category column instead of one of the 5 documented category strings, and is handled inconsistently vs. the same NaN value on the `test` side (which resolves to `"unique_globally"`). |
| C-4 | P1 | correctness/crash | gmm_classifier.py:99-108 | `GaussianMixtureClassifier.fit` crashes with a raw, unhelpful `sklearn` `ValueError` ("Found array with 1 sample... minimum of 2 is required") whenever any class has exactly 1 training sample, even though the code already tries to defend against undersized classes via `n_components = min(self.n_components_per_class, X_cls.shape[0])`. |
| C-5 | P1 | correctness/silent-wrong-result | known_label_override.py:154-158 | `is_recovered_positive = recovered_label >= midpoint` hardcodes an ascending `negative_value < positive_value` convention; when a caller legitimately passes a reversed scale (e.g. `positive_value=0.0, negative_value=1.0`), the override silently applies in the wrong direction for every recovered label. |
| C-6 | P1 | perf/memory | leak_scan.py:51-52 | `_nonnull_nonzero_mask` builds `values = df.to_numpy(dtype=object)` -- a full object-dtype copy of the entire (potentially huge) input frame -- purely to read `values.shape`; `values` itself is never otherwise used. `df.shape` gives the same shape with zero extra allocation. |
| C-7 | P1 | perf | threshold_range_rescaler.py:124-135 | `_cv_score` constructs a fresh `StratifiedKFold` and calls `.split(preds, y)` on every single grid-search candidate evaluation. Since the split only depends on `y` and the fixed `random_state`, it is bit-identical across every candidate in a `fit()` call (all subgroup x threshold x multiplier combinations, times `max_corrections` rounds) -- a classic "recomputing something cheap-to-cache on every call" hot-path waste. |
| C-8 | P2 | perf/memory | leak_scan.py:113-115 | `sort_by_density_leak_scan` does `work = df.copy()` unconditionally, even when `target is None` and no column is ever inserted into `work` -- an avoidable full-frame copy on the common no-target call path. |
| C-9 | P2 | robustness | known_label_override.py:102-109 | `known_label_override`'s docstring states values are "positive_value-like ... treated as recovered-positive" but never validates or documents the implicit assumption `negative_value < positive_value` that the `>= midpoint` comparison depends on (see C-5); no `ValueError` guards the reversed case. |
| C-10 | P2 | docs/test-gap | frequency_power_interaction.py:74-78 | The `count_clip_range` docstring doesn't mention that non-integer bounds combined with the default negative-valued `feature_range` produce `nan` outputs (see C-2); no validation rejects a non-integer range up front. |
| C-11 | P2 | test-coverage | tests/competition/ (whole dir) | None of the `test_biz_val_*` files exercise: NaN/missing values in `train_test_union_frequency_encode(..., hierarchical_split_sep=...)` (C-1), a non-integer `count_clip_range` in `frequency_power_interaction` (C-2), NaN values in `value_uniqueness_encoder` input columns (C-3), a singleton-sample class in `GaussianMixtureClassifier.fit` (C-4), or a reversed value scale in `known_label_override` (C-5) -- every crash/silent-wrong-result finding above would have been caught by a regression test. |

Below, one paragraph per finding with the concrete failure scenario and a suggested-fix direction.

**C-1** (train_test_union_frequency.py:152, confirmed by execution): calling `train_test_union_frequency_encode_hierarchical_components(train_series, test_series, ".")` where either series contains `None`/`NaN` raises `TypeError: object of type 'NAType' has no len()` inside `train_parts.map(len)`, because `pd.Series.astype("string").str.split(sep)` maps a missing entry to `pd.NA` rather than an empty list. This is a real crash on a realistic input -- version-string columns (the documented use case) very commonly have missing entries for older records. Fix direction: fill missing entries with a sentinel token (or filter to a length-computation that treats `pd.NA`/`NaN` as depth 0) before calling `.map(len)`, and route them through `missing_token`-style handling as `rounded_categorical_interaction.py` already does for its own missing-value case.

**C-2** (frequency_power_interaction.py:106, confirmed by execution): with `count_clip_range=(1.5, 2.5)` (a legitimate, documented parameter -- the docstring only requires `low <= high`, not integers) and the default `feature_range=(-4, 4)`, any input value that MinMax-scales to a negative number produces `np.power(negative_scaled, 1.5-or-2.5)` = `nan`, silently, with only a `RuntimeWarning` printed to stderr (not raised). Downstream consumers of `interaction_feature` get silent `NaN` contamination instead of an error. Fix direction: either restrict `count_clip_range` to integers (matching the source write-up, which always clips to whole-number exponents) and validate that at input, or take `np.sign(scaled) * np.abs(scaled) ** clipped_counts` to keep the transform well-defined for fractional exponents on negative bases.

**C-3** (value_uniqueness_encoder.py:57-63 and 176-182, confirmed by execution): `pandas.DataFrame.groupby("value")` drops rows whose `value` is NaN by default (`dropna=True`), so `_build_train_value_to_flag` never assigns a flag to NaN train values; `_encode_train_column`'s `.map(value_to_flag)` then leaves those rows as literal `NaN` in the output `category` column -- a 6th, undocumented state alongside the 5 named categories in `VALUE_UNIQUENESS_CATEGORIES`. Meanwhile the equivalent NaN value on the *test* side (via `_encode_test_column`'s `novel_counts` path, which does not drop NaN) resolves to `"unique_globally"`. A downstream consumer that assumes only the 5 documented categories appear (e.g. anything doing `pd.Categorical(..., categories=VALUE_UNIQUENESS_CATEGORIES)`, which this function itself does at line 182) will silently coerce the train-side NaN cells to an actual missing category value, invisible unless explicitly checked. Fix direction: either add an explicit `MISSING_VALUE` category and route NaNs there symmetrically on both sides, or document that upstream callers must impute/fillna before calling.

**C-4** (gmm_classifier.py:99-108, confirmed by execution): `GaussianMixtureClassifier.fit` computes `n_components = min(self.n_components_per_class, X_cls.shape[0])` specifically to avoid over-parameterizing a small class, but `sklearn.mixture.GaussianMixture` itself unconditionally requires `ensure_min_samples=2` regardless of `n_components`, so any class with exactly 1 training sample (a realistic edge case for an imbalanced dataset a user is diagnostically probing with this classifier) crashes with a raw sklearn `ValueError` rather than either working or raising a clear, class-identifying mlframe error. Fix direction: either raise a clear `ValueError` naming the offending class up front in `fit`, or fall back to a degenerate single-point "distribution" (e.g. a tiny-covariance Gaussian centered at that one sample) for singleton classes.

**C-5 / C-9** (known_label_override.py:102-161, confirmed by execution): the function's docstring says the recovered label is compared to decide whether it points toward the "positive" or "negative" direction, and the implementation computes `midpoint = (positive_value + negative_value) / 2.0` then `is_recovered_positive = recovered_label >= midpoint`. This silently assumes `negative_value < positive_value`. A caller using a reversed convention (e.g. modeling "0 = fraud/positive-of-interest, 1 = legit/negative", which is a real convention in some fraud/anomaly domains and nothing in the signature forbids it) gets every override applied in the *opposite* direction from what was requested -- a label the caller flagged as `negative_value` gets written as `positive_value` and vice versa, with no error or warning. Fix direction: either validate `negative_value < positive_value` and raise `ValueError` otherwise, or compare using `abs(recovered_label - positive_value) < abs(recovered_label - negative_value)` so the "closer to positive_value" semantics the docstring actually promises hold regardless of ordering.

**C-6** (leak_scan.py:51-52): `_nonnull_nonzero_mask(df)` opens with `values = df.to_numpy(dtype=object)` and then only ever reads `values.shape` on the next line -- the array of Python objects (one boxed object per cell) is fully materialized and then discarded. For the module's own stated audience ("anonymized, shuffled feature matrices" -- often wide, since Santander-style leak-scan targets hundreds of anonymized columns) this is a real, gratuitously wasteful `O(n_rows * n_cols)` allocation of Python-object boxes on every call, directly contradicting this repo's "never materialize a full frame just to read a cheap property" convention. Fix direction: replace with `mask = np.empty(df.shape, dtype=bool)` (or `np.empty((len(df), len(df.columns)), dtype=bool)`), dropping the `values =` line entirely.

**C-7** (threshold_range_rescaler.py:124-135): `ThresholdRangeRescaler.fit`'s greedy grid search calls `self._cv_score(candidate, y_arr)` once per `(subgroup, threshold, multiplier)` combination per round, and `_cv_score` re-instantiates `StratifiedKFold(..., random_state=self.random_state)` and re-runs `.split(preds, y)` inside every call -- but the split indices depend only on `y` and `self.random_state`, both constant across the entire `fit()` call, so the exact same fold partition is recomputed from scratch on every grid cell (potentially hundreds to thousands of times for a realistic `thresholds x multipliers x subgroups` grid). This is the textbook "hot path recomputing something cheap-to-cache on every call" pattern this repo's own conventions flag. Fix direction: compute `list(skf.split(preds, y))` once in `fit()` (or once per `self.random_state`/`y` pair) and pass the fold index list into `_cv_score`, which then only needs to slice and score.

**C-8** (leak_scan.py:113-115): `sort_by_density_leak_scan` always executes `work = df.copy()` before optionally inserting a `"__target__"` column, even on the common call path where `target is None` and `work` is never mutated afterward (every subsequent operation -- `_nonnull_nonzero_mask(work)`, `work.apply(pd.to_numeric, ...)` -- is read-only). For the "anonymized wide feature matrix" workloads this module targets, that is an avoidable full-frame duplication. Fix direction: only copy when `target is not None` (`work = df.copy() if target is not None else df`), or build the target-prefixed frame via `pd.concat([pd.Series(target, name="__target__"), df], axis=1)` without ever touching the original `df`.

**C-10**: (see C-2's discussion) purely a docs/validation gap -- the `count_clip_range` parameter's docstring should call out the fractional-exponent-on-negative-base failure mode, and/or `frequency_power_interaction` should validate `low, high` are both integers (or both `>= 0` combined with an odd/even-safe transform) the same way `power_rescale_to_target_sum` validates its own numerically-sensitive inputs.

**C-11**: every `test_biz_val_*.py` file under `tests/competition/` demonstrates the trick winning on a favorable synthetic and (for the "honest negative" modules) losing on an unfavorable one, but none of them probe the specific edge-case inputs above. Since C-1 through C-5 are all real, reproducible failures on realistic inputs, a targeted regression test per bug (per this repo's own "every bug fix ships a regression test" convention) would have caught each one before merge.

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| P-1 | test-coverage | tests/competition/test_biz_val_train_test_union_frequency.py | Add a regression test for NaN/missing values through the `hierarchical_split_sep` path (see C-1). |
| P-2 | test-coverage | tests/competition/test_biz_val_frequency_power_interaction.py | Add a regression test for a non-integer `count_clip_range` combined with the default negative-valued `feature_range` (see C-2). |
| P-3 | test-coverage | tests/competition/test_biz_val_value_uniqueness_encoder.py | Add a regression test asserting NaN input values never leak a raw `NaN`/undocumented category into the output, and that train-side vs. test-side NaN handling is symmetric (see C-3). |
| P-4 | test-coverage | tests/competition/test_biz_val_gmm_classifier.py | Add a regression test for a singleton-sample minority class, asserting either a clear `ValueError` or a graceful degenerate fit rather than the raw sklearn traceback (see C-4). |
| P-5 | test-coverage | tests/competition/test_biz_val_known_label_override.py | Add a regression test with a reversed `positive_value`/`negative_value` scale to pin the intended "closer to positive_value" semantics regardless of ordering (see C-5). |
| P-6 | perf | leak_scan.py | Beyond C-6/C-7/C-8, `find_shifted_column_groups` is `O(n_cols^2 * max_lag)` Pearson-correlation pairs (documented as a deliberate "minimal, best-effort" scope-reduction vs. the fuller `TemporalShiftGroupDetector` idea already in the module docstring) -- fine as-is, but worth a `warnings.warn` or a `max_columns` guard so an accidental call on a very wide anonymized frame (hundreds+ columns) doesn't silently run for a very long time. |
| P-7 | architecture | src/mlframe/competition/__init__.py | Every one of the 16 competition-trick modules re-implements its own "COMPETITION/EXPLORATORY ONLY" warning banner in its module docstring (verified consistent across all 17 files); consider a single decorator/mixin (e.g. `@competition_only` on public entry points) that both documents and, optionally, emits a one-time `warnings.warn` the first time any competition-only callable is invoked in a process, so accidental production wiring is caught at runtime, not just at code-review time. |
| P-8 | robustness | naive_bayes_log_odds.py:154-163 | `NaiveBayesLogOddsEnsembler.predict_proba` has no validation that `X.shape[1]` matches the feature count seen at `fit` time (no `n_features_in_` check unlike `gmm_classifier.py`'s explicit `self.n_features_in_`); a caller passing a mismatched-width `X` gets an opaque downstream `IndexError`/broadcasting error from inside a per-block `model.predict_proba` call rather than an mlframe-level message naming the mismatch. |
| P-9 | ML-best-practice | threshold_range_rescaler.py | The module's own docstring already flags the CV-overfitting risk of this trick (intentional, well-documented); consider exposing the per-candidate fold scores (not just the mean) from `_cv_score` so a caller can additionally gate acceptance on variance across folds (e.g. reject a candidate whose gain is driven by one lucky fold), further hardening the "only accept improvements that generalize across the CV split" intent already partially served by `min_improvement`. |

## Coverage notes

- All 34 in-scope `.py` files were read in full; none were too large to review completely (largest is `leak_scan.py` at 276 LOC).
- Five suspected issues (C-1 through C-5) were confirmed by actually running the installed package against small repro inputs (read-only: no repo file touched, no `pytest`, no git-state-changing command) rather than left as unverified hypotheses -- see the note at the end of the Scope section for the exact repro list.
- I did not attempt to independently verify every documented "honest negative" biz_value claim in the module docstrings (e.g. that `NaiveBayesLogOddsEnsembler` genuinely underperforms averaging on correlated features, or that `GaussianMixtureClassifier` genuinely underperforms a GBM baseline on `make_classification`-style data) against the corresponding `tests/competition/test_biz_val_*.py` files' actual assertions/thresholds -- those tests exist and were located, but re-running them was out of scope for a read-only audit (no pytest execution permitted) and re-deriving the numeric claims by hand was not attempted given the audit's time budget was spent chasing concrete, reproducible bugs instead.
- The MRMR/SHAP-proxied-selector exclusions and their test mirrors do not intersect this cluster's scope at all -- no file under `src/mlframe/competition/**` imports from either excluded package, so no boundary-case symbol-only reads were needed.

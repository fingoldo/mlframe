# training/targets -- mlframe audit

## Scope

All 16 files under `src/mlframe/training/targets/` were read in full (not sampled/skimmed):

- `__init__.py` (60 LOC)
- `_residual_normality_tests.py` (16 LOC)
- `_target_distribution_analyzer_modes.py` (126 LOC)
- `_target_temporal_plot.py` (134 LOC)
- `_target_temporal_audit_coerce.py` (143 LOC)
- `_target_temporal_audit_from_agg.py` (170 LOC)
- `_target_distribution_analyzer_stats.py` (183 LOC)
- `_ttr_eval_set_scaling.py` (187 LOC)
- `_target_distribution_analyzer.py` (192 LOC)
- `_target_temporal_audit_aggregate.py` (245 LOC)
- `_train_eval_select_target.py` (296 LOC)
- `_target_distribution_analyzer_target_fn.py` (307 LOC)
- `_target_temporal_changepoint.py` (413 LOC)
- `_target_distribution_analyzer_features.py` (450 LOC)
- `target_temporal_audit.py` (577 LOC)
- `regression_residual_audit.py` (776 LOC)

Total: **16 files, 4275 LOC**, all reviewed in full depth (no file was too large to read completely).

Two findings below (the polars/pandas target-rate parity bug and the string-labeled-target crash) were additionally verified empirically with standalone read-only Python reproductions (no repo files were modified; only `analyze_target_distribution` / `_aggregate_by_time_polars` / `_aggregate_by_time_pandas` were imported and called on synthetic in-memory data).

I also read (for caller-context only, not for findings) `src/mlframe/training/core/_main_train_suite_target_distribution.py` to confirm how `analyze_target_distribution` / `analyze_feature_distribution` are wired into the real training suite, and `src/mlframe/training/trainer.py:804-814` to confirm the one production call site of `_TTRWithEvalSetScaling`.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P0 | correctness / NaN-handling / GPU-CPU(backend)-parity | `_target_temporal_audit_aggregate.py:104-109` | Polars binary-classification target-rate aggregation counts null target rows as negative class, deflating `target_rate`; the identical bug was already found and fixed on the pandas twin but never mirrored to the polars path. |
| F2 | P1 | correctness / edge case | `_target_distribution_analyzer_target_fn.py:82-83` | `analyze_target_distribution` unconditionally casts `y` to `float64` before ever branching on `target_type`, so any classification target with string/object class labels crashes with `ValueError`, regardless of an explicit `target_type="classification"`. |
| F3 | P1 | docs/API contract mismatch / ML best practice | `_target_distribution_analyzer_features.py:251,414-440`; docstring `_target_distribution_analyzer.py:155-159` | `analyze_feature_distribution`'s documented "per-class AUC > 0.99 for classification" leakage detector was never implemented; the `target_type` parameter is dead (never read in the function body), and the only leakage check (Pearson `corrwith`) runs unconditionally against raw numeric `y`, which is statistically inappropriate for multiclass integer-coded targets. |
| F4 | P1 | silent-failure / robustness gap | `_ttr_eval_set_scaling.py:109-110` (vs. `44-64`) | The predict-time defensive extrapolation clip (`_y_train_clip_low_/_high_`, the class's whole documented purpose) is unconditionally computed in `fit()` but is skipped entirely in `predict()` whenever `self.transformer_ is None`, even though the clip bounds don't depend on a transformer being configured. |
| F5 | P2 | silent-failure pattern | `_ttr_eval_set_scaling.py:44-64` | The `except Exception:` guard around the y-train clip-bound computation in `fit()` silently disables the safety clip with no `logger.debug`/`warning` call at all, unlike the two other `except` blocks in the same file which do log. |
| F6 | P2 | correctness (cosmetic/diagnostic) | `_target_distribution_analyzer_target_fn.py:297` | `rare_classes = [int(c) for c, k in zip(classes, counts) ...]` truncates float class labels to `int`, so e.g. classes `1.4` and `1.6` would both display as rare class `1` in the pathology string (diagnostics text only, not a control-flow bug). |
| F7 | P2 | encoding bug (mojibake) | `target_temporal_audit.py` (e.g. lines 1,10,24,37,68,73,309,332,409-410), `_target_temporal_audit_from_agg.py` (lines 91,121,135), `_target_temporal_changepoint.py` (e.g. lines 10,17,24,28,33,46,94,100,176,228,252) | The multiplication sign `x` and an em-dash were corrupted into a `Г—`/`вЂ”`-style mojibake sequence (UTF-8 bytes misread) throughout docstrings and comments in three files; two of the corrupted strings (`_target_temporal_audit_from_agg.py:91,121,135`) are actual `logger.warning`/`warnings.append` text shown to operators at runtime, not just comments. |
| F8 | P2 | dead code / stale comment | `_target_temporal_plot.py:132-134`, `_target_temporal_audit_from_agg.py:168-170` | Both files end with an orphaned section-header comment (`# Human-readable text report` / `# Plotting`) left over from the "Wave 106" monolith split, with no matching code below it in that file -- the actual content now lives in a sibling module. Misleading to a reader navigating by these headers. |
| F9 | P2 | robustness / index-type ambiguity | `_train_eval_select_target.py:94-106` | `_select()`'s dispatch falls through to plain `target_arr[idx]` whenever `idx` has no `.dtype` attribute (e.g. a plain Python `list`); for a `pd.Series` target with a non-default index this is pandas *label*-based indexing, not positional -- silently wrong row selection if a caller ever passes a plain-list `train_idx`/`val_idx`/`test_idx`. All current internal callers appear to pass numpy arrays, so this is latent rather than confirmed-triggered. |
| F10 | P2 | perf/methodology nuance | `_target_distribution_analyzer_features.py:370-376` | The redundant-feature-pair correlation pass uses a fixed-stride systematic sample (`df[::stride]`) rather than a random sample when downsampling >100k rows; systematic striding can alias with periodic structure in the data (e.g. weekly/seasonal patterns), biasing the correlation estimate in a way random sampling would not. |
| F11 | P2 | efficiency (minor) | `_ttr_eval_set_scaling.py:65-67` vs `89-90` | `fit()` explicitly fits `self.transformer_` itself (to scale `eval_set`), then unconditionally calls `super().fit(...)`, which refits the *same* `transformer_` again internally -- a redundant second fit. Harmless for deterministic transformers like `StandardScaler` but wasteful, and would silently double-apply any stochastic transformer's randomness inconsistently between the two fits. |

### F1 -- polars/pandas target-rate parity bug on null target values (P0)

`_polars_rate_expr` (used by both `_aggregate_by_time_polars` and `_aggregate_by_time_polars_multi`, the code paths the module docstrings describe as "strongly preferred" and "the polars fastpath ... for >1M rows") computes the per-bin binary-classification rate as `(pl.col(target_col).fill_null(0) > 0).cast(pl.Float64).mean()`. This is exactly the pattern the sibling `_aggregate_by_time_pandas` function documents (lines 214-219) as a **previously-fixed bug**: "The prior `fillna(0) > 0` counted NaN rows as negatives, deflating the reported positive rate per bin." That fix was applied only to the pandas function; the polars expression builder was never updated to match.

Empirical reproduction (read-only, no repo mutation) with a target column containing genuine polars nulls (not NaN -- NaN happens to compare `> 0` as `True` in polars, so it does not trigger the bug; true nulls do):
```
polars target_rate: [0.645, 0.690, 0.667]
pandas  target_rate: [1.0, 1.0, 1.0]
```
Both aggregations ran on the exact same underlying data (all non-null values were `1.0`); the true rate is `1.0` per bin, which the pandas path reports correctly and the polars path silently deflates by roughly the null fraction. Because `audit_target_over_time`/`audit_targets_over_time` route through the polars path whenever the caller passes a `pl.DataFrame` (the documented, preferred, >1M-row path), and `"binary_classification"` is the default inferred target type in `audit_targets_over_time` for a bare string spec, this silently corrupts the primary temporal-drift-detection signal for any classification target with missing values -- exactly the "silently wrong result" bug class this audit was asked to hunt for. Suggested fix: mirror the pandas fix in `_polars_rate_expr`, e.g. `pl.col(target_col).drop_nulls()`-style mean, or explicitly compute `(pl.col(target_col).fill_null(0) > 0).cast(pl.Float64).sum() / pl.col(target_col).is_not_null().sum()` so the denominator matches the pandas semantics, and add a regression test pinning polars/pandas parity on a null-containing target.

### F2 -- `analyze_target_distribution` crashes on string-labeled classification targets (P1)

`y = np.asarray(y).reshape(-1); if y.dtype.kind != "f": y_for_stats = y.astype(np.float64)` runs before `target_type` is even inspected. Empirically confirmed:
```
y = np.array(['yes','no','yes','no']*20)
analyze_target_distribution(y, target_type='classification', has_time_axis=False)
-> ValueError: could not convert string to float: np.str_('yes')
```
This also means `target_type="auto"` never gets a chance to work either, since `_classify_target_type` (which DOES correctly fall through to `"regression"` for non-numeric dtypes, `_target_distribution_analyzer_modes.py:117-126`) is only consulted *after* the crashing cast. String-labeled classification targets (a completely standard scikit-learn pattern) can never be analyzed by this function. In the one production call site (`_main_train_suite_target_distribution.py:150-417`) the whole block is wrapped in a blanket `except Exception` that WARN-logs and continues, so this degrades to "the mini-HPT distribution analyzer silently never runs" for such targets rather than crashing the suite -- still a real capability gap with no test covering it (all `analyze_target_distribution` classification tests in `tests/training/targets/test_target_distribution_analyzer.py` use `np.int32`/integer target arrays only). Suggested fix: determine `ttype` from the raw `y` (or an explicit `target_type` override) before doing any numeric cast, and only cast to float on the regression path.

### F3 -- feature-leakage detector: dead `target_type` param + undocumented Pearson-only implementation (P1)

The module docstring (`_target_distribution_analyzer.py:155-159`) documents the leakage detector as "`|Pearson(x, y)| > 0.99` for regression OR per-class AUC > 0.99 for classification". `analyze_feature_distribution` accepts a `target_type` parameter (`_target_distribution_analyzer_features.py:251`) but **never reads it anywhere in the function body** (confirmed via grep -- the only occurrence of the string `target_type` in the file is the signature line). The leakage block (`414-440`) always uses `df[candidate_numeric].corrwith(y_series)` whenever `y_arr.dtype.kind in ("f","i","u","b")`, i.e. for ANY numeric-encoded target including multiclass integer-coded labels (`TargetTypes.MULTICLASS_CLASSIFICATION`, confirmed reachable via `_main_train_suite_target_distribution.py:355-358` which passes the raw picked target straight through). Pearson correlation against unordered integer class codes (e.g. 0/1/2 for three unrelated classes) is not a meaningful leakage signal and can both miss real leakage and flag spurious "leakage" depending on how class codes happen to be numbered. Binary targets are a partial exception (point-biserial correlation is a legitimate proxy), but the multiclass case is silently wrong. Suggested fix: either implement the documented per-class-AUC path when `target_type == "classification"` and `n_classes > 2`, or update the docstring to describe the actual (Pearson-only) behavior and have the function raise/skip explicitly for multiclass rather than silently running an inappropriate statistic.

### F4 -- OOD-extrapolation predict-time clip silently skipped when `transformer=None` (P1)

`_TTRWithEvalSetScaling.fit()` unconditionally computes `_y_train_clip_low_`/`_y_train_clip_high_` from the train-y distribution (lines 44-64), documented as protection against the "observed in prod" catastrophic-extrapolation incident. But `predict()` opens with `if self.transformer_ is None: return super().predict(X, **predict_params)` (lines 109-110), which bypasses both the `-17 sigma` sensor AND the y-space clip entirely -- not just the sensor (which legitimately only makes sense in transformer-scaled space). The y-space clip's rationale (bounding `y_hat` to `[y_min - 3std, y_max + 3std]`) is independent of whether a transformer is configured. `tests/training/targets/test_ttr_predict_no_double_invoke.py::test_ttr_predict_passthrough_when_no_transformer` exercises this branch but only asserts the inner regressor is called once -- it does not exercise or assert anything about the clip. Currently masked in production because the sole call site (`trainer.py:814`) always passes `transformer=StandardScaler()`, but the class is a general reusable component (tested with `transformer=None` explicitly) and any future/other caller configuring it without a transformer silently loses the documented safety net. Suggested fix: apply the y-space clip regardless of `transformer_ is None`, keeping only the scaled-space sensor conditional on the transformer.

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| P1 | test-coverage | `tests/training/targets/test_temporal_audit_polars_fastpath.py` | Add a regression test asserting `_aggregate_by_time_polars`/`_aggregate_by_time_polars_multi` produce the same `target_rate` as `_aggregate_by_time_pandas` on a target column with nulls -- would have caught F1 directly and prevents recurrence when either path is touched again. |
| P2 | test-coverage | `tests/training/targets/test_target_distribution_analyzer.py`, `tests/training/test_feature_distribution_analyzer.py` | Add a string-labeled-y test for `analyze_target_distribution` and a multiclass-integer-y leakage test for `analyze_feature_distribution` to lock in intended behavior once F2/F3 are fixed. |
| P3 | hygiene | `target_temporal_audit.py`, `_target_temporal_audit_from_agg.py`, `_target_temporal_changepoint.py` | Fix the mojibake source encoding (re-save the three affected files as clean UTF-8) and add a lightweight CI grep for the `Г—`/`вЂ”`-style corruption pattern so a future copy-paste through a lossy clipboard doesn't reintroduce it silently. |
| P4 | hygiene | `_target_temporal_plot.py:132-134`, `_target_temporal_audit_from_agg.py:168-170` | Delete the two orphaned "Wave 106" section-header comments that no longer correspond to any code in their own file. |
| P5 | observability | `_target_distribution_analyzer_stats.py:109-152` (`_lag1_autocorr_grouped`) | The function's own docstring says the `skipped` count of too-small groups is "not yet wired but reserved for future observability" -- it's computed (`skipped = 0` / `skipped += 1`) and then discarded. Either wire it into the caller's diagnostics dict or drop the dead local variable. |
| P6 | robustness | `_train_eval_select_target.py:94-106` | Harden `_select()` to normalise any array-like `idx` (including plain Python lists) to a numpy array up front, removing the pandas label-vs-positional indexing ambiguity noted in F9. |

## Coverage notes

- I did not execute `pytest` (per the read-only mandate for this audit); F1 and F2 were instead independently verified via standalone read-only Python reproductions run through the Bash tool (importing only the target modules under audit and calling them on synthetic in-memory data -- no repository files were read as fixtures, written, or mutated).
- I did not trace every call site of `select_target`/`_select` (F9) across the full `mlframe.training` tree to prove whether a plain-list `train_idx`/`val_idx`/`test_idx` is ever actually passed in production; the finding is reported as a latent/plausible risk (based on the code's own type-dispatch logic), not a confirmed-triggered bug, and is flagged as such (P2, not higher).
- `regression_residual_audit.py`'s numba-accelerated moments kernels (`_audit_moments_njit_seq`/`_par`) were read in full and found well-documented and internally consistent (matching the numpy fallback path's formulas); I did not independently re-derive or numerically re-verify the fused single-pass kurtosis/skew math beyond checking it against the equivalent numpy path already present as the fallback in the same file.
- The excluded MRMR/SHAP-proxied packages and their test mirrors were not read, per the audit's explicit scope boundary; no findings in this report touch those directories.

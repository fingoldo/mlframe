# preprocessing_data
Files reviewed: 44 | LOC: ~7,300 (of ~10,700 in cluster, benchmarks excluded)

## Summary
The cluster is unusually leakage-aware in its prose -- almost every transformer ships a `fit_*`/`apply_*` pair
plus an explicit LEAKAGE WARNING -- but several of those contracts are not implemented by the code underneath
them. The highest-severity finding is not leakage at all: `ensure_no_infinity_pd` uses `np.nan_to_num` with its
default `nan=0.0`, so any float column that happens to contain one `inf` has *every* NaN in it silently
rewritten to `0.0` -- a wrong value, in a default pre-fit path, contradicting both its own docstring and its
sibling ndarray implementation two functions above. Beyond that: `auto_transform_select` plugs its scaler-fit
leak per fold but median-imputes the whole column *before* the split; `apply_features_cleaning` casts to a
different dtype than `analyse_and_clean_features` learned with (float32-vs-float64 divergence, and a hard
`IntCastingNaNError` on integer columns under the default `default_na_val=np.nan`);
`apply_gaussian_power_transform` re-derives its NaN-fill median from the *apply* frame while promising it
replays exactly what was fitted; `smoothed_target_encode_column`'s OOF path threads the full-train global mean
into every fold; and `align_feature_direction`'s hand-rolled rank AUC breaks ties arbitrarily, so the sign-flip
decision on any tied/low-cardinality column is decided by argsort order rather than by the data. `pl.Categorical`
usage is already correct (Enum-preserving with warn-once fallbacks), no `min()==max()` constant-column predicate
exists in this cluster, and the one JSON-to-key path (`_param_oracle_store._stable_json`) correctly uses
`OPT_SORT_KEYS`.

## Findings

### PREPROCESSING_DATA-1 [P0] silent-value-corruption
**File:** src/mlframe/core/helpers.py:289
**Summary:** `ensure_no_infinity_pd` calls `np.nan_to_num` without `nan=`, so it silently replaces every NaN in an inf-bearing column with `0.0`, not just the infinities its docstring promises to touch.
**Failure scenario:** a float64 feature column holding `[1.0, NaN, inf, -inf]` (NaN = genuinely missing, inf = a divide-by-zero artifact). `ensure_no_infinity_pd` flags the column (`np.isinf(arr).any()` is True) and runs `np.nan_to_num(df[col], posinf=0, neginf=0)`. Verified live on pandas 2.3.3 / numpy 2.3.5: the result is `[1., 0., 0., 0.]` -- the NaN became `0.0`. CatBoost/LightGBM/XGBoost all treat NaN as a first-class "missing" branch; turning it into a real `0.0` moves those rows to the wrong side of every split, and `0.0` is a plausible value so nothing downstream errors. Only columns containing at least one inf are affected, so the corruption is data-dependent and invisible in most runs.
**Suggested fix:** pass `nan=np.nan` explicitly, or better, mirror the sibling `ensure_no_infinity_np` (lines 179-181) exactly: `arr = s.to_numpy(); arr[np.isinf(arr)] = nans_filler`. Add a regression test asserting a NaN survives an inf-bearing column.
**Evidence:** `np.nan_to_num`'s signature is `(x, copy=True, nan=0.0, posinf=None, neginf=None)` -- `nan` defaults to `0.0`. Reproduced in-process (output `[1. 0. 0. 0.]`). The function's docstring at line 219 says only "Replace +-inf with `nans_filler` in float columns"; `ensure_no_infinity_np`'s docstring at line 175 claims it "Mirrors `ensure_no_infinity_pd`'s contract" while using the mask form that correctly leaves NaN alone -- the two implementations disagree.

**Disposition:** RESOLVED. `np.putmask` on an explicit `np.isinf` mask, so NaN is left alone.

### PREPROCESSING_DATA-2 [P0] leakage
**File:** src/mlframe/preprocessing/auto_transform_select.py:229
**Summary:** The per-column NaN median fill is computed once over the WHOLE column before the CV split, so every held-out fold's own rows contribute to the value substituted into that fold -- the exact leak `_fit_transform_fold` was written to close, left open one line above it.
**Failure scenario:** a column with 30% missing values whose missing rows are concentrated in one target regime. `finite_fill[~np.isfinite(finite_fill)] = np.nanmedian(finite_fill[np.isfinite(finite_fill)])` (line 229) fills using a median computed over train+test folds. `fold_indices` are then applied to that already-imputed array (line 251), so each fold's "held-out" score is measured on rows imputed with a statistic that saw them. The resulting cross-validated AUC/RMSE is optimistically biased and the transform ranking (`best_transform`, line 289) can pick a different winner than an honest fit-on-train imputation would. Same pattern at lines 111 and 119 (context-column correlation ranking) and 239 (multivariate context matrix).
**Suggested fix:** move the median fill inside the fold loop -- compute `np.nanmedian` over `raw[train_idx]`'s finite values per fold and apply it to both slices, the same shape `_fit_transform_fold` already uses for scaler statistics. Keep the whole-column fill only for the final replay path.
**Evidence:** `_fit_transform_fold`'s own docstring (lines 73-78) states "fitting on the FULL column (train+test) before a CV split leaks the test fold's own statistics into the 'held-out' score, optimistically biasing it"; the inline comment at lines 248-250 repeats it. The median fill at line 229 does precisely that, on the array then passed as `finite_fill` into `_fit_transform_fold` at line 251.

**Disposition:** RESOLVED. `_fill_nonfinite_from_train(x, train_idx)` moves the fill inside the fold loop.

### PREPROCESSING_DATA-3 [P1] dtype-divergence-train-vs-apply
**File:** src/mlframe/preprocessing/cleaning.py:749 (learn) vs :907 and :913 (apply)
**Summary:** `analyse_and_clean_features` casts a rare-value-merged column to `default_float_type` (float32 by default) while `apply_features_cleaning` casts the same column to the *apply frame's own* dtype -- so train and val/test end up on different dtypes, and integer columns crash outright.
**Failure scenario (silent):** a float64 continuous column with rare values merged to `default_na_val=np.nan`. The train path takes `the_type = default_float_type = np.float32` (line 750) and does `.astype(np.float32)` (line 761). `apply_features_cleaning(X_val, res)` takes `head[col].dtype.name == "float64"` (line 913) and keeps float64. The model is fit on float32-rounded values and scored on float64 values; any downstream binning/edge computation (`core.binning.fit_bin_smoother`, MRMR edges) sees a different tie structure on train vs val.
**Failure scenario (loud):** an int64 discrete column with rare values. Train becomes float32; `apply_features_cleaning` runs `df[col].replace({rare: nan}).astype("int64")` and raises `pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer` -- reproduced in-process. This is the DEFAULT configuration (`default_na_val=np.nan`), so any integer feature with a rare value breaks the apply path.
**Suggested fix:** persist the learned target dtype per column in the returned dict (e.g. `features_dtypes[col]`) next to the `features_transforms[col].update(...)` at line 754, and have `apply_features_cleaning` use that instead of `head[col].dtype.name`. The `dtypes=df.dtypes` entry already returned at line 856 is close but reflects post-mutation train dtypes only when `update_data=True`.
**Evidence:** lines 749-752 choose `default_float_type` when `col_is_numeric and pd.isnull(default_na_val)`; lines 907 and 913 in `apply_features_cleaning` unconditionally use `head[col].dtype.name` with no equivalent branch. Verified `pd.Series([1,2,3],dtype='int64').replace({2: np.nan}).astype('int64')` raises `IntCastingNaNError` on pandas 2.3.3.

**Disposition:** RESOLVED. `features_dtypes` records the learned target dtype per column and `apply_features_cleaning` replays it. Reproduced the `IntCastingNaNError` before the fix. `tests/preprocessing/test_cleaning_apply_replays_the_learned_dtype.py`.

### PREPROCESSING_DATA-4 [P1] leakage
**File:** src/mlframe/preprocessing/gaussian_power_transform_search.py:219
**Summary:** `apply_gaussian_power_transform` recomputes the NaN-fill median from the frame it is applied to and writes those filled values into the output -- so the "replay the exact fitted transform" contract is violated and the function silently imputes with apply-frame statistics.
**Failure scenario:** the search runs on `X_train` (median of column `price` = 100). `apply_gaussian_power_transform(X_val, res)` computes `np.median(finite)` over `X_val` (line 219) -- say 140 -- fills `X_val`'s NaNs with 140, transforms, and assigns the filled result back via `out[col] = transformed` (line 246). At single-row inference the "median" is that row's own value, so a NaN cell is imputed with an unrelated observed cell. Two consequences: (a) the val/test imputation constant differs from train's, a train/serve statistic mismatch; (b) NaNs vanish from the output entirely without the docstring ever saying imputation happens.
**Suggested fix:** store `np.median(finite)` in the search result (`info["fill_median"]`, alongside `best_fitted_params` at line 195) and have `apply_gaussian_power_transform` use it. Better still, do not write the fill back: transform only the finite positions and restore NaN in the output, matching the docstring's "each searched column replaced by its best-scoring transform".
**Evidence:** line 153 `finite_fill[~np.isfinite(finite_fill)] = np.median(finite)` in the search; line 219 the identical expression in the apply, recomputed from `out[col]`. The apply docstring (lines 205-209) claims it "applies the SAME function that was measured and selected, not a freshly-refit one" -- true for the Box-Cox lambda / Yeo-Johnson power, false for the NaN fill.

**Disposition:** RESOLVED, both halves. `fill_median` is recorded at search time and replayed; and the filled values are no longer written into the output, so missingness survives the transform instead of silently becoming an imputation. `tests/preprocessing/test_power_transform_apply_replays_the_fitted_fill.py`.

### PREPROCESSING_DATA-5 [P1] leakage
**File:** src/mlframe/preprocessing/category_support.py:68
**Summary:** The OOF target-encoding path uses `global_mean` computed over the FULL `y_train` inside every fold's shrinkage formula and as the unseen-category fallback, so a row's own label does inform its own encoded value -- exactly what the docstring says cannot happen.
**Failure scenario:** `smoothed_target_encode_column(train_series, test_series, y_train, oof=True)` (the default). For a category appearing only in the held-out fold, `fold_shrunk` has no entry, so line 87's `.fillna(global_mean)` assigns that row *the mean of all train labels including its own*. For a category with `count` observations in the fold-train set, line 86 blends in `smoothing * global_mean`, again computed with the held-out row's label included. With a singleton category the encoded value is 100% own-label-contaminated; the contamination weight is `1/n_train` but is systematically aligned with the row's own label, so any correlation/AUC read off `train_encoded` is optimistically biased and the bias grows as `n_train` shrinks or the target becomes imbalanced.
**Suggested fix:** compute the global mean per fold from `y_train.loc[fold_train_idx].mean()` and use it both in `fold_shrunk` and in the `fillna`. Keep the full-train `global_mean` only for `test_encoded` (line 71), where it is correct.
**Evidence:** line 68 `global_mean = float(y_train.mean())` is computed once, outside the `kf.split` loop, and is referenced at lines 86 and 87 inside it. The docstring at lines 43-46 states "each train row's value comes from stats fit on every OTHER fold, so a row's own label never informs its own encoded value."

**Disposition:** RESOLVED. The shrinkage prior and the unseen-category fallback are both computed from the fold's own train rows. `tests/preprocessing/test_oof_target_encoding_uses_fold_local_stats.py`.

### PREPROCESSING_DATA-6 [P1] silent-wrong-value
**File:** src/mlframe/preprocessing/align_feature_direction.py:43
**Summary:** `batch_univariate_auc` assigns ranks from a raw `argsort` with no tie correction, so on any tied or low-cardinality column the computed AUC -- and therefore the sign-flip decision -- is determined by argsort ordering rather than by the data.
**Failure scenario:** a binary 0/1 indicator column (or any column with heavy ties, e.g. a count feature with a large zero mass) whose true tie-corrected AUC is 0.50-0.55. `np.argsort` breaks the tied block in an arbitrary, input-order-dependent order; the Mann-Whitney sum-of-ranks over positives can land on either side of 0.5, so `sign = -1 if auc < 0.5 else 1` (line 167) flips a column that should not be flipped, or fails to flip one that should -- and the flipped column then contributes the wrong sign to every pooled aggregate the module exists to protect. For a genuinely CONSTANT column the true AUC is exactly 0.5 (no flip), but with index-order ranks it is whatever the positive rows' positions dictate. `check_feature_direction_stability` (line 232) then reports the resulting per-fold sign churn as if it measured sampling noise. Second, independent failure: NaN sorts last under `np.argsort`, so a NaN cell receives the maximal rank -- a column whose missingness correlates with the positive class gets a spuriously inflated AUC, and nothing in the function rejects or drops non-finite input.
**Suggested fix:** use average (mid-)ranks -- `scipy.stats.rankdata(X, axis=0)`, which is what `roc_auc_score` effectively does for ties -- and either drop NaN rows pairwise per column or raise on non-finite input, since `batch_mutual_information` (line 83) will reject NaN anyway.
**Evidence:** the inline comment at line 43 explicitly says "ties broken arbitrarily (matches roc_auc_score's tie handling closely enough for a sign/threshold decision)". `roc_auc_score` uses mid-ranks; for a k-valued column with large tie blocks the two differ by O(tie-block-size / n) -- exactly the scale of the `|auc - 0.5|` quantity being thresholded at line 167 and at line 148 (`mi_near_chance_gap`).

**Disposition:** RESOLVED, and faster. Average ranks via a prange njit tie-block pass, bit-identical to `scipy.stats.rankdata` and matching `roc_auc_score` to 1.1e-16 including an exactly-0.5 constant column. Measured 1.42x/1.46x/1.12x FASTER than the untied ranks it replaces at (50k,200), (200k,100), (500k,50). Non-finite input is refused. `tests/preprocessing/test_univariate_auc_corrects_for_ties.py`.

### PREPROCESSING_DATA-7 [P1] silent-caller-frame-mutation
**File:** src/mlframe/preprocessing/unseen_category_imputer.py:156
**Summary:** In `similarity_mode="nearest"`, `out.loc[unreliable_mask, col] = replacement` writes through the shallow copy into the caller's input DataFrame, so `transform` mutates `df` -- the `"mode"` branch two lines below does not.
**Failure scenario:** `imp.transform(X_test)` with `similarity_mode="nearest"`. `out = df.copy(deep=False)` (line 123) shares blocks with `X_test`; `.loc[mask, col] = ...` on a shared block writes in place. Verified live on pandas 2.3.3 (Copy-on-Write off, the current default): after `out.loc[mask, 'a'] = ...`, the ORIGINAL frame's column read back `['Q','y','z']`. A caller that keeps `X_test` around for a second model, a baseline comparison, or an error-analysis join now has silently rewritten categories in it -- and a second `transform` call on the same frame sees the already-substituted values, so the fallback-rate diagnostic in `fallback_stats_` reads 0 the second time. The `else` branch at line 158 uses `out[col] = df[col].where(...)`, a whole-column rebind, which is safe -- so the bug fires only on the opt-in path.
**Suggested fix:** replace line 156 with a whole-column rebind: build a full-length replacement Series (mode/nearest per row) and do `out[col] = df[col].where(~unreliable_mask, replacement_full)`, matching the `mode` branch. (Under pandas 3 / CoW this would become correct on its own, but the module must not depend on that.)
**Evidence:** reproduced in-process on pandas 2.3.3: `df.copy(deep=False)` followed by `.loc[boolean_mask, col] = series` mutated the source frame. Note also the asymmetry with line 158 inside the same loop body.

**Disposition:** RESOLVED. Whole-column rebind, matching the `mode` branch. `tests/preprocessing/test_unseen_category_imputer_leaves_caller_frame_alone.py`.

### PREPROCESSING_DATA-8 [P1] silent-wrong-value
**File:** src/mlframe/preprocessing/cleaning_helpers.py:35
**Summary:** `map_elementwise_dedup`'s dedup fast path routes values through a Python dict, which collapses `True`/`1` and `False`/`0` into one key -- so for an object column mixing bools and ints it returns values that differ from the `s.map(fcn)` path it claims to be bit-identical to, and only above the 80k-row gate.
**Failure scenario:** an object-dtype column of raw scraped/CSV data holding a mix of `True`, `False`, `1`, `0` over 200k rows with low cardinality -- the exact regime this fast path targets. Verified in-process: `pd.unique` returns `[1, 0]` (the bools already collapsed), `{v: fcn(v) for v in u}` yields `{1: 'f_1', 0: 'f_0'}`, and `s.map(mapping)` returns `['f_1','f_1','f_0','f_0']` where `s.map(fcn)` returns `['f_1','f_True','f_0','f_False']`. The same frame at 50k rows takes the `n < 4 * sample` branch (line 26) and returns the CORRECT values -- so the answer depends on row count, the worst possible shape for a bug. The caller is `cleaning._clean_cat_and_obj_columns:165`, which runs on every object/string column whenever `obj_vars_clean_fcn` is supplied.
**Suggested fix:** key the mapping by `(type(v), v)` and reindex accordingly, or gate the dedup path on type-homogeneity of the probe (`probe.map(type).nunique() == 1`) and fall back to `s.map(fcn)` otherwise. The docstring's "Mapping over the *unique* values and reindexing back is bit-identical for a pure elementwise `fcn`" holds only when the values are hash-distinguishable.
**Evidence:** reproduced in-process (pandas 2.3.3): mapped `['f_1','f_1','f_0','f_0']` vs direct `['f_1','f_True','f_0','f_False']`.

**Disposition:** RESOLVED. The dedup mapping is keyed so `True` and `1` no longer collapse. `tests/preprocessing/test_map_elementwise_dedup_value_identity.py`.

### PREPROCESSING_DATA-9 [P1] memory
**File:** src/mlframe/utils/_param_oracle.py:268
**Summary:** `default_fingerprint` -- the DEFAULT fingerprint for every ParamOracle lookup -- materializes the caller's entire frame as a dense float64 array plus roughly eight full `(n, p)` temporaries, with no size cap or subsampling.
**Failure scenario:** ParamOracle wraps a function whose first array-like argument is a 100 GB training frame. `_as_2d_numeric` calls `obj.to_numpy()` (line 193, one full materialization), then line 268 `a = arr2d.astype(np.float64)` (a second), then in sequence: `finite` (269), `finite_mask` (277 -- the *identical* `np.isfinite(a)` expression computed a second time), `a0` (279), `dev` (282), `z` (287), `z**3` (288), `z**4` (289), `filled` (315), `inds` (318). That is ~2 float64 copies plus 2 bool masks plus ~6 float64 temporaries of the full frame simultaneously live, inside a function whose docstring bills itself as "stat-only" and "cheap". The failure is OOM, not a wrong answer.
**Suggested fix:** cap the fingerprint to a bounded row sample (head plus a stride sample to <=100k rows) before computing any statistic; reuse `finite` at line 277 instead of recomputing it; and compute the skew/kurtosis in a single fused pass (the repo already uses a numerically-stable centred-moments njit kernel for exactly this) instead of materializing `z`, `z**3` and `z**4`.
**Evidence:** lines 193 and 268-320 read end-to-end; there is no `n`-based gate anywhere in `default_fingerprint` or `_as_2d_numeric`. Line 269 `finite = np.isfinite(a)` and line 277 `finite_mask = np.isfinite(a)` are the same expression, and only `finite_mask` is used after line 274.

**Disposition:** RESOLVED. Statistics come from a deterministic strided sample bounded to 4M cells (with a 1000-row floor), the duplicate `np.isfinite` is gone, the `z`/`z**3`/`z**4` temporaries collapse to one squared-deviation array, and the correlation matrix is capped at 256 columns. `n` and `p` still report the true shape. Skew/kurtosis verified against scipy. `tests/utils/test_param_oracle_fingerprint_is_bounded.py`.

### PREPROCESSING_DATA-10 [P2] contract-drift
**File:** src/mlframe/preprocessing/outlier_capping_or_missing.py:108
**Summary:** The docstring tells callers to "apply the returned bounds' equivalent transform to test data", but the function returns only a DataFrame -- the bounds are never returned and there is no fit/apply split, so the module cannot be used leakage-free at all.
**Failure scenario:** a caller follows the documented discipline, passes only `X_train`, then looks for the bounds to reuse on `X_test` and finds none. The only options are (a) call `outlier_cap_or_missing(X_test)` again, which recomputes mean/std or IQR from the TEST distribution -- the exact "outlier threshold chosen on the data it then filters" pattern -- or (b) reimplement the private `_column_bounds` at the call site. Every sibling module in this package (`rare_count_pruning`, `regime_conditioned_imputation`, `missing_indicator_pairing`) ships an explicit fit/apply pair; this one does not.
**Suggested fix:** split into `fit_outlier_cap_or_missing(df, columns, rule)` returning per-column `(lower, upper, median)` and `apply_outlier_cap_or_missing(df, state, mode)`, keeping `outlier_cap_or_missing` as the single-frame wrapper -- the exact shape the three sibling modules already use. Note the `missing_impute` mode also needs its median persisted: line 152 recomputes `np.nanmedian(treated)` from the frame being transformed.
**Evidence:** docstring lines 108-110 vs the `-> pd.DataFrame` signature at line 102 and the sole `return out` at line 156. `_column_bounds` is module-private and never exposed.

### PREPROCESSING_DATA-11 [P2] contract-drift
**File:** src/mlframe/preprocessing/category_support.py:149
**Summary:** `target_col` is validated but never read by `train_test_support_screen`, and the documented constraint that it "must not be one of categorical_cols" is unenforced -- with the default `categorical_cols=None` the target column is itself screened as a categorical feature.
**Failure scenario:** `train_test_support_screen(train_df, test_df, target_col="y", enable_smoothed_target_encoding_fallback=True)` with `categorical_cols` left at its default. Line 147 sets `categorical_cols` to every column present in both frames, which includes the target whenever the test frame carries a label column, so the output contains a row recommending an encoding for the target itself. Meanwhile the `smoothed_target_encode` recommendation (line 184) is decided purely from `freq_cv`, never from `y` -- so the `target_col` requirement buys nothing except the ValueError at line 150.
**Suggested fix:** exclude `target_col` from the default `categorical_cols` at line 147 and raise if it appears in an explicit list; or drop the parameter and its validation, and state in the docstring that the caller supplies `y` only to the follow-up `smoothed_target_encode_column` call.
**Evidence:** grepping `target_col` across the file yields only the signature (line 98), the docstring, and the validation at line 149. It appears nowhere in the loop body (lines 153-199).

### PREPROCESSING_DATA-12 [P2] contract-drift
**File:** src/mlframe/preprocessing/missing_indicator_pairing.py:56
**Summary:** The docstring promises that `group_col` "itself is never imputed by this call", but with the default `columns=None` the column list is every column with at least one missing value, which includes `group_col`.
**Failure scenario:** `impute_with_missing_indicator(df, strategy="median", group_col="region")` where `region` has some nulls. Line 56 puts `region` into `cols`; line 84 stores a group-kind state for it, grouped by itself with `dropna=False`, so the NaN group's own stat is NaN; `apply_missing_indicator_imputation` line 113 maps `out[group_col]` through that stat, which misses for the NaN rows, and line 114 fills them with `global_fallback` -- the median of the region codes. The grouping key is now silently altered, and a `region_was_missing` indicator column appears that the docstring never promised.
**Suggested fix:** exclude `group_col` from the default `cols` computation at line 56, and skip it (or raise) if it appears in an explicit `columns` list.
**Evidence:** line 56 builds `cols` with no `group_col` exclusion; the guarantee is stated at line 152 of `impute_with_missing_indicator`'s docstring.

### PREPROCESSING_DATA-13 [P2] crash-on-natural-input-dtype
**File:** src/mlframe/preprocessing/rare_count_pruning.py:125
**Summary:** `apply_rare_category_collapse` uses `Series.where(cond, other_label)`, which raises TypeError on a pandas `category`-dtype column -- the natural input dtype for a rare-CATEGORY collapsing module. Same construct at adversarial_rebin.py:60-61.
**Failure scenario:** `apply_rare_category_collapse(df, mapping)` on a frame whose categorical columns carry `category` dtype -- which `cleaning.analyse_and_clean_features:675` in this same package produces automatically for fewly-valued object columns. Verified in-process on pandas 2.3.3: calling `.where(mask, "__other__")` on a Series built from a pandas Categorical raises `TypeError: Cannot setitem on a Categorical with a new category (__other__), set the categories first`. The `fit_rare_category_collapse` half succeeds, so the failure surfaces only at apply time -- potentially on the inference frame rather than during training.
**Suggested fix:** for a CategoricalDtype column, call `add_categories([other_label])` first (the pattern `transforms.prepare_df_for_catboost:195-198` already uses for exactly this reason), then `where`, then optionally `remove_unused_categories()`. Apply the same fix in `adversarial_rebin._merge_skewed_categories`.
**Evidence:** reproduced in-process (TypeError above). transforms.py:195-198 documents add-categories-first as the correct technique for Categorical columns in this codebase.

### PREPROCESSING_DATA-14 [P2] cache-key-collision
**File:** src/mlframe/utils/disk_cache.py:100
**Summary:** `hash_array_summary` is a deliberately sub-O(N) summary (shape, dtype, first/last 64 rows, per-column sum/min/max) that cannot distinguish two arrays differing only by a permutation of interior rows -- yet the module docstring promises caching "with zero correctness loss".
**Failure scenario:** the same feature matrix fed twice with a different interior row order (a re-sorted frame, a different CV shuffle applied before the call, a dedup pass that reorders). Permuting rows 64..n-64 leaves shape, dtype, head bytes, tail bytes, and every per-column sum/min/max byte-identical, so `compose_key` yields the same key and `DiskCache.get` returns the entry computed for the OTHER ordering. For the MRMR bin-edges consumer this is harmless (edges are permutation-invariant); for the ShapProxiedFS OOF-SHAP consumer named in the same docstring the payload is a per-ROW phi matrix, so every row attribution is silently misaligned -- no exception, plausible-looking numbers.
**Suggested fix:** either soften the "zero correctness loss" claim to state the summary-hash collision class explicitly, or add a cheap order-sensitive term -- e.g. fold a position-weighted accumulator over one column, or hash the bytes of a strided row sample -- so an interior permutation changes the key.
**Evidence:** module docstring line 11 ("Caching the result amortises the cost across re-fits with zero correctness loss") vs the hash inputs enumerated at lines 104-112 and implemented at lines 115-153. The head/tail defence noted at line 74 ("random row shuffles change the head/tail bytes") does not cover an interior-only permutation. Consumers are named at lines 5-6.

### PREPROCESSING_DATA-15 [P2] memory
**File:** src/mlframe/utils/_param_oracle_store.py:150
**Summary:** `_ParquetStore._aggregate` implements a weighted median by replicating each metric value `n_obs` times into a Python list, and `n_obs` accumulates monotonically across every `append`.
**Failure scenario:** a long-running tuning loop appends observations for the same (fn_name, host, fp_bucket, param_combo) key repeatedly. Each `append` calls `_aggregate` over ALL existing rows (line 99); for a row whose accumulated `n_obs` has reached, say, 500,000, line 150 builds a 500,000-element Python float list per metric per row just to take a median at line 151. With a few dozen keys and several metrics that is gigabytes of Python floats and a multi-second stall inside the cross-process file lock (line 103), blocking every other process appending to the same store.
**Suggested fix:** carry the weight alongside the value (append `(float(mv), w)` pairs) and compute a true weighted median by sorting the pairs and walking to the half-weight point -- O(rows log rows) instead of O(sum of n_obs). Separately, re-aggregating stored medians yields a median-of-medians, not the median of the underlying observations; the class docstring at line 50 says "MEDIAN" without that caveat.
**Evidence:** line 150 extends the list by `[float(mv)] * max(1, w)` with `w` read from `n_obs` at line 145; `total_obs` is summed at line 130 and written back into the aggregated row at line 159, so the multiplier grows without bound.

### PREPROCESSING_DATA-16 [P2] memory / contract-drift
**File:** src/mlframe/core/frame_compat.py:19
**Summary:** The dispatch table documents polars DataFrame conversion as "zero-copy where possible", but the implementation calls plain `.to_pandas()` (a full materialization into pandas blocks) while the repo already ships a genuinely zero-copy Arrow-backed bridge that this helper does not use.
**Failure scenario:** any caller routing a large polars frame through `to_pandas_or_array`. `X.to_pandas()` at line 105, without `use_pyarrow_extension_array=True`, copies every column into numpy-backed pandas blocks -- a full second copy of the frame. On a 100 GB frame this is an OOM inside a helper whose docstring implies otherwise. CLAUDE.md's own memory rule ("Frame-format conversions are the CALLER's decision, made once at the suite boundary -- inner wrappers must never silently down-convert on a hot path") makes this seam exactly where the guarantee matters.
**Suggested fix:** delegate the polars-DataFrame branch to `mlframe.training.utils.get_pandas_view_of_polars_df` -- the same bridge `feature_selection/boruta_shap/_fit_explain.py:145` and `:327` already use, and that CLAUDE.md documents as the validated zero-copy path -- falling back to `.to_pandas()` only if it raises. At minimum, drop the "zero-copy where possible" claim from line 19.
**Evidence:** module docstring line 19 vs the implementation at lines 103-105. `grep -rn get_pandas_view_of_polars_df src/` shows the zero-copy bridge exists in `mlframe.training.utils` and is used elsewhere; it is not referenced in frame_compat.py.

### PREPROCESSING_DATA-17 [P2] contract-drift
**File:** src/mlframe/preprocessing/temporal_drift_augment.py:125
**Summary:** The synthetic row's non-feature columns -- including the label -- are taken from the truncated-vintage row, not from the entity's TRUE last period the docstring promises.
**Failure scenario:** a panel with a per-period label (a rolling default flag, a next-period target). `synth = ordered.loc[eligible].copy()` selects rows at `rank_within_entity == count - n_drop - 1`, i.e. the truncated vintage, and only `feature_cols` are overwritten at line 131. Every other column, label included, is that earlier row's value, so the augmented rows are trained against the earlier period's target. For an entity-level (period-invariant) label the two coincide and nothing breaks; for a period-varying label the augmentation silently mislabels every synthetic row.
**Suggested fix:** either merge the true-last row's non-feature columns onto `synth` (join the entity's `rank == count - 1` row on `entity_col`), or amend the docstring to state that non-feature columns come from the truncated-vintage row and that the technique assumes an entity-level label.
**Evidence:** line 122 defines `eligible` as `rank_within_entity == new_last_rank`; line 125 copies those rows wholesale; line 131 overwrites only `feature_cols`. The claim is at docstring lines 33-35 ("the real label at that entity's TRUE last statement, per the source technique").

### PREPROCESSING_DATA-18 [P2] silent-wrong-value
**File:** src/mlframe/preprocessing/cleaning.py:841
**Summary:** `features_ranges[col]["median"]` is the median of the column's DISTINCT values, not the column's median -- the `value_counts` index is used as if it were the data.
**Failure scenario:** a manyvalued numeric column, 10M rows, values concentrated near 0 with a long sparse tail (a classic monetary or count feature). `col_unique_values` is a `value_counts` Series whose INDEX holds the distinct values; `np.nanmedian(col_unique_values.index)` returns the midpoint of the distinct-value SET, ignoring counts entirely -- for the described column that lands far out in the tail, not near 0. Any consumer of `features_ranges` (novelty detection, range checks, imputation defaults) then reads a number labelled "median" that is nowhere near the column's median. `min`/`max` computed off the same index are correct; only `median` is wrong.
**Suggested fix:** compute the median from the counts -- sort the index, cumsum `col_unique_values.values`, take the half-count crossing -- or restore the commented-out `df[col].describe()` form shown at lines 827-836.
**Evidence:** lines 838-842 build the dict from `col_unique_values.index`; `col_unique_values` is assigned at line 632 as `sub_df[col].value_counts(dropna=False)`.

### PREPROCESSING_DATA-19 [P2] contract-drift
**File:** src/mlframe/preprocessing/cleaning.py:761
**Summary:** A column converted to `category` dtype by step 3 is silently converted back to `object` by step 5, because `head` is a snapshot taken before any conversion -- undoing the documented memory saving.
**Failure scenario:** a fewly-valued object column, say 40 distinct country codes over 10M rows. Line 675 does `df[col] = df[col].astype("category")` and logs "converted to category type". If that column then has rare values merged, line 757 casts it to `object` and line 761 casts to `the_type = head[col].dtype.name` -- but `head = df.head(1)` was taken at line 596, BEFORE the category conversion, so `the_type` is `object`. The column ends the function as `object`, at full string-per-row memory, and the returned `dtypes=df.dtypes` records that. The docstring's step 3 promise (line 526, "Converts fewly-valued ... object features into categorical, to save space & increase processing speed") is silently reverted for exactly the columns that also needed rare-value merging.
**Suggested fix:** read the CURRENT dtype (`df[col].dtype.name`) rather than the stale `head[col].dtype.name` at lines 752, 761 and 808, or re-apply `.astype("category")` after the mask when the column was categorical on entry to the block.
**Evidence:** `head = df.head(1)` at line 596; the category conversion at line 675 mutates `df`, not `head`; lines 752, 761 and 808 all read `head[col].dtype.name`.

### PREPROCESSING_DATA-20 [P2] contract-drift
**File:** src/mlframe/utils/disk_cache.py:160
**Summary:** `hash_object`'s docstring describes an implementation that no longer exists ("Pickle protocol 0 is used to get a stable, key-sorted-ish representation"); the function actually delegates to the hand-rolled `_feed` byte encoder.
**Failure scenario:** a maintainer reasoning about hash stability across Python versions reads "pickle protocol 0", concludes the encoding is whatever `pickle` produces, and adds a new type to a params dict expecting pickle's handling. `_feed`'s actual last-resort branch (lines 218-222) is `repr(obj)`, which is NOT stable across runs for most objects -- a materially different stability contract from the documented one. The sort-keys guarantee the docstring claims is real (line 198), but the mechanism described is not.
**Suggested fix:** rewrite the docstring to describe `_feed`'s tagged, length-prefixed encoding and its `repr()` last resort, and state explicitly which types are safe to pass.
**Evidence:** docstring lines 160-165 vs the body at lines 166-168 (`_feed(h, obj)`); no `pickle` call appears anywhere in `hash_object` or `_feed`.

### PREPROCESSING_DATA-21 [P3] memory-guard-defeated
**File:** src/mlframe/preprocessing/cleaning.py:483
**Summary:** The broad `except Exception` around `df.memory_usage(deep=True)` sets `df_bytes = 0`, so a failure in the size probe falls through to `df.copy()` on a frame that may be far above the 2 GB guard the probe exists to enforce.
**Failure scenario:** `memory_usage(deep=True)` raises on an exotic extension dtype or an object column holding an un-sizeable payload. `df_bytes = 0` then passes the `> _DEFRAG_COPY_MAX_BYTES` test at line 486, and line 491 executes `df.copy()` on the full frame -- doubling peak RAM on exactly the huge, dtype-unusual frame the guard was written for. The failure is logged at DEBUG only (line 484), so it is invisible in production.
**Suggested fix:** fail closed on exception -- `return df, prev_mem_usage`, skipping the defrag copy -- and log at WARNING, since a silent skip of the size check is the dangerous branch.
**Evidence:** lines 481-491; the comment at lines 471-472 states the guard's purpose ("a 100+ GB prod frame OOMs the host").

### PREPROCESSING_DATA-22 [P3] cache-self-invalidation
**File:** src/mlframe/utils/disk_cache.py:438
**Summary:** The `protect` argument of `_evict_if_needed` shields only the `<key>.pkl` payload, not its `.sha256` sidecar, so eviction can drop the sidecar of the entry the caller just paid to compute.
**Failure scenario:** a `put` that pushes the directory over cap. `_evict_if_needed(protect=path)` scans every file including sidecars -- only `tmp_`-prefixed names are skipped, line 422 -- and unlinks oldest-first. The just-written payload is skipped by the `protect` check, but its sidecar is a separate file and is not, so it can be deleted in the same pass. The next `get` for that key then hits the fail-closed `safe_load` path, raises `PickleVerificationError`, and lines 346-353 delete BOTH files -- the entry is gone and the expensive compute is repeated.
**Suggested fix:** compare against both `protect` and `Path(str(protect) + ".sha256")` at line 438, and pair each payload with its sidecar in the eviction accounting so they are removed together.
**Evidence:** line 438 skips only `fpath.resolve() == protect.resolve()`; line 400 writes the separate sidecar file; line 422 skips only `tmp_`-prefixed names.

### PREPROCESSING_DATA-23 [P3] contract-drift
**File:** src/mlframe/preprocessing/gaussian_power_transform_search.py:172
**Summary:** The docstring says rows non-finite in either the column or the target are dropped pairwise for the correlation check, but the column's non-finite rows were already median-filled, so only the target's non-finites are actually dropped.
**Failure scenario:** a column with 40% missing feeding the `require_target_correlation_retention` guard. `pair_mask = np.isfinite(finite_fill) & np.isfinite(y_arr)` -- `finite_fill` is finite everywhere by construction (line 153), so 40% of the correlation's rows are a constant (the median), which mechanically attenuates `raw_target_corr` toward 0. The guard's `min_required` threshold (line 174) is then computed against an artificially weak baseline, so aggressive transforms pass a retention check they should have failed.
**Suggested fix:** build `pair_mask` from the RAW array (`np.isfinite(raw) & np.isfinite(y_arr)`) and correlate `raw[pair_mask]` against `transformed[pair_mask]`.
**Evidence:** line 153 fills every non-finite position; line 172 tests `np.isfinite(finite_fill)`, which is unconditionally True.

### PREPROCESSING_DATA-24 [P3] dtype-coercion
**File:** src/mlframe/preprocessing/outlier_capping_or_missing.py:148
**Summary:** Any treated column is written back from a float64 numpy array, so integer columns are silently widened to float64 even in cap mode, where no NaN is introduced.
**Failure scenario:** an int32 count feature with outliers. `values = out[col].to_numpy(dtype=np.float64)` (line 140), then `out[col] = np.clip(values, lower, upper)` (line 148) -- the column is now float64. Downstream, `cleaning.is_variable_truly_continuous` branches on the `np.modf` fractional structure, and a float64 column of integers produces different `n_unique_ints`/`n_unique_fracts` accounting than an int column, so the discrete-vs-continuous classification and therefore the rare-value cleaning path can change. The docstring at line 132 says only "with each treated column's outliers capped or replaced+imputed in place".
**Suggested fix:** in cap mode, cast the clipped result back to the original dtype when it is integral and the bounds are integral; otherwise document the widening explicitly.
**Evidence:** lines 140, 148 and 154; docstring lines 129-132 make no dtype statement.

### PREPROCESSING_DATA-25 [P3] misleading-diagnostic
**File:** src/mlframe/data_valuation/_adversarial_validation.py:89
**Summary:** When no fold's model exposes `feature_importances_`, `importances` stays all-zero and `top_shift_features` still returns the first 20 column names as the features driving the shift.
**Failure scenario:** a caller passes a LogisticRegression (or any estimator exposing `coef_` but not `feature_importances_`). The `if fi is not None` guard at line 82 skips every fold, `importances` remains `np.zeros(...)`, `np.argsort(-importances)[:20]` returns indices 0..19 in plain column order, and the returned `top_shift_features` is just the frame's first 20 column names -- presented by the docstring at lines 37-39 as the features driving the shift. An operator then acts on a ranking that carries no information at all.
**Suggested fix:** track whether any fold contributed importances; if none did, return an empty list (or None) for `top_shift_features` and log a warning naming the estimator type.
**Evidence:** lines 70, 81-84 and 89-90; `np.argsort` on an all-equal array returns index order.

### PREPROCESSING_DATA-26 [P3] contract-drift
**File:** src/mlframe/utils/nan_safe.py:76
**Summary:** The documented return of `argmax_classes_safe` is an "(N,) int64 array of class indices", but the 1-D input branch returns a 0-d array, and the all-finite sub-branch does not even set int64.
**Failure scenario:** a caller writes `preds = argmax_classes_safe(probs)` then `len(preds)` or `preds[mask] = ...` against a 1-D `probs`. Line 76 returns `np.asarray(np.argmax(probs))` -- a 0-d array of dtype `intp`, not int64 -- so `len()` raises "len() of unsized object", and any downstream assumption about int64 width is unmet on platforms where `intp` is int32. The 2-D path (lines 94 and 109) does honour the documented shape and dtype, so the inconsistency bites only the degenerate input.
**Suggested fix:** make the 1-D branch return a 1-element int64 array, or document explicitly that 1-D input yields a scalar; add `dtype=np.int64` at line 76 either way.
**Evidence:** docstring line 60 versus lines 76, 83 and 84.

### PREPROCESSING_DATA-27 [P3] fragility
**File:** src/mlframe/preprocessing/sibling_group_cold_start_fill.py:79
**Summary:** The interpolate path re-indexes by the raw `order_col` values and calls `interpolate(method="index")`, which requires a numeric/datetime, strictly-ordered, unique index -- none of which the parameter's documented contract guarantees.
**Failure scenario:** `order_col` is a string sequence id such as a quarter label, which the docstring explicitly blesses as "a sortable ordering across DISTINCT groups". `filled_per_group.index = order_values` then produces an object index and `interpolate(method="index")` raises. Alternatively, two groups sharing the same `order_col` value produce a duplicated index, where the distance weighting is undefined -- and distance weighting is the entire reason the comment at lines 71-75 gives for choosing `method="index"` over positional interpolation.
**Suggested fix:** validate at the top of the interpolate branch that `order_values` is numeric or datetime and unique, raising a named error otherwise; or fall back to positional linear interpolation with an explicit warning when it is not.
**Evidence:** lines 76-81; docstring lines 37-39 describe `order_col` only as a sortable ordering across distinct groups, giving a group-level sequence id or a first-seen timestamp as examples.

### PREPROCESSING_DATA-28 [P3] memory
**File:** src/mlframe/core/arrays.py:280
**Summary:** The `ascending=True` branch of `topk_by_partition` takes a full `.copy()` of the caller's array that nothing subsequently mutates.
**Failure scenario:** `topk_by_partition(big_scores, k, ascending=True)` on a large score matrix doubles peak memory for no reason -- after line 280, `arr` is only read (`arr.ravel()` at 288, `np.argpartition` at 308, `np.take` at 310, `np.take_along_axis` at 311). The `ascending=False` branch at line 278 already produces a fresh array as a side effect of negation and does not copy again, so the two branches have inconsistent allocation behaviour for no functional reason.
**Suggested fix:** replace line 280 with `arr = np.asarray(arr)`; the docstring promise that the function does not mutate the caller's array is already satisfied because no in-place operation remains.
**Evidence:** lines 276-280 and the read-only uses at lines 288, 291, 308, 310 and 311.

## Coverage

Read in full:

- src/mlframe/preprocessing/: `cleaning.py`, `cleaning_helpers.py`, `_cleaning_kernels.py`, `transforms.py`, `scalers.py`, `outliers.py`, `outlier_capping_or_missing.py`, `outlier_policy.py`, `outlier_detector_zoo.py`, `unseen_category_imputer.py`, `rare_count_pruning.py`, `category_support.py`, `regime_conditioned_imputation.py`, `missing_indicator_pairing.py`, `sibling_group_cold_start_fill.py`, `auto_transform_select.py`, `gaussian_power_transform_search.py`, `align_feature_direction.py`, `adversarial_rebin.py`, `degradation_augment.py`, `temporal_drift_augment.py`, `cluster.py`, `__init__.py`
- src/mlframe/core/: `helpers.py`, `stats.py`, `arrays.py`, `binning.py`, `frame_compat.py`, `category_encoders_compat.py`, `robust_location.py`
- src/mlframe/utils/: `disk_cache.py`, `safe_pickle.py`, `misc.py`, `nan_safe.py`, `_param_oracle_store.py`, `eda.py`, `text.py`
- src/mlframe/data/: `datasets.py`
- src/mlframe/data_valuation/: `_adversarial_validation.py`, `_weights.py`, `_training_weight_adapter.py`
- src/mlframe/config.py

Read in part, targeted at fingerprint / hash-key / leakage surfaces:

- src/mlframe/utils/`_param_oracle.py` -- lines 79-400 (bucket dims, `_host_key`, `default_store_dir`, `_as_2d_numeric`, `default_fingerprint`, `bucketize_fingerprint`, `_rss_mb`, store re-exports). The `ParamOracle` class body beyond line 400 was not read.
- src/mlframe/data_valuation/`_knn_shapley.py` -- lines 1-90.

Not read, and no findings are claimed against them: `data/synthetic.py`, `core/composite_similarity.py`, `core/ewma.py`, `core/matrix_seriation.py`, `core/proportion_stats.py`, `core/recency_weights.py`, `core/recency_step_weight.py`, `core/set_similarity.py`, `utils/log_throttle.py`, `utils/experiments.py`, `data_valuation/_adversarial_reweighting.py`, `_mc_sampling.py`, `_knn_shapley_multi_output.py`, `_knn_shapley_regression_binarize.py`, `_propagate_gpu_ktc.py`, and every `_benchmarks/` subtree.

Verification method: the pandas/numpy behaviours underpinning findings 1, 3, 7, 8 and 13 were confirmed by executing them in-process against the installed pandas 2.3.3 / numpy 2.3.5 (read-only, no files written) rather than reasoned from documentation.

Cross-cutting checks that came back CLEAN, with no finding raised:

- No `min() == max()` constant-column predicate exists anywhere in this cluster. The polars ones live in `training/_nan_processing.py` (out of scope) and already use `eq_missing`; the numeric-only guards inside this cluster (`arrayMinMax`, `_nanminmax_cols`, `fit_bin_smoother`) all handle the all-NaN case explicitly and were checked individually.
- `pl.Categorical` usage in `transforms.prepare_df_for_catboost` is correct: it preserves the per-Series `pl.Enum` domain across the string round-trip (lines 127-132) and warn-throttles the two remaining bare-`Categorical` casts as global-string-cache widening (lines 134-138 and 149-153).
- The one JSON-to-hash-key path in the cluster, `_param_oracle_store._stable_json` (line 31), correctly passes `orjson.OPT_SORT_KEYS`.
- No whole-frame pickle is used for caching: `DiskCache` stores only caller-supplied compute results, and `__getstate__` (line 284) correctly excludes the runtime lock dicts, matching the serialization-hygiene convention.
- `regime_conditioned_imputation`'s `out[col].where(out[col].notna(), fill_values)` was suspected of an int-to-float coercion; verified in-process that pandas preserves int64 when the condition is all-True, so no finding.
- No performance findings are reported. Per the njit-check rule, `_cleaning_kernels.py`, `outliers.py`, `core/arrays.py` and `core/robust_location.py` were grepped and all carry `@njit`/`prange` coverage with measured crossover gates (`_NANMINMAX_PARALLEL_MIN_ELEMS = 20_000`, `_ROBUST_MEAN_PARALLEL_MIN_N = 50_000`, `_PARALLEL_EDGES_MIN_COLS` in the sibling cluster), plus documented bench-attempt-rejected notes in `cleaning.py` at lines 273-275 and 385-388. Remaining candidates were not measured, so none are claimed.

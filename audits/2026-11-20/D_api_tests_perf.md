# D - MRMR public API, sklearn contract, persistence, tests, performance (read-only)

Scope: `src/mlframe/feature_selection/filters/` (mrmr package, `_mrmr_validate_transform.py`, `_mrmr_setstate_defaults.py`, `_mrmr_partial_fit.py`, `_fe_stability_vote.py`, `_univariate_route_device.py`) and `tests/feature_selection/filters/`. No source or test file was edited and no tests were run. `mrmr/_mrmr_class.py` and `MRMR.__init__` were read only as the reference for defaults and pickle state, and are not listed as debt.

Checked and dropped (no defect found):
- `get_feature_names_out` input-feature drift contract and the `_feature_names_in_synthesized_` sentinel.
- Pandas and polars transform name-based selection, missing-column RuntimeError, reordered-input handling (by name).
- Pickle: `__getstate__` drops the lock, the partial-fit buffers are reset on explicit fit, `_stability_replay_state_` is size-capped, `_engineered_names_cache_` is dropped.
- `_upload_columns` in `_univariate_route_device.py` already batches the H2D copy and documents its column-by-column fallback.
- `_column_all_missing` uses a strided sample before the full scan.
- Partial-fit `X_df.copy()` copies only the incoming batch, and only on column-name mismatch.
- The `_cat_fe_cache_` pickled payload is O(columns) scalars and signatures, not row data.
- Constructor parameters with zero readers: none found in the sample checked (the `fe_*`, `dcd_*`, `cluster_aggregate_*` families are read through `getattr` in their step modules).

### F01 - Identity fast-path returns input column order when support_ is a permutation, so transform disagrees with get_feature_names_out
- **Severity**: High
- **Location**: `src/mlframe/feature_selection/filters/_mrmr_validate_transform.py:506-517` (fast-path block in `transform`); `_mrmr_fit_impl/_assign_support.py:202,781` and `_mrmr_fit_impl/_finalise.py:257` (support_ written in selection order); `_mrmr_class_transform.py:117-121` (names built in support order)
- **Problem**: `support_` is built from `selected_vars` in greedy selection order and is not sorted before it is stored. When every input column is selected and no engineered recipes exist, the fast-path returns `X` itself whenever the incoming names equal `feature_names_in_` (same order). That skips the reordering step. `get_feature_names_out` and the non-fast-path pandas branch (`X[selected_cols]`, where `selected_cols` follows `support_`) both use selection order. So the fast path and the normal path return different column orders for the same fitted object, depending only on whether all columns were kept.
- **Failure scenario**: Fit on `df` with columns `[a, b, c]`. Greedy selection picks `c` first, then `a`, then `b`, so `support_ = [2, 0, 1]`, with no recipes. `transform(df)` returns a frame with columns `[a, b, c]` (identity fast-path), but `get_feature_names_out()` returns `['c', 'a', 'b']`. A downstream step that maps output columns to `get_feature_names_out()` by position gets the wrong feature names. If one column were dropped, the normal path would return `[c, a]` in the correct order, so the mismatch appears only in the all-selected case.
- **Suggested fix**: Apply the fast-path only when `support_` equals `arange(n_features_in_)` (sorted identity), or compare `np.asarray(support)` against `np.arange(len(support))`. Otherwise fall through to the normal path. Add a regression test that fits or stubs a permuted `support_` (see F02).
- **Disposition**: FIXED 2026-11-20 - the identity fast path of `transform` also requires `support_` to be the sorted identity (`_mrmr_validate_transform.py`); sensor `test_permuted_support_is_not_returned_as_the_identity` fails on the old code.

### F02 - Identity fast-path test stub uses sorted support_, so the permuted-selection case is never exercised
- **Severity**: Medium
- **Location**: `tests/feature_selection/filters/test_mrmr_transform_identity_fastpath_identity.py:25` (`self.support_ = np.arange(len(names))`); tests at lines 33-65
- **Problem**: The fast-path tests build the stub with `support_ = np.arange(...)`, which is always sorted. The reorder-in-support-order case behind F01 cannot be reached by any test in this file. The reorder test only covers an input frame whose columns are permuted; it does not cover a fitted `support_` that is permuted. The fast-path tests also do not check `get_feature_names_out()` against `transform(...).columns`, which is the contract that F01 breaks.
- **Failure scenario**: Any change or regression that makes the fast path ignore `support_` order passes this file. Concretely, the stub with `support_ = [2, 0, 1]` and an input `[a, b, c]` returns the input frame unchanged and no test fails.
- **Suggested fix**: Add a case with a permuted `support_` that asserts `list(transform(...).columns) == list(get_feature_names_out())`. Add the same assertion to the existing identity test.
- **Disposition**: FIXED 2026-11-20 - the stub of `test_mrmr_transform_identity_fastpath_identity.py` takes a `support` argument; a permuted-selection test and a boolean-mask test were added.

### F03 - Renamed-column test assertion is vacuous: it passes for almost any exception message
- **Severity**: Low
- **Location**: `tests/feature_selection/filters/test_mrmr_transform_identity_fastpath_identity.py:52`
- **Problem**: `assert "zzz" not in str(excinfo.value) or "a" in str(excinfo.value)` is satisfied by any message that contains the letter "a" (for example "expected", "column", "missing"), and the `pytest.raises(Exception)` above is broad. The test does not check that the error names the missing fitted column `a`, and it would also pass if the error were unrelated.
- **Failure scenario**: If the fast path were broken so that the renamed frame returned without error, the code path would raise nothing and the test would still fail with `excinfo` unset. The real weakness is the other direction: a RuntimeError about an unrelated problem (for example a dtype message containing "a") satisfies the assertion, so the test passes without verifying the column-drift diagnostic.
- **Suggested fix**: Assert on the specific diagnostic, for example `"a" in str(excinfo.value)` and `"missing" in str(excinfo.value)`, and narrow `pytest.raises` to `RuntimeError`.
- **Disposition**: OPEN

### F04 - Fitted getattr fallback for fe_optimizer disagrees with the constructor default
- **Severity**: Low
- **Location**: `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_core_parts.py:682` (`getattr(self, "fe_optimizer", "cma_batch")`); the constructor default is `"cupy_kernel"` in `mrmr/_mrmr_class.py:1183`; `_mrmr_setstate_defaults.py` has no `fe_optimizer` entry
- **Problem**: The reader's fallback names a different optimizer from the constructor default. In normal use the attribute always exists (the constructor sets it, and `__setstate__` re-sources ctor keys from `_ctor_defaults()`), so the fallback is reached only on an instance created without `__init__` (for example `object.__new__`, or a subclass that bypasses the constructor). In that case the fit silently runs the CMA-batch optimizer rather than the documented default.
- **Failure scenario**: `m = MRMR.__new__(MRMR); m.fit(X, y)` (attributes unset) runs `fe_optimizer="cma_batch"`, while `MRMR().fit(X, y)` runs `"cupy_kernel"`. The two fits can select different engineered features with no log line.
- **Suggested fix**: Read the default from the single source (`_ctor_defaults()["fe_optimizer"]`) or make the reader use the same literal as the constructor. Add a meta test that every `getattr(self, "<ctor param>", <default>)` fallback equals the constructor default.
- **Disposition**: OPEN

### F05 - Test fixtures change environment and reload a module without a full restore
- **Severity**: Low
- **Location**: `tests/feature_selection/filters/test_cmi_null_subsample.py:38-46` (env set, then `pop` in `finally`); `:57` (`importlib.reload(m)` in `test_excess_and_floor_cap_actually_subsamples`, restored only by the `m.__dict__` snapshot in its `finally`)
- **Problem**: The first test sets `MLFRAME_CMI_NULL_MAX_ROWS` and removes it with `pop` in `finally`, so a value that was already set in the environment (for example by CI) is lost for the rest of the session instead of restored. The reload-based tests rebuild the module in place and restore only `m.__dict__`. Other objects that imported names from this module keep the pre-reload classes, which breaks identity checks and violates the repository rule against reloading modules in tests.
- **Failure scenario**: CI exports `MLFRAME_CMI_NULL_MAX_ROWS=250000`. After the first test runs, the variable is gone for later tests in the same process, so any test that depends on it silently takes the full-n path. A later test that imports `_excess_and_floor` by name gets the pre-reload function object and tests a different implementation than the one it reads from the module.
- **Suggested fix**: Use `monkeypatch.setenv` / `monkeypatch.delenv` (as the second test already does for the env var) and replace `importlib.reload` with `monkeypatch.setattr` on the single knob the test needs.
- **Disposition**: OPEN

### F06 - ndarray replay wrap-failure warning claims wrong columns, but the code path raises a RuntimeError
- **Severity**: Low
- **Location**: `src/mlframe/feature_selection/filters/_mrmr_validate_transform.py:822-834` (`_recipe_replay_frame`), and `_replay_recipes_single_pass` (src_names check around line 790-800)
- **Problem**: When wrapping an ndarray with `feature_names_in_` fails, the code logs "engineered-recipe replay will run on an unnamed frame and may produce wrong columns" and keeps the unnamed ndarray. `_replay_recipes_single_pass` then reads `columns` from that ndarray (none), so every recipe with non-empty `src_names` fails the missing-source check and raises `RuntimeError`. The warning therefore describes a silent wrong-column outcome that does not occur, and the actual failure is an error that names every recipe source as missing.
- **Failure scenario**: `transform(X_ndarray)` with the wrong width for `feature_names_in_` (passing the shape check but not the wrap) produces the warning, then `RuntimeError: ... references source column(s) [...] missing from input X` for each recipe. An operator reading the warning looks for a silent corruption that is not there, and the real cause (the array does not match the fitted names) is not stated.
- **Suggested fix**: Raise the wrap error directly with the fit-time column count and the input shape, instead of falling through to the unnamed path. Correct the warning text to match the behaviour, and add a test that a mismatched ndarray raises with the shape in the message.
- **Disposition**: OPEN

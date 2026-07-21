# feature_engineering/ top-level B (grouped/timeseries/spatial, categorical encodings, misc extractors) -- mlframe audit

## Scope

All 39 files were read in full (not sampled/skimmed).

- src/mlframe/feature_engineering/grouped.py (972 LOC)
- src/mlframe/feature_engineering/timeseries.py (897 LOC)
- src/mlframe/feature_engineering/spatial.py (794 LOC)
- src/mlframe/feature_engineering/anchor.py (741 LOC)
- src/mlframe/feature_engineering/hurst.py (664 LOC)
- src/mlframe/feature_engineering/basic.py (577 LOC)
- src/mlframe/feature_engineering/wavelet_dwt.py (515 LOC)
- src/mlframe/feature_engineering/recency_aggregation.py (420 LOC)
- src/mlframe/feature_engineering/__init__.py (378 LOC)
- src/mlframe/feature_engineering/cat_cooccurrence_svd.py (314 LOC)
- src/mlframe/feature_engineering/bruteforce.py (306 LOC)
- src/mlframe/feature_engineering/_numerical_counts.py (284 LOC)
- src/mlframe/feature_engineering/relational_dfs.py (264 LOC)
- src/mlframe/feature_engineering/multi_decomposition_bank.py (250 LOC)
- src/mlframe/feature_engineering/nadaraya_watson.py (240 LOC)
- src/mlframe/feature_engineering/holiday_calendar_features.py (226 LOC)
- src/mlframe/feature_engineering/recency_density.py (222 LOC)
- src/mlframe/feature_engineering/multi_window_aggregate.py (207 LOC)
- src/mlframe/feature_engineering/categorical_group_concat.py (195 LOC)
- src/mlframe/feature_engineering/sentinel_missing_count.py (178 LOC)
- src/mlframe/feature_engineering/pysr_operators.py (177 LOC)
- src/mlframe/feature_engineering/auxiliary_feature_prediction.py (169 LOC)
- src/mlframe/feature_engineering/latent_parameter_recovery.py (163 LOC)
- src/mlframe/feature_engineering/drift_remediation.py (160 LOC)
- src/mlframe/feature_engineering/state_duration.py (146 LOC)
- src/mlframe/feature_engineering/tfidf_svd_entity_embedding.py (144 LOC)
- src/mlframe/feature_engineering/graph_construction.py (136 LOC)
- src/mlframe/feature_engineering/curated_fe.py (123 LOC)
- src/mlframe/feature_engineering/random_lag_augmentation.py (122 LOC)
- src/mlframe/feature_engineering/holiday_locale_target_encoding.py (116 LOC)
- src/mlframe/feature_engineering/gmm_bic_membership_features.py (114 LOC)
- src/mlframe/feature_engineering/binned_unique_count.py (107 LOC)
- src/mlframe/feature_engineering/categorical_powerset_concat.py (104 LOC)
- src/mlframe/feature_engineering/polars_dynamic_window.py (102 LOC)
- src/mlframe/feature_engineering/windowed_edge_diff.py (97 LOC)
- src/mlframe/feature_engineering/panel_sequence_tensor.py (95 LOC)
- src/mlframe/feature_engineering/rolling_target_correlation.py (84 LOC)
- src/mlframe/feature_engineering/entity_diff_features.py (74 LOC)
- src/mlframe/feature_engineering/magnitude_sample_weight.py (62 LOC)

**Total files reviewed: 39. Total LOC reviewed: 10939** (8644 for the 20 files >=178 LOC + 2295 for the 19 files <=177 LOC, both wc -l counts verified directly).

Note: `timeseries.py` re-exports 11 `_emit_*` helper functions from a sibling file `_timeseries_emit.py` (a prior monolith-split, per this repo's own convention). `_timeseries_emit.py` is NOT in my assigned 39-file scope list, so its internals were not audited here (only its call sites from `timeseries.py`, which are unremarkable pass-throughs) -- see Coverage notes.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P0 | correctness | holiday_calendar_features.py:82-85 | `is_holiday`/`is_eve` use exact datetime64 equality against midnight-normalized holiday dates; any input date with a time-of-day component silently always evaluates False, no error/warning. |
| F2 | P1 | perf/memory | multi_window_aggregate.py:69 | `windowed_history = history_df.copy()` is a full-frame deep copy repeated once per horizon in a loop, but the variable is never mutated -- pure wasted copy on every iteration. |
| F3 | P1 | silent-failure | grouped.py:316-324 | `per_group_apply`'s broad `except Exception` swallows any error raised by the caller-supplied `fn` (e.g. a signature/type bug), silently degrading the whole feature to `fill_value` with only a per-group log WARNING, not a raised error. |
| F4 | P2 | perf | anchor.py:227-289, 292-353 | `add_anchor_extrapolation_features` has no numba-accelerated core (unlike its 4 siblings) and recomputes the full K-anchor OLS regression from scratch on every row, not just at new anchors. |
| F5 | P2 | API surface | __init__.py:32-238 vs 239-378 | Dozens of functions imported into the package namespace are omitted from `__all__` (e.g. `per_group_rank`, `per_group_shift`, `per_group_cum_reduce`, `per_group_rolling_reduce`, `per_group_nth`; 5 of 6 `anchor.py` functions; 5 of 7 `spatial.py` functions; 3 `hurst.py` functions). |
| F6 | P2 | API surface | relational_dfs.py (whole file), __init__.py:223 | `stack_relational_chain`/`RelationalHop` -- the general N-depth API `stack_relational_features` now delegates to -- are never imported into `__init__.py`. |
| F7 | P2 | API surface | categorical_group_concat.py:73-193, __init__.py:228 | `discover_categorical_groups`/`auto_concat_categorical_groups` (first-class per the module's own docstring) are never imported into `__init__.py`. |
| F8 | P2 | memory convention | entity_diff_features.py:61 | `out = df.copy()` is a full DEEP copy; every sibling file doing the identical "copy then append columns" pattern uses `df.copy(deep=False)`. |
| F9 | P2 | perf (minor) | relational_dfs.py:78 | `result = parent_df.copy()` is redundant whenever `child_specs` is non-empty, since `pd.concat` on the first loop iteration immediately allocates a fresh object anyway. |
| F10 | P2 | code quality | timeseries.py:169-173,467-468,546-548,589-592; _numerical_counts.py:140,175 | "Wave NN (date)" process/audit markers embedded in code comments, which this repo's own CLAUDE.md explicitly bans. |
| F11 | P2 | dead/confusing code | timeseries.py:659-664 | `create_and_process_windows`'s `if forward_direction: ... else: ...` branch assigns the identical two statements in both arms, making the conditional a no-op and the "else-branch" comment stale. |
| F12 | P2 | API asymmetry | nadaraya_watson.py:180-240 | `per_group_nadaraya_watson_smooth` has no `sample_weight` parameter, unlike its flat sibling `nadaraya_watson_smooth`, despite the module's own motivating story being about composing kernel proximity with a recency/analog weight. |
| F13 | P2 | dead code | pysr_operators.py:31-60, 63-100 | 4 of 7 defined custom operators (`gauss`, `softplus`, `harmonic_mean`, `xlogy`) are never referenced by any of the 3 presets, yet their sympy mappings are unconditionally built every call. |

### F1 -- holiday_calendar_features.py: is_holiday/is_eve silently wrong for timestamped (non-midnight) dates

`_single_country_arrays` (holiday_calendar_features.py:82-85) computes `is_holiday = np.isin(dates_np, holiday_dates)` and `is_eve = np.isin(dates_np, eve_dates)` where `holiday_dates` come from `pd.Timestamp(date_object)` (always midnight). `dates_np` is `pd.to_datetime(dates).to_numpy()` with **no normalization** applied anywhere in the function or its caller `holiday_calendar_features`. If a caller passes a datetime column with a time-of-day component -- extremely common for the function's own stated use case (per-transaction / per-event timestamps, e.g. "Christmas Eve sales ~70x median") -- every row's exact-equality comparison against a midnight timestamp fails, so `is_holiday`/`is_eve` are silently **always False**, with zero error or warning (contrast with the function's proactive `_warn_on_mixed_tz` check for a different edge case). `days_since`/`days_until` still compute a (sub-day-precision, technically not exactly 0 on the holiday) continuous value, so only the two boolean flag columns are destroyed -- but those are the columns the module's own docstring leads with as the primary signal. Verified no existing test exercises this: `test_biz_val_holiday_calendar_features*.py` only use `pd.date_range(freq="D")` or plain date strings (both midnight). Suggested fix: `dates_dt = pd.to_datetime(dates).dt.normalize()` (or floor to day) before building `dates_np`, or explicitly document+enforce a date-only contract and raise/warn on any non-midnight input.

### F2 -- multi_window_aggregate.py: unconditional per-horizon full-frame copy that's never used

Inside `multi_window_aggregate`'s `for horizon in lookback_horizons:` loop (line 68), `windowed_history = history_df.copy()` (line 69) is executed on every iteration. `windowed_history` is subsequently passed unchanged (read-only) to two `leakage_safe_aggregate(...)` calls; it is never sliced, filtered, or mutated anywhere in the loop body. This means a full deep copy of `history_df` (potentially the largest input to this function -- the "history" table, per this repo's own 100+GB-frame convention) is paid `len(lookback_horizons)` times for zero functional benefit. Suggested fix: drop the copy entirely and pass `history_df` directly to both `leakage_safe_aggregate` calls.

### F3 -- grouped.py: per_group_apply's broad except Exception hides caller bugs

`per_group_apply` (grouped.py:259-332) is the shared primitive underlying several higher-level per-group feature helpers. Its per-group loop wraps the caller-supplied `fn(seg)` in `try: ... except Exception as err: logger.warning(...); continue` (lines 316-324). This is distinct from the function's documented "return `None` to skip" contract -- it additionally catches every *unexpected* exception (a typo, a wrong-shape return, a divide-by-zero the caller didn't anticipate) and converts it into a per-group log line plus silent `fill_value` fallback, rather than propagating. If `fn` has a systematic bug (wrong signature, always raises), the result is a feature column that is uniformly `fill_value` (e.g. all-NaN) across every group, with the only symptom being a WARNING logged once per group (easy to lose in output at scale, e.g. 100k+ groups) -- the exact "error silently swallowed into wrong downstream behavior" pattern this same codebase's MRMR audit found repeatedly. Suggested fix: narrow the catch to a documented, genuinely-recoverable exception set (or add a `raise_on_error: bool = True` opt-out), and/or escalate to `logger.exception` plus a hard failure after N consecutive group failures.

### F4 -- anchor.py: add_anchor_extrapolation_features has no numba path and redundantly recomputes its regression every row

All 4 sibling anchor-feature functions (`anchor_residual_rmse_features`, `anchor_quadratic_extrapolation_features`, `anchor_ewm_features`, `anchor_density_features`) dispatch to an `@numba.njit` core when numba is available (`_NUMBA_AVAILABLE` gate). `add_anchor_extrapolation_features` -- arguably the base/most commonly used member of the family -- has no such core; it always runs the pure-Python `_anchor_features_for_segment` (anchor.py:227-289). Within that function, the local-slope OLS fit over the last `K_slope` anchors (lines 268-276) is recomputed from scratch on **every row of the segment**, not just when a new anchor arrives -- i.e. O(n*K) work where the anchor set (and hence the slope) is actually constant between consecutive anchors, so O(A*K) (A = anchor count) would suffice, exactly as the sibling `_anchor_rmse_core`/`_anchor_ewm_core` numba kernels already do (the EWM core is explicitly O(1)-per-row via a running-accumulator recurrence). On a long gap between anchors this wastes real, easily-avoidable work. Suggested fix: add a numba core for this function following the same pattern as its siblings, and/or cache the slope and only recompute it when a new anchor is appended to the window.

### F5 -- __init__.py: large fraction of imported public functions missing from `__all__`

`__init__.py` imports many names it never lists in `__all__`, e.g. from `.grouped`: `per_group_rank`, `per_group_shift`, `per_group_cum_reduce`, `per_group_rolling_reduce`, `per_group_nth` (only `per_group_apply`/`per_group_sliding_window`/`iter_group_segments` make `__all__`); from `.anchor`: only `add_anchor_extrapolation_features` makes `__all__`, while `anchor_density_features`, `anchor_ewm_features`, `anchor_quadratic_extrapolation_features`, `anchor_residual_rmse_features`, `rows_until_next_anchor` do not; from `.spatial`: only `knn_aggregate`/`knn_within_bucket_aggregate` make it, while `inverse_distance_weighted_aggregate`, `knn_gradient_features`, `knn_label_dispersion_features`, `local_density_features`, `radius_aggregate` do not; from `.hurst`: `multi_scale_hurst`, `dfa_alpha2_quadratic`, `multifractal_dfa` are missing. The functions remain reachable via `mlframe.feature_engineering.<name>` attribute access (the import alone binds them), but `from mlframe.feature_engineering import *` and any `__all__`-driven tooling (doc generators, IDE surface discovery) silently under-report the real public API. Suggested fix: a small script asserting every name imported at module scope in `__init__.py` (that isn't prefixed `_`) is also present in `__all__`, run as a lightweight CI check.

### F6 -- relational_dfs.py: stack_relational_chain / RelationalHop unreachable from the package namespace

`relational_dfs.py`'s own module docstring frames `stack_relational_chain` as the generalized primitive that `stack_relational_features` (depth-2) now delegates to "so the two code paths are provably identical" -- yet `__init__.py:223` only imports `ChildTableSpec, compute_relational_features, stack_relational_features`, never `stack_relational_chain` or `RelationalHop`. Both are exercised in `tests/feature_engineering/test_biz_val_relational_dfs.py` via a direct `from mlframe.feature_engineering.relational_dfs import ...`, so the code itself is tested -- this is purely a package-surface omission, not an untested-code gap. Suggested fix: add both names to the `__init__.py` import block and `__all__`.

### F7 -- categorical_group_concat.py: discover_categorical_groups / auto_concat_categorical_groups unreachable from the package namespace

Same pattern as F6: the module's docstring describes `discover_categorical_groups`/`auto_concat_categorical_groups` as a first-class "Extension" of the base concatenator, and they are tested directly (`tests/feature_engineering/test_biz_val_categorical_group_concat.py`), but `__init__.py:228` only imports `concat_categorical_group`. Suggested fix: add both to `__init__.py`.

### F8 -- entity_diff_features.py: unnecessary deep copy vs the established shallow-copy convention

`entity_diff_features` (entity_diff_features.py:61) does `out = df.copy()` (default `deep=True`), then only ever appends new columns (`out[col_name] = ...`) in the loop below -- never mutates an existing column in place. Every other file in this cluster implementing the identical "copy df, then append new columns" shape uses `df.copy(deep=False)` explicitly with a comment explaining why a shallow copy is safe (e.g. `sentinel_missing_count.py:173`, `categorical_group_concat.py:56`/`186`, `categorical_powerset_concat.py:84`). This file is the one outlier doing a full deep copy of what can be a large panel/longitudinal frame (the module's own docstring cites "2604 diff features" on Amex-scale data). Suggested fix: switch to `df.copy(deep=False)` to match the established pattern.

### F9 -- relational_dfs.py: redundant upfront copy before the first pd.concat

`compute_relational_features` (relational_dfs.py:78) starts with `result = parent_df.copy()`, then inside the `for spec in child_specs:` loop immediately reassigns `result = pd.concat([result, agg], axis=1)` (line 91) on the very first iteration whenever `child_specs` is non-empty. `pd.concat` always allocates a fresh object rather than mutating its inputs, so the initial `.copy()` is redundant work in the common case (only the empty-`child_specs` return path actually needs it, to avoid returning a live alias to the caller's frame). Suggested fix: pass `parent_df` directly into the loop and only materialize a defensive copy on the `if not child_specs: return parent_df.copy()` early-return path.

### F10 -- "Wave NN (date)" process markers in code comments

CLAUDE.md states explicitly: "No process/audit metadata in code comments: no phase/wave markers, finding IDs, date stamps ... that belongs in git history / the PR description." `timeseries.py` has 4 such markers (lines 169-173 "Wave 96 (2026-05-21)", 467-468 "Wave 39 (2026-05-20)", 546-548 and 589-592 both "Wave 69 (2026-05-20)"), and `_numerical_counts.py` has 2 more (line 140 "Wave 58 (2026-05-20)", line 175 "Wave 21 P1"). Suggested fix: rewrite each comment to state only the WHY (the invariant/edge-case being handled), dropping the wave number and date stamp.

### F11 -- timeseries.py: dead/no-op if/else branch

In `create_and_process_windows` (timeseries.py:658-664):
```python
if forward_direction:
    windows_l = base_point
    windows_r = base_point  # initialised so the variable always exists in the else-branch
else:
    windows_l = base_point
    windows_r = base_point
```
Both branches perform the identical two assignments, so the `if forward_direction` test has no effect here (the actual forward/backward asymmetry is handled later in the function, inside the `for window_size in windows_lengths:` loop). The inline comment ("initialised so the variable always exists in the else-branch") is now stale/misleading since the if-branch does the same thing. Suggested fix: collapse to a single unconditional `windows_l = windows_r = base_point` before the loop.

### F12 -- nadaraya_watson.py: per-group variant missing sample_weight

The module docstring's central motivation is that NW is "a *query-dependent* weighted mean" and that `nadaraya_watson_smooth` "accepts an optional per-sample `sample_weight` that MULTIPLIES the kernel weight, letting a caller compose kernel proximity with a recency / seasonal-analog weight." `per_group_nadaraya_watson_smooth` (nadaraya_watson.py:201-240) -- the per-entity variant, arguably the more commonly useful one for panel data (e.g. "denoise each road-arc's speed over time", per its own docstring) -- has no `sample_weight` parameter at all; its numba core `_nw_per_group` always calls `_nw_at(..., w_dummy, ..., False)` (`use_w` hardcoded False). Suggested fix: thread an optional per-row `sample_weight` through `per_group_nadaraya_watson_smooth`/`_nw_per_group` matching the flat function's contract.

### F13 -- pysr_operators.py: unused custom-operator definitions

`OPERATOR_JULIA_SIGNATURES` defines 7 custom Julia operators, but `_operators_for_preset` (the only place operators are wired into a preset's `binary`/`unary` lists) only ever references `safe_log`, `safe_sqrt`, and `inv` across all 3 presets. `gauss`, `softplus`, `harmonic_mean`, `xlogy` are fully implemented (Julia signature + sympy mapping) but never selectable through `get_preset_kwargs`, and `_make_extra_sympy_mappings()` (line 42) unconditionally builds sympy mappings for all 7 regardless of which preset (or none of them) is in use. Not a functional bug (unused `extra_sympy_mappings` entries are harmless to PySR), but dead/orphaned code relative to the module's own documented "3 presets curated for tabular FE" scope. Suggested fix: either wire the 4 unused operators into a preset (if intentional future work) or remove them, and gate `_make_extra_sympy_mappings()`'s dict to only the operators the requested preset actually uses.

## Proposals

| ID | Category | File | Summary |
|----|----------|------|---------|
| PR1 | test-gap | holiday_calendar_features.py | Add a regression test with intraday timestamps (e.g. `pd.Timestamp("2024-12-25 14:30:00")`) pinning `is_holiday`/`is_eve` correctness -- would have caught F1 and guards the fix. |
| PR2 | architecture | grouped.py, timeseries.py | Both files sit at/above this repo's own ~900-1000 LOC split threshold (972 and 897 LOC respectively) with no sibling split yet (unlike `spatial.py`/`anchor.py`/`hurst.py`, still comfortably under). Worth a proactive split before the next feature lands, per CLAUDE.md's own "carve before ~800-900 LOC" guidance. |
| PR3 | ML best practice | multi_decomposition_bank.py | `auto_k`'s ICA/GRP/SRP elbow heuristic (`_select_k_by_reconstruction_elbow`) and `prune_uninformative_methods`'s AUC-based pruning are both good opt-in additions, but neither has a `test_biz_val_*` demonstrating the auto-selected k (or pruning decision) actually beats a fixed-k/no-pruning baseline on a synthetic with known true rank -- worth adding given the project's "every ML trick gets a quantitative biz_value test" convention. |
| PR4 | perf | anchor.py | Beyond F4, the whole anchor-feature family recomputes `iter_group_segments` (an O(n log n) sort) independently per function call; a caller wanting several anchor features on the same `(label, is_anchor, group_ids)` triple currently pays the sort N times. A combined "compute-all" entry point (or accepting a precomputed `(sort_idx, starts, ends)`) would amortize it, mirroring `per_group_recency_weighted_agg`'s `params` batching pattern elsewhere in this cluster. |
| PR5 | robustness | multi_window_aggregate.py | `_direct_window_agg`'s non-additive-agg fallback path (median/min/max/...) rebuilds a full `{entity: group}` dict via `history_df.groupby(entity_col, sort=False)` per horizon per (col, fn) combination that needs it -- for a caller mixing several non-additive aggs across several horizons this dict is rebuilt redundantly; hoisting it once per call (keyed only by `entity_col`) would avoid the repeated groupby. |
| PR6 | docs | tfidf_svd_entity_embedding.py, gmm_bic_membership_features.py | Both modules' "opt-in diagnostic" features (`FittedTfidfSvdEntityEmbedding.transform_new_entities`'s OOV fraction; `gmm_bic_membership_features`'s `gmm_shift_diagnostics` in `.attrs`) are good design but undocumented anywhere outside their own docstrings (no README/docs cross-reference) -- worth a short mention in the package-level `__init__.py` module docstring's one-line submodule index, which currently only describes the primary return value of each submodule. |

## Coverage notes

- `_timeseries_emit.py` (the sibling module `timeseries.py` re-exports 11 `_emit_*` helpers from) is not in my assigned 39-file list and was intentionally left un-audited beyond confirming the re-export wiring is sane; its own internals (the bulk of `create_aggregated_features`'s actual per-transform logic) belong to whichever cluster owns that file.
- No GPU/CPU-parity findings apply to this cluster: none of the 39 files contain a CUDA/cupy branch (confirmed by reading every file; the only GPU-adjacent comment, in `spatial.py`'s `knn_gradient_features` docstring, explains why a GPU port was deliberately NOT added).
- I did not execute any test file or benchmark (per the read-only mandate); all "verified no test covers X" statements (F1, and the F6/F7 "tested via direct import" notes) are based on reading the relevant test file's source directly (`test_biz_val_holiday_calendar_features*.py`, `test_biz_val_relational_dfs.py`, `test_biz_val_categorical_group_concat.py`), not on running them.
- I did not exhaustively cross-check every one of the ~39 files' functions against the full `tests/feature_engineering/` directory (it is large and a broad `Glob`/`Grep` over it repeatedly timed out in this environment); the spot-checks I did run (holiday_calendar_features, grouped, anchor, relational_dfs, categorical_group_concat, multi_window_aggregate, entity_diff_features, nadaraya_watson, pysr_operators) all had at least one corresponding test file, so I have no basis to claim a broader test-coverage gap beyond what's stated in the Findings/Proposals above.

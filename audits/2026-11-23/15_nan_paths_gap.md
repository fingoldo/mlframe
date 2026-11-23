# Audit 15: MRMR all-NaN, constant and missing-column fit paths (gap audit, Haiku, read-only, nothing executed)

Scope: `_missingness_fe.py`, `_mrmr_degenerate.py`, `_fe_stage_cascade_run.py`, `_fe_stage_cascade_early_b.py`, `_fit_impl_stages/_prelude.py`, `discretization/__init__.py`, `_discretization_dataset.py`, `_candidate_pool.py`, `_assign_support_tail.py`. Not traced: `local_mi_gate` and the relevance gate (findings that depend on them are hypotheses); `_grouped_*` and `_degenerate_combo_gpu.py` not read.

Tracker rows: AUD23-55 to AUD23-59.

## Finding 1: inf treated as missing at fit, not at transform. Medium. Confirmed.
- Fit: `_fit_impl_stages/_prelude.py:54-61` records `~np.isfinite(...)` as the NaN mask, so +inf and -inf count as missing.
- `_fe_stage_cascade_early_b.py:336-339` restores those positions as NaN in `_miss_frame()` for any column with no NaN; `missing_indicator_fit` (`_missingness_fe.py:187`) builds `is_missing__col` from that frame.
- Replay: `apply_missing_indicator` (`_missingness_fe.py:201`) uses `isna()`, False for inf.
- Input: a column with about 2% +inf and no NaN. Training indicator is 1 on those rows; on a clean transform frame the recipe returns all zeros, so the feature is constant at transform.

## Finding 2: explicit indicator columns bypass the all-NaN exclusion. Medium-low. Emission confirmed; drop is hypothesis.
- `_fe_stage_cascade_early_b.py:368-372`: with `fe_missingness_indicator_cols`, `_resolve_missing_cols` checks only membership in `X.columns` and `_engineered_seen_l37`; it does not consult `degenerate_columns_`, and the NaN-rate window applies only to auto-detect.
- `missing_indicator_fit` (`_missingness_fe.py:186-188`) has no all-NaN check; an all-NaN column yields an indicator constant 1 at fit, recipe emitted (`:436-442`).
- Default `fe_local_mi_gate=True` (`_mrmr_class.py:2635`) should drop a zero-MI indicator. Not confirmed that the gate removes it; if it survives, it replays as a value that varies at transform.

## Finding 3: count and pattern replay silently drop absent columns. Low-medium. Confirmed.
- `apply_missingness_count` (`_missingness_fe.py:247-250`) filters to `cols_present` and counts absent columns as 0; `apply_missingness_pattern` (`:356-359`) fills absent columns with all-False.
- Input: a fit-time source column missing at transform. Count undercounts; pattern labels shift toward "other". No error or warning: wrong output with no signal.

## Finding 4: non-finite columns bypass candidate judgment. Low. Confirmed.
- `_mrmr_fit_impl/_candidate_pool.py:63-66`: any column with a non-finite value, including all-NaN, goes into `pool` unjudged. Line 60 returns every numeric column unfiltered below 200 usable rows.
- Pre-screening does not remove all-NaN columns; they rely on the downstream MI gate, not traced (hypothesis on whether they are dropped).

## Finding 5: row count 0 or 1. Confirmed for the missingness path; hypothesis for the rest.
- `auto_detect_missing_cols` (`_missingness_fe.py:137-148`) returns `[]` silently for 0 rows; for 1 row the NaN rate is 0 or 1, outside `[0.01, 0.99]`, so also `[]` with no message.
- `missing_indicator_fit` raises `ValueError("X is empty")`, but the auto path returns before reaching it, so that error is unreachable from `MRMR.fit`.
- NaN in `y` is rejected (`_mrmr_class_fit_stages.py:188`); no minimum-row guard found in the MRMR fit path.
- Hypothesis: with 0 or 1 rows, MI is zero for every column and the fit selects nothing without raising; MI path not traced.

## Checked, not a finding
- `_handle_missing` with all-NaN columns (`discretization/__init__.py:116-124`) fills with 0 and `_assign_nan_bin_codes` re-routes them to the NaN bin; no crash.
- `_is_constant` and `_is_all_nan` in `_mrmr_degenerate.py` correctly separate the cases.

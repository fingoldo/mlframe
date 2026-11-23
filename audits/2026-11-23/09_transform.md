# Audit 09: MRMR transform and recipe replay (wave 3)

Scope: `MRMR.transform`, `fit_transform`, `get_feature_names_out`, `engineered_recipes/`, the code applying stored recipes to new data. Read-only, Haiku model, nothing executed. Items are code-reading results: confirmed (traced in code) or hypothesis (needs a fit-time check or run).

Tracker rows: AUD23-07, 08, 09, 10, 11 (categorical overlap with audit 10); and the checked-items below.

## Fitted state: checked and mostly sound
- `freeze_orth_fit_stats` (`engineered_recipes/_orth_basis_recipes.py:774-801`) freezes `nan_fill` and `pre_sd`, called at fit (`_mrmr_class_fit_stages.py:517`). Confirmed.
- `log_shift_*` and `gate_med_*_median` stamped at fit (`_recipe_unary_binary.py:390, 406-408`); the batch-dependent fallback (`:127-128`) fires only on legacy pickles. Confirmed.
- Target-encoding, count and frequency lookups fall back to a global default for unseen keys (`_encoding_recipes.py:163-164`, `_ratio_delta_fe.py:132-134`). Confirmed.
- `support_` always int64 (`_finalise.py:271`), so the pandas name path (`_mrmr_validate_transform.py:556`) is safe. Confirmed.
- Column drift detected for pandas and polars (`_mrmr_validate_transform.py:561-572, 591-600`). `ENSURE_ARROW_DF_SUPPORT` is True (`_internals.py:168`), so the positional pandas branch at 577 is dead by default. Confirmed.

## Findings
1. Silent all-zero codes for non-integer categorical input (High if reachable). `engineered_recipes/_recipe_extract.py:236-237`: a string or object source with no `cat_code_map` whose values do not cast to int gives every row code 0 unless `unknown_strategy="raise"`. Trigger: legacy pickle or recipe missing `cat_code_maps`, plus string values at transform. Result: a constant feature, no error. Confirmed in code; whether current fits omit the map is hypothesis.
2. Unseen category aliases a real code (Medium, hypothesis). `_recipe_extract.py:155` sends an unseen category to `n_bins - 1` ("sentinel bin"). If fit never reserved the slot, the last real category's encoding is reused. Confirmed as code; reservation unverified.
3. Numeric source without stored edges used as a raw code (Medium, hypothesis). `_recipe_extract.py:204-222`: with no `bin_edges` and no `cat_code_map`, a float is rounded and used directly as the code; factorize clips to the top bin. Depends on whether fit always writes `src_bin_edges` for factorize sources; not checked.
4. Positional ndarray input with reordered columns (Medium, inherent). `_mrmr_validate_transform.py:611-614` and `_recipe_replay_frame` (`:844-846`) assume an ndarray is in fit order. Width is checked (`:486-496`), order is not (no names). A permuted ndarray gives wrong base columns and recipe inputs without error. The docstring should say so.
5. Output container changes with `usability_aware_lists` (Medium, confirmed). `_mrmr_class.py:3644-3645` calls `_append_usability_union`; for an ndarray `base_out` it returns a pandas DataFrame (`_mrmr_validate_transform.py:204-209, 223`). Without the usability lists the same input returns an ndarray. Pandas-output conversion (`:617`) is skipped on this path.
6. Integer base columns promoted to float on ndarray input (Low, confirmed). `_mrmr_validate_transform.py:668-674` promotes the whole output to a common dtype; pandas and polars keep per-column dtypes. Values unchanged below 2^53.
7. Chained recipe with a pruned producer emits constant 0 (documented, Low). `_mrmr_validate_transform.py:827`, docstring `:909-913`. Intentional.

## Not examined
Constant or all-NaN columns at fit, and whether their recipes are dropped or replayed. `_missingness_fe` and `_grouped_*` replays for all-NaN fit inputs not traced.

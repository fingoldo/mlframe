# Audit 10: MRMR categorical handling (wave 3)

Scope: categorical encoding, target_stat fold, categorical paths of discretization, `_validate_categorical_fold`, cat-FE. Read-only, Haiku model, nothing executed. Findings A and B confirmed by reading the code path; C confirmed by reading; none run.

Tracker rows: AUD23-07 (= A), AUD23-08 (= B), AUD23-44 (= C), AUD23-45 (= target_stat leak, rejected as defect), AUD23-46 (NaN block shift).

## Finding A: 1 and "1" collapse in the cat-FE code map (requirement 3). High for cat-FE recipes on mixed-type object columns. Confirmed.
- `engineered_recipes/_recipe_extract.py:108`: `base = {str(u): ...}` from `pd.factorize` uniques. Factorize keeps `1` and `"1"` distinct, but `str()` maps both to `"1"`; the later overwrites the earlier.
- Input: object column `[1, "1", 2]`. Uniques `[1, "1", 2]`, codes 0,1,2; map becomes `{"1": 1, "2": 2}`. At transform `1` maps to code 1 instead of 0.
- Same collapse is deliberate in `_internals.py:237-261` (`canonical_group_token`): `1` and `"1"` both return `"1"`. This feeds `_target_encoding_fe.py:350-434` (`_column_to_str`), so target-encoding, count and frequency keys also merge int 1 with string "1". The comments show it is deliberate for int/float drift, but it is not limited to numeric drift. Medium.

## Finding B: min_support folding not reproduced at transform (requirement 4). High for cat-FE recipes with a thin level under `categorical_fold_strategy="min_support"`. Confirmed.
- `discretization/_categorical_encoding.py:89-99` folds thin levels into bucket `len(kept)`. `_recipe_extract.py:62-124` builds the transform map from every raw unique before any fold, in factorize order.
- Input: `["c", "c", "a"*10, "b"*10]`, `min_support=5`. Factorize: c=0, a=1, b=2. Fold keeps a->0, b->1, folds c into bucket 2. Transform map still says c->0, a->1, b->2.
- Result: a transform-time "c" gets code 0, the fit code of "a"; every level's code shifts in this case.
- Not checked: whether base categorical columns (not cat-FE) have a separate transform path applying the fold.

## Finding C: target_stat null does not refold boolean columns (requirement 2). Medium. Confirmed by reading.
- `_mrmr_fit_impl/_fit_impl_stages/_binning.py:70` and `:74` classify "categorical" as `not is_numeric_dtype(x[c])`. Pandas reports bool as numeric, so a bool column is never target-dependent.
- `_binning.py:292` keeps observed codes under every permuted target, but `_categorical_target_fold.py` can merge a two-level bool column using the real target, so its codes change under a permuted target.
- Effect: the permutation null for a bool source sees codes fitted on the true target, so the null is optimistic for that column. Size of the bias not measured.
- Object and category columns are refolded in the honest null (`_binning.py:315` passes `categorical_fold_target=_fold_target(self, y_perm)`).

## Requirement 2, leak in the screen (by design, confirmed, informational)
- `_binning.py:485-492` builds the target_stat fold from all rows (`_fold_codes`), as `_categorical_target_fold.py:8` documents. In-sample by design, partly bounded by the BIC threshold (`_categorical_target_fold.py:137`), but optimism remains. The honest null is the only correction.

## Requirement 1: NaN handling (no defect found)
- Fit: `_discretization_dataset.py:701-713` applies the +1 shift to the whole categorical block when any NaN exists; `_recipe_extract.py:73-92` mirrors it. `_stamp_cat_code_maps` (`_cat_fe.py:23-27`) computes the block flag once.
- Transform: `_NAN_CODE_KEY` stored only for columns owning a NaN; NaN not a base-map key. Consistent provided the block-level flag is the same at fit and transform.
- Not verified (hypothesis): a transform frame with NaN in a column that had none at fit; the block-level shift then comes from the transform frame.

## Not verified
- `PermutedTargetBinner` (fast null path, `_binning.py:267-280`) not located in `discretization/shared.py`; whether it refolds target_stat is unconfirmed.
- `_validate_categorical_fold` (`_mrmr_validate_strings.py:138-142`) checks only min_support >= 2 and int; correct as far as it goes.

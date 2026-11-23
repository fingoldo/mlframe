# Audit 02: MRMR discretization and binning (wave 1)

Scope: edge builders, code dtypes, NaN handling, honest-null binner, memo keys. Read-only, Haiku model, nothing executed.
Coverage gap (found by the auditor, and it was partly its own error): the auditor wrote that `_binning.py` does not exist. It does: `_mrmr_fit_impl/_fit_impl_stages/_binning.py`. Not read: `supervised_binning.py`, `_mdlp_validated_split.py`, `_code_storage_plan.py`, `_discretization_cuda.py`, the categorical files. `_discretization_dataset.py` was reviewed only for the NaN-bin and dtype helpers.

Tracker rows: AUD23-12, AUD23-13 (confirmed); AUD23-46 (NaN shift, cross-listed with categorical).

## Confirmed (full path read)

1. Knuth quantile edges are not de-duplicated (phantom or empty bins). Severity medium, conditional on the non-default knob.
   - `_discretization_edges.py:145-146`, reached via `_adaptive_nbins.py` `edges_knuth` (line 288) when `knuth_edge_type="quantile"`. Unlike `_edges_from_quantiles` (`_adaptive_nbins.py:178`), no `np.unique`.
   - Example: `x = [0]*90 + [1..10]`, M=5 gives percentiles `[0,0,0,0,0,10]`, inner edges `[0,0,0,0]`. With `searchsorted(side="right")` every value >= 0 gets code 4, so the column collapses to one occupied bin. `_sparse_dominance_edges` only fires when edges have size <= 1, so it does not catch this. The M value is illustrative; the collapse happens for any M >= 5.

2. `_safe_code_dtype` boundary with int8 (fails loudly, not silently). `_code_dtype.py:31-33` keeps int8 for `n_bins=128`. The legacy NaN code is 128 (`_discretization_dataset.py:295`), so `_assign_nan_bin_codes` (line 297) raises. Severity low.

3. Uniform path returns NaN edges for an all-NaN column; quantile path returns a single bin. `_discretization_edges.py:131-146` vs `:395-408`; `core/arrays.py:33-34` returns (nan, nan). Codes for NaN rows become `n_bins-1` (uniform) vs 0 (quantile). Both give one occupied bin, so MI is unaffected; the two strategies disagree. Severity low.

4. Memo keys in `_adaptive_nbins_columns.py` (no collision found). `_edge_memo_key` (282-293) keys on content hash (`_fe_resident_operands.py:100-111`), dtype, shape. `_FALLBACK_EDGES` (330-345) keyed on `id(uniq)` but holds a reference and checks `hit[0] is uniq`, so safe. Theoretical: kwargs values `1`, `1.0`, `True` compare equal; harmless.

## Suspected (not fully traced)

5. Fixed-column codes not range-checked: `_null_binner.py:85-86`, `_add_row` (185-200) write codes into `self._dtype` with no bound check. Depends on caller contract.
6. `get_binning_edges` quantile path does not de-duplicate (`_discretization_edges.py:409-410`), same class as item 1, reached from `discretize_sklearn` (428-429). Not checked whether `discretize_2d_array` de-duplicates downstream.
7. Knuth "BIT-IDENTICAL" claim. `_discretization_edges.py:84-91` computes edges as `a_min + (j+1)*width`, `np.histogram` uses `linspace`; can differ by one ulp, moving a value sitting on an edge. Docstring claim unverified. Low.
8. Bayesian Blocks forced median split (`_discretization_edges.py:309-312`): when median equals min or max, `np.unique` collapses the split to zero inner edges. Low.

## Checked, no train/apply skew found
- Fit NaN path: `_handle_missing` median-fills (`discretization/__init__.py:116-130`); NaN rows get code `per_col_bins[c]` (`_discretization_dataset.py:287, 305`).
- Honest OOS null uses the same NaN convention (`_mdlp_oos_honest_njit.py:297-302`, `_mdlp_honest_bycol_njit.py:131-148`).
- `PermutedTargetBinner._slot` (`_null_binner.py:161-183`) matches the fit path.
- int8 guards (`_mdlp_honest_bycol_njit.py:36,74,83`) are conservative.

## Not findings
- `_knuth_best_M` distinct-value cap and the `n<2` / `a_max<=a_min` guards are correct.
- `_percentiles_like_numpy` boundary handling at `lo >= n-1` is correct.

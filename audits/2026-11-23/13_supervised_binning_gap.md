# Audit 13: MRMR supervised binning (gap audit, Haiku, read-only, nothing executed)

Scope: `supervised_binning.py` (fast and classic path, best-split kernels at 398-579), `_mdlp_validated_split.py`, `_honest_spec.py`, `_binning.py` depth cap, `_adaptive_nbins_columns.py` pseudo-class count. The `_mdlp_best_split*.py` glob matched no files; the kernels are inside `supervised_binning.py`. Not read: the consumer `_mdlp_honest_njit` (so finding 5's effect on the null is a hypothesis).

Tracker rows: AUD23-60 to AUD23-64.

## Finding 1: continuous target with 2 to 64 distinct non-integer values is truncated to int. High. Confirmed.
- `supervised_binning.py:87-89`: quantisation runs only above `max_y_classes` distinct values; otherwise the float target passes through.
- `supervised_binning.py:138`: `_drop_nonfinite_xy` does `y_arr.astype(np.int64)`; `_mdlp_validated_split.py:689`: `y_i = _y_arr.astype(np.int64)` in the validated path.
- Input: y in {0.1, ..., 0.9} (9 distinct floats): every label becomes 0, one class, no split, one bin, no warning. y in {1.0, 1.5, 2.0} gives labels {1, 1, 2}, merging 1.0 and 1.5.
- The honest-null spec inherits it through `_encode_mdlp_target` and `_honest_spec.py:169`. Affects both fast and validated paths; the only place a target can come out degenerate.

## Finding 2: tie cuts and size floors. Low. Tie cuts confirmed absent; floor semantics differ; midpoint rounding is hypothesis.
- Tie cuts blocked: `x_sorted[i] == x_sorted[i+1]` skipped at `supervised_binning.py:508`, `:570`, `:435`.
- Both-side floors enforced at `:502`, `:566`, `:431` and in the holdout check `_mdlp_validated_split.py:732`.
- `min_split_size` counts de-duplicated (x, y) pairs (`_dedupe_xy`, `_mdlp_validated_split.py:692`, `supervised_binning.py:176`), not rows: a side can hold many more raw rows than the floor implies. Not an empty bin.
- Hypothesis: the midpoint `0.5*(a+b)` (`:641`, `_mdlp_validated_split.py:581`, `:799`) can round to `b` for adjacent floats; the holdout partition `x_holdout <= best_split` then sends rows at `b` left and could empty a holdout side. Not traced to a concrete input.

## Finding 3: silent depth cap, fewer bins than the target. Medium. Confirmed.
- Fast path `supervised_binning.py:105-106` caps `max_depth` to `ceil(log2(64)) = 6`: at most 64 bins, no log.
- Default path `_mrmr_fit_impl/_fit_impl_stages/_binning.py:411-415` caps `max_depth` to `floor(log2(max_categorical_cardinality))` for `mdlp` and `mdlp_validated`. `mrmr/_mrmr_class.py:375` default cardinality 127 gives depth 6, at most 64 intervals: the default MRMR configuration.
- No logger call for depth truncation in `supervised_binning.py` or `_mdlp_validated_split.py`.
- Trigger: a column whose signal supports more than 64 intervals under default MRMR yields 64 bins with no indication.

## Finding 4: target-class cap is order-independent; strategies disagree. Medium (disagreement). Order dependence not found.
- `np.quantile` sorts its input; `_encode_mdlp_target` memoised on content hash (`supervised_binning.py:43-61`): row order does not change labels.
- Low: quantile edges are computed on all finite y, including rows whose x is later dropped (`:88-93` before `:132-138`): edges depend on which rows are dropped, not on order.
- Disagreement: the same continuous target gets 16 pseudo-classes under default `mdlp` (`supervised_binning.py:89`, `_adaptive_nbins_columns.py:78`, default 16) and 64 under `mdlp_validated` (`_mdlp_validated_split.py:674`, `:902`; `_binning.py:26-38`). The result depends on the strategy name.

## Finding 5: fast_mode and the validated path disagree with no logged reason. Medium. Spec-level confirmed; effect on null is hypothesis.
- `mdlp_bin_edges_validated` has no `scaled_min_split` parameter; the default path applies it at `supervised_binning.py:341-342`. The same kwargs give different floors on the two entry points.
- `_honest_spec.py:162` hard-codes `fast_mode=False`, `max_y_classes=64`, `y_pseudo_classes=16` regardless of the caller. Trigger: `mdlp_fast_mode=True` with `max_depth=8`: production caps depth to 6, but the spec carries the uncapped depth (`:189`).
- `_honest_spec.py:190-196` hard-codes `min_split_size=5`, `val_min_split_size=5`, `alpha=0.05`, `n_permutations=15`, `seed=0`. `build_honest_spec` has no parameter for them, so `mdlp_alpha`, `mdlp_n_permutations`, `mdlp_validated_seed`, `scaled_min_split` never reach the honest null. Confirmed at spec level; whether `_mdlp_honest_njit` changes results from these values is hypothesis (consumer not read).

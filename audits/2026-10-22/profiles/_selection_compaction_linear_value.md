# Selection compaction: redundant in the binned information is not redundant for a linear model

`run_selection_compaction` (the last support-shrinking pass) has two moves, both judged only by the bias-corrected BINNED joint information of the selection:
`prune_redundant` drops an engineered feature whose removal leaves that information unchanged, `choose_compaction` replaces several engineered features by one
pool candidate that carries at least the same information. For a transform of a selected source both are exact by the data-processing inequality: with
`y = x**2 + 0.2 x`, `y` is a function of the quantile bins of `x`, so `x__pade2` (the x^2 basis a LINEAR downstream model needs) adds nothing to `x` and goes.

Provenance (`fe_provenance_`, `_predictors_log_`) of `test_default_recovers_quadratic_regression[42, 7]`, failing since before 2026-11-04: the greedy selected
`x__mod_7` (gain 0.954, rank 0), the final support was `['x']`; on the outer pass the compaction took the recipes `x__cos1`, `x__pade2` to 0 (prune) and, once
the prune was linear-aware, replaced both by the pool's `x__relu_gt0.0496404` (replacement; R^2 0.837 against the 0.85 the test pins).

Both moves now consult `_compaction_linear_value.make_linear_value_check` (3-fold held-out CV on at most 3000 rows, the retention's scorers: log-loss of a scaled
logistic model for a classification target, MAE of OLS otherwise; margin 1% like the greedy):
- `prune_redundant`: a feature that is information-redundant is still kept when removing it from the rest raises the CV loss by more than the margin;
- `choose_compaction`: a replacement is skipped when it raises the CV loss of `rest + candidate` over `rest + replaced` by more than the margin.
No continuous target (`_fe_prewarp_y_continuous_` unset) -> no check, the information-only rule as before. Cost: two small CV fits per droppable candidate.

Result: quadratic-regression recovery 5/5 (seeds 42 and 7 now), compaction tests 16/16, unit tests `test_compaction_linear_value.py`.

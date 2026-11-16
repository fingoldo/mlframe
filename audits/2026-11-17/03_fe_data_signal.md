# Audit 03: feature_engineering, data, data_valuation, signal, competition (MRMR excluded)

Scope note: a time-boxed pass. I grepped for broad excepts, additive-epsilon denominators and var/moment code, then read the hits. This was not a line-by-line read of all ~30k LOC.

| ID | file:line | finding | disposition | reason/test |
|----|-----------|---------|-------------|-------------|
| A1 | feature_engineering/ensemble_features.py:172 (`_entropy_from_counts`) | `log(p + 1e-12)` is an epsilon pad inside the log. A one-bin row gives entropy -1e-12 instead of 0, and every entropy is biased by about 1e-12. | RESOLVED | Replaced with `scipy.special.xlogy`. Empty rows still give NaN. Test: tests/test_ensemble_entropy_exact.py. |
| A2 | feature_engineering/ensemble_features.py:291 | MAD is padded with an absolute `+ 1e-12`. For predictors on a tiny scale the threshold is dominated by the pad. With MAD = 0 any deviation above 1.7e-12 counts as an outlier. | RESOLVED | Pad is now relative: `1e-12 * abs(median)`; MAD=0 flags any nonzero deviation on a zero-median row. Tests: tests/feature_engineering/test_mad_pad_relative_scale.py::test_mad_pad_relative_scale_flags_dissenter_on_tiny_scale_predictors, ::test_mad_pad_relative_scale_ordinary_scale_results_unchanged. |
| A3 | feature_engineering/anchor.py:262, 347 | The slope guard is `den > 1e-12` in absolute terms. If anchor positions are spread below about 1e-6 the slope silently becomes 0. The sibling branches at :213 and :707 already use a relative guard. | RESOLVED | Relative guard (`den > 1e-24 * sum(pos**2)`) in the numpy path (`_window_ols_slope`) and the njit kernel; test `test_tiny_spread_anchor_positions_give_true_slope` (tests/feature_engineering/test_anchor_relative_slope_guard.py). |
| A4 | feature_engineering/mps.py:695, 743; numerical.py:138, 436 | Broad `except Exception` blocks. | REJECTED | Each logs with traceback or at debug level and returns the documented sentinel. Nothing is hidden. |
| A5 | data_valuation/_mc_sampling.py:170; signal/dtw.py:617, 623 | Broad excepts around the GPU fallback. | REJECTED | The failure is logged and the CPU path runs. This is a deliberate fallback. |

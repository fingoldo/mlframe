# Wave 21 audit 2026-11-02 -- tracker

Five read-only reports by Opus 5.5 low-effort agents on areas no earlier wave read: the feature-engineering transformer subpackage and recency modules, `core` / `system` / the rest of `inference` / `competition` / the votenrank blends, `shap_proxied_fs` / `importance` / `forward_select` / `structure_discovery`, the discretization and GPU kernels, and the remaining `reporting` charts. The agents could not write files, so the reports are their returned text, condensed. Every claim was re-checked on HEAD before a fix. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Severity | Location |
|---|---|---|---|
| **RESOLVED** | W21FE-1 | High | transformer OOF buffers were zero-filled, so the first block of a `TimeSeriesSplit` read as a real 0.0; now NaN (the same zero-init pattern sits in `adaptive_bandwidth`, `anchor_attention`, `aux_mlp`, `class_conditional_anchor`, `class_mahalanobis`, `focal_lgb`, `ks_shift`: not traced, listed as FUTURE below) |
| **RESOLVED** | W21FE-1b | High | the seven sibling modules with zero-filled OOF outputs |
| **RESOLVED** | W21FE-2 | High | `per_group_recency_weighted_mean` / `_agg` gave every row its entity's whole-history value, future rows included; new `causal=True` default gives each row the aggregate over rows up to itself, `causal=False` keeps the one-shot value (`tests/feature_engineering/test_recency_aggregation_causal.py`) |
| **RESOLVED** | W21FE-3 | High | `per_group_recency_weighted_mode` / `_behavioral_stability` had the same future-row leak; same `causal` option, default True |
| **PARTIAL** | W21FE-4 | Medium | the rolling mean / std kernels and now the `recency_aggregation` mean/sum/min/max/std/var kernels (single and multi-decay) skip NaN (`tests/feature_engineering/test_recency_aggregation_causal.py`); the KDE kernels in `recency_density` still propagate it |
| **RESOLVED** | W21FE-5 | Medium | string / object group ids crashed `np.diff` in the recency rolling functions |
| **RESOLVED** | W21FE-6 | Low | `per_group_sliding_window` raised for a group shorter than the window |
| **RESOLVED** | W21CORE-1 | High | hard-mode logical constraints did one ordered pass, so a rule chain could end violated in either declaration order; passes now repeat to a fixpoint |
| **RESOLVED** | W21CORE-2 | Medium | `shapley_blend(coalition_blend='rank_mean')` learned weights on ranks but blended raw predictions; it now blends the per-row ranks (`tests/votenrank/test_shapley_blend_rank_mean_blends_ranks.py`) |
| **RESOLVED** | W21CORE-3 | Medium | `fastmath` removed from the `confidence_gated_blend` njit kernels, which assume no NaN; a divergence from the numpy path was not reproducible on this CPU, so the parity test over NaN / inf confidences and NaN inputs pins the behaviour rather than failing on the old code (`tests/votenrank/test_confidence_gated_blend_nan_confidence_parity.py`) |
| **RESOLVED** | W21CORE-4 | Medium | `explainability` OOF SHAP leaves `None` rows for never-tested samples |
| **RESOLVED** | W21CORE-5 | Low | `explainability` scored `oos_ts_max_size + 1` rows |
| **RESOLVED** | W21CORE-6 | Low | `inference/_ktc_dispatch` cache key omits `n_rules` |
| **RESOLVED** | W21CORE-7 | Low | `similarity_scale` was never validated |
| **FUTURE** | W21FS-1 | Medium | `discover_structure` p-values are conditional on the searched modulus / threshold / op; the null needs the same search per permutation |
| **RESOLVED** | W21FS-2 | Medium | `forward_select(patience=k)` kept the `patience-1` non-significant additions that preceded the stop |
| **RESOLVED** | W21FS-3 | Low | `plot_feature_importance(positive_fi_only=True)` still drew negative bars |
| **RESOLVED** | W21FS-4 | Low | the FI log line's `share=` used signed sums (1100%) |
| **FUTURE** | W21KRN-1 | Medium | GPU uniform binner emits undefined codes when the range is subnormal or infinite (CPU routes the NaN to bin 0) |
| **FUTURE** | W21KRN-2 | Low | `_accept_split_serial` reseeds the thread's numba RNG as a side effect |
| **RESOLVED** | W21REP-1 | Medium-High | slice finder divided supports by all rows although NaN-error rows were dropped |
| **RESOLVED** | W21REP-2 | Medium | multiclass P/R/F1 counted unknown true labels as false positives while the confusion panels dropped them |
| **FUTURE** | W21REP-3 | Medium | slice-finder "simultaneous" intervals use a Sidak count of only the degraded cells |
| **RESOLVED** | W21REP-4 | Low-Medium | multilabel Hamming histogram bins were equal-width over [min, max] and straddled the integer distances; it now draws one unit-wide bin per integer 0..L (`tests/reporting/test_multilabel_hamming_bins_are_integer_aligned.py`) |
| **RESOLVED** | W21REP-5 | Low | the quantile prediction-interval band subsampled all rows, so non-finite ones reached the line and band; it now draws from the finite rows only (`tests/reporting/test_interval_band_drops_non_finite_rows.py`) |

## Not covered by this wave

Most of `shap_proxied_fs/` (about 35 modules), `boruta_shap/__init__`, `wrappers/_helpers`; `transformer/` modules other than `_oof` / `_baseline_oof`; `timeseries.py`, `numerical.py`, `bayesian.py`; `core/` beyond `stats`, `competition/`, `system/` beyond `_policy`; `reporting` `shap_panels`, `error_analysis`, `_error_bias`, plotly renderers, `split_comparison`, `model_comparison`, `fairness_calibration`, `temporal`; the info-theory batch and class kernels, `_gpu_resident_*`, `_batch_mi_noise_gate_*`.

## Implemented

| ID | Regression test |
|---|---|
| W21FE-1 | `tests/feature_engineering/test_transformer_oof_uncovered_rows_are_nan.py` |
| W21FE-5, 6, W21CORE-5, 7 | `tests/test_wave21_small_fixes.py` |
| W21CORE-1 | `tests/inference/test_logical_constraints_chain_reaches_a_consistent_state.py` |
| W21FS-2 | `tests/feature_selection/test_forward_select_patience_drops_unconfirmed_additions.py` |
| W21FS-3, 4 | `tests/feature_selection/test_importance_signed_values_share_and_positive_only_chart.py` |
| W21REP-1, 2 | `tests/reporting/test_slice_finder_and_multiclass_known_rows.py` |
| W21FE-1b | `tests/feature_engineering/test_transformer_siblings_leave_uncovered_rows_nan.py` (ks_shift end to end; the other six use the same one-line change, their existing suites pass) |
| W21CORE-4 | `tests/inference/test_explainability_oof_rows_without_a_fold_are_nan.py` |
| W21CORE-6 | `tests/inference/test_logical_constraints_ktc_key_includes_rule_count.py` |

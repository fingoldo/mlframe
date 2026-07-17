"""Declarative xfail rule table shared with the tracked tests."""

from __future__ import annotations

from typing import Callable

from .combo import FuzzCombo


# _rule_cb_pool_reuse_with_mrmr_small_n_filtered REMOVED 2026-04-27 — fixed
# by empty-target + length-mismatch guards in
# trainer._maybe_get_or_build_cb_pool (rebuild on mismatch / empty target).
# The rule's TODO (rebuild cb Pool unconditionally when use_mrmr_fs=True) is
# subsumed by the per-fit length-mismatch check.
#
# _rule_cb_text_dict_collapse_with_full_quartet REMOVED 2026-04-27 — fixed
# by dynamic CB ``text_processing`` calibration in
# training/helpers.compute_cb_text_processing, applied at trainer fit-time
# AND in feature_selection/wrappers.py RFECV inner-fold. The rule's TODO
# ("auto-tune occurrence_lower_bound on small folds") is exactly what the
# new helper does — scaling the floor proportionally to fold rows so words
# that occur in 5%+ survive the prune. Permanent regression coverage:
# fuzz combos c0056 / c0070 / c0079 in tests/training/test_fuzz_suite.py.


KNOWN_XFAIL_RULES: list[tuple[Callable[[FuzzCombo], bool], str]] = [
    # _rule_linear_polars_gating_bug REMOVED 2026-04-22 (Fix 11).
    # Permanent regression guard: test_polars_full_combo_with_linear
    # (xfail removed) + test_sensor_linear_polars_gating_bug.
    # _rule_mrmr_plus_linear_multi_pandas REMOVED 2026-04-23.
    # _rule_cb_nan_in_cat_features_mrmr REMOVED 2026-04-23.
    # _rule_cb_multilabel_cat_nulls REMOVED 2026-04-26 — fixed by
    # defensive null-fill in trainer._train_model_with_fallback for the
    # CB pandas multilabel path.
    # _rule_empty_val_degenerate_cats_backward REMOVED 2026-04-26 — fixed
    # by min-rows guard in trainer._apply_pre_pipeline_transforms (skips
    # pre_pipeline.transform when val_df has 0 rows).
    # _rule_cb_only_mrmr_small_n_with_od REMOVED 2026-04-26 — fixed by
    # empty-target guard in _maybe_get_or_build_cb_pool.
    # _rule_cb_text_feature_full_quartet_heavy_inject REMOVED 2026-04-26
    # — fixed by feature-list filter in feature_selection/wrappers.py
    # (cat/text/embedding lists from outer fit_params are now narrowed to
    # current_features instead of overwriting the iteration-local lists).
    # _rule_cb_regression_polars_enum_mrmr_nulls_large REMOVED 2026-04-26 —
    # the matching combo (c0033 under default master_seed) now XPASSes; the
    # NaN-in-cat-feature path no longer reproduces. Removing the rule turns
    # any future regression into a real test failure instead of a silent
    # absorbed xpass.
    # _rule_multilabel_full_pipeline_deferred REMOVED 2026-04-25 — Session 6
    # full integration landed; all 42 multilabel combos pass end-to-end.
    # _rule_mrmr_plus_xgb_lgb_polars_utf8_small REMOVED 2026-04-23 — fixed by
    # `dt in set` → `dt == class` correction in filters.py categorize_dataset.
    # Permanent regression guard:
    # test_sensor_categorize_dataset_recognizes_polars_cat_dtypes.
    # _rule_cb_sparse_text_small REMOVED 2026-04-23 — same root cause; same sensor.
    # _rule_polars_schema_dispatch_bug REMOVED 2026-04-22: fixed in
    # core.py _build_tier_dfs (cache key now includes container kind).
    # Permanent regression guard: test_sensor_tier_cache_polars_pandas_collision.
    # _rule_mrmr_single_linear_pandas REMOVED 2026-04-22: fixed in MRMR.transform.
    # Permanent regression guard: test_sensor_mrmr_transform_handles_missing_support_.
]


def xfail_reason(combo: FuzzCombo) -> str | None:
    for predicate, reason in KNOWN_XFAIL_RULES:
        if predicate(combo):
            return reason
    return None

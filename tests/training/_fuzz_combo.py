"""Combo enumerator + results log for train_mlframe_models_suite fuzzing.

Design principles:
  * Deterministic: identical master_seed → identical combo list on any host.
  * Dedup-canonical: combos are canonicalized before hashing so
    semantically-equivalent combos (e.g. align_polars_dicts=True with
    pandas input) collapse to one.
  * Pairwise-covering: the greedy sampler guarantees every
    (axis_i=value_i, axis_j=value_j) pair is exercised at least once.
  * xfail-aware: combos hitting known bugs are auto-marked xfail via a
    declarative rule table — single source of truth shared with tracked
    tests elsewhere in the suite.

Results log: every fuzz run appends one JSONL row per combo to
``tests/training/_fuzz_results.jsonl`` capturing combo key, outcome
(pass/fail/xfail/skip), and — on failure — the exception class and a
one-line summary. That file is the audit trail used by human / agent
follow-ups to decide what to fix next.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import astuple, dataclass, field
from itertools import combinations as iter_combinations
from itertools import product as iter_product
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Axis space
# ---------------------------------------------------------------------------

MODELS: tuple[str, ...] = ("cb", "xgb", "lgb", "hgb", "linear")

AXES: dict[str, tuple[Any, ...]] = {
    "input_type": ("pandas", "polars_utf8", "polars_enum", "polars_nullable"),
    "n_rows": (300, 600, 1200),
    "cat_feature_count": (0, 1, 3, 8),
    "null_fraction_cats": (0.0, 0.1, 0.3),
    "use_mrmr_fs": (False, True),
    "weight_schemas": (("uniform",), ("uniform", "recency")),
    # 2026-04-24 Session 3: multilabel_classification re-added — FTE now
    # 2-D-aware (Session-2 landing); multilabel combos generate (N, 3)
    # targets via build_frame_for_combo's correlated-label logic and the
    # fuzz test runner uses MultilabelDispatchConfig via multilabel_strategy_cfg.
    "target_type": (
        "binary_classification",
        "regression",
        "multiclass_classification",
        "multilabel_classification",
    ),
    "multilabel_strategy_cfg": ("auto", "wrapper", "chain"),  # parametric on multilabel; canonicalised to "auto" for non-multilabel
    "auto_detect_cats": (True, False),
    "align_polars_categorical_dicts": (True, False),
    # 2026-04-24 expansion: flags that previously had NO coverage despite
    # being runtime-visible knobs. AXES-driven ≠ actually-wired — these
    # flags need corresponding prop-through in test_fuzz_suite.py runner.
    "use_polarsds_pipeline": (True, False),
    "use_text_features": (True, False),
    "honor_user_dtype": (True, False),
    # Rare column types that the frame builder previously omitted — this
    # is the gap the production 'skills_text' / embedding features caught
    # us on. A non-zero count forces the builder to emit at least one
    # high-cardinality text column or one pl.List embedding column.
    "text_col_count": (0, 1),
    "embedding_col_count": (0, 1),
    # 2026-04-24 combo extension — pull the test_suite_coverage_gaps
    # gap-analysis items into the fuzz axis space so cross-axis
    # interactions are exercised (e.g. OD × polars_utf8 × MRMR × linear).
    # Each axis doubles the pairwise-coverage work; the pairwise sampler
    # keeps the combo count at the target by selecting informative
    # combinations rather than a full cartesian product.
    "outlier_detection": (None, "isolation_forest"),             # #3
    "use_ensembles": (False, True),                              # #5
    "continue_on_model_failure": (False, True),                  # #21
    "iterations": (3, 30),                                       # #15
    "prefer_calibrated_classifiers": (False, True),              # #32
    "inject_degenerate_cols": (False, True),                     # #7 (const + all-null)
    "inject_inf_nan": (False, True),                             # #10
    "with_datetime_col": (False, True),                          # #11
    "inject_zero_col": (False, True),                            # #40 (uninformative)
    "fairness_col": (None, "cat_0"),                             # #31
    "custom_prep": (None, "pca2"),                               # #29
    "input_storage": ("memory", "parquet"),                      # #33
    # 2026-04-24 (round 2): config fields previously hard-coded to
    # defaults despite being user-facing knobs. Each axis exercises
    # a distinct code path that prior fuzz couldn't reach.
    "fillna_value_cfg": (None, 0.0),                             # PreprocessingConfig.fillna_value
    "scaler_name_cfg": ("standard", "robust", None),             # PolarsPipelineConfig.scaler_name
    "categorical_encoding_cfg": ("ordinal", "onehot"),           # PolarsPipelineConfig.categorical_encoding
    "skip_categorical_encoding_cfg": (False, True),              # PolarsPipelineConfig.skip_categorical_encoding
    "val_placement_cfg": ("forward", "backward"),                # TrainingSplitConfig.val_placement
    "test_size_cfg": (0.1, 0.2),                                 # TrainingSplitConfig.test_size
    "trainset_aging_limit_cfg": (None, 0.5),                     # TrainingSplitConfig.trainset_aging_limit
    "cat_text_card_threshold_cfg": (50, 300),                    # FeatureTypesConfig.cat_text_cardinality_threshold
    "early_stopping_rounds_cfg": (None, 20),                     # ModelHyperparamsConfig.early_stopping_rounds
    "use_robust_eval_metric_cfg": (False, True),                 # TrainingBehaviorConfig.use_robust_eval_metric
    # 2026-04-24 (Fix G): adversarial axis values — synthetic patterns
    # that stress-test the pipeline for bugs real-world synthetic data
    # alone cannot reach. Each of these is a 2-value axis.
    "inject_label_leak": (False, True),                          # feature = target + ε; val metric must be near-perfect
    "inject_rank_deficient": (False, True),                      # colinear feature pair; linear-model edge
    "inject_all_nan_col": (False, True),                         # whole column is NaN; pipeline guard test
    # 2026-04-24 (R3): drift, imbalance, weird-cat axes.
    "inject_test_drift": (None, "unseen_category", "out_of_range_numeric", "shifted_distribution"),  # R3-1
    "imbalance_ratio": ("balanced", "rare_5pct", "rare_1pct"),   # R3-4
    "weird_cat_content": (None, "empty", "unicode", "null_like"),# R3-5
}


# ---------------------------------------------------------------------------
# Combo dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuzzCombo:
    models: tuple[str, ...]
    input_type: str
    n_rows: int
    cat_feature_count: int
    null_fraction_cats: float
    use_mrmr_fs: bool
    weight_schemas: tuple[str, ...]
    target_type: str
    auto_detect_cats: bool
    align_polars_categorical_dicts: bool
    seed: int
    # New axes 2026-04-24 — have defaults so existing pinned sensor combos
    # and stored ``_fuzz_results.jsonl`` rows keep deserialising cleanly.
    use_polarsds_pipeline: bool = True
    use_text_features: bool = True
    honor_user_dtype: bool = False
    text_col_count: int = 0
    embedding_col_count: int = 0
    # 2026-04-24 combo extension from test_suite_coverage_gaps analysis
    outlier_detection: "str | None" = None
    use_ensembles: bool = False
    continue_on_model_failure: bool = False
    iterations: int = 3
    prefer_calibrated_classifiers: bool = False
    inject_degenerate_cols: bool = False
    inject_inf_nan: bool = False
    with_datetime_col: bool = False
    inject_zero_col: bool = False
    fairness_col: "str | None" = None
    custom_prep: "str | None" = None
    input_storage: str = "memory"
    # 2026-04-24 round 2 — config-field axes
    fillna_value_cfg: "float | None" = None
    scaler_name_cfg: "str | None" = "standard"
    categorical_encoding_cfg: str = "ordinal"
    skip_categorical_encoding_cfg: bool = False
    val_placement_cfg: str = "forward"
    test_size_cfg: float = 0.1
    trainset_aging_limit_cfg: "float | None" = None
    cat_text_card_threshold_cfg: int = 300
    early_stopping_rounds_cfg: "int | None" = None
    use_robust_eval_metric_cfg: bool = True
    # Fix G — adversarial axes
    inject_label_leak: bool = False
    inject_rank_deficient: bool = False
    inject_all_nan_col: bool = False
    # R3 — drift, imbalance, weird-cat axes
    inject_test_drift: "str | None" = None
    imbalance_ratio: str = "balanced"
    weird_cat_content: "str | None" = None
    # 2026-04-24 Phase H — multilabel dispatch axis. Only meaningful when
    # target_type == multilabel_classification.
    multilabel_strategy_cfg: str = "auto"

    def canonical_key(self) -> tuple:
        """Hashable tuple used for dedup. Canonicalizes semantically
        equivalent combos so e.g. ``align_polars_categorical_dicts=True`` with
        pandas input collapses to the False variant."""
        align = self.align_polars_categorical_dicts
        if self.input_type == "pandas":
            align = False
        null_frac = self.null_fraction_cats if self.cat_feature_count > 0 else 0.0
        # fairness_col is meaningful only if that column exists → None
        # when cat_feature_count == 0 (no cat_0 to reference).
        fairness = self.fairness_col if self.cat_feature_count > 0 else None
        # custom_prep=pca2 makes sense only on a clean all-numeric
        # frame — IncrementalPCA can't consume:
        #   * string/cat/text/embedding columns (no pre-encoding before
        #     custom_pre_pipeline in the mlframe pipeline),
        #   * NaN values (sklearn IncrementalPCA explicitly rejects NaN;
        #     the error message even suggests HistGradientBoosting as
        #     the alternative),
        #   * all-null / all-const columns (degenerate for PCA's
        #     variance computation).
        # Canonicalise custom_prep → None for any of those axes so
        # the pairwise sampler doesn't waste combos on a guaranteed-fail
        # configuration. Users who want PCA on real data must
        # pre-process upstream.
        pca_incompatible = (
            self.cat_feature_count > 0
            or self.text_col_count > 0
            or self.embedding_col_count > 0
            or self.inject_inf_nan          # injects np.nan → PCA rejects
            or self.inject_degenerate_cols  # adds all-null column → PCA rejects
        )
        custom_prep = self.custom_prep if not pca_incompatible else None
        use_ensembles = self.use_ensembles
        return (
            tuple(sorted(self.models)),
            self.input_type,
            self.n_rows,
            self.cat_feature_count,
            null_frac,
            self.use_mrmr_fs,
            tuple(sorted(self.weight_schemas)),
            self.target_type,
            self.auto_detect_cats,
            align,
            self.use_polarsds_pipeline,
            self.use_text_features,
            self.honor_user_dtype,
            self.text_col_count,
            self.embedding_col_count,
            # 2026-04-24 combo-extension axes
            self.outlier_detection,
            use_ensembles,
            self.continue_on_model_failure,
            self.iterations,
            self.prefer_calibrated_classifiers,
            self.inject_degenerate_cols,
            self.inject_inf_nan,
            self.with_datetime_col,
            self.inject_zero_col,
            fairness,
            custom_prep,
            self.input_storage,
            # 2026-04-24 round 2 — config field axes
            self.fillna_value_cfg,
            self.scaler_name_cfg,
            self.categorical_encoding_cfg,
            self.skip_categorical_encoding_cfg,
            self.val_placement_cfg,
            self.test_size_cfg,
            self.trainset_aging_limit_cfg,
            self.cat_text_card_threshold_cfg,
            self.early_stopping_rounds_cfg,
            self.use_robust_eval_metric_cfg,
            # Fix G — adversarial axes
            self.inject_label_leak,
            self.inject_rank_deficient,
            self.inject_all_nan_col,
            # R3 — drift, imbalance, weird-cat
            # inject_test_drift canonicalises to None when n_rows is too
            # small to meaningfully distinguish train from test slices.
            self.inject_test_drift if self.n_rows >= 300 else None,
            # imbalance_ratio canonicalisation: meaningful only on
            # binary classification. Extreme imbalance on small frames
            # causes random val/test splits to drop one class entirely
            # ("CatBoostError: Target contains only one unique value",
            # 2026-04-24 c0062/c0085). Clamp by expected minority count
            # per 10% slice — need ≥2 minority rows in each split's
            # worst-case unlucky draw, i.e. frac * slice_size ≥ ~4 → need
            # frac * n * 0.1 ≥ 4 → frac ≥ 40/n. At n=1200, rare_1pct
            # (0.01) = 12 total minority, slice=1.2 — unreliable → clamp
            # to rare_5pct. rare_5pct survives at n≥800.
            self._canonical_imbalance(),
            # weird_cat_content relevant only if there are cat columns.
            self.weird_cat_content if self.cat_feature_count > 0 else None,
            # Phase H: multilabel_strategy_cfg only meaningful for multilabel.
            self.multilabel_strategy_cfg if self.target_type == "multilabel_classification" else "auto",
        )

    def _canonical_imbalance(self) -> str:
        if "classification" not in self.target_type:
            return "balanced"
        # multiclass and multilabel: imbalance fuzz is its own can of worms
        # (per-class balancing for multiclass, per-label for multilabel —
        # neither is currently supported by the synthetic builder).
        # Collapse to balanced for these target types.
        if self.target_type in ("multiclass_classification", "multilabel_classification"):
            return "balanced"
        imb = self.imbalance_ratio
        frac = {"rare_5pct": 0.05, "rare_1pct": 0.01, "balanced": 0.5}.get(imb, 0.5)
        # Each split gets ~0.1×n rows. Require frac × 0.1 × n ≥ 4
        # (~4 minority rows expected in the smallest slice).
        if frac * 0.1 * self.n_rows < 4:
            # Try the next-safer rarity level.
            if imb == "rare_1pct":
                return "rare_5pct" if 0.05 * 0.1 * self.n_rows >= 4 else "balanced"
            if imb == "rare_5pct":
                return "balanced"
        return imb

    def short_id(self) -> str:
        h = hashlib.blake2s(repr(self.canonical_key()).encode(), digest_size=4).hexdigest()
        return f"c{self.seed:04d}_{h}"

    def pytest_id(self) -> str:
        # Include a human-readable prefix so failing IDs are diagnostic.
        tag = "_".join(sorted(self.models))
        short_input = self.input_type.replace("polars_", "pl_")
        return f"{self.short_id()}-{tag}-{short_input}-n{self.n_rows}"

    def to_json(self) -> dict:
        return {
            "short_id": self.short_id(),
            "models": list(self.models),
            "input_type": self.input_type,
            "n_rows": self.n_rows,
            "cat_feature_count": self.cat_feature_count,
            "null_fraction_cats": self.null_fraction_cats,
            "use_mrmr_fs": self.use_mrmr_fs,
            "weight_schemas": list(self.weight_schemas),
            "target_type": self.target_type,
            "auto_detect_cats": self.auto_detect_cats,
            "align_polars_categorical_dicts": self.align_polars_categorical_dicts,
            "seed": self.seed,
            "use_polarsds_pipeline": self.use_polarsds_pipeline,
            "use_text_features": self.use_text_features,
            "honor_user_dtype": self.honor_user_dtype,
            "text_col_count": self.text_col_count,
            "embedding_col_count": self.embedding_col_count,
            # 2026-04-24 combo-extension axes
            "outlier_detection": self.outlier_detection,
            "use_ensembles": self.use_ensembles,
            "continue_on_model_failure": self.continue_on_model_failure,
            "iterations": self.iterations,
            "prefer_calibrated_classifiers": self.prefer_calibrated_classifiers,
            "inject_degenerate_cols": self.inject_degenerate_cols,
            "inject_inf_nan": self.inject_inf_nan,
            "with_datetime_col": self.with_datetime_col,
            "inject_zero_col": self.inject_zero_col,
            "fairness_col": self.fairness_col,
            "custom_prep": self.custom_prep,
            "input_storage": self.input_storage,
            # 2026-04-24 round 2
            "fillna_value_cfg": self.fillna_value_cfg,
            "scaler_name_cfg": self.scaler_name_cfg,
            "categorical_encoding_cfg": self.categorical_encoding_cfg,
            "skip_categorical_encoding_cfg": self.skip_categorical_encoding_cfg,
            "val_placement_cfg": self.val_placement_cfg,
            "test_size_cfg": self.test_size_cfg,
            "trainset_aging_limit_cfg": self.trainset_aging_limit_cfg,
            "cat_text_card_threshold_cfg": self.cat_text_card_threshold_cfg,
            "early_stopping_rounds_cfg": self.early_stopping_rounds_cfg,
            "use_robust_eval_metric_cfg": self.use_robust_eval_metric_cfg,
            # Fix G
            "inject_label_leak": self.inject_label_leak,
            "inject_rank_deficient": self.inject_rank_deficient,
            "inject_all_nan_col": self.inject_all_nan_col,
            # R3
            "inject_test_drift": self.inject_test_drift,
            "imbalance_ratio": self.imbalance_ratio,
            "weird_cat_content": self.weird_cat_content,
            "multilabel_strategy_cfg": self.multilabel_strategy_cfg,
        }


# ---------------------------------------------------------------------------
# Known-xfail rules (single source of truth for auto-xfail)
# ---------------------------------------------------------------------------


# _rule_linear_polars_gating_bug REMOVED 2026-04-22 (Fix 11):
# core.py:3085 now computes polars_pipeline_applied per-strategy:
#   polars_pipeline_applied AND strategy.supports_polars
#                            AND NOT strategy.requires_encoding
# Linear (supports_polars=False, requires_encoding=True) always gets
# skip_preprocessing=False, so its pre_pipeline runs the encoder fully.
# Permanent regression guard: test_polars_full_combo_with_linear in
# test_integration_prod_like_polars.py (xfail removed) +
# test_sensor_linear_polars_gating_bug in test_fuzz_regression_sensors.py.


# _rule_mrmr_plus_linear_multi_pandas REMOVED 2026-04-23: new-seed fuzz
# showed 4/6 XPASS from this rule — the MRMR+linear+pandas combos that
# once failed on feature-name mismatch now pass (composite of all
# 2026-04-22/23 fixes — per-model pipeline cloning, _is_fitted
# Pipeline-aware check, MRMR in-place drop, Fix 10 polars-native MRMR,
# Fix 11 per-strategy polars_pipeline_applied, Fix 12 dt==class dispatch).
# Rule is now misleading; permanent regression guard: the integration
# test suite's test_polars_full_combo_with_linear (already un-xfailed).

# _rule_cb_nan_in_cat_features_mrmr REMOVED 2026-04-23: fixed in trainer.py
# `_polars_nullable_categorical_cols` — the candidate list for the fill_null
# pre-fit pass now includes pl.Utf8 / pl.String dtypes (previously only
# Categorical / Enum). Raw Utf8 cat columns with nulls are now filled
# before CB sees them.


def _rule_cb_regression_polars_enum_mrmr_nulls_large(c: FuzzCombo) -> bool:
    """New-seed (2026-04-24) fuzz c0086 residual: CB + polars_enum + MRMR +
    nulls + target_type='regression' + ncats>=8 + n>=1000.
    CB raises 'Invalid type for cat_feature ... NaN' despite
    _polars_nullable_categorical_cols covering pl.Enum. Hypothesis: MRMR
    path introduces new NaNs after the upstream fill, or the fill doesn't
    propagate through a specific regression-only branch. Needs deeper dig.
    Narrow rule so future regressions don't hide."""
    return (
        "cb" in c.models
        and c.input_type == "polars_enum"
        and c.use_mrmr_fs
        and c.null_fraction_cats > 0
        and c.cat_feature_count >= 8
        and c.n_rows >= 1000
        and c.target_type == "regression"
    )


# _rule_mrmr_plus_xgb_lgb_polars_utf8_small REMOVED 2026-04-23: same root
# cause as _rule_cb_sparse_text_small — categorize_dataset incorrectly
# routed polars Categorical / Utf8 columns through the numeric branch
# because `dt in set_of_dtype_classes` uses hash equality, and
# `pl.Categorical` instance hash differs from class hash. Permanent
# regression guard: test_sensor_categorize_dataset_recognizes_polars_cat_dtypes.


# _rule_cb_sparse_text_small REMOVED 2026-04-23: the underlying failures
# were a symptom of the categorize_dataset dt-in-set hash bug
# (filters.py:2660, `dt in {pl.Utf8, pl.Categorical, ...}` returns False
# for Categorical instances because class-vs-instance hash differs).
# Fixed there with explicit `==` checks per dtype. Both c0048 and c0098
# now PASS. Permanent regression guard:
# test_sensor_categorize_dataset_recognizes_polars_cat_dtypes.


# REMOVED 2026-04-22: _rule_polars_schema_dispatch_bug
#
# Root cause was: _build_tier_dfs in core.py cached tier-DFs keyed only on
# strategy.feature_tier() — which collides between Polars and pandas inputs
# when a non-polars-native strategy (Linear) runs before a polars-native
# strategy (XGB) in the same multi-model suite. Linear stashed pandas
# tier-DFs under tier=(False,False); XGB retrieved the same key and got
# pandas back; XGBoostStrategy.prepare_polars_dataframe then tried
# df.schema.items() on a pandas frame and raised AttributeError.
#
# Fix: cache key now = (tier, kind) where kind ∈ {"pl","pd"} sampled from
# the first non-None input. See test_sensor_tier_cache_polars_pandas_collision
# in test_fuzz_regression_sensors.py — permanent regression guard.


# _rule_mrmr_single_linear_pandas REMOVED 2026-04-22: MRMR.transform now
# uses getattr(self, 'support_', None) so a fit() that exits without
# setting support_ (e.g. early-exit on low-MI synthetic data) degrades
# to pass-through instead of raising. Regression guard:
# test_sensor_mrmr_transform_handles_missing_support_ in test_fuzz_regression_sensors.py.


# _rule_multilabel_full_pipeline_deferred REMOVED 2026-04-25 — full
# multilabel integration landed in Session 6. All 42 multilabel combos
# in the fuzz suite pass end-to-end after target_type plumbing into
# get_training_configs, MultiOutputClassifier wrapping for
# HGB/XGB/LGB/Linear, multilabel-aware report path, MRMR target injection
# fix, and supervised-encoder target collapse. See CHANGELOG.md
# "Session 6: multilabel full-pipeline integration" entry.


KNOWN_XFAIL_RULES: list[tuple[Callable[[FuzzCombo], bool], str]] = [
    # _rule_linear_polars_gating_bug REMOVED 2026-04-22 (Fix 11).
    # Permanent regression guard: test_polars_full_combo_with_linear
    # (xfail removed) + test_sensor_linear_polars_gating_bug.
    # _rule_mrmr_plus_linear_multi_pandas REMOVED 2026-04-23.
    # _rule_cb_nan_in_cat_features_mrmr REMOVED 2026-04-23.
    (
        _rule_cb_regression_polars_enum_mrmr_nulls_large,
        "CB + polars_enum + MRMR + nulls + regression + ncats>=8 + n>=1000 "
        "still raises 'Invalid type for cat_feature ... NaN' after the "
        "fill_null extension. Narrow window; likely MRMR introduces new "
        "NaNs after upstream fill, or a regression-only branch bypasses "
        "it. Needs deeper dig.",
    ),
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


# ---------------------------------------------------------------------------
# Enumerator: pairwise-greedy, deterministic, deduplicated
# ---------------------------------------------------------------------------


def _powerset_nonempty(items: tuple[str, ...]) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for r in range(1, len(items) + 1):
        for sub in iter_combinations(items, r):
            out.append(tuple(sorted(sub)))
    return out


def _sample_axes(rng: random.Random) -> dict[str, Any]:
    return {name: rng.choice(values) for name, values in AXES.items()}


def _build_combo(models: tuple[str, ...], axes: dict[str, Any], seed: int) -> FuzzCombo:
    return FuzzCombo(
        models=tuple(sorted(models)),
        input_type=axes["input_type"],
        n_rows=axes["n_rows"],
        cat_feature_count=axes["cat_feature_count"],
        null_fraction_cats=axes["null_fraction_cats"],
        use_mrmr_fs=axes["use_mrmr_fs"],
        weight_schemas=axes["weight_schemas"],
        target_type=axes["target_type"],
        auto_detect_cats=axes["auto_detect_cats"],
        align_polars_categorical_dicts=axes["align_polars_categorical_dicts"],
        seed=seed,
        use_polarsds_pipeline=axes.get("use_polarsds_pipeline", True),
        use_text_features=axes.get("use_text_features", True),
        honor_user_dtype=axes.get("honor_user_dtype", False),
        text_col_count=axes.get("text_col_count", 0),
        embedding_col_count=axes.get("embedding_col_count", 0),
        # 2026-04-24 combo-extension axes
        outlier_detection=axes.get("outlier_detection"),
        use_ensembles=axes.get("use_ensembles", False),
        continue_on_model_failure=axes.get("continue_on_model_failure", False),
        iterations=axes.get("iterations", 3),
        prefer_calibrated_classifiers=axes.get("prefer_calibrated_classifiers", False),
        inject_degenerate_cols=axes.get("inject_degenerate_cols", False),
        inject_inf_nan=axes.get("inject_inf_nan", False),
        with_datetime_col=axes.get("with_datetime_col", False),
        inject_zero_col=axes.get("inject_zero_col", False),
        fairness_col=axes.get("fairness_col"),
        custom_prep=axes.get("custom_prep"),
        input_storage=axes.get("input_storage", "memory"),
        # 2026-04-24 round 2
        fillna_value_cfg=axes.get("fillna_value_cfg"),
        scaler_name_cfg=axes.get("scaler_name_cfg", "standard"),
        categorical_encoding_cfg=axes.get("categorical_encoding_cfg", "ordinal"),
        skip_categorical_encoding_cfg=axes.get("skip_categorical_encoding_cfg", False),
        val_placement_cfg=axes.get("val_placement_cfg", "forward"),
        test_size_cfg=axes.get("test_size_cfg", 0.1),
        trainset_aging_limit_cfg=axes.get("trainset_aging_limit_cfg"),
        cat_text_card_threshold_cfg=axes.get("cat_text_card_threshold_cfg", 300),
        early_stopping_rounds_cfg=axes.get("early_stopping_rounds_cfg"),
        use_robust_eval_metric_cfg=axes.get("use_robust_eval_metric_cfg", True),
        # Fix G
        inject_label_leak=axes.get("inject_label_leak", False),
        inject_rank_deficient=axes.get("inject_rank_deficient", False),
        inject_all_nan_col=axes.get("inject_all_nan_col", False),
        # R3
        inject_test_drift=axes.get("inject_test_drift"),
        imbalance_ratio=axes.get("imbalance_ratio", "balanced"),
        weird_cat_content=axes.get("weird_cat_content"),
        multilabel_strategy_cfg=axes.get("multilabel_strategy_cfg", "auto"),
    )


def _all_axis_pairs() -> set[tuple[str, Any, str, Any]]:
    pairs: set[tuple[str, Any, str, Any]] = set()
    # Also include model-count ("n_models" pseudo-axis) to balance single vs
    # multi-model combos across other axes.
    axes_ext: dict[str, tuple[Any, ...]] = {**AXES, "n_models": (1, 2, 3, 4, 5)}
    axis_names = list(axes_ext.keys())
    for i in range(len(axis_names)):
        for j in range(i + 1, len(axis_names)):
            ai, aj = axis_names[i], axis_names[j]
            for vi in axes_ext[ai]:
                for vj in axes_ext[aj]:
                    pairs.add((ai, vi, aj, vj))
    return pairs


def _combo_pairs(combo: FuzzCombo) -> set[tuple[str, Any, str, Any]]:
    values = {
        "input_type": combo.input_type,
        "n_rows": combo.n_rows,
        "cat_feature_count": combo.cat_feature_count,
        "null_fraction_cats": combo.null_fraction_cats,
        "use_mrmr_fs": combo.use_mrmr_fs,
        "weight_schemas": combo.weight_schemas,
        "target_type": combo.target_type,
        "auto_detect_cats": combo.auto_detect_cats,
        "align_polars_categorical_dicts": combo.align_polars_categorical_dicts,
        "use_polarsds_pipeline": combo.use_polarsds_pipeline,
        "use_text_features": combo.use_text_features,
        "honor_user_dtype": combo.honor_user_dtype,
        "text_col_count": combo.text_col_count,
        "embedding_col_count": combo.embedding_col_count,
        # 2026-04-24 combo-extension axes
        "outlier_detection": combo.outlier_detection,
        "use_ensembles": combo.use_ensembles,
        "continue_on_model_failure": combo.continue_on_model_failure,
        "iterations": combo.iterations,
        "prefer_calibrated_classifiers": combo.prefer_calibrated_classifiers,
        "inject_degenerate_cols": combo.inject_degenerate_cols,
        "inject_inf_nan": combo.inject_inf_nan,
        "with_datetime_col": combo.with_datetime_col,
        "inject_zero_col": combo.inject_zero_col,
        "fairness_col": combo.fairness_col,
        "custom_prep": combo.custom_prep,
        "input_storage": combo.input_storage,
        # 2026-04-24 round 2
        "fillna_value_cfg": combo.fillna_value_cfg,
        "scaler_name_cfg": combo.scaler_name_cfg,
        "categorical_encoding_cfg": combo.categorical_encoding_cfg,
        "skip_categorical_encoding_cfg": combo.skip_categorical_encoding_cfg,
        "val_placement_cfg": combo.val_placement_cfg,
        "test_size_cfg": combo.test_size_cfg,
        "trainset_aging_limit_cfg": combo.trainset_aging_limit_cfg,
        "cat_text_card_threshold_cfg": combo.cat_text_card_threshold_cfg,
        "early_stopping_rounds_cfg": combo.early_stopping_rounds_cfg,
        "use_robust_eval_metric_cfg": combo.use_robust_eval_metric_cfg,
        # Fix G
        "inject_label_leak": combo.inject_label_leak,
        "inject_rank_deficient": combo.inject_rank_deficient,
        "inject_all_nan_col": combo.inject_all_nan_col,
        # R3
        "inject_test_drift": combo.inject_test_drift,
        "imbalance_ratio": combo.imbalance_ratio,
        "weird_cat_content": combo.weird_cat_content,
        # Phase H
        "multilabel_strategy_cfg": combo.multilabel_strategy_cfg,
        "n_models": len(combo.models),
    }
    names = list(values.keys())
    out: set[tuple[str, Any, str, Any]] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ai, aj = names[i], names[j]
            out.add((ai, values[ai], aj, values[aj]))
    return out


def enumerate_combos(
    target: int = 150,
    master_seed: int = 2026_04_22,
    model_universe: tuple[str, ...] = MODELS,
) -> list[FuzzCombo]:
    """Return `target` unique FuzzCombo instances covering the axis space.

    Phase A seeds with one combo per non-empty model-subset.
    Phase B greedy-fills until pairwise coverage is achieved.
    Phase C random-fills until len == target.
    """
    rng = random.Random(master_seed)
    seen: set[tuple] = set()
    combos: list[FuzzCombo] = []

    # Phase A — model subsets
    for subset in _powerset_nonempty(model_universe):
        axes = _sample_axes(rng)
        combo = _build_combo(subset, axes, len(combos))
        key = combo.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)

    # Phase B — pairwise coverage
    required = _all_axis_pairs()
    covered: set[tuple[str, Any, str, Any]] = set()
    for c in combos:
        covered.update(_combo_pairs(c))

    tries = 0
    max_tries = 10_000
    while covered < required and tries < max_tries and len(combos) < target:
        uncovered = required - covered
        best_combo = None
        best_new = 0
        for _ in range(50):
            subset = rng.choice(_powerset_nonempty(model_universe))
            axes = _sample_axes(rng)
            candidate = _build_combo(subset, axes, len(combos))
            if candidate.canonical_key() in seen:
                continue
            cand_pairs = _combo_pairs(candidate)
            new = len(cand_pairs & uncovered)
            if new > best_new:
                best_new = new
                best_combo = candidate
        if best_combo is None:
            break
        seen.add(best_combo.canonical_key())
        combos.append(best_combo)
        covered.update(_combo_pairs(best_combo))
        tries += 1

    # Phase C — random fill until target
    while len(combos) < target:
        subset = rng.choice(_powerset_nonempty(model_universe))
        axes = _sample_axes(rng)
        candidate = _build_combo(subset, axes, len(combos))
        key = candidate.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        combos.append(candidate)

    return combos[:target]


# ---------------------------------------------------------------------------
# Fix A — 3-wise covering over a curated subset of load-bearing axes
# ---------------------------------------------------------------------------

# Only the axes where 3-way interaction bugs have historically lived or
# are most plausible. Restricting the triple-space from the full 36
# axes to these 13 keeps the covering algorithm tractable (~286 axis-
# triples × ~12 value-triples = ~3.5k triples to cover) while still
# probing the interactions that matter. Expand cautiously — adding one
# axis bumps the triple count by ~C(N-1, 2) new axis-triples.
_3WAY_AXES: tuple[str, ...] = (
    "input_type",
    "n_rows",
    "cat_feature_count",
    "use_mrmr_fs",
    "target_type",
    "outlier_detection",
    "use_ensembles",
    "inject_inf_nan",
    "inject_degenerate_cols",
    "custom_prep",
    "categorical_encoding_cfg",
    "scaler_name_cfg",
    "inject_label_leak",
    "inject_rank_deficient",
    "inject_all_nan_col",
)


def _all_axis_triples() -> set[tuple[str, Any, str, Any, str, Any]]:
    axes_ext: dict[str, tuple[Any, ...]] = {
        name: AXES[name] for name in _3WAY_AXES if name in AXES
    }
    axes_ext["n_models"] = (1, 2, 3, 4, 5)
    names = list(axes_ext.keys())
    out: set[tuple] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for k in range(j + 1, len(names)):
                ai, aj, ak = names[i], names[j], names[k]
                for vi in axes_ext[ai]:
                    for vj in axes_ext[aj]:
                        for vk in axes_ext[ak]:
                            out.add((ai, vi, aj, vj, ak, vk))
    return out


def _combo_triples(combo: FuzzCombo) -> set[tuple[str, Any, str, Any, str, Any]]:
    values = {
        name: getattr(combo, name) for name in _3WAY_AXES if hasattr(combo, name)
    }
    values["n_models"] = len(combo.models)
    names = list(values.keys())
    out: set[tuple] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for k in range(j + 1, len(names)):
                out.add(
                    (names[i], values[names[i]],
                     names[j], values[names[j]],
                     names[k], values[names[k]])
                )
    return out


def enumerate_combos_3way(
    target: int = 600,
    master_seed: int = 2026_04_24,
    model_universe: tuple[str, ...] = MODELS,
) -> list[FuzzCombo]:
    """Greedy 3-wise (triple) covering over ``_3WAY_AXES``.

    Same shape as ``enumerate_combos`` but optimises for triple-coverage
    instead of pair-coverage. Seeded separately (default ``2026_04_24``
    so the 3-wise suite doesn't stomp the pairwise seed's sample).
    """
    rng = random.Random(master_seed)
    seen: set[tuple] = set()
    combos: list[FuzzCombo] = []

    # Phase A — model subsets
    for subset in _powerset_nonempty(model_universe):
        axes = _sample_axes(rng)
        combo = _build_combo(subset, axes, len(combos))
        key = combo.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)

    # Phase B — greedy triple coverage
    required = _all_axis_triples()
    covered: set[tuple] = set()
    for c in combos:
        covered.update(_combo_triples(c))

    tries = 0
    max_tries = 40_000
    while covered < required and tries < max_tries and len(combos) < target:
        uncovered = required - covered
        best_combo = None
        best_new = 0
        for _ in range(80):
            subset = rng.choice(_powerset_nonempty(model_universe))
            axes = _sample_axes(rng)
            candidate = _build_combo(subset, axes, len(combos))
            if candidate.canonical_key() in seen:
                continue
            cand = _combo_triples(candidate)
            new = len(cand & uncovered)
            if new > best_new:
                best_new = new
                best_combo = candidate
        if best_combo is None:
            break
        seen.add(best_combo.canonical_key())
        combos.append(best_combo)
        covered.update(_combo_triples(best_combo))
        tries += 1

    # Phase C — random fill
    while len(combos) < target:
        subset = rng.choice(_powerset_nonempty(model_universe))
        axes = _sample_axes(rng)
        candidate = _build_combo(subset, axes, len(combos))
        key = candidate.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        combos.append(candidate)

    return combos[:target]


# ---------------------------------------------------------------------------
# Results log — JSONL append-only
# ---------------------------------------------------------------------------

RESULTS_LOG = Path(__file__).parent / "_fuzz_results.jsonl"


def log_combo_outcome(
    combo: FuzzCombo,
    outcome: str,
    duration_s: float,
    error_class: str | None = None,
    error_summary: str | None = None,
    extra: dict | None = None,
) -> None:
    """Append one JSONL row with the combo's outcome.

    Columns: combo fields, outcome in {pass,fail,xpass,xfail,skip}, duration,
    error_class/error_summary (for fail/xpass rows), extra (free-form dict),
    and ``master_seed`` (Fix E: seed-rotation telemetry — the nightly cron
    passes a different ``FUZZ_SEED`` each run, we tag each row so failures
    stay attributable to their generating seed).
    """
    row: dict = {
        **combo.to_json(),
        "outcome": outcome,
        "duration_s": round(duration_s, 3),
        "master_seed": int(os.environ.get("FUZZ_SEED", "20260422")),
    }
    if error_class:
        row["error_class"] = error_class
    if error_summary:
        row["error_summary"] = error_summary[:300]
    if extra:
        row["extra"] = extra
    try:
        RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        pass  # never break a test because logging failed


def read_fail_summary() -> dict:
    """Return a summary of failures since the last run start marker."""
    if not RESULTS_LOG.exists():
        return {"fails": [], "totals": {}}
    totals: dict[str, int] = {}
    fails: list[dict] = []
    with RESULTS_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            totals[row.get("outcome", "?")] = totals.get(row.get("outcome", "?"), 0) + 1
            if row.get("outcome") == "fail":
                fails.append(row)
    return {"fails": fails, "totals": totals}


# ---------------------------------------------------------------------------
# Frame builder — turns a combo into (df, target_col, cat_feature_names)
# ---------------------------------------------------------------------------


def build_frame_for_combo(combo: FuzzCombo):
    """Build a pd / pl DataFrame matching the combo's input spec.

    Returns (df, target_col_name, cat_feature_names: list[str]).

    Text columns (``combo.text_col_count > 0``) are only emitted when
    ``"cb"`` is in ``combo.models`` — CatBoost is the only strategy that
    consumes ``text_features`` (see ``strategies.py``
    ``supports_text_features=True`` for CB; every other model either
    drops them via ``core.py:486-496`` or never looks at them). Same
    gate for embedding columns (``pl.List(pl.Float32)``). We still
    emit them SOMETIMES (not always) because the CB×text_features and
    CB×embeddings paths have their own TF-IDF / feature-dispatch
    edge cases that the earlier fuzz runs never exercised — pin them
    behind the "cb present" gate so a CB-less combo doesn't spuriously
    fail for a reason unrelated to what's being sampled.
    """
    import numpy as np

    rng = np.random.default_rng(combo.seed)
    n = combo.n_rows

    num_cols = {
        f"num_{i}": rng.standard_normal(n).astype("float32") for i in range(4)
    }
    cat_pools = [
        ["A", "B", "C"],
        ["X", "Y", "Z", "W"],
        ["alpha", "beta"],
        ["cat1", "cat2", "cat3", "cat4", "cat5"],
        ["US", "UK", "DE"],
        ["mon", "tue", "wed", "thu"],
        ["P", "Q"],
        ["r1", "r2", "r3"],
    ]
    cat_cols = {}
    cat_names: list[str] = []
    # R3-5 weird_cat_content: substitute specific pool entries with
    # pathological values that historically broke auto-detection, TF-IDF,
    # or encoder dispatch.
    def _apply_weird(pool: list[str], kind: "str | None") -> list[str]:
        if not kind:
            return pool
        pool = list(pool)
        if kind == "empty":
            # replace first entry with empty string
            if pool:
                pool[0] = ""
        elif kind == "unicode":
            # mix in a unicode-heavy value (emoji + CJK + combining marks)
            pool.append("кат́")  # cyrillic + combining acute
            pool.append("\U0001f600\U0001f4ca")        # emoji pair
        elif kind == "null_like":
            # strings that LOOK like nulls but are real string values.
            # Pipeline bugs sometimes treat these as actual nulls.
            pool.extend(["None", "NaN", "null", "NA"])
        return pool

    for i in range(combo.cat_feature_count):
        pool = _apply_weird(cat_pools[i], combo.weird_cat_content)
        values = [pool[j % len(pool)] for j in range(n)]
        if combo.null_fraction_cats > 0:
            mask = rng.random(n) < combo.null_fraction_cats
            values = [None if mask[j] else v for j, v in enumerate(values)]
        cat_cols[f"cat_{i}"] = values
        cat_names.append(f"cat_{i}")

    # Target: derive from num_0 + num_1 with noise so models have signal.
    # R3-2 multi_classification_{3,5}: discretise a continuous score into
    # N bins by quantile so distribution is approximately balanced.
    # R3-4 imbalance_ratio: on binary, shift threshold so minority class
    # is 5%/1% of rows instead of ~50/50. Not applied to multi-class
    # (implementation complexity not worth it — balanced multiclass is
    # the useful axis to exercise).
    if combo.target_type == "regression":
        target = 2.0 * num_cols["num_0"] - 1.5 * num_cols["num_1"] + rng.standard_normal(n) * 0.3
        target_col = "target_reg"
    elif combo.target_type == "binary_classification":
        logits = num_cols["num_0"] - 0.5 * num_cols["num_1"] + rng.standard_normal(n) * 0.3
        # Use the canonical imbalance value (clamped by n_rows via
        # _canonical_imbalance) so we never generate a target whose split
        # would reliably drop a class from val/test.
        imb = combo._canonical_imbalance()
        if imb == "rare_5pct":
            thresh = np.quantile(logits, 0.95)
        elif imb == "rare_1pct":
            thresh = np.quantile(logits, 0.99)
        else:
            thresh = 0.0
        target = (logits > thresh).astype("int32")
        target_col = "target"
    elif combo.target_type == "multiclass_classification":
        # 3-class quantile-cut to balanced classes (Phase H restoration of R3-2).
        score = num_cols["num_0"] + 0.3 * num_cols["num_1"] + rng.standard_normal(n) * 0.4
        k = 3  # default 3 classes; multi_5 deferred (resource-heavy)
        quantiles = [np.quantile(score, i / k) for i in range(1, k)]
        target = np.digitize(score, quantiles).astype("int32")
        target_col = "target"
    elif combo.target_type == "multilabel_classification":
        # K=3 binary labels with deliberate label correlation so chain ensemble
        # has a chance to win. Post-generation guarantee: no all-zero rows
        # (iterstrat / sklearn reject those silently).
        k = 3
        logit0 = num_cols["num_0"] - 0.4 * num_cols["num_1"] + rng.standard_normal(n) * 0.4
        y0 = (logit0 > 0).astype("int8")
        logit1 = 0.5 * y0 + num_cols["num_2"] + rng.standard_normal(n) * 0.4
        y1 = (logit1 > 0).astype("int8")
        logit2 = 0.5 * y0 + 0.5 * y1 + 0.3 * num_cols["num_3"] + rng.standard_normal(n) * 0.4
        y2 = (logit2 > 0.6).astype("int8")  # rarer
        Y = np.column_stack([y0, y1, y2])
        # Guarantee no all-zero rows (iterstrat, MultiOutputClassifier).
        zeros = (Y.sum(axis=1) == 0)
        if zeros.any():
            # flip a random label to 1 in zero rows (deterministic via rng)
            for i in np.where(zeros)[0]:
                Y[i, rng.integers(0, k)] = 1
        target = Y  # (N, K)
        target_col = "target"  # FTE will need to handle 2-D target
    else:
        raise ValueError(f"unknown target_type: {combo.target_type}")

    # Text columns: only emit when CB will actually consume them. Each
    # "text" row is a 3-word sentence drawn from a shared vocabulary so
    # CB's TF-IDF builds a non-empty dictionary (a single-word-per-row
    # column above the cardinality threshold would otherwise degenerate).
    want_text = combo.text_col_count > 0 and "cb" in combo.models
    text_vocab = [
        "python", "rust", "golang", "java", "swift", "kotlin",
        "backend", "frontend", "devops", "mlops", "dataeng", "platform",
        "cloud", "edge", "realtime", "batch", "stream", "vector",
        "search", "nlp", "vision", "audio", "robotics", "quantum",
    ]
    text_cols: dict[str, list] = {}
    if want_text:
        for i in range(combo.text_col_count):
            rows = []
            for _ in range(n):
                # 3 tokens per row, order randomized → non-trivial TF-IDF
                idxs = rng.integers(0, len(text_vocab), size=3)
                rows.append(" ".join(text_vocab[j] for j in idxs))
            text_cols[f"text_{i}"] = rows

    # Embedding columns: only Polars inputs support detection via
    # ``pl.List(pl.Float32)``; pandas has no robust native analog the
    # auto-detector recognises — skip for pandas to avoid spurious
    # xfails unrelated to the axis under test.
    want_embedding = (
        combo.embedding_col_count > 0
        and "cb" in combo.models
        and combo.input_type != "pandas"
    )

    # Data-axis injections (2026-04-24 combo extension).
    # inject_inf_nan: drop np.inf/-np.inf/np.nan into num_0's first 3 rows
    if combo.inject_inf_nan and n >= 3:
        num_cols["num_0"][0] = np.inf
        num_cols["num_0"][1] = -np.inf
        num_cols["num_0"][2] = np.nan
    # inject_degenerate_cols (#7): add one constant + one all-null numeric
    # column that the ``remove_constant_columns`` flag should strip.
    extra_num_cols: dict = {}
    if combo.inject_degenerate_cols:
        extra_num_cols["num_const"] = np.full(n, 7.5, dtype="float32")
        extra_num_cols["num_null"] = np.full(n, np.nan, dtype="float32")
    # inject_zero_col (#40): add an all-zero numeric column as an
    # uninformative feature. Triggers the per-model "constant feature"
    # handling in CB/XGB/LGB/HGB — not supposed to break anything.
    if combo.inject_zero_col:
        extra_num_cols["num_zero"] = np.zeros(n, dtype="float32")
    # Fix G — adversarial columns.
    # inject_rank_deficient: a colinear pair (num_dep = 2 * num_0).
    # Should NOT crash linear models or destabilise GBDTs — this is a
    # correctness guard, not a performance ask.
    if combo.inject_rank_deficient:
        extra_num_cols["num_dep"] = (2.0 * num_cols["num_0"]).astype("float32")
    # inject_all_nan_col: a column that is 100% NaN. Separate from
    # inject_degenerate_cols (which covers const + null together) so
    # combos can toggle it independently.
    if combo.inject_all_nan_col:
        extra_num_cols["num_all_nan"] = np.full(n, np.nan, dtype="float32")
    # inject_label_leak: a feature exactly equal to target + tiny noise.
    # A correctly-functioning suite trains on this happily; the val
    # metric must land near-perfect. Deliberately NOT asserted here —
    # the adversarial axis catches pipeline corruption that SILENTLY
    # suppresses the leak (e.g. label-column reordering, caller-frame
    # mutation); any crash is the real bug we're probing for.
    # For multilabel (target is (N, K)): leak label 0 specifically.
    if combo.inject_label_leak:
        if combo.target_type == "multilabel_classification":
            # Leak the first label only — 2-D target can't be broadcast as
            # a single feature. Single-label leak is still catastrophic for
            # a model that silently mis-uses the first target dimension.
            leak_src = target[:, 0]
        else:
            leak_src = target
        leak_col = leak_src.astype("float32") + (rng.standard_normal(n) * 0.01).astype("float32")
        extra_num_cols["num_leak"] = leak_col
    # R3-1 inject_test_drift: perturb the last 15% of rows so test/val
    # slices see a distribution mismatch. Real prod bug surface (unseen
    # categories, out-of-range values, feature shift) — catches pipelines
    # that memoise train stats without guarding against unseen state.
    if combo.inject_test_drift and n >= 20:
        tail = max(3, int(n * 0.15))
        tail_slice = slice(n - tail, n)
        if combo.inject_test_drift == "out_of_range_numeric":
            # scale last 15% of num_0 by 100× (values outside train range)
            num_cols["num_0"][tail_slice] = num_cols["num_0"][tail_slice] * 100.0
        elif combo.inject_test_drift == "shifted_distribution":
            # shift num_0 by +5 sigma (covariate shift)
            num_cols["num_0"][tail_slice] = num_cols["num_0"][tail_slice] + 5.0
        elif combo.inject_test_drift == "unseen_category" and combo.cat_feature_count > 0:
            # overwrite the FIRST cat column's tail values with a string
            # that didn't exist in the training portion.
            # (cat_cols[f"cat_0"] is already populated; mutate in place.)
            cat_cols["cat_0"] = list(cat_cols["cat_0"])
            unseen = "ZZZ_UNSEEN"
            for j in range(n - tail, n):
                cat_cols["cat_0"][j] = unseen

    if combo.input_type == "pandas":
        import pandas as pd
        data = {**num_cols, **extra_num_cols}
        for name, values in cat_cols.items():
            data[name] = pd.Categorical(values)
        for name, values in text_cols.items():
            # pandas object dtype with n_unique > threshold triggers text
            # auto-promotion inside ``_auto_detect_feature_types``.
            data[name] = pd.array(values, dtype="string")
        # with_datetime_col (#11): add a pandas datetime64 column.
        if combo.with_datetime_col:
            data["ts"] = pd.date_range("2026-01-01", periods=n, freq="h")
        # Multilabel target: 2-D (N, K) stored as an object column of list cells.
        # SimpleFeaturesAndTargetsExtractor unpacks back to (N, K) ndarray at
        # consumption time.
        if combo.target_type == "multilabel_classification":
            data[target_col] = pd.array([row.tolist() for row in target], dtype=object)
        else:
            data[target_col] = target
        return pd.DataFrame(data), target_col, cat_names

    import polars as pl
    data_pl: dict[str, Any] = {**num_cols, **extra_num_cols}
    for name, values in cat_cols.items():
        if combo.input_type == "polars_enum":
            pool_values = [v for v in values if v is not None]
            enum_type = pl.Enum(sorted(set(pool_values)))
            data_pl[name] = pl.Series(values).cast(enum_type)
        elif combo.input_type == "polars_nullable":
            data_pl[name] = pl.Series(values).cast(pl.Categorical)
        else:  # polars_utf8
            data_pl[name] = pl.Series(values, dtype=pl.Utf8)
    for name, values in text_cols.items():
        # Text columns are always pl.Utf8 — the auto-detector routes them
        # to text_features via cardinality threshold (hundreds of unique
        # 3-word sentences on 300+ rows) regardless of combo.input_type.
        data_pl[name] = pl.Series(values, dtype=pl.Utf8)
    if want_embedding:
        emb_dim = 4
        for i in range(combo.embedding_col_count):
            vecs = rng.standard_normal((n, emb_dim)).astype("float32")
            data_pl[f"emb_{i}"] = pl.Series(
                [vecs[j].tolist() for j in range(n)],
                dtype=pl.List(pl.Float32),
            )
    # with_datetime_col (#11): polars datetime64 column.
    if combo.with_datetime_col:
        import datetime as _dt
        start = _dt.datetime(2026, 1, 1)
        data_pl["ts"] = pl.Series(
            [start + _dt.timedelta(hours=i) for i in range(n)],
            dtype=pl.Datetime,
        )
    # Multilabel target: 2-D (N, K) stored as pl.List(pl.Int8) column.
    # SimpleFeaturesAndTargetsExtractor unpacks back to (N, K) ndarray.
    if combo.target_type == "multilabel_classification":
        data_pl[target_col] = pl.Series(
            [row.tolist() for row in target],
            dtype=pl.List(pl.Int8),
        )
    else:
        data_pl[target_col] = target
    return pl.DataFrame(data_pl), target_col, cat_names

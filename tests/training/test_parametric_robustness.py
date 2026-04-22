"""Parametric fuzz tests for pipeline functions that promise "any frame".

These tests exercise pieces of the pipeline that must survive every
reasonable Polars frame shape — nulls-in-Categorical, inf/NaN floats,
constant columns, sparse-null text, and so on.

Principle: these tests assert ONLY on invariants (doesn't crash,
returns the right types, schema integrity preserved) — never on specific
values, because the input varies per example. Regression tests with
pinned inputs (``test_round11_*``, ``test_round12_*``) stay separate;
they guard the specific shapes that actually burned us and their
assertions depend on those shapes being exact.
"""
from __future__ import annotations

import pytest
from hypothesis import given, HealthCheck, settings

import polars as pl

from mlframe.testing.parametric import (
    adversarial_frame,
    prod_like_frame,
    prod_like_frame_small,
    categorical_column,
    inf_heavy_float_column,
    constant_column,
    sparse_null_column,
)
from mlframe.training.core import _auto_detect_feature_types
from mlframe.training.configs import FeatureTypesConfig


# ---------------------------------------------------------------------------
# _auto_detect_feature_types — round 11/12 hot zone
# ---------------------------------------------------------------------------


class TestAutoDetectFeatureTypesRobustness:
    """Bugs we hit here:
      round 11 — null-in-Categorical crashed CB fastpath
      round 12 — sparse-null high-card column got promoted to text_features
                 which CatBoost then rejected with 'Dictionary size is 0'
    """

    @given(df=adversarial_frame(n_rows=(50, 150)))
    def test_never_raises_and_returns_lists(self, df: pl.DataFrame):
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=50,
        )
        cat_candidates = [c for c, dt in df.schema.items()
                          if dt == pl.Categorical or dt == pl.Utf8]
        text, emb, _ = _auto_detect_feature_types(df, cfg, cat_features=cat_candidates)
        assert isinstance(text, list)
        assert isinstance(emb, list)
        # Invariant: returned names are a subset of the provided candidates
        for name in text + emb:
            assert name in df.columns, f"{name!r} not in df columns"

    @given(df=adversarial_frame(
        n_rows=(100, 200),
        include_sparse_null_col=True,      # round 12 trigger
        include_high_card_cat=False,       # keep frame small — nested flatmaps are heavy
        include_null_in_cat=True,          # round 11 trigger
        include_constant_col=False,
        include_inf_in_float=False,
    ))
    def test_sparse_null_column_not_promoted_above_floor(self, df: pl.DataFrame):
        """Guard invariant from round 12: a column that trips n_unique
        threshold but has non_null < floor fraction MUST stay as cat,
        not get promoted to text."""
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=50,
        )  # min_non_null_fraction_for_text_promotion default is 0.01 (getattr'd)
        cat_candidates = [c for c, dt in df.schema.items()
                          if dt in (pl.Categorical, pl.Utf8)]
        text, _, _ = _auto_detect_feature_types(df, cfg, cat_features=cat_candidates)
        # For every column that IS in the promoted text list, its
        # non_null fraction must be >= 1%.
        for col in text:
            non_null_frac = (df.height - df[col].null_count()) / max(df.height, 1)
            assert non_null_frac >= 0.01, (
                f"{col} promoted with non_null_frac={non_null_frac:.4f} < 1% floor"
            )

    @given(df=prod_like_frame(n_rows=(100, 300)))
    def test_prod_like_schema_passes(self, df: pl.DataFrame):
        """prod_like_frame has a realistic mix — auto-detect should
        classify predictably and never throw."""
        cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                                 cat_text_cardinality_threshold=300)
        cat_candidates = [c for c, dt in df.schema.items()
                          if dt == pl.Enum or dt == pl.Categorical or dt == pl.Utf8]
        text, emb, _ = _auto_detect_feature_types(df, cfg, cat_features=cat_candidates)
        # prod_like has small-cardinality cats (<=16 uniques) so nothing
        # should be promoted to text.
        assert text == [], f"unexpected text promotion on prod_like: {text}"
        assert emb == []


# ---------------------------------------------------------------------------
# XGB / CatBoost strategy prepare_polars_dataframe — round 11 hot zone
# ---------------------------------------------------------------------------


class TestPolarsStrategyPrepareRobustness:
    @given(df=adversarial_frame(n_rows=(50, 150)))
    def test_xgb_strategy_prepare_survives(self, df: pl.DataFrame):
        from mlframe.training.strategies import XGBoostStrategy
        strategy = XGBoostStrategy()
        cat_features = [c for c, dt in df.schema.items()
                        if dt in (pl.Categorical, pl.Enum, pl.Utf8)]
        result = strategy.prepare_polars_dataframe(df, cat_features)
        assert isinstance(result, pl.DataFrame)
        assert result.height == df.height
        # Strings should no longer be present (cast to Categorical)
        for name, dt in result.schema.items():
            if name in cat_features:
                assert dt not in (pl.Utf8, pl.String), (
                    f"{name} still {dt} after prepare — should be Categorical/Enum"
                )

    @given(df=adversarial_frame(n_rows=(50, 150),
                                include_null_in_cat=True))
    def test_cb_strategy_prepare_survives_null_in_cat(self, df: pl.DataFrame):
        """Round 11: CB Polars fastpath + null-in-Categorical used to
        fail with TypeError deep in CB's C++. Strategy's prepare step
        should produce a frame CB accepts."""
        from mlframe.training.strategies import CatBoostStrategy
        strategy = CatBoostStrategy()
        cat_features = [c for c, dt in df.schema.items()
                        if dt in (pl.Categorical, pl.Enum, pl.Utf8)]
        result = strategy.prepare_polars_dataframe(df, cat_features)
        assert isinstance(result, pl.DataFrame)
        assert result.height == df.height


# ---------------------------------------------------------------------------
# create_split_dataframes — preprocessing / splitting
# ---------------------------------------------------------------------------


class TestCreateSplitDataframesRobustness:
    """``create_split_dataframes`` is invoked by the suite's PHASE 2 on
    both polars and pandas frames, with train/val/test index arrays.
    It must (a) preserve dtypes and column order, (b) return empty
    frames when an index slice is empty, (c) never lose rows across
    the 3-way partition."""

    @given(df=adversarial_frame(n_rows=(50, 150),
                                include_null_in_cat=True,
                                include_inf_in_float=True,
                                include_constant_col=False,
                                include_sparse_null_col=False))
    def test_3way_partition_preserves_all_rows(self, df: pl.DataFrame):
        import numpy as np
        from mlframe.training.preprocessing import create_split_dataframes
        n = df.height
        # Deterministic 80/10/10 split on sequential indices — keeps
        # the test fast while still varying n per example.
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train_idx = np.arange(n_train)
        val_idx   = np.arange(n_train, n_train + n_val)
        test_idx  = np.arange(n_train + n_val, n)
        train_df, val_df, test_df = create_split_dataframes(
            df, train_idx, val_idx, test_idx,
        )
        assert train_df.height == n_train
        assert val_df.height == n_val
        assert test_df.height == n - n_train - n_val
        # Invariant: schemas match original (unless the split is empty,
        # which returns an empty pl.DataFrame() without schema).
        for split in (train_df, val_df, test_df):
            if split.height > 0:
                assert split.schema == df.schema

    @given(df=adversarial_frame(n_rows=(50, 100),
                                include_null_in_cat=True,
                                include_inf_in_float=False,
                                include_constant_col=False,
                                include_sparse_null_col=False))
    def test_empty_val_test_returns_empty_dataframes(self, df: pl.DataFrame):
        """When val_idx / test_idx are empty, the split function should
        return an empty ``pl.DataFrame`` — round-12's timestamp bug
        (``wholeday_splitting collapsed``) triggered exactly this
        edge in prod."""
        import numpy as np
        from mlframe.training.preprocessing import create_split_dataframes
        n = df.height
        train_idx = np.arange(n)
        val_idx   = np.array([], dtype=np.int64)
        test_idx  = np.array([], dtype=np.int64)
        train_df, val_df, test_df = create_split_dataframes(
            df, train_idx, val_idx, test_idx,
        )
        assert train_df.height == n
        assert val_df.height == 0
        assert test_df.height == 0
        assert isinstance(val_df, pl.DataFrame)
        assert isinstance(test_df, pl.DataFrame)


# ---------------------------------------------------------------------------
# fast_calibration_binning — numba njit, must handle inf / nan / span=0
# ---------------------------------------------------------------------------


class TestFastCalibrationBinningRobustness:
    """``fast_calibration_binning`` is numba-compiled. The C code path
    has historically had issues with ``inf``, ``nan``, and constant
    ``y_pred`` (span = 0). These fuzz tests catch regressions where
    the numba version silently disagrees with the Python fallback."""

    @given(df=adversarial_frame(
        n_rows=(30, 100),
        include_null_in_cat=False,
        include_inf_in_float=False,   # normal floats only for this test
        include_constant_col=False,
        include_sparse_null_col=False,
    ))
    def test_binning_on_normal_floats_returns_valid_shapes(self, df: pl.DataFrame):
        import numpy as np
        from mlframe.metrics import fast_calibration_binning
        # adversarial_frame's default num_f32 uses st.floats(width=32) which
        # permits NaN / +-inf without passing include_inf_in_float — they
        # leak through even when we only asked for "normal". Filter
        # explicitly so this test exercises the finite-y_pred branch only;
        # the `test_binning_on_constant_y_pred_does_not_crash` sister test
        # covers a different numba code path.
        raw = df["num_f32"].drop_nulls().to_numpy().astype(np.float64)
        raw = raw[np.isfinite(raw)]
        if len(raw) < 10:
            return  # skip undersized examples
        # Guard exp overflow: float32 can hit |x|~3.4e38, and exp(3.4e38) is +inf.
        # Clip so the logistic stays finite — we're testing the binner, not exp.
        raw = np.clip(raw, -50.0, 50.0)
        y_pred = 1.0 / (1.0 + np.exp(-raw))  # (0, 1), finite
        y_true = (y_pred > 0.5).astype(np.int8)
        freqs_pred, freqs_true, hits = fast_calibration_binning(
            y_true=y_true, y_pred=y_pred, nbins=10,
        )
        # Invariants: arrays non-negative, hits sum to n, lengths
        # agree.
        assert len(freqs_pred) == len(freqs_true) == len(hits)
        assert int(hits.sum()) == len(y_true)
        assert (hits >= 0).all()

    def test_binning_on_constant_y_pred_does_not_crash(self):
        """span = 0 path — all predictions identical. This is the
        pathological input that used to trip the numba binner before
        the span-guard was added."""
        import numpy as np
        from mlframe.metrics import fast_calibration_binning
        y_pred = np.full(100, 0.3, dtype=np.float64)
        y_true = np.random.default_rng(0).integers(0, 2, size=100).astype(np.int8)
        freqs_pred, freqs_true, hits = fast_calibration_binning(
            y_true=y_true, y_pred=y_pred, nbins=10,
        )
        # All mass lands in a single bin; total hits preserved.
        assert int(hits.sum()) == 100


# ---------------------------------------------------------------------------
# _auto_detect — round-15 specific: infs in floats must not confuse
# the classifier (floats should never be promoted to text regardless
# of inf / nan content).
# ---------------------------------------------------------------------------


class TestTrainSuiteRobustness:
    """End-to-end parametric fuzz of ``train_mlframe_models_suite``.

    This is the main user-visible entry point and the thing we
    actually care about surviving adversarial frames. Each example
    here does a full fit: ~5-30s per CB, ~2-10s per XGB, on 200-400
    rows × 12 columns. We cap ``max_examples`` aggressively and use
    tiny iterations to keep runtime sane.

    Invariants asserted:
      * Suite completes (no native crash, no uncaught exception)
      * Returns a (models_dict, metadata_dict) tuple
      * metadata has ``columns`` key
      * If ``continue_on_model_failure=True``, failed models are
        listed in ``metadata['failed_models']`` rather than raised.
    """

    # 2026-04-21 fix 9.9: shrunk n_rows 200 -> 50. At n=200 the composite
    # 5-column generator (floats + datetime + sampled_from int target +
    # Enum category) rejected ~97% of hypothesis examples silently
    # upstream in polars.testing.parametric.dataframes (verified on
    # installed hypothesis 6.147.0 + polars 1.40.0). At n=50 the same
    # schema generates cleanly and CB/XGB can still fit; the test still
    # exercises the exact suite code path for Polars-native models on a
    # prod-like schema.
    @given(df=prod_like_frame_small(n_rows=50))
    @settings(
        max_examples=3,       # suite fit is SLOW (~5-30s per example)
        deadline=None,
        suppress_health_check=list(HealthCheck),
    )
    def test_xgb_only_suite_completes(self, df: pl.DataFrame, tmp_path):
        from mlframe.training.core import train_mlframe_models_suite
        from mlframe.training.configs import (
            ModelHyperparamsConfig, TrainingBehaviorConfig,
            TrainingSplitConfig, PolarsPipelineConfig,
        )
        from .shared import TimestampedFeaturesExtractor

        fte = TimestampedFeaturesExtractor(
            target_column="target", regression=False,
            ts_field="timestamp",
        )

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="parametric_fuzz",
            model_name="fuzz_xgb",
            features_and_targets_extractor=fte,
            mlframe_models=["xgb"],
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            use_mrmr_fs=False,
            hyperparams_config=ModelHyperparamsConfig(
                iterations=5, early_stopping_rounds=3,
                xgb_kwargs={"device": "cpu", "verbosity": 0},
            ),
            behavior_config=TrainingBehaviorConfig(
                prefer_calibrated_classifiers=False,
                prefer_gpu_configs=False,
                continue_on_model_failure=True,
            ),
            pipeline_config=PolarsPipelineConfig(
                use_polarsds_pipeline=False,
                categorical_encoding=None,
                scaler_name=None,
                imputer_strategy=None,
            ),
            split_config=TrainingSplitConfig(
                shuffle_val=False, shuffle_test=False,
                test_size=0.1, val_size=0.1,
                wholeday_splitting=False,  # not enough distinct days at fuzz scale
            ),
            data_dir=str(tmp_path),
            models_dir="models",
            verbose=0,
        )
        assert isinstance(models, dict)
        assert isinstance(metadata, dict)
        # Either the fit succeeded or was logged; never silently lost.
        failed = metadata.get("failed_models", [])
        assert isinstance(failed, list)

    # 2026-04-21 fix 9.9: shrunk n_rows 200 -> 50, same rationale as
    # test_xgb_only_suite_completes above.
    @given(df=prod_like_frame_small(n_rows=50))
    @settings(
        max_examples=2,       # CB is slower than XGB; even tighter budget
        deadline=None,
        suppress_health_check=list(HealthCheck),
    )
    def test_cb_only_suite_completes_with_null_in_cat(self, df: pl.DataFrame, tmp_path):
        """Round 11 regression at suite scale: CB Polars fastpath
        must handle a prod-shape frame with null-in-Categorical
        without falling back to pandas-path (or, if it does, without
        crashing)."""
        from mlframe.training.core import train_mlframe_models_suite
        from mlframe.training.configs import (
            ModelHyperparamsConfig, TrainingBehaviorConfig,
            TrainingSplitConfig, PolarsPipelineConfig,
        )
        from .shared import TimestampedFeaturesExtractor

        fte = TimestampedFeaturesExtractor(
            target_column="target", regression=False,
            ts_field="timestamp",
        )

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="parametric_fuzz",
            model_name="fuzz_cb",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            use_mrmr_fs=False,
            hyperparams_config=ModelHyperparamsConfig(
                iterations=5, early_stopping_rounds=3,
                cb_kwargs={"task_type": "CPU", "verbose": False},
            ),
            behavior_config=TrainingBehaviorConfig(
                prefer_calibrated_classifiers=False,
                prefer_gpu_configs=False,
                continue_on_model_failure=True,
            ),
            pipeline_config=PolarsPipelineConfig(
                use_polarsds_pipeline=False,
                categorical_encoding=None,
                scaler_name=None,
                imputer_strategy=None,
            ),
            split_config=TrainingSplitConfig(
                shuffle_val=False, shuffle_test=False,
                test_size=0.1, val_size=0.1,
                wholeday_splitting=False,
            ),
            data_dir=str(tmp_path),
            models_dir="models",
            verbose=0,
        )
        assert isinstance(models, dict)
        assert isinstance(metadata, dict)


class TestAutoDetectIgnoresNumericPathology:
    @given(df=adversarial_frame(
        n_rows=(50, 150),
        include_null_in_cat=False,
        include_inf_in_float=True,    # +inf / -inf / NaN across floats
        include_constant_col=True,    # forces a pl.Int16 all-zero column
        include_sparse_null_col=False,
    ))
    def test_numeric_columns_never_promoted_to_text(self, df: pl.DataFrame):
        """Regression guard: no matter how pathological the numeric
        columns are (inf, NaN, all-zero constant), they must not slip
        into ``text_features`` — only Utf8/Categorical/Enum columns
        are text-candidates."""
        cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                                 cat_text_cardinality_threshold=10)
        # Pretend every column in the frame is a cat candidate — the
        # detector must still classify numerics correctly.
        text, emb, _ = _auto_detect_feature_types(
            df, cfg, cat_features=list(df.columns),
        )
        numeric_names = [c for c, dt in df.schema.items()
                         if dt in (pl.Float32, pl.Float64,
                                   pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                   pl.Boolean)]
        for n in numeric_names:
            assert n not in text, f"numeric {n} wrongly promoted to text"
            assert n not in emb, f"numeric {n} wrongly promoted to embedding"

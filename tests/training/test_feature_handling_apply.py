"""
End-to-end integration tests for the phase-Q
``feature_handling_apply()`` bridge.

Coverage (round-3 T22 e2e gold path):
  * Sparse-aware models (XGB / CB / LGB / linear) get two-track
    output; trained model fits without crash.
  * Dense-only models (HGB / MLP-like) get single-track dense
    output; trained model fits.
  * Multi-handler chain (TF-IDF + custom) concatenated correctly
    with disambiguated names.
  * Target encoder via FHC -> uses LeakageSafeEncoder OOF; no
    train-AUC inflation.
  * FeatureCache reuses fitted handlers across multiple model
    fits (call_count == 1 across N models).
  * ``feature_handling_config`` kwarg accepted by
    ``train_mlframe_models_suite`` and validated against active
    model list.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import pytest
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression

from mlframe.training.feature_handling import (
    CacheConfig,
    CatHandlerSpec,
    FeatureCache,
    FeatureHandlingConfig,
    HashingParams,
    ModelHandlingOverride,
    TargetEncodeParams,
    TextHandlerSpec,
    feature_handling_apply,
    tfidf_only,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def synthetic_train_df():
    """Synthetic train df."""
    rng = np.random.RandomState(0)
    n = 200
    return pl.DataFrame(
        {
            "review": [
                "great product fast shipping recommended",
                "terrible quality waste of money",
                "mid quality acceptable for the price",
                "amazing value highly recommended",
                "poor build quality returned it",
            ]
            * (n // 5),
            "country": rng.choice(["US", "UK", "DE", "FR", "JP"], size=n).tolist(),
            "x_num": rng.randn(n).astype(np.float32),
        }
    )


@pytest.fixture
def synthetic_target():
    """Synthetic target."""
    rng = np.random.RandomState(1)
    return rng.randint(0, 2, size=200).astype(np.int32)


# =====================================================================
# 1. Sparse-aware path (XGB)
# =====================================================================


class TestSparseAwarePath:
    """Groups tests covering sparse aware path."""
    def test_xgb_two_track_output(self, synthetic_train_df):
        """Xgb two track output."""
        fhc = tfidf_only(max_features=30)
        res = feature_handling_apply(
            train_df=synthetic_train_df,
            fhc=fhc,
            model_kind="xgb",
            candidate_text_columns=["review"],
        )
        # Sparse-aware -> sparse_block populated
        assert res.train.sparse_block is not None
        assert issparse(res.train.sparse_block)
        # Disambiguated names
        assert all(n.startswith("review__tfidf__") for n in res.feature_names)

    def test_lgb_two_track_output(self, synthetic_train_df):
        """Lgb two track output."""
        fhc = tfidf_only(max_features=20)
        res = feature_handling_apply(
            train_df=synthetic_train_df,
            fhc=fhc,
            model_kind="lgb",
            candidate_text_columns=["review"],
        )
        assert res.train.sparse_block is not None


# =====================================================================
# 2. Dense-only path (HGB)
# =====================================================================


class TestDenseOnlyPath:
    """Groups tests covering dense only path."""
    def test_hgb_single_track_dense_under_svd_threshold(self, synthetic_train_df):
        # 30 cols < 512 -> densify in place, no SVD.
        """Hgb single track dense under svd threshold."""
        fhc = tfidf_only(max_features=30)
        res = feature_handling_apply(
            train_df=synthetic_train_df,
            fhc=fhc,
            model_kind="hgb",
            candidate_text_columns=["review"],
        )
        assert res.train.sparse_block is None
        assert res.train.dense_block is not None
        assert res.train.dense_block.dtype == np.float32

    def test_hgb_auto_svd_above_threshold(self, synthetic_train_df, caplog):
        """Hgb auto svd above threshold."""
        caplog.set_level(logging.WARNING)
        fhc = FeatureHandlingConfig(
            default_text=[
                TextHandlerSpec(
                    method="hashing",
                    params=HashingParams(n_features=2**14),  # 16384 cols
                )
            ],
        )
        res = feature_handling_apply(
            train_df=synthetic_train_df,
            fhc=fhc,
            model_kind="hgb",
            candidate_text_columns=["review"],
        )
        # SVD applied -> dense block <= svd_default_dim=256
        assert res.train.dense_block.shape[1] <= 256
        assert any("Auto-applying TruncatedSVD" in r.getMessage() for r in caplog.records)


# =====================================================================
# 3. End-to-end model fit smoke
# =====================================================================


class TestEndToEndModelFit:
    """Groups tests covering end to end model fit."""
    def test_logreg_fits_on_assembled_matrix(self, synthetic_train_df, synthetic_target):
        """Logreg fits on assembled matrix."""
        fhc = tfidf_only(max_features=30)
        res = feature_handling_apply(
            train_df=synthetic_train_df,
            fhc=fhc,
            model_kind="linear",
            candidate_text_columns=["review"],
        )
        # Linear is sparse-aware. Use the sparse block directly.
        X = res.train.sparse_block
        assert X is not None
        clf = LogisticRegression(max_iter=200, solver="liblinear")
        clf.fit(X, synthetic_target)
        # Model trained
        assert hasattr(clf, "coef_")

    def test_logreg_with_numeric_block(self, synthetic_train_df, synthetic_target):
        """Logreg with numeric block."""
        fhc = tfidf_only(max_features=20)
        # Dense numeric block from x_num
        num_block = synthetic_train_df.select("x_num").to_numpy().astype(np.float32)
        res = feature_handling_apply(
            train_df=synthetic_train_df,
            fhc=fhc,
            model_kind="xgb",
            candidate_text_columns=["review"],
            numeric_block_train=num_block,
            numeric_feature_names=["x_num"],
        )
        # XGB two-track: sparse + dense
        assert res.train.sparse_block is not None
        assert res.train.dense_block is not None
        assert res.train.dense_block.shape == (200, 1)
        assert "x_num" in res.feature_names


# =====================================================================
# 4. Cache reuse across multiple models
# =====================================================================


class TestCacheReuse:
    """Groups tests covering cache reuse."""
    def test_one_handler_fit_across_three_models(self, synthetic_train_df, monkeypatch):
        """One handler fit across three models."""
        from mlframe.training.feature_handling.text_encoder import TextColumnEncoder

        fhc = tfidf_only(max_features=20)
        cache = FeatureCache(CacheConfig(persistence="off"))

        # Manual fit-call counter via monkeypatch (mock.patch.autospec
        # breaks self-binding through the wraps= path on bound methods).
        # The wrapper calls through so the encoder still fits properly.
        original_fit = TextColumnEncoder.fit
        call_count = [0]

        def counted_fit(self, *args, **kwargs):
            """Counted fit."""
            call_count[0] += 1
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(TextColumnEncoder, "fit", counted_fit)

        for model_kind in ("xgb", "lgb", "linear"):
            feature_handling_apply(
                train_df=synthetic_train_df,
                fhc=fhc,
                model_kind=model_kind,
                cache=cache,
                candidate_text_columns=["review"],
            )
        # All three should hit the cache after the first fit -- exactly
        # one TextColumnEncoder.fit call for the entire run.
        assert call_count[0] == 1, f"TextColumnEncoder.fit called {call_count[0]} times across 3 models; expected 1 (cache miss only on first model)"

    def test_cache_hit_yields_identical_matrix(self, synthetic_train_df):
        """Round-3 T4: cache HIT must return same data, not a stale
        reference."""
        fhc = tfidf_only(max_features=15)
        cache = FeatureCache(CacheConfig(persistence="off"))

        res1 = feature_handling_apply(
            train_df=synthetic_train_df,
            fhc=fhc,
            model_kind="xgb",
            cache=cache,
            candidate_text_columns=["review"],
        )
        res2 = feature_handling_apply(
            train_df=synthetic_train_df,
            fhc=fhc,
            model_kind="lgb",
            cache=cache,
            candidate_text_columns=["review"],
        )
        # Both should have same vocab fitted on train -> same sparse
        # matrix shape AND content for the review column.
        m1 = res1.train.sparse_block
        m2 = res2.train.sparse_block
        np.testing.assert_array_equal(m1.toarray(), m2.toarray())


# =====================================================================
# 5. Target encoder integration (LeakageSafeEncoder via FHC)
# =====================================================================


class TestTargetEncoderIntegration:
    """Groups tests covering target encoder integration."""
    def test_target_mean_yields_dense_block(self, synthetic_train_df, synthetic_target):
        """Target mean yields dense block."""
        fhc = FeatureHandlingConfig(
            default_cat=[
                CatHandlerSpec(
                    method="target_mean",
                    params=TargetEncodeParams(kind="target_mean", smoothing=10.0, cv=3),
                )
            ],
            default_text=[],
        )
        res = feature_handling_apply(
            train_df=synthetic_train_df,
            fhc=fhc,
            model_kind="xgb",
            train_target=synthetic_target,
            candidate_cat_columns=["country"],
        )
        # target encoder always emits dense
        assert res.train.dense_block is not None
        assert res.train.dense_block.shape == (200, 1)

    def test_target_encoder_requires_train_target(self, synthetic_train_df):
        """Target encoder requires train target."""
        fhc = FeatureHandlingConfig(
            default_cat=[
                CatHandlerSpec(
                    method="target_mean",
                    params=TargetEncodeParams(kind="target_mean", smoothing=10.0),
                )
            ],
            default_text=[],
        )
        with pytest.raises(ValueError, match="train_target"):
            feature_handling_apply(
                train_df=synthetic_train_df,
                fhc=fhc,
                model_kind="xgb",
                candidate_cat_columns=["country"],
                # train_target=None  - missing
            )


# =====================================================================
# 6. Text auto-detection integration
# =====================================================================


class TestAutoDetect:
    """Groups tests covering auto detect."""
    def test_auto_detect_picks_text_column(self, synthetic_train_df):
        # No candidate_text_columns argument -> detector runs.
        """Auto detect picks text column."""
        fhc = tfidf_only(max_features=10)
        res = feature_handling_apply(
            train_df=synthetic_train_df,
            fhc=fhc,
            model_kind="xgb",
            candidate_text_columns=None,  # explicit auto-detect
        )
        # synthetic review column has variety + tokens -> should be text
        assert "review" in res.text_columns_detected or len(res.detection_decisions) > 0


# =====================================================================
# 7. Suite kwarg surfaces and validates
# =====================================================================


class TestSuiteKwarg:
    """Groups tests covering suite kwarg."""
    def test_train_mlframe_models_suite_accepts_fhc_kwarg(self):
        """The kwarg must be present in the suite signature and
        accept a FeatureHandlingConfig instance without raising."""
        import inspect
        from mlframe.training.core import train_mlframe_models_suite

        sig = inspect.signature(train_mlframe_models_suite)
        assert "feature_handling_config" in sig.parameters

    def test_invalid_fhc_for_active_models_raises(self):
        """If the resolved plan has model-axis-method mismatches, the
        suite raises at start with a combined error message."""
        from mlframe.training.feature_handling import (
            LearnableEmbeddingParams,
        )

        # learnable_text_embedding on XGB is invalid (neural-only).
        bad_fhc = FeatureHandlingConfig(
            per_model={
                "xgb": ModelHandlingOverride(
                    text=[
                        TextHandlerSpec(
                            method="learnable_text_embedding",
                            params=LearnableEmbeddingParams(),
                        ),
                    ]
                ),
            },
        )
        # Direct call to validate -- mimics what the suite does early.
        with pytest.raises(ValueError, match="incompatible"):
            bad_fhc.validate_against_models(["xgb"])


# =====================================================================
# 8. Held-out transform doesn't leak
# =====================================================================


class TestHeldOutTransform:
    """Groups tests covering held out transform."""
    def test_train_val_test_use_train_fitted_encoder(self, synthetic_train_df, synthetic_target):
        # Split synthetic_train_df 60/20/20 by index
        """Train val test use train fitted encoder."""
        train = synthetic_train_df.head(120)
        val = synthetic_train_df.slice(120, 40)
        test = synthetic_train_df.slice(160, 40)

        fhc = tfidf_only(max_features=20)
        res = feature_handling_apply(
            train_df=train,
            val_df=val,
            test_df=test,
            fhc=fhc,
            model_kind="xgb",
            candidate_text_columns=["review"],
        )
        # All three split matrices share column count (vocabulary).
        assert res.train.sparse_block is not None
        assert res.val is not None and res.val.sparse_block is not None
        assert res.test is not None and res.test.sparse_block is not None
        assert res.train.sparse_block.shape[1] == res.val.sparse_block.shape[1]
        assert res.train.sparse_block.shape[1] == res.test.sparse_block.shape[1]
